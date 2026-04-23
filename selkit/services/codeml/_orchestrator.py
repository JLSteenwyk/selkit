"""Family-agnostic codeml pipeline orchestrator.

This module factors the validate → fit → LRT → BEB → emit loop out of
:mod:`selkit.services.codeml.site_models` so that other model families
(branch, branch-site) can reuse it. Phase 1 adds the module but does not
wire it into the live code path yet — `run_site_models` continues to work
unchanged. Task 6 migrates the live path to call `run_family`.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional

import numpy as np

from selkit.engine.beb import BEBSite
from selkit.engine.beb.site import compute_neb
from selkit.engine.codon_model import SiteModel
from selkit.engine.fit import EngineFit, fit_model
from selkit.engine.genetic_code import GeneticCode
from selkit.engine.likelihood import per_class_site_log_likelihood
from selkit.engine.rate_matrix import estimate_f3x4
from selkit.io.config import RunConfig
from selkit.io.results import (
    BranchSiteModelFit,
    LRTResult,
    RunResult,
    SiteModelFit,
    StartResult,
)
from selkit.services.codeml.lrt import compute_lrt
from selkit.services.validate import ValidatedInputs


ModelFactory = Callable[[GeneticCode, np.ndarray], SiteModel]
LRTSpec = tuple[str, str, int, str]
Family = Literal["site", "branch", "branch-site"]


def run_family(
    *,
    family: Family,
    registry: dict[str, ModelFactory],
    default_lrts: tuple[LRTSpec, ...],
    default_beb_models: tuple[str, ...],
    inputs: ValidatedInputs,
    config: RunConfig,
    parallel: bool,
    progress: Optional[Callable[[str, str], None]] = None,
    preconditions: Optional[Callable[[ValidatedInputs, RunConfig], None]] = None,
) -> RunResult:
    """Run the validate → fit → LRT → BEB pipeline for a single model family.

    Parameters
    ----------
    family
        Tag stamped on the :class:`RunResult` and used by `_engine_to_public`
        to decide whether to emit `SiteModelFit` or `BranchSiteModelFit`.
    registry
        Mapping of model-name → factory producing a :class:`SiteModel`.
    default_lrts
        LRT specs to attempt; each is `(null, alt, df, test_type)`. A spec is
        fired only if both models are present in the fit set.
    default_beb_models
        Model names for which to compute NEB/BEB after fitting.
    inputs, config
        Alignment + tree + run configuration.
    parallel
        If True and `config.threads > 1` and multiple models are being fit,
        dispatch fits to a `ProcessPoolExecutor`.
    progress
        Optional callback `progress(event, model_name)` with event in
        `{"start", "done"}`.
    preconditions
        Optional family-specific precondition check invoked before any fit.
    """
    if preconditions is not None:
        preconditions(inputs, config)

    gc = GeneticCode.by_name(config.genetic_code)
    pi = estimate_f3x4(inputs.alignment.codons, gc)
    models_to_fit = config.models or tuple(registry.keys())

    for name in models_to_fit:
        if name not in registry:
            raise ValueError(
                f"unknown model {name!r} for family {family!r}"
            )

    engine_fits = _fit_all(
        registry=registry,
        models_to_fit=models_to_fit,
        inputs=inputs,
        pi=pi,
        config=config,
        parallel=parallel,
        progress=progress,
    )

    fits = {name: _engine_to_public(family, f) for name, f in engine_fits.items()}

    lrts = _compute_lrts(fits, default_lrts)
    beb = _compute_beb(
        engine_fits=engine_fits,
        default_beb_models=default_beb_models,
        registry=registry,
        inputs=inputs,
        pi=pi,
        gc=gc,
    )

    warnings: list[str] = []
    for name, f in fits.items():
        if not f.converged:
            warnings.append(
                f"{name}: multi-start disagreement > {config.convergence_tol} lnL"
            )

    return RunResult(
        config=config,
        family=family,
        fits=fits,
        lrts=lrts,
        beb=beb,
        warnings=warnings,
    )


def _engine_to_public(
    family: Family, fit: EngineFit
) -> "SiteModelFit | BranchSiteModelFit":
    """Convert an EngineFit into the tagged public dataclass for `family`."""
    starts = [
        StartResult(
            seed=i,
            final_lnL=-s.final_lnL,
            iterations=s.iterations,
            params=s.params,
        )
        for i, s in enumerate(fit.multi_start.starts)
    ]
    common = dict(
        model=fit.model,
        lnL=fit.lnL,
        n_params=fit.n_params,
        params=fit.params,
        branch_lengths=fit.branch_lengths,
        starts=starts,
        converged=fit.multi_start.converged,
        runtime_s=fit.runtime_s,
    )
    if family == "branch-site":
        p0 = fit.params.get("p0", 0.0)
        p1 = fit.params.get("p1", 0.0)
        p2 = max(0.0, 1.0 - p0 - p1)
        denom = p0 + p1 if (p0 + p1) > 0 else 1.0
        class_proportions = {
            "p0": p0,
            "p1": p1,
            "p2a": p0 * p2 / denom,
            "p2b": p1 * p2 / denom,
        }
        return BranchSiteModelFit(
            family="branch-site",
            class_proportions=class_proportions,
            **common,
        )
    if family == "branch":
        raise NotImplementedError("branch family lands in Phase 2")
    return SiteModelFit(family="site", **common)


def _fit_all(
    *,
    registry: dict[str, ModelFactory],
    models_to_fit: tuple[str, ...],
    inputs: ValidatedInputs,
    pi: np.ndarray,
    config: RunConfig,
    parallel: bool,
    progress: Optional[Callable[[str, str], None]],
) -> dict[str, EngineFit]:
    """Fit every model in `models_to_fit`, optionally in parallel.

    Mirrors the parallel/serial branches in
    :func:`selkit.services.codeml.site_models.run_site_models`.
    """
    engine_fits: dict[str, EngineFit] = {}
    if parallel and config.threads > 1 and len(models_to_fit) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        futures: dict = {}
        with ProcessPoolExecutor(max_workers=config.threads) as ex:
            for name in models_to_fit:
                if progress is not None:
                    progress("start", name)
                futures[
                    ex.submit(
                        _run_one,
                        name,
                        registry=registry,
                        inputs=inputs,
                        pi=pi,
                        cfg=config,
                    )
                ] = name
            for fut in as_completed(futures):
                name = futures[fut]
                engine_fits[name] = fut.result()
                if progress is not None:
                    progress("done", name)
    else:
        for name in models_to_fit:
            if progress is not None:
                progress("start", name)
            engine_fits[name] = _run_one(
                name, registry=registry, inputs=inputs, pi=pi, cfg=config
            )
            if progress is not None:
                progress("done", name)
    return engine_fits


def _run_one(
    name: str,
    *,
    registry: dict[str, ModelFactory],
    inputs: ValidatedInputs,
    pi: np.ndarray,
    cfg: RunConfig,
) -> EngineFit:
    """Fit a single model. Matches `site_models._run_one`, parameterised on `registry`."""
    gc = GeneticCode.by_name(cfg.genetic_code)
    model = registry[name](gc, pi)
    return fit_model(
        model=model,
        alignment_codons=inputs.alignment.codons,
        taxon_order=inputs.alignment.taxa,
        tree=inputs.tree,
        n_starts=cfg.n_starts,
        seed=cfg.seed + _model_seed_offset(name),
        convergence_tol=cfg.convergence_tol,
    )


def _model_seed_offset(name: str) -> int:
    """Deterministic per-model seed offset.

    Python's built-in ``hash()`` is process-randomized (unless PYTHONHASHSEED
    is set), so ``hash("M7") % 10_000`` differs between runs and the multi-start
    starting points for a given model are non-reproducible. Use a stable hash
    so corpus tests aren't optimizer-flaky across CI runs.
    """
    import hashlib
    digest = hashlib.sha1(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:2], "big") % 10_000


def _compute_lrts(
    fits: dict,
    default_lrts: tuple[LRTSpec, ...],
) -> list[LRTResult]:
    """Fire each registered LRT whose null and alt models are both present in `fits`."""
    lrts: list[LRTResult] = []
    for null, alt, df, test_type in default_lrts:
        if null in fits and alt in fits:
            r = compute_lrt(
                null=null,
                alt=alt,
                lnL_null=fits[null].lnL,
                lnL_alt=fits[alt].lnL,
                df=df,
                test_type=test_type,
            )
            lrts.append(
                LRTResult(
                    null=r.null,
                    alt=r.alt,
                    delta_lnL=r.delta_lnL,
                    df=r.df,
                    p_value=r.p_value,
                    test_type=r.test_type,
                    significant_at_0_05=r.significant_at_0_05,
                )
            )
    return lrts


def _compute_beb(
    *,
    engine_fits: dict[str, EngineFit],
    default_beb_models: tuple[str, ...],
    registry: dict[str, ModelFactory],
    inputs: ValidatedInputs,
    pi: np.ndarray,
    gc: GeneticCode,
) -> dict[str, list[BEBSite]]:
    """Compute NEB/BEB for each fit in `default_beb_models` that was actually fit.

    Mirrors `site_models._compute_beb_for` + the BEB block of
    `site_models.run_site_models`.
    """
    beb: dict[str, list[BEBSite]] = {}
    for name in default_beb_models:
        if name in engine_fits:
            beb[name] = _compute_beb_for(
                name,
                fit=engine_fits[name],
                registry=registry,
                inputs=inputs,
                pi=pi,
                gc=gc,
            )
    return beb


def _compute_beb_for(
    name: str,
    *,
    fit: EngineFit,
    registry: dict[str, ModelFactory],
    inputs: ValidatedInputs,
    pi: np.ndarray,
    gc: GeneticCode,
) -> list[BEBSite]:
    """NEB for M2a/M8 (Phase 1). Phase 3 will add real BEB."""
    model = registry[name](gc, pi)
    weights, Qs = model.build(params=fit.params)
    if name == "M2a":
        omegas = [fit.params["omega0"], 1.0, fit.params["omega2"]]
    elif name == "M8":
        from selkit.engine.codon_model import _beta_quantiles

        beta_omegas = _beta_quantiles(
            fit.params["p_beta"], fit.params["q_beta"], 10
        ).tolist()
        omegas = [float(o) for o in beta_omegas] + [fit.params["omega2"]]
    else:
        raise ValueError(f"NEB not supported for {name}")
    per_class = per_class_site_log_likelihood(
        tree=inputs.tree,
        codons=inputs.alignment.codons,
        taxon_order=inputs.alignment.taxa,
        Qs=Qs,
        pi=pi,
    )
    return compute_neb(
        per_class_site_logL=per_class,
        weights=weights,
        omegas=omegas,
    )
