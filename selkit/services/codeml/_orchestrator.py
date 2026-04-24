"""Family-agnostic codeml pipeline orchestrator.

This module factors the validate → fit → LRT → BEB → emit loop out of
:mod:`selkit.services.codeml.site_models` so that other model families
(branch, branch-site) can reuse it. Phase 1 adds the module but does not
wire it into the live code path yet — `run_site_models` continues to work
unchanged. Task 6 migrates the live path to call `run_family`.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional, Union

import numpy as np

from selkit.engine.beb import BEBSite, compute_neb
from selkit.engine.codon_model import SiteModel
from selkit.engine.fit import EngineFit, fit_model
from selkit.engine.genetic_code import GeneticCode
from selkit.engine.likelihood import per_class_site_log_likelihood
from selkit.engine.rate_matrix import estimate_f3x4
from selkit.io.config import RunConfig
from selkit.io.results import (
    BranchModelFit,
    BranchSiteModelFit,
    LRTResult,
    RunResult,
    SiteModelFit,
    StartResult,
)
from selkit.io.tree import BranchRecord, LabeledTree
from selkit.services.codeml.lrt import compute_lrt, resolve_df
from selkit.services.validate import ValidatedInputs


ModelFactory = Callable[..., SiteModel]
LRTSpec = tuple[str, str, Union[int, str], str]
Family = Literal["site", "branch", "branch-site"]


def _extract_per_branch_omega(
    *, model_name: str, fit: EngineFit, tree: LabeledTree,
) -> list[dict]:
    """Build the per_branch_omega list for a BranchModelFit.

    Uses the tree's branch_records() for stable keying. ``label`` is the
    display string: ``"foreground"``/``"background"`` for TwoRatios / NRatios,
    the tip name for tip branches in FreeRatios, and ``"mrca(...)"`` for
    internals in FreeRatios. ``omega`` is read out of fit.params using the
    model's naming convention. ``SE`` is populated from ``fit.hess_inv_diag``
    when available -- a natural-space standard error from scipy L-BFGS-B's
    inverse-Hessian diagonal. SE is ``None`` when (a) the optimizer fallback
    kicked in (older scipy without LinearOperator hess_inv) or (b) the
    parameter is pinned (e.g. omega_fg == 1 in TwoRatiosFixed).
    """
    se_for = fit.hess_inv_diag or {}  # dict[str, float] keyed by param name

    def _se(key: str | None) -> float | None:
        if key is None:
            return None
        v = se_for.get(key)
        return float(v) if v is not None else None

    recs = tree.branch_records()
    # Pre-build node-by-id map: branch_records() iterates B nodes and we look
    # up each one by id. The previous _node_by_id linear scan made the whole
    # routine O(B^2); a one-shot dict makes it O(B).
    node_by_id = {n.id: n for n in tree.all_nodes()}
    if model_name == "M0":
        # M0 has a single shared omega across the tree. Emit one row per
        # branch so downstream tooling sees a uniform per-branch table for
        # M0 too (e.g. when M0 is fit alongside TwoRatios / FreeRatios).
        om = float(fit.params["omega"])
        key = "omega" if "omega" in (se_for or {}) else None
        return [
            {
                "branch_id": r.branch_id,
                "tip_set": list(r.tip_set),
                "label": "M0",
                "paml_node_id": r.paml_node_id,
                "omega": om,
                "SE": _se(key),
            }
            for r in recs
        ]
    if model_name in {"TwoRatios", "TwoRatiosFixed"}:
        om_bg = fit.params["omega_bg"]
        om_fg = fit.params.get("omega_fg", 1.0)  # TwoRatiosFixed pins fg at 1.
        out: list[dict] = []
        for r in recs:
            node = node_by_id[r.node_id]
            is_fg = node.label == 1
            if is_fg:
                # TwoRatiosFixed: no omega_fg param -> SE is None (pinned).
                key = "omega_fg" if "omega_fg" in fit.params else None
            else:
                key = "omega_bg"
            out.append({
                "branch_id": r.branch_id,
                "tip_set": list(r.tip_set),
                "label": "foreground" if is_fg else "background",
                "paml_node_id": r.paml_node_id,
                "omega": float(om_fg if is_fg else om_bg),
                "SE": _se(key),
            })
        return out
    if model_name == "NRatios":
        out = []
        for r in recs:
            node = node_by_id[r.node_id]
            key = "omega_bg" if node.label == 0 else f"omega_{node.label}"
            label_str = "background" if node.label == 0 else f"#{node.label}"
            out.append({
                "branch_id": r.branch_id,
                "tip_set": list(r.tip_set),
                "label": label_str,
                "paml_node_id": r.paml_node_id,
                "omega": float(fit.params[key]),
                "SE": _se(key),
            })
        return out
    if model_name == "FreeRatios":
        out = []
        for r in recs:
            # After assign_unique_branch_labels, every branch's label IS its
            # branch_id. Two root-adjacent branches share one label (merged).
            node = node_by_id[r.node_id]
            key = f"omega_{node.label}"
            if node.is_tip:
                label_str = node.name or ""
            else:
                label_str = f"mrca({','.join(r.tip_set)})"
            out.append({
                "branch_id": r.branch_id,
                "tip_set": list(r.tip_set),
                "label": label_str,
                "paml_node_id": r.paml_node_id,
                "omega": float(fit.params[key]),
                "SE": _se(key),
            })
        return out
    raise ValueError(f"_extract_per_branch_omega: unknown branch model {model_name!r}")


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

    fits = {
        name: _engine_to_public(family, f, inputs=inputs)
        for name, f in engine_fits.items()
    }

    lrts = _compute_lrts(fits, default_lrts, tree=inputs.tree)
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
    family: Family, fit: EngineFit, *, inputs: ValidatedInputs | None = None,
) -> "SiteModelFit | BranchSiteModelFit | BranchModelFit":
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
        if inputs is None:
            raise ValueError(
                "branch family _engine_to_public requires inputs (for tree)"
            )
        # M0 fit appears in the branch family's run when LRTs need it (e.g.
        # M0-vs-TwoRatios, M0-vs-NRatios). M0 shares one omega across the
        # whole tree; emit one row per branch (all reporting the same omega)
        # so downstream tooling sees a uniform per-branch table for M0.
        per_branch = _extract_per_branch_omega(
            model_name=fit.model, fit=fit, tree=inputs.tree,
        )
        return BranchModelFit(
            family="branch", per_branch_omega=per_branch, **common,
        )
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
    model = registry[name](gc, pi, inputs.tree)
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
    *,
    tree: LabeledTree | None = None,
) -> list[LRTResult]:
    """Fire each registered LRT whose null and alt models are both present in `fits`."""
    lrts: list[LRTResult] = []
    for null, alt, df_spec, test_type in default_lrts:
        if null not in fits or alt not in fits:
            continue
        df = resolve_df(df_spec, tree) if tree is not None else df_spec
        if not isinstance(df, int):
            raise ValueError(
                f"lazy df spec {df_spec!r} requires a tree for resolution"
            )
        warning = None
        if alt == "FreeRatios":
            warning = (
                f"df={df} -- FreeRatios has one omega per branch; "
                "interpret per-branch omega with caution (Yang 1998)."
            )
        r = compute_lrt(
            null=null,
            alt=alt,
            lnL_null=fits[null].lnL,
            lnL_alt=fits[alt].lnL,
            df=df,
            test_type=test_type,
            warning=warning,
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
                warning=r.warning,
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
    model = registry[name](gc, pi, inputs.tree)
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
