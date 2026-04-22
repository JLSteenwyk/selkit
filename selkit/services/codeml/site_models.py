from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from selkit.engine.beb import compute_neb
from selkit.engine.codon_model import (
    M0,
    M1a,
    M2a,
    M7,
    M8,
    M8a,
    ModelA,
    ModelANull,
    SiteModel,
)
from selkit.engine.fit import EngineFit, fit_model
from selkit.engine.genetic_code import GeneticCode
from selkit.engine.likelihood import per_class_site_log_likelihood
from selkit.engine.rate_matrix import estimate_f3x4
from selkit.errors import SelkitConfigError
from selkit.io.config import RunConfig
from selkit.io.results import (
    BEBSite,
    LRTResult,
    ModelFit,
    RunResult,
    StartResult,
)
from selkit.services.codeml.lrt import STANDARD_SITE_LRTS, compute_lrt
from selkit.services.validate import ValidatedInputs


def _make_m0(gc: GeneticCode, pi: np.ndarray) -> SiteModel:
    return M0(gc=gc, pi=pi)


def _make_m1a(gc: GeneticCode, pi: np.ndarray) -> SiteModel:
    return M1a(gc=gc, pi=pi)


def _make_m2a(gc: GeneticCode, pi: np.ndarray) -> SiteModel:
    return M2a(gc=gc, pi=pi)


def _make_m7(gc: GeneticCode, pi: np.ndarray) -> SiteModel:
    return M7(gc=gc, pi=pi)


def _make_m8(gc: GeneticCode, pi: np.ndarray) -> SiteModel:
    return M8(gc=gc, pi=pi)


def _make_m8a(gc: GeneticCode, pi: np.ndarray) -> SiteModel:
    return M8a(gc=gc, pi=pi)


def _make_modela(gc: GeneticCode, pi: np.ndarray) -> SiteModel:
    return ModelA(gc=gc, pi=pi)


def _make_modela_null(gc: GeneticCode, pi: np.ndarray) -> SiteModel:
    return ModelANull(gc=gc, pi=pi)


_MODEL_CTORS: dict[str, Callable[[GeneticCode, np.ndarray], SiteModel]] = {
    "M0": _make_m0,
    "M1a": _make_m1a,
    "M2a": _make_m2a,
    "M7": _make_m7,
    "M8": _make_m8,
    "M8a": _make_m8a,
    "ModelA": _make_modela,
    "ModelA_null": _make_modela_null,
}

# Public alias used by the shared orchestrator (_orchestrator.run_family).
# Task 6 will migrate the live path to reference MODEL_REGISTRY directly.
MODEL_REGISTRY = _MODEL_CTORS

# Models that require a foreground-branch designation to be meaningful.
_BRANCH_SITE_MODELS: frozenset[str] = frozenset({"ModelA", "ModelA_null"})

_BUNDLE_DEFAULT: tuple[str, ...] = ("M0", "M1a", "M2a", "M7", "M8", "M8a")


def _engine_to_public(fit: EngineFit) -> ModelFit:
    starts = [
        StartResult(
            seed=i,
            final_lnL=-s.final_lnL,
            iterations=s.iterations,
            params=s.params,
        )
        for i, s in enumerate(fit.multi_start.starts)
    ]
    return ModelFit(
        model=fit.model,
        lnL=fit.lnL,
        n_params=fit.n_params,
        params=fit.params,
        branch_lengths=fit.branch_lengths,
        starts=starts,
        converged=fit.multi_start.converged,
        runtime_s=fit.runtime_s,
    )


def _run_one(
    name: str,
    *,
    inputs: ValidatedInputs,
    pi: np.ndarray,
    cfg: RunConfig,
) -> EngineFit:
    gc = GeneticCode.by_name(cfg.genetic_code)
    model = _MODEL_CTORS[name](gc, pi)
    return fit_model(
        model=model,
        alignment_codons=inputs.alignment.codons,
        taxon_order=inputs.alignment.taxa,
        tree=inputs.tree,
        n_starts=cfg.n_starts,
        seed=cfg.seed + hash(name) % 10_000,
        convergence_tol=cfg.convergence_tol,
    )


def _compute_beb_for(
    name: str, *, fit: EngineFit, inputs: ValidatedInputs, pi: np.ndarray,
    gc: GeneticCode,
) -> list[BEBSite]:
    model = _MODEL_CTORS[name](gc, pi)
    weights, Qs = model.build(params=fit.params)
    if name == "M2a":
        omegas = [fit.params["omega0"], 1.0, fit.params["omega2"]]
    elif name == "M8":
        from selkit.engine.codon_model import _beta_quantiles
        beta_omegas = _beta_quantiles(fit.params["p_beta"], fit.params["q_beta"], 10).tolist()
        omegas = [float(o) for o in beta_omegas] + [fit.params["omega2"]]
    else:
        raise ValueError(f"NEB not supported for {name}")
    per_class = per_class_site_log_likelihood(
        tree=inputs.tree,
        codons=inputs.alignment.codons,
        taxon_order=inputs.alignment.taxa,
        Qs=Qs, pi=pi,
    )
    return compute_neb(
        per_class_site_logL=per_class, weights=weights, omegas=omegas,
    )


def run_site_models(
    *,
    inputs: ValidatedInputs,
    config: RunConfig,
    parallel: bool,
    progress: Optional[Callable[[str, str], None]] = None,
) -> RunResult:
    gc = GeneticCode.by_name(config.genetic_code)
    pi = estimate_f3x4(inputs.alignment.codons, gc)
    models_to_fit = config.models or _BUNDLE_DEFAULT

    # Branch-site models (Model A, Model A null) require a foreground label on
    # the tree. Guard against the common mistake of requesting ModelA without
    # also passing --foreground / --labels-file / an in-Newick #1.
    has_foreground = any(n.label != 0 for n in inputs.tree.all_nodes())
    bs_requested = [m for m in models_to_fit if m in _BRANCH_SITE_MODELS]
    if bs_requested and not has_foreground:
        raise SelkitConfigError(
            f"branch-site models {bs_requested} require a foreground clade "
            "on the tree; supply --foreground / --foreground-tips / "
            "--labels-file, or use an in-Newick #1 annotation."
        )

    engine_fits: dict[str, EngineFit] = {}
    if parallel and config.threads > 1 and len(models_to_fit) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        futures = {}
        with ProcessPoolExecutor(max_workers=config.threads) as ex:
            for name in models_to_fit:
                if name not in _MODEL_CTORS:
                    raise ValueError(f"unknown model {name!r}")
                if progress:
                    progress("start", name)
                futures[ex.submit(_run_one, name, inputs=inputs, pi=pi, cfg=config)] = name
            for fut in as_completed(futures):
                name = futures[fut]
                engine_fits[name] = fut.result()
                if progress:
                    progress("done", name)
    else:
        for name in models_to_fit:
            if name not in _MODEL_CTORS:
                raise ValueError(f"unknown model {name!r}")
            if progress:
                progress("start", name)
            engine_fits[name] = _run_one(name, inputs=inputs, pi=pi, cfg=config)
            if progress:
                progress("done", name)

    fits = {name: _engine_to_public(f) for name, f in engine_fits.items()}

    lrts: list[LRTResult] = []
    for null, alt, df, test_type in STANDARD_SITE_LRTS:
        if null in fits and alt in fits:
            r = compute_lrt(
                null=null, alt=alt,
                lnL_null=fits[null].lnL, lnL_alt=fits[alt].lnL,
                df=df, test_type=test_type,
            )
            lrts.append(LRTResult(
                null=r.null, alt=r.alt, delta_lnL=r.delta_lnL,
                df=r.df, p_value=r.p_value, test_type=r.test_type,
                significant_at_0_05=r.significant_at_0_05,
            ))

    beb: dict[str, list[BEBSite]] = {}
    for name in ("M2a", "M8"):
        if name in engine_fits:
            beb[name] = _compute_beb_for(
                name, fit=engine_fits[name], inputs=inputs, pi=pi, gc=gc,
            )

    warnings: list[str] = []
    for name, f in fits.items():
        if not f.converged:
            warnings.append(f"{name}: multi-start disagreement > {config.convergence_tol} lnL")

    return RunResult(
        config=config, family="site", fits=fits, lrts=lrts, beb=beb, warnings=warnings,
    )
