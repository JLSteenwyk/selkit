from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from selkit.engine.codon_model import SiteModel
from selkit.engine.likelihood import (
    tree_log_likelihood_branch_family,
    tree_log_likelihood_mixture,
)
from selkit.engine.optimize import (
    MultiStartResult,
    Transform,
    fit_multi_start,
)
from selkit.io.tree import LabeledTree, Node


@dataclass(frozen=True)
class EngineFit:
    """Result of a single model fit on an alignment.

    ``hess_inv_diag``: per-parameter standard error in the natural (user-facing)
    parameter space, derived from scipy L-BFGS-B's inverse-Hessian diagonal
    with a delta-method Jacobian correction. ``None`` when scipy returns a
    non-LinearOperator ``hess_inv`` (older scipy or a degenerate fit). Treat
    these as a guide, not a rigorous confidence interval -- the L-BFGS-B
    inverse-Hessian is unreliable for parameters near a bound.
    """
    model: str
    lnL: float
    n_params: int
    params: dict[str, float]
    branch_lengths: dict[str, float]
    multi_start: MultiStartResult
    runtime_s: float
    hess_inv_diag: dict[str, float] | None = None


def _branch_key(node: Node) -> str:
    return f"bl_{node.id}"


def _branch_keys(tree: LabeledTree) -> list[str]:
    keys: list[str] = []
    for n in tree.all_nodes():
        if n is tree.root:
            continue
        keys.append(_branch_key(n))
    return keys


def fit_model(
    *,
    model: SiteModel,
    alignment_codons: np.ndarray,
    taxon_order: tuple[str, ...],
    tree: LabeledTree,
    n_starts: int,
    seed: int,
    convergence_tol: float = 0.5,
    max_iter: int = 500,
) -> EngineFit:
    t0 = time.perf_counter()
    branch_keys = _branch_keys(tree)
    bl_init: dict[str, float] = {}
    for n in tree.all_nodes():
        if n is tree.root:
            continue
        bl_init[_branch_key(n)] = (
            n.branch_length if n.branch_length and n.branch_length > 0 else 0.1
        )

    transform_spec: dict[str, Transform] = {k: "positive" for k in branch_keys}
    # Each SiteModel declares its own parameter-space constraints so that, e.g.,
    # M1a's ω0 is correctly bounded to (0, 1) and M2a's ω2 to (1, ∞) — enforcing
    # the model definition rather than letting L-BFGS-B drift into the wrong regime.
    for p in model.free_params:
        transform_spec[p] = model.transform_spec[p]

    def starting_values(s: int) -> dict[str, float]:
        start = dict(bl_init)
        rng = np.random.default_rng(s)
        for k in bl_init:
            start[k] *= float(rng.uniform(0.7, 1.3))
        start.update(model.starting_values(seed=s))
        return start

    def neg_lnL(params: dict[str, float]) -> float:
        for n in tree.all_nodes():
            if n is tree.root:
                continue
            n.branch_length = max(params[_branch_key(n)], 1e-8)
        model_params = {p: params[p] for p in model.free_params}
        weights, Qs = model.build(params=model_params)
        pi = getattr(model, "pi")
        if getattr(model, "branch_family", False):
            # branch family: weights = [1.0], Qs = [dict[int, ndarray]] by convention.
            return -tree_log_likelihood_branch_family(
                tree=tree,
                codons=alignment_codons,
                taxon_order=taxon_order,
                Q_by_label=Qs[0],
                pi=pi,
            )
        return -tree_log_likelihood_mixture(
            tree=tree,
            codons=alignment_codons,
            taxon_order=taxon_order,
            Qs=Qs,
            weights=weights,
            pi=pi,
        )

    result = fit_multi_start(
        neg_lnL=neg_lnL,
        starting_values=starting_values,
        transform_spec=transform_spec,
        n_starts=n_starts,
        seed=seed,
        convergence_tol=convergence_tol,
        max_iter=max_iter,
    )

    best_params = result.best.params
    model_only = {p: best_params[p] for p in model.free_params}
    bls = {k: best_params[k] for k in branch_keys}
    best_hess = result.best.hess_inv_diag
    # Filter to the model's free params (drop branch-length SEs -- those aren't
    # surfaced to the user in v0.3; per_branch_omega consumes only model params).
    if best_hess is not None:
        model_hess = {p: best_hess[p] for p in model.free_params if p in best_hess}
        if not model_hess:
            model_hess = None
    else:
        model_hess = None
    return EngineFit(
        model=model.name,
        lnL=-result.best.final_lnL,
        n_params=len(transform_spec),
        params=model_only,
        branch_lengths=bls,
        multi_start=result,
        runtime_s=time.perf_counter() - t0,
        hess_inv_diag=model_hess,
    )
