from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from selkit.engine.codon_model import SiteModel
from selkit.engine.likelihood import tree_log_likelihood_mixture
from selkit.engine.optimize import (
    MultiStartResult,
    Transform,
    fit_multi_start,
)
from selkit.io.tree import LabeledTree, Node


@dataclass(frozen=True)
class EngineFit:
    model: str
    lnL: float
    n_params: int
    params: dict[str, float]
    branch_lengths: dict[str, float]
    multi_start: MultiStartResult
    runtime_s: float


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
    model_transforms: dict[str, Transform] = {}
    _POSITIVE = {"omega", "omega0", "omega2", "kappa", "q_beta", "p_beta"}
    for p in model.free_params:
        model_transforms[p] = "positive" if p in _POSITIVE else "unit"
    transform_spec.update(model_transforms)

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
    return EngineFit(
        model=model.name,
        lnL=-result.best.final_lnL,
        n_params=len(transform_spec),
        params=model_only,
        branch_lengths=bls,
        multi_start=result,
        runtime_s=time.perf_counter() - t0,
    )
