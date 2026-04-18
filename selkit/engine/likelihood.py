from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from selkit.engine.rate_matrix import prob_transition_matrix
from selkit.errors import SelkitInputError
from selkit.io.tree import LabeledTree, Node


def _iter_postorder(root: Node) -> list[Node]:
    out: list[Node] = []

    def visit(n: Node) -> None:
        for c in n.children:
            visit(c)
        out.append(n)

    visit(root)
    return out


def _prune_tree_partials(
    tree: LabeledTree,
    codons: np.ndarray,
    taxon_order: tuple[str, ...],
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pruning with per-internal-node running scaling.

    Returns (L_root_scaled, log_scale_per_site) where L_root_scaled has each
    site's row normalized to max=1 across all codon states. lnL_site is then
    log(L_root_scaled @ pi) + log_scale_per_site, which never underflows.
    """
    n_sense = Q.shape[0]
    n_sites = codons.shape[1]

    tree_tips = {n.name for n in tree.tips if n.name}
    missing = tree_tips - set(taxon_order)
    if missing:
        raise SelkitInputError(
            f"tree tips not in taxon_order: {sorted(missing)}"
        )
    tip_to_row = {name: i for i, name in enumerate(taxon_order)}

    P_cache: dict[float, np.ndarray] = {}

    def P_for(bl: float) -> np.ndarray:
        if bl not in P_cache:
            P_cache[bl] = prob_transition_matrix(Q, bl)
        return P_cache[bl]

    partials: dict[int, np.ndarray] = {}
    log_scale = np.zeros(n_sites)

    for node in _iter_postorder(tree.root):
        if node.is_tip:
            L = np.zeros((n_sites, n_sense))
            row = tip_to_row[node.name or ""]
            for s in range(n_sites):
                c = int(codons[row, s])
                if c < 0:
                    L[s, :] = 1.0
                else:
                    L[s, c] = 1.0
            partials[node.id] = L
        else:
            L = np.ones((n_sites, n_sense))
            for child in node.children:
                bl = child.branch_length if child.branch_length is not None else 0.0
                P = P_for(bl)
                L_child = partials[child.id]
                contrib = L_child @ P.T
                L *= contrib
            row_max = L.max(axis=1)
            # Guard against all-zero rows (pathological; flag rather than silently underflow).
            safe_max = np.where(row_max > 0, row_max, 1.0)
            L = L / safe_max[:, None]
            log_scale += np.log(safe_max)
            partials[node.id] = L

    return partials[tree.root.id], log_scale


def tree_log_likelihood(
    tree: LabeledTree,
    codons: np.ndarray,
    taxon_order: tuple[str, ...],
    *,
    Q: np.ndarray,
    pi: np.ndarray,
) -> float:
    L_root, log_scale = _prune_tree_partials(tree, codons, taxon_order, Q)
    site_L = L_root @ pi
    # After running scaling, site_L is bounded below by min(pi); clip is a
    # defensive floor for pathological Q/pi, not a correctness substitute.
    log_site_L = np.log(np.clip(site_L, 1e-300, None)) + log_scale
    return float(log_site_L.sum())


def tree_log_likelihood_mixture(
    tree: LabeledTree,
    codons: np.ndarray,
    taxon_order: tuple[str, ...],
    *,
    Qs: list[np.ndarray],
    weights: list[float],
    pi: np.ndarray,
) -> float:
    per_class_log_site_L = []
    for Q in Qs:
        L_root, log_scale = _prune_tree_partials(tree, codons, taxon_order, Q)
        site_L = L_root @ pi
        per_class_log_site_L.append(
            np.log(np.clip(site_L, 1e-300, None)) + log_scale
        )
    logL_stack = np.vstack(per_class_log_site_L)
    with np.errstate(divide="ignore"):
        logW = np.log(np.asarray(weights))[:, None]
    site_log = logsumexp(logL_stack + logW, axis=0)
    return float(site_log.sum())
