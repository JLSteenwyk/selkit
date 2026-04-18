from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from selkit.engine.rate_matrix import prob_transition_matrix
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
    n_sense: int,
) -> np.ndarray:
    """Return L_root[site, codon] partial likelihoods at the root for a single Q."""
    n_sites = codons.shape[1]
    tip_to_row = {name: i for i, name in enumerate(taxon_order)}

    P_cache: dict[float, np.ndarray] = {}

    def P_for(bl: float) -> np.ndarray:
        if bl not in P_cache:
            P_cache[bl] = prob_transition_matrix(Q, bl)
        return P_cache[bl]

    partials: dict[int, np.ndarray] = {}

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
            partials[node.id] = L
    return partials[tree.root.id]


def tree_log_likelihood(
    tree: LabeledTree,
    codons: np.ndarray,
    taxon_order: tuple[str, ...],
    *,
    Q: np.ndarray,
    pi: np.ndarray,
) -> float:
    n_sense = Q.shape[0]
    L_root = _prune_tree_partials(tree, codons, taxon_order, Q, n_sense)
    site_L = L_root @ pi
    return float(np.sum(np.log(np.clip(site_L, 1e-300, None))))


def tree_log_likelihood_mixture(
    tree: LabeledTree,
    codons: np.ndarray,
    taxon_order: tuple[str, ...],
    *,
    Qs: list[np.ndarray],
    weights: list[float],
    pi: np.ndarray,
) -> float:
    n_sense = Qs[0].shape[0]
    per_class_logL = []
    for Q in Qs:
        L_root = _prune_tree_partials(tree, codons, taxon_order, Q, n_sense)
        site_L = L_root @ pi
        per_class_logL.append(np.log(np.clip(site_L, 1e-300, None)))
    logL_stack = np.vstack(per_class_logL)
    logW = np.log(np.asarray(weights))[:, None]
    site_log = logsumexp(logL_stack + logW, axis=0)
    return float(site_log.sum())
