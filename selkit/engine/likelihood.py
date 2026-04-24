from __future__ import annotations

from typing import Union

import numpy as np
from scipy.special import logsumexp

from selkit.engine.rate_matrix import prob_transition_matrix
from selkit.errors import SelkitInputError
from selkit.io.tree import LabeledTree, Node


# A "class Q" is either a single ndarray (homogeneous across branches — site
# models) or a dict mapping branch label to an ndarray (per-label — branch-site
# models). Callers use whichever is natural; _prune_tree_partials normalises.
ClassQ = Union[np.ndarray, dict[int, np.ndarray]]


def _iter_postorder(root: Node) -> list[Node]:
    out: list[Node] = []

    def visit(n: Node) -> None:
        for c in n.children:
            visit(c)
        out.append(n)

    visit(root)
    return out


def _normalize_class_q(class_q: ClassQ, tree: LabeledTree) -> dict[int, np.ndarray]:
    """Return a {label: Q} dict covering every label present in the tree.

    A plain ndarray is broadcast — same Q on every branch. A dict must already
    contain an entry for every label seen in the tree.
    """
    labels_in_tree = {n.label for n in tree.all_nodes()}
    if isinstance(class_q, np.ndarray):
        return {label: class_q for label in labels_in_tree}
    missing = labels_in_tree - set(class_q.keys())
    if missing:
        raise ValueError(
            f"branch-site Qs missing entries for labels {sorted(missing)}"
        )
    return class_q


def _prune_tree_partials(
    tree: LabeledTree,
    codons: np.ndarray,
    taxon_order: tuple[str, ...],
    Q: ClassQ,
) -> tuple[np.ndarray, np.ndarray]:
    """Pruning with per-internal-node running scaling.

    ``Q`` is either a single ndarray (site model: same Q on all branches) or a
    dict[label, ndarray] (branch-site model: Q depends on ``Node.label`` of the
    branch below the parent).

    Returns (L_root_scaled, log_scale_per_site) where L_root_scaled has each
    site's row normalized to max=1 across all codon states. lnL_site is then
    log(L_root_scaled @ pi) + log_scale_per_site, which never underflows.
    """
    Qs_by_label = _normalize_class_q(Q, tree)
    any_Q = next(iter(Qs_by_label.values()))
    n_sense = any_Q.shape[0]
    n_sites = codons.shape[1]

    tree_tips = {n.name for n in tree.tips if n.name}
    missing = tree_tips - set(taxon_order)
    if missing:
        raise SelkitInputError(
            f"tree tips not in taxon_order: {sorted(missing)}"
        )
    tip_to_row = {name: i for i, name in enumerate(taxon_order)}

    # Cache P(t) by (label, branch length) so branches that share both reuse
    # the same matrix exponential across site classes.
    P_cache: dict[tuple[int, float], np.ndarray] = {}

    def P_for(label: int, bl: float) -> np.ndarray:
        key = (label, bl)
        if key not in P_cache:
            P_cache[key] = prob_transition_matrix(Qs_by_label[label], bl)
        return P_cache[key]

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
                # The branch leading into ``child`` carries the label on ``child``.
                P = P_for(child.label, bl)
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
    Q: ClassQ,
    pi: np.ndarray,
) -> float:
    L_root, log_scale = _prune_tree_partials(tree, codons, taxon_order, Q)
    site_L = L_root @ pi
    # After running scaling, site_L is bounded below by min(pi); clip is a
    # defensive floor for pathological Q/pi, not a correctness substitute.
    log_site_L = np.log(np.clip(site_L, 1e-300, None)) + log_scale
    return float(log_site_L.sum())


def tree_log_likelihood_branch_family(
    tree: LabeledTree,
    codons: np.ndarray,
    taxon_order: tuple[str, ...],
    *,
    Q_by_label: dict[int, np.ndarray],
    pi: np.ndarray,
) -> float:
    """Total lnL for a branch-family model (Yang 1998): no site-class loop.

    Thin wrapper around :func:`tree_log_likelihood` that exists to make the
    single-class, per-label call shape explicit. Equivalent to
    ``tree_log_likelihood(Q=Q_by_label)``; kept as a distinct entry point so
    fit / BEB code paths for branch family, site mixture, and branch-site are
    all greppable by name.
    """
    return tree_log_likelihood(
        tree, codons, taxon_order, Q=Q_by_label, pi=pi,
    )


def tree_log_likelihood_mixture(
    tree: LabeledTree,
    codons: np.ndarray,
    taxon_order: tuple[str, ...],
    *,
    Qs: list[ClassQ],
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


def per_class_site_log_likelihood(
    tree: LabeledTree,
    codons: np.ndarray,
    taxon_order: tuple[str, ...],
    *,
    Qs: list[ClassQ],
    pi: np.ndarray,
) -> np.ndarray:
    """Return per-class per-site log-likelihoods with shape (n_classes, n_sites).

    Uses the same running-scale pruning as tree_log_likelihood_mixture so that
    sites with very small partial likelihoods do not underflow to zero.
    """
    rows = []
    for Q in Qs:
        L_root, log_scale = _prune_tree_partials(tree, codons, taxon_order, Q)
        site_L = L_root @ pi
        rows.append(np.log(np.clip(site_L, 1e-300, None)) + log_scale)
    return np.vstack(rows)
