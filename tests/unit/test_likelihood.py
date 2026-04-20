from __future__ import annotations

import numpy as np

from selkit.engine.genetic_code import GeneticCode
from selkit.engine.likelihood import tree_log_likelihood, tree_log_likelihood_mixture
from selkit.engine.rate_matrix import build_q
from selkit.io.tree import parse_newick


def _uniform_pi(gc: GeneticCode) -> np.ndarray:
    return np.full(gc.n_sense, 1.0 / gc.n_sense)


def test_lnl_on_constant_sites_is_finite() -> None:
    gc = GeneticCode.standard()
    tree = parse_newick("(a:0.1,b:0.1,c:0.1):0.0;")
    pi = _uniform_pi(gc)
    Q = build_q(gc, omega=0.5, kappa=2.0, pi=pi)
    idx = gc.codon_to_index("ATG")
    codons = np.full((3, 2), idx, dtype=np.int16)
    lnL = tree_log_likelihood(tree, codons, taxon_order=("a", "b", "c"), Q=Q, pi=pi)
    assert np.isfinite(lnL)


def test_gap_tip_marginalizes() -> None:
    gc = GeneticCode.standard()
    tree = parse_newick("(a:0.1,b:0.1,c:0.1):0.0;")
    pi = _uniform_pi(gc)
    Q = build_q(gc, omega=0.5, kappa=2.0, pi=pi)
    idx = gc.codon_to_index("ATG")
    all_observed = np.full((3, 1), idx, dtype=np.int16)
    with_gap = all_observed.copy()
    with_gap[2, 0] = -1
    lnL_obs = tree_log_likelihood(tree, all_observed, ("a", "b", "c"), Q=Q, pi=pi)
    lnL_gap = tree_log_likelihood(tree, with_gap, ("a", "b", "c"), Q=Q, pi=pi)
    assert lnL_gap > lnL_obs - 10


def test_mixture_matches_single_class_with_unit_weight() -> None:
    gc = GeneticCode.standard()
    tree = parse_newick("(a:0.1,b:0.1,c:0.1):0.0;")
    pi = _uniform_pi(gc)
    Q = build_q(gc, omega=0.5, kappa=2.0, pi=pi)
    idx = gc.codon_to_index("ATG")
    codons = np.full((3, 2), idx, dtype=np.int16)
    lnL_single = tree_log_likelihood(tree, codons, ("a", "b", "c"), Q=Q, pi=pi)
    lnL_mix = tree_log_likelihood_mixture(
        tree, codons, ("a", "b", "c"),
        Qs=[Q], weights=[1.0], pi=pi,
    )
    assert lnL_mix == pytest.approx(lnL_single, rel=1e-12)


import pytest  # noqa: E402


def test_lnl_on_bifurcating_tree_is_finite() -> None:
    gc = GeneticCode.standard()
    tree = parse_newick("((a:0.1,b:0.1):0.1,(c:0.1,d:0.1):0.1):0.0;")
    pi = _uniform_pi(gc)
    Q = build_q(gc, omega=0.5, kappa=2.0, pi=pi)
    idx = gc.codon_to_index("ATG")
    codons = np.full((4, 2), idx, dtype=np.int16)
    lnL = tree_log_likelihood(
        tree, codons, ("a", "b", "c", "d"), Q=Q, pi=pi,
    )
    assert np.isfinite(lnL)


def test_missing_taxon_in_order_raises() -> None:
    from selkit.errors import SelkitInputError
    gc = GeneticCode.standard()
    tree = parse_newick("(a:0.1,b:0.1,c:0.1):0.0;")
    pi = _uniform_pi(gc)
    Q = build_q(gc, omega=0.5, kappa=2.0, pi=pi)
    idx = gc.codon_to_index("ATG")
    codons = np.full((2, 1), idx, dtype=np.int16)
    with pytest.raises(SelkitInputError, match=r"not in taxon_order"):
        tree_log_likelihood(tree, codons, ("a", "b"), Q=Q, pi=pi)


def test_lnl_finite_on_large_tree_where_naive_underflows() -> None:
    # 300-taxon star tree, long branches (t=10). With this many branches each
    # contributing ~1/61 to the product, the pre-scaled site_L is far below
    # 1e-300 (theoretically ~10^-534). The naive clip-based pruner would
    # substitute 1e-300 and report lnL ~= -690 (the clip floor). Running-scale
    # pruning should return a lnL strictly less than -690, proving partial
    # likelihoods below the float floor are handled correctly.
    gc = GeneticCode.standard()
    n_taxa = 300
    taxa = tuple(f"t{i}" for i in range(n_taxa))
    newick = "(" + ",".join(f"{name}:10.0" for name in taxa) + "):0.0;"
    tree = parse_newick(newick)
    pi = _uniform_pi(gc)
    Q = build_q(gc, omega=0.5, kappa=2.0, pi=pi)
    idx_atg = gc.codon_to_index("ATG")
    codons = np.full((n_taxa, 1), idx_atg, dtype=np.int16)
    lnL = tree_log_likelihood(tree, codons, taxa, Q=Q, pi=pi)
    assert np.isfinite(lnL)
    assert lnL < -690.0


from selkit.engine.likelihood import per_class_site_log_likelihood


def test_per_class_site_log_likelihood_shape() -> None:
    gc = GeneticCode.standard()
    tree = parse_newick("(a:0.1,b:0.1,c:0.1):0.0;")
    pi = _uniform_pi(gc)
    Q0 = build_q(gc, omega=0.2, kappa=2.0, pi=pi)
    Q1 = build_q(gc, omega=1.0, kappa=2.0, pi=pi)
    idx = gc.codon_to_index("ATG")
    codons = np.full((3, 4), idx, dtype=np.int16)
    out = per_class_site_log_likelihood(
        tree, codons, ("a", "b", "c"), Qs=[Q0, Q1], pi=pi,
    )
    assert out.shape == (2, 4)
