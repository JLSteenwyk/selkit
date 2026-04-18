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
