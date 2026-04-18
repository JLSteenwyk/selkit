from __future__ import annotations

import numpy as np
import pytest

from selkit.engine.genetic_code import GeneticCode
from selkit.engine.rate_matrix import build_q, estimate_f3x4, prob_transition_matrix


def _uniform_pi(gc: GeneticCode) -> np.ndarray:
    return np.full(gc.n_sense, 1.0 / gc.n_sense)


def test_q_row_sums_to_zero() -> None:
    gc = GeneticCode.standard()
    pi = _uniform_pi(gc)
    Q = build_q(gc, omega=0.5, kappa=2.0, pi=pi)
    np.testing.assert_allclose(Q.sum(axis=1), 0.0, atol=1e-12)


def test_q_is_scaled() -> None:
    gc = GeneticCode.standard()
    pi = _uniform_pi(gc)
    Q = build_q(gc, omega=0.5, kappa=2.0, pi=pi)
    assert -float(pi @ np.diag(Q)) == pytest.approx(1.0, rel=1e-9)


def test_q_zero_for_multi_position_changes() -> None:
    gc = GeneticCode.standard()
    pi = _uniform_pi(gc)
    Q = build_q(gc, omega=1.0, kappa=1.0, pi=pi)
    i = gc.codon_to_index("ATG")
    j = gc.codon_to_index("CCC")
    assert Q[i, j] == 0.0


def test_q_respects_omega_on_nonsynonymous() -> None:
    gc = GeneticCode.standard()
    pi = _uniform_pi(gc)
    Q1 = build_q(gc, omega=1.0, kappa=1.0, pi=pi)
    Q2 = build_q(gc, omega=2.0, kappa=1.0, pi=pi)
    i_s, j_s = gc.codon_to_index("CTT"), gc.codon_to_index("CTC")
    i_n, j_n = gc.codon_to_index("ATG"), gc.codon_to_index("ACG")
    r1 = Q1[i_n, j_n] / Q1[i_s, j_s]
    r2 = Q2[i_n, j_n] / Q2[i_s, j_s]
    assert r2 / r1 == pytest.approx(2.0, rel=1e-9)


def test_prob_transition_matrix_is_stochastic() -> None:
    gc = GeneticCode.standard()
    pi = _uniform_pi(gc)
    Q = build_q(gc, omega=0.4, kappa=2.1, pi=pi)
    for t in (0.0, 0.01, 1.0, 100.0):
        P = prob_transition_matrix(Q, t)
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-9)
        assert np.all(P >= -1e-12)


def test_prob_transition_at_zero_is_identity() -> None:
    gc = GeneticCode.standard()
    pi = _uniform_pi(gc)
    Q = build_q(gc, omega=0.4, kappa=2.1, pi=pi)
    P = prob_transition_matrix(Q, 0.0)
    np.testing.assert_allclose(P, np.eye(gc.n_sense), atol=1e-9)


def test_estimate_f3x4_returns_probabilities() -> None:
    gc = GeneticCode.standard()
    idx = np.array([
        [gc.codon_to_index("ATG"), gc.codon_to_index("AAA")],
        [gc.codon_to_index("ATG"), gc.codon_to_index("AAG")],
        [gc.codon_to_index("ATG"), gc.codon_to_index("AAA")],
    ], dtype=np.int16)
    pi = estimate_f3x4(idx, gc)
    assert pi.shape == (gc.n_sense,)
    np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-12)
    assert np.all(pi > 0)
