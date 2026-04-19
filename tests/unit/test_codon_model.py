from __future__ import annotations

import numpy as np
import pytest

from selkit.engine.codon_model import M0, M1a, M2a, M7, M8, M8a
from selkit.engine.genetic_code import GeneticCode


def test_m0_produces_one_class() -> None:
    gc = GeneticCode.standard()
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    model = M0(gc=gc, pi=pi)
    weights, Qs = model.build(params={"omega": 0.5, "kappa": 2.0})
    assert weights == [1.0]
    assert len(Qs) == 1
    assert Qs[0].shape == (gc.n_sense, gc.n_sense)


def test_m0_free_params_matches_signature() -> None:
    gc = GeneticCode.standard()
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    model = M0(gc=gc, pi=pi)
    assert model.free_params == ("omega", "kappa")


def test_m0_default_starting_values() -> None:
    gc = GeneticCode.standard()
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    model = M0(gc=gc, pi=pi)
    starts = model.starting_values(seed=0)
    assert set(starts) == {"omega", "kappa"}
    assert starts["omega"] > 0
    assert starts["kappa"] > 0


def test_m1a_weights_sum_to_one_and_omega1_is_neutral() -> None:
    gc = GeneticCode.standard()
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    model = M1a(gc=gc, pi=pi)
    weights, Qs = model.build(params={"omega0": 0.2, "p0": 0.6, "kappa": 2.0})
    assert sum(weights) == pytest.approx(1.0, rel=1e-12)
    assert len(Qs) == 2
    assert weights[0] == pytest.approx(0.6)
    assert weights[1] == pytest.approx(0.4)


def test_m2a_weights_sum_to_one_with_three_classes() -> None:
    gc = GeneticCode.standard()
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    model = M2a(gc=gc, pi=pi)
    params = {"omega0": 0.2, "omega2": 2.5, "p0": 0.5, "p1_frac": 0.6, "kappa": 2.0}
    weights, Qs = model.build(params=params)
    assert sum(weights) == pytest.approx(1.0, rel=1e-12)
    assert len(Qs) == 3
    assert weights == pytest.approx([0.5, 0.3, 0.2], rel=1e-12)


def test_m2a_omega2_constrained_above_one_via_transform() -> None:
    gc = GeneticCode.standard()
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    model = M2a(gc=gc, pi=pi)
    params = {"omega0": 0.1, "omega2": 3.0, "p0": 0.8, "p1_frac": 0.5, "kappa": 2.0}
    weights, Qs = model.build(params=params)
    assert len(Qs) == 3


def test_m7_produces_10_classes_equal_weight() -> None:
    gc = GeneticCode.standard()
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    model = M7(gc=gc, pi=pi, n_categories=10)
    params = {"p_beta": 0.5, "q_beta": 1.5, "kappa": 2.0}
    weights, Qs = model.build(params=params)
    assert len(weights) == 10
    assert len(Qs) == 10
    np.testing.assert_allclose(weights, [0.1] * 10, atol=1e-12)


def test_m8_has_k_plus_one_classes_with_correct_total_mass() -> None:
    gc = GeneticCode.standard()
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    model = M8(gc=gc, pi=pi, n_categories=10)
    params = {"p_beta": 0.5, "q_beta": 1.5, "p0": 0.9, "omega2": 2.5, "kappa": 2.0}
    weights, Qs = model.build(params=params)
    assert len(weights) == 11
    assert sum(weights) == pytest.approx(1.0, rel=1e-12)
    np.testing.assert_allclose(weights[:10], [0.09] * 10, atol=1e-12)
    assert weights[10] == pytest.approx(0.1, rel=1e-12)


def test_m8a_has_no_omega2_free_param() -> None:
    gc = GeneticCode.standard()
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    model = M8a(gc=gc, pi=pi, n_categories=10)
    assert "omega2" not in model.free_params
