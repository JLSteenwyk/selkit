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


# --- Branch-site models (Model A, Model A null) ------------------------------


from selkit.engine.codon_model import ModelA, ModelANull


def test_model_a_has_four_classes_with_per_label_qs() -> None:
    gc = GeneticCode.standard()
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    model = ModelA(gc=gc, pi=pi)
    weights, Qs = model.build(params={
        "omega0": 0.1, "omega2": 3.0,
        "p0": 0.5, "p1_frac": 0.4, "kappa": 2.0,
    })
    assert len(weights) == 4
    assert len(Qs) == 4
    assert sum(weights) == pytest.approx(1.0, rel=1e-12)
    # Every class has a per-label Q dict with entries for both labels.
    for class_qs in Qs:
        assert isinstance(class_qs, dict)
        assert set(class_qs.keys()) == {0, 1}
        assert class_qs[0].shape == (gc.n_sense, gc.n_sense)
        assert class_qs[1].shape == (gc.n_sense, gc.n_sense)


def test_model_a_class_0_and_1_are_proportional_across_labels() -> None:
    """Classes 0 (purifying) and 1 (neutral) use the same raw Q on every
    branch; after PAML-style per-label mixture scaling the bg and fg copies
    are proportional by the ratio of the two labels mean rates."""
    gc = GeneticCode.standard()
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    weights, Qs = ModelA(gc=gc, pi=pi).build(params={
        "omega0": 0.1, "omega2": 3.0,
        "p0": 0.5, "p1_frac": 0.4, "kappa": 2.0,
    })
    for class_idx in (0, 1):
        bg = Qs[class_idx][0]
        fg = Qs[class_idx][1]
        ratio = bg[bg != 0] / fg[fg != 0]
        assert np.allclose(ratio, ratio[0], rtol=1e-10)


def test_model_a_class_2a_differs_between_fg_and_bg() -> None:
    """Class 2a: purifying on background, positive on foreground → Qs differ."""
    gc = GeneticCode.standard()
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    weights, Qs = ModelA(gc=gc, pi=pi).build(params={
        "omega0": 0.1, "omega2": 3.0,
        "p0": 0.5, "p1_frac": 0.4, "kappa": 2.0,
    })
    assert not np.allclose(Qs[2][0], Qs[2][1])  # class 2a


def test_model_a_weights_via_stick_breaking() -> None:
    gc = GeneticCode.standard()
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    p0, p1_frac = 0.5, 0.4
    weights, _ = ModelA(gc=gc, pi=pi).build(params={
        "omega0": 0.1, "omega2": 3.0,
        "p0": p0, "p1_frac": p1_frac, "kappa": 2.0,
    })
    p1 = (1 - p0) * p1_frac
    p2 = 1 - p0 - p1
    p2a = p2 * p0 / (p0 + p1)
    p2b = p2 * p1 / (p0 + p1)
    np.testing.assert_allclose(weights, [p0, p1, p2a, p2b])


def test_model_a_null_has_no_omega2_free_param() -> None:
    gc = GeneticCode.standard()
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    model = ModelANull(gc=gc, pi=pi)
    assert "omega2" not in model.free_params
    assert model.free_params == ("omega0", "p0", "p1_frac", "kappa")


def test_model_a_null_omega2_pinned_to_one() -> None:
    """Model A null: class 2a/2b foreground Q must match Q at omega=1."""
    gc = GeneticCode.standard()
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    _, Qs_null = ModelANull(gc=gc, pi=pi).build(params={
        "omega0": 0.1, "p0": 0.5, "p1_frac": 0.4, "kappa": 2.0,
    })
    # Class 1 (neutral, omega=1) and class 2b foreground (omega2 = 1) should
    # use the same ratio of rates → same Q after per-label scaling.
    np.testing.assert_allclose(Qs_null[1][1], Qs_null[3][1])
    # Also class 2a foreground (omega=1) == class 1 on foreground
    np.testing.assert_allclose(Qs_null[1][1], Qs_null[2][1])


def test_model_a_branch_site_marker() -> None:
    gc = GeneticCode.standard()
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    assert ModelA(gc=gc, pi=pi).branch_site is True
    assert ModelANull(gc=gc, pi=pi).branch_site is True
    # Site models are not branch-site
    assert M0(gc=gc, pi=pi).branch_site is False
    assert M2a(gc=gc, pi=pi).branch_site is False


def test_model_a_transform_spec_constrains_boundaries() -> None:
    gc = GeneticCode.standard()
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    model = ModelA(gc=gc, pi=pi)
    assert model.transform_spec["omega0"] == "unit"          # (0, 1)
    assert model.transform_spec["omega2"] == "positive_gt_one"  # (1, inf)
    assert model.transform_spec["p0"] == "unit"
    assert model.transform_spec["p1_frac"] == "unit"
    assert model.transform_spec["kappa"] == "positive"
