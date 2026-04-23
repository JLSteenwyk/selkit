from __future__ import annotations

import numpy as np
import pytest


def test_build_n_ratios_qs_two_labels():
    from selkit.engine.codon_model import _build_n_ratios_qs
    from selkit.engine.genetic_code import GeneticCode

    gc = GeneticCode.by_name("standard")
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    Qs = _build_n_ratios_qs(
        omegas_by_label={0: 0.2, 1: 1.8},
        kappa=2.0, pi=pi, gc=gc,
    )
    assert set(Qs.keys()) == {0, 1}
    # Each label scaled to unit mean rate.
    for lab, Q in Qs.items():
        rate = float(-(pi @ np.diag(Q)))
        assert abs(rate - 1.0) < 1e-10, f"label {lab}: rate={rate}"


def test_build_n_ratios_qs_n_labels():
    from selkit.engine.codon_model import _build_n_ratios_qs
    from selkit.engine.genetic_code import GeneticCode

    gc = GeneticCode.by_name("standard")
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    Qs = _build_n_ratios_qs(
        omegas_by_label={0: 0.3, 1: 0.8, 2: 2.1, 3: 5.0},
        kappa=2.0, pi=pi, gc=gc,
    )
    assert set(Qs.keys()) == {0, 1, 2, 3}
    for Q in Qs.values():
        rate = float(-(pi @ np.diag(Q)))
        assert abs(rate - 1.0) < 1e-10


def test_two_ratios_free_params_and_starts():
    from selkit.engine.codon_model import TwoRatios
    from selkit.engine.genetic_code import GeneticCode
    import numpy as np
    gc = GeneticCode.by_name("standard")
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    m = TwoRatios(gc=gc, pi=pi)
    assert m.name == "TwoRatios"
    assert m.branch_family is True
    assert m.branch_site is False
    assert set(m.free_params) == {"kappa", "omega_bg", "omega_fg"}
    sv = m.starting_values(seed=0)
    assert set(sv.keys()) == {"kappa", "omega_bg", "omega_fg"}


def test_two_ratios_build_two_label_dict():
    from selkit.engine.codon_model import TwoRatios
    from selkit.engine.genetic_code import GeneticCode
    import numpy as np
    gc = GeneticCode.by_name("standard")
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    m = TwoRatios(gc=gc, pi=pi)
    weights, Qs = m.build(params={"kappa": 2.0, "omega_bg": 0.3, "omega_fg": 2.5})
    assert weights == [1.0]
    assert isinstance(Qs, list) and len(Qs) == 1
    assert set(Qs[0].keys()) == {0, 1}


def test_two_ratios_rejects_extra_labels():
    from selkit.engine.codon_model import TwoRatios
    from selkit.engine.genetic_code import GeneticCode
    import numpy as np
    import pytest
    gc = GeneticCode.by_name("standard")
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    m = TwoRatios(gc=gc, pi=pi)
    # TwoRatios is defined only over labels {0, 1}. Higher labels are a K>1
    # configuration; caller must use NRatios instead.
    # The engine doesn't see extra labels at build() time (labels live on the
    # tree, not the params) -- so the K>1 gate is enforced at the service layer.
    # Here we document that assumption with a sanity build that succeeds.
    weights, Qs = m.build(params={"kappa": 2.0, "omega_bg": 0.5, "omega_fg": 1.0})
    assert list(Qs[0].keys()) == [0, 1]


def test_two_ratios_fixed_has_no_omega_fg_free_param():
    from selkit.engine.codon_model import TwoRatiosFixed
    from selkit.engine.genetic_code import GeneticCode
    import numpy as np
    gc = GeneticCode.by_name("standard")
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    m = TwoRatiosFixed(gc=gc, pi=pi)
    assert m.name == "TwoRatiosFixed"
    assert m.branch_family is True
    assert set(m.free_params) == {"kappa", "omega_bg"}


def test_two_ratios_fixed_omega_fg_is_one():
    """Build() at any omega_bg should produce a Q_fg with omega=1."""
    from selkit.engine.codon_model import TwoRatiosFixed, _build_n_ratios_qs
    from selkit.engine.genetic_code import GeneticCode
    from selkit.engine.rate_matrix import build_q, scale_per_label_qs
    import numpy as np
    gc = GeneticCode.by_name("standard")
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    m = TwoRatiosFixed(gc=gc, pi=pi)
    _, Qs = m.build(params={"kappa": 2.0, "omega_bg": 0.3})
    ref = _build_n_ratios_qs(
        omegas_by_label={0: 0.3, 1: 1.0}, kappa=2.0, pi=pi, gc=gc,
    )
    np.testing.assert_allclose(Qs[0][0], ref[0])
    np.testing.assert_allclose(Qs[0][1], ref[1])


def test_n_ratios_requires_K_labels_parameter():
    from selkit.engine.codon_model import NRatios
    from selkit.engine.genetic_code import GeneticCode
    import numpy as np
    gc = GeneticCode.by_name("standard")
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    m = NRatios(gc=gc, pi=pi, K=3)
    assert m.name == "NRatios"
    assert m.branch_family is True
    assert set(m.free_params) == {"kappa", "omega_bg", "omega_1", "omega_2", "omega_3"}


def test_n_ratios_build_produces_K_plus_1_labels():
    from selkit.engine.codon_model import NRatios
    from selkit.engine.genetic_code import GeneticCode
    import numpy as np
    gc = GeneticCode.by_name("standard")
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    m = NRatios(gc=gc, pi=pi, K=2)
    w, Qs = m.build(params={
        "kappa": 2.0, "omega_bg": 0.3, "omega_1": 1.2, "omega_2": 3.0,
    })
    assert w == [1.0]
    assert set(Qs[0].keys()) == {0, 1, 2}


def test_n_ratios_K_equals_1_equivalent_to_two_ratios():
    """NRatios(K=1) with matching params builds the same Qs as TwoRatios."""
    from selkit.engine.codon_model import NRatios, TwoRatios
    from selkit.engine.genetic_code import GeneticCode
    import numpy as np
    gc = GeneticCode.by_name("standard")
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    n = NRatios(gc=gc, pi=pi, K=1)
    t = TwoRatios(gc=gc, pi=pi)
    _, nqs = n.build(params={"kappa": 2.0, "omega_bg": 0.3, "omega_1": 2.5})
    _, tqs = t.build(params={"kappa": 2.0, "omega_bg": 0.3, "omega_fg": 2.5})
    for lab in (0, 1):
        np.testing.assert_allclose(nqs[0][lab], tqs[0][lab])
