from __future__ import annotations

import numpy as np
import pytest

from selkit.engine.codon_model import M0
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
