from __future__ import annotations

import pytest


def test_branch_models_registry_has_four_models():
    from selkit.services.codeml import branch_models
    # M0 is also in the registry so M0-vs-<branch-model> LRTs auto-wire.
    expected = {"M0", "TwoRatios", "TwoRatiosFixed", "NRatios", "FreeRatios"}
    assert expected.issubset(set(branch_models.MODEL_REGISTRY.keys()))


def test_branch_models_default_lrts_include_lazy_entries():
    from selkit.services.codeml import branch_models
    lrts = branch_models.DEFAULT_LRTS
    as_tuples = [tuple(row) for row in lrts]
    assert ("M0", "TwoRatios", 1, "chi2") in as_tuples
    assert ("TwoRatiosFixed", "TwoRatios", 1, "mixed_chi2") in as_tuples
    assert ("M0", "NRatios", "lazy_K", "chi2") in as_tuples
    assert ("M0", "FreeRatios", "lazy_Bminus2", "chi2") in as_tuples


def test_branch_models_require_k_eq_1_for_two_ratios():
    from selkit.services.codeml.branch_models import _require_branch_preconditions
    from selkit.errors import SelkitConfigError
    from selkit.io.tree import parse_newick
    # Build a 2-label tree (K=2) and request TwoRatios -- must error.
    tree = parse_newick("((A:0.1,B:0.1)#1,(C:0.1,D:0.1)#2);")
    # Use a minimal ValidatedInputs stub: tree is the only attribute read.
    class _Stub:
        def __init__(self, t):
            self.tree = t
    inputs = _Stub(tree)

    class _Cfg:
        models = ("TwoRatios",)
    with pytest.raises(SelkitConfigError, match="K=1"):
        _require_branch_preconditions(inputs, _Cfg())
