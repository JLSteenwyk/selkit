from __future__ import annotations

from pathlib import Path

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


def test_free_ratios_precondition_assigns_unique_labels_in_parent():
    """C1 regression: assign_unique_branch_labels must run in the parent
    process so ProcessPoolExecutor workers inherit the labelling via pickle.

    Before the fix, the label-rewrite happened inside _mk_free_ratios which
    runs in the worker subprocess; the parent's tree still had every label==0
    and per_branch_omega reported the same omega for every branch.
    """
    from selkit.services.codeml.branch_models import _require_branch_preconditions
    from selkit.io.tree import parse_newick

    tree = parse_newick("((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);")
    # All labels start at 0.
    assert all(n.label == 0 for n in tree.all_nodes())

    class _Stub:
        def __init__(self, t):
            self.tree = t

    class _Cfg:
        models = ("M0", "FreeRatios")

    _require_branch_preconditions(_Stub(tree), _Cfg())

    # After precondition: non-root branches have unique labels (root-adjacent
    # pair share one label under merge_root=True).
    non_root_labels = [n.label for n in tree.all_nodes() if n is not tree.root]
    # 2 root-adjacent + 4 tip branches under merge_root: 5 distinct labels.
    assert len(set(non_root_labels)) >= 4


def test_free_ratios_parallel_no_corruption(tmp_path):
    """C1 regression (end-to-end): FreeRatios in parallel mode reports
    distinct per-branch omegas, not the same omega for every branch.
    """
    from selkit import codeml_branch_models

    corpus = (
        Path(__file__).parent.parent / "validation" / "corpus" / "hiv_4s"
    )
    out = tmp_path / "out"
    result = codeml_branch_models(
        alignment=corpus / "alignment.fa",
        tree=corpus / "tree.nwk",
        output_dir=out,
        models=("M0", "FreeRatios"),
        threads=2,
        n_starts=1,
        seed=0,
    )
    fit = result.fits["FreeRatios"]
    omegas = [r["omega"] for r in fit.per_branch_omega]
    assert len(set(omegas)) > 1, (
        f"FreeRatios parallel mode produced identical omegas across all "
        f"branches: {omegas}"
    )
