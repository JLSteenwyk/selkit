from __future__ import annotations

import pytest


def test_lazy_k_resolves_to_n_label_classes():
    from selkit.io.tree import parse_newick
    from selkit.services.codeml.lrt import resolve_df
    tree = parse_newick("((A:0.1,B:0.1)#1,C:0.1,(D:0.1,E:0.1)#2);")
    assert resolve_df("lazy_K", tree) == 2


def test_lazy_bminus2_resolves_to_n_branches_minus_2():
    from selkit.io.tree import parse_newick
    from selkit.services.codeml.lrt import resolve_df
    tree = parse_newick("((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);")
    assert resolve_df("lazy_Bminus2", tree) == tree.n_branches - 2


def test_concrete_int_df_passes_through():
    from selkit.io.tree import parse_newick
    from selkit.services.codeml.lrt import resolve_df
    tree = parse_newick("(A:0.1,B:0.1);")
    assert resolve_df(2, tree) == 2


def test_free_ratios_warning_is_populated():
    """compute_lrt threads a 'warning' string through to the LRTResult."""
    from selkit.services.codeml.lrt import compute_lrt
    lrt = compute_lrt(
        null="M0", alt="FreeRatios",
        lnL_null=-100.0, lnL_alt=-80.0,
        df=37, test_type="chi2",
        warning="df=37 -- interpret per-branch omega with caution (Yang 1998).",
    )
    assert lrt.warning is not None
    assert "caution" in lrt.warning.lower()
