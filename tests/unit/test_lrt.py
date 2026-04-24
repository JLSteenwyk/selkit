from __future__ import annotations

import pytest

from selkit.services.codeml.lrt import LRTResult, compute_lrt


def test_simple_chi2_lrt() -> None:
    r = compute_lrt(null="M1a", alt="M2a", lnL_null=-100.0, lnL_alt=-95.0, df=2)
    assert r.delta_lnL == pytest.approx(10.0)
    assert r.test_type == "chi2"
    assert 0.005 < r.p_value < 0.01
    assert r.significant_at_0_05


def test_negative_delta_clamped_to_zero() -> None:
    r = compute_lrt(null="M1a", alt="M2a", lnL_null=-100.0, lnL_alt=-100.1, df=2)
    assert r.delta_lnL == 0.0
    assert r.p_value == 1.0


def test_mixed_chi2_halves_pvalue() -> None:
    r_reg = compute_lrt(null="M8a", alt="M8", lnL_null=-100.0, lnL_alt=-97.0, df=1)
    r_mix = compute_lrt(
        null="M8a", alt="M8",
        lnL_null=-100.0, lnL_alt=-97.0, df=1, test_type="mixed_chi2",
    )
    assert r_mix.p_value == pytest.approx(r_reg.p_value / 2, rel=1e-9)
    assert r_mix.test_type == "mixed_chi2"


def test_resolve_df_lazy_K():
    from selkit.services.codeml.lrt import resolve_df
    from selkit.io.tree import parse_newick
    tree = parse_newick("((A:0.1,B:0.1)#1,(C:0.1,D:0.1)#2);")
    assert resolve_df("lazy_K", tree) == 2


def test_resolve_df_lazy_Bminus2():
    from selkit.services.codeml.lrt import resolve_df
    from selkit.io.tree import parse_newick
    tree = parse_newick("((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);")
    # 4 tips + 2 internals = 6 non-root branches; lazy_Bminus2 = 4.
    assert resolve_df("lazy_Bminus2", tree) == tree.n_branches - 2


def test_resolve_df_integer_passthrough():
    from selkit.services.codeml.lrt import resolve_df
    from selkit.io.tree import parse_newick
    tree = parse_newick("(A:0.1,B:0.1);")
    assert resolve_df(1, tree) == 1


def test_lrt_result_has_optional_warning():
    from selkit.services.codeml.lrt import LRTResult
    import dataclasses
    fields = {f.name for f in dataclasses.fields(LRTResult)}
    assert "warning" in fields
