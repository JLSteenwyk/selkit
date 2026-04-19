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
