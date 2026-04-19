from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from scipy.stats import chi2


@dataclass(frozen=True)
class LRTResult:
    null: str
    alt: str
    delta_lnL: float
    df: int
    p_value: float
    test_type: Literal["chi2", "mixed_chi2"]
    significant_at_0_05: bool


def compute_lrt(
    *,
    null: str,
    alt: str,
    lnL_null: float,
    lnL_alt: float,
    df: int,
    test_type: Literal["chi2", "mixed_chi2"] = "chi2",
    alpha: float = 0.05,
) -> LRTResult:
    stat = 2.0 * (lnL_alt - lnL_null)
    stat = max(0.0, stat)
    if test_type == "chi2":
        p = float(chi2.sf(stat, df))
    elif test_type == "mixed_chi2":
        p = float(0.5 * chi2.sf(stat, df))
    else:
        raise ValueError(f"unknown test_type: {test_type}")
    return LRTResult(
        null=null, alt=alt,
        delta_lnL=stat,
        df=df, p_value=p,
        test_type=test_type,
        significant_at_0_05=(p < alpha),
    )


STANDARD_SITE_LRTS: tuple[tuple[str, str, int, str], ...] = (
    ("M1a", "M2a", 2, "chi2"),
    ("M7", "M8", 2, "chi2"),
    ("M8a", "M8", 1, "mixed_chi2"),
)
