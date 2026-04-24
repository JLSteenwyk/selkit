from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

from scipy.stats import chi2

from selkit.io.tree import LabeledTree


DFSpec = Union[int, Literal["lazy_K", "lazy_Bminus2"]]


@dataclass(frozen=True)
class LRTResult:
    null: str
    alt: str
    delta_lnL: float
    df: int
    p_value: float
    test_type: Literal["chi2", "mixed_chi2"]
    significant_at_0_05: bool
    warning: str | None = None


def resolve_df(df: DFSpec, tree: LabeledTree) -> int:
    """Resolve a lazy-df specifier against a concrete tree."""
    if isinstance(df, int):
        return df
    if df == "lazy_K":
        return tree.n_label_classes
    if df == "lazy_Bminus2":
        return tree.n_branches - 2
    raise ValueError(f"unknown df specifier: {df!r}")


def compute_lrt(
    *,
    null: str,
    alt: str,
    lnL_null: float,
    lnL_alt: float,
    df: int,
    test_type: Literal["chi2", "mixed_chi2"] = "chi2",
    alpha: float = 0.05,
    warning: str | None = None,
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
        warning=warning,
    )


STANDARD_SITE_LRTS: tuple[tuple[str, str, DFSpec, str], ...] = (
    ("M1a", "M2a", 2, "chi2"),
    ("M7", "M8", 2, "chi2"),
    ("M8a", "M8", 1, "mixed_chi2"),
    # Branch-site test of positive selection (Zhang et al. 2005).
    # 1 df because ModelA_null fixes omega2 = 1; boundary test -> mixed 50:50 chi2.
    ("ModelA_null", "ModelA", 1, "mixed_chi2"),
)
