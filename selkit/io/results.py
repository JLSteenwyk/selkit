from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

from selkit.io.config import RunConfig


@dataclass(frozen=True)
class StartResult:
    seed: int
    final_lnL: float
    iterations: int
    params: dict[str, float]


@dataclass(frozen=True)
class ModelFit:
    model: str
    lnL: float
    n_params: int
    params: dict[str, float]
    branch_lengths: dict[str, float]
    starts: list[StartResult]
    converged: bool
    runtime_s: float


@dataclass(frozen=True)
class LRTResult:
    null: str
    alt: str
    delta_lnL: float
    df: int
    p_value: float
    test_type: Literal["chi2", "mixed_chi2"]
    significant_at_0_05: bool


@dataclass(frozen=True)
class BEBSite:
    site: int
    p_positive: float
    mean_omega: float


@dataclass(frozen=True)
class RunResult:
    config: RunConfig
    fits: dict[str, ModelFit]
    lrts: list[LRTResult]
    beb: dict[str, list[BEBSite]]
    warnings: list[str]


def to_json(result: RunResult) -> dict:
    from selkit.io.config import _to_primitive
    return {
        "config": _to_primitive(result.config),
        "fits": {k: asdict(v) for k, v in result.fits.items()},
        "lrts": [asdict(l) for l in result.lrts],
        "beb": {k: [asdict(s) for s in v] for k, v in result.beb.items()},
        "warnings": list(result.warnings),
    }
