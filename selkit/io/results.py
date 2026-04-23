from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
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
class SiteModelFit:
    model: str
    family: Literal["site"]
    lnL: float
    n_params: int
    params: dict[str, float]
    branch_lengths: dict[str, float]
    starts: list[StartResult]
    converged: bool
    runtime_s: float


@dataclass(frozen=True)
class BranchSiteModelFit:
    model: str
    family: Literal["branch-site"]
    lnL: float
    n_params: int
    params: dict[str, float]
    class_proportions: dict[str, float]
    branch_lengths: dict[str, float]
    starts: list[StartResult]
    converged: bool
    runtime_s: float


# Placeholder for Phase 2. Branch models emit a per-branch ω array instead of
# a scalar in `params`.
@dataclass(frozen=True)
class BranchModelFit:
    model: str
    family: Literal["branch"]
    lnL: float
    n_params: int
    params: dict[str, float]
    per_branch_omega: list[dict]  # [{branch_id, tip_set, label, paml_node_id, omega, SE}, ...]
    branch_lengths: dict[str, float]
    starts: list[StartResult]
    converged: bool
    runtime_s: float


AnyModelFit = SiteModelFit | BranchModelFit | BranchSiteModelFit


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


@dataclass(frozen=True)
class BEBSite:
    site: int
    p_positive: float
    posterior_mean_omega: float
    p_class_2a: float | None = None
    p_class_2b: float | None = None
    beb_grid_size: int | None = None


@dataclass(frozen=True)
class RunResult:
    config: RunConfig
    family: Literal["site", "branch", "branch-site"]
    fits: dict[str, ModelFit]
    lrts: list[LRTResult]
    beb: dict[str, list[BEBSite]]
    warnings: list[str]


def to_json(result: RunResult) -> dict:
    from selkit.io.config import _to_primitive
    return {
        "config": _to_primitive(result.config),
        "family": result.family,
        "fits": {k: asdict(v) for k, v in result.fits.items()},
        "lrts": [asdict(l) for l in result.lrts],
        "beb": {k: [asdict(s) for s in v] for k, v in result.beb.items()},
        "warnings": list(result.warnings),
    }


def emit_tsv_files(result: RunResult, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_rows = ["\t".join(["model", "lnL", "n_params", "converged", "runtime_s", "params"])]
    for name, fit in result.fits.items():
        fit_rows.append("\t".join([
            fit.model, f"{fit.lnL:.6f}", str(fit.n_params),
            str(fit.converged).lower(), f"{fit.runtime_s:.3f}",
            json.dumps(fit.params, sort_keys=True),
        ]))
    (output_dir / "fits.tsv").write_text("\n".join(fit_rows) + "\n")
    lrt_rows = ["\t".join(["null", "alt", "delta_lnL", "df", "p_value", "test_type", "significant_at_0_05"])]
    for l in result.lrts:
        lrt_rows.append("\t".join([
            l.null, l.alt, f"{l.delta_lnL:.6f}", str(l.df),
            f"{l.p_value:.6g}", l.test_type, str(l.significant_at_0_05).lower(),
        ]))
    (output_dir / "lrts.tsv").write_text("\n".join(lrt_rows) + "\n")
    for model, sites in result.beb.items():
        rows = ["\t".join(["site", "p_positive", "posterior_mean_omega", "p_class_2a", "p_class_2b", "beb_grid_size"])]
        for s in sites:
            p_class_2a_str = "" if s.p_class_2a is None else f"{s.p_class_2a:.6f}"
            p_class_2b_str = "" if s.p_class_2b is None else f"{s.p_class_2b:.6f}"
            grid_size_str = "" if s.beb_grid_size is None else str(s.beb_grid_size)
            rows.append("\t".join([str(s.site), f"{s.p_positive:.6f}", f"{s.posterior_mean_omega:.6f}", p_class_2a_str, p_class_2b_str, grid_size_str]))
        (output_dir / f"beb_{model}.tsv").write_text("\n".join(rows) + "\n")
