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
    family = result.family

    if family == "site":
        _emit_site_fits_tsv(result, output_dir)
    elif family == "branch":
        _emit_branch_fits_tsv(result, output_dir)
        _emit_branch_per_branch_tsv(result, output_dir)
    elif family == "branch-site":
        _emit_branch_site_fits_tsv(result, output_dir)
    else:
        raise ValueError(f"unknown family for TSV emission: {family!r}")

    _emit_lrts_tsv(result, output_dir)
    _emit_beb_tsvs(result, output_dir)


def _emit_site_fits_tsv(result: RunResult, output_dir: Path) -> None:
    rows = ["\t".join([
        "model", "lnL", "n_params", "converged",
        "kappa", "omega0", "omega2", "p0", "p1", "p", "q",
    ])]
    for name, fit in result.fits.items():
        p = fit.params
        def g(k):  # get-or-blank helper for optional params per model
            v = p.get(k)
            return "" if v is None else f"{v:.6f}"
        rows.append("\t".join([
            fit.model, f"{fit.lnL:.6f}", str(fit.n_params),
            str(fit.converged).lower(),
            g("kappa"), g("omega0"), g("omega2"), g("p0"), g("p1"),
            g("p_beta"), g("q_beta"),
        ]))
    (output_dir / "fits_site.tsv").write_text("\n".join(rows) + "\n")


def _emit_branch_fits_tsv(result: RunResult, output_dir: Path) -> None:
    rows = ["\t".join(["model", "lnL", "n_params", "converged", "kappa"])]
    for name, fit in result.fits.items():
        rows.append("\t".join([
            fit.model, f"{fit.lnL:.6f}", str(fit.n_params),
            str(fit.converged).lower(),
            f"{fit.params.get('kappa', float('nan')):.6f}",
        ]))
    (output_dir / "fits_branch.tsv").write_text("\n".join(rows) + "\n")


def _emit_branch_per_branch_tsv(result: RunResult, output_dir: Path) -> None:
    rows = ["\t".join([
        "model", "branch_id", "tip_set", "label", "paml_node_id", "omega", "SE",
    ])]
    for name, fit in result.fits.items():
        per_branch = getattr(fit, "per_branch_omega", None) or []
        for r in per_branch:
            se = "" if r["SE"] is None else f"{r['SE']:.6f}"
            rows.append("\t".join([
                fit.model, str(r["branch_id"]),
                "|".join(r["tip_set"]),
                r["label"], str(r["paml_node_id"]),
                f"{r['omega']:.6f}", se,
            ]))
    (output_dir / "fits_branch_per_branch.tsv").write_text("\n".join(rows) + "\n")


def _emit_branch_site_fits_tsv(result: RunResult, output_dir: Path) -> None:
    rows = ["\t".join([
        "model", "lnL", "n_params", "converged",
        "kappa", "omega0", "omega2", "p0", "p1", "p2a", "p2b",
    ])]
    for name, fit in result.fits.items():
        p = fit.params
        cp = getattr(fit, "class_proportions", {}) or {}
        rows.append("\t".join([
            fit.model, f"{fit.lnL:.6f}", str(fit.n_params),
            str(fit.converged).lower(),
            f"{p.get('kappa', float('nan')):.6f}",
            f"{p.get('omega0', float('nan')):.6f}",
            f"{p.get('omega2', float('nan')):.6f}",
            f"{cp.get('p0', float('nan')):.6f}",
            f"{cp.get('p1', float('nan')):.6f}",
            f"{cp.get('p2a', float('nan')):.6f}",
            f"{cp.get('p2b', float('nan')):.6f}",
        ]))
    (output_dir / "fits_branch_site.tsv").write_text("\n".join(rows) + "\n")


def _emit_lrts_tsv(result: RunResult, output_dir: Path) -> None:
    rows = ["\t".join([
        "null", "alt", "delta_lnL", "df", "p_value", "test_type",
        "significant_0_05", "warning",
    ])]
    for l in result.lrts:
        rows.append("\t".join([
            l.null, l.alt, f"{l.delta_lnL:.6f}", str(l.df),
            f"{l.p_value:.6g}", l.test_type,
            str(l.significant_at_0_05).lower(),
            getattr(l, "warning", None) or "",
        ]))
    (output_dir / "lrts.tsv").write_text("\n".join(rows) + "\n")


def _emit_beb_tsvs(result: RunResult, output_dir: Path) -> None:
    for model, sites in result.beb.items():
        rows = ["\t".join([
            "site", "p_positive", "posterior_mean_omega",
            "p_class_2a", "p_class_2b", "beb_grid_size",
        ])]
        for s in sites:
            rows.append("\t".join([
                str(s.site), f"{s.p_positive:.6f}",
                f"{s.posterior_mean_omega:.6f}",
                "" if s.p_class_2a is None else f"{s.p_class_2a:.6f}",
                "" if s.p_class_2b is None else f"{s.p_class_2b:.6f}",
                "" if s.beb_grid_size is None else str(s.beb_grid_size),
            ]))
        (output_dir / f"beb_{model}.tsv").write_text("\n".join(rows) + "\n")
