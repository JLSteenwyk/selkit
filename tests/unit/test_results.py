from __future__ import annotations

import json
from pathlib import Path

from selkit.io.config import RunConfig, StrictFlags
from selkit.io.results import (
    BEBSite,
    LRTResult,
    ModelFit,
    RunResult,
    StartResult,
    emit_tsv_files,
    to_json,
)


def _minimal_config() -> RunConfig:
    return RunConfig(
        alignment=Path("/x.fa"), alignment_dir=None, tree=Path("/x.nwk"),
        foreground=None, subcommand="codeml.site-models",
        models=("M0",), tests=(), genetic_code="standard",
        output_dir=Path("/out"), threads=1, seed=0, n_starts=1,
        convergence_tol=0.5,
        strict=StrictFlags(True, False, False, False),
        selkit_version="0.0.1", git_sha=None,
    )


def test_run_result_roundtrips_through_json(tmp_path: Path) -> None:
    cfg = _minimal_config()
    fit = ModelFit(
        model="M0", lnL=-123.4, n_params=5,
        params={"omega": 0.5, "kappa": 2.0},
        branch_lengths={"bl_1": 0.1, "bl_2": 0.2},
        starts=[StartResult(seed=1, final_lnL=-123.4, iterations=20,
                            params={"omega": 0.5, "kappa": 2.0})],
        converged=True, runtime_s=0.01,
    )
    beb = {"M0": [BEBSite(site=1, p_positive=0.0, posterior_mean_omega=0.5)]}
    result = RunResult(
        config=cfg, fits={"M0": fit}, lrts=[], beb=beb, warnings=[],
    )
    path = tmp_path / "results.json"
    path.write_text(json.dumps(to_json(result), indent=2))
    loaded = json.loads(path.read_text())
    assert loaded["fits"]["M0"]["lnL"] == -123.4
    assert loaded["fits"]["M0"]["params"]["omega"] == 0.5
    assert loaded["beb"]["M0"][0]["p_positive"] == 0.0


def test_emit_tsv_files(tmp_path: Path) -> None:
    cfg = _minimal_config()
    fit = ModelFit(
        model="M0", lnL=-123.4, n_params=5,
        params={"omega": 0.5, "kappa": 2.0},
        branch_lengths={"bl_1": 0.1},
        starts=[], converged=True, runtime_s=0.01,
    )
    result = RunResult(
        config=cfg, fits={"M0": fit},
        lrts=[LRTResult("M1a", "M2a", 5.0, 2, 0.05, "chi2", True)],
        beb={"M2a": [BEBSite(1, 0.95, 3.2)]}, warnings=[],  # positional args: site, p_positive, posterior_mean_omega
    )
    emit_tsv_files(result, tmp_path)
    fits_tsv = (tmp_path / "fits.tsv").read_text().splitlines()
    assert fits_tsv[0].split("\t") == [
        "model", "lnL", "n_params", "converged", "runtime_s", "params"
    ]
    assert fits_tsv[1].split("\t")[0] == "M0"
    lrts_tsv = (tmp_path / "lrts.tsv").read_text().splitlines()
    assert lrts_tsv[0].split("\t") == [
        "null", "alt", "delta_lnL", "df", "p_value", "test_type", "significant_at_0_05"
    ]
    beb_tsv = (tmp_path / "beb_M2a.tsv").read_text().splitlines()
    assert beb_tsv[0].split("\t") == ["site", "p_positive", "posterior_mean_omega"]


def test_beb_site_uses_posterior_mean_omega():
    s = BEBSite(site=1, p_positive=0.9, posterior_mean_omega=2.3)
    assert s.posterior_mean_omega == 2.3
    assert not hasattr(s, "mean_omega")


def test_site_model_fit_tagged():
    from selkit.io.results import SiteModelFit
    f = SiteModelFit(
        model="M8", family="site", lnL=-3421.17, n_params=5,
        params={"kappa": 2.14, "omega2": 2.1},
        branch_lengths={}, starts=[], converged=True, runtime_s=0.1,
    )
    assert f.family == "site"


def test_branch_site_model_fit_has_class_proportions():
    from selkit.io.results import BranchSiteModelFit
    f = BranchSiteModelFit(
        model="ModelA", family="branch-site", lnL=-3255.44, n_params=5,
        params={"p0": 0.6, "p1": 0.25, "omega0": 0.12, "omega2": 4.3, "kappa": 2.1},
        class_proportions={"p0": 0.6, "p1": 0.25, "p2a": 0.1, "p2b": 0.05},
        branch_lengths={}, starts=[], converged=True, runtime_s=0.1,
    )
    assert f.family == "branch-site"
    assert abs(sum(f.class_proportions.values()) - 1.0) < 1e-9


def test_branch_model_fit_stub_exists():
    from selkit.io.results import BranchModelFit
    f = BranchModelFit(
        model="TwoRatios", family="branch", lnL=-100.0, n_params=3,
        params={"kappa": 2.0}, per_branch_omega=[],
        branch_lengths={}, starts=[], converged=True, runtime_s=0.1,
    )
    assert f.family == "branch"


def test_any_model_fit_union_importable():
    from selkit.io.results import AnyModelFit, SiteModelFit, BranchModelFit, BranchSiteModelFit
    # sanity: AnyModelFit union includes all three
    # (type unions don't have an `__args__` on the instance, but the module-level
    # alias should be a types.UnionType or typing.Union and include the three classes)
    import typing
    args = typing.get_args(AnyModelFit)
    assert SiteModelFit in args
    assert BranchModelFit in args
    assert BranchSiteModelFit in args
