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
