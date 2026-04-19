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
    beb = {"M0": [BEBSite(site=1, p_positive=0.0, mean_omega=0.5)]}
    result = RunResult(
        config=cfg, fits={"M0": fit}, lrts=[], beb=beb, warnings=[],
    )
    path = tmp_path / "results.json"
    path.write_text(json.dumps(to_json(result), indent=2))
    loaded = json.loads(path.read_text())
    assert loaded["fits"]["M0"]["lnL"] == -123.4
    assert loaded["fits"]["M0"]["params"]["omega"] == 0.5
    assert loaded["beb"]["M0"][0]["p_positive"] == 0.0
