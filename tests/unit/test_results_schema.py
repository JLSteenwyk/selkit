from __future__ import annotations

import json


def test_branch_model_fit_roundtrips_through_json():
    from selkit.io.results import BranchModelFit, StartResult
    # Mix a real float SE (common case) and None (fallback / pinned case)
    # to regression-guard the JSON shape for both.
    fit = BranchModelFit(
        model="TwoRatios", family="branch",
        lnL=-100.0, n_params=3,
        params={"kappa": 2.0, "omega_bg": 0.3, "omega_fg": 2.5},
        per_branch_omega=[
            {"branch_id": 0, "tip_set": ["A"], "label": "background",
             "paml_node_id": 1, "omega": 0.3, "SE": 0.0421},
            {"branch_id": 1, "tip_set": ["A", "B"], "label": "foreground",
             "paml_node_id": 6, "omega": 2.5, "SE": None},
        ],
        branch_lengths={"bl_0": 0.1, "bl_1": 0.2},
        starts=[], converged=True, runtime_s=0.01,
    )
    # dataclasses.asdict -> json-serializable dict
    import dataclasses
    d = dataclasses.asdict(fit)
    payload = json.dumps(d)
    parsed = json.loads(payload)
    assert parsed["family"] == "branch"
    assert len(parsed["per_branch_omega"]) == 2
    assert parsed["per_branch_omega"][1]["label"] == "foreground"
    assert parsed["per_branch_omega"][0]["SE"] == 0.0421
    assert parsed["per_branch_omega"][1]["SE"] is None


def test_emit_tsv_files_splits_by_family(tmp_path):
    from selkit.io.results import (
        BranchModelFit, RunResult, StartResult, emit_tsv_files,
    )
    from selkit.io.config import RunConfig, StrictFlags
    from pathlib import Path
    fit = BranchModelFit(
        model="TwoRatios", family="branch", lnL=-100.0, n_params=3,
        params={"kappa": 2.0, "omega_bg": 0.3, "omega_fg": 2.5},
        per_branch_omega=[
            {"branch_id": 0, "tip_set": ["A"], "label": "background",
             "paml_node_id": 1, "omega": 0.3, "SE": 0.0421},
            {"branch_id": 1, "tip_set": ["A", "B"], "label": "foreground",
             "paml_node_id": 6, "omega": 2.5, "SE": None},
        ],
        branch_lengths={}, starts=[], converged=True, runtime_s=0.0,
    )
    cfg = RunConfig(
        alignment=Path("x.fa"), alignment_dir=None, tree=Path("y.nwk"),
        foreground=None, subcommand="codeml.branch",
        models=("TwoRatios",), tests=(), genetic_code="standard",
        output_dir=tmp_path, threads=1, seed=0, n_starts=1,
        convergence_tol=0.5,
        strict=StrictFlags(True, False, False, False),
        selkit_version="0.3.0", git_sha=None,
    )
    result = RunResult(
        config=cfg, family="branch",
        fits={"TwoRatios": fit}, lrts=[], beb={}, warnings=[],
    )
    emit_tsv_files(result, tmp_path)
    assert (tmp_path / "fits_branch.tsv").exists()
    assert (tmp_path / "fits_branch_per_branch.tsv").exists()
    pb = (tmp_path / "fits_branch_per_branch.tsv").read_text().splitlines()
    assert pb[0].split("\t") == [
        "model", "branch_id", "tip_set", "label", "paml_node_id", "omega", "SE",
    ]
    # tip_set uses pipe as delimiter per spec section 6.5.
    assert "A|B" in pb[2]
    # SE column: real float is formatted at %.6f; None renders as empty.
    row_bg = pb[1].split("\t")
    row_fg = pb[2].split("\t")
    assert row_bg[-1] == "0.042100"  # real SE, %.6f
    assert row_fg[-1] == ""  # pinned/fallback -> empty cell
