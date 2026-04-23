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
