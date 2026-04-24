from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from selkit.io.config import RunConfig, StrictFlags
from selkit.io.tree import ForegroundSpec
from selkit.services.codeml.site_models import run_site_models
from selkit.services.validate import validate_inputs
from selkit.version import __version__


CORPUS = Path(__file__).parent / "corpus" / "hiv_4s_beb_m8"


@pytest.mark.validation
def test_m8_beb_matches_paml() -> None:
    expected_path = CORPUS / "expected.json"
    if not expected_path.exists():
        pytest.skip(
            "FIXTURE MISSING: tests/validation/corpus/hiv_4s_beb_m8/expected.json. "
            "Run PAML (codeml_m8.ctl with BEB enabled) locally and populate "
            "expected.json with per-site p_positive / posterior_mean_omega. "
            "See meta.yaml for the PAML command."
        )
    meta = yaml.safe_load((CORPUS / "meta.yaml").read_text())
    expected = json.loads(expected_path.read_text())

    cfg = RunConfig(
        alignment=CORPUS / "alignment.fa", alignment_dir=None,
        tree=CORPUS / "tree.nwk", foreground=None,
        subcommand="codeml.site",
        models=("M8",), tests=(),
        genetic_code=meta.get("genetic_code", "standard"),
        output_dir=CORPUS / "_out",
        threads=1, seed=meta.get("seed", 0),
        n_starts=meta.get("n_starts", 5),
        convergence_tol=0.5,
        strict=StrictFlags(True, False, False, False),
        selkit_version=__version__, git_sha=None,
        beb=True, beb_grid=int(meta.get("beb_grid", 10)),
    )
    validated = validate_inputs(
        alignment_path=cfg.alignment, tree_path=cfg.tree,
        foreground_spec=ForegroundSpec(), genetic_code_name=cfg.genetic_code,
    )
    result = run_site_models(inputs=validated, config=cfg, parallel=False, progress=None)

    tol = float(meta.get("beb_tolerance", 0.01))
    got = {s.site: (s.p_positive, s.posterior_mean_omega) for s in result.beb["M8"]}
    for rec in expected["beb"]["M8"]:
        site = int(rec["site"])
        assert site in got, f"site {site} missing from selkit BEB output"
        got_pp, got_mean_w = got[site]
        assert abs(got_pp - rec["p_positive"]) <= tol, (
            f"site {site}: p_positive diff {abs(got_pp - rec['p_positive']):.4f} > {tol}"
        )
