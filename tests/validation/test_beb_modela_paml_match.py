from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from selkit.io.config import RunConfig, StrictFlags
from selkit.io.tree import ForegroundSpec
from selkit.services.codeml.branch_site import run_branch_site_models
from selkit.services.validate import validate_inputs
from selkit.version import __version__


CORPUS = Path(__file__).parent / "corpus" / "lysozyme_beb_modelA"


@pytest.mark.validation
def test_modela_beb_matches_paml() -> None:
    expected_path = CORPUS / "expected.json"
    if not expected_path.exists():
        pytest.skip(
            "FIXTURE MISSING: tests/validation/corpus/lysozyme_beb_modelA/expected.json. "
            "Run PAML (codeml.ctl model=2 NSsites=2 with BEB) locally and populate "
            "expected.json with per-site p_class_2a, p_class_2b, p_positive. "
            "See meta.yaml."
        )
    meta = yaml.safe_load((CORPUS / "meta.yaml").read_text())
    expected = json.loads(expected_path.read_text())

    fg_meta = meta.get("foreground") or {}
    fg_spec = ForegroundSpec(mrca=tuple(fg_meta.get("mrca") or ()))

    cfg = RunConfig(
        alignment=CORPUS / "alignment.fa", alignment_dir=None,
        tree=CORPUS / "tree.nwk", foreground=None,
        subcommand="codeml.branch-site",
        models=("ModelA",), tests=(),
        genetic_code=meta.get("genetic_code", "standard"),
        output_dir=CORPUS / "_out",
        threads=1, seed=meta.get("seed", 0),
        n_starts=meta.get("n_starts", 3),
        convergence_tol=0.5,
        strict=StrictFlags(True, False, False, False),
        selkit_version=__version__, git_sha=None,
        beb=True, beb_grid=int(meta.get("beb_grid", 10)),
    )
    # The lysozyme tree carries #1 in-Newick foreground labels; if the test
    # is invoked with an external ForegroundSpec selkit will reject the
    # double-spec. Detect in-tree labels and prefer them.
    from selkit.io.tree import parse_newick
    raw = parse_newick(cfg.tree.read_text())
    if any(n.label != 0 for n in raw.all_nodes()):
        fg_spec = ForegroundSpec()
    validated = validate_inputs(
        alignment_path=cfg.alignment, tree_path=cfg.tree,
        foreground_spec=fg_spec, genetic_code_name=cfg.genetic_code,
    )
    result = run_branch_site_models(inputs=validated, config=cfg, parallel=False, progress=None)

    tol = float(meta.get("beb_tolerance", 0.01))
    got = {
        s.site: (s.p_positive, s.p_class_2a, s.p_class_2b, s.posterior_mean_omega)
        for s in result.beb["ModelA"]
    }
    for rec in expected["beb"]["ModelA"]:
        site = int(rec["site"])
        assert site in got, f"site {site} missing"
        got_pp, got_2a, got_2b, _ = got[site]
        assert abs(got_pp - rec["p_positive"]) <= tol, (
            f"site {site}: p_positive diff {abs(got_pp - rec['p_positive']):.4f} > {tol}"
        )
        # Per-class posteriors: assert if PAML fixture provides them.
        if "p_class_2a" in rec:
            assert abs(got_2a - rec["p_class_2a"]) <= tol
        if "p_class_2b" in rec:
            assert abs(got_2b - rec["p_class_2b"]) <= tol
