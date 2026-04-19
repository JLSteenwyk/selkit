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


LNL_TOL = 0.01
OMEGA_TOL = 1e-3
BL_TOL = 1e-2
BEB_TOL = 1e-3


@pytest.mark.validation
def test_case_matches_paml(paml_case: Path) -> None:
    meta = yaml.safe_load((paml_case / "meta.yaml").read_text())
    expected = json.loads((paml_case / "expected.json").read_text())
    aln = paml_case / "alignment.fa"
    tree = paml_case / "tree.nwk"

    cfg = RunConfig(
        alignment=aln, alignment_dir=None, tree=tree,
        foreground=None, subcommand="codeml.site-models",
        models=tuple(meta["models"]),
        tests=tuple(meta.get("tests") or ()),
        genetic_code=meta.get("genetic_code", "standard"),
        output_dir=paml_case / "_out",
        threads=1, seed=meta.get("seed", 0),
        n_starts=meta.get("n_starts", 3),
        convergence_tol=0.5,
        strict=StrictFlags(True, False, False, False),
        selkit_version=__version__, git_sha=None,
    )
    validated = validate_inputs(
        alignment_path=aln, tree_path=tree,
        foreground_spec=ForegroundSpec(),
        genetic_code_name=cfg.genetic_code,
    )
    result = run_site_models(
        inputs=validated, config=cfg, parallel=False, progress=None,
    )

    for name, exp in expected["fits"].items():
        fit = result.fits[name]
        assert abs(fit.lnL - exp["lnL"]) <= LNL_TOL, (
            f"{name}: lnL diff {abs(fit.lnL - exp['lnL']):.4f} > {LNL_TOL}"
        )
        for k, v in exp["params"].items():
            if k in ("omega", "omega0", "omega2"):
                assert abs(fit.params[k] - v) <= OMEGA_TOL, (
                    f"{name}: {k} diff {abs(fit.params[k] - v):.6f} > {OMEGA_TOL}"
                )

    for name, sites in (expected.get("beb") or {}).items():
        assert name in result.beb
        got = {s.site: s.p_positive for s in result.beb[name]}
        for s in sites:
            assert abs(got[s["site"]] - s["p_positive"]) <= BEB_TOL
