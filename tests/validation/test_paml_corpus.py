from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from selkit.io.config import RunConfig, StrictFlags
from selkit.io.tree import ForegroundSpec
from selkit.services.codeml.branch_site import run_branch_site_models
from selkit.services.codeml.site_models import run_site_models
from selkit.services.validate import validate_inputs
from selkit.version import __version__


# v0.3 split site_models into site + branch-site services. Dispatch by the
# models requested in meta.yaml: ModelA / ModelA_null route to the branch-site
# service; everything else stays on site.
_BRANCH_SITE_MODELS = frozenset({"ModelA", "ModelA_null"})


def _dispatch(*, inputs, config, parallel, progress):
    requested = set(config.models)
    if requested & _BRANCH_SITE_MODELS:
        if not requested.issubset(_BRANCH_SITE_MODELS):
            raise AssertionError(
                f"corpus case mixes branch-site and site-family models: {sorted(requested)}"
            )
        return run_branch_site_models(
            inputs=inputs, config=config, parallel=parallel, progress=progress,
        )
    return run_site_models(
        inputs=inputs, config=config, parallel=parallel, progress=progress,
    )


LNL_TOL = 0.01
# omega comparison uses relative tolerance: large-omega classes (e.g. M2a's
# positive-selection class with omega ~ 5-10) sit on a flatter ridge of the
# likelihood surface than small-omega classes, so PAML and selkit converge
# to slightly different optimizers' stopping points. The lnL agreement
# (within LNL_TOL) is the primary correctness signal; omega within ~0.5%
# relative is expected optimizer variance.
OMEGA_REL_TOL = 5e-3
OMEGA_ABS_TOL = 1e-3  # floor for values near 0
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
        foreground=None, subcommand="codeml.site",
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
    # Branch-site corpus cases either encode the foreground in-Newick (#1)
    # OR via meta.yaml — never both, since apply_foreground_spec rejects the
    # double-spec case. Read the tree first to detect in-Newick labels and
    # only pass an external spec when none are present.
    from selkit.io.tree import parse_newick
    raw_tree = parse_newick(tree.read_text())
    has_in_tree_labels = any(n.label != 0 for n in raw_tree.all_nodes())
    if has_in_tree_labels:
        fg_spec = ForegroundSpec()
    else:
        fg_meta = meta.get("foreground") or {}
        fg_spec = ForegroundSpec(
            tips=tuple(fg_meta.get("tips") or ()),
            mrca=tuple(fg_meta.get("mrca") or ()),
        )
    validated = validate_inputs(
        alignment_path=aln, tree_path=tree,
        foreground_spec=fg_spec,
        genetic_code_name=cfg.genetic_code,
    )
    # Use the branch-site service when meta.yaml requests ModelA / ModelA_null;
    # site service otherwise. v0.2 routed everything through run_site_models.
    if set(cfg.models) & _BRANCH_SITE_MODELS:
        cfg = RunConfig(**{**cfg.__dict__, "subcommand": "codeml.branch-site"})
    result = _dispatch(
        inputs=validated, config=cfg, parallel=False, progress=None,
    )

    for name, exp in expected["fits"].items():
        # expected.json may carry reference fits for models that this case's
        # meta.yaml deliberately omits (e.g. hiv_4s skips M7/M8 because PAML's
        # optimum lands on a beta boundary). Silently skip those — the
        # integration-tier test_paml_lnl_match.py covers static-point lnL match
        # for the omitted models.
        if name not in result.fits:
            continue
        fit = result.fits[name]
        assert abs(fit.lnL - exp["lnL"]) <= LNL_TOL, (
            f"{name}: lnL diff {abs(fit.lnL - exp['lnL']):.4f} > {LNL_TOL}"
        )
        for k, v in exp["params"].items():
            if k in ("omega", "omega0", "omega2"):
                tol = max(OMEGA_ABS_TOL, OMEGA_REL_TOL * abs(v))
                assert abs(fit.params[k] - v) <= tol, (
                    f"{name}: {k} diff {abs(fit.params[k] - v):.6f} > {tol:.6f} "
                    f"(rel {OMEGA_REL_TOL}, abs floor {OMEGA_ABS_TOL})"
                )

    for name, sites in (expected.get("beb") or {}).items():
        assert name in result.beb
        got = {s.site: s.p_positive for s in result.beb[name]}
        for s in sites:
            assert abs(got[s["site"]] - s["p_positive"]) <= BEB_TOL
