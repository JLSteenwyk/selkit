from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from selkit.io.config import RunConfig, StrictFlags
from selkit.io.tree import ForegroundSpec
from selkit.services.codeml.branch_models import run_branch_models
from selkit.services.codeml.branch_site import run_branch_site_models
from selkit.services.codeml.site_models import run_site_models
from selkit.services.validate import validate_inputs
from selkit.version import __version__


# v0.3 split site_models into site + branch-site services. Dispatch by the
# models requested in meta.yaml: ModelA / ModelA_null route to the branch-site
# service; TwoRatios/TwoRatiosFixed/NRatios/FreeRatios route to the branch
# service; everything else stays on site. M0 is admissible in any family.
_BRANCH_SITE_MODELS = frozenset({"ModelA", "ModelA_null"})
_BRANCH_FAMILY_MODELS = frozenset({"TwoRatios", "TwoRatiosFixed", "NRatios", "FreeRatios"})


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
    if requested & _BRANCH_FAMILY_MODELS:
        return run_branch_models(
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
    if meta.get("blocked"):
        pytest.skip(
            f"awaiting PAML fixture for {paml_case.name}: {meta['blocked']}"
        )
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
    # branch service for branch-family models; site service otherwise.
    if set(cfg.models) & _BRANCH_SITE_MODELS:
        cfg = RunConfig(**{**cfg.__dict__, "subcommand": "codeml.branch-site"})
    elif set(cfg.models) & _BRANCH_FAMILY_MODELS:
        cfg = RunConfig(**{**cfg.__dict__, "subcommand": "codeml.branch"})
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
        # Optional branch-model per-branch omega comparison.
        per_branch_exp = exp.get("per_branch_omega")
        if per_branch_exp is not None:
            got = {r["paml_node_id"]: r["omega"] for r in fit.per_branch_omega}
            for e in per_branch_exp:
                tol = max(OMEGA_ABS_TOL, OMEGA_REL_TOL * abs(e["omega"]))
                assert abs(got[e["paml_node_id"]] - e["omega"]) <= tol, (
                    f"{name} node {e['paml_node_id']}: "
                    f"omega diff {abs(got[e['paml_node_id']] - e['omega']):.4f} > {tol:.4f}"
                )

    for name, sites in (expected.get("beb") or {}).items():
        assert name in result.beb
        got = {s.site: s.p_positive for s in result.beb[name]}
        for s in sites:
            assert abs(got[s["site"]] - s["p_positive"]) <= BEB_TOL


@pytest.mark.validation
def test_free_ratios_se_non_null_on_hiv_4s(tmp_path):
    """End-to-end: FreeRatios on hiv_4s produces non-null SE for every branch.

    This guards the hess_inv -> per_branch_omega['SE'] wiring (Tasks 12 + 15)
    against regressions. Lives under tests/validation/ since it needs a real
    corpus alignment, but it does NOT compare to a PAML reference -- it only
    asserts the SE is populated.
    """
    import numpy as np
    from selkit import codeml_branch_models
    from selkit.io.tree import ForegroundSpec

    corpus = Path(__file__).parent / "corpus" / "hiv_4s"
    result = codeml_branch_models(
        alignment=corpus / "alignment.fa",
        tree=corpus / "tree.nwk",
        output_dir=tmp_path,
        models=("FreeRatios",),
        foreground=ForegroundSpec(),  # FreeRatios ignores foreground
        n_starts=2, seed=0, threads=1,
    )
    fr = result.fits["FreeRatios"]
    ses = [r["SE"] for r in fr.per_branch_omega]
    assert all(se is not None for se in ses), (
        f"FreeRatios hiv_4s: expected every branch to carry a non-null SE, "
        f"got SEs = {ses} (None entries indicate the hess_inv fallback path "
        f"kicked in -- check scipy version or see Task 12 guard)"
    )
    assert all(np.isfinite(se) and se > 0 for se in ses), (
        f"FreeRatios hiv_4s: every SE must be finite and positive, got {ses}"
    )
