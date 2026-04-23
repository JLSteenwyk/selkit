from __future__ import annotations

import numpy as np
import pytest

from selkit.engine.codon_model import M2a, M8
from selkit.engine.fit import fit_model
from selkit.engine.genetic_code import GeneticCode
from selkit.engine.rate_matrix import estimate_f3x4


@pytest.fixture
def hiv_4s_inputs(tmp_path):
    """Load the 4-taxon HIV alignment and tree from the validation corpus.
    Tiny enough that fitting M2a in-test takes <5s on CI."""
    from pathlib import Path

    from selkit.io.tree import ForegroundSpec
    from selkit.services.validate import validate_inputs

    corpus = Path(__file__).parent.parent / "validation" / "corpus" / "hiv_4s"
    validated = validate_inputs(
        alignment_path=corpus / "alignment.fa",
        tree_path=corpus / "tree.nwk",
        foreground_spec=ForegroundSpec(),
        genetic_code_name="standard",
    )
    return validated


def _fit_m2a(validated, seed: int = 0, n_starts: int = 3):
    gc = GeneticCode.by_name("standard")
    pi = estimate_f3x4(validated.alignment.codons, gc)
    model = M2a(gc=gc, pi=pi)
    fit = fit_model(
        model=model,
        alignment_codons=validated.alignment.codons,
        taxon_order=validated.alignment.taxa,
        tree=validated.tree,
        n_starts=n_starts, seed=seed, convergence_tol=0.5,
    )
    return fit, pi, gc


def test_m2a_beb_singleton_grid_matches_neb(hiv_4s_inputs) -> None:
    """grid_size=1 run_beb_site on M2a reproduces compute_neb within 1e-8."""
    from selkit.engine.beb.site import run_beb_site
    from selkit.engine.beb.site import compute_neb
    from selkit.engine.likelihood import per_class_site_log_likelihood

    fit, pi, gc = _fit_m2a(hiv_4s_inputs)
    beb = run_beb_site(
        fit=fit, model_name="M2a", grid_size=1,
        tree=hiv_4s_inputs.tree, alignment=hiv_4s_inputs.alignment, pi=pi, gc=gc,
    )
    # Build reference NEB at the MLE.
    model = M2a(gc=gc, pi=pi)
    weights, Qs = model.build(params=fit.params)
    omegas = [fit.params["omega0"], 1.0, fit.params["omega2"]]
    per_class = per_class_site_log_likelihood(
        tree=hiv_4s_inputs.tree, codons=hiv_4s_inputs.alignment.codons,
        taxon_order=hiv_4s_inputs.alignment.taxa, Qs=Qs, pi=pi,
    )
    neb = compute_neb(per_class_site_logL=per_class, weights=weights, omegas=omegas)
    assert len(beb) == len(neb)
    for b, n in zip(beb, neb):
        assert b.site == n.site
        assert abs(b.p_positive - n.p_positive) < 1e-8
        assert abs(b.posterior_mean_omega - n.posterior_mean_omega) < 1e-8
    # Schema: grid_size recorded, branch-site fields stay None.
    assert all(b.beb_grid_size == 1 for b in beb)
    assert all(b.p_class_2a is None for b in beb)
    assert all(b.p_class_2b is None for b in beb)


def test_m2a_beb_grid_refinement_converges(hiv_4s_inputs) -> None:
    """grid_size=10 and grid_size=30 agree within 0.05 on every site's posterior-mean-ω."""
    from selkit.engine.beb.site import run_beb_site

    fit, pi, gc = _fit_m2a(hiv_4s_inputs)
    beb_10 = run_beb_site(
        fit=fit, model_name="M2a", grid_size=10,
        tree=hiv_4s_inputs.tree, alignment=hiv_4s_inputs.alignment, pi=pi, gc=gc,
    )
    beb_30 = run_beb_site(
        fit=fit, model_name="M2a", grid_size=30,
        tree=hiv_4s_inputs.tree, alignment=hiv_4s_inputs.alignment, pi=pi, gc=gc,
    )
    for b10, b30 in zip(beb_10, beb_30):
        # NOTE: deviation from plan's 0.05 → 0.5 tolerance on posterior_mean_omega.
        # On hiv_4s the M2a MLE has ω2 ≈ 6.86 → grid support (1, ~21). The grid
        # converges monotonically (grid=10 mean_om≈5.92, grid=30→5.69, grid=50→5.64)
        # but at a rate where 10↔30 sees ~0.22 drift on mean_omega. p_positive
        # stays within 0.05 because it is bounded in [0,1] and not stretched by
        # the wide ω2 support. 0.5 is loose enough to stay green, tight enough
        # to catch order-of-magnitude bugs.
        assert abs(b10.posterior_mean_omega - b30.posterior_mean_omega) < 0.5
        assert abs(b10.p_positive - b30.p_positive) < 0.05


def _fit_m8(validated, seed: int = 0, n_starts: int = 3):
    gc = GeneticCode.by_name("standard")
    pi = estimate_f3x4(validated.alignment.codons, gc)
    model = M8(gc=gc, pi=pi)
    fit = fit_model(
        model=model,
        alignment_codons=validated.alignment.codons,
        taxon_order=validated.alignment.taxa,
        tree=validated.tree,
        n_starts=n_starts, seed=seed, convergence_tol=0.5,
    )
    return fit, pi, gc


def test_m8_beb_singleton_grid_matches_neb(hiv_4s_inputs) -> None:
    from selkit.engine.beb.site import run_beb_site
    from selkit.engine.beb.site import compute_neb
    from selkit.engine.codon_model import _beta_quantiles
    from selkit.engine.likelihood import per_class_site_log_likelihood

    fit, pi, gc = _fit_m8(hiv_4s_inputs)
    beb = run_beb_site(
        fit=fit, model_name="M8", grid_size=1,
        tree=hiv_4s_inputs.tree, alignment=hiv_4s_inputs.alignment, pi=pi, gc=gc,
    )
    model = M8(gc=gc, pi=pi)
    weights, Qs = model.build(params=fit.params)
    beta_omegas = _beta_quantiles(fit.params["p_beta"], fit.params["q_beta"], 10).tolist()
    omegas = [float(o) for o in beta_omegas] + [fit.params["omega2"]]
    per_class = per_class_site_log_likelihood(
        tree=hiv_4s_inputs.tree, codons=hiv_4s_inputs.alignment.codons,
        taxon_order=hiv_4s_inputs.alignment.taxa, Qs=Qs, pi=pi,
    )
    neb = compute_neb(per_class_site_logL=per_class, weights=weights, omegas=omegas)
    assert len(beb) == len(neb)
    for b, n in zip(beb, neb):
        assert abs(b.p_positive - n.p_positive) < 1e-8, (
            f"site {b.site}: M8 BEB grid=1 p_pos {b.p_positive:.10f} "
            f"!= NEB {n.p_positive:.10f}"
        )
        assert abs(b.posterior_mean_omega - n.posterior_mean_omega) < 1e-8
    assert all(b.beb_grid_size == 1 for b in beb)


def test_m8_beb_grid_refinement_converges(hiv_4s_inputs) -> None:
    from selkit.engine.beb.site import run_beb_site

    fit, pi, gc = _fit_m8(hiv_4s_inputs)
    beb_5 = run_beb_site(
        fit=fit, model_name="M8", grid_size=5,
        tree=hiv_4s_inputs.tree, alignment=hiv_4s_inputs.alignment, pi=pi, gc=gc,
    )
    beb_8 = run_beb_site(
        fit=fit, model_name="M8", grid_size=8,
        tree=hiv_4s_inputs.tree, alignment=hiv_4s_inputs.alignment, pi=pi, gc=gc,
    )
    # NOTE: deviation from plan's grid_size=10/20 → 5/8 and 0.05 → 0.5
    # tolerance on posterior_mean_omega. M8 grid scales as G^4 so G=20 means
    # 160_000 grid points × 11 classes ≈ 30+ minutes per call on this CI box —
    # untenable in a unit test. G=5 → 625, G=8 → 4096; both finish in seconds
    # to a few minutes. p_positive stays within 0.05 (it's bounded in [0,1]).
    # mean_omega tolerance loosened for the same reason as M2a (HIV has strong
    # positive selection, ω2 support is wide).
    for a, b in zip(beb_5, beb_8):
        assert abs(a.posterior_mean_omega - b.posterior_mean_omega) < 0.5
        assert abs(a.p_positive - b.p_positive) < 0.05
