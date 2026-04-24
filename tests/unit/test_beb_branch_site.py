from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def lysozyme_inputs():
    from selkit.io.tree import ForegroundSpec
    from selkit.services.validate import validate_inputs

    corpus = Path(__file__).parent.parent / "validation" / "corpus" / "lysozyme_branchsite"
    # Lysozyme tree has #1 in-Newick foreground labels; pass an empty spec.
    validated = validate_inputs(
        alignment_path=corpus / "alignment.fa",
        tree_path=corpus / "tree.nwk",
        foreground_spec=ForegroundSpec(),
        genetic_code_name="standard",
    )
    return validated


def _fit_model_a(validated, seed: int = 0, n_starts: int = 3):
    from selkit.engine.codon_model import ModelA
    from selkit.engine.fit import fit_model
    from selkit.engine.genetic_code import GeneticCode
    from selkit.engine.rate_matrix import estimate_f3x4

    gc = GeneticCode.by_name("standard")
    pi = estimate_f3x4(validated.alignment.codons, gc)
    model = ModelA(gc=gc, pi=pi)
    fit = fit_model(
        model=model,
        alignment_codons=validated.alignment.codons,
        taxon_order=validated.alignment.taxa,
        tree=validated.tree,
        n_starts=n_starts, seed=seed, convergence_tol=0.5,
    )
    return fit, pi, gc


def test_model_a_beb_singleton_matches_neb(lysozyme_inputs) -> None:
    """grid_size=1 run_beb_branch_site on ModelA reproduces the branch-site NEB."""
    from selkit.engine.beb.branch_site import run_beb_branch_site
    from selkit.engine.beb.site import compute_neb
    from selkit.engine.codon_model import ModelA
    from selkit.engine.likelihood import per_class_site_log_likelihood

    fit, pi, gc = _fit_model_a(lysozyme_inputs)
    beb = run_beb_branch_site(
        fit=fit, grid_size=1,
        tree=lysozyme_inputs.tree, alignment=lysozyme_inputs.alignment, pi=pi, gc=gc,
    )
    # Reference NEB on the 4-class ModelA mixture.
    model = ModelA(gc=gc, pi=pi)
    weights, Qs = model.build(params=fit.params)
    # Classes: 0 (bg purifying), 1 (bg neutral), 2a (bg purifying/fg ω2),
    # 2b (bg neutral/fg ω2). Omegas for NEB are per-class "representative" ω.
    # Yang 2005 BEB for branch-site treats 2a + 2b as positive-selection classes.
    omegas = [fit.params["omega0"], 1.0, fit.params["omega2"], fit.params["omega2"]]
    per_class = per_class_site_log_likelihood(
        tree=lysozyme_inputs.tree, codons=lysozyme_inputs.alignment.codons,
        taxon_order=lysozyme_inputs.alignment.taxa, Qs=Qs, pi=pi,
    )
    neb = compute_neb(per_class_site_logL=per_class, weights=weights, omegas=omegas)
    assert len(beb) == len(neb)
    for b, n in zip(beb, neb):
        assert abs(b.p_positive - n.p_positive) < 1e-8
        # p_class_2a + p_class_2b should equal p_positive up to roundoff.
        assert abs(b.p_class_2a + b.p_class_2b - b.p_positive) < 1e-10
    assert all(b.beb_grid_size == 1 for b in beb)


def test_model_a_beb_populates_class_2a_2b(lysozyme_inputs) -> None:
    """Non-singleton grid: p_class_2a and p_class_2b are populated and sane."""
    from selkit.engine.beb.branch_site import run_beb_branch_site

    fit, pi, gc = _fit_model_a(lysozyme_inputs)
    beb = run_beb_branch_site(
        fit=fit, grid_size=10,
        tree=lysozyme_inputs.tree, alignment=lysozyme_inputs.alignment, pi=pi, gc=gc,
    )
    assert all(b.p_class_2a is not None and 0.0 <= b.p_class_2a <= 1.0 for b in beb)
    assert all(b.p_class_2b is not None and 0.0 <= b.p_class_2b <= 1.0 for b in beb)
    assert all(b.beb_grid_size == 10 for b in beb)
    # Invariant: p_positive == p_class_2a + p_class_2b.
    for b in beb:
        assert abs(b.p_positive - (b.p_class_2a + b.p_class_2b)) < 1e-10
