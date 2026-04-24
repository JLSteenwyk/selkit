from __future__ import annotations

import numpy as np


def test_fit_m0_populates_hess_inv_diag_with_finite_positive_se():
    """M0 on a tiny alignment should yield a finite positive SE for omega."""
    from selkit.engine.codon_model import M0
    from selkit.engine.fit import fit_model
    from selkit.engine.genetic_code import GeneticCode
    from selkit.engine.rate_matrix import estimate_f3x4
    from selkit.io.tree import parse_newick

    gc = GeneticCode.by_name("standard")
    tree = parse_newick("((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);")
    # Several distinct codons across 4 taxa -> non-degenerate pi and non-trivial fit.
    codons = np.tile(np.arange(4).reshape(4, 1), (1, 12))
    pi = estimate_f3x4(codons, gc, pseudocount=1.0)
    m = M0(gc=gc, pi=pi)
    fit = fit_model(
        model=m, alignment_codons=codons, taxon_order=("A", "B", "C", "D"),
        tree=tree, n_starts=1, seed=0,
    )
    assert fit.hess_inv_diag is not None
    assert "omega" in fit.hess_inv_diag
    se = fit.hess_inv_diag["omega"]
    assert np.isfinite(se)
    assert se > 0.0
    # kappa should also have an SE.
    assert "kappa" in fit.hess_inv_diag
    assert fit.hess_inv_diag["kappa"] > 0.0


def test_fit_two_ratios_populates_hess_inv_diag_for_both_omegas():
    """TwoRatios fit surfaces SE for omega_bg and omega_fg."""
    from selkit.engine.codon_model import TwoRatios
    from selkit.engine.fit import fit_model
    from selkit.engine.genetic_code import GeneticCode
    from selkit.engine.rate_matrix import estimate_f3x4
    from selkit.io.tree import apply_foreground_spec, parse_newick, ForegroundSpec

    gc = GeneticCode.by_name("standard")
    tree = parse_newick("((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);")
    tree = apply_foreground_spec(tree, ForegroundSpec(mrca=("A", "B")))
    codons = np.tile(np.arange(4).reshape(4, 1), (1, 12))
    pi = estimate_f3x4(codons, gc, pseudocount=1.0)
    m = TwoRatios(gc=gc, pi=pi)
    fit = fit_model(
        model=m, alignment_codons=codons, taxon_order=("A", "B", "C", "D"),
        tree=tree, n_starts=1, seed=0,
    )
    assert fit.hess_inv_diag is not None
    for key in ("omega_bg", "omega_fg", "kappa"):
        assert key in fit.hess_inv_diag, f"missing SE for {key}"
        assert np.isfinite(fit.hess_inv_diag[key]) and fit.hess_inv_diag[key] > 0.0
