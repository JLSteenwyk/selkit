from __future__ import annotations

import numpy as np
import pytest


def test_fit_dispatches_branch_family_to_branch_family_caller(monkeypatch):
    import selkit.engine.fit as fit_mod
    from selkit.engine.codon_model import TwoRatios
    from selkit.engine.genetic_code import GeneticCode
    from selkit.engine.rate_matrix import estimate_f3x4
    from selkit.io.tree import apply_foreground_spec, parse_newick, ForegroundSpec
    import numpy as np

    calls = {"branch_family": 0, "mixture": 0}
    orig_bf = fit_mod.tree_log_likelihood_branch_family
    orig_mx = fit_mod.tree_log_likelihood_mixture

    def spy_bf(*a, **kw):
        calls["branch_family"] += 1
        return orig_bf(*a, **kw)

    def spy_mx(*a, **kw):
        calls["mixture"] += 1
        return orig_mx(*a, **kw)

    monkeypatch.setattr(fit_mod, "tree_log_likelihood_branch_family", spy_bf)
    monkeypatch.setattr(fit_mod, "tree_log_likelihood_mixture", spy_mx)

    gc = GeneticCode.by_name("standard")
    tree = parse_newick("((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);")
    tree = apply_foreground_spec(tree, ForegroundSpec(mrca=("A", "B")))
    codons = np.tile(np.arange(4).reshape(4, 1), (1, 6))
    pi = estimate_f3x4(codons, gc, pseudocount=1.0)
    m = TwoRatios(gc=gc, pi=pi)
    fit_mod.fit_model(
        model=m, alignment_codons=codons, taxon_order=("A", "B", "C", "D"),
        tree=tree, n_starts=1, seed=0,
    )
    assert calls["branch_family"] >= 1
    assert calls["mixture"] == 0
