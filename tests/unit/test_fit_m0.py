from __future__ import annotations

import numpy as np

from selkit.engine.codon_model import M0
from selkit.engine.fit import fit_model
from selkit.engine.genetic_code import GeneticCode
from selkit.engine.rate_matrix import estimate_f3x4
from selkit.io.tree import parse_newick


def test_fit_m0_recovers_omega_from_simulated_like_data() -> None:
    gc = GeneticCode.standard()
    tree = parse_newick("(a:0.2,b:0.2,(c:0.2,d:0.2):0.1);")
    # Use a synthetic alignment that observes every nucleotide at every codon
    # position (raw-count F3X4, PAML-compatible, needs this).
    codon_strings = [
        "ACG", "TCA", "GGT", "CAC", "ATG", "TGG", "CCT", "GAT",
    ] * 7  # 56 codons; trim to 50
    codon_strings = codon_strings[:50]
    row = [gc.codon_to_index(c) for c in codon_strings]
    codons = np.array([row, row, row, row], dtype=np.int16)
    pi = estimate_f3x4(codons, gc)
    model = M0(gc=gc, pi=pi)
    fit = fit_model(
        model=model,
        alignment_codons=codons,
        taxon_order=("a", "b", "c", "d"),
        tree=tree,
        n_starts=2,
        seed=7,
    )
    assert np.isfinite(fit.lnL)
    assert "omega" in fit.params
    assert "kappa" in fit.params
    assert fit.params["omega"] > 0
    assert fit.params["kappa"] > 0
    assert len(fit.branch_lengths) >= 4
