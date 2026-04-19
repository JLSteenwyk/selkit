from __future__ import annotations

import numpy as np

from selkit.engine.genetic_code import GeneticCode
from selkit.engine.rate_matrix import estimate_f3x4
from selkit.io.config import RunConfig, StrictFlags
from selkit.io.tree import parse_newick
from selkit.io.alignment import CodonAlignment
from selkit.services.validate import ValidatedInputs
from selkit.services.codeml.site_models import run_site_models


def _make_inputs() -> ValidatedInputs:
    gc = GeneticCode.standard()
    tree = parse_newick("(a:0.1,b:0.1,c:0.1,d:0.1);")
    idx = gc.codon_to_index("ATG")
    codons = np.full((4, 20), idx, dtype=np.int16)
    aln = CodonAlignment(taxa=("a","b","c","d"), codons=codons, genetic_code="standard", stripped_sites=())
    return ValidatedInputs(alignment=aln, tree=tree)


def _cfg() -> RunConfig:
    from pathlib import Path
    return RunConfig(
        alignment=Path("/x.fa"), alignment_dir=None, tree=Path("/x.nwk"),
        foreground=None, subcommand="codeml.site-models",
        models=("M0", "M1a", "M2a"), tests=("M1a-vs-M2a",),
        genetic_code="standard",
        output_dir=Path("/out"), threads=1, seed=1, n_starts=2,
        convergence_tol=0.5,
        strict=StrictFlags(True, False, False, False),
        selkit_version="0.0.1", git_sha=None,
    )


def test_run_site_models_sequential_returns_fits_and_lrts() -> None:
    inputs = _make_inputs()
    result = run_site_models(inputs=inputs, config=_cfg(), parallel=False, progress=None)
    assert set(result.fits) == {"M0", "M1a", "M2a"}
    lrt_names = {(l.null, l.alt) for l in result.lrts}
    assert ("M1a", "M2a") in lrt_names
    assert "M2a" in result.beb
