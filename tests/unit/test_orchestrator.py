from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _make_validated_inputs_and_config(tmp_path: Path):
    """Construct the smallest valid ValidatedInputs + RunConfig for orchestrator tests.

    Mirrors the fixture shape used by tests/unit/test_site_models_service.py:
    a 4-taxon alignment observing every nucleotide at every codon position so
    raw-count F3X4 yields pi > 0 everywhere, plus a flat 4-tip Newick.
    """
    from selkit.io.alignment import CodonAlignment
    from selkit.io.config import RunConfig, StrictFlags
    from selkit.io.tree import parse_newick
    from selkit.services.validate import ValidatedInputs
    from selkit.engine.genetic_code import GeneticCode

    gc = GeneticCode.standard()
    tree = parse_newick("(a:0.1,b:0.1,c:0.1,d:0.1);")
    codon_strings = [
        "ACG", "TCA", "GGT", "CAC", "ATG", "TGG", "CCT", "GAT",
        "AAG", "TTC", "GCA", "CGT", "ATC", "TAC", "GTA", "CAT",
        "ACC", "TCG", "GGA", "CAA",
    ]
    row = [gc.codon_to_index(c) for c in codon_strings]
    codons = np.array([row, row, row, row], dtype=np.int16)
    aln = CodonAlignment(
        taxa=("a", "b", "c", "d"),
        codons=codons,
        genetic_code="standard",
        stripped_sites=(),
    )
    inputs = ValidatedInputs(alignment=aln, tree=tree)

    config = RunConfig(
        alignment=Path("/x.fa"),
        alignment_dir=None,
        tree=Path("/x.nwk"),
        foreground=None,
        subcommand="codeml.site-models",
        models=("M0",),
        tests=(),
        genetic_code="standard",
        output_dir=Path("/out"),
        threads=1,
        seed=1,
        n_starts=2,
        convergence_tol=0.5,
        strict=StrictFlags(True, False, False, False),
        selkit_version="0.0.1",
        git_sha=None,
    )
    return inputs, config


def test_run_family_dispatches_by_registry(tmp_path):
    """run_family fits only models in the provided registry and tags the RunResult."""
    from selkit.services.codeml._orchestrator import run_family
    from selkit.services.codeml import site_models
    inputs, config = _make_validated_inputs_and_config(tmp_path)
    # Override models to only fit M0 — fast.
    config = type(config)(**{**config.__dict__, "models": ("M0",)})
    result = run_family(
        family="site",
        registry=site_models.MODEL_REGISTRY,
        default_lrts=(),
        default_beb_models=(),
        inputs=inputs, config=config,
        parallel=False, progress=None,
    )
    assert result.family == "site"
    assert set(result.fits.keys()) == {"M0"}
    # The orchestrator returns SiteModelFit (tagged) — or still ModelFit if tagging
    # wasn't wired yet. Accept either for now; Task 6 makes the tagging mandatory.


def test_run_family_rejects_unknown_model(tmp_path):
    """Unknown model name raises ValueError mentioning the family."""
    from selkit.services.codeml._orchestrator import run_family
    from selkit.services.codeml import site_models
    inputs, config = _make_validated_inputs_and_config(tmp_path)
    config = type(config)(**{**config.__dict__, "models": ("NotAModel",)})
    with pytest.raises(ValueError, match="unknown model"):
        run_family(
            family="site",
            registry=site_models.MODEL_REGISTRY,
            default_lrts=(),
            default_beb_models=(),
            inputs=inputs, config=config,
            parallel=False, progress=None,
        )


def test_run_family_invokes_preconditions(tmp_path):
    """The preconditions callback fires before any fit work."""
    from selkit.services.codeml._orchestrator import run_family
    from selkit.services.codeml import site_models
    inputs, config = _make_validated_inputs_and_config(tmp_path)
    config = type(config)(**{**config.__dict__, "models": ("M0",)})
    calls = []
    def my_preconds(inputs, config):
        calls.append((inputs, config))
    run_family(
        family="site",
        registry=site_models.MODEL_REGISTRY,
        default_lrts=(),
        default_beb_models=(),
        inputs=inputs, config=config,
        parallel=False, progress=None,
        preconditions=my_preconds,
    )
    assert len(calls) == 1
