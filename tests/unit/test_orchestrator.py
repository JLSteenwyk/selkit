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
        subcommand="codeml.site",
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


def test_orchestrator_runs_branch_family_on_tiny_tree(tmp_path):
    from selkit.io.config import RunConfig, StrictFlags
    from selkit.io.tree import ForegroundSpec
    from selkit.services.codeml.branch_models import run_branch_models
    from selkit.services.validate import validate_inputs
    from selkit.version import __version__
    import numpy as np

    aln = tmp_path / "aln.fa"
    aln.write_text(">A\nATGAAAGGG\n>B\nATGAAAGGG\n>C\nATGAAAGGG\n>D\nATGAAAGGG\n")
    nwk = tmp_path / "tree.nwk"
    nwk.write_text("((A:0.1,B:0.1)#1,(C:0.1,D:0.1):0.1);\n")

    cfg = RunConfig(
        alignment=aln, alignment_dir=None, tree=nwk,
        foreground=None, subcommand="codeml.branch",
        models=("TwoRatios",), tests=(), genetic_code="standard",
        output_dir=tmp_path, threads=1, seed=0, n_starts=1,
        convergence_tol=0.5,
        strict=StrictFlags(True, False, False, False),
        selkit_version=__version__, git_sha=None,
    )
    inputs = validate_inputs(
        alignment_path=aln, tree_path=nwk,
        foreground_spec=ForegroundSpec(),
        genetic_code_name="standard",
    )
    result = run_branch_models(
        inputs=inputs, config=cfg, parallel=False, progress=None,
    )
    assert result.family == "branch"
    from selkit.io.results import BranchModelFit
    assert isinstance(result.fits["TwoRatios"], BranchModelFit)
    # 4 tips + 2 internals + root = 7 nodes; 6 non-root branches.
    assert len(result.fits["TwoRatios"].per_branch_omega) == 6
    per_labels = {r["label"] for r in result.fits["TwoRatios"].per_branch_omega}
    assert per_labels == {"foreground", "background"}


def test_orchestrator_populates_per_branch_SE_from_hess_inv_diag(tmp_path):
    """A successful TwoRatios fit should emit real (non-None) SE values
    on every foreground and background branch record, sourced from
    EngineFit.hess_inv_diag via _extract_per_branch_omega.
    """
    from selkit.io.config import RunConfig, StrictFlags
    from selkit.io.tree import ForegroundSpec
    from selkit.services.codeml.branch_models import run_branch_models
    from selkit.services.validate import validate_inputs
    from selkit.version import __version__
    import numpy as np

    aln = tmp_path / "aln.fa"
    # Non-degenerate alignment so the Hessian is informative. Use 30 codons
    # with substitutions on both fg and bg branches so neither omega is
    # estimated at ~0 (where the inv-Hessian collapses).
    aln.write_text(
        ">A\nATGAAAGGGCCCTTTACGCATAGCGCCATTAACTACTGCAGCGGTGGCATA\n"
        ">B\nATGAAGGGACCCTTCACCCACAGTGCCATCAACTATTGTAGTGGAGGAATC\n"
        ">C\nATGAAAGGGCCATTTACGCATAGCGCCATTAACTACTGCAGCGGTGGCATA\n"
        ">D\nATGAAGGGTCCATTCACCCATAGAGCGATCAACTATTGTAGTGGAGGCATC\n"
    )
    nwk = tmp_path / "tree.nwk"
    nwk.write_text("((A:0.1,B:0.1)#1,(C:0.1,D:0.1):0.1);\n")
    cfg = RunConfig(
        alignment=aln, alignment_dir=None, tree=nwk,
        foreground=None, subcommand="codeml.branch",
        models=("TwoRatios",), tests=(), genetic_code="standard",
        output_dir=tmp_path, threads=1, seed=0, n_starts=2,
        convergence_tol=0.5,
        strict=StrictFlags(True, False, False, False),
        selkit_version=__version__, git_sha=None,
    )
    inputs = validate_inputs(
        alignment_path=aln, tree_path=nwk,
        foreground_spec=ForegroundSpec(),
        genetic_code_name="standard",
    )
    result = run_branch_models(
        inputs=inputs, config=cfg, parallel=False, progress=None,
    )
    fit = result.fits["TwoRatios"]
    # Every per-branch record should carry a finite SE.
    ses = [r["SE"] for r in fit.per_branch_omega]
    assert all(se is not None and np.isfinite(se) and se > 0 for se in ses), (
        f"expected non-None SE on every branch; got {ses}"
    )


def test_branch_family_M0_populates_per_branch_omega_uniformly(tmp_path):
    """I2 regression: M0 in the branch family should emit one per_branch_omega
    row per non-root branch, all carrying the M0-shared omega. Previously the
    orchestrator special-cased M0 to return per_branch_omega=[].
    """
    from selkit.io.config import RunConfig, StrictFlags
    from selkit.io.tree import ForegroundSpec
    from selkit.io.results import BranchModelFit
    from selkit.services.codeml.branch_models import run_branch_models
    from selkit.services.validate import validate_inputs
    from selkit.version import __version__

    aln = tmp_path / "aln.fa"
    aln.write_text(">A\nATGAAAGGG\n>B\nATGAAAGGG\n>C\nATGAAAGGG\n>D\nATGAAAGGG\n")
    nwk = tmp_path / "tree.nwk"
    nwk.write_text("((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);\n")

    cfg = RunConfig(
        alignment=aln, alignment_dir=None, tree=nwk,
        foreground=None, subcommand="codeml.branch",
        models=("M0",), tests=(), genetic_code="standard",
        output_dir=tmp_path, threads=1, seed=0, n_starts=1,
        convergence_tol=0.5,
        strict=StrictFlags(True, False, False, False),
        selkit_version=__version__, git_sha=None,
    )
    inputs = validate_inputs(
        alignment_path=aln, tree_path=nwk,
        foreground_spec=ForegroundSpec(),
        genetic_code_name="standard",
    )
    result = run_branch_models(
        inputs=inputs, config=cfg, parallel=False, progress=None,
    )
    fit = result.fits["M0"]
    assert isinstance(fit, BranchModelFit)
    # 4 tips + 2 internals + root => 6 non-root branches.
    assert len(fit.per_branch_omega) == 6
    # Every row reports the same M0-shared omega.
    omegas = {r["omega"] for r in fit.per_branch_omega}
    assert len(omegas) == 1
    assert omegas == {float(fit.params["omega"])}
    # Label is "M0" so consumers can distinguish from per-class branch labels.
    assert all(r["label"] == "M0" for r in fit.per_branch_omega)


def test_run_branch_preconditions_K_mismatch(tmp_path):
    from selkit.errors import SelkitConfigError
    from selkit.io.config import RunConfig, StrictFlags
    from selkit.io.tree import ForegroundSpec
    from selkit.services.codeml.branch_models import run_branch_models
    from selkit.services.validate import validate_inputs
    from selkit.version import __version__
    import pytest

    aln = tmp_path / "aln.fa"
    aln.write_text(">A\nATG\n>B\nATG\n>C\nATG\n>D\nATG\n")
    nwk = tmp_path / "t.nwk"
    # Tree with #1 and #2 -> K=2
    nwk.write_text("((A:0.1,B:0.1)#1,(C:0.1,D:0.1)#2);\n")
    cfg = RunConfig(
        alignment=aln, alignment_dir=None, tree=nwk,
        foreground=None, subcommand="codeml.branch",
        models=("TwoRatios",), tests=(), genetic_code="standard",
        output_dir=tmp_path, threads=1, seed=0, n_starts=1,
        convergence_tol=0.5,
        strict=StrictFlags(True, False, False, False),
        selkit_version=__version__, git_sha=None,
    )
    inputs = validate_inputs(
        alignment_path=aln, tree_path=nwk,
        foreground_spec=ForegroundSpec(), genetic_code_name="standard",
    )
    with pytest.raises(SelkitConfigError, match="K=1"):
        run_branch_models(
            inputs=inputs, config=cfg, parallel=False, progress=None,
        )
