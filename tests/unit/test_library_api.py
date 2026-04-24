from __future__ import annotations

from pathlib import Path

import selkit
from selkit import (
    BEBSite,
    CodonAlignment,
    LRTResult,
    LabeledTree,
    ModelFit,
    RunConfig,
    RunResult,
    codeml_site_models,
)


def test_public_api_exports_expected_names() -> None:
    assert hasattr(selkit, "__version__")
    assert callable(codeml_site_models)
    for cls in (CodonAlignment, LabeledTree, RunConfig, RunResult, ModelFit, LRTResult, BEBSite):
        assert isinstance(cls, type)


def test_codeml_site_models_returns_run_result(tmp_path: Path) -> None:
    aln = tmp_path / "a.fa"
    aln.write_text(
        ">a\nATGAAAGCACGT\n>b\nATGAAGGCCCGT\n>c\nATGAAAGCACGT\n>d\nATGAAAGCCCGC\n"
    )
    tree = tmp_path / "t.nwk"
    tree.write_text("((a:0.1,b:0.1):0.05,(c:0.1,d:0.1):0.05);")
    out = tmp_path / "out"
    result = codeml_site_models(
        alignment=aln, tree=tree, output_dir=out,
        models=("M0", "M1a"), n_starts=2, seed=1,
    )
    assert isinstance(result, RunResult)
    assert set(result.fits) == {"M0", "M1a"}


def test_codeml_branch_site_models_library_function_exists():
    from selkit import codeml_branch_site_models
    import inspect
    sig = inspect.signature(codeml_branch_site_models)
    assert "alignment" in sig.parameters
    assert "tree" in sig.parameters
    assert "foreground" in sig.parameters
    assert "output_dir" in sig.parameters


def test_codeml_branch_models_library_happy_path(tmp_path):
    from selkit import codeml_branch_models
    from selkit.io.tree import ForegroundSpec
    aln = tmp_path / "a.fa"
    aln.write_text(
        ">A\nATGAAAGGG\n>B\nATGAAAGGG\n>C\nATGAAAGGG\n>D\nATGAAAGGG\n"
    )
    nwk = tmp_path / "t.nwk"
    nwk.write_text("((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);\n")
    result = codeml_branch_models(
        alignment=aln, tree=nwk, output_dir=tmp_path / "out",
        models=("TwoRatios",),
        foreground=ForegroundSpec(mrca=("A", "B")),
        n_starts=1, seed=0,
    )
    assert result.family == "branch"
    assert "TwoRatios" in result.fits


def test_codeml_site_models_accepts_beb_kwargs() -> None:
    import inspect

    from selkit import codeml_site_models
    sig = inspect.signature(codeml_site_models)
    assert "beb" in sig.parameters
    assert sig.parameters["beb"].default is True
    assert "beb_grid" in sig.parameters
    assert sig.parameters["beb_grid"].default == 10


def test_codeml_branch_site_models_accepts_beb_kwargs() -> None:
    import inspect

    from selkit import codeml_branch_site_models
    sig = inspect.signature(codeml_branch_site_models)
    assert sig.parameters["beb"].default is True
    assert sig.parameters["beb_grid"].default == 10
