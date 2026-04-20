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
