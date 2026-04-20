from __future__ import annotations

from pathlib import Path

import pytest

from selkit.errors import SelkitInputError
from selkit.io.tree import ForegroundSpec
from selkit.services.validate import validate_inputs


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def test_validate_success(tmp_path: Path) -> None:
    aln = _write(tmp_path, "a.fa", ">a\nATGAAA\n>b\nATGAAG\n>c\nATGAAA\n")
    tree = _write(tmp_path, "t.nwk", "(a:0.1,b:0.1,c:0.1);")
    result = validate_inputs(
        alignment_path=aln, tree_path=tree, foreground_spec=ForegroundSpec(),
        genetic_code_name="standard",
    )
    assert result.alignment.taxa == ("a", "b", "c")
    assert result.tree.tip_names == ("a", "b", "c")


def test_validate_taxon_mismatch(tmp_path: Path) -> None:
    aln = _write(tmp_path, "a.fa", ">a\nATGAAA\n>b\nATGAAG\n")
    tree = _write(tmp_path, "t.nwk", "(a:0.1,b:0.1,c:0.1);")
    with pytest.raises(SelkitInputError, match=r"taxon"):
        validate_inputs(
            alignment_path=aln, tree_path=tree, foreground_spec=ForegroundSpec(),
            genetic_code_name="standard",
        )
