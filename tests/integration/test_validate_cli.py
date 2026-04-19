from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def _run_selkit(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "selkit", *args],
        capture_output=True, text=True,
    )


def test_validate_reports_pass(tmp_path: Path) -> None:
    aln = _write(tmp_path, "a.fa", ">a\nATGAAA\n>b\nATGAAG\n>c\nATGAAA\n")
    tree = _write(tmp_path, "t.nwk", "(a:0.1,b:0.1,c:0.1);")
    r = _run_selkit("validate", "--alignment", str(aln), "--tree", str(tree))
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout or "pass" in r.stdout.lower()


def test_validate_reports_mismatch(tmp_path: Path) -> None:
    aln = _write(tmp_path, "a.fa", ">a\nATGAAA\n>b\nATGAAG\n")
    tree = _write(tmp_path, "t.nwk", "(a:0.1,b:0.1,c:0.1);")
    r = _run_selkit("validate", "--alignment", str(aln), "--tree", str(tree))
    assert r.returncode == 1
    assert "taxon" in r.stderr.lower()
