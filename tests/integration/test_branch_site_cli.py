from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


CORPUS_DIR = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "corpus"
    / "lysozyme_branchsite"
)


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "selkit", *args],
        capture_output=True, text=True,
    )


def test_branch_site_end_to_end(tmp_path: Path) -> None:
    """Happy path: branch-site ModelA + ModelA_null on the lysozyme corpus
    (foreground encoded in-Newick via #1 on the colobine clade)."""
    aln = CORPUS_DIR / "alignment.fa"
    tree = CORPUS_DIR / "tree.nwk"
    out = tmp_path / "out"

    r = _run(
        "codeml", "branch-site",
        "--alignment", str(aln),
        "--tree", str(tree),
        "--output", str(out),
        "--threads", "1",
        "--n-starts", "1",
        "--models", "ModelA,ModelA_null",
        "--allow-unconverged",
    )
    assert r.returncode == 0, r.stderr

    results = json.loads((out / "results.json").read_text())
    assert results["family"] == "branch-site"
    assert set(results["fits"]) == {"ModelA", "ModelA_null"}
    assert (out / "fits.tsv").exists()
    assert (out / "lrts.tsv").exists()
    assert (out / "run.yaml").exists()


def test_branch_site_requires_foreground(tmp_path: Path) -> None:
    """Branch-site rejects calls with no foreground annotation via the
    _require_foreground precondition (SelkitConfigError -> exit 1)."""
    aln = _write(tmp_path, "a.fa",
        ">a\nATGAAAGCACGTTTAGGCAAACCACGTATG\n"
        ">b\nATGAAGGCCCGTCTAGGGAAGCCTCGTATG\n"
        ">c\nATGAAAGCACGTTTGGGGAAGCCACGTATG\n"
        ">d\nATGAAAGCCCGCTTAGGCAAACCGCGTATG\n"
    )
    # No #1 in this tree and no --foreground/--foreground-tips/--labels-file.
    tree = _write(tmp_path, "t.nwk", "((a:0.1,b:0.1):0.05,(c:0.1,d:0.1):0.05);")
    out = tmp_path / "out"

    r = _run(
        "codeml", "branch-site",
        "--alignment", str(aln),
        "--tree", str(tree),
        "--output", str(out),
        "--threads", "1",
        "--n-starts", "1",
        "--models", "ModelA,ModelA_null",
        "--allow-unconverged",
    )
    assert r.returncode == 1, (r.returncode, r.stderr)
    assert "foreground" in r.stderr.lower()
