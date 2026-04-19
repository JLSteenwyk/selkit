from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "selkit", *args],
        capture_output=True, text=True,
    )


def test_site_models_end_to_end(tmp_path: Path) -> None:
    aln = _write(tmp_path, "a.fa",
        ">a\nATGAAAGCACGTTTAGGCAAACCACGTATG\n"
        ">b\nATGAAGGCCCGTCTAGGGAAGCCTCGTATG\n"
        ">c\nATGAAAGCACGTTTGGGGAAGCCACGTATG\n"
        ">d\nATGAAAGCCCGCTTAGGCAAACCGCGTATG\n"
    )
    tree = _write(tmp_path, "t.nwk", "((a:0.1,b:0.1):0.05,(c:0.1,d:0.1):0.05);")
    out = tmp_path / "out"

    r = _run(
        "codeml", "site-models",
        "--alignment", str(aln),
        "--tree", str(tree),
        "--output", str(out),
        "--threads", "1",
        "--n-starts", "2",
        "--models", "M0,M1a,M2a",
        "--allow-unconverged",
    )
    assert r.returncode == 0, r.stderr

    results = json.loads((out / "results.json").read_text())
    assert set(results["fits"]) == {"M0", "M1a", "M2a"}
    assert (out / "fits.tsv").exists()
    assert (out / "lrts.tsv").exists()
    assert (out / "run.yaml").exists()
