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


def test_rerun_reproduces_results(tmp_path: Path) -> None:
    aln = _write(tmp_path, "a.fa",
        ">a\nATGAAAGCACGTTTAGGC\n>b\nATGAAGGCCCGTCTAGGG\n>c\nATGAAAGCACGTTTGGGG\n>d\nATGAAAGCCCGCTTAGGC\n"
    )
    tree = _write(tmp_path, "t.nwk", "((a:0.1,b:0.1):0.05,(c:0.1,d:0.1):0.05);")
    out1 = tmp_path / "out1"
    r1 = _run(
        "codeml", "site-models",
        "--alignment", str(aln), "--tree", str(tree),
        "--output", str(out1), "--threads", "1", "--n-starts", "2",
        "--models", "M0,M1a", "--seed", "42", "--allow-unconverged",
    )
    assert r1.returncode == 0, r1.stderr
    out2 = tmp_path / "out2"
    r2 = _run("rerun", str(out1 / "run.yaml"), "--output", str(out2))
    assert r2.returncode == 0, r2.stderr
    j1 = json.loads((out1 / "results.json").read_text())
    j2 = json.loads((out2 / "results.json").read_text())
    for m in j1["fits"]:
        assert abs(j1["fits"][m]["lnL"] - j2["fits"][m]["lnL"]) < 1e-6
