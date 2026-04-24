from __future__ import annotations

import json
import subprocess
from pathlib import Path


CORPUS_HIV_4S = Path(__file__).parent.parent / "validation" / "corpus" / "hiv_4s"


def _run_site(args, tmp_out: Path) -> subprocess.CompletedProcess:
    cmd = [
        "selkit", "codeml", "site",
        "--alignment", str(CORPUS_HIV_4S / "alignment.fa"),
        "--tree", str(CORPUS_HIV_4S / "tree.nwk"),
        "--output", str(tmp_out),
        "--models", "M0,M2a",
        "--n-starts", "1",
        *args,
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_no_beb_flag_skips_beb_emission(tmp_path) -> None:
    out = tmp_path / "no_beb"
    r = _run_site(["--no-beb"], out)
    assert r.returncode in (0, 2), r.stderr
    assert not (out / "beb_M2a.tsv").exists(), "BEB should be skipped with --no-beb"
    data = json.loads((out / "results.json").read_text())
    assert data["beb"] == {} or data["beb"] == {"M2a": []}


def test_beb_grid_flag_threads_through(tmp_path) -> None:
    out = tmp_path / "beb_grid5"
    r = _run_site(["--beb-grid", "5"], out)
    assert r.returncode in (0, 2), r.stderr
    data = json.loads((out / "results.json").read_text())
    assert "M2a" in data["beb"]
    # Every BEB record records the grid size.
    for rec in data["beb"]["M2a"]:
        assert rec["beb_grid_size"] == 5


def test_default_runs_beb_with_grid_10(tmp_path) -> None:
    out = tmp_path / "default"
    r = _run_site([], out)
    assert r.returncode in (0, 2), r.stderr
    data = json.loads((out / "results.json").read_text())
    assert "M2a" in data["beb"] and len(data["beb"]["M2a"]) > 0
    for rec in data["beb"]["M2a"]:
        assert rec["beb_grid_size"] == 10
