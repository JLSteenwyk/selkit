from __future__ import annotations

import subprocess
import sys
from pathlib import Path


FIXTURE = Path(__file__).parent / "fixtures" / "v02_run.yaml"


def test_rerun_rejects_v02_run_yaml():
    result = subprocess.run(
        [sys.executable, "-m", "selkit", "rerun", str(FIXTURE)],
        capture_output=True, text=True,
    )
    assert result.returncode == 1, (
        f"expected exit 1, got {result.returncode}\nstdout:{result.stdout}\nstderr:{result.stderr}"
    )
    assert "v0.2" in result.stderr or "0.2.0" in result.stderr, result.stderr
    assert (
        "codeml.site-models" in result.stderr or "site-models" in result.stderr
    ), result.stderr
    # migration hint mentions the new branch-site subcommand
    assert "branch-site" in result.stderr, result.stderr
