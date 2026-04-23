from __future__ import annotations

import subprocess
from pathlib import Path


def test_rerun_codeml_branch(tmp_path):
    """Running codeml branch then rerun on its run.yaml should succeed."""
    corpus = Path(__file__).parent.parent / "validation" / "corpus" / "lysozyme_branchsite"
    out1 = tmp_path / "out1"
    r1 = subprocess.run(
        [
            "selkit", "codeml", "branch",
            "--alignment", str(corpus / "alignment.fa"),
            "--tree", str(corpus / "tree.nwk"),
            "--output", str(out1),
            "--models", "TwoRatios",
            "--n-starts", "1", "--seed", "0",
        ],
        capture_output=True, text=True,
    )
    assert r1.returncode == 0, r1.stderr
    out2 = tmp_path / "out2"
    r2 = subprocess.run(
        [
            "selkit", "rerun", str(out1 / "run.yaml"),
            "--output", str(out2),
        ],
        capture_output=True, text=True,
    )
    assert r2.returncode == 0, r2.stderr
    assert (out2 / "fits_branch.tsv").exists()
