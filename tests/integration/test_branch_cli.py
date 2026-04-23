from __future__ import annotations

import subprocess
from pathlib import Path


LYS = Path(__file__).parent.parent / "validation" / "corpus" / "lysozyme_branchsite"


def test_branch_cli_runs_two_ratios(tmp_path):
    out = tmp_path / "out"
    result = subprocess.run(
        [
            "selkit", "codeml", "branch",
            "--alignment", str(LYS / "alignment.fa"),
            "--tree", str(LYS / "tree.nwk"),
            "--output", str(out),
            "--models", "TwoRatios",
            "--n-starts", "1", "--seed", "0",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "fits_branch.tsv").exists()
    assert (out / "fits_branch_per_branch.tsv").exists()


def test_branch_cli_rejects_two_ratios_on_K_gt_1_tree(tmp_path):
    """TwoRatios errors cleanly when the tree has K=2."""
    nwk = tmp_path / "k2.nwk"
    nwk.write_text("((A:0.1,B:0.1)#1,(C:0.1,D:0.1)#2);\n")
    aln = tmp_path / "aln.fa"
    aln.write_text(">A\nATG\n>B\nATG\n>C\nATG\n>D\nATG\n")
    out = tmp_path / "out"
    result = subprocess.run(
        [
            "selkit", "codeml", "branch",
            "--alignment", str(aln),
            "--tree", str(nwk),
            "--output", str(out),
            "--models", "TwoRatios",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "K=1" in result.stderr or "K=1" in result.stdout
