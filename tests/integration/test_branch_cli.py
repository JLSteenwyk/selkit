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


def test_branch_cli_defaults_to_trio(tmp_path):
    """No --models flag -> defaults to M0,TwoRatios,TwoRatiosFixed."""
    import json
    corpus = LYS
    out = tmp_path / "out"
    r = subprocess.run(
        [
            "selkit", "codeml", "branch",
            "--alignment", str(corpus / "alignment.fa"),
            "--tree", str(corpus / "tree.nwk"),
            "--output", str(out),
            "--n-starts", "1", "--seed", "0",
        ],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    results = json.loads((out / "results.json").read_text())
    assert set(results["fits"].keys()) == {"M0", "TwoRatios", "TwoRatiosFixed"}


def test_branch_cli_foreground_flag_wires_through(tmp_path):
    """--foreground A,B,C designates a foreground via MRCA on an unlabelled tree."""
    corpus = LYS
    # Strip inline #1 from the tree so --foreground provides labels (not double-spec).
    raw = (corpus / "tree.nwk").read_text()
    nwk = tmp_path / "tree_unlabelled.nwk"
    nwk.write_text(raw.replace("#1", ""))
    out = tmp_path / "out"
    r = subprocess.run(
        [
            "selkit", "codeml", "branch",
            "--alignment", str(corpus / "alignment.fa"),
            "--tree", str(nwk),
            "--output", str(out),
            "--models", "TwoRatios",
            "--foreground",
            "5.colobus_Cgu&Can,6.langur_Sen&Sve,7.langur_Tob&Tfr,"
            "8.Douc_langur_Pne,9.probiscis_Nla",
            "--n-starts", "1", "--seed", "0",
        ],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr


def test_branch_cli_unknown_model_errors(tmp_path):
    corpus = LYS
    out = tmp_path / "out"
    r = subprocess.run(
        [
            "selkit", "codeml", "branch",
            "--alignment", str(corpus / "alignment.fa"),
            "--tree", str(corpus / "tree.nwk"),
            "--output", str(out),
            "--models", "NotAModel",
        ],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "NotAModel" in (r.stderr + r.stdout)


def test_branch_cli_free_ratios_reports_df_warning(tmp_path):
    """FreeRatios LRT carries a 'caution' warning. Uses the small hiv_4s
    corpus so the fit is fast (4 taxa -> 5 branches -> 4 free omegas)."""
    import json
    corpus = Path(__file__).parent.parent / "validation" / "corpus" / "hiv_4s"
    out = tmp_path / "out"
    r = subprocess.run(
        [
            "selkit", "codeml", "branch",
            "--alignment", str(corpus / "alignment.fa"),
            "--tree", str(corpus / "tree.nwk"),
            "--output", str(out),
            "--models", "M0,FreeRatios",
            "--n-starts", "1", "--seed", "0",
        ],
        capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, r.stderr
    lrts = json.loads((out / "results.json").read_text())["lrts"]
    free = [l for l in lrts if l["alt"] == "FreeRatios"]
    assert free, "FreeRatios LRT must be present"
    assert "caution" in (free[0]["warning"] or "").lower()
