from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from selkit.engine.genetic_code import GeneticCode
from selkit.errors import SelkitInputError
from selkit.io.alignment import CodonAlignment, read_fasta


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def test_read_simple_fasta(tmp_path: Path) -> None:
    path = _write(
        tmp_path, "ok.fa",
        ">a\nATGAAA\n>b\nATGAAG\n",
    )
    aln = read_fasta(path, genetic_code=GeneticCode.standard())
    assert aln.taxa == ("a", "b")
    assert aln.codons.shape == (2, 2)
    gc = GeneticCode.standard()
    assert aln.codons[0, 0] == gc.codon_to_index("ATG")
    assert aln.codons[0, 1] == gc.codon_to_index("AAA")
    assert aln.codons[1, 1] == gc.codon_to_index("AAG")


def test_read_fasta_lower_case_is_upcased(tmp_path: Path) -> None:
    path = _write(tmp_path, "lc.fa", ">a\natgaaa\n>b\natgaag\n")
    aln = read_fasta(path, genetic_code=GeneticCode.standard())
    gc = GeneticCode.standard()
    assert aln.codons[0, 0] == gc.codon_to_index("ATG")


def test_read_fasta_gap_codon_becomes_minus_one(tmp_path: Path) -> None:
    path = _write(tmp_path, "gap.fa", ">a\nATG---\n>b\nATGAAA\n")
    aln = read_fasta(path, genetic_code=GeneticCode.standard())
    assert aln.codons[0, 1] == -1
    assert aln.codons[1, 1] != -1


def test_length_not_multiple_of_three_errors(tmp_path: Path) -> None:
    path = _write(tmp_path, "bad.fa", ">a\nATGAA\n>b\nATGAAG\n")
    with pytest.raises(SelkitInputError, match=r"multiple of 3"):
        read_fasta(path, genetic_code=GeneticCode.standard())


def test_mismatched_sequence_lengths_errors(tmp_path: Path) -> None:
    path = _write(tmp_path, "mm.fa", ">a\nATG\n>b\nATGAAA\n")
    with pytest.raises(SelkitInputError, match=r"same length"):
        read_fasta(path, genetic_code=GeneticCode.standard())


def test_mid_sequence_stop_codon_errors_by_default(tmp_path: Path) -> None:
    path = _write(tmp_path, "mid.fa", ">a\nATGTAAATG\n>b\nATGAAAATG\n")
    with pytest.raises(SelkitInputError, match=r"[Ss]top"):
        read_fasta(path, genetic_code=GeneticCode.standard())


def test_terminal_stop_on_all_taxa_is_stripped(tmp_path: Path) -> None:
    path = _write(tmp_path, "term.fa", ">a\nATGAAATAA\n>b\nATGAAGTAA\n")
    aln = read_fasta(path, genetic_code=GeneticCode.standard())
    assert aln.codons.shape == (2, 2)
    assert 2 in aln.stripped_sites


def test_duplicate_taxon_errors(tmp_path: Path) -> None:
    path = _write(tmp_path, "dup.fa", ">a\nATGAAA\n>a\nATGAAG\n")
    with pytest.raises(SelkitInputError, match=r"duplicate"):
        read_fasta(path, genetic_code=GeneticCode.standard())


def test_empty_alignment_errors(tmp_path: Path) -> None:
    path = _write(tmp_path, "empty.fa", "")
    with pytest.raises(SelkitInputError, match=r"empty"):
        read_fasta(path, genetic_code=GeneticCode.standard())


def test_codon_alignment_codons_dtype_is_int16() -> None:
    arr = np.array([[0, 1, -1]], dtype=np.int16)
    aln = CodonAlignment(
        taxa=("a",), codons=arr, genetic_code="standard", stripped_sites=()
    )
    assert aln.codons.dtype == np.int16


def test_bom_prefixed_fasta_does_not_drop_first_taxon(tmp_path: Path) -> None:
    p = tmp_path / "bom.fa"
    p.write_bytes("\ufeff>a\nATGAAA\n>b\nATGAAG\n".encode("utf-8"))
    aln = read_fasta(p, genetic_code=GeneticCode.standard())
    assert aln.taxa == ("a", "b")
    assert aln.codons.shape == (2, 2)


def test_bare_gt_header_raises_selkit_input_error(tmp_path: Path) -> None:
    path = _write(tmp_path, "bare.fa", ">\nATGAAA\n>b\nATGAAG\n")
    with pytest.raises(SelkitInputError, match=r"(?i)header"):
        read_fasta(path, genetic_code=GeneticCode.standard())


def test_terminal_stop_stripped_even_when_mid_stops_exist(tmp_path: Path) -> None:
    # Universal terminal TAA + a mid-stop in taxon a at codon 2.
    # With strip_stop_codons=False (default), terminal should be stripped,
    # then mid-stop should raise.
    path = _write(
        tmp_path, "both.fa",
        ">a\nATGTAAAAATAA\n>b\nATGAAAAAATAA\n",
    )
    with pytest.raises(SelkitInputError, match=r"(?i)stop") as ei:
        read_fasta(path, genetic_code=GeneticCode.standard())
    # The error should name codon 2 (the mid-stop), not codon 4 (the terminal).
    assert "codon 2" in str(ei.value)


def test_strip_stop_codons_drops_columns_without_leakage(tmp_path: Path) -> None:
    path = _write(
        tmp_path, "midstop.fa",
        ">a\nATGTAAATG\n>b\nATGAAAATG\n",
    )
    aln = read_fasta(
        path, genetic_code=GeneticCode.standard(), strip_stop_codons=True
    )
    # Column 1 (the TAA/AAA column) is dropped from every taxon.
    assert aln.codons.shape == (2, 2)
    assert 1 in aln.stripped_sites
    # The internal -2 stop sentinel must never surface in the returned array.
    assert -2 not in aln.codons.tolist()


def test_strip_stop_codons_with_terminal_and_mid_combined(tmp_path: Path) -> None:
    # Universal terminal TAA + a mid-stop in taxon a; with strip_stop_codons=True
    # terminal is stripped AND the mid column is dropped. Both indices appear in
    # stripped_sites, and no -2 leaks.
    path = _write(
        tmp_path, "combo.fa",
        ">a\nATGTAAAAATAA\n>b\nATGAAAAAATAA\n",
    )
    aln = read_fasta(
        path, genetic_code=GeneticCode.standard(), strip_stop_codons=True
    )
    assert -2 not in aln.codons.tolist()
    # Terminal (index 3) and mid (index 1) are both recorded.
    assert 1 in aln.stripped_sites
    assert 3 in aln.stripped_sites
    assert aln.codons.shape == (2, 2)


from selkit.io.alignment import read_phylip, read_alignment


def test_read_sequential_phylip(tmp_path: Path) -> None:
    content = "2 6\na         ATGAAA\nb         ATGAAG\n"
    path = _write(tmp_path, "s.phy", content)
    aln = read_phylip(path, genetic_code=GeneticCode.standard())
    assert aln.taxa == ("a", "b")
    assert aln.codons.shape == (2, 2)


def test_read_interleaved_phylip(tmp_path: Path) -> None:
    content = (
        "2 9\n"
        "a         ATG AAA\n"
        "b         ATG AAG\n"
        "\n"
        "          ATG\n"
        "          ATG\n"
    )
    path = _write(tmp_path, "i.phy", content)
    aln = read_phylip(path, genetic_code=GeneticCode.standard())
    assert aln.taxa == ("a", "b")
    assert aln.codons.shape == (2, 3)


def test_read_alignment_dispatches_to_fasta(tmp_path: Path) -> None:
    path = _write(tmp_path, "x.fa", ">a\nATGAAA\n>b\nATGAAG\n")
    aln = read_alignment(path, genetic_code=GeneticCode.standard())
    assert aln.taxa == ("a", "b")


def test_read_alignment_dispatches_to_phylip(tmp_path: Path) -> None:
    path = _write(tmp_path, "x.phy", "2 6\na ATGAAA\nb ATGAAG\n")
    aln = read_alignment(path, genetic_code=GeneticCode.standard())
    assert aln.taxa == ("a", "b")


def test_read_alignment_garbage_errors(tmp_path: Path) -> None:
    path = _write(tmp_path, "x.txt", "this is not an alignment\n")
    with pytest.raises(SelkitInputError, match=r"unrecognized"):
        read_alignment(path, genetic_code=GeneticCode.standard())
