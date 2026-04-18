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
