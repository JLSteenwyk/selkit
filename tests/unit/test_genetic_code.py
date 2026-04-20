from __future__ import annotations

import pytest

from selkit.engine.genetic_code import GeneticCode


def test_standard_code_has_61_sense_codons() -> None:
    gc = GeneticCode.standard()
    assert gc.n_sense == 61
    assert len(gc.sense_codons) == 61


def test_standard_code_stop_codons() -> None:
    gc = GeneticCode.standard()
    assert gc.is_stop("TAA")
    assert gc.is_stop("TAG")
    assert gc.is_stop("TGA")
    assert not gc.is_stop("ATG")


def test_translate_methionine() -> None:
    gc = GeneticCode.standard()
    assert gc.translate("ATG") == "M"


def test_codon_index_round_trip() -> None:
    gc = GeneticCode.standard()
    for codon in gc.sense_codons:
        idx = gc.codon_to_index(codon)
        assert gc.index_to_codon(idx) == codon


def test_stop_codon_has_no_index() -> None:
    gc = GeneticCode.standard()
    with pytest.raises(KeyError):
        gc.codon_to_index("TAA")


def test_is_synonymous() -> None:
    gc = GeneticCode.standard()
    # Leucine: CTT, CTC, CTA, CTG, TTA, TTG
    assert gc.is_synonymous("CTT", "CTC")
    assert gc.is_synonymous("CTA", "TTA")
    # ATG (Met) vs ATA (Ile): nonsynonymous
    assert not gc.is_synonymous("ATG", "ATA")


def test_is_transition() -> None:
    gc = GeneticCode.standard()
    # ATG -> ACG differs at position 1 (T->C, pyrimidine->pyrimidine = transition)
    assert gc.is_transition("ATG", "ACG")
    # ATG -> AAG differs at position 1 (T->A, pyrimidine->purine = transversion)
    assert not gc.is_transition("ATG", "AAG")


def test_codons_differ_at_more_than_one_position() -> None:
    gc = GeneticCode.standard()
    assert gc.n_differences("ATG", "CTG") == 1
    assert gc.n_differences("ATG", "CCG") == 2
    assert gc.n_differences("ATG", "CCC") == 3
    assert gc.n_differences("ATG", "ATG") == 0


def test_unknown_table_name_raises() -> None:
    with pytest.raises(KeyError):
        GeneticCode.by_name("nonsense_table")


def test_vertebrate_mitochondrial_code_differs_from_standard() -> None:
    std = GeneticCode.standard()
    mito = GeneticCode.by_name("vertebrate_mitochondrial")
    # TGA is Trp in vertebrate mito, stop in standard
    assert std.is_stop("TGA")
    assert not mito.is_stop("TGA")
    assert mito.translate("TGA") == "W"
