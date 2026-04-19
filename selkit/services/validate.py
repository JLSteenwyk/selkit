from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from selkit.engine.genetic_code import GeneticCode
from selkit.errors import SelkitInputError
from selkit.io.alignment import CodonAlignment, read_alignment
from selkit.io.tree import ForegroundSpec, LabeledTree, apply_foreground_spec, parse_newick


@dataclass(frozen=True)
class ValidatedInputs:
    alignment: CodonAlignment
    tree: LabeledTree


def validate_inputs(
    *,
    alignment_path: Path,
    tree_path: Path,
    foreground_spec: ForegroundSpec,
    genetic_code_name: str,
    strip_terminal_stop: bool = True,
    strip_stop_codons: bool = False,
) -> ValidatedInputs:
    gc = GeneticCode.by_name(genetic_code_name)
    aln = read_alignment(
        alignment_path, genetic_code=gc,
        strip_terminal_stop=strip_terminal_stop,
        strip_stop_codons=strip_stop_codons,
    )
    tree_text = Path(tree_path).read_text()
    tree = parse_newick(tree_text)
    tree = apply_foreground_spec(tree, foreground_spec)
    aln_taxa = set(aln.taxa)
    tree_taxa = {n for n in tree.tip_names if n}
    if aln_taxa != tree_taxa:
        missing_in_tree = aln_taxa - tree_taxa
        missing_in_aln = tree_taxa - aln_taxa
        raise SelkitInputError(
            "taxon mismatch between alignment and tree\n"
            f"  alignment-only: {sorted(missing_in_tree)}\n"
            f"  tree-only:      {sorted(missing_in_aln)}\n"
            "  hint: add --prune-unmatched to drop missing tips/taxa"
        )
    return ValidatedInputs(alignment=aln, tree=tree)
