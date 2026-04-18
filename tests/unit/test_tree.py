from __future__ import annotations

from pathlib import Path

import pytest

from selkit.errors import SelkitInputError
from selkit.io.tree import LabeledTree, parse_newick


def test_parse_simple_tree() -> None:
    tree = parse_newick("(a,b,c);")
    assert tree.tip_names == ("a", "b", "c")
    assert len(tree.internal_nodes) == 1


def test_parse_with_branch_lengths() -> None:
    tree = parse_newick("(a:0.1,b:0.2,c:0.3):0.0;")
    bls = {n.name: n.branch_length for n in tree.tips}
    assert bls == {"a": 0.1, "b": 0.2, "c": 0.3}


def test_parse_nested_tree() -> None:
    tree = parse_newick("((a:0.1,b:0.2):0.3,(c:0.4,d:0.5):0.6);")
    assert tree.tip_names == ("a", "b", "c", "d")
    assert len(tree.internal_nodes) == 3


def test_paml_hash_label_on_tip() -> None:
    tree = parse_newick("((a #1,b):0.1,c);")
    labels = {n.name: n.label for n in tree.tips}
    assert labels["a"] == 1
    assert labels["b"] == 0


def test_paml_hash_label_on_clade() -> None:
    tree = parse_newick("((a,b) #1,c);")
    clade = next(n for n in tree.internal_nodes if n.label == 1)
    assert {n.name for n in clade.tips_beneath()} == {"a", "b"}


def test_dollar_label() -> None:
    tree = parse_newick("((a,b) $1,c);")
    clade = next(n for n in tree.internal_nodes if n.label == 1)
    assert {n.name for n in clade.tips_beneath()} == {"a", "b"}


def test_comment_is_stripped() -> None:
    tree = parse_newick("(a[branch comment]:0.1,b:0.2,c);")
    assert tree.tip_names == ("a", "b", "c")


def test_duplicate_tip_raises() -> None:
    with pytest.raises(SelkitInputError, match=r"duplicate"):
        parse_newick("(a,a,b);")


def test_missing_semicolon_ok() -> None:
    # PAML sometimes omits trailing ';'
    tree = parse_newick("(a,b,c)")
    assert tree.tip_names == ("a", "b", "c")


def test_labeled_tree_has_unique_node_ids() -> None:
    tree = parse_newick("((a,b),(c,d));")
    ids = [n.id for n in tree.all_nodes()]
    assert len(ids) == len(set(ids))
