from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from selkit.errors import SelkitInputError

_SPECIAL = set("(),:;#$[]")


@dataclass
class Node:
    id: int
    name: str | None = None
    branch_length: float | None = None
    label: int = 0
    children: list["Node"] = field(default_factory=list)
    parent: "Node | None" = None

    @property
    def is_tip(self) -> bool:
        return not self.children

    def tips_beneath(self) -> Iterator["Node"]:
        if self.is_tip:
            yield self
            return
        for child in self.children:
            yield from child.tips_beneath()


@dataclass
class LabeledTree:
    root: Node
    newick: str
    labels: dict[int, int]
    tip_order: tuple[str, ...]

    @property
    def tips(self) -> list[Node]:
        return [n for n in self.all_nodes() if n.is_tip]

    @property
    def tip_names(self) -> tuple[str, ...]:
        return tuple(n.name or "" for n in self.tips)

    @property
    def internal_nodes(self) -> list[Node]:
        return [n for n in self.all_nodes() if not n.is_tip]

    def all_nodes(self) -> list[Node]:
        out: list[Node] = []
        stack: list[Node] = [self.root]
        while stack:
            n = stack.pop()
            out.append(n)
            # reversed() preserves left-to-right document order under stack-based DFS.
            stack.extend(reversed(n.children))
        return out


def _strip_comments(s: str) -> str:
    return re.sub(r"\[[^\]]*\]", "", s)


def _tokenize(s: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c in _SPECIAL:
            tokens.append(c)
            i += 1
            continue
        j = i
        while j < len(s) and s[j] not in _SPECIAL and not s[j].isspace():
            j += 1
        tokens.append(s[i:j])
        i = j
    return tokens


class _Parser:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.pos = 0
        self.next_id = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self) -> str:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def parse(self) -> Node:
        root = self._parse_subtree()
        if self.peek() == ";":
            self.consume()
        return root

    def _parse_subtree(self) -> Node:
        node = Node(id=self.next_id)
        self.next_id += 1
        if self.peek() == "(":
            self.consume()
            while True:
                child = self._parse_subtree()
                child.parent = node
                node.children.append(child)
                if self.peek() == ",":
                    self.consume()
                    continue
                break
            if self.peek() != ")":
                raise SelkitInputError(f"expected ')' near token {self.peek()!r}")
            self.consume()
        if self.peek() is not None and self.peek() not in set(",():;#$"):
            node.name = self.consume()
        if self.peek() == ":":
            self.consume()
            try:
                node.branch_length = float(self.consume())
            except ValueError as e:
                raise SelkitInputError(f"bad branch length: {e}") from e
        if self.peek() in {"#", "$"}:
            self.consume()
            try:
                node.label = int(self.consume())
            except ValueError as e:
                raise SelkitInputError(f"bad branch label: {e}") from e
        return node


def parse_newick(s: str) -> LabeledTree:
    stripped = _strip_comments(s)
    tokens = _tokenize(stripped)
    if not tokens:
        raise SelkitInputError("empty tree string")
    root = _Parser(tokens).parse()
    tree = LabeledTree(
        root=root,
        newick=_canonicalize(root),
        labels={n.id: n.label for n in _iter_nodes(root) if n.label},
        tip_order=tuple(n.name or "" for n in _iter_nodes(root) if not n.children),
    )
    names = [n for n in tree.tip_names if n]
    if len(set(names)) != len(names):
        raise SelkitInputError("duplicate tip names in tree")
    return tree


def _iter_nodes(root: Node) -> Iterator[Node]:
    stack: list[Node] = [root]
    while stack:
        n = stack.pop()
        yield n
        # reversed() preserves document order under stack-based DFS.
        stack.extend(reversed(n.children))


def _canonicalize(root: Node) -> str:
    def fmt(n: Node) -> str:
        if n.is_tip:
            base = n.name or ""
        else:
            base = "(" + ",".join(fmt(c) for c in n.children) + ")"
        if n.branch_length is not None:
            base += f":{n.branch_length:g}"
        return base
    return fmt(root) + ";"


@dataclass(frozen=True)
class ForegroundSpec:
    tips: tuple[str, ...] = ()
    mrca: tuple[str, ...] = ()
    labels: dict[int, int] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not (self.tips or self.mrca or self.labels)


def _mrca(tree: LabeledTree, names: tuple[str, ...]) -> Node:
    target = set(names)

    def contains_all(n: Node) -> bool:
        return target.issubset({t.name for t in n.tips_beneath() if t.name})

    candidates = [n for n in tree.all_nodes() if contains_all(n)]
    if not candidates:
        raise SelkitInputError(f"no node covers tips {names}")
    candidates.sort(key=lambda n: sum(1 for _ in n.tips_beneath()))
    return candidates[0]


def apply_foreground_spec(tree: LabeledTree, spec: ForegroundSpec) -> LabeledTree:
    if spec.is_empty:
        return tree
    has_in_tree_label = any(n.label != 0 for n in tree.all_nodes())
    if has_in_tree_label and (spec.tips or spec.mrca or spec.labels):
        raise SelkitInputError(
            "conflicting branch labels: tree already has #-labels and an external "
            "ForegroundSpec was also provided"
        )
    known_tips = {n.name for n in tree.tips if n.name}
    for t in (*spec.tips, *spec.mrca):
        if t not in known_tips:
            raise SelkitInputError(f"unknown tip in foreground spec: {t!r}")
    if spec.tips:
        target = set(spec.tips)
        for n in tree.all_nodes():
            if n.is_tip and n.name in target:
                n.label = 1
    if spec.mrca:
        _mrca(tree, spec.mrca).label = 1
    for node_id, lab in spec.labels.items():
        for n in tree.all_nodes():
            if n.id == node_id:
                n.label = lab
                break
    new_labels = {n.id: n.label for n in tree.all_nodes() if n.label}
    return LabeledTree(
        root=tree.root,
        newick=tree.newick,
        labels=new_labels,
        tip_order=tree.tip_order,
    )


def load_labels_file(path: Path) -> ForegroundSpec:
    lines = Path(path).read_text().splitlines()
    if not lines:
        raise SelkitInputError(f"empty labels file: {path}")
    header = lines[0].split("\t")
    if header != ["taxon", "label"]:
        raise SelkitInputError(
            f"labels file must have header 'taxon\\tlabel', got {header}"
        )
    tips: list[str] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2 or parts[1].strip() != "1":
            raise SelkitInputError(f"labels file only supports label=1 rows; got {line!r}")
        tips.append(parts[0].strip())
    return ForegroundSpec(tips=tuple(tips))
