from __future__ import annotations

import re
from dataclasses import dataclass, field
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
