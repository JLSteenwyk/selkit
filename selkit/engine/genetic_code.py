from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

NUCS = ("T", "C", "A", "G")
PURINES = frozenset({"A", "G"})
PYRIMIDINES = frozenset({"C", "T"})

# NCBI standard code (table 1), in canonical TCAG × TCAG × TCAG order (64 codons).
_STANDARD_AAS = (
    "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
)
# Vertebrate mitochondrial (table 2): TGA=W, AGA/AGG=*, ATA=M.
_VERT_MITO_AAS = (
    "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIMMTTTTNNKKSS**VVVVAAAADDEEGGGG"
)

_ALL_CODONS: tuple[str, ...] = tuple(
    a + b + c for a in NUCS for b in NUCS for c in NUCS
)


@dataclass(frozen=True)
class GeneticCode:
    name: str
    aa_table: str  # 64-char string aligned with _ALL_CODONS

    sense_codons: tuple[str, ...] = field(init=False, compare=False)
    stop_codons: frozenset[str] = field(init=False, compare=False)
    _codon_to_idx: dict[str, int] = field(init=False, compare=False, repr=False)
    _idx_to_codon: tuple[str, ...] = field(init=False, compare=False, repr=False)

    _REGISTRY: ClassVar[dict[str, str]] = {
        "standard": _STANDARD_AAS,
        "vertebrate_mitochondrial": _VERT_MITO_AAS,
    }

    def __post_init__(self) -> None:
        if len(self.aa_table) != 64:
            raise ValueError(f"aa_table must be length 64, got {len(self.aa_table)}")
        sense: list[str] = []
        stops: set[str] = set()
        for codon, aa in zip(_ALL_CODONS, self.aa_table):
            if aa == "*":
                stops.add(codon)
            else:
                sense.append(codon)
        object.__setattr__(self, "sense_codons", tuple(sense))
        object.__setattr__(self, "stop_codons", frozenset(stops))
        object.__setattr__(
            self, "_codon_to_idx", {c: i for i, c in enumerate(sense)}
        )
        object.__setattr__(self, "_idx_to_codon", tuple(sense))

    @classmethod
    def standard(cls) -> GeneticCode:
        return cls.by_name("standard")

    @classmethod
    def by_name(cls, name: str) -> GeneticCode:
        if name not in cls._REGISTRY:
            raise KeyError(f"unknown genetic code: {name!r}")
        return cls(name=name, aa_table=cls._REGISTRY[name])

    @property
    def n_sense(self) -> int:
        return len(self.sense_codons)

    def is_stop(self, codon: str) -> bool:
        return codon.upper() in self.stop_codons

    def translate(self, codon: str) -> str:
        idx = _ALL_CODONS.index(codon.upper())
        return self.aa_table[idx]

    def codon_to_index(self, codon: str) -> int:
        codon = codon.upper()
        if codon not in self._codon_to_idx:
            raise KeyError(f"{codon!r} is not a sense codon in {self.name}")
        return self._codon_to_idx[codon]

    def index_to_codon(self, idx: int) -> str:
        return self._idx_to_codon[idx]

    def is_synonymous(self, a: str, b: str) -> bool:
        return self.translate(a) == self.translate(b)

    def is_transition(self, a: str, b: str) -> bool:
        diffs = [(x, y) for x, y in zip(a.upper(), b.upper()) if x != y]
        if len(diffs) != 1:
            raise ValueError("is_transition requires codons differing at exactly one position")
        x, y = diffs[0]
        return (x in PURINES and y in PURINES) or (x in PYRIMIDINES and y in PYRIMIDINES)

    def n_differences(self, a: str, b: str) -> int:
        return sum(1 for x, y in zip(a.upper(), b.upper()) if x != y)
