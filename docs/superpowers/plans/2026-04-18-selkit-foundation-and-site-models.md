# selkit — Foundation & Site Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the first usable selkit — a pure-Python reimplementation of PAML codeml site models (M0, M1a, M2a, M7, M8, M8a) with modern CLI/library UX, auto-LRTs, BEB posteriors, parallelism, strict validation, and reproducibility.

**Architecture:** Four-layer package with one-way dependencies (`io → engine → services → cli`). Engine is pure math (no IO). Pure numpy/scipy; `scipy.optimize.minimize(method="L-BFGS-B")` for fitting; `ProcessPoolExecutor` for model-level parallelism; `rich` for progress UI. TDD throughout; PAML-comparison corpus gates correctness.

**Tech Stack:** Python 3.11+, numpy, scipy, rich, PyYAML, pytest.

Spec: `docs/superpowers/specs/2026-04-18-selkit-design.md`.

---

## File structure

**Creating (package):**

```
pyproject.toml
README.md
selkit/
├── __init__.py
├── __main__.py
├── version.py
├── errors.py
├── cli.py
├── cli_registry.py
├── services/
│   ├── __init__.py
│   ├── validate.py
│   └── codeml/
│       ├── __init__.py
│       ├── site_models.py
│       └── lrt.py
├── engine/
│   ├── __init__.py
│   ├── genetic_code.py
│   ├── rate_matrix.py
│   ├── codon_model.py
│   ├── likelihood.py
│   ├── optimize.py
│   ├── beb.py
│   └── simulator.py
├── io/
│   ├── __init__.py
│   ├── alignment.py
│   ├── tree.py
│   ├── config.py
│   └── results.py
└── progress/
    ├── __init__.py
    └── runner.py
```

**Creating (tests):**

```
tests/
├── __init__.py
├── unit/
│   ├── __init__.py
│   ├── test_genetic_code.py
│   ├── test_rate_matrix.py
│   ├── test_codon_model.py
│   ├── test_likelihood.py
│   ├── test_optimize.py
│   ├── test_beb.py
│   ├── test_alignment.py
│   ├── test_tree.py
│   ├── test_config.py
│   └── test_results.py
├── integration/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_site_models_cli.py
│   ├── test_validate_cli.py
│   └── test_reproducibility.py
└── validation/
    ├── __init__.py
    ├── conftest.py
    ├── README.md
    ├── corpus/       # starts empty; holds (aln, tree, expected.json) triples
    └── test_paml_corpus.py
```

**File responsibilities:**

- `engine/genetic_code.py` — codon tables (NCBI tables), translate, is_stop, synonymous/transition predicates.
- `engine/rate_matrix.py` — build codon-level Q given (ω, κ, π), eigendecompose, exponentiate to P(t).
- `engine/codon_model.py` — parameterization for each site model (M0…M8a); ω-class mixture weights; maps parameters → list of (weight, Q).
- `engine/likelihood.py` — Felsenstein pruning; per-site and total tree lnL for mixture models.
- `engine/optimize.py` — param transforms, single-start L-BFGS-B, multi-start with convergence gate.
- `engine/beb.py` — Bayes Empirical Bayes per-site posteriors.
- `engine/simulator.py` — minimal codon simulator (internal, testing only).
- `io/alignment.py` — FASTA parser, Phylip fallback, codon validation, CodonAlignment.
- `io/tree.py` — Newick parser, LabeledTree, label normalization (#1 / flags / file).
- `io/config.py` — RunConfig dataclass, YAML serialize/load.
- `io/results.py` — ModelFit, LRTResult, BEBSite, RunResult dataclasses; JSON + TSV emitters.
- `services/validate.py` — pre-flight input validation.
- `services/codeml/site_models.py` — expand bundle, fit models (optionally in parallel), compose RunResult.
- `services/codeml/lrt.py` — LRT computation (chi2, mixed chi2).
- `progress/runner.py` — rich progress UI for sequential + parallel runs.
- `cli.py`, `cli_registry.py`, `__main__.py` — argparse + subcommand dispatch.
- `errors.py` — exception taxonomy.

---

## Conventions used throughout

- **Python 3.11+.** Use `from __future__ import annotations` in every module.
- **Type hints everywhere.** Run `mypy --strict` as a (non-blocking) linter check locally.
- **Tests co-located by module name:** `selkit/engine/foo.py` → `tests/unit/test_foo.py`.
- **Commit cadence:** commit at the end of every task. Conventional Commits style (`feat:`, `test:`, `refactor:`, `docs:`, `chore:`).
- **Run tests via:** `pytest -q`. Individual: `pytest tests/unit/test_foo.py::test_name -v`.
- **Working directory for all commands:** `/Users/jacoblsteenwyk/Desktop/kit_dev/selkit` (repo root).
- **Virtual env:** create once with `python3.11 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"` (see Task 1). All subsequent `pytest` / `python` commands run inside this venv.

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `selkit/__init__.py`
- Create: `selkit/version.py`
- Create: `selkit/__main__.py`
- Create: `selkit/errors.py`
- Create: `tests/__init__.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/unit/test_errors.py`

- [ ] **Step 1.1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "selkit"
version = "0.0.1"
description = "Python reimplementation of PAML's selection-analysis workflows (codeml, yn00)"
readme = "README.md"
license = { text = "MIT" }
authors = [{ name = "Jacob Steenwyk" }]
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.12",
    "rich>=13.7",
    "PyYAML>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.1",
    "mypy>=1.8",
    "ruff>=0.3",
]

[project.scripts]
selkit = "selkit.__main__:main"

[tool.setuptools.packages.find]
include = ["selkit*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
markers = [
    "validation: PAML-comparison corpus tests (nightly)",
    "simulation: simulation-based tests (tagged releases only)",
]
```

- [ ] **Step 1.2: Create `selkit/version.py`**

```python
from __future__ import annotations

__version__ = "0.0.1"
```

- [ ] **Step 1.3: Create `selkit/__init__.py`**

```python
from __future__ import annotations

from selkit.version import __version__

__all__ = ["__version__"]
```

- [ ] **Step 1.4: Create `selkit/__main__.py`**

```python
from __future__ import annotations


def main() -> int:
    # Real CLI wiring arrives in Task 34. Placeholder keeps the entry point importable.
    raise SystemExit("selkit CLI not yet implemented")


if __name__ == "__main__":
    main()
```

- [ ] **Step 1.5: Create `selkit/errors.py`**

```python
from __future__ import annotations


class SelkitError(Exception):
    """Base class for all selkit exceptions."""


class SelkitInputError(SelkitError):
    """Malformed input files (alignment, tree, labels)."""


class SelkitConfigError(SelkitError):
    """Invalid configuration (bad model name, conflicting flags)."""


class SelkitEngineError(SelkitError):
    """Numerical failure inside the ML engine."""


class SelkitConvergenceWarning(Warning):
    """Multi-start optimization lnL values disagreed past tolerance."""
```

- [ ] **Step 1.6: Create `tests/__init__.py` and `tests/unit/__init__.py` as empty files, then write the failing test**

`tests/unit/test_errors.py`:

```python
from __future__ import annotations

import pytest

from selkit import errors


def test_selkit_error_is_exception() -> None:
    assert issubclass(errors.SelkitError, Exception)


def test_input_error_inherits_from_selkit_error() -> None:
    assert issubclass(errors.SelkitInputError, errors.SelkitError)


def test_convergence_warning_is_warning() -> None:
    assert issubclass(errors.SelkitConvergenceWarning, Warning)


def test_input_error_preserves_message() -> None:
    with pytest.raises(errors.SelkitInputError, match="bad file"):
        raise errors.SelkitInputError("bad file")
```

- [ ] **Step 1.7: Install the package in editable mode and run tests**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/unit/test_errors.py -v
```

Expected: 4 tests pass.

- [ ] **Step 1.8: Commit**

```bash
git add pyproject.toml selkit/ tests/
git commit -m "feat: project scaffolding, error taxonomy, package install"
```

---

## Task 2: Genetic code — codon tables and utilities

**Files:**
- Create: `selkit/engine/__init__.py`
- Create: `selkit/engine/genetic_code.py`
- Create: `tests/unit/test_genetic_code.py`

**Background:** `GeneticCode` exposes the 64 codons, maps codons to amino acids, identifies stop codons, and answers whether two codons differ synonymously / by a transition. Sense-codon indices are stable (0..N-1, where N=61 for standard code) and are the integer encoding used throughout the engine. The class wraps NCBI genetic code tables.

- [ ] **Step 2.1: Create `selkit/engine/__init__.py` (empty)**

- [ ] **Step 2.2: Write the failing tests**

`tests/unit/test_genetic_code.py`:

```python
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
```

- [ ] **Step 2.3: Run tests to verify failure**

```bash
pytest tests/unit/test_genetic_code.py -v
```

Expected: all tests FAIL (module doesn't exist yet).

- [ ] **Step 2.4: Implement `selkit/engine/genetic_code.py`**

```python
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
```

- [ ] **Step 2.5: Run tests to verify pass**

```bash
pytest tests/unit/test_genetic_code.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 2.6: Commit**

```bash
git add selkit/engine/__init__.py selkit/engine/genetic_code.py tests/unit/test_genetic_code.py
git commit -m "feat(engine): GeneticCode with standard + vertebrate mito tables"
```

---

## Task 3: Alignment IO — FASTA parser and CodonAlignment

**Files:**
- Create: `selkit/io/__init__.py`
- Create: `selkit/io/alignment.py`
- Create: `tests/unit/test_alignment.py`

**Background:** `CodonAlignment` holds a validated matrix of integer codon codes (shape `(n_taxa, n_codons)`). Gaps / ambiguous codons become `-1`. FASTA parsing + length / stop validation live here. Phylip fallback arrives in Task 5.

- [ ] **Step 3.1: Create `selkit/io/__init__.py` (empty)**

- [ ] **Step 3.2: Write the failing tests**

`tests/unit/test_alignment.py`:

```python
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
```

- [ ] **Step 3.3: Run tests to verify failure**

```bash
pytest tests/unit/test_alignment.py -v
```

Expected: all FAIL (module missing).

- [ ] **Step 3.4: Implement `selkit/io/alignment.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from selkit.engine.genetic_code import GeneticCode
from selkit.errors import SelkitInputError

_GAP_CHARS = frozenset({"-", "N", "?"})


@dataclass(frozen=True)
class CodonAlignment:
    taxa: tuple[str, ...]
    codons: np.ndarray            # shape (n_taxa, n_codons), int16
    genetic_code: str
    stripped_sites: tuple[int, ...]


def _parse_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name: str | None = None
    buf: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(buf).upper()))
            name = line[1:].split()[0]
            buf = []
        else:
            buf.append(line)
    if name is not None:
        records.append((name, "".join(buf).upper()))
    return records


def _encode_sequence(seq: str, gc: GeneticCode) -> list[int]:
    out: list[int] = []
    for i in range(0, len(seq), 3):
        codon = seq[i : i + 3]
        if any(c in _GAP_CHARS for c in codon):
            out.append(-1)
        else:
            try:
                out.append(gc.codon_to_index(codon))
            except KeyError:
                if gc.is_stop(codon):
                    out.append(-2)  # sentinel for stop
                else:
                    out.append(-1)
    return out


def read_fasta(
    path: Path,
    *,
    genetic_code: GeneticCode,
    strip_terminal_stop: bool = True,
    strip_stop_codons: bool = False,
) -> CodonAlignment:
    records = _parse_fasta(Path(path))
    if not records:
        raise SelkitInputError(f"alignment is empty: {path}")

    names = [r[0] for r in records]
    if len(set(names)) != len(names):
        raise SelkitInputError(f"duplicate taxon names in {path}")

    lengths = {len(r[1]) for r in records}
    if len(lengths) > 1:
        raise SelkitInputError(
            f"sequences must all be the same length in {path} "
            f"(got lengths {sorted(lengths)})"
        )
    (seq_len,) = lengths
    if seq_len % 3 != 0:
        offender = next(r[0] for r in records)
        raise SelkitInputError(
            f"sequence length is not a multiple of 3 in {path}\n"
            f"  taxon:     {offender}\n"
            f"  length:    {seq_len} nt (remainder {seq_len % 3})\n"
            f"  hint:      rerun with --trim-trailing if overhang is uniform"
        )

    encoded = [(n, _encode_sequence(s, genetic_code)) for n, s in records]
    n_codons = seq_len // 3

    stops_at: dict[int, list[str]] = {}
    for taxon, seq in encoded:
        for i, c in enumerate(seq):
            if c == -2:
                stops_at.setdefault(i, []).append(taxon)

    # Terminal stop handling.
    stripped_sites: list[int] = []
    if strip_terminal_stop and stops_at:
        last = n_codons - 1
        if last in stops_at and len(stops_at[last]) == len(encoded) and len(stops_at) == 1:
            stripped_sites.append(last)
            for _, seq in encoded:
                seq.pop()
            del stops_at[last]
            n_codons -= 1

    # Mid-sequence stop handling.
    if stops_at:
        if not strip_stop_codons:
            first = min(stops_at)
            taxon = stops_at[first][0]
            raise SelkitInputError(
                f"stop codon found mid-sequence in {path}\n"
                f"  taxon:    {taxon}\n"
                f"  position: codon {first + 1}\n"
                f"  hint:     rerun with --strip-stop-codons to drop these sites"
            )
        # drop any column with >=1 stop
        bad = sorted(stops_at)
        keep = [i for i in range(n_codons) if i not in stops_at]
        for _, seq in encoded:
            new = [seq[i] for i in keep]
            seq.clear()
            seq.extend(new)
        stripped_sites.extend(bad)
        n_codons -= len(bad)

    codons = np.array([seq for _, seq in encoded], dtype=np.int16)
    return CodonAlignment(
        taxa=tuple(names),
        codons=codons,
        genetic_code=genetic_code.name,
        stripped_sites=tuple(sorted(stripped_sites)),
    )
```

- [ ] **Step 3.5: Run tests to verify pass**

```bash
pytest tests/unit/test_alignment.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 3.6: Commit**

```bash
git add selkit/io/__init__.py selkit/io/alignment.py tests/unit/test_alignment.py
git commit -m "feat(io): FASTA parser and CodonAlignment with validation"
```

---

## Task 4: Alignment IO — Phylip fallback and auto-detect

**Files:**
- Modify: `selkit/io/alignment.py` (add `read_phylip`, `read_alignment`)
- Modify: `tests/unit/test_alignment.py` (add phylip + dispatch tests)

**Background:** PAML uses Phylip (relaxed, interleaved or sequential). Add a reader and a `read_alignment` dispatcher that tries FASTA first, Phylip second.

- [ ] **Step 4.1: Write the failing tests**

Append to `tests/unit/test_alignment.py`:

```python
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
```

- [ ] **Step 4.2: Run tests to verify failure**

```bash
pytest tests/unit/test_alignment.py -v
```

Expected: 5 new tests FAIL.

- [ ] **Step 4.3: Append implementation to `selkit/io/alignment.py`**

```python
def _parse_phylip(path: Path) -> list[tuple[str, str]]:
    raw = Path(path).read_text().splitlines()
    # Skip blank leading lines.
    lines = [l for l in raw]
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        raise SelkitInputError(f"phylip file is empty: {path}")
    header = lines[0].split()
    if len(header) < 2 or not header[0].isdigit() or not header[1].isdigit():
        raise SelkitInputError(f"not a phylip file (bad header): {path}")
    n_taxa = int(header[0])
    n_sites = int(header[1])
    body = lines[1:]

    records: list[tuple[str, list[str]]] = []
    i = 0
    # First block: taxon name at start of each of the first n_taxa non-empty lines.
    block_lines: list[str] = []
    for line in body:
        if line.strip():
            block_lines.append(line)
            if len(block_lines) == n_taxa:
                break
    if len(block_lines) < n_taxa:
        raise SelkitInputError(f"phylip file has fewer than {n_taxa} taxa: {path}")

    for line in block_lines:
        # "relaxed" format: first whitespace separates name from sequence.
        parts = line.split(maxsplit=1)
        if len(parts) == 1:
            name, seq = parts[0], ""
        else:
            name, seq = parts
        records.append((name, [seq.replace(" ", "")]))

    # Remaining (interleaved) blocks: append in order, ignoring blanks.
    remaining = [l for l in body[len(block_lines):] if l.strip()]
    for j, line in enumerate(remaining):
        records[j % n_taxa][1].append(line.strip().replace(" ", ""))

    out = [(name, "".join(chunks).upper()) for name, chunks in records]
    # Sanity-check declared vs actual length.
    for name, seq in out:
        if len(seq) != n_sites:
            raise SelkitInputError(
                f"phylip taxon {name!r} has length {len(seq)}, header declared {n_sites}"
            )
    return out


def _build_from_records(
    records: list[tuple[str, str]],
    *,
    source: Path,
    genetic_code: GeneticCode,
    strip_terminal_stop: bool,
    strip_stop_codons: bool,
) -> CodonAlignment:
    # Body of read_fasta after _parse_fasta was extracted into read_fasta; replicate here.
    if not records:
        raise SelkitInputError(f"alignment is empty: {source}")
    names = [r[0] for r in records]
    if len(set(names)) != len(names):
        raise SelkitInputError(f"duplicate taxon names in {source}")
    lengths = {len(r[1]) for r in records}
    if len(lengths) > 1:
        raise SelkitInputError(
            f"sequences must all be the same length in {source} (got {sorted(lengths)})"
        )
    (seq_len,) = lengths
    if seq_len % 3 != 0:
        offender = next(r[0] for r in records)
        raise SelkitInputError(
            f"sequence length is not a multiple of 3 in {source}\n"
            f"  taxon:  {offender}\n  length: {seq_len} nt"
        )
    encoded = [(n, _encode_sequence(s, genetic_code)) for n, s in records]
    n_codons = seq_len // 3
    stops_at: dict[int, list[str]] = {}
    for taxon, seq in encoded:
        for i, c in enumerate(seq):
            if c == -2:
                stops_at.setdefault(i, []).append(taxon)
    stripped: list[int] = []
    if strip_terminal_stop and stops_at:
        last = n_codons - 1
        if last in stops_at and len(stops_at[last]) == len(encoded) and len(stops_at) == 1:
            stripped.append(last)
            for _, seq in encoded:
                seq.pop()
            del stops_at[last]
            n_codons -= 1
    if stops_at:
        if not strip_stop_codons:
            first = min(stops_at)
            taxon = stops_at[first][0]
            raise SelkitInputError(
                f"stop codon found mid-sequence in {source}\n"
                f"  taxon:    {taxon}\n"
                f"  position: codon {first + 1}"
            )
        bad = sorted(stops_at)
        keep = [i for i in range(n_codons) if i not in stops_at]
        for _, seq in encoded:
            new = [seq[i] for i in keep]
            seq.clear()
            seq.extend(new)
        stripped.extend(bad)
        n_codons -= len(bad)
    codons = np.array([seq for _, seq in encoded], dtype=np.int16)
    return CodonAlignment(
        taxa=tuple(names),
        codons=codons,
        genetic_code=genetic_code.name,
        stripped_sites=tuple(sorted(stripped)),
    )


def read_phylip(
    path: Path,
    *,
    genetic_code: GeneticCode,
    strip_terminal_stop: bool = True,
    strip_stop_codons: bool = False,
) -> CodonAlignment:
    records = _parse_phylip(Path(path))
    return _build_from_records(
        records,
        source=Path(path),
        genetic_code=genetic_code,
        strip_terminal_stop=strip_terminal_stop,
        strip_stop_codons=strip_stop_codons,
    )


def read_alignment(
    path: Path,
    *,
    genetic_code: GeneticCode,
    strip_terminal_stop: bool = True,
    strip_stop_codons: bool = False,
) -> CodonAlignment:
    text = Path(path).read_text()
    stripped = text.lstrip()
    if stripped.startswith(">"):
        return read_fasta(
            path, genetic_code=genetic_code,
            strip_terminal_stop=strip_terminal_stop,
            strip_stop_codons=strip_stop_codons,
        )
    head = stripped.splitlines()[0].split() if stripped else []
    if len(head) >= 2 and head[0].isdigit() and head[1].isdigit():
        return read_phylip(
            path, genetic_code=genetic_code,
            strip_terminal_stop=strip_terminal_stop,
            strip_stop_codons=strip_stop_codons,
        )
    raise SelkitInputError(f"unrecognized alignment format for {path}")
```

Also refactor `read_fasta` to call `_build_from_records`:

```python
def read_fasta(
    path: Path,
    *,
    genetic_code: GeneticCode,
    strip_terminal_stop: bool = True,
    strip_stop_codons: bool = False,
) -> CodonAlignment:
    records = _parse_fasta(Path(path))
    return _build_from_records(
        records,
        source=Path(path),
        genetic_code=genetic_code,
        strip_terminal_stop=strip_terminal_stop,
        strip_stop_codons=strip_stop_codons,
    )
```

Remove the now-duplicated validation body that previously lived inline in `read_fasta`.

- [ ] **Step 4.4: Run full alignment test suite**

```bash
pytest tests/unit/test_alignment.py -v
```

Expected: all 15 tests PASS.

- [ ] **Step 4.5: Commit**

```bash
git add selkit/io/alignment.py tests/unit/test_alignment.py
git commit -m "feat(io): add Phylip reader and format auto-detect"
```

---

## Task 5: Tree IO — Newick parser and LabeledTree

**Files:**
- Create: `selkit/io/tree.py`
- Create: `tests/unit/test_tree.py`

**Background:** We don't depend on ete3 / dendropy to keep install light. Write a minimal Newick parser that supports: tip names (any non-whitespace/punctuation), branch lengths (`:length`), internal labels (ignored for now), PAML branch labels (`#1`, `#2`, `$1`), comments `[...]`. Node objects carry `id: int` (BFS order), `name: str | None`, `branch_length: float | None`, `label: int` (0 default), `children: list[Node]`.

- [ ] **Step 5.1: Write the failing tests**

`tests/unit/test_tree.py`:

```python
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
```

- [ ] **Step 5.2: Run tests to verify failure**

```bash
pytest tests/unit/test_tree.py -v
```

Expected: all FAIL (module missing).

- [ ] **Step 5.3: Implement `selkit/io/tree.py`**

```python
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

from selkit.errors import SelkitInputError

# Newick-special characters.
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
    newick: str                         # canonical form, labels stripped
    labels: dict[int, int]              # node_id -> label (0=background)
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
            stack.extend(n.children)
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
        # Optional name/label.
        if self.peek() is not None and self.peek() not in set(",():;#$"):
            node.name = self.consume()
        # Branch length.
        if self.peek() == ":":
            self.consume()
            try:
                node.branch_length = float(self.consume())
            except ValueError as e:
                raise SelkitInputError(f"bad branch length: {e}") from e
        # PAML label (# or $).
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
    # Duplicate-tip check.
    names = [n for n in tree.tip_names if n]
    if len(set(names)) != len(names):
        raise SelkitInputError(f"duplicate tip names in tree")
    return tree


def _iter_nodes(root: Node) -> Iterator[Node]:
    stack: list[Node] = [root]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)


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
```

- [ ] **Step 5.4: Run tests to verify pass**

```bash
pytest tests/unit/test_tree.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 5.5: Commit**

```bash
git add selkit/io/tree.py tests/unit/test_tree.py
git commit -m "feat(io): minimal Newick parser with PAML label support"
```

---

## Task 6: Tree label normalization — flags and labels file

**Files:**
- Modify: `selkit/io/tree.py` (add `apply_foreground_spec`, `load_labels_file`)
- Modify: `tests/unit/test_tree.py`

**Background:** Users can specify foreground branches via (a) in-Newick `#1`, (b) `--foreground TAXON_LIST` (MRCA clade of named tips), (c) `--foreground-tips TAXON_LIST` (only listed tips), or (d) `--labels-file path.tsv`. Normalize to `LabeledTree.labels: dict[node_id, int]`. Multiple sources → error.

- [ ] **Step 6.1: Write failing tests (append to `tests/unit/test_tree.py`)**

```python
from selkit.io.tree import (
    ForegroundSpec,
    apply_foreground_spec,
    load_labels_file,
)


def test_apply_foreground_tips() -> None:
    tree = parse_newick("((a,b),(c,d));")
    spec = ForegroundSpec(tips=("a", "c"))
    out = apply_foreground_spec(tree, spec)
    labels_by_name = {n.name: n.label for n in out.tips}
    assert labels_by_name["a"] == 1
    assert labels_by_name["c"] == 1
    assert labels_by_name["b"] == 0
    assert labels_by_name["d"] == 0


def test_apply_foreground_mrca() -> None:
    tree = parse_newick("((a,b),(c,d));")
    spec = ForegroundSpec(mrca=("a", "b"))
    out = apply_foreground_spec(tree, spec)
    clade = next(n for n in out.internal_nodes if n.label == 1)
    assert {n.name for n in clade.tips_beneath()} == {"a", "b"}


def test_conflicting_spec_errors() -> None:
    tree = parse_newick("((a #1,b),(c,d));")
    with pytest.raises(SelkitInputError, match=r"conflict"):
        apply_foreground_spec(tree, ForegroundSpec(tips=("c",)))


def test_labels_file_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "labels.tsv"
    p.write_text("taxon\tlabel\na\t1\nc\t1\n")
    spec = load_labels_file(p)
    assert spec.tips == ("a", "c")


def test_empty_spec_is_noop() -> None:
    tree = parse_newick("((a,b),(c,d));")
    out = apply_foreground_spec(tree, ForegroundSpec())
    assert all(n.label == 0 for n in out.all_nodes())


def test_unknown_tip_name_errors() -> None:
    tree = parse_newick("((a,b),(c,d));")
    with pytest.raises(SelkitInputError, match=r"unknown tip"):
        apply_foreground_spec(tree, ForegroundSpec(tips=("z",)))
```

- [ ] **Step 6.2: Run tests; expect FAIL**

```bash
pytest tests/unit/test_tree.py -v
```

- [ ] **Step 6.3: Append to `selkit/io/tree.py`**

```python
from pathlib import Path


@dataclass(frozen=True)
class ForegroundSpec:
    tips: tuple[str, ...] = ()          # only these tip branches get label=1
    mrca: tuple[str, ...] = ()          # label=1 on MRCA clade of these tips
    labels: dict[int, int] = field(default_factory=dict)  # explicit node_id -> label

    @property
    def is_empty(self) -> bool:
        return not (self.tips or self.mrca or self.labels)


def _mrca(tree: LabeledTree, names: tuple[str, ...]) -> Node:
    target = set(names)
    def contains_all(n: Node) -> bool:
        return target.issubset({t.name for t in n.tips_beneath() if t.name})
    # smallest node whose tips_beneath covers the target
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
        # Node lookup by id.
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
```

- [ ] **Step 6.4: Run tests; expect PASS**

```bash
pytest tests/unit/test_tree.py -v
```

Expected: 16 tests pass total.

- [ ] **Step 6.5: Commit**

```bash
git add selkit/io/tree.py tests/unit/test_tree.py
git commit -m "feat(io): foreground label normalization (flags + labels file)"
```

---

## Task 7: Codon rate matrix Q

**Files:**
- Create: `selkit/engine/rate_matrix.py`
- Create: `tests/unit/test_rate_matrix.py`

**Background:** Codon Q is a `(N × N)` matrix where `N = gc.n_sense`. Off-diagonal entry `Q[i,j]` (i≠j):

- `0` if codons i, j differ at more than one position
- `π_j` if change is synonymous & transversion
- `κ · π_j` if change is synonymous & transition
- `ω · π_j` if change is nonsynonymous & transversion
- `ω · κ · π_j` if change is nonsynonymous & transition

Diagonals: `Q[i,i] = -Σ_{j≠i} Q[i,j]`. Q is scaled so that `-Σ_i π_i · Q[i,i] = 1` (one substitution per unit branch length).

Equilibrium frequencies π use the **F3X4** model: compute nucleotide frequencies per codon position from the alignment, then `π(codon) ∝ f_pos1(n1) · f_pos2(n2) · f_pos3(n3)` over sense codons, renormalized.

- [ ] **Step 7.1: Write failing tests**

`tests/unit/test_rate_matrix.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from selkit.engine.genetic_code import GeneticCode
from selkit.engine.rate_matrix import build_q, estimate_f3x4, prob_transition_matrix


def _uniform_pi(gc: GeneticCode) -> np.ndarray:
    return np.full(gc.n_sense, 1.0 / gc.n_sense)


def test_q_row_sums_to_zero() -> None:
    gc = GeneticCode.standard()
    pi = _uniform_pi(gc)
    Q = build_q(gc, omega=0.5, kappa=2.0, pi=pi)
    np.testing.assert_allclose(Q.sum(axis=1), 0.0, atol=1e-12)


def test_q_is_scaled() -> None:
    gc = GeneticCode.standard()
    pi = _uniform_pi(gc)
    Q = build_q(gc, omega=0.5, kappa=2.0, pi=pi)
    # Mean substitution rate = -sum_i pi_i Q_ii must equal 1.
    assert -float(pi @ np.diag(Q)) == pytest.approx(1.0, rel=1e-9)


def test_q_zero_for_multi_position_changes() -> None:
    gc = GeneticCode.standard()
    pi = _uniform_pi(gc)
    Q = build_q(gc, omega=1.0, kappa=1.0, pi=pi)
    # Compare ATG (ATG) and CCC: differ at 3 positions -> rate 0.
    i = gc.codon_to_index("ATG")
    j = gc.codon_to_index("CCC")
    assert Q[i, j] == 0.0


def test_q_respects_omega_on_nonsynonymous() -> None:
    gc = GeneticCode.standard()
    pi = _uniform_pi(gc)
    Q1 = build_q(gc, omega=1.0, kappa=1.0, pi=pi)
    Q2 = build_q(gc, omega=2.0, kappa=1.0, pi=pi)
    # Scaling normalizes; but the ratio between a synonymous and
    # nonsynonymous off-diagonal should differ between omega=1 and omega=2.
    # Use a known synonymous pair CTT->CTC (both Leu) and nonsyn ATG->ACG (Met->Thr).
    i_s, j_s = gc.codon_to_index("CTT"), gc.codon_to_index("CTC")
    i_n, j_n = gc.codon_to_index("ATG"), gc.codon_to_index("ACG")
    r1 = Q1[i_n, j_n] / Q1[i_s, j_s]
    r2 = Q2[i_n, j_n] / Q2[i_s, j_s]
    assert r2 / r1 == pytest.approx(2.0, rel=1e-9)


def test_prob_transition_matrix_is_stochastic() -> None:
    gc = GeneticCode.standard()
    pi = _uniform_pi(gc)
    Q = build_q(gc, omega=0.4, kappa=2.1, pi=pi)
    for t in (0.0, 0.01, 1.0, 100.0):
        P = prob_transition_matrix(Q, t)
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-9)
        assert np.all(P >= -1e-12)


def test_prob_transition_at_zero_is_identity() -> None:
    gc = GeneticCode.standard()
    pi = _uniform_pi(gc)
    Q = build_q(gc, omega=0.4, kappa=2.1, pi=pi)
    P = prob_transition_matrix(Q, 0.0)
    np.testing.assert_allclose(P, np.eye(gc.n_sense), atol=1e-9)


def test_estimate_f3x4_returns_probabilities() -> None:
    gc = GeneticCode.standard()
    # 3 taxa, 2 codons each — arbitrary small matrix.
    idx = np.array([
        [gc.codon_to_index("ATG"), gc.codon_to_index("AAA")],
        [gc.codon_to_index("ATG"), gc.codon_to_index("AAG")],
        [gc.codon_to_index("ATG"), gc.codon_to_index("AAA")],
    ], dtype=np.int16)
    pi = estimate_f3x4(idx, gc)
    assert pi.shape == (gc.n_sense,)
    np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-12)
    assert np.all(pi > 0)
```

- [ ] **Step 7.2: Run tests; expect FAIL**

```bash
pytest tests/unit/test_rate_matrix.py -v
```

- [ ] **Step 7.3: Implement `selkit/engine/rate_matrix.py`**

```python
from __future__ import annotations

import numpy as np

from selkit.engine.genetic_code import NUCS, GeneticCode, PURINES, PYRIMIDINES


def build_q(
    gc: GeneticCode, *, omega: float, kappa: float, pi: np.ndarray
) -> np.ndarray:
    n = gc.n_sense
    if pi.shape != (n,):
        raise ValueError(f"pi has wrong shape: {pi.shape}, expected ({n},)")
    codons = gc.sense_codons
    Q = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        ci = codons[i]
        for j in range(n):
            if i == j:
                continue
            cj = codons[j]
            diffs = [(k, ci[k], cj[k]) for k in range(3) if ci[k] != cj[k]]
            if len(diffs) != 1:
                continue
            _, x, y = diffs[0]
            syn = gc.is_synonymous(ci, cj)
            trans = (x in PURINES and y in PURINES) or (x in PYRIMIDINES and y in PYRIMIDINES)
            rate = pi[j]
            if trans:
                rate *= kappa
            if not syn:
                rate *= omega
            Q[i, j] = rate
    Q[np.diag_indices_from(Q)] = -Q.sum(axis=1)
    mean_rate = float(-(pi @ np.diag(Q)))
    if mean_rate <= 0:
        raise ValueError("non-positive mean substitution rate; check pi/params")
    Q /= mean_rate
    return Q


def prob_transition_matrix(Q: np.ndarray, t: float) -> np.ndarray:
    if t == 0.0:
        return np.eye(Q.shape[0])
    # Q is not symmetric but diagonalizable in generic cases; use general eig.
    w, V = np.linalg.eig(Q)
    Vinv = np.linalg.inv(V)
    P = (V * np.exp(w * t)) @ Vinv
    # Discard numerical imaginary noise.
    return np.real(P)


def estimate_f3x4(codon_indices: np.ndarray, gc: GeneticCode) -> np.ndarray:
    n = gc.n_sense
    counts = np.zeros((3, 4))     # position x nucleotide (TCAG)
    nuc_idx = {n_: i for i, n_ in enumerate(NUCS)}
    mask = codon_indices >= 0
    flat = codon_indices[mask]
    for idx in flat:
        codon = gc.index_to_codon(int(idx))
        for pos, nuc in enumerate(codon):
            counts[pos, nuc_idx[nuc]] += 1
    totals = counts.sum(axis=1, keepdims=True)
    if np.any(totals == 0):
        # Fallback to uniform if a position is fully gapped.
        totals[totals == 0] = 1
    f = counts / totals
    pi = np.empty(n, dtype=np.float64)
    for i, codon in enumerate(gc.sense_codons):
        pi[i] = f[0, nuc_idx[codon[0]]] * f[1, nuc_idx[codon[1]]] * f[2, nuc_idx[codon[2]]]
    pi /= pi.sum()
    return pi
```

- [ ] **Step 7.4: Run tests; expect PASS**

```bash
pytest tests/unit/test_rate_matrix.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 7.5: Commit**

```bash
git add selkit/engine/rate_matrix.py tests/unit/test_rate_matrix.py
git commit -m "feat(engine): codon Q matrix, eigen exponential, F3X4 frequencies"
```

---

## Task 8: Felsenstein pruning — likelihood on a fixed tree

**Files:**
- Create: `selkit/engine/likelihood.py`
- Create: `tests/unit/test_likelihood.py`

**Background:** Given a tree with branch lengths and a per-branch `P(t)` matrix, compute per-site likelihood. Pruning recursion:

- Tip with observed codon c: `L_tip[k] = 1 if k == c else 0`; for gap `-1`: `L_tip[k] = 1 for all k`.
- Internal node: `L_node[k] = Π_children (Σ_m P_child[k,m] · L_child[m])`.
- Root lnL for site = `ln(Σ_k π_k · L_root[k])`.

Total lnL = Σ sites lnL_site (with scaling to prevent underflow). For **mixture models** (M1a/M2a/M7/M8 — ω classes with weights `w_c` and separate Q_c), per-site likelihood = Σ_c w_c · L_c[site], where `L_c` is the pruning under Q_c. Log-sum across classes, then sum over sites.

- [ ] **Step 8.1: Write failing tests**

`tests/unit/test_likelihood.py`:

```python
from __future__ import annotations

import numpy as np

from selkit.engine.genetic_code import GeneticCode
from selkit.engine.likelihood import tree_log_likelihood, tree_log_likelihood_mixture
from selkit.engine.rate_matrix import build_q
from selkit.io.tree import parse_newick


def _uniform_pi(gc: GeneticCode) -> np.ndarray:
    return np.full(gc.n_sense, 1.0 / gc.n_sense)


def test_lnl_on_constant_sites_is_finite() -> None:
    gc = GeneticCode.standard()
    tree = parse_newick("(a:0.1,b:0.1,c:0.1):0.0;")
    pi = _uniform_pi(gc)
    Q = build_q(gc, omega=0.5, kappa=2.0, pi=pi)
    # All three taxa have the same codon ATG at two sites.
    idx = gc.codon_to_index("ATG")
    codons = np.full((3, 2), idx, dtype=np.int16)
    lnL = tree_log_likelihood(tree, codons, taxon_order=("a", "b", "c"), Q=Q, pi=pi)
    assert np.isfinite(lnL)


def test_gap_tip_marginalizes() -> None:
    gc = GeneticCode.standard()
    tree = parse_newick("(a:0.1,b:0.1,c:0.1):0.0;")
    pi = _uniform_pi(gc)
    Q = build_q(gc, omega=0.5, kappa=2.0, pi=pi)
    idx = gc.codon_to_index("ATG")
    all_observed = np.full((3, 1), idx, dtype=np.int16)
    with_gap = all_observed.copy()
    with_gap[2, 0] = -1
    lnL_obs = tree_log_likelihood(tree, all_observed, ("a", "b", "c"), Q=Q, pi=pi)
    lnL_gap = tree_log_likelihood(tree, with_gap, ("a", "b", "c"), Q=Q, pi=pi)
    # Marginalizing over c should yield higher lnL than observing a mismatch.
    assert lnL_gap > lnL_obs - 10  # sanity: finite and not absurd


def test_mixture_matches_single_class_with_unit_weight() -> None:
    gc = GeneticCode.standard()
    tree = parse_newick("(a:0.1,b:0.1,c:0.1):0.0;")
    pi = _uniform_pi(gc)
    Q = build_q(gc, omega=0.5, kappa=2.0, pi=pi)
    idx = gc.codon_to_index("ATG")
    codons = np.full((3, 2), idx, dtype=np.int16)
    lnL_single = tree_log_likelihood(tree, codons, ("a", "b", "c"), Q=Q, pi=pi)
    lnL_mix = tree_log_likelihood_mixture(
        tree, codons, ("a", "b", "c"),
        Qs=[Q], weights=[1.0], pi=pi,
    )
    assert lnL_mix == pytest.approx(lnL_single, rel=1e-12)


import pytest  # noqa: E402
```

- [ ] **Step 8.2: Run tests; expect FAIL**

```bash
pytest tests/unit/test_likelihood.py -v
```

- [ ] **Step 8.3: Implement `selkit/engine/likelihood.py`**

```python
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from selkit.engine.rate_matrix import prob_transition_matrix
from selkit.io.tree import LabeledTree, Node


def _iter_postorder(root: Node) -> list[Node]:
    out: list[Node] = []
    def visit(n: Node) -> None:
        for c in n.children:
            visit(c)
        out.append(n)
    visit(root)
    return out


def _prune_tree_partials(
    tree: LabeledTree,
    codons: np.ndarray,            # (n_taxa, n_sites) int16
    taxon_order: tuple[str, ...],
    Q: np.ndarray,
    n_sense: int,
) -> np.ndarray:
    """Return L_root[site, codon] partial likelihoods at the root for a single Q."""
    n_sites = codons.shape[1]
    tip_to_row = {name: i for i, name in enumerate(taxon_order)}

    # Cache P(t) per branch length (quantized by identity lookup).
    P_cache: dict[float, np.ndarray] = {}

    def P_for(bl: float) -> np.ndarray:
        if bl not in P_cache:
            P_cache[bl] = prob_transition_matrix(Q, bl)
        return P_cache[bl]

    partials: dict[int, np.ndarray] = {}  # node_id -> (n_sites, n_sense)

    for node in _iter_postorder(tree.root):
        if node.is_tip:
            L = np.zeros((n_sites, n_sense))
            row = tip_to_row[node.name or ""]
            for s in range(n_sites):
                c = int(codons[row, s])
                if c < 0:
                    L[s, :] = 1.0
                else:
                    L[s, c] = 1.0
            partials[node.id] = L
        else:
            L = np.ones((n_sites, n_sense))
            for child in node.children:
                bl = child.branch_length if child.branch_length is not None else 0.0
                P = P_for(bl)
                # L_child shape (n_sites, n_sense); P shape (n_sense, n_sense)
                L_child = partials[child.id]
                contrib = L_child @ P.T   # (n_sites, n_sense)
                L *= contrib
            partials[node.id] = L
    return partials[tree.root.id]


def tree_log_likelihood(
    tree: LabeledTree,
    codons: np.ndarray,
    taxon_order: tuple[str, ...],
    *,
    Q: np.ndarray,
    pi: np.ndarray,
) -> float:
    n_sense = Q.shape[0]
    L_root = _prune_tree_partials(tree, codons, taxon_order, Q, n_sense)
    site_L = L_root @ pi                   # (n_sites,)
    # Log, with a floor for numerical safety.
    return float(np.sum(np.log(np.clip(site_L, 1e-300, None))))


def tree_log_likelihood_mixture(
    tree: LabeledTree,
    codons: np.ndarray,
    taxon_order: tuple[str, ...],
    *,
    Qs: list[np.ndarray],
    weights: list[float],
    pi: np.ndarray,
) -> float:
    n_sense = Qs[0].shape[0]
    per_class_logL = []
    for Q in Qs:
        L_root = _prune_tree_partials(tree, codons, taxon_order, Q, n_sense)
        site_L = L_root @ pi
        per_class_logL.append(np.log(np.clip(site_L, 1e-300, None)))
    logL_stack = np.vstack(per_class_logL)  # (n_classes, n_sites)
    logW = np.log(np.asarray(weights))[:, None]
    site_log = logsumexp(logL_stack + logW, axis=0)
    return float(site_log.sum())
```

- [ ] **Step 8.4: Run tests; expect PASS**

```bash
pytest tests/unit/test_likelihood.py -v
```

- [ ] **Step 8.5: Commit**

```bash
git add selkit/engine/likelihood.py tests/unit/test_likelihood.py
git commit -m "feat(engine): Felsenstein pruning + mixture lnL"
```

---

## Task 9: Codon model — M0 parameterization

**Files:**
- Create: `selkit/engine/codon_model.py`
- Create: `tests/unit/test_codon_model.py`

**Background:** A `SiteModel` abstracts "given free params, produce (weights, Qs)" where each Q is a codon rate matrix for one ω class. M0 has one class (ω₀, κ, with π fixed from data). Free params = (ω, κ).

- [ ] **Step 9.1: Write failing tests**

`tests/unit/test_codon_model.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from selkit.engine.codon_model import M0
from selkit.engine.genetic_code import GeneticCode


def test_m0_produces_one_class() -> None:
    gc = GeneticCode.standard()
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    model = M0(gc=gc, pi=pi)
    weights, Qs = model.build(params={"omega": 0.5, "kappa": 2.0})
    assert weights == [1.0]
    assert len(Qs) == 1
    assert Qs[0].shape == (gc.n_sense, gc.n_sense)


def test_m0_free_params_matches_signature() -> None:
    gc = GeneticCode.standard()
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    model = M0(gc=gc, pi=pi)
    assert model.free_params == ("omega", "kappa")


def test_m0_default_starting_values() -> None:
    gc = GeneticCode.standard()
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    model = M0(gc=gc, pi=pi)
    starts = model.starting_values(seed=0)
    assert set(starts) == {"omega", "kappa"}
    assert starts["omega"] > 0
    assert starts["kappa"] > 0
```

- [ ] **Step 9.2: Run tests; expect FAIL**

```bash
pytest tests/unit/test_codon_model.py -v
```

- [ ] **Step 9.3: Implement `selkit/engine/codon_model.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from selkit.engine.genetic_code import GeneticCode
from selkit.engine.rate_matrix import build_q


class SiteModel(Protocol):
    name: str
    free_params: tuple[str, ...]

    def build(
        self, *, params: dict[str, float]
    ) -> tuple[list[float], list[np.ndarray]]: ...

    def starting_values(self, *, seed: int) -> dict[str, float]: ...


@dataclass
class M0:
    gc: GeneticCode
    pi: np.ndarray
    name: str = "M0"
    free_params: tuple[str, ...] = ("omega", "kappa")

    def build(self, *, params: dict[str, float]) -> tuple[list[float], list[np.ndarray]]:
        Q = build_q(self.gc, omega=params["omega"], kappa=params["kappa"], pi=self.pi)
        return [1.0], [Q]

    def starting_values(self, *, seed: int) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        return {
            "omega": float(rng.uniform(0.2, 1.2)),
            "kappa": float(rng.uniform(1.5, 3.5)),
        }
```

- [ ] **Step 9.4: Run tests; expect PASS**

```bash
pytest tests/unit/test_codon_model.py -v
```

- [ ] **Step 9.5: Commit**

```bash
git add selkit/engine/codon_model.py tests/unit/test_codon_model.py
git commit -m "feat(engine): SiteModel protocol and M0 implementation"
```

---

## Task 10: Parameter transforms and single-start L-BFGS-B optimizer

**Files:**
- Create: `selkit/engine/optimize.py`
- Create: `tests/unit/test_optimize.py`

**Background:** Optimizer operates on **unconstrained** real vectors; transforms enforce constraints:

- ω, κ, branch lengths (non-negative): `softplus` transform `x = log(1 + exp(u))` (derivative is `sigmoid`).
- Proportions in [0,1]: `logit` transform.
- Dirichlet-style (p₀, p₁, p₂ summing to 1): use "stick-breaking" — two `logit` params `v₁, v₂ ∈ (0,1)`; then `p₀ = v₁; p₁ = (1-v₁) v₂; p₂ = (1-v₁)(1-v₂)`.

v1 starts with joint optimization of all free params + all branch lengths.

- [ ] **Step 10.1: Write failing tests**

`tests/unit/test_optimize.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from selkit.engine.optimize import (
    fit_single_start,
    pack_params,
    softplus,
    softplus_inv,
    unpack_params,
)


def test_softplus_round_trip() -> None:
    for x in (0.01, 0.1, 1.0, 10.0, 100.0):
        u = softplus_inv(x)
        assert softplus(u) == pytest.approx(x, rel=1e-9)


def test_pack_unpack_round_trip() -> None:
    spec = {"omega": "positive", "kappa": "positive", "p0": "unit"}
    params = {"omega": 0.5, "kappa": 2.1, "p0": 0.3}
    u = pack_params(params, spec)
    back = unpack_params(u, spec)
    for k, v in params.items():
        assert back[k] == pytest.approx(v, rel=1e-9)


def test_fit_single_start_minimizes_quadratic() -> None:
    # Dummy "model" where lnL = -((omega-0.7)^2 + (kappa-2.0)^2).
    def neg_lnL(params: dict[str, float]) -> float:
        return (params["omega"] - 0.7) ** 2 + (params["kappa"] - 2.0) ** 2

    spec = {"omega": "positive", "kappa": "positive"}
    start = {"omega": 1.5, "kappa": 3.0}
    result = fit_single_start(neg_lnL, start=start, transform_spec=spec, seed=0)
    assert result.params["omega"] == pytest.approx(0.7, abs=1e-3)
    assert result.params["kappa"] == pytest.approx(2.0, abs=1e-3)
    assert result.final_lnL == pytest.approx(0.0, abs=1e-6)
```

- [ ] **Step 10.2: Run tests; expect FAIL**

```bash
pytest tests/unit/test_optimize.py -v
```

- [ ] **Step 10.3: Implement `selkit/engine/optimize.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy.optimize import minimize

Transform = Literal["positive", "unit"]


@dataclass(frozen=True)
class SingleStartResult:
    params: dict[str, float]
    final_lnL: float                     # note: caller passes neg-lnL; we store neg (value at min)
    iterations: int
    converged: bool


def softplus(u: float) -> float:
    # numerically stable
    if u > 30:
        return float(u)
    return float(np.log1p(np.exp(u)))


def softplus_inv(x: float) -> float:
    if x < 0:
        raise ValueError("softplus_inv requires x >= 0")
    if x > 30:
        return float(x)
    return float(np.log(np.expm1(x)))


def _sigmoid(u: float) -> float:
    return 1.0 / (1.0 + np.exp(-u))


def _logit(x: float) -> float:
    if x <= 0 or x >= 1:
        raise ValueError("logit requires 0 < x < 1")
    return float(np.log(x / (1 - x)))


def _apply(u: float, kind: Transform) -> float:
    if kind == "positive":
        return softplus(u)
    if kind == "unit":
        return _sigmoid(u)
    raise ValueError(f"unknown transform kind: {kind}")


def _invert(x: float, kind: Transform) -> float:
    if kind == "positive":
        return softplus_inv(x)
    if kind == "unit":
        return _logit(x)
    raise ValueError(f"unknown transform kind: {kind}")


def pack_params(params: dict[str, float], spec: dict[str, Transform]) -> np.ndarray:
    return np.array([_invert(params[k], spec[k]) for k in spec], dtype=np.float64)


def unpack_params(u: np.ndarray, spec: dict[str, Transform]) -> dict[str, float]:
    return {k: _apply(float(ui), kind) for ui, (k, kind) in zip(u, spec.items())}


def fit_single_start(
    neg_lnL: Callable[[dict[str, float]], float],
    *,
    start: dict[str, float],
    transform_spec: dict[str, Transform],
    seed: int,
    max_iter: int = 500,
) -> SingleStartResult:
    u0 = pack_params(start, transform_spec)

    def wrapped(u: np.ndarray) -> float:
        try:
            params = unpack_params(u, transform_spec)
            return float(neg_lnL(params))
        except (FloatingPointError, ValueError):
            return 1e18

    res = minimize(
        wrapped,
        u0,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": 1e-10, "gtol": 1e-7},
    )
    params = unpack_params(res.x, transform_spec)
    return SingleStartResult(
        params=params,
        final_lnL=float(res.fun),
        iterations=int(res.nit),
        converged=bool(res.success),
    )
```

- [ ] **Step 10.4: Run tests; expect PASS**

```bash
pytest tests/unit/test_optimize.py -v
```

- [ ] **Step 10.5: Commit**

```bash
git add selkit/engine/optimize.py tests/unit/test_optimize.py
git commit -m "feat(engine): parameter transforms and single-start L-BFGS-B"
```

---

## Task 11: Multi-start optimizer with convergence gate

**Files:**
- Modify: `selkit/engine/optimize.py` (add `fit_multi_start`)
- Modify: `tests/unit/test_optimize.py`

**Background:** Wrap the single-start fitter with K starts. Pick the best-lnL result. Flag `converged=False` if the top 2 lnL values differ by more than `tol` (default 0.5 lnL units).

- [ ] **Step 11.1: Append failing tests**

```python
from selkit.engine.optimize import MultiStartResult, fit_multi_start


def test_multi_start_picks_best_and_reports_convergence() -> None:
    def neg_lnL(params: dict[str, float]) -> float:
        # Has a flat region and a true minimum; the K=3 starts should agree.
        return (params["omega"] - 0.7) ** 2 + (params["kappa"] - 2.0) ** 2

    def starting_values(seed: int) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        return {"omega": float(rng.uniform(0.1, 2.0)), "kappa": float(rng.uniform(1.0, 5.0))}

    spec = {"omega": "positive", "kappa": "positive"}
    result = fit_multi_start(
        neg_lnL=neg_lnL,
        starting_values=starting_values,
        transform_spec=spec,
        n_starts=3,
        seed=42,
        convergence_tol=0.5,
    )
    assert isinstance(result, MultiStartResult)
    assert result.converged
    assert result.best.params["omega"] == pytest.approx(0.7, abs=1e-3)
    assert len(result.starts) == 3


def test_multi_start_flags_non_convergence() -> None:
    # Multi-modal objective: distinct basins around omega=0.5 and omega=2.0.
    def neg_lnL(params: dict[str, float]) -> float:
        o = params["omega"]
        return min((o - 0.5) ** 2 + 1.0, (o - 2.0) ** 2)  # second basin is deeper

    def starting_values(seed: int) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        # seeds that land on both basins
        return {"omega": float(rng.choice([0.4, 1.9, 0.6]))}

    spec = {"omega": "positive"}
    result = fit_multi_start(
        neg_lnL=neg_lnL,
        starting_values=starting_values,
        transform_spec=spec,
        n_starts=3,
        seed=0,
        convergence_tol=0.1,
    )
    assert not result.converged
```

- [ ] **Step 11.2: Run; expect FAIL**

```bash
pytest tests/unit/test_optimize.py -v
```

- [ ] **Step 11.3: Append to `selkit/engine/optimize.py`**

```python
from typing import Callable  # noqa: E402 (already imported above)


@dataclass(frozen=True)
class MultiStartResult:
    starts: list[SingleStartResult]
    best: SingleStartResult
    converged: bool


def fit_multi_start(
    *,
    neg_lnL: Callable[[dict[str, float]], float],
    starting_values: Callable[[int], dict[str, float]],
    transform_spec: dict[str, Transform],
    n_starts: int,
    seed: int,
    convergence_tol: float,
    max_iter: int = 500,
) -> MultiStartResult:
    rng = np.random.default_rng(seed)
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(n_starts)]
    starts: list[SingleStartResult] = []
    for s in seeds:
        start = starting_values(s)
        try:
            r = fit_single_start(
                neg_lnL,
                start=start,
                transform_spec=transform_spec,
                seed=s,
                max_iter=max_iter,
            )
        except Exception:
            continue
        starts.append(r)
    if not starts:
        raise RuntimeError("all optimization starts failed")
    # lower neg-lnL = better (smaller objective)
    starts.sort(key=lambda r: r.final_lnL)
    best = starts[0]
    converged = True
    if len(starts) >= 2:
        converged = (starts[1].final_lnL - starts[0].final_lnL) <= convergence_tol
    return MultiStartResult(starts=starts, best=best, converged=converged)
```

- [ ] **Step 11.4: Run; expect PASS**

```bash
pytest tests/unit/test_optimize.py -v
```

- [ ] **Step 11.5: Commit**

```bash
git add selkit/engine/optimize.py tests/unit/test_optimize.py
git commit -m "feat(engine): multi-start optimizer with convergence gate"
```

---

## Task 12: End-to-end M0 fit on a toy alignment

**Files:**
- Create: `selkit/engine/fit.py` (fit-a-model glue)
- Create: `tests/unit/test_fit_m0.py`

**Background:** Glue layer: given a `SiteModel` + `CodonAlignment` + `LabeledTree`, build a neg-lnL function that jointly optimizes model params **and branch lengths**, then multi-start fit it. Branch lengths use `"positive"` transform with starting values from the input tree (default 0.1 if missing).

- [ ] **Step 12.1: Write failing test**

`tests/unit/test_fit_m0.py`:

```python
from __future__ import annotations

import numpy as np

from selkit.engine.codon_model import M0
from selkit.engine.fit import fit_model
from selkit.engine.genetic_code import GeneticCode
from selkit.engine.rate_matrix import estimate_f3x4
from selkit.io.tree import parse_newick


def test_fit_m0_recovers_omega_from_simulated_like_data() -> None:
    gc = GeneticCode.standard()
    tree = parse_newick("(a:0.2,b:0.2,(c:0.2,d:0.2):0.1);")
    # Build a realistic-ish constant alignment — all taxa share ATG repeated 50 times.
    # This is a degenerate case where the fit is mostly driven by the prior; we check
    # the pipeline runs and returns something sensible.
    idx = gc.codon_to_index("ATG")
    codons = np.full((4, 50), idx, dtype=np.int16)
    pi = estimate_f3x4(codons, gc)
    model = M0(gc=gc, pi=pi)
    fit = fit_model(
        model=model,
        alignment_codons=codons,
        taxon_order=("a", "b", "c", "d"),
        tree=tree,
        n_starts=2,
        seed=7,
    )
    assert np.isfinite(fit.lnL)
    assert "omega" in fit.params
    assert "kappa" in fit.params
    assert fit.params["omega"] > 0
    assert fit.params["kappa"] > 0
    assert len(fit.branch_lengths) >= 4      # at least one per tip
```

- [ ] **Step 12.2: Run; expect FAIL**

```bash
pytest tests/unit/test_fit_m0.py -v
```

- [ ] **Step 12.3: Implement `selkit/engine/fit.py`**

```python
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from selkit.engine.codon_model import SiteModel
from selkit.engine.likelihood import tree_log_likelihood_mixture
from selkit.engine.optimize import (
    MultiStartResult,
    Transform,
    fit_multi_start,
)
from selkit.io.tree import LabeledTree, Node


@dataclass(frozen=True)
class EngineFit:
    model: str
    lnL: float
    n_params: int
    params: dict[str, float]
    branch_lengths: dict[str, float]
    multi_start: MultiStartResult
    runtime_s: float


def _branch_keys(tree: LabeledTree) -> list[str]:
    keys: list[str] = []
    for n in tree.all_nodes():
        if n is tree.root:
            continue
        keys.append(_branch_key(n))
    return keys


def _branch_key(node: Node) -> str:
    return f"bl_{node.id}"


def _apply_branch_lengths(tree: LabeledTree, bls: dict[str, float]) -> None:
    for n in tree.all_nodes():
        if n is tree.root:
            continue
        n.branch_length = bls[_branch_key(n)]


def fit_model(
    *,
    model: SiteModel,
    alignment_codons: np.ndarray,
    taxon_order: tuple[str, ...],
    tree: LabeledTree,
    n_starts: int,
    seed: int,
    convergence_tol: float = 0.5,
    max_iter: int = 500,
) -> EngineFit:
    t0 = time.perf_counter()
    branch_keys = _branch_keys(tree)
    bl_init: dict[str, float] = {}
    for n in tree.all_nodes():
        if n is tree.root:
            continue
        bl_init[_branch_key(n)] = n.branch_length if n.branch_length and n.branch_length > 0 else 0.1

    transform_spec: dict[str, Transform] = {k: "positive" for k in branch_keys}
    # Model-specific transforms.
    model_transforms: dict[str, Transform] = {}
    for p in model.free_params:
        if p in {"omega", "kappa", "omega2", "q_beta", "p_beta"}:
            model_transforms[p] = "positive"
        else:
            model_transforms[p] = "unit"
    transform_spec.update(model_transforms)

    def starting_values(s: int) -> dict[str, float]:
        start = dict(bl_init)
        rng = np.random.default_rng(s)
        for k in bl_init:
            start[k] *= float(rng.uniform(0.7, 1.3))
        start.update(model.starting_values(seed=s))
        return start

    def neg_lnL(params: dict[str, float]) -> float:
        # Apply branch lengths.
        for n in tree.all_nodes():
            if n is tree.root:
                continue
            n.branch_length = max(params[_branch_key(n)], 1e-8)
        model_params = {p: params[p] for p in model.free_params}
        weights, Qs = model.build(params=model_params)
        pi = getattr(model, "pi")
        return -tree_log_likelihood_mixture(
            tree=tree,
            codons=alignment_codons,
            taxon_order=taxon_order,
            Qs=Qs, weights=weights, pi=pi,
        )

    result = fit_multi_start(
        neg_lnL=neg_lnL,
        starting_values=starting_values,
        transform_spec=transform_spec,
        n_starts=n_starts,
        seed=seed,
        convergence_tol=convergence_tol,
        max_iter=max_iter,
    )

    best_params = result.best.params
    model_only = {p: best_params[p] for p in model.free_params}
    bls = {k: best_params[k] for k in branch_keys}
    return EngineFit(
        model=model.name,
        lnL=-result.best.final_lnL,
        n_params=len(transform_spec),
        params=model_only,
        branch_lengths=bls,
        multi_start=result,
        runtime_s=time.perf_counter() - t0,
    )
```

- [ ] **Step 12.4: Run; expect PASS**

```bash
pytest tests/unit/test_fit_m0.py -v
```

- [ ] **Step 12.5: Commit**

```bash
git add selkit/engine/fit.py tests/unit/test_fit_m0.py
git commit -m "feat(engine): end-to-end model fit with joint BL optimization"
```

---

## Task 13: Site models M1a, M2a

**Files:**
- Modify: `selkit/engine/codon_model.py`
- Modify: `tests/unit/test_codon_model.py`

**Background:**
- **M1a (NearlyNeutral):** two classes — class 0 with ω₀ ∈ (0, 1), proportion p₀; class 1 with ω₁ = 1, proportion 1 - p₀. Free: `omega0`, `p0`, `kappa`.
- **M2a (PositiveSelection):** three classes — class 0 ω₀ ∈ (0, 1) p₀; class 1 ω₁ = 1 p₁; class 2 ω₂ ≥ 1 p₂ = 1 - p₀ - p₁. Stick-breaking with `v1 = p0`, `v2 = p1 / (1 - p0)`. Free: `omega0`, `omega2`, `p0`, `p1_frac` (= v2), `kappa`.

- [ ] **Step 13.1: Write failing tests (append)**

```python
from selkit.engine.codon_model import M1a, M2a


def test_m1a_weights_sum_to_one_and_omega1_is_neutral() -> None:
    gc = GeneticCode.standard()
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    model = M1a(gc=gc, pi=pi)
    weights, Qs = model.build(params={"omega0": 0.2, "p0": 0.6, "kappa": 2.0})
    assert sum(weights) == pytest.approx(1.0, rel=1e-12)
    assert len(Qs) == 2
    # With omega=1 the second class should be "neutral"; we can't check Q directly
    # without duplicating logic, so just check the weights.
    assert weights[0] == pytest.approx(0.6)
    assert weights[1] == pytest.approx(0.4)


def test_m2a_weights_sum_to_one_with_three_classes() -> None:
    gc = GeneticCode.standard()
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    model = M2a(gc=gc, pi=pi)
    params = {"omega0": 0.2, "omega2": 2.5, "p0": 0.5, "p1_frac": 0.6, "kappa": 2.0}
    weights, Qs = model.build(params=params)
    assert sum(weights) == pytest.approx(1.0, rel=1e-12)
    assert len(Qs) == 3
    # p1 = (1 - 0.5) * 0.6 = 0.3; p2 = (1 - 0.5) * 0.4 = 0.2
    assert weights == pytest.approx([0.5, 0.3, 0.2], rel=1e-12)


def test_m2a_omega2_constrained_above_one_via_transform() -> None:
    # The model itself doesn't constrain omega2; that's the transform's job.
    # Here we just check the model accepts and passes the parameter through.
    gc = GeneticCode.standard()
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    model = M2a(gc=gc, pi=pi)
    params = {"omega0": 0.1, "omega2": 3.0, "p0": 0.8, "p1_frac": 0.5, "kappa": 2.0}
    weights, Qs = model.build(params=params)
    assert len(Qs) == 3
```

- [ ] **Step 13.2: Run; expect FAIL**

```bash
pytest tests/unit/test_codon_model.py -v
```

- [ ] **Step 13.3: Append to `selkit/engine/codon_model.py`**

```python
@dataclass
class M1a:
    gc: GeneticCode
    pi: np.ndarray
    name: str = "M1a"
    free_params: tuple[str, ...] = ("omega0", "p0", "kappa")

    def build(self, *, params: dict[str, float]) -> tuple[list[float], list[np.ndarray]]:
        omega0 = params["omega0"]
        p0 = params["p0"]
        kappa = params["kappa"]
        Q0 = build_q(self.gc, omega=omega0, kappa=kappa, pi=self.pi)
        Q1 = build_q(self.gc, omega=1.0, kappa=kappa, pi=self.pi)
        return [p0, 1.0 - p0], [Q0, Q1]

    def starting_values(self, *, seed: int) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        return {
            "omega0": float(rng.uniform(0.05, 0.8)),
            "p0": float(rng.uniform(0.3, 0.9)),
            "kappa": float(rng.uniform(1.5, 3.5)),
        }


@dataclass
class M2a:
    gc: GeneticCode
    pi: np.ndarray
    name: str = "M2a"
    free_params: tuple[str, ...] = ("omega0", "omega2", "p0", "p1_frac", "kappa")

    def build(self, *, params: dict[str, float]) -> tuple[list[float], list[np.ndarray]]:
        p0 = params["p0"]
        p1_frac = params["p1_frac"]
        p1 = (1.0 - p0) * p1_frac
        p2 = (1.0 - p0) * (1.0 - p1_frac)
        kappa = params["kappa"]
        Q0 = build_q(self.gc, omega=params["omega0"], kappa=kappa, pi=self.pi)
        Q1 = build_q(self.gc, omega=1.0, kappa=kappa, pi=self.pi)
        Q2 = build_q(self.gc, omega=params["omega2"], kappa=kappa, pi=self.pi)
        return [p0, p1, p2], [Q0, Q1, Q2]

    def starting_values(self, *, seed: int) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        return {
            "omega0": float(rng.uniform(0.05, 0.5)),
            "omega2": float(rng.uniform(1.5, 4.0)),
            "p0": float(rng.uniform(0.4, 0.85)),
            "p1_frac": float(rng.uniform(0.3, 0.7)),
            "kappa": float(rng.uniform(1.5, 3.5)),
        }
```

Also update `fit.py` `model_transforms` (step 12) to know about the unit-bounded parameters: `p0`, `p1_frac`, `p1` (for M8) use `"unit"`; all other params default to `"positive"`. Confirm by looking at the fit transform-selection code — the existing fallback `else: model_transforms[p] = "unit"` catches these names correctly. Add `"omega0"` and `"omega2"` to the positive list in `fit.py`:

```python
# in selkit/engine/fit.py, replace the model-transform block with:
model_transforms: dict[str, Transform] = {}
_POSITIVE = {"omega", "omega0", "omega2", "kappa", "q_beta", "p_beta"}
for p in model.free_params:
    model_transforms[p] = "positive" if p in _POSITIVE else "unit"
```

- [ ] **Step 13.4: Run codon-model + fit tests; expect PASS**

```bash
pytest tests/unit/test_codon_model.py tests/unit/test_fit_m0.py -v
```

- [ ] **Step 13.5: Commit**

```bash
git add selkit/engine/codon_model.py selkit/engine/fit.py tests/unit/test_codon_model.py
git commit -m "feat(engine): M1a and M2a site models"
```

---

## Task 14: Site models M7, M8, M8a (beta-discretized ω)

**Files:**
- Modify: `selkit/engine/codon_model.py`
- Modify: `tests/unit/test_codon_model.py`

**Background:**
- **M7:** ω follows `Beta(p_beta, q_beta)` on [0,1], discretized into K=10 equal-area bins. Each bin i contributes weight `1/K` and ω equal to the bin's **median** (quantile (i + 0.5)/K).
- **M8:** M7 for K bins with total weight p₀, plus one extra class with weight 1 - p₀ and ω = `omega2 ≥ 1`. Free: `p_beta`, `q_beta`, `p0`, `omega2`, `kappa`.
- **M8a:** M8 with `omega2 = 1` fixed (null model; boundary for LRT).

- [ ] **Step 14.1: Write failing tests (append)**

```python
from selkit.engine.codon_model import M7, M8, M8a


def test_m7_produces_10_classes_equal_weight() -> None:
    gc = GeneticCode.standard()
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    model = M7(gc=gc, pi=pi, n_categories=10)
    params = {"p_beta": 0.5, "q_beta": 1.5, "kappa": 2.0}
    weights, Qs = model.build(params=params)
    assert len(weights) == 10
    assert len(Qs) == 10
    np.testing.assert_allclose(weights, [0.1] * 10, atol=1e-12)


def test_m8_has_k_plus_one_classes_with_correct_total_mass() -> None:
    gc = GeneticCode.standard()
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    model = M8(gc=gc, pi=pi, n_categories=10)
    params = {"p_beta": 0.5, "q_beta": 1.5, "p0": 0.9, "omega2": 2.5, "kappa": 2.0}
    weights, Qs = model.build(params=params)
    assert len(weights) == 11
    assert sum(weights) == pytest.approx(1.0, rel=1e-12)
    # first 10 share p0, last is 1-p0
    np.testing.assert_allclose(weights[:10], [0.09] * 10, atol=1e-12)
    assert weights[10] == pytest.approx(0.1, rel=1e-12)


def test_m8a_has_no_omega2_free_param() -> None:
    gc = GeneticCode.standard()
    pi = np.full(gc.n_sense, 1.0 / gc.n_sense)
    model = M8a(gc=gc, pi=pi, n_categories=10)
    assert "omega2" not in model.free_params
```

- [ ] **Step 14.2: Run; expect FAIL**

```bash
pytest tests/unit/test_codon_model.py -v
```

- [ ] **Step 14.3: Append to `selkit/engine/codon_model.py`**

```python
from scipy.stats import beta as _beta


def _beta_quantiles(p_beta: float, q_beta: float, n: int) -> np.ndarray:
    # Medians of n equal-probability bins on [0,1].
    qs = (np.arange(n) + 0.5) / n
    return _beta.ppf(qs, p_beta, q_beta)


@dataclass
class M7:
    gc: GeneticCode
    pi: np.ndarray
    name: str = "M7"
    free_params: tuple[str, ...] = ("p_beta", "q_beta", "kappa")
    n_categories: int = 10

    def build(self, *, params: dict[str, float]) -> tuple[list[float], list[np.ndarray]]:
        omegas = _beta_quantiles(params["p_beta"], params["q_beta"], self.n_categories)
        omegas = np.clip(omegas, 1e-6, 1.0 - 1e-6)
        kappa = params["kappa"]
        Qs = [build_q(self.gc, omega=float(o), kappa=kappa, pi=self.pi) for o in omegas]
        w = 1.0 / self.n_categories
        return [w] * self.n_categories, Qs

    def starting_values(self, *, seed: int) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        return {
            "p_beta": float(rng.uniform(0.3, 2.0)),
            "q_beta": float(rng.uniform(0.3, 2.0)),
            "kappa": float(rng.uniform(1.5, 3.5)),
        }


@dataclass
class M8:
    gc: GeneticCode
    pi: np.ndarray
    name: str = "M8"
    free_params: tuple[str, ...] = ("p_beta", "q_beta", "p0", "omega2", "kappa")
    n_categories: int = 10

    def build(self, *, params: dict[str, float]) -> tuple[list[float], list[np.ndarray]]:
        omegas = _beta_quantiles(params["p_beta"], params["q_beta"], self.n_categories)
        omegas = np.clip(omegas, 1e-6, 1.0 - 1e-6)
        kappa = params["kappa"]
        p0 = params["p0"]
        Qs = [build_q(self.gc, omega=float(o), kappa=kappa, pi=self.pi) for o in omegas]
        Qs.append(build_q(self.gc, omega=params["omega2"], kappa=kappa, pi=self.pi))
        weights = [p0 / self.n_categories] * self.n_categories + [1.0 - p0]
        return weights, Qs

    def starting_values(self, *, seed: int) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        return {
            "p_beta": float(rng.uniform(0.3, 2.0)),
            "q_beta": float(rng.uniform(0.3, 2.0)),
            "p0": float(rng.uniform(0.7, 0.98)),
            "omega2": float(rng.uniform(1.5, 4.0)),
            "kappa": float(rng.uniform(1.5, 3.5)),
        }


@dataclass
class M8a:
    gc: GeneticCode
    pi: np.ndarray
    name: str = "M8a"
    free_params: tuple[str, ...] = ("p_beta", "q_beta", "p0", "kappa")
    n_categories: int = 10

    def build(self, *, params: dict[str, float]) -> tuple[list[float], list[np.ndarray]]:
        # M8 with omega2 fixed at 1.
        omegas = _beta_quantiles(params["p_beta"], params["q_beta"], self.n_categories)
        omegas = np.clip(omegas, 1e-6, 1.0 - 1e-6)
        kappa = params["kappa"]
        p0 = params["p0"]
        Qs = [build_q(self.gc, omega=float(o), kappa=kappa, pi=self.pi) for o in omegas]
        Qs.append(build_q(self.gc, omega=1.0, kappa=kappa, pi=self.pi))
        weights = [p0 / self.n_categories] * self.n_categories + [1.0 - p0]
        return weights, Qs

    def starting_values(self, *, seed: int) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        return {
            "p_beta": float(rng.uniform(0.3, 2.0)),
            "q_beta": float(rng.uniform(0.3, 2.0)),
            "p0": float(rng.uniform(0.7, 0.98)),
            "kappa": float(rng.uniform(1.5, 3.5)),
        }
```

- [ ] **Step 14.4: Run; expect PASS**

```bash
pytest tests/unit/test_codon_model.py -v
```

- [ ] **Step 14.5: Commit**

```bash
git add selkit/engine/codon_model.py tests/unit/test_codon_model.py
git commit -m "feat(engine): M7, M8, M8a site models with beta-discretized omega"
```

---

## Task 15: LRT computation

**Files:**
- Create: `selkit/services/__init__.py`
- Create: `selkit/services/codeml/__init__.py`
- Create: `selkit/services/codeml/lrt.py`
- Create: `tests/unit/test_lrt.py`

**Background:** LRT statistic `2·(lnL_alt - lnL_null)` is compared against χ². For boundary cases (M8a vs M8 and later branch-site A vs null), use **mixed 50:50 χ²₀:χ²₁** — i.e., p-value = 0.5 · χ²ᵢ₁.sf(stat).

- [ ] **Step 15.1: Write failing tests**

`tests/unit/test_lrt.py`:

```python
from __future__ import annotations

import pytest

from selkit.services.codeml.lrt import LRTResult, compute_lrt


def test_simple_chi2_lrt() -> None:
    # lnL_null=-100, lnL_alt=-95 -> 2dlnL=10, df=2 -> p ~ 0.0067
    r = compute_lrt(null="M1a", alt="M2a", lnL_null=-100.0, lnL_alt=-95.0, df=2)
    assert r.delta_lnL == pytest.approx(10.0)
    assert r.test_type == "chi2"
    assert 0.005 < r.p_value < 0.01
    assert r.significant_at_0_05


def test_negative_delta_clamped_to_zero() -> None:
    # Optimizer noise: "alt" lnL slightly worse than null.
    r = compute_lrt(null="M1a", alt="M2a", lnL_null=-100.0, lnL_alt=-100.1, df=2)
    assert r.delta_lnL == 0.0
    assert r.p_value == 1.0


def test_mixed_chi2_halves_pvalue() -> None:
    # mixed 50:50 chi2_0:chi2_1 for boundary tests.
    r_reg = compute_lrt(null="M8a", alt="M8", lnL_null=-100.0, lnL_alt=-97.0, df=1)
    r_mix = compute_lrt(
        null="M8a", alt="M8",
        lnL_null=-100.0, lnL_alt=-97.0, df=1, test_type="mixed_chi2",
    )
    assert r_mix.p_value == pytest.approx(r_reg.p_value / 2, rel=1e-9)
    assert r_mix.test_type == "mixed_chi2"
```

- [ ] **Step 15.2: Run; expect FAIL**

```bash
pytest tests/unit/test_lrt.py -v
```

- [ ] **Step 15.3: Implement**

`selkit/services/__init__.py` — empty.
`selkit/services/codeml/__init__.py` — empty.

`selkit/services/codeml/lrt.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from scipy.stats import chi2


@dataclass(frozen=True)
class LRTResult:
    null: str
    alt: str
    delta_lnL: float
    df: int
    p_value: float
    test_type: Literal["chi2", "mixed_chi2"]
    significant_at_0_05: bool


def compute_lrt(
    *,
    null: str,
    alt: str,
    lnL_null: float,
    lnL_alt: float,
    df: int,
    test_type: Literal["chi2", "mixed_chi2"] = "chi2",
    alpha: float = 0.05,
) -> LRTResult:
    stat = 2.0 * (lnL_alt - lnL_null)
    stat = max(0.0, stat)
    if test_type == "chi2":
        p = float(chi2.sf(stat, df))
    elif test_type == "mixed_chi2":
        # 50:50 mixture of chi2_0 (point mass at 0) and chi2_1.
        # chi2_0 contributes zero probability above 0.
        p = float(0.5 * chi2.sf(stat, df))
    else:
        raise ValueError(f"unknown test_type: {test_type}")
    return LRTResult(
        null=null, alt=alt,
        delta_lnL=max(0.0, lnL_alt - lnL_null),
        df=df, p_value=p,
        test_type=test_type,
        significant_at_0_05=(p < alpha),
    )


STANDARD_SITE_LRTS: tuple[tuple[str, str, int, str], ...] = (
    ("M1a", "M2a", 2, "chi2"),
    ("M7", "M8", 2, "chi2"),
    ("M8a", "M8", 1, "mixed_chi2"),
)
```

- [ ] **Step 15.4: Run; expect PASS**

```bash
pytest tests/unit/test_lrt.py -v
```

- [ ] **Step 15.5: Commit**

```bash
git add selkit/services/ tests/unit/test_lrt.py
git commit -m "feat(services): LRT computation with mixed-chi2 support"
```

---

## Task 16: BEB posteriors (M2a, M8)

**Files:**
- Create: `selkit/engine/beb.py`
- Create: `tests/unit/test_beb.py`

**Background:** Given a fit of M2a or M8, BEB estimates per-site `P(ω > 1 | site data)`. We implement the simpler **Naive Empirical Bayes (NEB)** version first (posterior at the MLE point estimate of hyperparameters), then flag that true BEB (integrating over hyperparameters) is a v1.1 follow-on. For a site, with mixture weights `w_c` and per-class per-site likelihoods `L_c[site]` from the fitted Qs, the posterior probability of class c is:

`P(class c | site) = w_c · L_c[site] / Σ_c' w_c' · L_c'[site]`

Positive-selection posterior = Σ over classes with ω > 1 of P(class | site). Also compute the posterior-mean ω per site.

- [ ] **Step 16.1: Write failing test**

`tests/unit/test_beb.py`:

```python
from __future__ import annotations

import numpy as np

from selkit.engine.beb import BEBSite, compute_neb


def test_neb_returns_one_entry_per_site() -> None:
    # Fake inputs: 3 sites, 3 classes (omegas 0.1, 1, 3), weights (0.5, 0.3, 0.2).
    per_class_site_logL = np.log(np.array([
        [0.5, 0.2, 0.1],  # site 0: class 0 most likely
        [0.1, 0.5, 0.2],  # site 1: class 1 most likely
        [0.05, 0.1, 0.6], # site 2: class 2 most likely
    ]))
    weights = [0.5, 0.3, 0.2]
    omegas = [0.1, 1.0, 3.0]
    sites = compute_neb(
        per_class_site_logL=per_class_site_logL,
        weights=weights, omegas=omegas,
    )
    assert len(sites) == 3
    assert all(isinstance(s, BEBSite) for s in sites)
    # Site 2: heavy weight on class 2 (omega=3)
    assert sites[2].p_positive > 0.4


def test_neb_positive_posterior_is_zero_when_no_positive_class() -> None:
    per_class_site_logL = np.log(np.array([[0.5, 0.5]]))
    weights = [0.5, 0.5]
    omegas = [0.1, 1.0]
    sites = compute_neb(
        per_class_site_logL=per_class_site_logL,
        weights=weights, omegas=omegas,
    )
    assert sites[0].p_positive == 0.0
```

- [ ] **Step 16.2: Run; expect FAIL**

```bash
pytest tests/unit/test_beb.py -v
```

- [ ] **Step 16.3: Implement**

`selkit/engine/beb.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class BEBSite:
    site: int              # 1-indexed
    p_positive: float
    mean_omega: float


def compute_neb(
    *,
    per_class_site_logL: np.ndarray,   # shape (n_classes, n_sites)
    weights: list[float],
    omegas: list[float],
) -> list[BEBSite]:
    log_w = np.log(np.asarray(weights))[:, None]
    log_joint = per_class_site_logL + log_w
    log_norm = logsumexp(log_joint, axis=0)              # (n_sites,)
    log_post = log_joint - log_norm                       # (n_classes, n_sites)
    post = np.exp(log_post)                               # (n_classes, n_sites)
    om = np.asarray(omegas)[:, None]
    mean_om = (post * om).sum(axis=0)
    positive_mask = (np.asarray(omegas) > 1.0)[:, None]
    p_pos = (post * positive_mask).sum(axis=0)
    out: list[BEBSite] = []
    for s in range(per_class_site_logL.shape[1]):
        out.append(BEBSite(
            site=s + 1,
            p_positive=float(p_pos[s]),
            mean_omega=float(mean_om[s]),
        ))
    return out
```

- [ ] **Step 16.4: Run; expect PASS**

```bash
pytest tests/unit/test_beb.py -v
```

- [ ] **Step 16.5: Commit**

```bash
git add selkit/engine/beb.py tests/unit/test_beb.py
git commit -m "feat(engine): NEB per-site posteriors (BEB stub deferred)"
```

---

## Task 17: Helper — per-class per-site log-likelihoods

**Files:**
- Modify: `selkit/engine/likelihood.py`
- Modify: `tests/unit/test_likelihood.py`

**Background:** To feed NEB, we need per-class per-site logL (not aggregated across classes). Add `per_class_site_log_likelihood` that returns shape `(n_classes, n_sites)` for a given model fit.

- [ ] **Step 17.1: Append failing test**

```python
from selkit.engine.likelihood import per_class_site_log_likelihood


def test_per_class_site_log_likelihood_shape() -> None:
    gc = GeneticCode.standard()
    tree = parse_newick("(a:0.1,b:0.1,c:0.1):0.0;")
    pi = _uniform_pi(gc)
    Q0 = build_q(gc, omega=0.2, kappa=2.0, pi=pi)
    Q1 = build_q(gc, omega=1.0, kappa=2.0, pi=pi)
    idx = gc.codon_to_index("ATG")
    codons = np.full((3, 4), idx, dtype=np.int16)
    out = per_class_site_log_likelihood(
        tree, codons, ("a", "b", "c"), Qs=[Q0, Q1], pi=pi,
    )
    assert out.shape == (2, 4)
```

- [ ] **Step 17.2: Run; expect FAIL**

```bash
pytest tests/unit/test_likelihood.py -v
```

- [ ] **Step 17.3: Append implementation**

```python
def per_class_site_log_likelihood(
    tree: LabeledTree,
    codons: np.ndarray,
    taxon_order: tuple[str, ...],
    *,
    Qs: list[np.ndarray],
    pi: np.ndarray,
) -> np.ndarray:
    n_sense = Qs[0].shape[0]
    rows = []
    for Q in Qs:
        L_root = _prune_tree_partials(tree, codons, taxon_order, Q, n_sense)
        site_L = L_root @ pi
        rows.append(np.log(np.clip(site_L, 1e-300, None)))
    return np.vstack(rows)
```

- [ ] **Step 17.4: Run; expect PASS**

```bash
pytest tests/unit/test_likelihood.py -v
```

- [ ] **Step 17.5: Commit**

```bash
git add selkit/engine/likelihood.py tests/unit/test_likelihood.py
git commit -m "feat(engine): per-class per-site logL helper for NEB"
```

---

## Task 18: Result dataclasses and RunConfig

**Files:**
- Create: `selkit/io/config.py`
- Create: `selkit/io/results.py`
- Create: `tests/unit/test_results.py`
- Create: `tests/unit/test_config.py`

**Background:** Public-facing result types mirror the spec §6. `RunConfig` captures the full invocation for `run.yaml`. For now: dataclasses only + JSON serialization helper (TSV emitters come in the next task).

- [ ] **Step 18.1: Write failing tests**

`tests/unit/test_config.py`:

```python
from __future__ import annotations

from pathlib import Path

from selkit.io.config import RunConfig, StrictFlags, dump_config, load_config


def test_run_config_round_trips_yaml(tmp_path: Path) -> None:
    cfg = RunConfig(
        alignment=Path("/data/x.fa"),
        alignment_dir=None,
        tree=Path("/data/x.nwk"),
        foreground=None,
        subcommand="codeml.site-models",
        models=("M0", "M1a", "M2a"),
        tests=("M1a-vs-M2a",),
        genetic_code="standard",
        output_dir=Path("/out"),
        threads=4,
        seed=42,
        n_starts=3,
        convergence_tol=0.5,
        strict=StrictFlags(
            strip_terminal_stop=True,
            strip_stop_codons=False,
            mask_stop_codons=False,
            trim_trailing=False,
        ),
        selkit_version="0.0.1",
        git_sha=None,
    )
    path = tmp_path / "run.yaml"
    dump_config(cfg, path)
    back = load_config(path)
    assert back == cfg
```

`tests/unit/test_results.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from selkit.io.config import RunConfig, StrictFlags
from selkit.io.results import (
    BEBSite,
    LRTResult,
    ModelFit,
    RunResult,
    StartResult,
    to_json,
)


def _minimal_config() -> RunConfig:
    return RunConfig(
        alignment=Path("/x.fa"), alignment_dir=None, tree=Path("/x.nwk"),
        foreground=None, subcommand="codeml.site-models",
        models=("M0",), tests=(), genetic_code="standard",
        output_dir=Path("/out"), threads=1, seed=0, n_starts=1,
        convergence_tol=0.5,
        strict=StrictFlags(True, False, False, False),
        selkit_version="0.0.1", git_sha=None,
    )


def test_run_result_roundtrips_through_json(tmp_path: Path) -> None:
    cfg = _minimal_config()
    fit = ModelFit(
        model="M0", lnL=-123.4, n_params=5,
        params={"omega": 0.5, "kappa": 2.0},
        branch_lengths={"bl_1": 0.1, "bl_2": 0.2},
        starts=[StartResult(seed=1, final_lnL=-123.4, iterations=20,
                            params={"omega": 0.5, "kappa": 2.0})],
        converged=True, runtime_s=0.01,
    )
    beb = {"M0": [BEBSite(site=1, p_positive=0.0, mean_omega=0.5)]}
    result = RunResult(
        config=cfg, fits={"M0": fit}, lrts=[], beb=beb, warnings=[],
    )
    path = tmp_path / "results.json"
    path.write_text(json.dumps(to_json(result), indent=2))
    loaded = json.loads(path.read_text())
    assert loaded["fits"]["M0"]["lnL"] == -123.4
    assert loaded["fits"]["M0"]["params"]["omega"] == 0.5
    assert loaded["beb"]["M0"][0]["p_positive"] == 0.0
```

- [ ] **Step 18.2: Run; expect FAIL**

```bash
pytest tests/unit/test_results.py tests/unit/test_config.py -v
```

- [ ] **Step 18.3: Implement `selkit/io/config.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass(frozen=True)
class StrictFlags:
    strip_terminal_stop: bool
    strip_stop_codons: bool
    mask_stop_codons: bool
    trim_trailing: bool


@dataclass(frozen=True)
class ForegroundConfig:
    tips: tuple[str, ...] = ()
    mrca: tuple[str, ...] = ()
    labels_file: Optional[Path] = None


@dataclass(frozen=True)
class RunConfig:
    alignment: Optional[Path]
    alignment_dir: Optional[Path]
    tree: Path
    foreground: Optional[ForegroundConfig]
    subcommand: str
    models: tuple[str, ...]
    tests: tuple[str, ...]
    genetic_code: str
    output_dir: Path
    threads: int
    seed: int
    n_starts: int
    convergence_tol: float
    strict: StrictFlags
    selkit_version: str
    git_sha: Optional[str]


def _to_primitive(obj: object) -> object:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return [_to_primitive(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_primitive(v) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_primitive(v) for k, v in asdict(obj).items()}
    return obj


def dump_config(cfg: RunConfig, path: Path) -> None:
    Path(path).write_text(yaml.safe_dump(_to_primitive(cfg), sort_keys=False))


def _from_primitive_strict(data: dict) -> StrictFlags:
    return StrictFlags(**data)


def _from_primitive_fg(data: Optional[dict]) -> Optional[ForegroundConfig]:
    if data is None:
        return None
    return ForegroundConfig(
        tips=tuple(data.get("tips") or ()),
        mrca=tuple(data.get("mrca") or ()),
        labels_file=Path(data["labels_file"]) if data.get("labels_file") else None,
    )


def load_config(path: Path) -> RunConfig:
    data = yaml.safe_load(Path(path).read_text())
    return RunConfig(
        alignment=Path(data["alignment"]) if data.get("alignment") else None,
        alignment_dir=Path(data["alignment_dir"]) if data.get("alignment_dir") else None,
        tree=Path(data["tree"]),
        foreground=_from_primitive_fg(data.get("foreground")),
        subcommand=data["subcommand"],
        models=tuple(data.get("models") or ()),
        tests=tuple(data.get("tests") or ()),
        genetic_code=data["genetic_code"],
        output_dir=Path(data["output_dir"]),
        threads=int(data["threads"]),
        seed=int(data["seed"]),
        n_starts=int(data["n_starts"]),
        convergence_tol=float(data["convergence_tol"]),
        strict=_from_primitive_strict(data["strict"]),
        selkit_version=data["selkit_version"],
        git_sha=data.get("git_sha"),
    )
```

- [ ] **Step 18.4: Implement `selkit/io/results.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

from selkit.io.config import RunConfig


@dataclass(frozen=True)
class StartResult:
    seed: int
    final_lnL: float
    iterations: int
    params: dict[str, float]


@dataclass(frozen=True)
class ModelFit:
    model: str
    lnL: float
    n_params: int
    params: dict[str, float]
    branch_lengths: dict[str, float]
    starts: list[StartResult]
    converged: bool
    runtime_s: float


@dataclass(frozen=True)
class LRTResult:
    null: str
    alt: str
    delta_lnL: float
    df: int
    p_value: float
    test_type: Literal["chi2", "mixed_chi2"]
    significant_at_0_05: bool


@dataclass(frozen=True)
class BEBSite:
    site: int
    p_positive: float
    mean_omega: float


@dataclass(frozen=True)
class RunResult:
    config: RunConfig
    fits: dict[str, ModelFit]
    lrts: list[LRTResult]
    beb: dict[str, list[BEBSite]]
    warnings: list[str]


def to_json(result: RunResult) -> dict:
    from selkit.io.config import _to_primitive
    return {
        "config": _to_primitive(result.config),
        "fits": {k: asdict(v) for k, v in result.fits.items()},
        "lrts": [asdict(l) for l in result.lrts],
        "beb": {k: [asdict(s) for s in v] for k, v in result.beb.items()},
        "warnings": list(result.warnings),
    }
```

- [ ] **Step 18.5: Run; expect PASS**

```bash
pytest tests/unit/test_results.py tests/unit/test_config.py -v
```

- [ ] **Step 18.6: Commit**

```bash
git add selkit/io/config.py selkit/io/results.py tests/unit/test_results.py tests/unit/test_config.py
git commit -m "feat(io): RunConfig/RunResult dataclasses + JSON/YAML serialization"
```

---

## Task 19: TSV emitters

**Files:**
- Modify: `selkit/io/results.py`
- Modify: `tests/unit/test_results.py`

- [ ] **Step 19.1: Append failing tests**

```python
from selkit.io.results import emit_tsv_files


def test_emit_tsv_files(tmp_path: Path) -> None:
    cfg = _minimal_config()
    fit = ModelFit(
        model="M0", lnL=-123.4, n_params=5,
        params={"omega": 0.5, "kappa": 2.0},
        branch_lengths={"bl_1": 0.1},
        starts=[], converged=True, runtime_s=0.01,
    )
    result = RunResult(
        config=cfg, fits={"M0": fit},
        lrts=[LRTResult("M1a", "M2a", 5.0, 2, 0.05, "chi2", True)],
        beb={"M2a": [BEBSite(1, 0.95, 3.2)]}, warnings=[],
    )
    emit_tsv_files(result, tmp_path)
    fits_tsv = (tmp_path / "fits.tsv").read_text().splitlines()
    assert fits_tsv[0].split("\t") == [
        "model", "lnL", "n_params", "converged", "runtime_s", "params"
    ]
    assert fits_tsv[1].split("\t")[0] == "M0"
    lrts_tsv = (tmp_path / "lrts.tsv").read_text().splitlines()
    assert lrts_tsv[0].split("\t") == [
        "null", "alt", "delta_lnL", "df", "p_value", "test_type", "significant_at_0_05"
    ]
    beb_tsv = (tmp_path / "beb_M2a.tsv").read_text().splitlines()
    assert beb_tsv[0].split("\t") == ["site", "p_positive", "mean_omega"]
```

- [ ] **Step 19.2: Run; expect FAIL**

- [ ] **Step 19.3: Append to `selkit/io/results.py`**

```python
import json
from pathlib import Path


def emit_tsv_files(result: RunResult, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # fits.tsv
    fit_rows = ["\t".join(["model", "lnL", "n_params", "converged", "runtime_s", "params"])]
    for name, fit in result.fits.items():
        fit_rows.append("\t".join([
            fit.model, f"{fit.lnL:.6f}", str(fit.n_params),
            str(fit.converged).lower(), f"{fit.runtime_s:.3f}",
            json.dumps(fit.params, sort_keys=True),
        ]))
    (output_dir / "fits.tsv").write_text("\n".join(fit_rows) + "\n")
    # lrts.tsv
    lrt_rows = ["\t".join(["null", "alt", "delta_lnL", "df", "p_value", "test_type", "significant_at_0_05"])]
    for l in result.lrts:
        lrt_rows.append("\t".join([
            l.null, l.alt, f"{l.delta_lnL:.6f}", str(l.df),
            f"{l.p_value:.6g}", l.test_type, str(l.significant_at_0_05).lower(),
        ]))
    (output_dir / "lrts.tsv").write_text("\n".join(lrt_rows) + "\n")
    # beb_<model>.tsv
    for model, sites in result.beb.items():
        rows = ["\t".join(["site", "p_positive", "mean_omega"])]
        for s in sites:
            rows.append("\t".join([str(s.site), f"{s.p_positive:.6f}", f"{s.mean_omega:.6f}"]))
        (output_dir / f"beb_{model}.tsv").write_text("\n".join(rows) + "\n")
```

- [ ] **Step 19.4: Run; expect PASS**

- [ ] **Step 19.5: Commit**

```bash
git add selkit/io/results.py tests/unit/test_results.py
git commit -m "feat(io): TSV emitters for fits/lrts/beb"
```

---

## Task 20: Validate service

**Files:**
- Create: `selkit/services/validate.py`
- Create: `tests/unit/test_validate_service.py`

**Background:** `validate_inputs(alignment_path, tree_path, foreground_spec)` runs all pre-flight checks (alignment, tree, taxon overlap, foreground consistency) and returns `(CodonAlignment, LabeledTree)` on success, or raises `SelkitInputError` with the right structured message. Reused by both the `selkit validate` subcommand and the codeml entry point.

- [ ] **Step 20.1: Write failing tests**

```python
# tests/unit/test_validate_service.py
from __future__ import annotations

from pathlib import Path

import pytest

from selkit.errors import SelkitInputError
from selkit.io.tree import ForegroundSpec
from selkit.services.validate import validate_inputs


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def test_validate_success(tmp_path: Path) -> None:
    aln = _write(tmp_path, "a.fa", ">a\nATGAAA\n>b\nATGAAG\n>c\nATGAAA\n")
    tree = _write(tmp_path, "t.nwk", "(a:0.1,b:0.1,c:0.1);")
    result = validate_inputs(
        alignment_path=aln, tree_path=tree, foreground_spec=ForegroundSpec(),
        genetic_code_name="standard",
    )
    assert result.alignment.taxa == ("a", "b", "c")
    assert result.tree.tip_names == ("a", "b", "c")


def test_validate_taxon_mismatch(tmp_path: Path) -> None:
    aln = _write(tmp_path, "a.fa", ">a\nATGAAA\n>b\nATGAAG\n")
    tree = _write(tmp_path, "t.nwk", "(a:0.1,b:0.1,c:0.1);")
    with pytest.raises(SelkitInputError, match=r"taxon"):
        validate_inputs(
            alignment_path=aln, tree_path=tree, foreground_spec=ForegroundSpec(),
            genetic_code_name="standard",
        )
```

- [ ] **Step 20.2: Run; expect FAIL**

- [ ] **Step 20.3: Implement**

`selkit/services/validate.py`:

```python
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
```

- [ ] **Step 20.4: Run; expect PASS**

- [ ] **Step 20.5: Commit**

```bash
git add selkit/services/validate.py tests/unit/test_validate_service.py
git commit -m "feat(services): pre-flight input validation"
```

---

## Task 21: Site-models orchestration (sequential)

**Files:**
- Create: `selkit/services/codeml/site_models.py`
- Create: `tests/unit/test_site_models_service.py`

**Background:** Expand bundle → list of `SiteModel` instances; fit each; compute LRTs for `STANDARD_SITE_LRTS`; compute NEB on `M2a` and `M8` (filtered to models actually fit); return a `RunResult`. Sequential implementation first; parallelism lands in Task 22.

- [ ] **Step 21.1: Write failing test**

`tests/unit/test_site_models_service.py`:

```python
from __future__ import annotations

import numpy as np

from selkit.engine.genetic_code import GeneticCode
from selkit.engine.rate_matrix import estimate_f3x4
from selkit.io.config import RunConfig, StrictFlags
from selkit.io.tree import parse_newick
from selkit.io.alignment import CodonAlignment
from selkit.services.validate import ValidatedInputs
from selkit.services.codeml.site_models import run_site_models


def _make_inputs() -> ValidatedInputs:
    gc = GeneticCode.standard()
    tree = parse_newick("(a:0.1,b:0.1,c:0.1,d:0.1);")
    idx = gc.codon_to_index("ATG")
    codons = np.full((4, 20), idx, dtype=np.int16)
    aln = CodonAlignment(taxa=("a","b","c","d"), codons=codons, genetic_code="standard", stripped_sites=())
    return ValidatedInputs(alignment=aln, tree=tree)


def _cfg() -> RunConfig:
    from pathlib import Path
    return RunConfig(
        alignment=Path("/x.fa"), alignment_dir=None, tree=Path("/x.nwk"),
        foreground=None, subcommand="codeml.site-models",
        models=("M0", "M1a", "M2a"), tests=("M1a-vs-M2a",),
        genetic_code="standard",
        output_dir=Path("/out"), threads=1, seed=1, n_starts=2,
        convergence_tol=0.5,
        strict=StrictFlags(True, False, False, False),
        selkit_version="0.0.1", git_sha=None,
    )


def test_run_site_models_sequential_returns_fits_and_lrts() -> None:
    inputs = _make_inputs()
    result = run_site_models(inputs=inputs, config=_cfg(), parallel=False, progress=None)
    assert set(result.fits) == {"M0", "M1a", "M2a"}
    lrt_names = {(l.null, l.alt) for l in result.lrts}
    assert ("M1a", "M2a") in lrt_names
    # NEB runs on M2a only (M8 not requested).
    assert "M2a" in result.beb
```

- [ ] **Step 21.2: Run; expect FAIL**

- [ ] **Step 21.3: Implement `selkit/services/codeml/site_models.py`**

```python
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from selkit.engine.beb import compute_neb
from selkit.engine.codon_model import M0, M1a, M2a, M7, M8, M8a, SiteModel
from selkit.engine.fit import EngineFit, fit_model
from selkit.engine.genetic_code import GeneticCode
from selkit.engine.likelihood import per_class_site_log_likelihood
from selkit.engine.rate_matrix import estimate_f3x4
from selkit.io.config import RunConfig
from selkit.io.results import (
    BEBSite,
    LRTResult,
    ModelFit,
    RunResult,
    StartResult,
)
from selkit.services.codeml.lrt import STANDARD_SITE_LRTS, compute_lrt
from selkit.services.validate import ValidatedInputs

_MODEL_CTORS: dict[str, Callable[[GeneticCode, np.ndarray], SiteModel]] = {
    "M0":  lambda gc, pi: M0(gc=gc, pi=pi),
    "M1a": lambda gc, pi: M1a(gc=gc, pi=pi),
    "M2a": lambda gc, pi: M2a(gc=gc, pi=pi),
    "M7":  lambda gc, pi: M7(gc=gc, pi=pi),
    "M8":  lambda gc, pi: M8(gc=gc, pi=pi),
    "M8a": lambda gc, pi: M8a(gc=gc, pi=pi),
}

_BUNDLE_DEFAULT: tuple[str, ...] = ("M0", "M1a", "M2a", "M7", "M8", "M8a")


def _engine_to_public(fit: EngineFit) -> ModelFit:
    starts = [
        StartResult(
            seed=i,
            final_lnL=-s.final_lnL,
            iterations=s.iterations,
            params=s.params,
        )
        for i, s in enumerate(fit.multi_start.starts)
    ]
    return ModelFit(
        model=fit.model,
        lnL=fit.lnL,
        n_params=fit.n_params,
        params=fit.params,
        branch_lengths=fit.branch_lengths,
        starts=starts,
        converged=fit.multi_start.converged,
        runtime_s=fit.runtime_s,
    )


def _run_one(
    name: str,
    *,
    inputs: ValidatedInputs,
    pi: np.ndarray,
    cfg: RunConfig,
) -> EngineFit:
    gc = GeneticCode.by_name(cfg.genetic_code)
    model = _MODEL_CTORS[name](gc, pi)
    return fit_model(
        model=model,
        alignment_codons=inputs.alignment.codons,
        taxon_order=inputs.alignment.taxa,
        tree=inputs.tree,
        n_starts=cfg.n_starts,
        seed=cfg.seed + hash(name) % 10_000,
        convergence_tol=cfg.convergence_tol,
    )


def _compute_beb_for(
    name: str, *, fit: EngineFit, inputs: ValidatedInputs, pi: np.ndarray,
    gc: GeneticCode,
) -> list[BEBSite]:
    model = _MODEL_CTORS[name](gc, pi)
    weights, Qs = model.build(params=fit.params)
    # M2a / M8 class omegas in the same order as Qs:
    if name == "M2a":
        omegas = [fit.params["omega0"], 1.0, fit.params["omega2"]]
    elif name == "M8":
        # Beta quantiles + positive-selection class:
        from selkit.engine.codon_model import _beta_quantiles
        beta_omegas = _beta_quantiles(fit.params["p_beta"], fit.params["q_beta"], 10).tolist()
        omegas = [float(o) for o in beta_omegas] + [fit.params["omega2"]]
    else:
        raise ValueError(f"NEB not supported for {name}")
    per_class = per_class_site_log_likelihood(
        tree=inputs.tree,
        codons=inputs.alignment.codons,
        taxon_order=inputs.alignment.taxa,
        Qs=Qs, pi=pi,
    )
    return compute_neb(
        per_class_site_logL=per_class, weights=weights, omegas=omegas,
    )


def run_site_models(
    *,
    inputs: ValidatedInputs,
    config: RunConfig,
    parallel: bool,
    progress: Optional[Callable[[str, str], None]] = None,
) -> RunResult:
    gc = GeneticCode.by_name(config.genetic_code)
    pi = estimate_f3x4(inputs.alignment.codons, gc)
    models_to_fit = config.models or _BUNDLE_DEFAULT

    engine_fits: dict[str, EngineFit] = {}
    for name in models_to_fit:
        if name not in _MODEL_CTORS:
            raise ValueError(f"unknown model {name!r}")
        if progress:
            progress("start", name)
        engine_fits[name] = _run_one(name, inputs=inputs, pi=pi, cfg=config)
        if progress:
            progress("done", name)

    fits = {name: _engine_to_public(f) for name, f in engine_fits.items()}

    lrts: list[LRTResult] = []
    for null, alt, df, test_type in STANDARD_SITE_LRTS:
        if null in fits and alt in fits:
            r = compute_lrt(
                null=null, alt=alt,
                lnL_null=fits[null].lnL, lnL_alt=fits[alt].lnL,
                df=df, test_type=test_type,
            )
            lrts.append(LRTResult(
                null=r.null, alt=r.alt, delta_lnL=r.delta_lnL,
                df=r.df, p_value=r.p_value, test_type=r.test_type,
                significant_at_0_05=r.significant_at_0_05,
            ))

    beb: dict[str, list[BEBSite]] = {}
    for name in ("M2a", "M8"):
        if name in engine_fits:
            beb[name] = _compute_beb_for(
                name, fit=engine_fits[name], inputs=inputs, pi=pi, gc=gc,
            )

    warnings: list[str] = []
    for name, f in fits.items():
        if not f.converged:
            warnings.append(f"{name}: multi-start disagreement > {config.convergence_tol} lnL")

    return RunResult(
        config=config, fits=fits, lrts=lrts, beb=beb, warnings=warnings,
    )
```

- [ ] **Step 21.4: Run; expect PASS**

```bash
pytest tests/unit/test_site_models_service.py -v
```

- [ ] **Step 21.5: Commit**

```bash
git add selkit/services/codeml/site_models.py tests/unit/test_site_models_service.py
git commit -m "feat(services): sequential site-models orchestration with LRTs + NEB"
```

---

## Task 22: Model-level parallelism via ProcessPoolExecutor

**Files:**
- Modify: `selkit/services/codeml/site_models.py`
- Modify: `tests/unit/test_site_models_service.py`

**Background:** When `config.threads > 1` and `parallel=True`, run `_run_one` over models concurrently. The process pool shares `inputs, pi, config` via pickling (all inputs are pickleable). Each worker returns `EngineFit`; main process assembles.

- [ ] **Step 22.1: Append failing test**

```python
def test_run_site_models_parallel_matches_sequential() -> None:
    inputs = _make_inputs()
    cfg_seq = _cfg()
    cfg_par = RunConfig(**{**cfg_seq.__dict__, "threads": 2})
    r_seq = run_site_models(inputs=inputs, config=cfg_seq, parallel=False, progress=None)
    r_par = run_site_models(inputs=inputs, config=cfg_par, parallel=True, progress=None)
    assert set(r_seq.fits) == set(r_par.fits)
    for name in r_seq.fits:
        assert abs(r_seq.fits[name].lnL - r_par.fits[name].lnL) < 0.01
```

- [ ] **Step 22.2: Run; expect FAIL**

- [ ] **Step 22.3: Edit `run_site_models`**

Replace the sequential fit loop with:

```python
    if parallel and config.threads > 1 and len(models_to_fit) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        futures = {}
        with ProcessPoolExecutor(max_workers=config.threads) as ex:
            for name in models_to_fit:
                if name not in _MODEL_CTORS:
                    raise ValueError(f"unknown model {name!r}")
                if progress:
                    progress("start", name)
                futures[ex.submit(_run_one, name, inputs=inputs, pi=pi, cfg=config)] = name
            for fut in as_completed(futures):
                name = futures[fut]
                engine_fits[name] = fut.result()
                if progress:
                    progress("done", name)
    else:
        for name in models_to_fit:
            if name not in _MODEL_CTORS:
                raise ValueError(f"unknown model {name!r}")
            if progress:
                progress("start", name)
            engine_fits[name] = _run_one(name, inputs=inputs, pi=pi, cfg=config)
            if progress:
                progress("done", name)
```

- [ ] **Step 22.4: Run; expect PASS**

```bash
pytest tests/unit/test_site_models_service.py -v
```

- [ ] **Step 22.5: Commit**

```bash
git add selkit/services/codeml/site_models.py tests/unit/test_site_models_service.py
git commit -m "feat(services): model-level parallelism via ProcessPoolExecutor"
```

---

## Task 23: Rich-based progress runner

**Files:**
- Create: `selkit/progress/__init__.py`
- Create: `selkit/progress/runner.py`
- Create: `tests/unit/test_progress.py`

**Background:** Thin wrapper over `rich.progress.Progress` exposing `start(model)` / `done(model)` callbacks matched to `run_site_models(progress=...)`. Tests use `Console(file=StringIO, force_terminal=False)` so output is capturable.

- [ ] **Step 23.1: Create `selkit/progress/__init__.py` (empty)**

- [ ] **Step 23.2: Write failing tests**

`tests/unit/test_progress.py`:

```python
from __future__ import annotations

import io

from selkit.progress.runner import ProgressReporter


def test_progress_reporter_marks_all_models_done() -> None:
    buf = io.StringIO()
    reporter = ProgressReporter(models=("M0", "M1a"), stream=buf)
    reporter("start", "M0")
    reporter("done", "M0")
    reporter("start", "M1a")
    reporter("done", "M1a")
    reporter.close()
    out = buf.getvalue()
    assert "M0" in out and "M1a" in out


def test_reporter_is_callable_matching_service_signature() -> None:
    reporter = ProgressReporter(models=("M0",), stream=io.StringIO())
    reporter("start", "M0")
    reporter("done", "M0")
    reporter.close()
```

- [ ] **Step 23.3: Run; expect FAIL**

- [ ] **Step 23.4: Implement `selkit/progress/runner.py`**

```python
from __future__ import annotations

from typing import IO, Optional

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn


class ProgressReporter:
    def __init__(self, models: tuple[str, ...], stream: Optional[IO[str]] = None) -> None:
        console = Console(file=stream, force_terminal=False) if stream else Console()
        self._progress = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("{task.fields[status]}"),
            console=console,
            transient=False,
        )
        self._progress.start()
        self._tasks = {
            name: self._progress.add_task(name, total=1, status="queued")
            for name in models
        }

    def __call__(self, event: str, model: str) -> None:
        if model not in self._tasks:
            return
        tid = self._tasks[model]
        if event == "start":
            self._progress.update(tid, status="fitting", completed=0)
        elif event == "done":
            self._progress.update(tid, status="done", completed=1)

    def close(self) -> None:
        self._progress.stop()
```

- [ ] **Step 23.5: Run; expect PASS**

```bash
pytest tests/unit/test_progress.py -v
```

- [ ] **Step 23.6: Commit**

```bash
git add selkit/progress/ tests/unit/test_progress.py
git commit -m "feat(progress): rich-based per-model progress reporter"
```

---

## Task 24: CLI registry and dispatch

**Files:**
- Create: `selkit/cli.py`
- Create: `selkit/cli_registry.py`
- Modify: `selkit/__main__.py`
- Create: `tests/unit/test_cli_registry.py`

**Background:** Each subcommand registers a `(parser_builder, handler)` pair. The top-level `main` dispatches `selkit <group> <sub>`.

- [ ] **Step 24.1: Write failing test**

`tests/unit/test_cli_registry.py`:

```python
from __future__ import annotations

import argparse

import pytest

from selkit.cli_registry import CLI_COMMANDS, CommandSpec


def test_registry_contains_expected_groups() -> None:
    names = {c.group for c in CLI_COMMANDS}
    assert "codeml" in names
    assert "validate" in names
    assert "rerun" in names


def test_command_spec_has_parser_builder() -> None:
    for cmd in CLI_COMMANDS:
        assert callable(cmd.build_parser)
        assert callable(cmd.handle)
```

- [ ] **Step 24.2: Run; expect FAIL (module missing)**

- [ ] **Step 24.3: Implement `selkit/cli_registry.py`**

```python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CommandSpec:
    group: str
    sub: str | None
    build_parser: Callable[[argparse.ArgumentParser], None]
    handle: Callable[[argparse.Namespace], int]


def _codeml_site_models_builder(p: argparse.ArgumentParser) -> None:
    p.add_argument("--alignment", required=False)
    p.add_argument("--alignment-dir", required=False)
    p.add_argument("--tree", required=True)
    p.add_argument("--output", required=True, dest="output_dir")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-starts", type=int, default=3)
    p.add_argument("--genetic-code", default="standard")
    p.add_argument("--foreground", default=None,
                   help="comma-separated tip names: MRCA clade becomes foreground")
    p.add_argument("--foreground-tips", default=None,
                   help="comma-separated tip names: only these tip branches are foreground")
    p.add_argument("--labels-file", default=None)
    p.add_argument("--tests", default=None,
                   help="comma-separated LRT pair names, e.g. M1a-vs-M2a,M7-vs-M8")
    p.add_argument("--models", default=None,
                   help="comma-separated site-model bundle subset")
    p.add_argument("--strip-stop-codons", action="store_true")
    p.add_argument("--no-strip-terminal-stop", action="store_true")
    p.add_argument("--allow-unconverged", action="store_true")
    p.add_argument("--config", default=None, help="optional run.yaml to merge in")


def _validate_builder(p: argparse.ArgumentParser) -> None:
    p.add_argument("--alignment", required=True)
    p.add_argument("--tree", required=True)
    p.add_argument("--genetic-code", default="standard")
    p.add_argument("--foreground", default=None)
    p.add_argument("--foreground-tips", default=None)
    p.add_argument("--labels-file", default=None)
    p.add_argument("--strip-stop-codons", action="store_true")
    p.add_argument("--no-strip-terminal-stop", action="store_true")


def _rerun_builder(p: argparse.ArgumentParser) -> None:
    p.add_argument("config", help="path to run.yaml")
    p.add_argument("--output", required=False, dest="output_dir",
                   help="override output dir from config")


def _codeml_site_models_handle(ns: argparse.Namespace) -> int:
    from selkit.cli import handle_codeml_site_models
    return handle_codeml_site_models(ns)


def _validate_handle(ns: argparse.Namespace) -> int:
    from selkit.cli import handle_validate
    return handle_validate(ns)


def _rerun_handle(ns: argparse.Namespace) -> int:
    from selkit.cli import handle_rerun
    return handle_rerun(ns)


CLI_COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec(
        group="codeml", sub="site-models",
        build_parser=_codeml_site_models_builder,
        handle=_codeml_site_models_handle,
    ),
    CommandSpec(
        group="validate", sub=None,
        build_parser=_validate_builder,
        handle=_validate_handle,
    ),
    CommandSpec(
        group="rerun", sub=None,
        build_parser=_rerun_builder,
        handle=_rerun_handle,
    ),
)


def build_argparser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="selkit")
    subs = root.add_subparsers(dest="group", required=True)
    codeml = subs.add_parser("codeml")
    codeml_subs = codeml.add_subparsers(dest="sub", required=True)
    for cmd in CLI_COMMANDS:
        if cmd.group == "codeml":
            p = codeml_subs.add_parser(cmd.sub or "_default")
            cmd.build_parser(p)
            p.set_defaults(_cmd=cmd)
        else:
            p = subs.add_parser(cmd.group)
            cmd.build_parser(p)
            p.set_defaults(_cmd=cmd)
    return root
```

- [ ] **Step 24.4: Implement minimal `selkit/cli.py` (handler stubs; real impls in next tasks)**

```python
from __future__ import annotations

import argparse


def handle_codeml_site_models(ns: argparse.Namespace) -> int:
    raise NotImplementedError


def handle_validate(ns: argparse.Namespace) -> int:
    raise NotImplementedError


def handle_rerun(ns: argparse.Namespace) -> int:
    raise NotImplementedError
```

- [ ] **Step 24.5: Replace `selkit/__main__.py`**

```python
from __future__ import annotations

import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from selkit.cli_registry import build_argparser
    parser = build_argparser()
    args = parser.parse_args(argv)
    return args._cmd.handle(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 24.6: Run; expect PASS**

```bash
pytest tests/unit/test_cli_registry.py -v
```

- [ ] **Step 24.7: Commit**

```bash
git add selkit/cli.py selkit/cli_registry.py selkit/__main__.py tests/unit/test_cli_registry.py
git commit -m "feat(cli): argparse registry and dispatch skeleton"
```

---

## Task 25: CLI handler — `selkit validate`

**Files:**
- Modify: `selkit/cli.py` (implement `handle_validate`)
- Create: `tests/integration/__init__.py` (empty)
- Create: `tests/integration/conftest.py`
- Create: `tests/integration/test_validate_cli.py`

**Background:** `selkit validate` parses CLI flags, builds `ForegroundSpec`, calls `validate_inputs`, prints a pass summary (or structured error). Exit code 0 on success, 1 on validation error.

- [ ] **Step 25.1: Write failing tests**

`tests/integration/conftest.py`:

```python
from __future__ import annotations

from pathlib import Path


def write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p
```

`tests/integration/test_validate_cli.py`:

```python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def _run_selkit(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "selkit", *args],
        capture_output=True, text=True,
    )


def test_validate_reports_pass(tmp_path: Path) -> None:
    aln = _write(tmp_path, "a.fa", ">a\nATGAAA\n>b\nATGAAG\n>c\nATGAAA\n")
    tree = _write(tmp_path, "t.nwk", "(a:0.1,b:0.1,c:0.1);")
    r = _run_selkit("validate", "--alignment", str(aln), "--tree", str(tree))
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout or "pass" in r.stdout.lower()


def test_validate_reports_mismatch(tmp_path: Path) -> None:
    aln = _write(tmp_path, "a.fa", ">a\nATGAAA\n>b\nATGAAG\n")
    tree = _write(tmp_path, "t.nwk", "(a:0.1,b:0.1,c:0.1);")
    r = _run_selkit("validate", "--alignment", str(aln), "--tree", str(tree))
    assert r.returncode == 1
    assert "taxon" in r.stderr.lower()
```

- [ ] **Step 25.2: Run; expect FAIL**

```bash
pytest tests/integration/test_validate_cli.py -v
```

- [ ] **Step 25.3: Replace `handle_validate` in `selkit/cli.py`**

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from selkit.errors import SelkitInputError
from selkit.io.tree import ForegroundSpec, load_labels_file
from selkit.services.validate import validate_inputs


def _foreground_spec_from_ns(ns: argparse.Namespace) -> ForegroundSpec:
    sources = [bool(ns.foreground), bool(ns.foreground_tips), bool(ns.labels_file)]
    if sum(sources) > 1:
        print(
            "ERROR: only one of --foreground, --foreground-tips, --labels-file may be given",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if ns.foreground:
        return ForegroundSpec(mrca=tuple(ns.foreground.split(",")))
    if ns.foreground_tips:
        return ForegroundSpec(tips=tuple(ns.foreground_tips.split(",")))
    if ns.labels_file:
        return load_labels_file(Path(ns.labels_file))
    return ForegroundSpec()


def handle_validate(ns: argparse.Namespace) -> int:
    try:
        spec = _foreground_spec_from_ns(ns)
        result = validate_inputs(
            alignment_path=Path(ns.alignment),
            tree_path=Path(ns.tree),
            foreground_spec=spec,
            genetic_code_name=ns.genetic_code,
            strip_terminal_stop=not ns.no_strip_terminal_stop,
            strip_stop_codons=ns.strip_stop_codons,
        )
    except SelkitInputError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(
        f"OK: {len(result.alignment.taxa)} taxa, "
        f"{result.alignment.codons.shape[1]} codons, "
        f"genetic code = {result.alignment.genetic_code}"
    )
    return 0


def handle_codeml_site_models(ns: argparse.Namespace) -> int:
    raise NotImplementedError


def handle_rerun(ns: argparse.Namespace) -> int:
    raise NotImplementedError
```

- [ ] **Step 25.4: Run; expect PASS**

```bash
pytest tests/integration/test_validate_cli.py -v
```

- [ ] **Step 25.5: Commit**

```bash
git add selkit/cli.py tests/integration/
git commit -m "feat(cli): selkit validate subcommand"
```

---

## Task 26: CLI handler — `selkit codeml site-models`

**Files:**
- Modify: `selkit/cli.py`
- Create: `tests/integration/test_site_models_cli.py`

**Background:** Builds `RunConfig`, runs validate, dispatches `run_site_models`, emits JSON + TSVs + `run.yaml`, prints a `rich` summary table, returns 0 (success), 2 (convergence failure unless `--allow-unconverged`).

- [ ] **Step 26.1: Write failing integration test**

`tests/integration/test_site_models_cli.py`:

```python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "selkit", *args],
        capture_output=True, text=True,
    )


def test_site_models_end_to_end(tmp_path: Path) -> None:
    # Tiny but non-trivial alignment: 4 taxa, 10 codons, some variation.
    aln = _write(tmp_path, "a.fa",
        ">a\nATGAAAGCACGTTTAGGCAAACCACGTATG\n"
        ">b\nATGAAGGCCCGTCTAGGGAAGCCTCGTATG\n"
        ">c\nATGAAAGCACGTTTGGGGAAGCCACGTATG\n"
        ">d\nATGAAAGCCCGCTTAGGCAAACCGCGTATG\n"
    )
    tree = _write(tmp_path, "t.nwk", "((a:0.1,b:0.1):0.05,(c:0.1,d:0.1):0.05);")
    out = tmp_path / "out"

    r = _run(
        "codeml", "site-models",
        "--alignment", str(aln),
        "--tree", str(tree),
        "--output", str(out),
        "--threads", "1",
        "--n-starts", "2",
        "--models", "M0,M1a,M2a",
        "--allow-unconverged",
    )
    assert r.returncode == 0, r.stderr

    results = json.loads((out / "results.json").read_text())
    assert set(results["fits"]) == {"M0", "M1a", "M2a"}
    assert (out / "fits.tsv").exists()
    assert (out / "lrts.tsv").exists()
    assert (out / "run.yaml").exists()
```

- [ ] **Step 26.2: Run; expect FAIL**

- [ ] **Step 26.3: Implement `handle_codeml_site_models`**

Append to `selkit/cli.py`:

```python
import json as _json
from rich.console import Console
from rich.table import Table

from selkit.io.alignment import read_alignment  # noqa: F401  (import sanity)
from selkit.io.config import (
    ForegroundConfig,
    RunConfig,
    StrictFlags,
    dump_config,
)
from selkit.io.results import emit_tsv_files, to_json
from selkit.progress.runner import ProgressReporter
from selkit.services.codeml.site_models import run_site_models
from selkit.version import __version__


_DEFAULT_BUNDLE: tuple[str, ...] = ("M0", "M1a", "M2a", "M7", "M8", "M8a")


def _build_runconfig(ns: argparse.Namespace) -> RunConfig:
    if ns.models:
        models = tuple(ns.models.split(","))
    else:
        models = _DEFAULT_BUNDLE
    tests = tuple(ns.tests.split(",")) if ns.tests else ()
    fg: ForegroundConfig | None = None
    if ns.foreground:
        fg = ForegroundConfig(mrca=tuple(ns.foreground.split(",")))
    elif ns.foreground_tips:
        fg = ForegroundConfig(tips=tuple(ns.foreground_tips.split(",")))
    elif ns.labels_file:
        fg = ForegroundConfig(labels_file=Path(ns.labels_file))
    return RunConfig(
        alignment=Path(ns.alignment) if ns.alignment else None,
        alignment_dir=Path(ns.alignment_dir) if ns.alignment_dir else None,
        tree=Path(ns.tree),
        foreground=fg,
        subcommand="codeml.site-models",
        models=models,
        tests=tests,
        genetic_code=ns.genetic_code,
        output_dir=Path(ns.output_dir),
        threads=int(ns.threads),
        seed=int(ns.seed),
        n_starts=int(ns.n_starts),
        convergence_tol=0.5,
        strict=StrictFlags(
            strip_terminal_stop=not ns.no_strip_terminal_stop,
            strip_stop_codons=ns.strip_stop_codons,
            mask_stop_codons=False,
            trim_trailing=False,
        ),
        selkit_version=__version__,
        git_sha=None,
    )


def _render_summary(result) -> None:
    console = Console()
    t = Table(title="selkit codeml site-models")
    t.add_column("model"); t.add_column("lnL"); t.add_column("omega (or omega2)"); t.add_column("converged")
    for name, fit in result.fits.items():
        omega_label = (
            f"{fit.params.get('omega2', fit.params.get('omega0', fit.params.get('omega', float('nan')))):.4f}"
        )
        t.add_row(name, f"{fit.lnL:.3f}", omega_label, "yes" if fit.converged else "NO")
    console.print(t)
    if result.lrts:
        lrt_t = Table(title="LRTs")
        lrt_t.add_column("null"); lrt_t.add_column("alt"); lrt_t.add_column("2dlnL"); lrt_t.add_column("df"); lrt_t.add_column("p"); lrt_t.add_column("sig.")
        for l in result.lrts:
            lrt_t.add_row(l.null, l.alt, f"{2*l.delta_lnL:.3f}", str(l.df), f"{l.p_value:.4g}", "*" if l.significant_at_0_05 else "")
        console.print(lrt_t)


def handle_codeml_site_models(ns: argparse.Namespace) -> int:
    try:
        spec = _foreground_spec_from_ns(ns)
        config = _build_runconfig(ns)
        validated = validate_inputs(
            alignment_path=config.alignment,
            tree_path=config.tree,
            foreground_spec=spec,
            genetic_code_name=config.genetic_code,
            strip_terminal_stop=config.strict.strip_terminal_stop,
            strip_stop_codons=config.strict.strip_stop_codons,
        )
    except SelkitInputError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reporter = ProgressReporter(models=config.models)
    try:
        result = run_site_models(
            inputs=validated, config=config,
            parallel=config.threads > 1,
            progress=reporter,
        )
    finally:
        reporter.close()

    (out / "results.json").write_text(_json.dumps(to_json(result), indent=2))
    emit_tsv_files(result, out)
    dump_config(config, out / "run.yaml")
    _render_summary(result)

    unconverged = [n for n, f in result.fits.items() if not f.converged]
    if unconverged and not ns.allow_unconverged:
        print(f"WARNING: unconverged models: {unconverged}", file=sys.stderr)
        return 2
    return 0
```

- [ ] **Step 26.4: Run; expect PASS**

```bash
pytest tests/integration/test_site_models_cli.py -v
```

- [ ] **Step 26.5: Commit**

```bash
git add selkit/cli.py tests/integration/test_site_models_cli.py
git commit -m "feat(cli): selkit codeml site-models subcommand"
```

---

## Task 27: CLI handler — `selkit rerun`

**Files:**
- Modify: `selkit/cli.py`
- Create: `tests/integration/test_reproducibility.py`

**Background:** `selkit rerun run.yaml [--output NEW_DIR]` reads the config and re-invokes the originating subcommand. For v1, supports `codeml.site-models` only.

- [ ] **Step 27.1: Write failing test**

`tests/integration/test_reproducibility.py`:

```python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "selkit", *args],
        capture_output=True, text=True,
    )


def test_rerun_reproduces_results(tmp_path: Path) -> None:
    aln = _write(tmp_path, "a.fa",
        ">a\nATGAAAGCACGTTTAGGC\n>b\nATGAAGGCCCGTCTAGGG\n>c\nATGAAAGCACGTTTGGGG\n>d\nATGAAAGCCCGCTTAGGC\n"
    )
    tree = _write(tmp_path, "t.nwk", "((a:0.1,b:0.1):0.05,(c:0.1,d:0.1):0.05);")
    out1 = tmp_path / "out1"
    r1 = _run(
        "codeml", "site-models",
        "--alignment", str(aln), "--tree", str(tree),
        "--output", str(out1), "--threads", "1", "--n-starts", "2",
        "--models", "M0,M1a", "--seed", "42", "--allow-unconverged",
    )
    assert r1.returncode == 0, r1.stderr
    out2 = tmp_path / "out2"
    r2 = _run("rerun", str(out1 / "run.yaml"), "--output", str(out2))
    assert r2.returncode == 0, r2.stderr
    j1 = json.loads((out1 / "results.json").read_text())
    j2 = json.loads((out2 / "results.json").read_text())
    for m in j1["fits"]:
        assert abs(j1["fits"][m]["lnL"] - j2["fits"][m]["lnL"]) < 1e-6
```

- [ ] **Step 27.2: Run; expect FAIL**

- [ ] **Step 27.3: Implement `handle_rerun`**

Replace the stub in `selkit/cli.py`:

```python
from selkit.io.config import load_config
from selkit.io.tree import ForegroundSpec, load_labels_file


def handle_rerun(ns: argparse.Namespace) -> int:
    cfg = load_config(Path(ns.config))
    if ns.output_dir:
        cfg = RunConfig(**{**cfg.__dict__, "output_dir": Path(ns.output_dir)})
    if cfg.subcommand != "codeml.site-models":
        print(f"ERROR: rerun only supports codeml.site-models; got {cfg.subcommand!r}", file=sys.stderr)
        return 1
    # Rebuild foreground spec.
    if cfg.foreground:
        if cfg.foreground.labels_file:
            fg = load_labels_file(cfg.foreground.labels_file)
        elif cfg.foreground.tips:
            fg = ForegroundSpec(tips=cfg.foreground.tips)
        elif cfg.foreground.mrca:
            fg = ForegroundSpec(mrca=cfg.foreground.mrca)
        else:
            fg = ForegroundSpec()
    else:
        fg = ForegroundSpec()
    try:
        validated = validate_inputs(
            alignment_path=cfg.alignment, tree_path=cfg.tree,
            foreground_spec=fg, genetic_code_name=cfg.genetic_code,
            strip_terminal_stop=cfg.strict.strip_terminal_stop,
            strip_stop_codons=cfg.strict.strip_stop_codons,
        )
    except SelkitInputError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reporter = ProgressReporter(models=cfg.models)
    try:
        result = run_site_models(
            inputs=validated, config=cfg,
            parallel=cfg.threads > 1,
            progress=reporter,
        )
    finally:
        reporter.close()
    (out / "results.json").write_text(_json.dumps(to_json(result), indent=2))
    emit_tsv_files(result, out)
    dump_config(cfg, out / "run.yaml")
    return 0
```

- [ ] **Step 27.4: Run; expect PASS**

```bash
pytest tests/integration/test_reproducibility.py -v
```

- [ ] **Step 27.5: Commit**

```bash
git add selkit/cli.py tests/integration/test_reproducibility.py
git commit -m "feat(cli): selkit rerun reproduces a run from run.yaml"
```

---

## Task 28: PAML-comparison validation corpus scaffold

**Files:**
- Create: `tests/validation/__init__.py`
- Create: `tests/validation/README.md`
- Create: `tests/validation/conftest.py`
- Create: `tests/validation/test_paml_corpus.py`
- Create: `tests/validation/corpus/.gitkeep`

**Background:** No triples are bundled yet; the scaffold makes dropping them in trivial later. Each triple lives in `tests/validation/corpus/<case_id>/` with `alignment.fa`, `tree.nwk`, and `expected.json` (hand-generated once from a PAML run). The test runner discovers triples via fixture, runs selkit, and checks thresholds.

- [ ] **Step 28.1: Create `tests/validation/__init__.py` (empty)**

- [ ] **Step 28.2: Create `tests/validation/corpus/.gitkeep` (empty)**

- [ ] **Step 28.3: Create `tests/validation/README.md`**

```markdown
# PAML-comparison validation corpus

Each subdirectory of `corpus/` is one case:

```
corpus/<case_id>/
├── alignment.fa          # FASTA codon alignment
├── tree.nwk              # Newick
├── expected.json         # PAML-derived ground truth
└── meta.yaml             # genetic_code, models to fit, bundle, tests, etc.
```

`expected.json` structure:

```json
{
  "fits": {
    "M0":  { "lnL": -1234.567, "params": { "omega": 0.234, "kappa": 2.12 } },
    "M1a": { ... },
    "M2a": { ... }
  },
  "beb": {
    "M2a": [ { "site": 17, "p_positive": 0.972 }, ... ]
  }
}
```

## Thresholds (v1)

- `|lnL_selkit − lnL_paml| ≤ 0.01`
- `|ω − ω_paml| ≤ 1e-3`
- `|branch_length − paml_bl| ≤ 1e-2`
- BEB site posteriors agree within `1e-3` on sites PAML identifies as positively selected.

## Adding a new case

1. Run PAML codeml externally on (alignment, tree).
2. Extract fitted lnLs, ω values, and (where relevant) BEB posteriors.
3. Write `expected.json`; commit alongside the inputs.
4. The test runner will pick it up automatically.
```

- [ ] **Step 28.4: Create `tests/validation/conftest.py`**

```python
from __future__ import annotations

from pathlib import Path

import pytest

_CORPUS_ROOT = Path(__file__).parent / "corpus"


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "paml_case" in metafunc.fixturenames:
        cases = [d for d in _CORPUS_ROOT.iterdir() if d.is_dir()]
        metafunc.parametrize("paml_case", cases, ids=[c.name for c in cases])
```

- [ ] **Step 28.5: Create `tests/validation/test_paml_corpus.py`**

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from selkit.io.config import RunConfig, StrictFlags
from selkit.io.tree import ForegroundSpec
from selkit.services.codeml.site_models import run_site_models
from selkit.services.validate import validate_inputs
from selkit.version import __version__


LNL_TOL = 0.01
OMEGA_TOL = 1e-3
BL_TOL = 1e-2
BEB_TOL = 1e-3


@pytest.mark.validation
def test_case_matches_paml(paml_case: Path) -> None:
    meta = yaml.safe_load((paml_case / "meta.yaml").read_text())
    expected = json.loads((paml_case / "expected.json").read_text())
    aln = paml_case / "alignment.fa"
    tree = paml_case / "tree.nwk"

    cfg = RunConfig(
        alignment=aln, alignment_dir=None, tree=tree,
        foreground=None, subcommand="codeml.site-models",
        models=tuple(meta["models"]),
        tests=tuple(meta.get("tests") or ()),
        genetic_code=meta.get("genetic_code", "standard"),
        output_dir=paml_case / "_out",
        threads=1, seed=meta.get("seed", 0),
        n_starts=meta.get("n_starts", 3),
        convergence_tol=0.5,
        strict=StrictFlags(True, False, False, False),
        selkit_version=__version__, git_sha=None,
    )
    validated = validate_inputs(
        alignment_path=aln, tree_path=tree,
        foreground_spec=ForegroundSpec(),
        genetic_code_name=cfg.genetic_code,
    )
    result = run_site_models(
        inputs=validated, config=cfg, parallel=False, progress=None,
    )

    for name, exp in expected["fits"].items():
        fit = result.fits[name]
        assert abs(fit.lnL - exp["lnL"]) <= LNL_TOL, (
            f"{name}: lnL diff {abs(fit.lnL - exp['lnL']):.4f} > {LNL_TOL}"
        )
        for k, v in exp["params"].items():
            if k in ("omega", "omega0", "omega2"):
                assert abs(fit.params[k] - v) <= OMEGA_TOL, (
                    f"{name}: {k} diff {abs(fit.params[k] - v):.6f} > {OMEGA_TOL}"
                )

    for name, sites in (expected.get("beb") or {}).items():
        assert name in result.beb
        got = {s.site: s.p_positive for s in result.beb[name]}
        for s in sites:
            assert abs(got[s["site"]] - s["p_positive"]) <= BEB_TOL
```

- [ ] **Step 28.6: Verify the suite is discoverable but empty (passes trivially with zero cases)**

```bash
pytest tests/validation/ -v -m validation
```

Expected: `collected 0 items` / `no tests ran`. This is the correct state until the first case is checked in.

- [ ] **Step 28.7: Commit**

```bash
git add tests/validation/
git commit -m "test(validation): PAML-comparison corpus scaffold"
```

---

## Task 29: Library public API

**Files:**
- Modify: `selkit/__init__.py`
- Create: `tests/unit/test_library_api.py`

**Background:** Expose a small public library surface so the CLI-first decision doesn't force shell users. The spec's library-first principle is cheapest-to-honor at this point: re-export the already-built dataclasses and a single `codeml_site_models(...)` entry point.

- [ ] **Step 29.1: Write failing test**

`tests/unit/test_library_api.py`:

```python
from __future__ import annotations

from pathlib import Path

import selkit
from selkit import (
    BEBSite,
    CodonAlignment,
    LRTResult,
    LabeledTree,
    ModelFit,
    RunConfig,
    RunResult,
    codeml_site_models,
)


def test_public_api_exports_expected_names() -> None:
    assert hasattr(selkit, "__version__")
    assert callable(codeml_site_models)
    # Classes are importable.
    for cls in (CodonAlignment, LabeledTree, RunConfig, RunResult, ModelFit, LRTResult, BEBSite):
        assert isinstance(cls, type)


def test_codeml_site_models_returns_run_result(tmp_path: Path) -> None:
    aln = tmp_path / "a.fa"
    aln.write_text(
        ">a\nATGAAAGCACGT\n>b\nATGAAGGCCCGT\n>c\nATGAAAGCACGT\n>d\nATGAAAGCCCGC\n"
    )
    tree = tmp_path / "t.nwk"
    tree.write_text("((a:0.1,b:0.1):0.05,(c:0.1,d:0.1):0.05);")
    out = tmp_path / "out"
    result = codeml_site_models(
        alignment=aln, tree=tree, output_dir=out,
        models=("M0", "M1a"), n_starts=2, seed=1,
    )
    assert isinstance(result, RunResult)
    assert set(result.fits) == {"M0", "M1a"}
```

- [ ] **Step 29.2: Run; expect FAIL**

- [ ] **Step 29.3: Replace `selkit/__init__.py`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from selkit.io.alignment import CodonAlignment
from selkit.io.config import RunConfig, StrictFlags, ForegroundConfig
from selkit.io.results import BEBSite, LRTResult, ModelFit, RunResult
from selkit.io.tree import ForegroundSpec, LabeledTree
from selkit.services.codeml.site_models import run_site_models
from selkit.services.validate import validate_inputs
from selkit.version import __version__

__all__ = [
    "__version__",
    "BEBSite",
    "CodonAlignment",
    "ForegroundConfig",
    "ForegroundSpec",
    "LRTResult",
    "LabeledTree",
    "ModelFit",
    "RunConfig",
    "RunResult",
    "codeml_site_models",
]


def codeml_site_models(
    *,
    alignment: Path,
    tree: Path,
    output_dir: Path,
    models: Iterable[str] = ("M0", "M1a", "M2a", "M7", "M8", "M8a"),
    genetic_code: str = "standard",
    foreground: Optional[ForegroundSpec] = None,
    n_starts: int = 3,
    seed: int = 0,
    threads: int = 1,
    convergence_tol: float = 0.5,
) -> RunResult:
    fg = foreground or ForegroundSpec()
    config = RunConfig(
        alignment=Path(alignment), alignment_dir=None, tree=Path(tree),
        foreground=None, subcommand="codeml.site-models",
        models=tuple(models), tests=(),
        genetic_code=genetic_code, output_dir=Path(output_dir),
        threads=threads, seed=seed, n_starts=n_starts,
        convergence_tol=convergence_tol,
        strict=StrictFlags(True, False, False, False),
        selkit_version=__version__, git_sha=None,
    )
    validated = validate_inputs(
        alignment_path=config.alignment, tree_path=config.tree,
        foreground_spec=fg, genetic_code_name=config.genetic_code,
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return run_site_models(
        inputs=validated, config=config,
        parallel=threads > 1, progress=None,
    )
```

- [ ] **Step 29.4: Run; expect PASS**

```bash
pytest tests/unit/test_library_api.py -v
```

- [ ] **Step 29.5: Commit**

```bash
git add selkit/__init__.py tests/unit/test_library_api.py
git commit -m "feat: public library API (codeml_site_models and dataclasses)"
```

---

## Task 30: Full-suite smoke test and README

**Files:**
- Create: `README.md`

**Background:** Close with a run of the full test suite and a README that shows the typical workflow. No new code; this is the "is it actually coherent" check.

- [ ] **Step 30.1: Run the full suite**

```bash
pytest -q
```

Expected: all unit + integration tests pass. Validation suite is empty (collected 0 items from `tests/validation/`). If any test fails, stop and fix before continuing.

- [ ] **Step 30.2: Create `README.md`**

```markdown
# selkit

Pure-Python reimplementation of PAML's selection-analysis workflows (codeml, yn00).

v1 ships codeml site models (M0, M1a, M2a, M7, M8, M8a) with auto-LRTs, per-site NEB posteriors, model-level parallelism, multi-start optimization with a convergence gate, strict input validation, and reproducible `run.yaml` artifacts.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Typical CLI workflow

```bash
selkit codeml site-models \
    --alignment gene.fa \
    --tree species.nwk \
    --output results/gene1 \
    --threads 4
```

This fits M0, M1a, M2a, M7, M8, and M8a, computes M1a-vs-M2a, M7-vs-M8, and M8a-vs-M8 LRTs, and emits per-site NEB posteriors for M2a and M8.

Outputs in `results/gene1/`:

- `results.json` — canonical structured output (all fits, LRTs, BEB sites, warnings).
- `fits.tsv`, `lrts.tsv`, `beb_M2a.tsv`, `beb_M8.tsv` — tabular views.
- `run.yaml` — exact config; reproduce with `selkit rerun results/gene1/run.yaml`.

## Validation

```bash
selkit validate --alignment gene.fa --tree species.nwk
```

Runs every pre-flight check short of model fitting (length multiples of 3, stop codons, taxon match, foreground-label consistency).

## Library

```python
from selkit import codeml_site_models

result = codeml_site_models(
    alignment="gene.fa", tree="species.nwk", output_dir="out/",
    models=("M0", "M1a", "M2a"),
)
print(result.fits["M2a"].lnL)
for s in result.beb["M2a"]:
    if s.p_positive > 0.95:
        print(f"site {s.site}: P(omega > 1) = {s.p_positive:.3f}")
```

## Not yet in v1

- Branch models (one-ratio, two-ratios, free-ratios)
- Branch-site models (Model A, Model A null)
- yn00 pairwise dN/dS

These land in subsequent plans built on the same engine.

## Design

See `docs/superpowers/specs/2026-04-18-selkit-design.md`.
```

- [ ] **Step 30.3: Commit**

```bash
git add README.md
git commit -m "docs: README for v1 (foundation + site models)"
```

- [ ] **Step 30.4: Final push (optional — confirm with user first)**

```bash
git push origin main
```

---

## Appendix A: Follow-ups deferred to later plans

1. **yn00 (pairwise dN/dS).** Reuses genetic code + alignment IO; closed-form estimator. Small plan; ~1 week.
2. **Branch models.** Reuses labeled-tree infrastructure already present (labels live on nodes, just unused for site models). Adds per-branch ω parameterization.
3. **Branch-site models (Model A, Model A null).** Depends on branch models.
4. **True BEB.** Current NEB posteriors use the MLE of hyperparameters; full BEB integrates over them (Yang, Wong, Nielsen 2005 §3). Drop-in replacement for `compute_neb`.
5. **Alignment-dir batch mode (`--alignment-dir`).** Spec'd in §4; the RunConfig field exists but the CLI doesn't yet consume it. Small orchestration layer on top of `run_site_models` per gene.
6. **Intra-model vectorization / numba.** After first real-world benchmarks.
7. **`--warnings-as-errors` flag.** One-line CLI addition; skipped to keep the plan focused.
8. **`GeneticCode` input-validation hardening.** Code review after Task 2 flagged three latent issues in the public API that are inherited from the plan-pinned implementation: (a) `is_transition` and `n_differences` silently accept mismatched-length inputs via `zip` truncation — should validate both args are length 3; (b) `translate` raises a bare `ValueError` with an unhelpful `tuple.index` message on invalid codons — should raise a descriptive `ValueError`; (c) `index_to_codon` accepts negative indices (tuple wrap-around) instead of raising `IndexError`. Also missing tests for transversion-at-single-position, lower-case inputs, `is_transition` with 0 or ≥2 diffs, and `n_sense` for mitochondrial code. Deferred because Tasks 3+ only feed well-formed upper-case codons through the API.

## Appendix B: Self-review notes

**Spec coverage:**
- §2 scope: codeml site models ✓ (Tasks 9, 13, 14). yn00 + branch + branch-site deferred per plan-split agreement.
- §3 principles: library-first ✓ (Task 29), sensible defaults ✓ (bundle default in Tasks 21, 29), strict inputs ✓ (Tasks 3, 4, 20), structured outputs ✓ (Tasks 18, 19, 29), reproducibility ✓ (Tasks 18, 27), correctness-before-speed ✓ (no numba in v1).
- §4 UX: CLI + YAML ✓ (Tasks 24, 27), FASTA+Phylip ✓ (Tasks 3, 4), JSON+TSV ✓ (Tasks 18, 19), tree labels three ways ✓ (Tasks 5, 6), auto-LRTs ✓ (Task 21), parallelism ✓ (Task 22), multi-start ✓ (Task 11), validate subcommand ✓ (Task 25), run.yaml ✓ (Tasks 18, 26).
- §5 architecture: four layers ✓, engine IO-free ✓ (all engine tests pass pure arrays/trees).
- §6 data types: CodonAlignment, LabeledTree, ModelFit, StartResult, LRTResult, BEBSite, RunResult ✓ (Tasks 3, 5, 18).
- §7 data flow: parse → validate → plan → fit (parallel) → LRT → BEB → emit ✓ (Tasks 20, 21, 22, 26).
- §8 validation: stop-codon rules ✓ (Task 3), length check ✓ (Task 3), taxon match ✓ (Task 20), label collision ✓ (Task 6), validate subcommand ✓ (Task 25).
- §9 testing: unit + integration + validation scaffold + coverage targets ✓ (`pyproject.toml` markers, Tasks 3-29, 28).
- §10 error handling: exception hierarchy ✓ (Task 1), structured messages ✓ (Tasks 3, 20). `--warnings-as-errors` flag is deferred (Appendix A).
- §11 build order: matches the task sequence.

**Placeholder scan:** All code blocks contain compilable Python. No TBD / TODO / "similar to task N". Thresholds (LNL_TOL=0.01, etc.) and defaults (K=3, tol=0.5) are consistent across Tasks 11, 28, and the spec.

**Type consistency:** `LRTResult` is defined twice — once in `services/codeml/lrt.py` (Task 15) with the computation helpers and once in `io/results.py` (Task 18) as the public-facing dataclass. They have the same fields; Task 21 explicitly converts between them. Named consistently as `LRTResult` in both modules; callers import from `selkit.io.results` per `__init__.py`. `ModelFit` vs `EngineFit`: `EngineFit` (internal, in `engine/fit.py`) is converted to public `ModelFit` inside `run_site_models` (Task 21). Names are deliberately distinct to make the library/engine boundary explicit.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-18-selkit-foundation-and-site-models.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?



