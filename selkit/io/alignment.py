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
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(buf).upper()))
            tokens = line[1:].split()
            if not tokens:
                raise SelkitInputError(f"FASTA header missing taxon name in {path}")
            name = tokens[0]
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


def _build_from_records(
    records: list[tuple[str, str]],
    *,
    source: Path,
    genetic_code: GeneticCode,
    strip_terminal_stop: bool,
    strip_stop_codons: bool,
) -> CodonAlignment:
    if not records:
        raise SelkitInputError(f"alignment is empty: {source}")

    names = [r[0] for r in records]
    if len(set(names)) != len(names):
        raise SelkitInputError(f"duplicate taxon names in {source}")

    for taxon, seq in records:
        if len(seq) % 3 != 0:
            raise SelkitInputError(
                f"sequence length is not a multiple of 3 in {source}\n"
                f"  taxon:     {taxon}\n"
                f"  length:    {len(seq)} nt (remainder {len(seq) % 3})\n"
                f"  hint:      rerun with --trim-trailing if overhang is uniform"
            )

    lengths = {len(r[1]) for r in records}
    if len(lengths) > 1:
        raise SelkitInputError(
            f"sequences must all be the same length in {source} "
            f"(got lengths {sorted(lengths)})"
        )
    (seq_len,) = lengths

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
        if last in stops_at and len(stops_at[last]) == len(encoded):
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
                f"stop codon found mid-sequence in {source}\n"
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


def _parse_phylip(path: Path) -> list[tuple[str, str]]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()

    # Filter to non-empty lines for header detection, but keep track of position
    non_empty = [line for line in lines if line.strip()]

    if not non_empty:
        raise SelkitInputError(f"not a phylip file (bad header): {path}")

    # Skip leading blank lines; first non-empty line is the header
    header_tokens = non_empty[0].split()
    if len(header_tokens) < 2 or not header_tokens[0].isdigit() or not header_tokens[1].isdigit():
        raise SelkitInputError(f"not a phylip file (bad header): {path}")

    n_taxa = int(header_tokens[0])
    n_sites = int(header_tokens[1])

    # Lines after the header (non-empty only considered after we pick the first block)
    # We iterate through the original lines skipping the header line (first non-empty)
    # to preserve blank-line block boundaries for interleaved format.

    # Find header line index in original lines
    header_idx = next(i for i, line in enumerate(lines) if line.strip())
    remaining_lines = lines[header_idx + 1:]

    # Take the first n_taxa non-empty lines as the first block
    first_block: list[str] = []
    rest: list[str] = []
    seen_first_block = False
    for line in remaining_lines:
        if not seen_first_block:
            if line.strip():
                first_block.append(line)
                if len(first_block) == n_taxa:
                    seen_first_block = True
        else:
            rest.append(line)

    if len(first_block) < n_taxa:
        raise SelkitInputError(f"phylip file has fewer than {n_taxa} taxa: {path}")

    # Parse first block: split at most once on whitespace → (name, seq_fragment)
    taxa_names: list[str] = []
    fragments: list[list[str]] = []
    for line in first_block:
        parts = line.split(None, 1)
        name = parts[0]
        seq_fragment = parts[1].replace(" ", "") if len(parts) > 1 else ""
        taxa_names.append(name)
        fragments.append([seq_fragment])

    # Remaining non-empty lines are interleaved continuation blocks,
    # distributed round-robin to taxa j % n_taxa
    j = 0
    for line in rest:
        stripped = line.strip()
        if not stripped:
            continue
        # Continuation lines have no name prefix; strip inner spaces
        fragments[j % n_taxa].append(stripped.replace(" ", ""))
        j += 1

    # Concatenate fragments, upcase, and validate lengths
    records: list[tuple[str, str]] = []
    for i, name in enumerate(taxa_names):
        seq = "".join(fragments[i]).upper()
        if len(seq) != n_sites:
            raise SelkitInputError(
                f"phylip taxon {name!r} has length {len(seq)}, header declared {n_sites}"
            )
        records.append((name, seq))

    return records


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
    text = Path(path).read_text(encoding="utf-8-sig")
    stripped = text.lstrip()

    if stripped.startswith(">"):
        return read_fasta(
            path,
            genetic_code=genetic_code,
            strip_terminal_stop=strip_terminal_stop,
            strip_stop_codons=strip_stop_codons,
        )

    # Check if first non-empty line looks like a Phylip header
    first_line = next((line for line in stripped.splitlines() if line.strip()), "")
    tokens = first_line.split()
    if len(tokens) >= 2 and tokens[0].isdigit() and tokens[1].isdigit():
        return read_phylip(
            path,
            genetic_code=genetic_code,
            strip_terminal_stop=strip_terminal_stop,
            strip_stop_codons=strip_stop_codons,
        )

    raise SelkitInputError(f"unrecognized alignment format for {path}")
