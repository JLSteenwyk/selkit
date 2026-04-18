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

    for taxon, seq in records:
        if len(seq) % 3 != 0:
            raise SelkitInputError(
                f"sequence length is not a multiple of 3 in {path}\n"
                f"  taxon:     {taxon}\n"
                f"  length:    {len(seq)} nt (remainder {len(seq) % 3})\n"
                f"  hint:      rerun with --trim-trailing if overhang is uniform"
            )

    lengths = {len(r[1]) for r in records}
    if len(lengths) > 1:
        raise SelkitInputError(
            f"sequences must all be the same length in {path} "
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
