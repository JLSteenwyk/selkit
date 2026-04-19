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
