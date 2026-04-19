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
