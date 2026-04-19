from __future__ import annotations

import argparse
import json as _json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from selkit.errors import SelkitInputError
from selkit.io.alignment import read_alignment  # noqa: F401
from selkit.io.config import (
    ForegroundConfig,
    RunConfig,
    StrictFlags,
    dump_config,
)
from selkit.io.results import emit_tsv_files, to_json
from selkit.io.tree import ForegroundSpec, load_labels_file
from selkit.progress.runner import ProgressReporter
from selkit.services.codeml.site_models import run_site_models
from selkit.services.validate import validate_inputs
from selkit.version import __version__


_DEFAULT_BUNDLE: tuple[str, ...] = ("M0", "M1a", "M2a", "M7", "M8", "M8a")


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


def handle_rerun(ns: argparse.Namespace) -> int:
    raise NotImplementedError
