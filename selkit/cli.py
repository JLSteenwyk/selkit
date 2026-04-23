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
    load_config,
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
        subcommand="codeml.site",
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
    t = Table(title="selkit codeml site")
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


def handle_codeml_site(ns: argparse.Namespace) -> int:
    try:
        spec = _foreground_spec_from_ns(ns)
        config = _build_runconfig(ns)
        config = RunConfig(**{**config.__dict__, "subcommand": "codeml.site"})
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


def handle_codeml_branch_site(ns: argparse.Namespace) -> int:
    from selkit.services.codeml.branch_site import run_branch_site_models
    from selkit.errors import SelkitConfigError
    try:
        spec = _foreground_spec_from_ns(ns)
        config = _build_runconfig(ns)
        config = RunConfig(**{**config.__dict__, "subcommand": "codeml.branch-site"})
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
    reporter = ProgressReporter(models=config.models or ("ModelA", "ModelA_null"))
    try:
        try:
            result = run_branch_site_models(
                inputs=validated, config=config,
                parallel=config.threads > 1, progress=reporter,
            )
        except SelkitConfigError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
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


def handle_codeml_branch(ns: argparse.Namespace) -> int:
    from selkit.services.codeml.branch_models import run_branch_models
    from selkit.errors import SelkitConfigError
    try:
        spec = _foreground_spec_from_ns(ns)
        config = _build_runconfig(ns)
        # Defaults for branch: canonical positive-selection trio.
        if not ns.models:
            config = RunConfig(**{
                **config.__dict__,
                "models": ("M0", "TwoRatios", "TwoRatiosFixed"),
                "subcommand": "codeml.branch",
            })
        else:
            config = RunConfig(**{**config.__dict__, "subcommand": "codeml.branch"})
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
        try:
            result = run_branch_models(
                inputs=validated, config=config,
                parallel=config.threads > 1, progress=reporter,
            )
        except SelkitConfigError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
    finally:
        reporter.close()

    (out / "results.json").write_text(_json.dumps(to_json(result), indent=2))
    emit_tsv_files(result, out)
    dump_config(config, out / "run.yaml")
    _render_summary_branch(result)

    unconverged = [n for n, f in result.fits.items() if not f.converged]
    if unconverged and not ns.allow_unconverged:
        print(f"WARNING: unconverged models: {unconverged}", file=sys.stderr)
        return 2
    return 0


def _render_summary_branch(result) -> None:
    console = Console()
    t = Table(title="selkit codeml branch")
    t.add_column("model"); t.add_column("lnL")
    t.add_column("omega"); t.add_column("converged")
    for name, fit in result.fits.items():
        if name == "M0":
            omega_str = f"{fit.params.get('omega', float('nan')):.4f}"
        elif name in {"TwoRatios", "TwoRatiosFixed"}:
            bg = fit.params.get("omega_bg", float("nan"))
            fg = fit.params.get("omega_fg", 1.0)
            omega_str = f"fg={fg:.3f}, bg={bg:.3f}"
        elif name == "NRatios":
            bg = fit.params.get("omega_bg", float("nan"))
            extras = ", ".join(
                f"#{i}={fit.params[f'omega_{i}']:.3f}"
                for i in range(1, 1_000_000)
                if f"omega_{i}" in fit.params
            )
            omega_str = f"bg={bg:.3f}" + (f", {extras}" if extras else "")
        elif name == "FreeRatios":
            import statistics
            omegas = [r["omega"] for r in fit.per_branch_omega]
            omega_str = (
                f"B={len(omegas)} branches, "
                f"median omega={statistics.median(omegas):.3f}, "
                f"max omega={max(omegas):.3f}"
            )
        else:
            omega_str = "?"
        t.add_row(name, f"{fit.lnL:.3f}", omega_str, "yes" if fit.converged else "NO")
    console.print(t)
    if result.lrts:
        lrt_t = Table(title="LRTs")
        lrt_t.add_column("null"); lrt_t.add_column("alt")
        lrt_t.add_column("2dlnL"); lrt_t.add_column("df")
        lrt_t.add_column("p"); lrt_t.add_column("sig.")
        lrt_t.add_column("warning")
        for l in result.lrts:
            lrt_t.add_row(
                l.null, l.alt, f"{2*l.delta_lnL:.3f}", str(l.df),
                f"{l.p_value:.4g}",
                "*" if l.significant_at_0_05 else "",
                getattr(l, "warning", None) or "",
            )
        console.print(lrt_t)


def _foreground_spec_from_cfg(cfg: RunConfig) -> ForegroundSpec:
    if cfg.foreground:
        if cfg.foreground.labels_file:
            return load_labels_file(cfg.foreground.labels_file)
        if cfg.foreground.tips:
            return ForegroundSpec(tips=cfg.foreground.tips)
        if cfg.foreground.mrca:
            return ForegroundSpec(mrca=cfg.foreground.mrca)
    return ForegroundSpec()


def _rerun_site(cfg: RunConfig) -> int:
    fg = _foreground_spec_from_cfg(cfg)
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


def _rerun_branch(cfg: RunConfig) -> int:
    from selkit.services.codeml.branch_models import run_branch_models
    from selkit.errors import SelkitConfigError

    fg = _foreground_spec_from_cfg(cfg)
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
        try:
            result = run_branch_models(
                inputs=validated, config=cfg,
                parallel=cfg.threads > 1, progress=reporter,
            )
        except SelkitConfigError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
    finally:
        reporter.close()
    (out / "results.json").write_text(_json.dumps(to_json(result), indent=2))
    emit_tsv_files(result, out)
    dump_config(cfg, out / "run.yaml")
    return 0


def _rerun_branch_site(cfg: RunConfig) -> int:
    from selkit.services.codeml.branch_site import run_branch_site_models
    from selkit.errors import SelkitConfigError

    fg = _foreground_spec_from_cfg(cfg)
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
    reporter = ProgressReporter(models=cfg.models or ("ModelA", "ModelA_null"))
    try:
        try:
            result = run_branch_site_models(
                inputs=validated, config=cfg,
                parallel=cfg.threads > 1,
                progress=reporter,
            )
        except SelkitConfigError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
    finally:
        reporter.close()
    (out / "results.json").write_text(_json.dumps(to_json(result), indent=2))
    emit_tsv_files(result, out)
    dump_config(cfg, out / "run.yaml")
    return 0


def handle_rerun(ns: argparse.Namespace) -> int:
    cfg = load_config(Path(ns.config))
    if ns.output_dir:
        cfg = RunConfig(**{**cfg.__dict__, "output_dir": Path(ns.output_dir)})
    if cfg.subcommand == "codeml.site-models":
        print(
            "ERROR: this run.yaml was produced by selkit <= 0.2.0. The "
            "codeml.site-models subcommand was split in v0.3.0. Re-create "
            "the run manually:\n"
            "  selkit codeml site        ... (for site models)\n"
            "  selkit codeml branch-site ... (for ModelA / ModelA_null)\n"
            "See CHANGELOG.md 0.3.0 for migration details.",
            file=sys.stderr,
        )
        return 1
    if cfg.subcommand == "codeml.site":
        return _rerun_site(cfg)
    if cfg.subcommand == "codeml.branch":
        return _rerun_branch(cfg)
    if cfg.subcommand == "codeml.branch-site":
        return _rerun_branch_site(cfg)
    print(f"ERROR: rerun does not support {cfg.subcommand!r}", file=sys.stderr)
    return 1
