from __future__ import annotations

from io import StringIO

from rich.console import Console


def _render(result):
    from selkit.cli import _render_summary_branch
    buf = StringIO()
    # monkey-patch Console to capture; easier: call with file=
    import selkit.cli as cli_mod
    orig = cli_mod.Console
    try:
        cli_mod.Console = lambda **kw: Console(file=buf, force_terminal=False)
        _render_summary_branch(result)
    finally:
        cli_mod.Console = orig
    return buf.getvalue()


def test_render_two_ratios_summary_format(tmp_path):
    from selkit.io.results import BranchModelFit, RunResult
    from selkit.io.config import RunConfig, StrictFlags
    from pathlib import Path
    fit = BranchModelFit(
        model="TwoRatios", family="branch",
        lnL=-100.0, n_params=3,
        params={"kappa": 2.0, "omega_bg": 0.21, "omega_fg": 1.74},
        per_branch_omega=[], branch_lengths={}, starts=[],
        converged=True, runtime_s=0.0,
    )
    cfg = RunConfig(
        alignment=Path("x"), alignment_dir=None, tree=Path("y"),
        foreground=None, subcommand="codeml.branch",
        models=("TwoRatios",), tests=(), genetic_code="standard",
        output_dir=tmp_path, threads=1, seed=0, n_starts=1,
        convergence_tol=0.5,
        strict=StrictFlags(True, False, False, False),
        selkit_version="0.3.0", git_sha=None,
    )
    result = RunResult(
        config=cfg, family="branch",
        fits={"TwoRatios": fit}, lrts=[], beb={}, warnings=[],
    )
    out = _render(result)
    assert "fg=1.740" in out and "bg=0.210" in out
