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
