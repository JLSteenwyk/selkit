from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from selkit.engine.codon_model import ModelA, ModelANull, SiteModel
from selkit.engine.genetic_code import GeneticCode
from selkit.errors import SelkitConfigError
from selkit.io.config import RunConfig
from selkit.io.results import RunResult
from selkit.services.codeml._orchestrator import ModelFactory, run_family
from selkit.services.validate import ValidatedInputs


def _make_modela(gc, pi):      return ModelA(gc=gc, pi=pi)
def _make_modela_null(gc, pi): return ModelANull(gc=gc, pi=pi)


MODEL_REGISTRY: dict[str, ModelFactory] = {
    "ModelA":      _make_modela,
    "ModelA_null": _make_modela_null,
}

DEFAULT_LRTS: tuple[tuple[str, str, int, str], ...] = (
    ("ModelA_null", "ModelA", 1, "mixed_chi2"),
)

# Phase 3 will set this to ("ModelA",). Phase 1 leaves it empty so the orchestrator
# does not attempt BEB for branch-site (true BEB for ModelA isn't implemented yet).
DEFAULT_BEB_MODELS: tuple[str, ...] = ()


def _require_foreground(inputs: ValidatedInputs, config: RunConfig) -> None:
    has_foreground = any(n.label != 0 for n in inputs.tree.all_nodes())
    if not has_foreground:
        raise SelkitConfigError(
            "branch-site models require a foreground clade on the tree; "
            "supply --foreground / --foreground-tips / --labels-file, or "
            "use an in-Newick #1 annotation."
        )


def run_branch_site_models(
    *,
    inputs: ValidatedInputs,
    config: RunConfig,
    parallel: bool,
    progress: Optional[Callable[[str, str], None]] = None,
) -> RunResult:
    return run_family(
        family="branch-site",
        registry=MODEL_REGISTRY,
        default_lrts=DEFAULT_LRTS,
        default_beb_models=DEFAULT_BEB_MODELS,
        inputs=inputs, config=config,
        parallel=parallel, progress=progress,
        preconditions=_require_foreground,
    )
