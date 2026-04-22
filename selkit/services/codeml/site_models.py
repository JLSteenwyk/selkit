from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from selkit.engine.codon_model import M0, M1a, M2a, M7, M8, M8a, SiteModel
from selkit.engine.genetic_code import GeneticCode
from selkit.io.config import RunConfig
from selkit.io.results import RunResult
from selkit.services.codeml._orchestrator import ModelFactory, run_family
from selkit.services.validate import ValidatedInputs


# Named factory functions (not lambdas) so MODEL_REGISTRY pickles cleanly when
# passed through ProcessPoolExecutor in the orchestrator's parallel branch.
def _make_m0(gc, pi):  return M0(gc=gc, pi=pi)
def _make_m1a(gc, pi): return M1a(gc=gc, pi=pi)
def _make_m2a(gc, pi): return M2a(gc=gc, pi=pi)
def _make_m7(gc, pi):  return M7(gc=gc, pi=pi)
def _make_m8(gc, pi):  return M8(gc=gc, pi=pi)
def _make_m8a(gc, pi): return M8a(gc=gc, pi=pi)


MODEL_REGISTRY: dict[str, ModelFactory] = {
    "M0":  _make_m0,
    "M1a": _make_m1a,
    "M2a": _make_m2a,
    "M7":  _make_m7,
    "M8":  _make_m8,
    "M8a": _make_m8a,
}

DEFAULT_LRTS: tuple[tuple[str, str, int, str], ...] = (
    ("M1a", "M2a", 2, "chi2"),
    ("M7",  "M8",  2, "chi2"),
    ("M8a", "M8",  1, "mixed_chi2"),
)

DEFAULT_BEB_MODELS: tuple[str, ...] = ("M2a", "M8")


def run_site_models(
    *,
    inputs: ValidatedInputs,
    config: RunConfig,
    parallel: bool,
    progress: Optional[Callable[[str, str], None]] = None,
) -> RunResult:
    return run_family(
        family="site",
        registry=MODEL_REGISTRY,
        default_lrts=DEFAULT_LRTS,
        default_beb_models=DEFAULT_BEB_MODELS,
        inputs=inputs, config=config,
        parallel=parallel, progress=progress,
    )
