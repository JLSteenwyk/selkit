from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from selkit.engine.codon_model import (
    FreeRatios,
    M0,
    NRatios,
    SiteModel,
    TwoRatios,
    TwoRatiosFixed,
)
from selkit.engine.genetic_code import GeneticCode
from selkit.errors import SelkitConfigError
from selkit.io.config import RunConfig
from selkit.io.results import RunResult
from selkit.io.tree import BranchRecord, LabeledTree
from selkit.services.codeml._orchestrator import run_family
from selkit.services.codeml.lrt import DFSpec
from selkit.services.validate import ValidatedInputs


# Named factory functions (not lambdas) so MODEL_REGISTRY pickles cleanly when
# passed through ProcessPoolExecutor in the orchestrator's parallel branch.
# Branch-family factories take (gc, pi, tree). Static models ignore tree;
# NRatios and FreeRatios inspect it for K / n_branches.
def _mk_m0(gc, pi, tree):
    return M0(gc=gc, pi=pi)


def _mk_two_ratios(gc, pi, tree):
    return TwoRatios(gc=gc, pi=pi)


def _mk_two_ratios_fixed(gc, pi, tree):
    return TwoRatiosFixed(gc=gc, pi=pi)


def _mk_n_ratios(gc, pi, tree: LabeledTree):
    K = tree.n_label_classes
    if K < 1:
        raise SelkitConfigError(
            "NRatios requires at least one #-label class on the tree; "
            "supply --foreground / --labels-file / use inline #1 ... #K."
        )
    return NRatios(gc=gc, pi=pi, K=K)


def _mk_free_ratios(gc, pi, tree: LabeledTree):
    # FreeRatios ignores any pre-existing labels; the service rewrites labels
    # to be unique-per-branch before fit. n_branches is the distinct-label count
    # after that rewrite.
    recs = tree.assign_unique_branch_labels(merge_root=True)
    n = len({r.branch_id for r in recs})
    return FreeRatios(gc=gc, pi=pi, n_branches=n)


MODEL_REGISTRY: dict[str, Callable] = {
    "M0":              _mk_m0,  # present so M0-vs-<branch-model> LRTs auto-wire
    "TwoRatios":       _mk_two_ratios,
    "TwoRatiosFixed":  _mk_two_ratios_fixed,
    "NRatios":         _mk_n_ratios,
    "FreeRatios":      _mk_free_ratios,
}


DEFAULT_LRTS: tuple[tuple[str, str, DFSpec, str], ...] = (
    ("M0", "TwoRatios", 1, "chi2"),
    ("TwoRatiosFixed", "TwoRatios", 1, "mixed_chi2"),
    ("M0", "NRatios", "lazy_K", "chi2"),
    ("M0", "FreeRatios", "lazy_Bminus2", "chi2"),
)


DEFAULT_BEB_MODELS: tuple[str, ...] = ()  # branch family has no BEB


def _require_branch_preconditions(
    inputs: ValidatedInputs, config: RunConfig
) -> None:
    """Enforce per-model tree preconditions before fit."""
    tree = inputs.tree
    models = tuple(config.models) if config.models else ()
    if "TwoRatios" in models or "TwoRatiosFixed" in models:
        if tree.n_label_classes != 1:
            raise SelkitConfigError(
                f"TwoRatios / TwoRatiosFixed require exactly one #-label class "
                f"(K=1) on the tree; got K={tree.n_label_classes}. "
                "Use NRatios for K>1 or re-label the tree with a single "
                "foreground class."
            )
    if "NRatios" in models:
        if tree.n_label_classes < 1:
            raise SelkitConfigError(
                "NRatios requires at least one #-label class on the tree; "
                "supply --foreground / --labels-file / use inline #1 ... #K."
            )
    # FreeRatios: no precondition; ignores labels.


def run_branch_models(
    *,
    inputs: ValidatedInputs,
    config: RunConfig,
    parallel: bool,
    progress: Optional[Callable[[str, str], None]] = None,
) -> RunResult:
    return run_family(
        family="branch",
        registry=MODEL_REGISTRY,
        default_lrts=DEFAULT_LRTS,
        default_beb_models=DEFAULT_BEB_MODELS,
        inputs=inputs, config=config,
        parallel=parallel, progress=progress,
        preconditions=_require_branch_preconditions,
    )
