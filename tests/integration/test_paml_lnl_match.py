"""Static PAML lnL-correctness gate.

For each corpus case, evaluate selkit's log-likelihood at PAML's exact
reported (branch lengths, model parameters) point and assert it matches
PAML's reported lnL. This is fast (seconds) because no optimizer runs —
it only exercises Q construction, eigendecomposition, and Felsenstein
pruning. It is the tightest automated regression gate for numerical
correctness of the core likelihood calculation.

For the slower full-fit validation (also runs optimizer + multi-start
convergence), see ``tests/validation/test_paml_corpus.py`` which is not
run by default.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from selkit.engine.codon_model import M0, M1a, M2a, M7, M8
from selkit.engine.genetic_code import GeneticCode
from selkit.engine.likelihood import tree_log_likelihood_mixture
from selkit.engine.rate_matrix import estimate_f3x4
from selkit.io.alignment import read_alignment
from selkit.io.tree import parse_newick

_CORPUS = Path(__file__).parent.parent / "validation" / "corpus"

_MODEL_CTORS = {
    "M0": M0,
    "M1a": M1a,
    "M2a": M2a,
    "M7": M7,
    "M8": M8,
}


def _cases() -> list[tuple[str, str]]:
    """Flatten corpus into (case_id, model_name) pairs so each model gets its own test."""
    out: list[tuple[str, str]] = []
    if not _CORPUS.exists():
        return out
    for case_dir in sorted(_CORPUS.iterdir()):
        if not case_dir.is_dir():
            continue
        exp_path = case_dir / "expected.json"
        if not exp_path.exists():
            continue
        expected = json.loads(exp_path.read_text())
        for model_name in expected.get("fits", {}):
            if model_name in _MODEL_CTORS:
                out.append((case_dir.name, model_name))
    return out


@pytest.mark.parametrize("case_id,model_name", _cases())
def test_lnl_matches_paml_at_reported_point(case_id: str, model_name: str) -> None:
    case_dir = _CORPUS / case_id
    expected = json.loads((case_dir / "expected.json").read_text())
    fit_spec = expected["fits"][model_name]
    if "newick_with_bls" not in fit_spec:
        pytest.skip(f"{case_id}/{model_name}: newick_with_bls missing from expected.json")

    gc = GeneticCode.standard()
    aln = read_alignment(case_dir / "alignment.fa", genetic_code=gc)
    pi = estimate_f3x4(aln.codons, gc)
    tree = parse_newick(fit_spec["newick_with_bls"])

    model = _MODEL_CTORS[model_name](gc=gc, pi=pi)
    weights, Qs = model.build(params=fit_spec["params"])
    lnL = tree_log_likelihood_mixture(
        tree=tree,
        codons=aln.codons,
        taxon_order=aln.taxa,
        Qs=Qs,
        weights=weights,
        pi=pi,
    )
    assert abs(lnL - fit_spec["lnL"]) < 1e-3, (
        f"{case_id}/{model_name}: selkit lnL={lnL:.6f} vs PAML={fit_spec['lnL']:.6f}, "
        f"diff={abs(lnL - fit_spec['lnL']):.6f}"
    )
