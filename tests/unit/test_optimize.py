from __future__ import annotations

import numpy as np
import pytest

from selkit.engine.optimize import (
    fit_single_start,
    pack_params,
    softplus,
    softplus_inv,
    unpack_params,
)


def test_softplus_round_trip() -> None:
    for x in (0.01, 0.1, 1.0, 10.0, 100.0):
        u = softplus_inv(x)
        assert softplus(u) == pytest.approx(x, rel=1e-9)


def test_pack_unpack_round_trip() -> None:
    spec = {"omega": "positive", "kappa": "positive", "p0": "unit"}
    params = {"omega": 0.5, "kappa": 2.1, "p0": 0.3}
    u = pack_params(params, spec)
    back = unpack_params(u, spec)
    for k, v in params.items():
        assert back[k] == pytest.approx(v, rel=1e-9)


def test_fit_single_start_minimizes_quadratic() -> None:
    def neg_lnL(params: dict[str, float]) -> float:
        return (params["omega"] - 0.7) ** 2 + (params["kappa"] - 2.0) ** 2

    spec = {"omega": "positive", "kappa": "positive"}
    start = {"omega": 1.5, "kappa": 3.0}
    result = fit_single_start(neg_lnL, start=start, transform_spec=spec, seed=0)
    assert result.params["omega"] == pytest.approx(0.7, abs=1e-3)
    assert result.params["kappa"] == pytest.approx(2.0, abs=1e-3)
    assert result.final_lnL == pytest.approx(0.0, abs=1e-6)


from selkit.engine.optimize import MultiStartResult, fit_multi_start


def test_multi_start_picks_best_and_reports_convergence() -> None:
    def neg_lnL(params: dict[str, float]) -> float:
        return (params["omega"] - 0.7) ** 2 + (params["kappa"] - 2.0) ** 2

    def starting_values(seed: int) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        return {"omega": float(rng.uniform(0.1, 2.0)), "kappa": float(rng.uniform(1.0, 5.0))}

    spec = {"omega": "positive", "kappa": "positive"}
    result = fit_multi_start(
        neg_lnL=neg_lnL,
        starting_values=starting_values,
        transform_spec=spec,
        n_starts=3,
        seed=42,
        convergence_tol=0.5,
    )
    assert isinstance(result, MultiStartResult)
    assert result.converged
    assert result.best.params["omega"] == pytest.approx(0.7, abs=1e-3)
    assert len(result.starts) == 3


def test_multi_start_flags_non_convergence() -> None:
    def neg_lnL(params: dict[str, float]) -> float:
        o = params["omega"]
        return min((o - 0.5) ** 2 + 1.0, (o - 2.0) ** 2)

    def starting_values(seed: int) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        return {"omega": float(rng.choice([0.4, 1.9, 0.6]))}

    spec = {"omega": "positive"}
    result = fit_multi_start(
        neg_lnL=neg_lnL,
        starting_values=starting_values,
        transform_spec=spec,
        n_starts=3,
        seed=3,
        convergence_tol=0.1,
    )
    assert not result.converged
