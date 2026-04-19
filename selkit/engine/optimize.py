from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy.optimize import minimize

Transform = Literal["positive", "unit"]


@dataclass(frozen=True)
class SingleStartResult:
    params: dict[str, float]
    final_lnL: float
    iterations: int
    converged: bool


def softplus(u: float) -> float:
    if u > 30:
        return float(u)
    return float(np.log1p(np.exp(u)))


def softplus_inv(x: float) -> float:
    if x < 0:
        raise ValueError("softplus_inv requires x >= 0")
    if x > 30:
        return float(x)
    return float(np.log(np.expm1(x)))


def _sigmoid(u: float) -> float:
    return 1.0 / (1.0 + np.exp(-u))


def _logit(x: float) -> float:
    if x <= 0 or x >= 1:
        raise ValueError("logit requires 0 < x < 1")
    return float(np.log(x / (1 - x)))


def _apply(u: float, kind: Transform) -> float:
    if kind == "positive":
        return softplus(u)
    if kind == "unit":
        return _sigmoid(u)
    raise ValueError(f"unknown transform kind: {kind}")


def _invert(x: float, kind: Transform) -> float:
    if kind == "positive":
        return softplus_inv(x)
    if kind == "unit":
        return _logit(x)
    raise ValueError(f"unknown transform kind: {kind}")


def pack_params(params: dict[str, float], spec: dict[str, Transform]) -> np.ndarray:
    return np.array([_invert(params[k], spec[k]) for k in spec], dtype=np.float64)


def unpack_params(u: np.ndarray, spec: dict[str, Transform]) -> dict[str, float]:
    return {k: _apply(float(ui), kind) for ui, (k, kind) in zip(u, spec.items())}


def fit_single_start(
    neg_lnL: Callable[[dict[str, float]], float],
    *,
    start: dict[str, float],
    transform_spec: dict[str, Transform],
    seed: int,
    max_iter: int = 500,
) -> SingleStartResult:
    u0 = pack_params(start, transform_spec)

    def wrapped(u: np.ndarray) -> float:
        try:
            params = unpack_params(u, transform_spec)
            return float(neg_lnL(params))
        except (FloatingPointError, ValueError):
            return 1e18

    res = minimize(
        wrapped,
        u0,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": 1e-10, "gtol": 1e-7},
    )
    params = unpack_params(res.x, transform_spec)
    return SingleStartResult(
        params=params,
        final_lnL=float(res.fun),
        iterations=int(res.nit),
        converged=bool(res.success),
    )
