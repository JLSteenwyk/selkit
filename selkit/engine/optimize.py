from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy.optimize import minimize

Transform = Literal["positive", "unit", "positive_gt_one"]


@dataclass(frozen=True)
class SingleStartResult:
    params: dict[str, float]
    final_lnL: float
    iterations: int
    converged: bool
    hess_inv_diag: dict[str, float] | None = None  # natural-space SE per param


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
    if kind == "positive_gt_one":
        # x = 1 + softplus(u), so x ∈ (1, ∞)
        return 1.0 + softplus(u)
    raise ValueError(f"unknown transform kind: {kind}")


def _invert(x: float, kind: Transform) -> float:
    if kind == "positive":
        return softplus_inv(x)
    if kind == "unit":
        return _logit(x)
    if kind == "positive_gt_one":
        if x <= 1.0:
            raise ValueError("positive_gt_one requires x > 1")
        return softplus_inv(x - 1.0)
    raise ValueError(f"unknown transform kind: {kind}")


def pack_params(params: dict[str, float], spec: dict[str, Transform]) -> np.ndarray:
    return np.array([_invert(params[k], spec[k]) for k in spec], dtype=np.float64)


def unpack_params(u: np.ndarray, spec: dict[str, Transform]) -> dict[str, float]:
    return {k: _apply(float(ui), kind) for ui, (k, kind) in zip(u, spec.items())}


def _hess_inv_diag_u_space(res) -> np.ndarray | None:
    """Extract the diagonal of scipy's L-BFGS-B inverse-Hessian in u-space.

    Returns None if scipy did not return a LinearOperator-compatible
    hess_inv (older scipy, different optimizer, or a degenerate fit).
    """
    hi = getattr(res, "hess_inv", None)
    if hi is None:
        return None
    # scipy's LbfgsInvHessProduct is a LinearOperator; todense() gives the full
    # matrix (cheap for the small parameter counts we use in selkit).
    if hasattr(hi, "todense"):
        try:
            dense = np.asarray(hi.todense())
            return np.diag(dense).astype(np.float64, copy=True)
        except Exception:
            pass
    # Fallback: matvec with unit basis vectors.
    if hasattr(hi, "matvec"):
        n = res.x.size
        diag = np.empty(n, dtype=np.float64)
        try:
            for i in range(n):
                e = np.zeros(n, dtype=np.float64)
                e[i] = 1.0
                diag[i] = float(hi.matvec(e)[i])
            return diag
        except Exception:
            return None
    # Dense ndarray (rare for L-BFGS-B but permitted by scipy).
    if isinstance(hi, np.ndarray) and hi.ndim == 2:
        return np.diag(hi).astype(np.float64, copy=True)
    return None


def _natural_space_se(
    u: np.ndarray, var_u: np.ndarray, spec: dict[str, "Transform"],
) -> dict[str, float]:
    """Apply the delta-method transform Jacobian so SE is in natural (x) space."""
    out: dict[str, float] = {}
    for i, (name, kind) in enumerate(spec.items()):
        ui = float(u[i])
        vu = float(var_u[i])
        if not np.isfinite(vu) or vu < 0.0:
            continue  # skip this parameter; leave absent from dict
        if kind == "positive":
            # x = softplus(u); dx/du = sigmoid(u)
            dxdu = _sigmoid(ui)
        elif kind == "unit":
            # x = sigmoid(u); dx/du = x*(1-x)
            x = _sigmoid(ui)
            dxdu = x * (1.0 - x)
        elif kind == "positive_gt_one":
            # x = 1 + softplus(u); dx/du = sigmoid(u)
            dxdu = _sigmoid(ui)
        else:
            continue
        var_x = (dxdu ** 2) * vu
        if var_x >= 0.0 and np.isfinite(var_x):
            out[name] = float(np.sqrt(var_x))
    return out


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
    diag_u = _hess_inv_diag_u_space(res)
    hess_inv_diag: dict[str, float] | None
    if diag_u is None:
        warnings.warn(
            "scipy L-BFGS-B did not return a LinearOperator-compatible hess_inv; "
            "per-parameter SE will be unavailable for this fit. "
            "Requires scipy >= 1.0 (LbfgsInvHessProduct).",
            RuntimeWarning,
            stacklevel=2,
        )
        hess_inv_diag = None
    else:
        hess_inv_diag = _natural_space_se(res.x, diag_u, transform_spec)
        if not hess_inv_diag:  # all entries skipped (non-finite variances)
            hess_inv_diag = None
    return SingleStartResult(
        params=params,
        final_lnL=float(res.fun),
        iterations=int(res.nit),
        converged=bool(res.success),
        hess_inv_diag=hess_inv_diag,
    )


@dataclass(frozen=True)
class MultiStartResult:
    starts: list[SingleStartResult]
    best: SingleStartResult
    converged: bool


def fit_multi_start(
    *,
    neg_lnL: Callable[[dict[str, float]], float],
    starting_values: Callable[[int], dict[str, float]],
    transform_spec: dict[str, Transform],
    n_starts: int,
    seed: int,
    convergence_tol: float,
    max_iter: int = 500,
) -> MultiStartResult:
    rng = np.random.default_rng(seed)
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(n_starts)]
    starts: list[SingleStartResult] = []
    for s in seeds:
        start = starting_values(s)
        try:
            r = fit_single_start(
                neg_lnL,
                start=start,
                transform_spec=transform_spec,
                seed=s,
                max_iter=max_iter,
            )
        except Exception:
            continue
        starts.append(r)
    if not starts:
        raise RuntimeError("all optimization starts failed")
    starts.sort(key=lambda r: r.final_lnL)
    best = starts[0]
    converged = True
    if len(starts) >= 2:
        converged = (starts[1].final_lnL - starts[0].final_lnL) <= convergence_tol
    return MultiStartResult(starts=starts, best=best, converged=converged)
