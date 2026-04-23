from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.special import logsumexp
from scipy.stats import beta as _beta

from selkit.engine.beb._grid import integrate_posteriors_over_grid
from selkit.engine.codon_model import M2a, M8, _beta_quantiles
from selkit.engine.fit import EngineFit
from selkit.engine.genetic_code import GeneticCode
from selkit.engine.likelihood import per_class_site_log_likelihood
from selkit.engine.rate_matrix import build_q, scale_mixture_qs
from selkit.io.alignment import CodonAlignment
from selkit.io.results import BEBSite
from selkit.io.tree import LabeledTree


def compute_neb(
    *,
    per_class_site_logL: np.ndarray,
    weights: list[float],
    omegas: list[float],
) -> list[BEBSite]:
    """NEB oracle retained for self-consistency tests; not re-exported."""
    log_w = np.log(np.asarray(weights))[:, None]
    log_joint = per_class_site_logL + log_w
    log_norm = logsumexp(log_joint, axis=0)
    log_post = log_joint - log_norm
    post = np.exp(log_post)
    om = np.asarray(omegas)[:, None]
    mean_om = (post * om).sum(axis=0)
    positive_mask = (np.asarray(omegas) > 1.0)[:, None]
    p_pos = (post * positive_mask).sum(axis=0)
    out: list[BEBSite] = []
    for s in range(per_class_site_logL.shape[1]):
        out.append(BEBSite(
            site=s + 1,
            p_positive=float(p_pos[s]),
            posterior_mean_omega=float(mean_om[s]),
        ))
    return out


def _m2a_grid(grid_size: int, mle: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """M2a grid: (p0, p1) on the 2-simplex × ω2 ∈ (1, max(3·ω2_mle, 11)).

    When grid_size == 1, returns a single point at the MLE.

    Layout of H: [p0, p1, omega2]. p0, p1 are unconstrained draws from (0, 1);
    points with p0 + p1 >= 1 are dropped. omega2 is drawn log-uniformly from
    (1, ω2_max). Yang 2005 uses equal-probability bins of (p0, p1) on a
    triangle partition; the PAML implementation uses 10 points per axis.
    """
    if grid_size == 1:
        grid = np.array([[mle["p0"], (1.0 - mle["p0"]) * mle["p1_frac"], mle["omega2"]]])
        prior = np.array([1.0])
        return grid, prior

    omega2_mle = mle["omega2"]
    omega2_max = max(3.0 * omega2_mle, 11.0)
    # Midpoints of equal-probability bins per Yang 2005 eq. (7).
    edges = (np.arange(grid_size) + 0.5) / grid_size     # e.g. {0.05, 0.15, ..., 0.95} for G=10
    p0_vals = edges                                        # ∈ (0, 1)
    p1_vals = edges                                        # ∈ (0, 1)
    # log-uniform midpoints on (1, omega2_max):
    omega2_vals = np.exp(np.log(1.0) + edges * (np.log(omega2_max) - np.log(1.0)))

    pts: list[tuple[float, float, float]] = []
    for p0 in p0_vals:
        for p1 in p1_vals:
            if p0 + p1 >= 1.0:
                continue
            for w2 in omega2_vals:
                pts.append((float(p0), float(p1), float(w2)))
    grid = np.asarray(pts)                                # (G, 3)
    prior = np.full(len(pts), 1.0 / len(pts))
    return grid, prior


def _m2a_site_class_loglik_factory(
    *, gc: GeneticCode, pi: np.ndarray, kappa: float,
    tree: LabeledTree, alignment: CodonAlignment,
    omega0: float,
):
    """Return a callable that, given a grid point (p0, p1, ω2), computes the
    (n_sites, 3) matrix of per-class log-likelihoods log P(D_h | class k, θ_g).

    The κ is held at its MLE (Yang 2005 convention — the grid integrates only
    the mixture hyperparameters; κ is a nuisance parameter). ω0 is also held
    at its MLE.
    """
    def f(point: np.ndarray) -> np.ndarray:
        p0, p1, w2 = float(point[0]), float(point[1]), float(point[2])
        # p2 used for class weights; integrand uses these.
        p2 = max(1.0 - p0 - p1, 0.0)
        weights = [p0, p1, p2]
        # Yang 2005 M2a holds ω0 (purifying) at its MLE across the grid;
        # ω1 is fixed at 1.0; ω2 varies on the grid.
        Q0 = build_q(gc, omega=omega0, kappa=kappa, pi=pi, unscaled=True)
        Q1 = build_q(gc, omega=1.0, kappa=kappa, pi=pi, unscaled=True)
        Q2 = build_q(gc, omega=w2, kappa=kappa, pi=pi, unscaled=True)
        Qs = scale_mixture_qs([Q0, Q1, Q2], weights, pi)
        per_class = per_class_site_log_likelihood(
            tree=tree, codons=alignment.codons,
            taxon_order=alignment.taxa, Qs=Qs, pi=pi,
        )
        return per_class.T  # (n_sites, 3)
    return f


def run_beb_site(
    *,
    fit: EngineFit,
    model_name: Literal["M2a", "M8"],
    grid_size: int,
    tree: LabeledTree,
    alignment: CodonAlignment,
    pi: np.ndarray,
    gc: GeneticCode,
) -> list[BEBSite]:
    """True BEB (Yang 2005) for site models M2a and M8.

    M2a grid: (p0, p1, ω2) — 2-simplex × positive-selection-ω interval.
    M8 grid:  (p0, p, q, ω2) — β-distribution shape × positive-selection-ω.

    For grid_size == 1, the grid collapses to a single point at the MLE and
    the result is equivalent to NEB-at-MLE.
    """
    if model_name == "M2a":
        return _run_m2a(fit=fit, grid_size=grid_size, tree=tree, alignment=alignment, pi=pi, gc=gc)
    if model_name == "M8":
        return _run_m8(fit=fit, grid_size=grid_size, tree=tree, alignment=alignment, pi=pi, gc=gc)
    raise ValueError(f"run_beb_site: unsupported model {model_name!r}")


def _run_m2a(
    *, fit: EngineFit, grid_size: int, tree: LabeledTree,
    alignment: CodonAlignment, pi: np.ndarray, gc: GeneticCode,
) -> list[BEBSite]:
    kappa = fit.params["kappa"]
    omega0_mle = fit.params["omega0"]
    grid, prior = _m2a_grid(grid_size, fit.params)
    # class_weights_over_grid and class_omegas_over_grid — per grid point, 3 classes.
    cw = np.stack([
        grid[:, 0],                                      # p0 (class 0, ω0 purifying)
        grid[:, 1],                                      # p1 (class 1, ω=1)
        np.clip(1.0 - grid[:, 0] - grid[:, 1], 0.0, 1.0),# p2 (class 2, ω2>1)
    ], axis=1)
    # For the positive-selection indicator and posterior-mean-ω, each grid
    # point's class-0 ω is the MLE ω0 held fixed; class-1 ω=1; class-2 ω=ω2(g).
    co = np.stack([
        np.full(grid.shape[0], omega0_mle),
        np.ones(grid.shape[0]),
        grid[:, 2],
    ], axis=1)
    site_class_loglik = _m2a_site_class_loglik_factory(
        gc=gc, pi=pi, kappa=kappa, tree=tree, alignment=alignment,
        omega0=omega0_mle,
    )
    out = integrate_posteriors_over_grid(
        hyperparameter_grid=grid,
        site_class_loglik=site_class_loglik,
        class_weights_over_grid=cw,
        class_omegas_over_grid=co,
        prior_on_grid=prior,
    )
    n_sites = out["p_positive"].shape[0]
    return [
        BEBSite(
            site=s + 1,
            p_positive=float(out["p_positive"][s]),
            posterior_mean_omega=float(out["posterior_mean_omega"][s]),
            beb_grid_size=grid_size,
        )
        for s in range(n_sites)
    ]


def _run_m8(*args, **kwargs) -> list[BEBSite]:
    raise NotImplementedError("M8 BEB lands in Task 6")
