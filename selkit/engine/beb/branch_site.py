"""True BEB (Yang 2005) for branch-site Model A.

ModelA site classes (foreground-labeled tree):
  0  : ω₀ < 1 everywhere (purifying)
  1  : ω = 1 everywhere   (neutral)
  2a : bg ω₀, fg ω₂ > 1
  2b : bg ω = 1, fg ω₂ > 1

BEB integrates posteriors over (p0, p1, ω₂) per Yang 2005 supplementary §2.
ω₀ and κ are held at their MLE values (nuisance parameters).
"""
from __future__ import annotations

import numpy as np

from selkit.engine.beb._grid import integrate_posteriors_over_grid
from selkit.engine.codon_model import _model_a_build
from selkit.engine.fit import EngineFit
from selkit.engine.genetic_code import GeneticCode
from selkit.engine.likelihood import per_class_site_log_likelihood
from selkit.io.alignment import CodonAlignment
from selkit.io.results import BEBSite
from selkit.io.tree import LabeledTree


def _model_a_grid(grid_size: int, mle: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """ModelA grid: (p0, p1, ω2). p0, p1 ∈ (0,1) with p0 + p1 < 1.

    When grid_size == 1, returns a single point placed at the MLE.
    """
    if grid_size == 1:
        p0 = mle["p0"]
        p1 = (1.0 - mle["p0"]) * mle["p1_frac"]
        grid = np.array([[p0, p1, mle["omega2"]]])
        prior = np.array([1.0])
        return grid, prior

    omega2_max = max(3.0 * mle["omega2"], 11.0)
    edges = (np.arange(grid_size) + 0.5) / grid_size
    p0_vals = edges
    p1_vals = edges
    omega2_vals = np.exp(edges * (np.log(omega2_max) - np.log(1.0)))

    pts: list[tuple[float, float, float]] = []
    for p0 in p0_vals:
        for p1 in p1_vals:
            if p0 + p1 >= 1.0:
                continue
            for w2 in omega2_vals:
                pts.append((float(p0), float(p1), float(w2)))
    grid = np.asarray(pts)
    prior = np.full(len(pts), 1.0 / len(pts))
    return grid, prior


def _model_a_site_class_loglik_factory(
    *, gc: GeneticCode, pi: np.ndarray, kappa: float, omega0: float,
    tree: LabeledTree, alignment: CodonAlignment,
):
    """Build the 4-class per-site log-likelihood matrix for a ModelA grid point.

    Class weights at the grid point (p0, p1, ω₂):
        class 0:  p0
        class 1:  p1
        class 2a: p2 · p0 / (p0 + p1)
        class 2b: p2 · p1 / (p0 + p1)
      where p2 = 1 - p0 - p1.
    """
    def f(point: np.ndarray) -> np.ndarray:
        p0, p1, w2 = float(point[0]), float(point[1]), float(point[2])
        # Back-transform to the p1_frac stick-breaking that _model_a_build expects.
        denom_pf = max(1.0 - p0, 1e-12)
        p1_frac = max(min(p1 / denom_pf, 1.0 - 1e-12), 1e-12)
        weights, Qs = _model_a_build(
            gc=gc, pi=pi,
            omega0=omega0, omega2=w2,
            p0=p0, p1_frac=p1_frac, kappa=kappa,
        )
        per_class = per_class_site_log_likelihood(
            tree=tree, codons=alignment.codons,
            taxon_order=alignment.taxa, Qs=Qs, pi=pi,
        )
        return per_class.T  # (n_sites, 4)
    return f


def run_beb_branch_site(
    *,
    fit: EngineFit,
    grid_size: int,
    tree: LabeledTree,
    alignment: CodonAlignment,
    pi: np.ndarray,
    gc: GeneticCode,
) -> list[BEBSite]:
    """True BEB for branch-site Model A.

    Returns per-site ``BEBSite`` with ``p_class_2a``, ``p_class_2b`` populated
    and ``p_positive = p_class_2a + p_class_2b``.
    """
    kappa = fit.params["kappa"]
    omega0 = fit.params["omega0"]
    grid, prior = _model_a_grid(grid_size, fit.params)
    G = grid.shape[0]
    # Class weights per grid point.
    p0 = grid[:, 0]
    p1 = grid[:, 1]
    w2 = grid[:, 2]
    p2 = np.clip(1.0 - p0 - p1, 0.0, None)
    denom = np.where((p0 + p1) > 0.0, p0 + p1, 1.0)
    p2a = p2 * p0 / denom
    p2b = p2 * p1 / denom
    cw = np.stack([p0, p1, p2a, p2b], axis=1)  # (G, 4)

    # Per-class ω for the p_positive indicator / posterior-mean-ω. Classes 2a
    # and 2b both carry ω = ω₂ on the foreground; since BEB reports over the
    # mixed foreground+background sites, we use the (foreground) positive-
    # selection ω for these classes. Classes 0 / 1 carry ω₀ and 1 respectively.
    co = np.stack([
        np.full(G, omega0),    # class 0: ω₀
        np.ones(G),            # class 1: ω = 1
        w2,                    # class 2a: ω₂ (positive on foreground)
        w2,                    # class 2b: ω₂ (positive on foreground)
    ], axis=1)

    site_class_loglik = _model_a_site_class_loglik_factory(
        gc=gc, pi=pi, kappa=kappa, omega0=omega0,
        tree=tree, alignment=alignment,
    )
    out = integrate_posteriors_over_grid(
        hyperparameter_grid=grid,
        site_class_loglik=site_class_loglik,
        class_weights_over_grid=cw,
        class_omegas_over_grid=co,
        prior_on_grid=prior,
    )
    posterior = out["posterior"]  # (n_sites, 4)
    n_sites = posterior.shape[0]
    result: list[BEBSite] = []
    for s in range(n_sites):
        p2a_post = float(posterior[s, 2])
        p2b_post = float(posterior[s, 3])
        result.append(BEBSite(
            site=s + 1,
            p_positive=p2a_post + p2b_post,
            posterior_mean_omega=float(out["posterior_mean_omega"][s]),
            p_class_2a=p2a_post,
            p_class_2b=p2b_post,
            beb_grid_size=grid_size,
        ))
    return result
