from __future__ import annotations

import numpy as np
import pytest


def test_singleton_grid_at_mle_reproduces_neb() -> None:
    """With a 1-point grid, grid-BEB must equal NEB-at-MLE bit-for-bit.

    This is the self-consistency invariant: BEB is NEB plus integration.
    When the integration collapses to a point mass, the two must agree.
    """
    from selkit.engine.beb._grid import integrate_posteriors_over_grid
    from selkit.engine.beb.site import compute_neb

    rng = np.random.default_rng(42)
    n_sites, n_classes = 20, 3
    mle_weights = np.array([0.6, 0.3, 0.1])
    mle_omegas = np.array([0.1, 1.0, 3.5])
    per_class_site_logL = np.log(rng.uniform(0.01, 1.0, size=(n_classes, n_sites)))

    def site_class_loglik(point: np.ndarray) -> np.ndarray:
        # Singleton grid ignores point; return the MLE-evaluated per-class logL.
        # Shape must be (n_sites, n_classes).
        return per_class_site_logL.T

    grid = np.zeros((1, 1))           # 1 grid point, 1 dummy hyperparameter
    prior = np.array([1.0])           # uniform over 1 point
    class_weights = np.ones((1, n_classes)) * mle_weights[None, :]
    omegas_over_grid = np.ones((1, n_classes)) * mle_omegas[None, :]

    beb = integrate_posteriors_over_grid(
        hyperparameter_grid=grid,
        site_class_loglik=site_class_loglik,
        class_weights_over_grid=class_weights,
        class_omegas_over_grid=omegas_over_grid,
        prior_on_grid=prior,
    )
    # beb: dict with 'posterior' (n_sites, n_classes), 'p_positive' (n_sites,),
    #      'posterior_mean_omega' (n_sites,).

    neb = compute_neb(
        per_class_site_logL=per_class_site_logL,
        weights=mle_weights.tolist(),
        omegas=mle_omegas.tolist(),
    )
    for s, site_neb in enumerate(neb):
        assert abs(beb["p_positive"][s] - site_neb.p_positive) < 1e-10, (
            f"site {s}: BEB p_positive {beb['p_positive'][s]:.15f} != "
            f"NEB {site_neb.p_positive:.15f}"
        )
        assert abs(beb["posterior_mean_omega"][s] - site_neb.posterior_mean_omega) < 1e-10


def test_log_sum_exp_stability_on_extreme_values() -> None:
    """Two grid points with log-likelihoods differing by 1e6 must not underflow
    to NaN or 0. The dominant point should contribute ~all the mass."""
    from selkit.engine.beb._grid import integrate_posteriors_over_grid

    n_sites, n_classes = 3, 2
    per_class_site_logL_a = np.full((n_classes, n_sites), -1_000_000.0)
    per_class_site_logL_b = np.full((n_classes, n_sites), +1_000_000.0)
    per_class_site_logL_b[1, :] += 5.0   # class 1 dominates at point b

    def site_class_loglik(point: np.ndarray) -> np.ndarray:
        idx = int(point[0])
        return (per_class_site_logL_a if idx == 0 else per_class_site_logL_b).T

    grid = np.array([[0.0], [1.0]])
    prior = np.array([0.5, 0.5])
    class_weights = np.array([[0.5, 0.5], [0.5, 0.5]])
    class_omegas = np.array([[0.1, 3.0], [0.1, 3.0]])

    beb = integrate_posteriors_over_grid(
        hyperparameter_grid=grid,
        site_class_loglik=site_class_loglik,
        class_weights_over_grid=class_weights,
        class_omegas_over_grid=class_omegas,
        prior_on_grid=prior,
    )
    assert np.all(np.isfinite(beb["p_positive"]))
    assert np.all(np.isfinite(beb["posterior_mean_omega"]))
    # Point b dominates; class 1 (omega=3) dominates at b; p_positive ≈ 1.
    assert np.all(beb["p_positive"] > 0.99)


def test_posterior_rows_sum_to_one() -> None:
    """At every site, posterior P(class | site) integrated over grid must sum to 1."""
    from selkit.engine.beb._grid import integrate_posteriors_over_grid

    rng = np.random.default_rng(7)
    n_grid, n_sites, n_classes = 5, 8, 4
    logLs = rng.normal(size=(n_grid, n_classes, n_sites))

    def site_class_loglik(point: np.ndarray) -> np.ndarray:
        idx = int(point[0])
        return logLs[idx].T

    grid = np.arange(n_grid, dtype=float).reshape(-1, 1)
    prior = np.ones(n_grid) / n_grid
    class_weights = rng.dirichlet(np.ones(n_classes), size=n_grid)
    class_omegas = rng.uniform(0.01, 5.0, size=(n_grid, n_classes))

    beb = integrate_posteriors_over_grid(
        hyperparameter_grid=grid,
        site_class_loglik=site_class_loglik,
        class_weights_over_grid=class_weights,
        class_omegas_over_grid=class_omegas,
        prior_on_grid=prior,
    )
    row_sums = beb["posterior"].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-10)
