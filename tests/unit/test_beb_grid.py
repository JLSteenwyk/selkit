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


def test_two_grid_point_oracle() -> None:
    """At G=2 with hand-computed marginal likelihoods, posterior matches Yang 2005 eq. 5.

    Construct a tiny synthetic problem (2 grid points, 3 sites, 2 classes) and
    compute the integrated posterior independently from raw arrays via

        f(D|θ_g) = Π_h Σ_k w_k(θ_g) · L_{h,k}(θ_g)
        P(k | D_h) = Σ_g f(D|θ_g)·π(θ_g)·w_k(θ_g)·L_{h,k}(θ_g)
                     ─────────────────────────────────────────
                     Σ_g f(D|θ_g)·π(θ_g)·Σ_{k'} w_{k'}(θ_g)·L_{h,k'}(θ_g)

    and verify ``integrate_posteriors_over_grid`` agrees to <1e-12.

    This is the oracle that proves the f(D|θ_g) marginal-likelihood weight
    is wired in correctly; the singleton-grid (G=1) test passes vacuously
    because at G=1 any constant scalar cancels.
    """
    from selkit.engine.beb._grid import integrate_posteriors_over_grid

    # Deterministic, hand-checkable values. Choose log-likelihoods on a
    # moderate scale so f(D|θ) differs noticeably between grid points
    # (otherwise the bug would barely show up).
    n_grid, n_sites, n_classes = 2, 3, 2
    # Per-grid logL[g, h, k]:
    logL = np.array(
        [
            # grid point 0 — site/class likelihoods are mostly small
            [[-3.0, -1.0],
             [-2.0, -4.0],
             [-1.5, -2.5]],
            # grid point 1 — overall higher likelihoods → f(D|θ_1) ≫ f(D|θ_0)
            [[-0.5, -2.0],
             [-1.0, -0.8],
             [-0.7, -1.2]],
        ]
    )
    L = np.exp(logL)  # (G, n_sites, n_classes)

    weights = np.array([[0.7, 0.3], [0.4, 0.6]])      # (G, K)
    omegas = np.array([[0.2, 2.5], [0.2, 4.0]])       # (G, K), class 1 is positive
    prior = np.array([0.5, 0.5])                      # uniform 2-pt grid
    grid = np.array([[0.0], [1.0]])                   # G=2, H=1

    def site_class_loglik(point: np.ndarray) -> np.ndarray:
        idx = int(point[0])
        return logL[idx]   # (n_sites, n_classes)

    beb = integrate_posteriors_over_grid(
        hyperparameter_grid=grid,
        site_class_loglik=site_class_loglik,
        class_weights_over_grid=weights,
        class_omegas_over_grid=omegas,
        prior_on_grid=prior,
    )

    # ---- Hand-compute the oracle in linear space (small enough — no underflow). ----
    # f(D|θ_g) = Π_h Σ_k w_k(θ_g) · L_{h,k}(θ_g)
    f_data = np.empty(n_grid)
    for g in range(n_grid):
        per_site_mix = (weights[g][None, :] * L[g]).sum(axis=1)   # (n_sites,)
        f_data[g] = np.prod(per_site_mix)

    # Numerator(g, h, k) = f(D|θ_g) · π(θ_g) · w_k(θ_g) · L_{h,k}(θ_g)
    num = np.empty((n_grid, n_sites, n_classes))
    for g in range(n_grid):
        num[g] = f_data[g] * prior[g] * weights[g][None, :] * L[g]
    num_per_class = num.sum(axis=0)                                # (n_sites, K)
    denom = num_per_class.sum(axis=1, keepdims=True)               # (n_sites, 1)
    expected_posterior = num_per_class / denom                     # (n_sites, K)

    # Posterior-mean ω: Σ_{g,k} mass(g,h,k) · ω_k(θ_g) where mass = num/denom_h.
    mass = num / denom[None, :, :]                                  # (G, n_sites, K)
    expected_mean_omega = (mass * omegas[:, None, :]).sum(axis=(0, 2))   # (n_sites,)

    # p_positive: Σ_{g,k: ω_k>1} mass(g,h,k)
    pos_mask = (omegas > 1.0)[:, None, :]                            # (G, 1, K)
    expected_p_positive = (mass * pos_mask).sum(axis=(0, 2))         # (n_sites,)

    np.testing.assert_allclose(beb["posterior"], expected_posterior, atol=1e-12)
    np.testing.assert_allclose(beb["posterior_mean_omega"], expected_mean_omega, atol=1e-12)
    np.testing.assert_allclose(beb["p_positive"], expected_p_positive, atol=1e-12)

    # Sanity: f(D|θ_1) ≫ f(D|θ_0), so the buggy implementation (which omits f)
    # would give a clearly different answer. Confirm by reproducing the buggy
    # numerator (drop f_data) and showing the posterior would differ noticeably.
    assert f_data[1] > 5 * f_data[0], (
        f"oracle setup too benign: f_data ratio is {f_data[1]/f_data[0]:.2f}; "
        "pick logLs that make f(D|θ_g) differ across grid points."
    )
    buggy_num = np.empty_like(num)
    for g in range(n_grid):
        buggy_num[g] = prior[g] * weights[g][None, :] * L[g]    # missing f(D|θ_g)
    buggy_per_class = buggy_num.sum(axis=0)
    buggy_post = buggy_per_class / buggy_per_class.sum(axis=1, keepdims=True)
    # The bug must move at least one site's posterior by >0.05 — otherwise
    # the singleton invariant alone would have been "proof enough" and this
    # test wouldn't be a real oracle.
    assert np.max(np.abs(buggy_post - expected_posterior)) > 0.05, (
        "oracle does not actually distinguish buggy from correct integrand; "
        "make the per-grid-point logLs differ more so f(D|θ_g) varies."
    )


def test_public_api_does_not_expose_compute_neb() -> None:
    """compute_neb is a test oracle, not public API."""
    import selkit.engine.beb as beb_pkg
    assert "compute_neb" not in getattr(beb_pkg, "__all__", [])
    # Still importable from its concrete module — that is fine.
    from selkit.engine.beb.site import compute_neb  # noqa: F401
