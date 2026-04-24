"""Grid-integration helper for Yang 2005 Bayes Empirical Bayes (BEB).

The integrand (eq. 5 of Yang 2005, MBE 22:1107) is:

    P(class k | site h) = ∫ P(class k | site h, θ) f(D|θ) π(θ) dθ
                         ÷ ∫ f(D|θ) π(θ) dθ

where θ are the mixture hyperparameters (e.g. (p0, p1, ω2) for M2a). Per
Yang 2005, the prior π(θ) is uniform on a discrete grid that partitions
each hyperparameter's support into equal bins. The integrand is evaluated
at each grid point θ_g; numerator/denominator are accumulated in log space.

This helper is model-agnostic. Per-model grid construction (including
hyperparameter support, per-class ω per grid point, and per-class weight
per grid point) is the caller's responsibility.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.special import logsumexp


def integrate_posteriors_over_grid(
    *,
    hyperparameter_grid: np.ndarray,         # (G, H)
    site_class_loglik: Callable[[np.ndarray], np.ndarray],
    class_weights_over_grid: np.ndarray,      # (G, K)
    class_omegas_over_grid: np.ndarray,       # (G, K)
    prior_on_grid: np.ndarray,                # (G,)
) -> dict[str, np.ndarray]:
    """Integrate per-site class posteriors over a hyperparameter grid.

    Parameters
    ----------
    hyperparameter_grid : shape (G, H)
        G grid points; H hyperparameters per point. Passed row-by-row to
        ``site_class_loglik``.
    site_class_loglik : callable
        Given one grid point (H,), returns the (n_sites, n_classes) array of
        log-likelihoods P(data at site | class k, θ_g). The caller pre-builds
        the Q matrices and per-class partials at each θ_g.
    class_weights_over_grid : shape (G, K)
        Mixture weights w_k(θ_g). Rows need not sum to 1 in the PAML sense;
        the function normalizes per-site posteriors as part of the integrand.
    class_omegas_over_grid : shape (G, K)
        ω values ω_k(θ_g) per class per grid point. Used for the posterior-
        mean-ω and the p_positive indicator (ω > 1).
    prior_on_grid : shape (G,)
        Grid-prior weights. Yang 2005 uses uniform (1/G for every g); callers
        can override if they use a non-uniform partition (not used in v0.3).

    Returns
    -------
    dict with keys
        'posterior':              (n_sites, n_classes) — Σ_g-integrated class posteriors
        'p_positive':             (n_sites,)           — Σ_{k: ω_k > 1 at θ_g} posterior mass
        'posterior_mean_omega':   (n_sites,)           — E[ω | site], integrated over θ

    Numerical method
    ----------------
    Per Yang 2005 eq. 5, the BEB posterior is

        P(k | D_h) ≈ Σ_g f(D|θ_g) · π(θ_g) · w_k(θ_g) · L_{h,k}(θ_g)
                     ───────────────────────────────────────────────
                     Σ_g f(D|θ_g) · π(θ_g) · Σ_{k'} w_{k'}(θ_g) · L_{h,k'}(θ_g)

    where ``f(D|θ_g) = ∏_h Σ_k w_k(θ_g) · L_{h,k}(θ_g)`` is the per-grid-point
    marginal likelihood of the data under the mixture at θ_g. This term weights
    each grid point by how well it explains the data; it does NOT cancel after
    integrating over θ (it cancels only at fixed θ). Without this weight every
    grid point gets equal mass, producing a uniform-prior-averaged estimate
    rather than a data-conditioned BEB posterior.

    For each site h, class k, grid point g, the numerator integrand is

        exp(log f(D|θ_g) + log_prior[g] + log w_k(θ_g) + logL(h | k, θ_g)).

    The denominator integrand sums over k first (exp then sum) before summing
    over g. Both are accumulated via scipy.special.logsumexp so that single
    extreme grid points do not underflow or overflow.

    Singleton-grid invariant. At G=1 the (constant) ``f(D|θ_1)`` factor cancels
    between numerator and denominator, so the posterior collapses to NEB-at-θ_1,
    bit-for-bit. The fix preserves this invariant by construction.
    """
    G, H = hyperparameter_grid.shape
    assert prior_on_grid.shape == (G,), (
        f"prior_on_grid shape {prior_on_grid.shape} != ({G},)"
    )
    assert class_weights_over_grid.shape[0] == G
    assert class_omegas_over_grid.shape == class_weights_over_grid.shape

    K = class_weights_over_grid.shape[1]
    log_prior = np.log(np.clip(prior_on_grid, 1e-300, None))  # (G,)

    # Probe shape with the first grid point.
    first = site_class_loglik(hyperparameter_grid[0])  # (n_sites, K)
    n_sites = first.shape[0]
    assert first.shape == (n_sites, K), (
        f"site_class_loglik returned {first.shape}; expected (n_sites, K=({K}))"
    )

    # Accumulators in log space: one stack of (n_sites, K) log-integrand terms per grid point.
    log_num_stack = np.empty((G, n_sites, K))
    # For the positive-selection indicator, track per grid point which classes have ω > 1.
    positive_mask_over_grid = class_omegas_over_grid > 1.0   # (G, K)

    # log_data_at_g[g] = log f(D | θ_g) = Σ_h log Σ_k w_k(θ_g) · L_{h,k}(θ_g)
    # (Yang 2005 eq. 5 marginal-likelihood weight per grid point.)
    log_data_at_g = np.empty(G)

    for g in range(G):
        logL_g = first if g == 0 else site_class_loglik(hyperparameter_grid[g])  # (n_sites, K)
        log_w_g = np.log(np.clip(class_weights_over_grid[g], 1e-300, None))      # (K,)
        # Per-site mixture loglik at θ_g: log Σ_k w_k(θ_g) · L_{h,k}(θ_g).
        log_per_site_g = logsumexp(log_w_g[None, :] + logL_g, axis=1)            # (n_sites,)
        log_data_at_g[g] = log_per_site_g.sum()
        log_num_stack[g] = (
            log_data_at_g[g] + log_prior[g] + log_w_g[None, :] + logL_g
        )                                                                         # (n_sites, K)

    # Posterior P(class k | site h) = Σ_g num(g, h, k) / Σ_{g,k'} num(g, h, k')
    # Numerator in log space: logsumexp over g for each (site, class).
    log_num_per_class = logsumexp(log_num_stack, axis=0)             # (n_sites, K)
    log_denom = logsumexp(log_num_per_class, axis=1)                 # (n_sites,)
    log_post = log_num_per_class - log_denom[:, None]                # (n_sites, K)
    posterior = np.exp(log_post)                                     # (n_sites, K)

    # Posterior-mean ω. Each grid point contributes its own ω per class, so we
    # weight per (g, k) by the integrand mass exp(log_num_stack) / denom.
    # This collapses to Σ_g p(g, k | h) · ω_k(θ_g) when rewritten.
    # Implement directly: masses per (g, h, k) = exp(log_num_stack - log_denom[newaxis,:,newaxis]).
    mass = np.exp(log_num_stack - log_denom[None, :, None])          # (G, n_sites, K)
    omega_over = class_omegas_over_grid[:, None, :]                  # (G, 1, K)
    posterior_mean_omega = (mass * omega_over).sum(axis=(0, 2))      # (n_sites,)

    pos_mask = positive_mask_over_grid[:, None, :]                   # (G, 1, K)
    p_positive = (mass * pos_mask).sum(axis=(0, 2))                  # (n_sites,)

    return {
        "posterior": posterior,
        "p_positive": p_positive,
        "posterior_mean_omega": posterior_mean_omega,
    }
