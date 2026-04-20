from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.stats import beta as _beta

from selkit.engine.genetic_code import GeneticCode
from selkit.engine.rate_matrix import build_q, scale_branch_site_qs, scale_mixture_qs


# Branch-site models return per-class per-label Q dicts; site models return a
# plain ndarray per class. Both forms are accepted downstream by
# tree_log_likelihood_mixture / per_class_site_log_likelihood.
class SiteModel(Protocol):
    name: str
    free_params: tuple[str, ...]
    transform_spec: dict[str, str]
    branch_site: bool  # True iff Qs are per-label dicts (branch-site)

    def build(
        self, *, params: dict[str, float]
    ) -> tuple[list[float], list]: ...

    def starting_values(self, *, seed: int) -> dict[str, float]: ...


# Transform specs encode each model's parameter-space constraints:
#   "positive":        x ∈ (0, ∞)    e.g. kappa, omega (M0 only), beta shape params
#   "unit":            x ∈ (0, 1)    proportions, omega0 in M1a/M2a
#   "positive_gt_one": x ∈ (1, ∞)    omega2 in M2a/M8 (positive-selection class)


@dataclass
class M0:
    gc: GeneticCode
    pi: np.ndarray
    name: str = "M0"
    branch_site: bool = False
    free_params: tuple[str, ...] = ("omega", "kappa")
    transform_spec: dict[str, str] = field(default_factory=lambda: {
        "omega": "positive",
        "kappa": "positive",
    })

    def build(self, *, params: dict[str, float]) -> tuple[list[float], list[np.ndarray]]:
        Q = build_q(self.gc, omega=params["omega"], kappa=params["kappa"], pi=self.pi)
        return [1.0], [Q]

    def starting_values(self, *, seed: int) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        return {
            "omega": float(rng.uniform(0.2, 1.2)),
            "kappa": float(rng.uniform(1.5, 3.5)),
        }


@dataclass
class M1a:
    gc: GeneticCode
    pi: np.ndarray
    name: str = "M1a"
    branch_site: bool = False
    free_params: tuple[str, ...] = ("omega0", "p0", "kappa")
    transform_spec: dict[str, str] = field(default_factory=lambda: {
        "omega0": "unit",       # omega0 ∈ (0, 1)
        "p0": "unit",
        "kappa": "positive",
    })

    def build(self, *, params: dict[str, float]) -> tuple[list[float], list[np.ndarray]]:
        omega0 = params["omega0"]
        p0 = params["p0"]
        kappa = params["kappa"]
        Q0 = build_q(self.gc, omega=omega0, kappa=kappa, pi=self.pi, unscaled=True)
        Q1 = build_q(self.gc, omega=1.0, kappa=kappa, pi=self.pi, unscaled=True)
        weights = [p0, 1.0 - p0]
        Qs = scale_mixture_qs([Q0, Q1], weights, self.pi)
        return weights, Qs

    def starting_values(self, *, seed: int) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        # Log-uniform on omega0 so boundary starts (ω0 near 0) are sampled
        # as densely as mid-range — needed to escape the ω0→1 collapse basin.
        return {
            "omega0": float(np.exp(rng.uniform(np.log(1e-3), np.log(0.95)))),
            "p0": float(rng.uniform(0.3, 0.9)),
            "kappa": float(rng.uniform(1.5, 3.5)),
        }


@dataclass
class M2a:
    gc: GeneticCode
    pi: np.ndarray
    name: str = "M2a"
    branch_site: bool = False
    free_params: tuple[str, ...] = ("omega0", "omega2", "p0", "p1_frac", "kappa")
    transform_spec: dict[str, str] = field(default_factory=lambda: {
        "omega0": "unit",             # ω0 ∈ (0, 1)
        "omega2": "positive_gt_one",  # ω2 > 1
        "p0": "unit",
        "p1_frac": "unit",
        "kappa": "positive",
    })

    def build(self, *, params: dict[str, float]) -> tuple[list[float], list[np.ndarray]]:
        p0 = params["p0"]
        p1_frac = params["p1_frac"]
        p1 = (1.0 - p0) * p1_frac
        p2 = (1.0 - p0) * (1.0 - p1_frac)
        kappa = params["kappa"]
        Q0 = build_q(self.gc, omega=params["omega0"], kappa=kappa, pi=self.pi, unscaled=True)
        Q1 = build_q(self.gc, omega=1.0, kappa=kappa, pi=self.pi, unscaled=True)
        Q2 = build_q(self.gc, omega=params["omega2"], kappa=kappa, pi=self.pi, unscaled=True)
        weights = [p0, p1, p2]
        Qs = scale_mixture_qs([Q0, Q1, Q2], weights, self.pi)
        return weights, Qs

    def starting_values(self, *, seed: int) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        return {
            "omega0": float(np.exp(rng.uniform(np.log(1e-3), np.log(0.9)))),
            "omega2": float(rng.uniform(1.5, 4.0)),
            "p0": float(rng.uniform(0.4, 0.85)),
            "p1_frac": float(rng.uniform(0.3, 0.7)),
            "kappa": float(rng.uniform(1.5, 3.5)),
        }


def _beta_quantiles(p_beta: float, q_beta: float, n: int) -> np.ndarray:
    qs = (np.arange(n) + 0.5) / n
    return _beta.ppf(qs, p_beta, q_beta)


@dataclass
class M7:
    gc: GeneticCode
    pi: np.ndarray
    name: str = "M7"
    branch_site: bool = False
    free_params: tuple[str, ...] = ("p_beta", "q_beta", "kappa")
    n_categories: int = 10
    transform_spec: dict[str, str] = field(default_factory=lambda: {
        "p_beta": "positive",
        "q_beta": "positive",
        "kappa": "positive",
    })

    def build(self, *, params: dict[str, float]) -> tuple[list[float], list[np.ndarray]]:
        omegas = _beta_quantiles(params["p_beta"], params["q_beta"], self.n_categories)
        omegas = np.clip(omegas, 1e-6, 1.0 - 1e-6)
        kappa = params["kappa"]
        Qs_raw = [
            build_q(self.gc, omega=float(o), kappa=kappa, pi=self.pi, unscaled=True)
            for o in omegas
        ]
        w = 1.0 / self.n_categories
        weights = [w] * self.n_categories
        Qs = scale_mixture_qs(Qs_raw, weights, self.pi)
        return weights, Qs

    def starting_values(self, *, seed: int) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        return {
            "p_beta": float(rng.uniform(0.3, 2.0)),
            "q_beta": float(rng.uniform(0.3, 2.0)),
            "kappa": float(rng.uniform(1.5, 3.5)),
        }


@dataclass
class M8:
    gc: GeneticCode
    pi: np.ndarray
    name: str = "M8"
    branch_site: bool = False
    free_params: tuple[str, ...] = ("p_beta", "q_beta", "p0", "omega2", "kappa")
    n_categories: int = 10
    transform_spec: dict[str, str] = field(default_factory=lambda: {
        "p_beta": "positive",
        "q_beta": "positive",
        "p0": "unit",
        "omega2": "positive_gt_one",  # ω2 > 1
        "kappa": "positive",
    })

    def build(self, *, params: dict[str, float]) -> tuple[list[float], list[np.ndarray]]:
        omegas = _beta_quantiles(params["p_beta"], params["q_beta"], self.n_categories)
        omegas = np.clip(omegas, 1e-6, 1.0 - 1e-6)
        kappa = params["kappa"]
        p0 = params["p0"]
        Qs_raw = [
            build_q(self.gc, omega=float(o), kappa=kappa, pi=self.pi, unscaled=True)
            for o in omegas
        ]
        Qs_raw.append(build_q(self.gc, omega=params["omega2"], kappa=kappa, pi=self.pi, unscaled=True))
        weights = [p0 / self.n_categories] * self.n_categories + [1.0 - p0]
        Qs = scale_mixture_qs(Qs_raw, weights, self.pi)
        return weights, Qs

    def starting_values(self, *, seed: int) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        return {
            "p_beta": float(rng.uniform(0.3, 2.0)),
            "q_beta": float(rng.uniform(0.3, 2.0)),
            "p0": float(rng.uniform(0.7, 0.98)),
            "omega2": float(rng.uniform(1.5, 4.0)),
            "kappa": float(rng.uniform(1.5, 3.5)),
        }


@dataclass
class M8a:
    gc: GeneticCode
    pi: np.ndarray
    name: str = "M8a"
    branch_site: bool = False
    free_params: tuple[str, ...] = ("p_beta", "q_beta", "p0", "kappa")
    n_categories: int = 10
    transform_spec: dict[str, str] = field(default_factory=lambda: {
        "p_beta": "positive",
        "q_beta": "positive",
        "p0": "unit",
        "kappa": "positive",
    })

    def build(self, *, params: dict[str, float]) -> tuple[list[float], list[np.ndarray]]:
        omegas = _beta_quantiles(params["p_beta"], params["q_beta"], self.n_categories)
        omegas = np.clip(omegas, 1e-6, 1.0 - 1e-6)
        kappa = params["kappa"]
        p0 = params["p0"]
        Qs_raw = [
            build_q(self.gc, omega=float(o), kappa=kappa, pi=self.pi, unscaled=True)
            for o in omegas
        ]
        Qs_raw.append(build_q(self.gc, omega=1.0, kappa=kappa, pi=self.pi, unscaled=True))
        weights = [p0 / self.n_categories] * self.n_categories + [1.0 - p0]
        Qs = scale_mixture_qs(Qs_raw, weights, self.pi)
        return weights, Qs

    def starting_values(self, *, seed: int) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        return {
            "p_beta": float(rng.uniform(0.3, 2.0)),
            "q_beta": float(rng.uniform(0.3, 2.0)),
            "p0": float(rng.uniform(0.7, 0.98)),
            "kappa": float(rng.uniform(1.5, 3.5)),
        }


# Branch-site models (Yang & Nielsen 2002; Zhang et al. 2005; Yang et al. 2005).
# Foreground branches have Node.label == 1 (set by apply_foreground_spec in
# selkit.io.tree). Model A has 4 site classes:
#
#   Class 0:  ω = ω₀ (purifying) on every branch. Proportion p₀.
#   Class 1:  ω = 1 (neutral)   on every branch. Proportion p₁.
#   Class 2a: ω = ω₀ on background; ω = ω₂ ≥ 1 on foreground. Proportion p₂a.
#   Class 2b: ω = 1  on background; ω = ω₂ ≥ 1 on foreground. Proportion p₂b.
#
# with p₂a = p₂ · p₀ / (p₀ + p₁) and p₂b = p₂ · p₁ / (p₀ + p₁),
# where p₂ = 1 - p₀ - p₁. Free params: ω₀ ∈ (0,1), ω₂ ≥ 1, p₀, p₁ (via p1_frac
# stick-breaking: p₁ = (1-p₀)·p1_frac, matching M2a's convention), κ.
#
# Model A null is the same with ω₂ fixed at 1 (boundary). The Model A null vs
# Model A LRT is the standard branch-site test of positive selection (1 df,
# mixed 50:50 χ²₀:χ²₁).


def _model_a_build(
    gc: GeneticCode,
    pi: np.ndarray,
    omega0: float,
    omega2: float,
    p0: float,
    p1_frac: float,
    kappa: float,
) -> tuple[list[float], list[dict[int, np.ndarray]]]:
    """Shared construction for Model A and Model A null."""
    p1 = (1.0 - p0) * p1_frac
    p2 = 1.0 - p0 - p1
    denom = p0 + p1
    if denom <= 0:
        raise ValueError("Model A: p0 + p1 must be > 0")
    p2a = p2 * p0 / denom
    p2b = p2 * p1 / denom
    weights = [p0, p1, p2a, p2b]

    Q_bg_w0 = build_q(gc, omega=omega0, kappa=kappa, pi=pi, unscaled=True)
    Q_bg_w1 = build_q(gc, omega=1.0, kappa=kappa, pi=pi, unscaled=True)
    Q_fg_w2 = build_q(gc, omega=omega2, kappa=kappa, pi=pi, unscaled=True)

    # Label 0 = background, label 1 = foreground. apply_foreground_spec sets
    # label=1 on foreground branches; everything else stays at the default 0.
    Qs_by_class_by_label: list[dict[int, np.ndarray]] = [
        {0: Q_bg_w0, 1: Q_bg_w0},  # class 0: purifying everywhere
        {0: Q_bg_w1, 1: Q_bg_w1},  # class 1: neutral everywhere
        {0: Q_bg_w0, 1: Q_fg_w2},  # class 2a: purifying bg, positive fg
        {0: Q_bg_w1, 1: Q_fg_w2},  # class 2b: neutral   bg, positive fg
    ]
    scaled = scale_branch_site_qs(Qs_by_class_by_label, weights, pi)
    return weights, scaled


@dataclass
class ModelA:
    """Branch-site Model A. Alternative hypothesis for the branch-site test."""

    gc: GeneticCode
    pi: np.ndarray
    name: str = "ModelA"
    branch_site: bool = True
    free_params: tuple[str, ...] = ("omega0", "omega2", "p0", "p1_frac", "kappa")
    transform_spec: dict[str, str] = field(default_factory=lambda: {
        "omega0": "unit",              # ω₀ ∈ (0, 1)
        "omega2": "positive_gt_one",   # ω₂ ≥ 1
        "p0": "unit",
        "p1_frac": "unit",
        "kappa": "positive",
    })

    def build(
        self, *, params: dict[str, float]
    ) -> tuple[list[float], list[dict[int, np.ndarray]]]:
        return _model_a_build(
            self.gc, self.pi,
            omega0=params["omega0"],
            omega2=params["omega2"],
            p0=params["p0"],
            p1_frac=params["p1_frac"],
            kappa=params["kappa"],
        )

    def starting_values(self, *, seed: int) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        return {
            "omega0": float(np.exp(rng.uniform(np.log(1e-3), np.log(0.9)))),
            "omega2": float(rng.uniform(1.5, 4.0)),
            "p0": float(rng.uniform(0.4, 0.85)),
            "p1_frac": float(rng.uniform(0.3, 0.7)),
            "kappa": float(rng.uniform(1.5, 3.5)),
        }


@dataclass
class ModelANull:
    """Branch-site Model A with ω₂ fixed at 1 (null for the branch-site test)."""

    gc: GeneticCode
    pi: np.ndarray
    name: str = "ModelA_null"
    branch_site: bool = True
    free_params: tuple[str, ...] = ("omega0", "p0", "p1_frac", "kappa")
    transform_spec: dict[str, str] = field(default_factory=lambda: {
        "omega0": "unit",
        "p0": "unit",
        "p1_frac": "unit",
        "kappa": "positive",
    })

    def build(
        self, *, params: dict[str, float]
    ) -> tuple[list[float], list[dict[int, np.ndarray]]]:
        return _model_a_build(
            self.gc, self.pi,
            omega0=params["omega0"],
            omega2=1.0,  # pinned to the boundary
            p0=params["p0"],
            p1_frac=params["p1_frac"],
            kappa=params["kappa"],
        )

    def starting_values(self, *, seed: int) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        return {
            "omega0": float(np.exp(rng.uniform(np.log(1e-3), np.log(0.9)))),
            "p0": float(rng.uniform(0.4, 0.85)),
            "p1_frac": float(rng.uniform(0.3, 0.7)),
            "kappa": float(rng.uniform(1.5, 3.5)),
        }
