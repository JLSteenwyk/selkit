from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from selkit.engine.genetic_code import GeneticCode
from selkit.engine.rate_matrix import build_q


class SiteModel(Protocol):
    name: str
    free_params: tuple[str, ...]

    def build(
        self, *, params: dict[str, float]
    ) -> tuple[list[float], list[np.ndarray]]: ...

    def starting_values(self, *, seed: int) -> dict[str, float]: ...


@dataclass
class M0:
    gc: GeneticCode
    pi: np.ndarray
    name: str = "M0"
    free_params: tuple[str, ...] = ("omega", "kappa")

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
    free_params: tuple[str, ...] = ("omega0", "p0", "kappa")

    def build(self, *, params: dict[str, float]) -> tuple[list[float], list[np.ndarray]]:
        omega0 = params["omega0"]
        p0 = params["p0"]
        kappa = params["kappa"]
        Q0 = build_q(self.gc, omega=omega0, kappa=kappa, pi=self.pi)
        Q1 = build_q(self.gc, omega=1.0, kappa=kappa, pi=self.pi)
        return [p0, 1.0 - p0], [Q0, Q1]

    def starting_values(self, *, seed: int) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        return {
            "omega0": float(rng.uniform(0.05, 0.8)),
            "p0": float(rng.uniform(0.3, 0.9)),
            "kappa": float(rng.uniform(1.5, 3.5)),
        }


@dataclass
class M2a:
    gc: GeneticCode
    pi: np.ndarray
    name: str = "M2a"
    free_params: tuple[str, ...] = ("omega0", "omega2", "p0", "p1_frac", "kappa")

    def build(self, *, params: dict[str, float]) -> tuple[list[float], list[np.ndarray]]:
        p0 = params["p0"]
        p1_frac = params["p1_frac"]
        p1 = (1.0 - p0) * p1_frac
        p2 = (1.0 - p0) * (1.0 - p1_frac)
        kappa = params["kappa"]
        Q0 = build_q(self.gc, omega=params["omega0"], kappa=kappa, pi=self.pi)
        Q1 = build_q(self.gc, omega=1.0, kappa=kappa, pi=self.pi)
        Q2 = build_q(self.gc, omega=params["omega2"], kappa=kappa, pi=self.pi)
        return [p0, p1, p2], [Q0, Q1, Q2]

    def starting_values(self, *, seed: int) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        return {
            "omega0": float(rng.uniform(0.05, 0.5)),
            "omega2": float(rng.uniform(1.5, 4.0)),
            "p0": float(rng.uniform(0.4, 0.85)),
            "p1_frac": float(rng.uniform(0.3, 0.7)),
            "kappa": float(rng.uniform(1.5, 3.5)),
        }
