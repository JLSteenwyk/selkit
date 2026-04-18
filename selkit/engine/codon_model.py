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
