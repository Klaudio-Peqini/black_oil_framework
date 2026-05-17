from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Rock:
    """Rock model with pressure-dependent porosity.

    The porosity law is

        phi(p) = phi_ref * exp(c_r * (p - p_ref)).

    The exponential form avoids negative porosity for large pressure excursions.
    For small compressibility it is equivalent to the usual linearized law.
    """

    porosity_ref: float | np.ndarray
    permeability: float | np.ndarray
    compressibility: float = 0.0
    p_ref: float = 0.0

    def porosity(self, p: float | np.ndarray) -> np.ndarray:
        p_arr = np.asarray(p, dtype=float)
        return np.asarray(self.porosity_ref, dtype=float) * np.exp(
            self.compressibility * (p_arr - self.p_ref)
        )

    def pore_volume(self, volumes: np.ndarray, p: float | np.ndarray) -> np.ndarray:
        return volumes * self.porosity(p)
