from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CartesianGrid1D:
    """Uniform 1D Cartesian finite-volume grid.

    Parameters
    ----------
    nx:
        Number of cells.
    length:
        Domain length in metres.
    area:
        Cross-sectional area in square metres.
    depth:
        Cell-centre depth. Positive downward. Used later for gravity terms.
    """

    nx: int
    length: float
    area: float = 1.0
    depth: float | np.ndarray = 0.0

    def __post_init__(self) -> None:
        if self.nx <= 0:
            raise ValueError("nx must be positive")
        if self.length <= 0.0:
            raise ValueError("length must be positive")
        if self.area <= 0.0:
            raise ValueError("area must be positive")

    @property
    def dx(self) -> float:
        return self.length / self.nx

    @property
    def centers(self) -> np.ndarray:
        return (np.arange(self.nx) + 0.5) * self.dx

    @property
    def volumes(self) -> np.ndarray:
        return np.full(self.nx, self.area * self.dx, dtype=float)

    @property
    def neighbors(self) -> np.ndarray:
        """Return interior-face neighbour pairs as an array with shape (nx-1, 2)."""
        left = np.arange(self.nx - 1, dtype=int)
        right = left + 1
        return np.column_stack([left, right])

    @property
    def depths(self) -> np.ndarray:
        if np.isscalar(self.depth):
            return np.full(self.nx, float(self.depth), dtype=float)
        depth = np.asarray(self.depth, dtype=float)
        if depth.shape != (self.nx,):
            raise ValueError("depth array must have shape (nx,)")
        return depth

    def geometric_transmissibility(self, permeability: float | np.ndarray) -> np.ndarray:
        """Compute interior-face geometric transmissibilities.

        The returned value is the harmonic-average TPFA factor

            T_geo = A / (dx_i/(2K_i) + dx_j/(2K_j)),

        with units of m^3. Phase mobility and pressure difference are applied later.
        """
        k = np.asarray(permeability, dtype=float)
        if k.ndim == 0:
            k = np.full(self.nx, float(k), dtype=float)
        if k.shape != (self.nx,):
            raise ValueError("permeability must be scalar or have shape (nx,)")
        if np.any(k <= 0.0):
            raise ValueError("permeability must be strictly positive")

        i = np.arange(self.nx - 1)
        j = i + 1
        half_dx = 0.5 * self.dx
        return self.area / (half_dx / k[i] + half_dx / k[j])
