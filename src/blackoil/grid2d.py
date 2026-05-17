from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CartesianGrid2D:
    """Uniform 2D Cartesian finite-volume grid.

    The grid is logically Cartesian and cell-centred. It is intended as the
    Step 5C bridge from the previous 1D models to heterogeneous 2D reservoir
    cases. Cells are flattened with x as the fast index:

        cell(i, j) = j * nx + i.

    Depth is positive downward. A scalar depth gives a horizontal reservoir;
    an array of shape ``(ny, nx)`` or ``(n_cells,)`` can represent structural
    dip or topography. The reservoir is areal 2D with constant thickness.
    """

    nx: int
    ny: int
    lx: float
    ly: float
    thickness: float = 1.0
    depth: float | np.ndarray = 0.0

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("nx and ny must be positive")
        if self.lx <= 0.0 or self.ly <= 0.0:
            raise ValueError("lx and ly must be positive")
        if self.thickness <= 0.0:
            raise ValueError("thickness must be positive")

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny

    @property
    def volumes(self) -> np.ndarray:
        return np.full(self.n_cells, self.dx * self.dy * self.thickness, dtype=float)

    @property
    def centers_x(self) -> np.ndarray:
        return (np.arange(self.nx, dtype=float) + 0.5) * self.dx

    @property
    def centers_y(self) -> np.ndarray:
        return (np.arange(self.ny, dtype=float) + 0.5) * self.dy

    @property
    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        x, y = np.meshgrid(self.centers_x, self.centers_y, indexing="xy")
        return x.ravel(), y.ravel()

    @property
    def depths(self) -> np.ndarray:
        if np.isscalar(self.depth):
            return np.full(self.n_cells, float(self.depth), dtype=float)
        depth = np.asarray(self.depth, dtype=float)
        if depth.shape == (self.ny, self.nx):
            return depth.ravel()
        if depth.shape == (self.n_cells,):
            return depth.copy()
        raise ValueError("depth must be scalar, shape (ny, nx), or shape (n_cells,)")

    def cell_index(self, i: int, j: int) -> int:
        if not (0 <= i < self.nx and 0 <= j < self.ny):
            raise IndexError("2D cell index out of range")
        return j * self.nx + i

    def unravel_cell(self, cell: int) -> tuple[int, int]:
        if not (0 <= cell < self.n_cells):
            raise IndexError("flat cell index out of range")
        return int(cell % self.nx), int(cell // self.nx)

    @property
    def x_face_neighbors(self) -> np.ndarray:
        """Pairs across vertical faces, positive direction west-to-east."""
        pairs = []
        for j in range(self.ny):
            for i in range(self.nx - 1):
                pairs.append((self.cell_index(i, j), self.cell_index(i + 1, j)))
        return np.asarray(pairs, dtype=int).reshape((-1, 2))

    @property
    def y_face_neighbors(self) -> np.ndarray:
        """Pairs across horizontal faces, positive direction south-to-north in grid coordinates.

        The coordinate y is only a logical areal direction. Gravity uses the
        separate ``depth`` field, not y itself.
        """
        pairs = []
        for j in range(self.ny - 1):
            for i in range(self.nx):
                pairs.append((self.cell_index(i, j), self.cell_index(i, j + 1)))
        return np.asarray(pairs, dtype=int).reshape((-1, 2))

    @property
    def neighbors(self) -> np.ndarray:
        if self.n_cells == 1:
            return np.empty((0, 2), dtype=int)
        return np.vstack([self.x_face_neighbors, self.y_face_neighbors])

    def _permeability_components(self, permeability) -> tuple[np.ndarray, np.ndarray]:
        """Return flattened Kx and Ky arrays from scalar/array/tuple/dict input."""
        if isinstance(permeability, dict):
            kx = permeability.get("kx", permeability.get("Kx"))
            ky = permeability.get("ky", permeability.get("Ky"))
            if kx is None or ky is None:
                raise ValueError("permeability dict must contain kx and ky")
        elif isinstance(permeability, (tuple, list)) and len(permeability) == 2:
            kx, ky = permeability
        else:
            kx = ky = permeability

        def arr(k):
            a = np.asarray(k, dtype=float)
            if a.ndim == 0:
                return np.full(self.n_cells, float(a), dtype=float)
            if a.shape == (self.ny, self.nx):
                return a.ravel()
            if a.shape == (self.n_cells,):
                return a.copy()
            raise ValueError("permeability components must be scalar, (ny,nx), or (n_cells,)")

        kx_arr = arr(kx)
        ky_arr = arr(ky)
        if np.any(kx_arr <= 0.0) or np.any(ky_arr <= 0.0):
            raise ValueError("permeability must be strictly positive")
        return kx_arr, ky_arr

    def geometric_transmissibility(self, permeability) -> tuple[np.ndarray, np.ndarray]:
        """Return x-face and y-face geometric transmissibility factors.

        The factors are pure geometric/absolute-permeability terms. Phase
        mobility and potential differences are applied later.
        """
        kx, ky = self._permeability_components(permeability)
        x_pairs = self.x_face_neighbors
        y_pairs = self.y_face_neighbors

        if len(x_pairs):
            i, j = x_pairs[:, 0], x_pairs[:, 1]
            ax = self.dy * self.thickness
            tx = ax / ((0.5 * self.dx) / kx[i] + (0.5 * self.dx) / kx[j])
        else:
            tx = np.empty(0, dtype=float)

        if len(y_pairs):
            i, j = y_pairs[:, 0], y_pairs[:, 1]
            ay = self.dx * self.thickness
            ty = ay / ((0.5 * self.dy) / ky[i] + (0.5 * self.dy) / ky[j])
        else:
            ty = np.empty(0, dtype=float)
        return tx, ty

    def reshape(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.size != self.n_cells:
            raise ValueError("field size is incompatible with grid")
        return values.reshape((self.ny, self.nx))
