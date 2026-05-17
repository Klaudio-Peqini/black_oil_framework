from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CartesianGrid3D:
    """Uniform structured Cartesian finite-volume grid in 3D.

    The grid is logically Cartesian and cell-centred. Cells are flattened with
    x as the fast index, then y, then z/layer:

        cell(i, j, k) = k * nx * ny + j * nx + i.

    The physical ``z`` coordinate is geometric. The reservoir depth used in
    gravity terms is stored separately and is positive downward. By default,
    depth is ``top_depth + z_cell_center``. A scalar or array depth can be
    supplied to represent structural dip without changing the logical mesh.
    """

    nx: int
    ny: int
    nz: int
    lx: float
    ly: float
    lz: float
    top_depth: float = 0.0
    depth: float | np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.ny <= 0 or self.nz <= 0:
            raise ValueError("nx, ny and nz must be positive")
        if self.lx <= 0.0 or self.ly <= 0.0 or self.lz <= 0.0:
            raise ValueError("lx, ly and lz must be positive")

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny

    @property
    def dz(self) -> float:
        return self.lz / self.nz

    @property
    def volumes(self) -> np.ndarray:
        return np.full(self.n_cells, self.dx * self.dy * self.dz, dtype=float)

    @property
    def centers_x(self) -> np.ndarray:
        return (np.arange(self.nx, dtype=float) + 0.5) * self.dx

    @property
    def centers_y(self) -> np.ndarray:
        return (np.arange(self.ny, dtype=float) + 0.5) * self.dy

    @property
    def centers_z(self) -> np.ndarray:
        return (np.arange(self.nz, dtype=float) + 0.5) * self.dz

    @property
    def centers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, z = np.meshgrid(self.centers_x, self.centers_y, self.centers_z, indexing="xy")
        # meshgrid with indexing='xy' returns shape (ny, nx, nz). Transpose to
        # the native reservoir ordering (nz, ny, nx) before flattening.
        return x.transpose(2, 0, 1).ravel(), y.transpose(2, 0, 1).ravel(), z.transpose(2, 0, 1).ravel()

    @property
    def depths(self) -> np.ndarray:
        if self.depth is None:
            _x, _y, z = self.centers
            return self.top_depth + z
        if np.isscalar(self.depth):
            return np.full(self.n_cells, float(self.depth), dtype=float)
        depth = np.asarray(self.depth, dtype=float)
        if depth.shape == (self.nz, self.ny, self.nx):
            return depth.ravel()
        if depth.shape == (self.n_cells,):
            return depth.copy()
        raise ValueError("depth must be None, scalar, shape (nz,ny,nx), or shape (n_cells,)")

    def cell_index(self, i: int, j: int, k: int) -> int:
        if not (0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz):
            raise IndexError("3D cell index out of range")
        return k * self.nx * self.ny + j * self.nx + i

    def unravel_cell(self, cell: int) -> tuple[int, int, int]:
        if not (0 <= cell < self.n_cells):
            raise IndexError("flat cell index out of range")
        k, rem = divmod(int(cell), self.nx * self.ny)
        j, i = divmod(rem, self.nx)
        return i, j, k

    @property
    def x_face_neighbors(self) -> np.ndarray:
        """Pairs across x-normal faces, positive direction west-to-east."""
        pairs: list[tuple[int, int]] = []
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx - 1):
                    pairs.append((self.cell_index(i, j, k), self.cell_index(i + 1, j, k)))
        return np.asarray(pairs, dtype=int).reshape((-1, 2))

    @property
    def y_face_neighbors(self) -> np.ndarray:
        """Pairs across y-normal faces, positive direction south-to-north."""
        pairs: list[tuple[int, int]] = []
        for k in range(self.nz):
            for j in range(self.ny - 1):
                for i in range(self.nx):
                    pairs.append((self.cell_index(i, j, k), self.cell_index(i, j + 1, k)))
        return np.asarray(pairs, dtype=int).reshape((-1, 2))

    @property
    def z_face_neighbors(self) -> np.ndarray:
        """Pairs across z-normal faces, positive direction from shallow to deep."""
        pairs: list[tuple[int, int]] = []
        for k in range(self.nz - 1):
            for j in range(self.ny):
                for i in range(self.nx):
                    pairs.append((self.cell_index(i, j, k), self.cell_index(i, j, k + 1)))
        return np.asarray(pairs, dtype=int).reshape((-1, 2))

    @property
    def neighbors(self) -> np.ndarray:
        parts = [self.x_face_neighbors, self.y_face_neighbors, self.z_face_neighbors]
        parts = [p for p in parts if len(p)]
        if not parts:
            return np.empty((0, 2), dtype=int)
        return np.vstack(parts)

    def _permeability_components(self, permeability) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return flattened Kx, Ky and Kz arrays from scalar/array/tuple/dict input."""
        if isinstance(permeability, dict):
            kx = permeability.get("kx", permeability.get("Kx"))
            ky = permeability.get("ky", permeability.get("Ky"))
            kz = permeability.get("kz", permeability.get("Kz"))
            if kx is None or ky is None or kz is None:
                raise ValueError("permeability dict must contain kx, ky and kz")
        elif isinstance(permeability, (tuple, list)) and len(permeability) == 3:
            kx, ky, kz = permeability
        else:
            kx = ky = kz = permeability

        def arr(k):
            a = np.asarray(k, dtype=float)
            if a.ndim == 0:
                return np.full(self.n_cells, float(a), dtype=float)
            if a.shape == (self.nz, self.ny, self.nx):
                return a.ravel()
            if a.shape == (self.n_cells,):
                return a.copy()
            raise ValueError("permeability components must be scalar, (nz,ny,nx), or (n_cells,)")

        kx_arr, ky_arr, kz_arr = arr(kx), arr(ky), arr(kz)
        if np.any(kx_arr <= 0.0) or np.any(ky_arr <= 0.0) or np.any(kz_arr <= 0.0):
            raise ValueError("permeability must be strictly positive")
        return kx_arr, ky_arr, kz_arr

    def geometric_transmissibility(self, permeability) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return x-, y- and z-face geometric transmissibility factors.

        These are absolute-permeability/geometric terms only. Phase mobility,
        capillary pressure and gravity potentials are applied by later flow
        modules. Harmonic averaging is used across each face.
        """
        kx, ky, kz = self._permeability_components(permeability)
        xp, yp, zp = self.x_face_neighbors, self.y_face_neighbors, self.z_face_neighbors

        if len(xp):
            i, j = xp[:, 0], xp[:, 1]
            ax = self.dy * self.dz
            tx = ax / ((0.5 * self.dx) / kx[i] + (0.5 * self.dx) / kx[j])
        else:
            tx = np.empty(0, dtype=float)

        if len(yp):
            i, j = yp[:, 0], yp[:, 1]
            ay = self.dx * self.dz
            ty = ay / ((0.5 * self.dy) / ky[i] + (0.5 * self.dy) / ky[j])
        else:
            ty = np.empty(0, dtype=float)

        if len(zp):
            i, j = zp[:, 0], zp[:, 1]
            az = self.dx * self.dy
            tz = az / ((0.5 * self.dz) / kz[i] + (0.5 * self.dz) / kz[j])
        else:
            tz = np.empty(0, dtype=float)
        return tx, ty, tz

    def boundary_cells(self, side: str) -> tuple[np.ndarray, float, float, str]:
        """Return boundary cells, half-distance, area and normal direction.

        Sides are ``left``, ``right``, ``front``, ``back``, ``top`` and
        ``bottom``. ``top`` is shallow; ``bottom`` is deep. The returned normal
        direction is one of ``x``, ``y`` or ``z`` and identifies the permeability
        component used by pressure-boundary or aquifer modules.
        """
        side = side.lower()
        cells: list[int] = []
        if side == "left":
            for k in range(self.nz):
                for j in range(self.ny):
                    cells.append(self.cell_index(0, j, k))
            return np.asarray(cells, dtype=int), 0.5 * self.dx, self.dy * self.dz, "x"
        if side == "right":
            for k in range(self.nz):
                for j in range(self.ny):
                    cells.append(self.cell_index(self.nx - 1, j, k))
            return np.asarray(cells, dtype=int), 0.5 * self.dx, self.dy * self.dz, "x"
        if side == "front":
            for k in range(self.nz):
                for i in range(self.nx):
                    cells.append(self.cell_index(i, 0, k))
            return np.asarray(cells, dtype=int), 0.5 * self.dy, self.dx * self.dz, "y"
        if side == "back":
            for k in range(self.nz):
                for i in range(self.nx):
                    cells.append(self.cell_index(i, self.ny - 1, k))
            return np.asarray(cells, dtype=int), 0.5 * self.dy, self.dx * self.dz, "y"
        if side == "top":
            for j in range(self.ny):
                for i in range(self.nx):
                    cells.append(self.cell_index(i, j, 0))
            return np.asarray(cells, dtype=int), 0.5 * self.dz, self.dx * self.dy, "z"
        if side == "bottom":
            for j in range(self.ny):
                for i in range(self.nx):
                    cells.append(self.cell_index(i, j, self.nz - 1))
            return np.asarray(cells, dtype=int), 0.5 * self.dz, self.dx * self.dy, "z"
        raise ValueError(f"Unknown 3D boundary side {side!r}")

    def reshape(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.size != self.n_cells:
            raise ValueError("field size is incompatible with grid")
        return values.reshape((self.nz, self.ny, self.nx))

    def pore_volume(self, porosity) -> np.ndarray:
        phi = np.asarray(porosity, dtype=float)
        if phi.ndim == 0:
            phi = np.full(self.n_cells, float(phi), dtype=float)
        elif phi.shape == (self.nz, self.ny, self.nx):
            phi = phi.ravel()
        elif phi.shape == (self.n_cells,):
            phi = phi.copy()
        else:
            raise ValueError("porosity must be scalar, (nz,ny,nx), or (n_cells,)")
        if np.any(phi <= 0.0) or np.any(phi >= 1.0):
            raise ValueError("porosity should lie in the open interval (0, 1)")
        return phi * self.volumes
