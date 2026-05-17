from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CornerPointGridSpec:
    """Lightweight container for future corner-point/corner-grid imports.

    Full corner-point flow support is intentionally deferred to later stages.
    This object records the essential ECLIPSE/GRDECL-style ingredients that a
    future importer must validate and map into the finite-volume connectivity:

    * ``coord``: pillar coordinates, expected shape ``((ny+1)*(nx+1), 6)``;
    * ``zcorn``: corner depths, expected size ``8*nx*ny*nz``;
    * ``actnum``: optional active-cell mask of size ``nx*ny*nz``.

    The Step 6A goal is to define this interface early so that structured
    Cartesian and later corner-point grids can share property arrays, active
    masks, cell-centred fields and VTK export conventions.
    """

    nx: int
    ny: int
    nz: int
    coord: np.ndarray
    zcorn: np.ndarray
    actnum: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.ny <= 0 or self.nz <= 0:
            raise ValueError("nx, ny and nz must be positive")
        coord = np.asarray(self.coord, dtype=float)
        zcorn = np.asarray(self.zcorn, dtype=float)
        expected_coord = (self.nx + 1) * (self.ny + 1)
        if coord.shape != (expected_coord, 6):
            raise ValueError(f"coord must have shape ({expected_coord}, 6)")
        if zcorn.size != 8 * self.nx * self.ny * self.nz:
            raise ValueError("zcorn must contain 8*nx*ny*nz entries")
        if self.actnum is not None:
            act = np.asarray(self.actnum, dtype=int)
            if act.size != self.nx * self.ny * self.nz:
                raise ValueError("actnum must contain nx*ny*nz entries")

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def active_mask(self) -> np.ndarray:
        if self.actnum is None:
            return np.ones(self.n_cells, dtype=bool)
        return np.asarray(self.actnum, dtype=int).ravel() != 0


def make_cartesian_cornerpoint_spec(grid) -> CornerPointGridSpec:
    """Build a simple corner-point specification equivalent to a Cartesian grid.

    This is mainly a testing and interoperability bridge. A real GRDECL import
    function can later return the same ``CornerPointGridSpec`` type after
    parsing COORD/ZCORN/ACTNUM from file.
    """
    xs = np.linspace(0.0, grid.lx, grid.nx + 1)
    ys = np.linspace(0.0, grid.ly, grid.ny + 1)
    ztop = float(grid.top_depth)
    zbot = float(grid.top_depth + grid.lz)
    coord = []
    for j in range(grid.ny + 1):
        for i in range(grid.nx + 1):
            x, y = xs[i], ys[j]
            coord.append([x, y, ztop, x, y, zbot])
    coord_arr = np.asarray(coord, dtype=float)

    # Store eight corner depths per cell. Ordering is deliberately documented as
    # local Cartesian corners: bottom/top in x-y pairs. Later GRDECL conversion
    # can replace this helper without affecting users of CornerPointGridSpec.
    zcorn = []
    z_edges = np.linspace(ztop, zbot, grid.nz + 1)
    for k in range(grid.nz):
        z0, z1 = z_edges[k], z_edges[k + 1]
        for _j in range(grid.ny):
            for _i in range(grid.nx):
                zcorn.extend([z0, z0, z0, z0, z1, z1, z1, z1])
    return CornerPointGridSpec(grid.nx, grid.ny, grid.nz, coord_arr, np.asarray(zcorn, dtype=float))
