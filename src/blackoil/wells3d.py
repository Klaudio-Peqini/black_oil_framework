from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np

from .reservoir3d import as_cell_array_3d, ActiveCellMap3D


@dataclass(frozen=True)
class Completion3D:
    """One perforated/completed cell in a 3D well."""

    cell: int
    well_index: float
    status: str = "open"
    skin: float = 0.0
    segment: int | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if self.well_index < 0.0:
            raise ValueError("well_index must be non-negative")
        status = self.status.lower()
        if status not in {"open", "shut"}:
            raise ValueError("completion status must be 'open' or 'shut'")
        object.__setattr__(self, "status", status)

    @property
    def is_open(self) -> bool:
        return self.status == "open" and self.well_index > 0.0


@dataclass(frozen=True)
class WellTrajectory3D:
    """Polyline well trajectory in physical coordinates.

    Coordinates are interpreted in the same Cartesian coordinate system as the
    structured grid. The third coordinate is geometric z, not necessarily true
    vertical depth; the grid already stores the depth field used in gravity.
    """

    name: str
    points: np.ndarray

    def __post_init__(self) -> None:
        pts = np.asarray(self.points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 2:
            raise ValueError("trajectory points must have shape (n_points>=2, 3)")
        object.__setattr__(self, "points", pts)

    @classmethod
    def vertical(cls, name: str, x: float, y: float, z_top: float, z_bottom: float) -> "WellTrajectory3D":
        return cls(name=name, points=np.asarray([[x, y, z_top], [x, y, z_bottom]], dtype=float))

    @classmethod
    def horizontal_x(cls, name: str, x0: float, x1: float, y: float, z: float) -> "WellTrajectory3D":
        return cls(name=name, points=np.asarray([[x0, y, z], [x1, y, z]], dtype=float))

    @property
    def length(self) -> float:
        diff = np.diff(self.points, axis=0)
        return float(np.sum(np.linalg.norm(diff, axis=1)))

    def sampled_points(self, grid, samples_per_cell: int = 3) -> np.ndarray:
        """Sample the trajectory densely enough to identify crossed cells."""
        if samples_per_cell <= 0:
            raise ValueError("samples_per_cell must be positive")
        nominal = min(grid.dx, grid.dy, grid.dz) / float(samples_per_cell)
        chunks = []
        for a, b in zip(self.points[:-1], self.points[1:]):
            length = float(np.linalg.norm(b - a))
            n = max(2, int(np.ceil(length / max(nominal, 1.0e-30))) + 1)
            t = np.linspace(0.0, 1.0, n, endpoint=False)
            chunks.append(a[None, :] + t[:, None] * (b - a)[None, :])
        chunks.append(self.points[-1][None, :])
        return np.vstack(chunks)

    def cells_intersected(self, grid, *, samples_per_cell: int = 3, active: ActiveCellMap3D | None = None) -> np.ndarray:
        pts = self.sampled_points(grid, samples_per_cell=samples_per_cell)
        i = np.floor(pts[:, 0] / grid.dx).astype(int)
        j = np.floor(pts[:, 1] / grid.dy).astype(int)
        k = np.floor(pts[:, 2] / grid.dz).astype(int)
        i = np.clip(i, 0, grid.nx - 1); j = np.clip(j, 0, grid.ny - 1); k = np.clip(k, 0, grid.nz - 1)
        cells = np.array([grid.cell_index(ii, jj, kk) for ii, jj, kk in zip(i, j, k)], dtype=int)
        cells = np.unique(cells)
        if active is not None:
            cells = cells[active.active_mask[cells]]
        return cells


def peaceman_well_index_3d(
    grid,
    permeability: Mapping[str, Sequence[float] | np.ndarray | float],
    cell: int,
    *,
    well_radius: float = 0.1,
    skin: float = 0.0,
    orientation: str = "z",
    thickness: float | None = None,
) -> float:
    """Estimate a single-cell Peaceman well index for a Cartesian 3D grid.

    This is an engineering approximation intended for schedule/control and 3D
    completion testing. ``orientation='z'`` is a vertical well completion,
    ``'x'`` and ``'y'`` are horizontal completions aligned with the respective
    axis. Permeabilities must be in SI units.
    """
    if well_radius <= 0.0:
        raise ValueError("well_radius must be positive")
    orientation = orientation.lower()
    kx = as_cell_array_3d(grid, permeability["kx"], name="kx")[cell]
    ky = as_cell_array_3d(grid, permeability["ky"], name="ky")[cell]
    kz = as_cell_array_3d(grid, permeability["kz"], name="kz")[cell]
    if orientation == "z":
        k1, k2 = kx, ky
        d1, d2 = grid.dx, grid.dy
        h = grid.dz if thickness is None else float(thickness)
    elif orientation == "x":
        k1, k2 = ky, kz
        d1, d2 = grid.dy, grid.dz
        h = grid.dx if thickness is None else float(thickness)
    elif orientation == "y":
        k1, k2 = kx, kz
        d1, d2 = grid.dx, grid.dz
        h = grid.dy if thickness is None else float(thickness)
    else:
        raise ValueError("orientation must be 'x', 'y' or 'z'")
    if k1 <= 0.0 or k2 <= 0.0:
        raise ValueError("permeability must be positive")
    ratio = np.sqrt(k2 / k1)
    re = 0.28 * np.sqrt(np.sqrt(k2 / k1) * d1 * d1 + np.sqrt(k1 / k2) * d2 * d2) / ((k2 / k1) ** 0.25 + (k1 / k2) ** 0.25)
    denom = np.log(max(re / well_radius, 1.0000001)) + skin
    if denom <= 0.0:
        raise ValueError("well-index denominator must be positive; check radius/skin/grid size")
    return float(2.0 * np.pi * h * np.sqrt(k1 * k2) / denom)


def completions_from_trajectory(
    grid,
    trajectory: WellTrajectory3D,
    permeability: Mapping[str, Sequence[float] | np.ndarray | float],
    *,
    well_radius: float = 0.1,
    skin: float = 0.0,
    orientation: str = "z",
    active: ActiveCellMap3D | None = None,
    samples_per_cell: int = 3,
) -> list[Completion3D]:
    cells = trajectory.cells_intersected(grid, samples_per_cell=samples_per_cell, active=active)
    completions: list[Completion3D] = []
    for n, cell in enumerate(cells):
        wi = peaceman_well_index_3d(grid, permeability, int(cell), well_radius=well_radius, skin=skin, orientation=orientation)
        completions.append(Completion3D(cell=int(cell), well_index=wi, skin=skin, segment=n, label=f"{trajectory.name}:{n}"))
    return completions


@dataclass
class FieldWell3D:
    """Field-style 3D well with multiple completions and a surface control."""

    name: str
    well_type: str
    control: str
    target: float
    completions: list[Completion3D]
    min_bhp: float | None = None
    max_bhp: float | None = None
    status: str = "open"

    def __post_init__(self) -> None:
        self.well_type = self.well_type.lower()
        self.control = self.control.lower()
        self.status = self.status.lower()
        if self.well_type not in {"producer", "injector"}:
            raise ValueError("well_type must be 'producer' or 'injector'")
        if self.status not in {"open", "shut"}:
            raise ValueError("well status must be 'open' or 'shut'")
        if self.control not in {"bhp", "water_rate", "oil_rate", "gas_rate", "liquid_rate", "total_rate"}:
            raise ValueError("unsupported well control")

    @property
    def is_open(self) -> bool:
        return self.status == "open" and any(c.is_open for c in self.completions)

    @property
    def cells(self) -> np.ndarray:
        return np.asarray([c.cell for c in self.completions if c.is_open], dtype=int)

    @property
    def total_well_index(self) -> float:
        return float(sum(c.well_index for c in self.completions if c.is_open))

    def completion_table(self, grid) -> list[dict[str, float | int | str]]:
        rows = []
        for c in self.completions:
            i, j, k = grid.unravel_cell(c.cell)
            rows.append({
                "well": self.name,
                "cell": int(c.cell),
                "i": int(i), "j": int(j), "k": int(k),
                "well_index": float(c.well_index),
                "status": c.status,
                "skin": float(c.skin),
                "segment": -1 if c.segment is None else int(c.segment),
                "label": c.label,
            })
        return rows
