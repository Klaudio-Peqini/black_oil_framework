from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence
import numpy as np

from .reservoir3d import ActiveCellMap3D


@dataclass(frozen=True)
class FaultPlane3D:
    """Structured-grid transmissibility multiplier across a logical plane.

    ``axis='x'`` identifies x-normal faces between ``i=index`` and
    ``i=index+1``. Similarly, ``axis='y'`` acts across faces between
    ``j=index`` and ``j=index+1`` and ``axis='z'`` acts across faces between
    ``k=index`` and ``k=index+1``. Ranges are half-open logical intervals.
    A multiplier of zero represents a sealing fault; values between zero and one
    represent partial transmissibility reduction; values greater than one can be
    used for enhanced communication.
    """

    name: str
    axis: str
    index: int
    multiplier: float
    i_range: tuple[int, int] | None = None
    j_range: tuple[int, int] | None = None
    k_range: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        axis = self.axis.lower()
        if axis not in {"x", "y", "z"}:
            raise ValueError("axis must be 'x', 'y' or 'z'")
        if self.multiplier < 0.0:
            raise ValueError("fault multiplier must be non-negative")
        object.__setattr__(self, "axis", axis)

    def _ranges(self, grid) -> tuple[range, range, range]:
        ir = self.i_range or (0, grid.nx)
        jr = self.j_range or (0, grid.ny)
        kr = self.k_range or (0, grid.nz)
        if not (0 <= ir[0] <= ir[1] <= grid.nx and 0 <= jr[0] <= jr[1] <= grid.ny and 0 <= kr[0] <= kr[1] <= grid.nz):
            raise ValueError(f"fault {self.name!r} has invalid ranges")
        return range(*ir), range(*jr), range(*kr)

    def affected_face_indices(self, grid) -> np.ndarray:
        faces: list[int] = []
        if self.axis == "x":
            if not (0 <= self.index < grid.nx - 1):
                raise ValueError("x fault index must satisfy 0 <= index < nx-1")
            x_pairs = grid.x_face_neighbors
            lookup = {tuple(pair): n for n, pair in enumerate(x_pairs)}
            _, jr, kr = self._ranges(grid)
            for k in kr:
                for j in jr:
                    left = grid.cell_index(self.index, j, k)
                    right = grid.cell_index(self.index + 1, j, k)
                    if (left, right) in lookup:
                        faces.append(lookup[(left, right)])
        elif self.axis == "y":
            if not (0 <= self.index < grid.ny - 1):
                raise ValueError("y fault index must satisfy 0 <= index < ny-1")
            y_pairs = grid.y_face_neighbors
            lookup = {tuple(pair): n for n, pair in enumerate(y_pairs)}
            ir, _, kr = self._ranges(grid)
            for k in kr:
                for i in ir:
                    front = grid.cell_index(i, self.index, k)
                    back = grid.cell_index(i, self.index + 1, k)
                    if (front, back) in lookup:
                        faces.append(lookup[(front, back)])
        else:
            if not (0 <= self.index < grid.nz - 1):
                raise ValueError("z fault index must satisfy 0 <= index < nz-1")
            z_pairs = grid.z_face_neighbors
            lookup = {tuple(pair): n for n, pair in enumerate(z_pairs)}
            ir, jr, _ = self._ranges(grid)
            for j in jr:
                for i in ir:
                    top = grid.cell_index(i, j, self.index)
                    bottom = grid.cell_index(i, j, self.index + 1)
                    if (top, bottom) in lookup:
                        faces.append(lookup[(top, bottom)])
        return np.asarray(faces, dtype=int)


@dataclass
class TransmissibilityMultipliers3D:
    """Face-multiplier model for 3D TPFA transmissibilities."""

    grid: object
    tx_multiplier: np.ndarray = field(init=False)
    ty_multiplier: np.ndarray = field(init=False)
    tz_multiplier: np.ndarray = field(init=False)
    faults: list[FaultPlane3D] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.tx_multiplier = np.ones(len(self.grid.x_face_neighbors), dtype=float)
        self.ty_multiplier = np.ones(len(self.grid.y_face_neighbors), dtype=float)
        self.tz_multiplier = np.ones(len(self.grid.z_face_neighbors), dtype=float)

    def add_fault(self, fault: FaultPlane3D) -> None:
        idx = fault.affected_face_indices(self.grid)
        if fault.axis == "x":
            self.tx_multiplier[idx] *= fault.multiplier
        elif fault.axis == "y":
            self.ty_multiplier[idx] *= fault.multiplier
        else:
            self.tz_multiplier[idx] *= fault.multiplier
        self.faults.append(fault)

    def add_faults(self, faults: Sequence[FaultPlane3D]) -> None:
        for fault in faults:
            self.add_fault(fault)

    def apply_to(self, tx: np.ndarray, ty: np.ndarray, tz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tx = np.asarray(tx, dtype=float); ty = np.asarray(ty, dtype=float); tz = np.asarray(tz, dtype=float)
        if tx.size != self.tx_multiplier.size or ty.size != self.ty_multiplier.size or tz.size != self.tz_multiplier.size:
            raise ValueError("transmissibility arrays are incompatible with multiplier sizes")
        return tx * self.tx_multiplier, ty * self.ty_multiplier, tz * self.tz_multiplier

    def apply_active_mask(self, active: ActiveCellMap3D) -> None:
        """Zero multipliers for faces touching inactive cells."""
        for pairs, mult in (
            (self.grid.x_face_neighbors, self.tx_multiplier),
            (self.grid.y_face_neighbors, self.ty_multiplier),
            (self.grid.z_face_neighbors, self.tz_multiplier),
        ):
            if len(pairs):
                keep = active.active_mask[pairs[:, 0]] & active.active_mask[pairs[:, 1]]
                mult[~keep] = 0.0

    def nonzero_counts(self) -> dict[str, int]:
        return {
            "x": int(np.count_nonzero(self.tx_multiplier)),
            "y": int(np.count_nonzero(self.ty_multiplier)),
            "z": int(np.count_nonzero(self.tz_multiplier)),
        }

    def to_cell_indicator(self) -> np.ndarray:
        """Return cells adjacent to any faulted or inactive-barrier face."""
        indicator = np.zeros(self.grid.n_cells, dtype=float)
        for pairs, mult in (
            (self.grid.x_face_neighbors, self.tx_multiplier),
            (self.grid.y_face_neighbors, self.ty_multiplier),
            (self.grid.z_face_neighbors, self.tz_multiplier),
        ):
            bad = np.flatnonzero(mult < 0.999999)
            if bad.size:
                indicator[pairs[bad, 0]] = 1.0
                indicator[pairs[bad, 1]] = 1.0
        return indicator


def multipliers_from_faults(grid, faults: Sequence[FaultPlane3D], *, active: ActiveCellMap3D | None = None) -> TransmissibilityMultipliers3D:
    model = TransmissibilityMultipliers3D(grid)
    model.add_faults(faults)
    if active is not None:
        model.apply_active_mask(active)
    return model
