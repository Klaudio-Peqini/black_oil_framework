from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np


ArrayLike = float | Sequence[float] | np.ndarray


def as_cell_array_3d(grid, values: ArrayLike, *, name: str = "field", dtype=float) -> np.ndarray:
    """Return a flattened cell-centred array with one value per 3D grid cell.

    Accepted input shapes are scalar, ``(nz, ny, nx)``, or ``(n_cells,)``. The
    returned array follows the grid's native flattened ordering
    ``cell(i,j,k)=k*nx*ny+j*nx+i``.
    """
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim == 0:
        return np.full(grid.n_cells, arr.item(), dtype=dtype)
    if arr.shape == (grid.nz, grid.ny, grid.nx):
        return arr.ravel().astype(dtype, copy=True)
    if arr.shape == (grid.n_cells,):
        return arr.astype(dtype, copy=True)
    raise ValueError(f"{name} must be scalar, shape (nz,ny,nx), or shape (n_cells,)")


@dataclass(frozen=True)
class ActiveCellMap3D:
    """Mapping between full-grid and active-cell indexing.

    A realistic geological grid usually contains cells outside the reservoir,
    pinched-out cells, or cells removed by ACTNUM. This object stores the full
    logical grid but exposes compact active indices for future sparse 3D flow
    assembly.
    """

    grid: object
    active_mask: np.ndarray

    def __post_init__(self) -> None:
        mask = np.asarray(self.active_mask, dtype=bool).ravel()
        if mask.size != self.grid.n_cells:
            raise ValueError("active_mask must contain one value per grid cell")
        object.__setattr__(self, "active_mask", mask)

    @property
    def active_cells(self) -> np.ndarray:
        return np.flatnonzero(self.active_mask)

    @property
    def inactive_cells(self) -> np.ndarray:
        return np.flatnonzero(~self.active_mask)

    @property
    def n_active(self) -> int:
        return int(np.count_nonzero(self.active_mask))

    @property
    def full_to_active(self) -> np.ndarray:
        mapping = -np.ones(self.grid.n_cells, dtype=int)
        mapping[self.active_cells] = np.arange(self.n_active, dtype=int)
        return mapping

    def compress(self, values: ArrayLike, *, name: str = "field") -> np.ndarray:
        arr = as_cell_array_3d(self.grid, values, name=name)
        return arr[self.active_mask]

    def expand(self, active_values: ArrayLike, *, fill_value: float = np.nan) -> np.ndarray:
        arr = np.asarray(active_values, dtype=float).ravel()
        if arr.size != self.n_active:
            raise ValueError("active_values must contain one value per active cell")
        full = np.full(self.grid.n_cells, fill_value, dtype=float)
        full[self.active_mask] = arr
        return full

    def active_neighbor_pairs(self, face_neighbors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return active-only neighbor pairs and the original face indices."""
        pairs = np.asarray(face_neighbors, dtype=int)
        if pairs.size == 0:
            return np.empty((0, 2), dtype=int), np.empty(0, dtype=int)
        keep = self.active_mask[pairs[:, 0]] & self.active_mask[pairs[:, 1]]
        return pairs[keep], np.flatnonzero(keep)


def active_mask_from_indices(grid, inactive_indices: Sequence[int] | None = None) -> np.ndarray:
    """Build an active mask from a list of inactive flat-cell indices."""
    mask = np.ones(grid.n_cells, dtype=bool)
    if inactive_indices is not None:
        idx = np.asarray(inactive_indices, dtype=int).ravel()
        if np.any((idx < 0) | (idx >= grid.n_cells)):
            raise IndexError("inactive cell index out of range")
        mask[idx] = False
    return mask


def active_mask_from_boxes(grid, boxes: Sequence[Mapping[str, int]]) -> np.ndarray:
    """Build an active mask by deactivating logical boxes.

    Each box may contain ``i0,i1,j0,j1,k0,k1``. Upper bounds are exclusive. Any
    omitted lower bound defaults to 0 and any omitted upper bound defaults to the
    corresponding grid dimension.
    """
    mask = np.ones((grid.nz, grid.ny, grid.nx), dtype=bool)
    for box in boxes:
        i0 = int(box.get("i0", 0)); i1 = int(box.get("i1", grid.nx))
        j0 = int(box.get("j0", 0)); j1 = int(box.get("j1", grid.ny))
        k0 = int(box.get("k0", 0)); k1 = int(box.get("k1", grid.nz))
        if not (0 <= i0 <= i1 <= grid.nx and 0 <= j0 <= j1 <= grid.ny and 0 <= k0 <= k1 <= grid.nz):
            raise ValueError("inactive box bounds are invalid")
        mask[k0:k1, j0:j1, i0:i1] = False
    return mask.ravel()


def zone_from_layers(grid, layer_ids: Sequence[int]) -> np.ndarray:
    """Assign integer zone IDs by vertical layer."""
    ids = np.asarray(layer_ids, dtype=int).ravel()
    if ids.size != grid.nz:
        raise ValueError("layer_ids must contain one zone ID per vertical layer")
    zones = np.empty((grid.nz, grid.ny, grid.nx), dtype=int)
    for k, zid in enumerate(ids):
        zones[k, :, :] = zid
    return zones.ravel()


def zone_from_depth_intervals(grid, intervals: Sequence[tuple[float, float, int]], *, default: int = 0) -> np.ndarray:
    """Assign integer zone IDs from positive-downward depth intervals.

    Each interval is ``(top_depth, bottom_depth, zone_id)`` and follows
    ``top_depth <= depth < bottom_depth``. Later intervals overwrite earlier
    ones, which is convenient for local refinements.
    """
    depth = grid.depths
    zones = np.full(grid.n_cells, int(default), dtype=int)
    for top, bottom, zid in intervals:
        if bottom < top:
            raise ValueError("depth interval bottom must be >= top")
        zones[(depth >= float(top)) & (depth < float(bottom))] = int(zid)
    return zones


def map_property_by_zone(grid, zone_ids: ArrayLike, values_by_zone: Mapping[int, float], *, default: float | None = None, name: str = "property") -> np.ndarray:
    """Map a dictionary of zone values to a cell-centred property array."""
    zones = as_cell_array_3d(grid, zone_ids, name="zone_ids", dtype=int)
    out = np.empty(grid.n_cells, dtype=float)
    if default is None:
        missing = sorted(set(np.unique(zones).tolist()) - set(int(k) for k in values_by_zone))
        if missing:
            raise ValueError(f"No {name} value provided for zone(s): {missing}")
        out.fill(np.nan)
    else:
        out.fill(float(default))
    for zid, value in values_by_zone.items():
        out[zones == int(zid)] = float(value)
    if np.any(~np.isfinite(out)):
        raise ValueError(f"{name} mapping produced non-finite values")
    return out


def apply_region_multiplier(grid, values: ArrayLike, multiplier: float, *, box: Mapping[str, int]) -> np.ndarray:
    """Apply a multiplicative factor to a logical box in a 3D cell field."""
    if multiplier < 0.0:
        raise ValueError("multiplier must be non-negative")
    arr = as_cell_array_3d(grid, values, name="values").reshape((grid.nz, grid.ny, grid.nx))
    i0 = int(box.get("i0", 0)); i1 = int(box.get("i1", grid.nx))
    j0 = int(box.get("j0", 0)); j1 = int(box.get("j1", grid.ny))
    k0 = int(box.get("k0", 0)); k1 = int(box.get("k1", grid.nz))
    if not (0 <= i0 <= i1 <= grid.nx and 0 <= j0 <= j1 <= grid.ny and 0 <= k0 <= k1 <= grid.nz):
        raise ValueError("box bounds are invalid")
    out = arr.copy()
    out[k0:k1, j0:j1, i0:i1] *= float(multiplier)
    return out.ravel()


@dataclass(frozen=True)
class ReservoirPropertyModel3D:
    """Container for 3D reservoir property arrays and ACTNUM-style activity."""

    grid: object
    porosity: np.ndarray
    permeability: dict[str, np.ndarray]
    active: ActiveCellMap3D
    zone_ids: np.ndarray | None = None
    net_to_gross: np.ndarray | None = None

    @classmethod
    def from_arrays(
        cls,
        grid,
        *,
        porosity: ArrayLike,
        permeability: Mapping[str, ArrayLike],
        active_mask: ArrayLike | None = None,
        zone_ids: ArrayLike | None = None,
        net_to_gross: ArrayLike | None = None,
    ) -> "ReservoirPropertyModel3D":
        phi = as_cell_array_3d(grid, porosity, name="porosity")
        if np.any((phi <= 0.0) | (phi >= 1.0)):
            raise ValueError("porosity must lie in the open interval (0,1)")
        perm = {}
        for key in ("kx", "ky", "kz"):
            if key not in permeability:
                raise ValueError("permeability must contain kx, ky and kz")
            arr = as_cell_array_3d(grid, permeability[key], name=key)
            if np.any(arr <= 0.0):
                raise ValueError(f"{key} must be positive")
            perm[key] = arr
        if active_mask is None:
            active_bool = np.ones(grid.n_cells, dtype=bool)
        else:
            active_bool = as_cell_array_3d(grid, active_mask, name="active_mask", dtype=bool).astype(bool)
        zones = None if zone_ids is None else as_cell_array_3d(grid, zone_ids, name="zone_ids", dtype=int)
        ntg = None if net_to_gross is None else as_cell_array_3d(grid, net_to_gross, name="net_to_gross")
        if ntg is not None and np.any((ntg < 0.0) | (ntg > 1.0)):
            raise ValueError("net_to_gross must lie in [0,1]")
        return cls(grid=grid, porosity=phi, permeability=perm, active=ActiveCellMap3D(grid, active_bool), zone_ids=zones, net_to_gross=ntg)

    @property
    def pore_volume(self) -> np.ndarray:
        ntg = 1.0 if self.net_to_gross is None else self.net_to_gross
        pv = self.grid.volumes * self.porosity * ntg
        return np.where(self.active.active_mask, pv, 0.0)

    def active_property(self, name: str) -> np.ndarray:
        if name == "porosity":
            return self.active.compress(self.porosity, name=name)
        if name == "pore_volume":
            return self.active.compress(self.pore_volume, name=name)
        if name in self.permeability:
            return self.active.compress(self.permeability[name], name=name)
        if name == "zone_ids" and self.zone_ids is not None:
            return self.zone_ids[self.active.active_mask]
        raise KeyError(f"unknown property {name!r}")

    def summary(self) -> dict[str, float | int]:
        return {
            "n_cells": int(self.grid.n_cells),
            "n_active": int(self.active.n_active),
            "n_inactive": int(self.grid.n_cells - self.active.n_active),
            "pore_volume_m3": float(np.sum(self.pore_volume)),
            "phi_min": float(np.min(self.porosity[self.active.active_mask])),
            "phi_max": float(np.max(self.porosity[self.active.active_mask])),
            "kx_min": float(np.min(self.permeability["kx"][self.active.active_mask])),
            "kx_max": float(np.max(self.permeability["kx"][self.active.active_mask])),
        }
