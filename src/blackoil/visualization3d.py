from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence
import html
import json

import numpy as np


ArrayLike = np.ndarray | Sequence[float] | float


def _as_cell_array(grid, values: ArrayLike, *, name: str = "field") -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.full(grid.n_cells, float(arr), dtype=float)
    if arr.shape == (grid.nz, grid.ny, grid.nx):
        return arr.ravel()
    if arr.size == grid.n_cells:
        return arr.reshape(-1).astype(float, copy=True)
    raise ValueError(f"{name!r} must be scalar, shape (nz,ny,nx), or size n_cells")


def _sanitize_vtk_name(name: str) -> str:
    out = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(name).strip())
    return out or "field"


def write_vtk_rectilinear_grid_3d(path, grid, cell_data: Mapping[str, ArrayLike]) -> Path:
    """Write a legacy VTK rectilinear-grid file for ParaView.

    This dependency-free writer exports cell-centred scalar fields on a
    structured 3D Cartesian grid. The output can be opened directly in ParaView.
    The writer intentionally targets the legacy ASCII VTK format because it is
    easy to inspect, robust on clusters, and does not require VTK/PyVista.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x_coords = np.linspace(0.0, grid.lx, grid.nx + 1)
    y_coords = np.linspace(0.0, grid.ly, grid.ny + 1)
    z_coords = np.linspace(0.0, grid.lz, grid.nz + 1)

    with path.open("w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("blackoil-framework 3D rectilinear-grid output\n")
        f.write("ASCII\n")
        f.write("DATASET RECTILINEAR_GRID\n")
        f.write(f"DIMENSIONS {grid.nx + 1} {grid.ny + 1} {grid.nz + 1}\n")
        f.write(f"X_COORDINATES {grid.nx + 1} float\n")
        f.write(" ".join(f"{v:.9e}" for v in x_coords) + "\n")
        f.write(f"Y_COORDINATES {grid.ny + 1} float\n")
        f.write(" ".join(f"{v:.9e}" for v in y_coords) + "\n")
        f.write(f"Z_COORDINATES {grid.nz + 1} float\n")
        f.write(" ".join(f"{v:.9e}" for v in z_coords) + "\n")
        f.write(f"CELL_DATA {grid.n_cells}\n")
        for name, values in cell_data.items():
            arr = _as_cell_array(grid, values, name=name)
            safe_name = _sanitize_vtk_name(name)
            f.write(f"SCALARS {safe_name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for value in arr:
                f.write(f"{value:.9e}\n")
    return path


def write_vtk_structured_points_3d(path, grid, cell_data: Mapping[str, ArrayLike]) -> Path:
    """Alias kept for users who think of the Cartesian grid as image data."""
    return write_vtk_rectilinear_grid_3d(path, grid, cell_data)


def write_pvd_collection(path, datasets: Sequence[tuple[float, str | Path]]) -> Path:
    """Write a ParaView ``.pvd`` collection file for a VTK time series.

    Parameters
    ----------
    path:
        Destination ``.pvd`` file.
    datasets:
        Sequence of ``(time, file_path)`` pairs. File paths are written relative
        to the directory containing the PVD file when possible.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>',
    ]
    for time, file_path in datasets:
        fp = Path(file_path)
        try:
            rel = fp.relative_to(path.parent)
        except ValueError:
            rel = fp
        lines.append(f'    <DataSet timestep="{float(time):.12g}" group="" part="0" file="{html.escape(str(rel))}"/>')
    lines.extend(['  </Collection>', '</VTKFile>', ''])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@dataclass
class VTKTimeSeriesWriter3D:
    """Small helper for writing ParaView-readable 3D state sequences.

    Each call to :meth:`write_state` writes one legacy ``.vtk`` rectilinear-grid
    file. Calling :meth:`write_pvd` creates the collection file that ParaView can
    open as a single time-dependent dataset.
    """

    output_dir: str | Path
    grid: object
    prefix: str = "state"
    digits: int = 5
    datasets: list[tuple[float, Path]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_state(self, time: float, cell_data: Mapping[str, ArrayLike], *, step: int | None = None) -> Path:
        if step is None:
            step = len(self.datasets)
        file_path = self.output_dir / f"{self.prefix}_{int(step):0{self.digits}d}.vtk"
        write_vtk_rectilinear_grid_3d(file_path, self.grid, cell_data)
        self.datasets.append((float(time), file_path))
        return file_path

    def write_pvd(self, filename: str | Path | None = None) -> Path:
        if filename is None:
            filename = f"{self.prefix}_series.pvd"
        path = Path(filename)
        if not path.is_absolute():
            path = self.output_dir / path
        return write_pvd_collection(path, self.datasets)

    def write_manifest(self, filename: str | Path = "vtk_time_series_manifest.json") -> Path:
        path = Path(filename)
        if not path.is_absolute():
            path = self.output_dir / path
        payload = {
            "prefix": self.prefix,
            "n_steps": len(self.datasets),
            "datasets": [{"time": t, "file": str(p)} for t, p in self.datasets],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


def save_state_time_series_npz(path, times: Sequence[float], states: Mapping[str, Sequence[ArrayLike] | np.ndarray]) -> Path:
    """Save a compact NumPy archive containing scalar-field time series."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"times": np.asarray(times, dtype=float)}
    for name, values in states.items():
        arrays[_sanitize_vtk_name(name)] = np.asarray(values, dtype=float)
    np.savez(path, **arrays)
    return path


def cell_center_coordinates(grid, cells: Sequence[int] | np.ndarray | None = None) -> np.ndarray:
    """Return cell-centre coordinates as an ``(n,3)`` array."""
    x, y, z = grid.centers
    coords = np.column_stack([x, y, z])
    if cells is None:
        return coords
    return coords[np.asarray(cells, dtype=int)]


def build_completion_cell_fields(grid, wells) -> dict[str, np.ndarray]:
    """Build cell-centred fields that identify well completions.

    Returned fields are useful both for VTK export and diagnostic maps:
    ``well_id``, ``completion_count`` and ``completion_wi``.
    """
    well_id = np.zeros(grid.n_cells, dtype=float)
    completion_count = np.zeros(grid.n_cells, dtype=float)
    completion_wi = np.zeros(grid.n_cells, dtype=float)
    for wid, well in enumerate(wells, start=1):
        for comp in well.completions:
            cell = int(comp.cell)
            well_id[cell] = float(wid)
            completion_count[cell] += 1.0
            completion_wi[cell] += float(comp.well_index)
    return {"well_id": well_id, "completion_count": completion_count, "completion_wi": completion_wi}


def write_vtk_well_trajectories_3d(path, trajectories: Sequence[object], *, well_ids: Sequence[int] | None = None) -> Path:
    """Write well trajectories as legacy VTK ``POLYDATA`` lines.

    ``trajectories`` are expected to expose ``name`` and ``points`` attributes,
    as provided by :class:`blackoil.wells3d.WellTrajectory3D`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if well_ids is None:
        well_ids = list(range(1, len(trajectories) + 1))
    if len(well_ids) != len(trajectories):
        raise ValueError("well_ids must have the same length as trajectories")

    all_points: list[np.ndarray] = []
    line_records: list[tuple[int, int, int]] = []  # start, n_points, well_id
    offset = 0
    for trajectory, wid in zip(trajectories, well_ids):
        pts = np.asarray(trajectory.points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 2:
            raise ValueError("each trajectory must contain an (n>=2,3) point array")
        all_points.append(pts)
        line_records.append((offset, pts.shape[0], int(wid)))
        offset += pts.shape[0]
    stacked = np.vstack(all_points) if all_points else np.empty((0, 3), dtype=float)

    with path.open("w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("blackoil-framework 3D well trajectories\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {stacked.shape[0]} float\n")
        for x, y, z in stacked:
            f.write(f"{x:.9e} {y:.9e} {z:.9e}\n")
        total_ints = sum(n + 1 for _, n, _ in line_records)
        f.write(f"LINES {len(line_records)} {total_ints}\n")
        for start, n, _wid in line_records:
            ids = " ".join(str(i) for i in range(start, start + n))
            f.write(f"{n} {ids}\n")
        f.write(f"CELL_DATA {len(line_records)}\n")
        f.write("SCALARS well_id int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for _start, _n, wid in line_records:
            f.write(f"{wid}\n")
    return path


def write_vtk_completion_points_3d(path, grid, wells) -> Path:
    """Write completion locations as legacy VTK ``POLYDATA`` vertices."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, float | int]] = []
    coords_all = cell_center_coordinates(grid)
    for wid, well in enumerate(wells, start=1):
        for cid, comp in enumerate(well.completions):
            cell = int(comp.cell)
            i, j, k = grid.unravel_cell(cell)
            records.append({
                "well_id": wid,
                "completion_id": cid,
                "cell": cell,
                "i": i,
                "j": j,
                "k": k,
                "well_index": float(comp.well_index),
                "is_open": 1 if comp.is_open else 0,
                "x": float(coords_all[cell, 0]),
                "y": float(coords_all[cell, 1]),
                "z": float(coords_all[cell, 2]),
            })
    points = np.asarray([[r["x"], r["y"], r["z"]] for r in records], dtype=float) if records else np.empty((0, 3))
    with path.open("w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("blackoil-framework 3D completion points\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {points.shape[0]} float\n")
        for x, y, z in points:
            f.write(f"{x:.9e} {y:.9e} {z:.9e}\n")
        f.write(f"VERTICES {points.shape[0]} {2 * points.shape[0]}\n")
        for n in range(points.shape[0]):
            f.write(f"1 {n}\n")
        f.write(f"POINT_DATA {points.shape[0]}\n")
        for name in ["well_id", "completion_id", "cell", "i", "j", "k", "well_index", "is_open"]:
            dtype = "float" if name == "well_index" else "int"
            f.write(f"SCALARS {name} {dtype} 1\n")
            f.write("LOOKUP_TABLE default\n")
            for rec in records:
                f.write(f"{rec[name]}\n")
    return path


def state_statistics(grid, cell_data: Mapping[str, ArrayLike], *, pore_volume: ArrayLike | None = None) -> dict[str, float]:
    """Return min/max/mean and optional PV-weighted mean for scalar fields."""
    stats: dict[str, float] = {}
    weights = None if pore_volume is None else _as_cell_array(grid, pore_volume, name="pore_volume")
    if weights is not None and np.sum(weights) <= 0.0:
        weights = None
    for name, values in cell_data.items():
        arr = _as_cell_array(grid, values, name=name)
        safe = _sanitize_vtk_name(name)
        stats[f"{safe}_min"] = float(np.nanmin(arr))
        stats[f"{safe}_max"] = float(np.nanmax(arr))
        stats[f"{safe}_mean"] = float(np.nanmean(arr))
        if weights is not None:
            stats[f"{safe}_pvmean"] = float(np.sum(arr * weights) / np.sum(weights))
    return stats


def to_pyvista_rectilinear_grid(grid, cell_data: Mapping[str, ArrayLike] | None = None):
    """Return a PyVista RectilinearGrid when PyVista is installed.

    The function is optional by design. Importing :mod:`blackoil` never requires
    PyVista, but users who install ``blackoil-framework[visualization]`` can use
    this adapter for interactive notebooks, screenshots, clipping, streamlines,
    and modern VTK XML export.
    """
    try:
        import pyvista as pv  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise ImportError("PyVista is optional. Install with `pip install pyvista` or `pip install -e .[visualization]`.") from exc
    x_coords = np.linspace(0.0, grid.lx, grid.nx + 1)
    y_coords = np.linspace(0.0, grid.ly, grid.ny + 1)
    z_coords = np.linspace(0.0, grid.lz, grid.nz + 1)
    pv_grid = pv.RectilinearGrid(x_coords, y_coords, z_coords)
    if cell_data:
        for name, values in cell_data.items():
            pv_grid.cell_data[_sanitize_vtk_name(name)] = _as_cell_array(grid, values, name=name)
    return pv_grid
