from __future__ import annotations

from pathlib import Path
import numpy as np


def write_vtk_structured_grid_2d(
    path,
    grid,
    cell_data: dict[str, np.ndarray],
) -> Path:
    """Write a simple legacy VTK rectilinear-grid file for ParaView.

    The areal 2D reservoir is exported as a single-layer rectilinear grid with
    cell-centred data. This is deliberately dependency-free; PyVista support can
    later be added on top of the same cell-data dictionary.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x_coords = np.linspace(0.0, grid.lx, grid.nx + 1)
    y_coords = np.linspace(0.0, grid.ly, grid.ny + 1)
    z_coords = np.array([0.0, grid.thickness], dtype=float)
    n_cells = grid.n_cells

    with path.open("w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("blackoil-framework Step 5C 2D output\n")
        f.write("ASCII\n")
        f.write("DATASET RECTILINEAR_GRID\n")
        f.write(f"DIMENSIONS {grid.nx + 1} {grid.ny + 1} 2\n")
        f.write(f"X_COORDINATES {grid.nx + 1} float\n")
        f.write(" ".join(f"{v:.9e}" for v in x_coords) + "\n")
        f.write(f"Y_COORDINATES {grid.ny + 1} float\n")
        f.write(" ".join(f"{v:.9e}" for v in y_coords) + "\n")
        f.write("Z_COORDINATES 2 float\n")
        f.write(" ".join(f"{v:.9e}" for v in z_coords) + "\n")
        f.write(f"CELL_DATA {n_cells}\n")
        for name, values in cell_data.items():
            arr = np.asarray(values, dtype=float).ravel()
            if arr.size != n_cells:
                raise ValueError(f"cell-data field {name!r} has incompatible size")
            safe_name = name.replace(" ", "_")
            f.write(f"SCALARS {safe_name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for value in arr:
                f.write(f"{value:.9e}\n")
    return path
