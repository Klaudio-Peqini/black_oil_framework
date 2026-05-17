"""Step 6A example: structured 3D mesh, properties and visualization.

This example does not yet run the full 3D black-oil flow solver. Its purpose is
mesh infrastructure: 3D indexing, anisotropic transmissibilities, synthetic rock
properties, sparse stencil preparation and ParaView-compatible VTK export.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from blackoil.grid3d import CartesianGrid3D
from blackoil.properties3d import (
    anisotropic_permeability_3d,
    gaussian_channel_permeability_3d,
    layered_permeability_3d,
    lognormal_permeability_3d,
    porosity_from_permeability_3d,
)
from blackoil.visualization3d import write_vtk_rectilinear_grid_3d
from blackoil.cornerpoint import make_cartesian_cornerpoint_spec
from blackoil.sparse_jacobian import structured_grid_black_oil_sparsity
from blackoil.flux3d import total_transmissibility_count_3d
from blackoil.units import md


def save_slice(path: Path, grid: CartesianGrid3D, values, title: str, layer: int, cmap: str = "viridis"):
    arr = grid.reshape(values)[layer]
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    im = ax.imshow(arr, origin="lower", extent=[0, grid.lx, 0, grid.ly], aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    out = ROOT / "outputs" / "example_11_step6a_3d_mesh"
    out.mkdir(parents=True, exist_ok=True)

    nx, ny, nz = 24, 16, 6
    lx, ly, lz = 720.0, 480.0, 36.0

    # Structural dip: depth increases in x and weakly in y, while layer depth is
    # inherited from the cell-centre z coordinate. Positive depth is downward.
    tmp_grid = CartesianGrid3D(nx, ny, nz, lx, ly, lz, top_depth=1450.0)
    x, y, z = tmp_grid.centers
    depth = 1450.0 + z + 0.035 * x + 0.015 * y
    grid = CartesianGrid3D(nx, ny, nz, lx, ly, lz, top_depth=1450.0, depth=depth)

    # Synthetic property model: vertical layering plus a high-permeability
    # channel and a reproducible stochastic component. Units are converted from
    # mD to m^2 for transmissibility calculations.
    layered_md = layered_permeability_3d(grid, [350, 250, 80, 40, 120, 180], direction="z")
    channel_md = gaussian_channel_permeability_3d(grid, k_background=40.0, k_channel=700.0, width_y_fraction=0.12)
    stochastic_md = lognormal_permeability_3d(grid, geometric_mean=1.0, sigma_log=0.25, seed=42)
    kh_md = np.maximum(5.0, 0.55 * layered_md + 0.45 * channel_md) * stochastic_md
    perm_md = anisotropic_permeability_3d(grid, kh_md, ky_kx=0.65, kvkh=0.04)
    perm_m2 = {name: value * md for name, value in perm_md.items()}
    porosity = porosity_from_permeability_3d(kh_md, phi_min=0.09, phi_max=0.26)
    pore_volume = grid.pore_volume(porosity)

    tx, ty, tz = grid.geometric_transmissibility(perm_m2)
    stencil = structured_grid_black_oil_sparsity(grid, n_components=3)
    cp_spec = make_cartesian_cornerpoint_spec(grid)

    # A synthetic pressure field for visualization and later flow-solver tests.
    pressure_bar = 260.0 - 0.015 * x + 0.08 * (depth - depth.min())
    sw = 0.18 + 0.18 * (x / grid.lx) + 0.04 * np.sin(2.0 * np.pi * y / grid.ly)
    sg = np.clip(0.03 + 0.10 * (depth - depth.min()) / (depth.max() - depth.min()), 0.0, 0.2)

    write_vtk_rectilinear_grid_3d(
        out / "step6a_3d_mesh_state.vtk",
        grid,
        {
            "pressure_bar": pressure_bar,
            "water_saturation": sw,
            "gas_saturation": sg,
            "porosity": porosity,
            "kx_mD": perm_md["kx"],
            "ky_mD": perm_md["ky"],
            "kz_mD": perm_md["kz"],
            "depth_m": grid.depths,
            "pore_volume_m3": pore_volume,
        },
    )

    mid_layer = nz // 2
    save_slice(out / "kx_middle_layer_md.png", grid, perm_md["kx"], "Kx [mD], middle layer", mid_layer)
    save_slice(out / "porosity_middle_layer.png", grid, porosity, "Porosity [-], middle layer", mid_layer)
    save_slice(out / "pressure_middle_layer_bar.png", grid, pressure_bar, "Pressure [bar], middle layer", mid_layer)
    save_slice(out / "gas_saturation_middle_layer.png", grid, sg, "Gas saturation [-], middle layer", mid_layer)

    summary = out / "step6a_mesh_summary.txt"
    summary.write_text(
        "Step 6A structured 3D mesh summary\n"
        "===================================\n"
        f"cells: {grid.n_cells} ({nx} x {ny} x {nz})\n"
        f"cell size: dx={grid.dx:.3f} m, dy={grid.dy:.3f} m, dz={grid.dz:.3f} m\n"
        f"internal faces: {total_transmissibility_count_3d(grid)}\n"
        f"x/y/z transmissibilities: {tx.size}/{ty.size}/{tz.size}\n"
        f"total pore volume: {pore_volume.sum():.6e} m3\n"
        f"sparse black-oil Jacobian pattern: shape={stencil.shape}, nnz={stencil.nnz}\n"
        f"corner-point bridge: coord={cp_spec.coord.shape}, zcorn={cp_spec.zcorn.size}, active={cp_spec.active_mask.sum()}\n"
        f"VTK file: {out / 'step6a_3d_mesh_state.vtk'}\n",
        encoding="utf-8",
    )

    print("Step 6A 3D mesh example finished successfully")
    print(summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
