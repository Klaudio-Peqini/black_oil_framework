"""Step 6B example: 3D property mapping, inactive cells, faults and wells.

This example prepares the 3D reservoir description used by the later full 3D
black-oil simulator. It creates zones, maps rock properties, removes inactive
cells, applies transmissibility multipliers/faults, builds well trajectories and
exports a ParaView VTK file with diagnostic fields.
"""

from pathlib import Path
import sys
import csv

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from blackoil.grid3d import CartesianGrid3D
from blackoil.units import md
from blackoil.properties3d import anisotropic_permeability_3d, gaussian_channel_permeability_3d, porosity_from_permeability_3d
from blackoil.reservoir3d import (
    ReservoirPropertyModel3D,
    active_mask_from_boxes,
    apply_region_multiplier,
    map_property_by_zone,
    zone_from_layers,
)
from blackoil.faults3d import FaultPlane3D, multipliers_from_faults
from blackoil.wells3d import FieldWell3D, WellTrajectory3D, completions_from_trajectory
from blackoil.visualization3d import write_vtk_rectilinear_grid_3d


def save_slice(path: Path, grid: CartesianGrid3D, values, title: str, layer: int, cmap: str = "viridis"):
    arr = grid.reshape(values)[layer]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    im = ax.imshow(arr, origin="lower", extent=[0, grid.lx, 0, grid.ly], aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_completion_csv(path: Path, grid: CartesianGrid3D, wells: list[FieldWell3D]) -> None:
    rows = []
    for well in wells:
        rows.extend(well.completion_table(grid))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["well", "cell", "i", "j", "k", "well_index", "status", "skin", "segment", "label"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    out = ROOT / "outputs" / "example_12_step6b_3d_reservoir_description"
    out.mkdir(parents=True, exist_ok=True)

    nx, ny, nz = 32, 20, 8
    lx, ly, lz = 960.0, 600.0, 64.0
    base = CartesianGrid3D(nx, ny, nz, lx, ly, lz, top_depth=1600.0)
    x, y, z = base.centers
    depth = 1600.0 + z + 0.025 * x - 0.010 * y
    grid = CartesianGrid3D(nx, ny, nz, lx, ly, lz, top_depth=1600.0, depth=depth)

    # Geological zoning and rock properties. Units for permeability are built in
    # mD for readability and converted to m^2 before constructing the model.
    zones = zone_from_layers(grid, [1, 1, 2, 2, 3, 3, 4, 4])
    k_layer_md = map_property_by_zone(grid, zones, {1: 420.0, 2: 160.0, 3: 65.0, 4: 25.0}, name="layer kx")
    channel_md = gaussian_channel_permeability_3d(grid, 20.0, 850.0, y_center_fraction=0.52, z_center_fraction=0.35, width_y_fraction=0.10, width_z_fraction=0.22)
    kh_md = np.maximum(3.0, 0.65 * k_layer_md + 0.35 * channel_md)
    # A local low-permeability baffle across part of the reservoir.
    kh_md = apply_region_multiplier(grid, kh_md, 0.20, box={"i0": 12, "i1": 23, "j0": 8, "j1": 13, "k0": 3, "k1": 6})
    porosity = porosity_from_permeability_3d(kh_md, phi_min=0.08, phi_max=0.27)
    perm_m2 = anisotropic_permeability_3d(grid, kh_md * md, ky_kx=0.72, kvkh=0.035)

    # Inactive cells: a non-reservoir corner and a thin pinched-out streak.
    active_mask = active_mask_from_boxes(
        grid,
        [
            {"i0": 0, "i1": 4, "j0": 0, "j1": 4, "k0": 0, "k1": 8},
            {"i0": 18, "i1": 22, "j0": 0, "j1": 20, "k0": 6, "k1": 8},
        ],
    )
    model = ReservoirPropertyModel3D.from_arrays(grid, porosity=porosity, permeability=perm_m2, active_mask=active_mask, zone_ids=zones)

    # Faults and transmissibility multipliers. F_MAIN is sealing over a vertical
    # panel; F_LEAKY is a partially communicating y-normal fault.
    faults = [
        FaultPlane3D("F_MAIN", axis="x", index=15, multiplier=0.0, j_range=(3, 17), k_range=(1, 7)),
        FaultPlane3D("F_LEAKY", axis="y", index=10, multiplier=0.25, i_range=(8, 30), k_range=(0, 6)),
    ]
    tx, ty, tz = grid.geometric_transmissibility(model.permeability)
    multiplier_model = multipliers_from_faults(grid, faults, active=model.active)
    txm, tym, tzm = multiplier_model.apply_to(tx, ty, tz)
    fault_indicator = multiplier_model.to_cell_indicator()

    # Well trajectories and completions. These objects do not yet perform 3D
    # flow; they are the completion-level representation that Step 7 will use.
    prod_traj = WellTrajectory3D.vertical("PROD_A", x=820.0, y=420.0, z_top=0.0, z_bottom=lz)
    inj_traj = WellTrajectory3D.vertical("INJ_A", x=120.0, y=140.0, z_top=0.0, z_bottom=lz)
    hor_traj = WellTrajectory3D.horizontal_x("PROD_H", x0=460.0, x1=880.0, y=250.0, z=24.0)
    wells = [
        FieldWell3D("PROD_A", "producer", "liquid_rate", -900.0, completions_from_trajectory(grid, prod_traj, model.permeability, orientation="z", active=model.active), min_bhp=120e5),
        FieldWell3D("INJ_A", "injector", "water_rate", 1100.0, completions_from_trajectory(grid, inj_traj, model.permeability, orientation="z", active=model.active), max_bhp=330e5),
        FieldWell3D("PROD_H", "producer", "bhp", 145e5, completions_from_trajectory(grid, hor_traj, model.permeability, orientation="x", active=model.active), min_bhp=115e5),
    ]
    well_cells = np.zeros(grid.n_cells, dtype=float)
    well_index_field = np.zeros(grid.n_cells, dtype=float)
    for n, well in enumerate(wells, start=1):
        for comp in well.completions:
            well_cells[comp.cell] = n
            well_index_field[comp.cell] += comp.well_index

    # Synthetic pressure/saturation state for visualization only.
    pressure_bar = 285.0 - 0.018 * x + 0.065 * (grid.depths - grid.depths.min())
    sw = np.clip(0.18 + 0.32 * x / grid.lx + 0.05 * np.sin(2 * np.pi * y / grid.ly), 0.1, 0.75)
    sg = np.where(grid.depths > np.percentile(grid.depths, 55), 0.08, 0.02)

    write_vtk_rectilinear_grid_3d(
        out / "step6b_3d_reservoir_description.vtk",
        grid,
        {
            "active": model.active.active_mask.astype(float),
            "zone_id": zones.astype(float),
            "fault_indicator": fault_indicator,
            "well_id": well_cells,
            "completion_wi": well_index_field,
            "pressure_bar": pressure_bar,
            "water_saturation": sw,
            "gas_saturation": sg,
            "porosity": model.porosity,
            "pore_volume_m3": model.pore_volume,
            "kx_mD": model.permeability["kx"] / md,
            "ky_mD": model.permeability["ky"] / md,
            "kz_mD": model.permeability["kz"] / md,
            "depth_m": grid.depths,
        },
    )

    write_completion_csv(out / "well_completions.csv", grid, wells)
    mid = nz // 2
    save_slice(out / "active_cells_layer.png", grid, model.active.active_mask.astype(float), "Active cells [-]", mid, cmap="gray")
    save_slice(out / "zone_ids_layer.png", grid, zones, "Zone ID [-]", mid, cmap="tab10")
    save_slice(out / "kx_layer_md.png", grid, model.permeability["kx"] / md, "Kx [mD]", mid)
    save_slice(out / "fault_indicator_layer.png", grid, fault_indicator, "Fault/inactive face indicator [-]", mid, cmap="magma")
    save_slice(out / "well_cells_layer.png", grid, well_cells, "Well cells [-]", mid, cmap="tab10")

    summary = model.summary()
    report = out / "step6b_reservoir_description_summary.txt"
    report.write_text(
        "Step 6B 3D reservoir-description summary\n"
        "========================================\n"
        f"grid: {nx} x {ny} x {nz} = {grid.n_cells} cells\n"
        f"active cells: {summary['n_active']}\n"
        f"inactive cells: {summary['n_inactive']}\n"
        f"total active pore volume: {summary['pore_volume_m3']:.6e} m3\n"
        f"kx range: {summary['kx_min'] / md:.3f} to {summary['kx_max'] / md:.3f} mD\n"
        f"faults: {', '.join(f.name for f in faults)}\n"
        f"nonzero multiplier counts: {multiplier_model.nonzero_counts()}\n"
        f"raw transmissibilities: x={tx.size}, y={ty.size}, z={tz.size}\n"
        f"post-multiplier zeroed faces: x={int(np.count_nonzero(txm == 0.0))}, y={int(np.count_nonzero(tym == 0.0))}, z={int(np.count_nonzero(tzm == 0.0))}\n"
        f"wells: {', '.join(w.name for w in wells)}\n"
        f"total completions: {sum(len(w.completions) for w in wells)}\n"
        f"VTK: {out / 'step6b_3d_reservoir_description.vtk'}\n",
        encoding="utf-8",
    )
    print("Step 6B 3D reservoir-description example finished successfully")
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
