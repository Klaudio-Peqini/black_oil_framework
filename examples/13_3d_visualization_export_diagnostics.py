"""Step 6C example: 3D visualization/export and diagnostics.

The example starts from the 3D reservoir-description objects introduced in Step
6B and creates a synthetic time sequence of black-oil state variables. Its goal
is not to solve the full 3D flow problem yet; it demonstrates the export and
visualization layer that the final 3D simulator will use:

* ParaView legacy VTK state files for each report time;
* a ParaView PVD collection file for time-series loading;
* well trajectory and completion-point VTK exports;
* compact NPZ time-series storage;
* diagnostic layer maps, well trajectory projections, histograms, and state
  history plots.
"""

from pathlib import Path
import csv
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from blackoil.grid3d import CartesianGrid3D
from blackoil.units import md, day
from blackoil.properties3d import anisotropic_permeability_3d, gaussian_channel_permeability_3d, porosity_from_permeability_3d
from blackoil.reservoir3d import ReservoirPropertyModel3D, active_mask_from_boxes, map_property_by_zone, zone_from_layers
from blackoil.faults3d import FaultPlane3D, multipliers_from_faults
from blackoil.wells3d import FieldWell3D, WellTrajectory3D, completions_from_trajectory
from blackoil.visualization3d import (
    VTKTimeSeriesWriter3D,
    build_completion_cell_fields,
    save_state_time_series_npz,
    state_statistics,
    write_vtk_completion_points_3d,
    write_vtk_rectilinear_grid_3d,
    write_vtk_well_trajectories_3d,
)
from blackoil.diagnostics3d import (
    plot_layer_map_3d,
    plot_pore_volume_by_zone,
    plot_property_histograms,
    plot_state_time_series,
    plot_well_trajectories_3d,
)


def write_diagnostics_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    out = ROOT / "outputs" / "example_13_step6c_3d_visualization"
    vtk_dir = out / "vtk_timeseries"
    out.mkdir(parents=True, exist_ok=True)
    vtk_dir.mkdir(parents=True, exist_ok=True)

    # A moderately sized structured 3D grid. This remains small enough for fast
    # tests and examples but large enough to show layers, faults, and well paths.
    nx, ny, nz = 30, 18, 7
    lx, ly, lz = 900.0, 540.0, 56.0
    base = CartesianGrid3D(nx, ny, nz, lx, ly, lz, top_depth=1500.0)
    x, y, z = base.centers
    depth = 1500.0 + z + 0.018 * x - 0.012 * y
    grid = CartesianGrid3D(nx, ny, nz, lx, ly, lz, top_depth=1500.0, depth=depth)

    zones = zone_from_layers(grid, [1, 1, 2, 2, 3, 3, 4])
    background_md = map_property_by_zone(grid, zones, {1: 360.0, 2: 180.0, 3: 75.0, 4: 30.0}, name="kx_md")
    channel_md = gaussian_channel_permeability_3d(grid, 25.0, 780.0, y_center_fraction=0.55, z_center_fraction=0.42, width_y_fraction=0.12, width_z_fraction=0.25)
    kx_md = 0.72 * background_md + 0.28 * channel_md
    porosity = porosity_from_permeability_3d(kx_md, phi_min=0.075, phi_max=0.265)
    perm = anisotropic_permeability_3d(grid, kx_md * md, ky_kx=0.68, kvkh=0.04)
    active_mask = active_mask_from_boxes(
        grid,
        [
            {"i0": 0, "i1": 4, "j0": 0, "j1": 4, "k0": 0, "k1": nz},
            {"i0": 20, "i1": 24, "j0": 0, "j1": ny, "k0": 5, "k1": nz},
        ],
    )
    model = ReservoirPropertyModel3D.from_arrays(grid, porosity=porosity, permeability=perm, active_mask=active_mask, zone_ids=zones)

    faults = [
        FaultPlane3D("F_MAIN", axis="x", index=14, multiplier=0.0, j_range=(3, 16), k_range=(1, 6)),
        FaultPlane3D("F_LEAKY", axis="y", index=9, multiplier=0.35, i_range=(7, 28), k_range=(0, 6)),
    ]
    multiplier_model = multipliers_from_faults(grid, faults, active=model.active)
    fault_indicator = multiplier_model.to_cell_indicator()

    trajectories = [
        WellTrajectory3D.vertical("INJ_W", x=105.0, y=120.0, z_top=0.0, z_bottom=lz),
        WellTrajectory3D.vertical("PROD_E", x=790.0, y=410.0, z_top=0.0, z_bottom=lz),
        WellTrajectory3D.horizontal_x("PROD_H", x0=470.0, x1=830.0, y=245.0, z=28.0),
    ]
    wells = [
        FieldWell3D("INJ_W", "injector", "water_rate", 1400.0, completions_from_trajectory(grid, trajectories[0], model.permeability, orientation="z", active=model.active), max_bhp=340e5),
        FieldWell3D("PROD_E", "producer", "liquid_rate", -950.0, completions_from_trajectory(grid, trajectories[1], model.permeability, orientation="z", active=model.active), min_bhp=120e5),
        FieldWell3D("PROD_H", "producer", "bhp", 145e5, completions_from_trajectory(grid, trajectories[2], model.permeability, orientation="x", active=model.active), min_bhp=115e5),
    ]
    completion_fields = build_completion_cell_fields(grid, wells)

    # Static VTK files: one cell-centred reservoir description, one line file for
    # trajectories, one point file for completions. These can be loaded together
    # in ParaView and overlaid with the time-dependent states.
    static_fields = {
        "active": model.active.active_mask.astype(float),
        "zone_id": zones.astype(float),
        "fault_indicator": fault_indicator,
        "porosity": model.porosity,
        "pore_volume_m3": model.pore_volume,
        "kx_mD": model.permeability["kx"] / md,
        "ky_mD": model.permeability["ky"] / md,
        "kz_mD": model.permeability["kz"] / md,
        "depth_m": grid.depths,
        **completion_fields,
    }
    write_vtk_rectilinear_grid_3d(out / "reservoir_static_description.vtk", grid, static_fields)
    write_vtk_well_trajectories_3d(out / "well_trajectories.vtk", trajectories)
    write_vtk_completion_points_3d(out / "well_completions_points.vtk", grid, wells)

    # Synthetic state sequence that mimics pressure depletion, water invasion
    # from the injector side, and gas liberation in the deeper/lower-pressure
    # region. The final 3D simulator will replace this block with real states.
    times_days = np.asarray([0.0, 60.0, 180.0, 360.0, 720.0, 1080.0])
    writer = VTKTimeSeriesWriter3D(vtk_dir, grid, prefix="blackoil_state")
    state_archive: dict[str, list[np.ndarray]] = {"pressure_bar": [], "water_saturation": [], "gas_saturation": [], "oil_saturation": [], "Rs_sm3_sm3": []}
    rows: list[dict[str, float]] = []

    xnorm = x / grid.lx
    ynorm = y / grid.ly
    depth_norm = (grid.depths - grid.depths.min()) / max(np.ptp(grid.depths), 1.0)
    for step, tday in enumerate(times_days):
        tau = tday / max(times_days[-1], 1.0)
        injector_plume = np.exp(-((xnorm - 0.12) ** 2 / 0.030 + (ynorm - 0.22) ** 2 / 0.055))
        producer_drawdown = np.exp(-((xnorm - 0.88) ** 2 / 0.050 + (ynorm - 0.75) ** 2 / 0.055))
        pressure_bar = 292.0 - 28.0 * tau * producer_drawdown + 7.5 * tau * injector_plume + 0.055 * (grid.depths - grid.depths.min())
        sw = np.clip(0.18 + 0.34 * tau * injector_plume + 0.06 * xnorm, 0.10, 0.82)
        sg = np.clip(0.015 + 0.16 * tau * producer_drawdown * (0.35 + 0.65 * depth_norm), 0.0, 0.28)
        so = np.clip(1.0 - sw - sg, 0.0, 1.0)
        rs = np.clip(92.0 - 32.0 * tau * producer_drawdown + 5.0 * depth_norm, 30.0, 120.0)
        # Inactive cells are exported too, but flagged; set their dynamic state
        # fields to NaN-like sentinel values that ParaView can threshold out.
        inactive = ~model.active.active_mask
        pressure_bar = pressure_bar.copy(); sw = sw.copy(); sg = sg.copy(); so = so.copy(); rs = rs.copy()
        pressure_bar[inactive] = 0.0; sw[inactive] = 0.0; sg[inactive] = 0.0; so[inactive] = 0.0; rs[inactive] = 0.0

        cell_data = {
            **static_fields,
            "pressure_bar": pressure_bar,
            "water_saturation": sw,
            "gas_saturation": sg,
            "oil_saturation": so,
            "Rs_sm3_sm3": rs,
        }
        writer.write_state(tday * day, cell_data, step=step)
        for key in state_archive:
            state_archive[key].append(np.asarray(cell_data[key], dtype=float))
        stats = state_statistics(grid, {"pressure_bar": pressure_bar, "water_saturation": sw, "gas_saturation": sg, "oil_saturation": so}, pore_volume=model.pore_volume)
        rows.append({"time_days": float(tday), **stats})

    pvd_path = writer.write_pvd("blackoil_state_series.pvd")
    writer.write_manifest()
    save_state_time_series_npz(out / "blackoil_state_timeseries.npz", times_days, state_archive)
    write_diagnostics_csv(out / "state_diagnostics.csv", rows)

    final = {key: values[-1] for key, values in state_archive.items()}
    mid = nz // 2
    plot_layer_map_3d(out / "final_pressure_layer.png", grid, final["pressure_bar"], layer=mid, title="Final pressure", label="pressure [bar]", wells=wells, fault_indicator=fault_indicator)
    plot_layer_map_3d(out / "final_water_saturation_layer.png", grid, final["water_saturation"], layer=mid, title="Final water saturation", label="Sw [-]", wells=wells)
    plot_layer_map_3d(out / "final_gas_saturation_layer.png", grid, final["gas_saturation"], layer=mid, title="Final gas saturation", label="Sg [-]", wells=wells)
    plot_layer_map_3d(out / "static_kx_layer.png", grid, static_fields["kx_mD"], layer=mid, title="Kx", label="Kx [mD]", wells=wells, fault_indicator=fault_indicator)
    plot_well_trajectories_3d(out / "well_trajectories_projection.png", grid, trajectories, wells=wells)
    plot_property_histograms(out / "property_histograms.png", {"Kx [mD]": static_fields["kx_mD"][model.active.active_mask], "porosity [-]": model.porosity[model.active.active_mask], "pore volume [m3]": model.pore_volume[model.active.active_mask]})
    plot_pore_volume_by_zone(out / "pore_volume_by_zone.png", zones, model.pore_volume)
    history = {
        "time_days": times_days,
        "pressure_pvmean_bar": [r["pressure_bar_pvmean"] for r in rows],
        "Sw_pvmean": [r["water_saturation_pvmean"] for r in rows],
        "Sg_pvmean": [r["gas_saturation_pvmean"] for r in rows],
        "So_pvmean": [r["oil_saturation_pvmean"] for r in rows],
    }
    plot_state_time_series(out / "state_history_diagnostics.png", history, title="PV-weighted 3D state diagnostics")

    report = out / "step6c_visualization_summary.txt"
    report.write_text(
        "Step 6C 3D visualization/export summary\n"
        "=======================================\n"
        f"grid: {nx} x {ny} x {nz} = {grid.n_cells} cells\n"
        f"active cells: {model.active.n_active}\n"
        f"time steps exported: {len(times_days)}\n"
        f"PVD collection: {pvd_path}\n"
        f"static VTK: {out / 'reservoir_static_description.vtk'}\n"
        f"trajectory VTK: {out / 'well_trajectories.vtk'}\n"
        f"completion VTK: {out / 'well_completions_points.vtk'}\n"
        f"total completions: {sum(len(w.completions) for w in wells)}\n"
        f"diagnostics CSV: {out / 'state_diagnostics.csv'}\n"
        f"NPZ archive: {out / 'blackoil_state_timeseries.npz'}\n",
        encoding="utf-8",
    )
    print("Step 6C 3D visualization/export example finished successfully")
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
