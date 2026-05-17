"""Step 7 example: integrated full 3D black-oil reservoir simulation.

This example exercises the final integrated framework layer:

* 3D structured Cartesian grid;
* heterogeneous anisotropic rock properties;
* inactive ACTNUM-like cells;
* fault/transmissibility multipliers;
* live-oil phase-state switching with solution gas Rs;
* gravity and capillary pressure;
* multi-completion 3D wells;
* field-style schedule changes;
* sparse Newton/GMRES/ILU solve;
* restart files;
* ParaView VTK/PVD export and diagnostic plots.

The case is deliberately small so it can run quickly on a laptop while still
using all final-framework infrastructure.
"""

from pathlib import Path
import csv
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from blackoil.grid3d import CartesianGrid3D
from blackoil.units import md, day, bar
from blackoil.properties3d import anisotropic_permeability_3d, gaussian_channel_permeability_3d, porosity_from_permeability_3d
from blackoil.reservoir3d import ReservoirPropertyModel3D, active_mask_from_boxes, map_property_by_zone, zone_from_layers
from blackoil.faults3d import FaultPlane3D, multipliers_from_faults
from blackoil.wells3d import FieldWell3D, WellTrajectory3D, completions_from_trajectory
from blackoil.pvt import BlackOilPVTTable, TabulatedFluid
from blackoil.relperm import CoreyThreePhaseRelPerm
from blackoil.capillary import LinearCapillaryPressure
from blackoil.rock import Rock
from blackoil.state import StateBlackOil
from blackoil.sparse_solver import SparseNewtonSolver
from blackoil.schedule3d import FieldWellControlEvent3D, FieldWellSchedule3D
from blackoil.black_oil_7 import FullBlackOilSimulator3D
from blackoil.visualization3d import (
    VTKTimeSeriesWriter3D,
    build_completion_cell_fields,
    save_state_time_series_npz,
    state_statistics,
    write_vtk_completion_points_3d,
    write_vtk_rectilinear_grid_3d,
    write_vtk_well_trajectories_3d,
)
from blackoil.diagnostics3d import plot_layer_map_3d, plot_state_time_series, plot_well_trajectories_3d


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def reports_to_rows(reports) -> list[dict]:
    rows = []
    for r in reports:
        rows.append({
            "time_days": r.time / day,
            "dt_days": r.dt / day,
            "newton_iterations": r.newton.iterations,
            "residual_norm": r.newton.residual_norm,
            "linear_iterations_total": r.linear_iterations_total,
            "jacobian_nnz": r.jacobian_nnz,
            "jacobian_colors": r.jacobian_colors,
            "pressure_min_bar": r.min_pressure / bar,
            "pressure_max_bar": r.max_pressure / bar,
            "sw_min": r.min_sw,
            "sw_max": r.max_sw,
            "sg_min": r.min_sg,
            "sg_max": r.max_sg,
            "so_min": r.min_so,
            "so_max": r.max_so,
            "rs_min": r.min_rs,
            "rs_max": r.max_rs,
            "saturated_cells": r.saturated_cells,
            "undersaturated_cells": r.undersaturated_cells,
            "oil_rate_m3_day": r.oil_rate * day,
            "water_prod_rate_m3_day": r.water_rate * day,
            "water_inj_rate_m3_day": r.water_injection_rate * day,
            "gas_component_rate_day": r.gas_component_rate * day,
            "producing_gor": r.producing_gor,
            "recovery_factor": r.recovery_factor,
            "oil_mb_rel": r.oil_material_balance_error_relative,
            "water_mb_rel": r.water_material_balance_error_relative,
            "gas_mb_rel": r.gas_material_balance_error_relative,
            "active_cells": r.active_cells,
            "inactive_cells": r.inactive_cells,
        })
    return rows


def main() -> None:
    out = ROOT / "outputs" / "example_14_step7_full_3d_black_oil"
    vtk_dir = out / "vtk_timeseries"
    restart_dir = out / "restart"
    out.mkdir(parents=True, exist_ok=True)
    vtk_dir.mkdir(parents=True, exist_ok=True)
    restart_dir.mkdir(parents=True, exist_ok=True)

    # 1. Grid and static 3D reservoir description.
    nx, ny, nz = 5, 4, 3
    lx, ly, lz = 260.0, 220.0, 36.0
    base_grid = CartesianGrid3D(nx, ny, nz, lx, ly, lz, top_depth=1580.0)
    x, y, z = base_grid.centers
    depth = 1580.0 + z + 0.015 * x - 0.010 * y
    grid = CartesianGrid3D(nx, ny, nz, lx, ly, lz, top_depth=1580.0, depth=depth)

    zones = zone_from_layers(grid, [1, 2, 3])
    background_md = map_property_by_zone(grid, zones, {1: 165.0, 2: 95.0, 3: 45.0}, name="Kx")
    channel_md = gaussian_channel_permeability_3d(grid, 380.0, 40.0, y_center_fraction=0.45, z_center_fraction=0.45, width_y_fraction=0.24, width_z_fraction=0.35)
    kx_md = 0.70 * background_md + 0.30 * channel_md
    porosity = porosity_from_permeability_3d(kx_md, phi_min=0.16, phi_max=0.255)
    permeability = anisotropic_permeability_3d(grid, kx_md * md, ky_kx=0.70, kvkh=0.08)
    active_mask = active_mask_from_boxes(grid, [{"i0": 0, "i1": 1, "j0": 0, "j1": 1, "k0": 2, "k1": 3}])
    reservoir = ReservoirPropertyModel3D.from_arrays(grid, porosity=porosity, permeability=permeability, active_mask=active_mask, zone_ids=zones)

    faults = [
        FaultPlane3D("F_SEALING", axis="x", index=2, multiplier=0.20, j_range=(1, 4), k_range=(0, 3)),
        FaultPlane3D("F_LEAKY_LAYER", axis="z", index=1, multiplier=0.55, i_range=(1, 5), j_range=(0, 4)),
    ]
    multipliers = multipliers_from_faults(grid, faults, active=reservoir.active)
    fault_indicator = multipliers.to_cell_indicator()

    # 2. Wells and completions.
    trajectories = [
        WellTrajectory3D.vertical("INJ_W", x=38.0, y=42.0, z_top=0.0, z_bottom=lz),
        WellTrajectory3D.vertical("PROD_E", x=226.0, y=184.0, z_top=0.0, z_bottom=lz),
        WellTrajectory3D.horizontal_x("PROD_H", x0=120.0, x1=242.0, y=104.0, z=18.0),
    ]
    wells = [
        FieldWell3D("INJ_W", "injector", "water_rate", 3.0 / day, completions_from_trajectory(grid, trajectories[0], permeability, orientation="z", active=reservoir.active), max_bhp=355.0 * bar),
        FieldWell3D("PROD_E", "producer", "liquid_rate", -2.2 / day, completions_from_trajectory(grid, trajectories[1], permeability, orientation="z", active=reservoir.active), min_bhp=125.0 * bar),
        FieldWell3D("PROD_H", "producer", "bhp", 175.0 * bar, completions_from_trajectory(grid, trajectories[2], permeability, orientation="x", active=reservoir.active), min_bhp=115.0 * bar),
    ]
    schedule = FieldWellSchedule3D(
        base_wells=wells,
        events=[
            FieldWellControlEvent3D(0.0, "INJ_W", control="water_rate", target=3.0 / day),
            FieldWellControlEvent3D(0.0, "PROD_E", control="liquid_rate", target=-2.2 / day),
            FieldWellControlEvent3D(0.0, "PROD_H", control="bhp", target=175.0 * bar),
            FieldWellControlEvent3D(10.0 * day, "PROD_E", control="liquid_rate", target=-2.8 / day),
            FieldWellControlEvent3D(15.0 * day, "PROD_H", control="bhp", target=165.0 * bar),
        ],
        report_times=[0.0, 5.0 * day, 10.0 * day, 15.0 * day, 20.0 * day],
    )

    # 3. Fluids, relperm, capillarity and initial state.
    table = BlackOilPVTTable.from_csv(ROOT / "data" / "pvt" / "live_oil_pvt.csv")
    water = TabulatedFluid("water", table, "Bw", "muw_pa_s", density_key="rhow_kg_m3")
    oil = TabulatedFluid("live_oil", table, "Bo", "muo_pa_s", density_key="rhoo_kg_m3", rs_key="Rs_sm3_sm3")
    gas = TabulatedFluid("gas", table, "Bg", "mug_pa_s", density_key="rhog_kg_m3")
    relperm = CoreyThreePhaseRelPerm(swc=0.18, sor=0.20, sgc=0.02, krw0=0.32, kro0=0.86, krg0=0.72)
    capillary = LinearCapillaryPressure(swc=relperm.swc, sor=relperm.sor, sgc=relperm.sgc, pcow_max=1.4e5, pcgo_max=1.0e5)

    p0 = 260.0 * bar
    rs_initial = float(oil.solution_gas_ratio(p0)) * 0.94
    is_saturated = np.zeros(grid.n_cells, dtype=bool)
    x_primary = np.full(grid.n_cells, rs_initial, dtype=float)
    # Seed a small gas region around producer completions so the gas equation
    # and switching logic are active from the first report.
    for c in wells[1].cells:
        is_saturated[c] = True
        x_primary[c] = 0.025
    state = StateBlackOil(
        p=np.full(grid.n_cells, p0, dtype=float),
        sw=np.full(grid.n_cells, relperm.swc + 0.05, dtype=float),
        x=x_primary,
        is_saturated=is_saturated,
    )
    rock = Rock(porosity_ref=reservoir.porosity, permeability=reservoir.permeability, compressibility=4.0e-10, p_ref=p0)
    solver = SparseNewtonSolver(
        tol=1.0e-7,
        acceptable_tol=8.0e-3,
        acceptable_min_iterations=2,
        max_iter=9,
        jacobian_strategy="sparse_fd",
        linear_solver="gmres",
        preconditioner="ilu",
        krylov_rtol=1.0e-6,
        krylov_maxiter=120,
        regularization=1.0e-20,
    )

    sim = FullBlackOilSimulator3D(
        grid=grid,
        rock=rock,
        water=water,
        oil=oil,
        gas=gas,
        relperm=relperm,
        state=state,
        wells=wells,
        schedule=schedule,
        capillary=capillary,
        gravity=9.80665,
        transmissibility_multipliers=multipliers,
        active=reservoir.active,
        solver=solver,
    )

    results = sim.run(20.0 * day, dt_initial=5.0 * day, dt_min=0.10 * day, dt_max=5.0 * day, max_ds=0.08, restart_dir=restart_dir, restart_interval=10.0 * day)

    # 4. Tables and VTK export.
    report_rows = reports_to_rows(results["reports"])
    write_csv(out / "step7_timestep_report.csv", report_rows)
    write_csv(out / "step7_schedule_history.csv", results["schedule_history"])
    control_rows = []
    for entry in results["control_history"]:
        for c in entry["controls"]:
            control_rows.append({"time_days": entry["time"] / day, "dt_days": entry["dt"] / day, **c})
    write_csv(out / "step7_control_history.csv", control_rows)

    static_fields = {
        "active": reservoir.active.active_mask.astype(float),
        "zone_id": zones.astype(float),
        "fault_indicator": fault_indicator,
        "porosity": reservoir.porosity,
        "pore_volume_m3": reservoir.pore_volume,
        "kx_mD": reservoir.permeability["kx"] / md,
        "ky_mD": reservoir.permeability["ky"] / md,
        "kz_mD": reservoir.permeability["kz"] / md,
        "depth_m": grid.depths,
        **build_completion_cell_fields(grid, wells),
    }
    write_vtk_rectilinear_grid_3d(out / "step7_static_reservoir.vtk", grid, static_fields)
    write_vtk_well_trajectories_3d(out / "step7_well_trajectories.vtk", trajectories)
    write_vtk_completion_points_3d(out / "step7_well_completions.vtk", grid, wells)

    writer = VTKTimeSeriesWriter3D(vtk_dir, grid, prefix="step7_blackoil_state")
    archive = {"pressure_bar": [], "water_saturation": [], "oil_saturation": [], "gas_saturation": [], "Rs_sm3_sm3": [], "phase_state_saturated": []}
    stats_rows = []
    for step, time_s in enumerate(results["time"]):
        fields = {
            **static_fields,
            "pressure_bar": results["pressure"][step] / bar,
            "water_saturation": results["sw"][step],
            "oil_saturation": results["so"][step],
            "gas_saturation": results["sg"][step],
            "Rs_sm3_sm3": results["rs"][step],
            "phase_state_saturated": results["is_saturated"][step].astype(float),
        }
        writer.write_state(time_s, fields, step=step)
        for key in archive:
            archive[key].append(fields[key])
        stats = state_statistics(grid, {"pressure_bar": fields["pressure_bar"], "water_saturation": fields["water_saturation"], "gas_saturation": fields["gas_saturation"], "oil_saturation": fields["oil_saturation"]}, pore_volume=reservoir.pore_volume)
        stats_rows.append({"time_days": time_s / day, **stats})
    pvd_path = writer.write_pvd("step7_blackoil_state_series.pvd")
    writer.write_manifest()
    save_state_time_series_npz(out / "step7_state_timeseries.npz", results["time"] / day, archive)
    write_csv(out / "step7_state_statistics.csv", stats_rows)

    # 5. Diagnostic plots.
    final = {key: values[-1] for key, values in archive.items()}
    mid_layer = grid.nz // 2
    plot_layer_map_3d(out / "final_pressure_layer.png", grid, final["pressure_bar"], layer=mid_layer, title="Step 7 final pressure", label="pressure [bar]", wells=wells, fault_indicator=fault_indicator)
    plot_layer_map_3d(out / "final_water_saturation_layer.png", grid, final["water_saturation"], layer=mid_layer, title="Step 7 final water saturation", label="Sw [-]", wells=wells)
    plot_layer_map_3d(out / "final_gas_saturation_layer.png", grid, final["gas_saturation"], layer=mid_layer, title="Step 7 final gas saturation", label="Sg [-]", wells=wells)
    plot_well_trajectories_3d(out / "step7_well_trajectory_projection.png", grid, trajectories, wells=wells)
    history = {
        "time_days": results["time"] / day,
        "pressure_pvmean_bar": [r["pressure_bar_pvmean"] for r in stats_rows],
        "Sw_pvmean": [r["water_saturation_pvmean"] for r in stats_rows],
        "Sg_pvmean": [r["gas_saturation_pvmean"] for r in stats_rows],
        "So_pvmean": [r["oil_saturation_pvmean"] for r in stats_rows],
    }
    plot_state_time_series(out / "step7_state_history.png", history, title="Step 7 PV-weighted state diagnostics")

    df = pd.DataFrame(report_rows)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(df["time_days"], df["recovery_factor"], marker="o")
    ax.set_xlabel("time [days]")
    ax.set_ylabel("recovery factor [-]")
    ax.set_title("Step 7 recovery factor")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "step7_recovery_factor.png", dpi=180)
    plt.close(fig)

    summary = out / "step7_summary.txt"
    summary.write_text(
        "Step 7 integrated full 3D black-oil example\n"
        "===========================================\n"
        f"grid: {nx} x {ny} x {nz} = {grid.n_cells} cells\n"
        f"active cells: {reservoir.active.n_active}\n"
        f"inactive cells: {grid.n_cells - reservoir.active.n_active}\n"
        f"faults: {len(faults)}\n"
        f"wells: {len(wells)}\n"
        f"completions: {sum(len(w.completions) for w in wells)}\n"
        f"accepted timesteps: {len(results['reports'])}\n"
        f"final recovery factor: {results['reports'][-1].recovery_factor:.6e}\n"
        f"final oil material-balance relative error: {results['reports'][-1].oil_material_balance_error_relative:.6e}\n"
        f"final water material-balance relative error: {results['reports'][-1].water_material_balance_error_relative:.6e}\n"
        f"final gas material-balance relative error: {results['reports'][-1].gas_material_balance_error_relative:.6e}\n"
        f"PVD time series: {pvd_path}\n"
        f"restart files: {len(results['restart_files'])}\n",
        encoding="utf-8",
    )
    print("Step 7 integrated full 3D black-oil example finished successfully")
    print(summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
