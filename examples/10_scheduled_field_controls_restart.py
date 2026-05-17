"""Example 10: Step 5D schedules, control histories, and restart files.

This example upgrades the 2D heterogeneous black-oil case into a small
field-style operational run:

* multiple wells are controlled by a schedule,
* targets and statuses change at prescribed dates,
* adaptive timesteps land exactly on schedule/report milestones,
* active rate/BHP switching is logged,
* restart files are written and can be loaded into a cloned simulator.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from blackoil import (
    CartesianGrid2D,
    Rock,
    BlackOilPVTTable,
    TabulatedFluid,
    CoreyThreePhaseRelPerm,
    StateBlackOil,
    SparseNewtonSolver,
    BrooksCoreyCapillaryPressure,
    BoundaryConditions2D,
    ScheduledBlackOilSimulator5D,
    WellControlEvent,
    WellSchedule,
    layered_permeability_2d,
    write_vtk_structured_grid_2d,
    load_black_oil_restart,
    apply_black_oil_restart,
    day,
    md_to_m2,
    bar,
)
from blackoil.io import ensure_dir


def reports_to_frame(reports) -> pd.DataFrame:
    rows = []
    for r in reports:
        rows.append(
            {
                "time_days": r.time / day,
                "dt_days": r.dt / day,
                "newton_iterations": r.newton.iterations,
                "residual_norm": r.newton.residual_norm,
                "linear_iterations_total": getattr(r.newton, "linear_iterations_total", 0),
                "p_min_bar": r.min_pressure / bar,
                "p_max_bar": r.max_pressure / bar,
                "sw_min": r.min_sw,
                "sw_max": r.max_sw,
                "sg_min": r.min_sg,
                "sg_max": r.max_sg,
                "saturated_cells": r.saturated_cells,
                "oil_rate_m3_day": r.oil_rate * day,
                "water_rate_m3_day": r.water_rate * day,
                "free_gas_rate_m3_day": r.free_gas_rate * day,
                "gas_component_rate_sm3_day": r.gas_component_rate * day,
                "water_injection_rate_m3_day": r.water_injection_rate * day,
                "producing_gor_sm3_sm3": r.producing_gor,
                "recovery_factor": r.recovery_factor,
                "oil_mb_error_relative": r.oil_material_balance_error_relative,
                "water_mb_error_relative": r.water_material_balance_error_relative,
                "gas_mb_error_relative": r.gas_material_balance_error_relative,
            }
        )
    return pd.DataFrame(rows)


def save_map(grid, values, path, title, label):
    arr = grid.reshape(values)
    plt.figure(figsize=(6.2, 4.8))
    im = plt.imshow(arr, origin="lower", extent=[0, grid.lx, 0, grid.ly], aspect="auto")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title)
    cbar = plt.colorbar(im)
    cbar.set_label(label)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def make_simulator(schedule: WellSchedule) -> ScheduledBlackOilSimulator5D:
    grid = CartesianGrid2D(nx=5, ny=4, lx=250.0, ly=200.0, thickness=18.0, depth=1500.0)
    p0 = 265.0 * bar

    kx = layered_permeability_2d(
        grid,
        [md_to_m2(35.0), md_to_m2(140.0), md_to_m2(60.0)],
        direction="y",
    )
    ky = 0.30 * kx
    rock = Rock(
        porosity_ref=0.21 + 0.025 * (kx / np.max(kx)),
        permeability={"kx": kx, "ky": ky},
        compressibility=4.5e-10,
        p_ref=p0,
    )

    table = BlackOilPVTTable.from_csv(ROOT / "data" / "pvt" / "live_oil_pvt.csv")
    water = TabulatedFluid("water", table, "Bw", "muw_pa_s", density_key="rhow_kg_m3")
    oil = TabulatedFluid("live_oil", table, "Bo", "muo_pa_s", density_key="rhoo_kg_m3", rs_key="Rs_sm3_sm3")
    gas = TabulatedFluid("gas", table, "Bg", "mug_pa_s", density_key="rhog_kg_m3")

    relperm = CoreyThreePhaseRelPerm(
        swc=0.18,
        sor=0.20,
        sgc=0.02,
        krw0=0.32,
        kro0=0.86,
        krg0=0.72,
        nw=2.4,
        no=2.1,
        ng=2.0,
    )
    capillary = BrooksCoreyCapillaryPressure(
        swc=relperm.swc,
        sor=relperm.sor,
        sgc=relperm.sgc,
        pe_w=1.2e4,
        pe_g=8.0e3,
        pcow_max=1.6e5,
        pcgo_max=1.2e5,
    )

    n = grid.n_cells
    is_saturated = np.zeros(n, dtype=bool)
    is_saturated[grid.cell_index(grid.nx - 1, grid.ny - 1)] = True
    x_primary = np.full(n, 120.0, dtype=float)
    x_primary[is_saturated] = 0.020
    state = StateBlackOil(
        p=np.full(n, p0, dtype=float),
        sw=np.full(n, relperm.swc + 0.03, dtype=float),
        x=x_primary,
        is_saturated=is_saturated,
    )

    solver = SparseNewtonSolver(
        tol=1.0e-7,
        max_iter=14,
        acceptable_tol=8.0e-2,
        acceptable_min_iterations=2,
        jacobian_strategy="sparse_fd",
        linear_solver="gmres",
        preconditioner="ilu",
        krylov_rtol=1.0e-6,
        krylov_maxiter=100,
        gmres_restart=50,
        regularization=1.0e-18,
    )

    return ScheduledBlackOilSimulator5D(
        grid,
        rock,
        water,
        oil,
        gas,
        relperm,
        state,
        schedule=schedule,
        solver=solver,
        capillary=capillary,
        gravity=9.80665,
        boundaries=BoundaryConditions2D.no_flow(),
        sg_switch_tol=2.0e-5,
    )


def main() -> None:
    out = ensure_dir(ROOT / "outputs" / "example_10_step5d_schedules_restart")
    restart_dir = ensure_dir(out / "restart")

    grid_probe = CartesianGrid2D(nx=5, ny=4, lx=250.0, ly=200.0, thickness=18.0, depth=1500.0)
    inj1 = grid_probe.cell_index(0, 0)
    inj2 = grid_probe.cell_index(0, grid_probe.ny - 1)
    prod1 = grid_probe.cell_index(grid_probe.nx - 1, grid_probe.ny - 1)
    prod2 = grid_probe.cell_index(grid_probe.nx - 1, 0)

    schedule = WellSchedule(
        report_times=[0.5 * day, 1.0 * day, 1.5 * day, 2.0 * day, 2.5 * day, 3.0 * day],
        events=[
            WellControlEvent(0.0, "INJ_SW", cell=inj1, well_index=2.3e-15, control="water_rate", target=2.2 / day, max_bhp=345.0 * bar, group="INJECTORS"),
            WellControlEvent(0.0, "PROD_NE", cell=prod1, well_index=2.8e-15, control="total_rate", target=-3.0 / day, min_bhp=145.0 * bar, group="PRODUCERS"),
            WellControlEvent(0.8 * day, "PROD_SE", cell=prod2, well_index=2.2e-15, control="liquid_rate", target=-1.4 / day, min_bhp=145.0 * bar, group="PRODUCERS"),
            WellControlEvent(1.2 * day, "INJ_NW", cell=inj2, well_index=2.0e-15, control="water_rate", target=1.2 / day, max_bhp=350.0 * bar, group="INJECTORS"),
            WellControlEvent(1.6 * day, "PROD_NE", control="bhp", target=155.0 * bar, min_bhp=145.0 * bar, group="PRODUCERS"),
            WellControlEvent(2.2 * day, "INJ_SW", control="water_rate", target=1.0 / day, max_bhp=335.0 * bar, group="INJECTORS"),
            WellControlEvent(2.6 * day, "PROD_SE", control="shut", group="PRODUCERS"),
        ],
    )
    schedule.to_csv(out / "field_schedule.csv")

    sim = make_simulator(schedule)
    results = sim.run_scheduled(
        t_final=3.0 * day,
        dt_initial=0.20 * day,
        dt_min=0.002 * day,
        dt_max=0.50 * day,
        max_ds=0.08,
        restart_dir=restart_dir,
        restart_interval=1.5 * day,
        write_restart_at_report_times=True,
    )

    df = reports_to_frame(results["reports"])
    df.to_csv(out / "step5d_timestep_report.csv", index=False)
    pd.DataFrame(results["schedule_history"]).drop_duplicates().to_csv(out / "step5d_schedule_history.csv", index=False)
    control_rows = []
    for k, entry in enumerate(sim.active_control_log):
        for c in entry.get("controls", []):
            row = {"step": k, "dt_days": entry.get("dt", np.nan) / day}
            row.update(c)
            if "bhp" in row and row["bhp"] == row["bhp"]:
                row["bhp_bar"] = row["bhp"] / bar
            control_rows.append(row)
    pd.DataFrame(control_rows).to_csv(out / "step5d_active_control_history.csv", index=False)

    final = -1
    np.savez(
        out / "step5d_results.npz",
        time=results["time"],
        pressure=results["pressure"],
        sw=results["sw"],
        sg=results["sg"],
        so=results["so"],
        rs=results["rs"],
        is_saturated=results["is_saturated"],
    )

    # Restart demonstration: load the last restart into a fresh simulator and
    # continue from that time to the final schedule time. This verifies that the
    # dynamic state and cumulative quantities are self-contained.
    restart_files = sorted(Path(p) for p in results["restart_files"])
    if restart_files:
        clone = make_simulator(schedule)
        restart_data = load_black_oil_restart(restart_files[-1])
        t_restart = apply_black_oil_restart(clone, restart_data)
        clone.current_time = t_restart
        clone_results = clone.run_scheduled(
            t_final=3.0 * day,
            dt_initial=0.20 * day,
            dt_min=0.002 * day,
            dt_max=0.50 * day,
            max_ds=0.08,
        )
        pd.DataFrame({"restart_time_days": [t_restart / day], "continued_steps": [len(clone_results["reports"])]}).to_csv(
            out / "restart_demo_summary.csv", index=False
        )

    save_map(sim.grid, sim.rock.permeability["kx"] / md_to_m2(1.0), out / "kx_map_md.png", "Step 5D Kx", "Kx [mD]")
    save_map(sim.grid, results["pressure"][final] / bar, out / "final_pressure_map.png", "Final oil pressure", "p_o [bar]")
    save_map(sim.grid, results["sw"][final], out / "final_water_saturation_map.png", "Final water saturation", "Sw")
    save_map(sim.grid, results["sg"][final], out / "final_gas_saturation_map.png", "Final gas saturation", "Sg")

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(df["time_days"], df["oil_rate_m3_day"], marker="o", label="oil production")
    plt.plot(df["time_days"], df["water_rate_m3_day"], marker="^", label="water production")
    plt.plot(df["time_days"], df["water_injection_rate_m3_day"], marker="s", label="water injection")
    for t_event in schedule.event_times:
        plt.axvline(t_event / day, linestyle="--", alpha=0.25)
    plt.xlabel("time [days]")
    plt.ylabel("rate [m3/day]")
    plt.title("Step 5D scheduled field rates")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "scheduled_well_rates.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(df["time_days"], df["recovery_factor"], marker="o")
    plt.xlabel("time [days]")
    plt.ylabel("recovery factor [-]")
    plt.title("Step 5D recovery factor")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "recovery_factor.png", dpi=180)
    plt.close()

    write_vtk_structured_grid_2d(
        out / "final_state.vtk",
        sim.grid,
        {
            "pressure_bar": results["pressure"][final] / bar,
            "water_saturation": results["sw"][final],
            "oil_saturation": results["so"][final],
            "gas_saturation": results["sg"][final],
            "solution_gas_ratio": results["rs"][final],
            "kx_md": sim.rock.permeability["kx"] / md_to_m2(1.0),
            "ky_md": sim.rock.permeability["ky"] / md_to_m2(1.0),
        },
    )

    print(f"Example 10 finished. Outputs written to {out}")
    print(df.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
