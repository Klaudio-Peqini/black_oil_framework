"""Example 09: Step 5C 2D heterogeneous black-oil validation case.

This example exercises the new 2D finite-volume infrastructure:

* Cartesian 2D grid,
* heterogeneous and anisotropic permeability,
* no-flow boundaries,
* corner injector/producer pair,
* gravity/capillary/live-oil phase switching,
* sparse Newton/Krylov solution,
* 2D maps and ParaView-compatible VTK export.
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
    ControlledWell,
    SparseNewtonSolver,
    BrooksCoreyCapillaryPressure,
    BoundaryConditions2D,
    HeterogeneousBlackOilSimulator5C,
    gaussian_channel_permeability_2d,
    write_vtk_structured_grid_2d,
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
                "linear_solver": getattr(r.newton, "linear_solver", "unknown"),
                "preconditioner": getattr(r.newton, "preconditioner", "unknown"),
                "linear_iterations_total": getattr(r.newton, "linear_iterations_total", 0),
                "jacobian_nnz_last": getattr(r.newton, "jacobian_nnz_last", 0),
                "jacobian_colors": getattr(r.newton, "jacobian_colors", 0),
                "p_min_bar": r.min_pressure / bar,
                "p_max_bar": r.max_pressure / bar,
                "sw_min": r.min_sw,
                "sw_max": r.max_sw,
                "sg_min": r.min_sg,
                "sg_max": r.max_sg,
                "saturated_cells": r.saturated_cells,
                "oil_rate_m3_day": r.oil_rate * day,
                "water_rate_m3_day": r.water_rate * day,
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


def main() -> None:
    out = ensure_dir(ROOT / "outputs" / "example_09_step5c_2d_heterogeneous")

    grid = CartesianGrid2D(nx=6, ny=5, lx=300.0, ly=250.0, thickness=18.0, depth=1500.0)
    p0 = 270.0 * bar

    # A smooth high-permeability channel in x, with vertical anisotropy in the
    # areal y direction. This is intentionally simple but nontrivial enough to
    # test anisotropic transmissibilities and 2D sweep behavior.
    kx = gaussian_channel_permeability_2d(
        grid,
        k_background=md_to_m2(40.0),
        k_channel=md_to_m2(220.0),
        y_center_fraction=0.55,
        width_fraction=0.18,
    )
    ky = 0.35 * kx
    rock = Rock(
        porosity_ref=0.22 + 0.03 * (kx / np.max(kx)),
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
        pe_w=1.5e4,
        pe_g=9.0e3,
        pcow_max=1.8e5,
        pcgo_max=1.4e5,
    )

    inj_cell = grid.cell_index(0, 0)
    prod_cell = grid.cell_index(grid.nx - 1, grid.ny - 1)
    wells = [
        ControlledWell(
            name="INJ_SW_CORNER",
            cell=inj_cell,
            control="water_rate",
            target=4.0 / day,
            well_index=2.5e-15,
            max_bhp=350.0 * bar,
        ),
        ControlledWell(
            name="PROD_NE_CORNER",
            cell=prod_cell,
            control="total_rate",
            target=-5.5 / day,
            well_index=3.0e-15,
            min_bhp=130.0 * bar,
        ),
    ]

    n = grid.n_cells
    is_saturated = np.zeros(n, dtype=bool)
    # A small gas cap near the producer-side top row makes phase switching and
    # gas-component conservation visible without making the test too stiff.
    for i in range(grid.nx):
        if i >= grid.nx // 2:
            is_saturated[grid.cell_index(i, grid.ny - 1)] = True
    x_primary = np.full(n, 125.0, dtype=float)
    x_primary[is_saturated] = 0.025
    state = StateBlackOil(
        p=np.full(n, p0, dtype=float),
        sw=np.full(n, relperm.swc + 0.035, dtype=float),
        x=x_primary,
        is_saturated=is_saturated,
    )

    solver = SparseNewtonSolver(
        tol=1.0e-7,
        max_iter=14,
        acceptable_tol=8.0e-3,
        acceptable_min_iterations=2,
        jacobian_strategy="sparse_fd",
        linear_solver="gmres",
        preconditioner="ilu",
        krylov_rtol=1.0e-6,
        krylov_maxiter=100,
        gmres_restart=50,
        regularization=1.0e-18,
    )

    sim = HeterogeneousBlackOilSimulator5C(
        grid,
        rock,
        water,
        oil,
        gas,
        relperm,
        state,
        wells=wells,
        solver=solver,
        capillary=capillary,
        gravity=9.80665,
        boundaries=BoundaryConditions2D.no_flow(),
        sg_switch_tol=2.0e-5,
    )

    results = sim.run_adaptive(
        t_final=3.0 * day,
        dt_initial=0.25 * day,
        dt_min=0.002 * day,
        dt_max=0.75 * day,
        max_ds=0.08,
    )

    df = reports_to_frame(results["reports"])
    df.to_csv(out / "step5c_2d_timestep_report.csv", index=False)
    np.savez(
        out / "step5c_2d_results.npz",
        time=results["time"],
        pressure=results["pressure"],
        sw=results["sw"],
        sg=results["sg"],
        so=results["so"],
        rs=results["rs"],
        is_saturated=results["is_saturated"],
        kx=kx,
        ky=ky,
    )

    final = -1
    save_map(grid, kx / md_to_m2(1.0), out / "kx_map_md.png", "Step 5C permeability Kx", "Kx [mD]")
    save_map(grid, ky / md_to_m2(1.0), out / "ky_map_md.png", "Step 5C permeability Ky", "Ky [mD]")
    save_map(grid, results["pressure"][final] / bar, out / "final_pressure_map.png", "Final oil pressure", "p_o [bar]")
    save_map(grid, results["sw"][final], out / "final_water_saturation_map.png", "Final water saturation", "Sw")
    save_map(grid, results["sg"][final], out / "final_gas_saturation_map.png", "Final gas saturation", "Sg")
    save_map(grid, results["rs"][final], out / "final_solution_gas_ratio_map.png", "Final solution gas ratio", "Rs [sm3/sm3]")

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(df["time_days"], df["oil_rate_m3_day"], marker="o", label="oil production")
    plt.plot(df["time_days"], df["water_injection_rate_m3_day"], marker="s", label="water injection")
    plt.xlabel("time [days]")
    plt.ylabel("rate [m3/day]")
    plt.title("Step 5C controlled well rates")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "well_rates.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(df["time_days"], df["newton_iterations"], marker="o", label="Newton")
    plt.plot(df["time_days"], df["linear_iterations_total"], marker="s", label="linear total")
    plt.xlabel("time [days]")
    plt.ylabel("iterations")
    plt.title("Step 5C sparse nonlinear/linear diagnostics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "solver_diagnostics.png", dpi=180)
    plt.close()

    write_vtk_structured_grid_2d(
        out / "final_state.vtk",
        grid,
        {
            "pressure_bar": results["pressure"][final] / bar,
            "water_saturation": results["sw"][final],
            "oil_saturation": results["so"][final],
            "gas_saturation": results["sg"][final],
            "solution_gas_ratio": results["rs"][final],
            "kx_md": kx / md_to_m2(1.0),
            "ky_md": ky / md_to_m2(1.0),
        },
    )

    print(f"Example 09 finished. Outputs written to {out}")
    print(df.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
