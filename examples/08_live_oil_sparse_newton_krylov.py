"""Example 08: Step 5B sparse Newton / Newton-Krylov black-oil solve.

The physics is inherited from Step 5A: live-oil phase switching, gravity,
capillary pressure, and controlled wells. The numerical infrastructure is new:
structural sparse Jacobians, sparse linear solvers, optional Krylov methods, and
preconditioning.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from blackoil import (
    CartesianGrid1D,
    Rock,
    BlackOilPVTTable,
    TabulatedFluid,
    CoreyThreePhaseRelPerm,
    StateBlackOil,
    ControlledWell,
    ScalableBlackOilSimulator5B,
    SparseNewtonSolver,
    BrooksCoreyCapillaryPressure,
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
                "jacobian_strategy": getattr(r.newton, "jacobian_strategy", "unknown"),
                "preconditioner": getattr(r.newton, "preconditioner", "unknown"),
                "linear_iterations_total": getattr(r.newton, "linear_iterations_total", 0),
                "linear_iterations_last": getattr(r.newton, "linear_iterations_last", 0),
                "jacobian_nnz_last": getattr(r.newton, "jacobian_nnz_last", 0),
                "jacobian_colors": getattr(r.newton, "jacobian_colors", 0),
                "p_min_bar": r.min_pressure / bar,
                "p_max_bar": r.max_pressure / bar,
                "sw_min": r.min_sw,
                "sw_max": r.max_sw,
                "sg_min": r.min_sg,
                "sg_max": r.max_sg,
                "oil_rate_m3_day": r.oil_rate * day,
                "water_injection_rate_m3_day": r.water_injection_rate * day,
                "producing_gor_sm3_sm3": r.producing_gor,
                "recovery_factor": r.recovery_factor,
                "oil_mb_error_relative": r.oil_material_balance_error_relative,
                "water_mb_error_relative": r.water_material_balance_error_relative,
                "gas_mb_error_relative": r.gas_material_balance_error_relative,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    out = ensure_dir(ROOT / "outputs" / "example_08_step5b_sparse_newton_krylov")

    nx = 10
    depth = np.linspace(1400.0, 1480.0, nx)
    grid = CartesianGrid1D(nx=nx, length=500.0, area=1200.0, depth=depth)
    p0 = 275.0 * bar

    rock = Rock(
        porosity_ref=0.23,
        permeability=np.linspace(md_to_m2(75.0), md_to_m2(150.0), nx),
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
        pe_w=1.8e4,
        pe_g=1.2e4,
        pcow_max=2.2e5,
        pcgo_max=1.8e5,
    )

    wells = [
        ControlledWell(
            name="INJ_WATER_RATE_MAXBHP",
            cell=0,
            control="water_rate",
            target=8.0 / day,
            well_index=3.5e-15,
            max_bhp=355.0 * bar,
        ),
        ControlledWell(
            name="PROD_TOTAL_RATE_MINBHP",
            cell=nx - 1,
            control="total_rate",
            target=-13.0 / day,
            well_index=4.0e-15,
            min_bhp=120.0 * bar,
        ),
    ]

    is_saturated = np.zeros(nx, dtype=bool)
    is_saturated[-4:] = True
    x_primary = np.full(nx, 135.0, dtype=float)
    x_primary[is_saturated] = 0.025
    state = StateBlackOil(
        p=np.full(nx, p0, dtype=float),
        sw=np.full(nx, relperm.swc + 0.035, dtype=float),
        x=x_primary,
        is_saturated=is_saturated,
    )

    # Use sparse FD + GMRES + ILU here to exercise the Step 5B path. For very
    # small teaching examples, linear_solver="spsolve" is also available.
    solver = SparseNewtonSolver(
        tol=1.0e-7,
        max_iter=16,
        acceptable_tol=7.0e-3,
        acceptable_min_iterations=2,
        jacobian_strategy="sparse_fd",
        linear_solver="gmres",
        preconditioner="ilu",
        krylov_rtol=1.0e-7,
        krylov_maxiter=80,
        gmres_restart=40,
    )

    sim = ScalableBlackOilSimulator5B(
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
        sg_switch_tol=2.0e-5,
    )

    results = sim.run_adaptive(
        t_final=5.0 * day,
        dt_initial=0.25 * day,
        dt_min=0.001 * day,
        dt_max=1.0 * day,
        max_ds=0.075,
    )

    df = reports_to_frame(results["reports"])
    df.to_csv(out / "step5b_sparse_timestep_report.csv", index=False)
    np.savez(
        out / "step5b_results.npz",
        time=results["time"],
        pressure=results["pressure"],
        sw=results["sw"],
        sg=results["sg"],
        so=results["so"],
        rs=results["rs"],
        is_saturated=results["is_saturated"],
        depth=grid.depths,
    )

    x = grid.centers
    time_days = results["time"] / day
    pressure_bar = results["pressure"] / bar
    sample_ids = np.unique(np.array([0, len(time_days) // 2, -1]))

    plt.figure(figsize=(7.8, 4.8))
    for idx in sample_ids:
        plt.plot(x, pressure_bar[idx], label=f"t = {time_days[idx]:.1f} d")
    plt.xlabel("x [m]")
    plt.ylabel("Oil pressure [bar]")
    plt.title("Step 5B: sparse Newton pressure profiles")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "pressure_profiles.png", dpi=180)

    plt.figure(figsize=(7.8, 4.8))
    for idx in sample_ids:
        plt.plot(x, results["sw"][idx], label=f"t = {time_days[idx]:.1f} d")
    plt.xlabel("x [m]")
    plt.ylabel("Water saturation")
    plt.title("Water saturation with sparse Newton infrastructure")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "water_saturation_profiles.png", dpi=180)

    plt.figure(figsize=(7.8, 4.8))
    for idx in sample_ids:
        plt.plot(x, results["sg"][idx], label=f"t = {time_days[idx]:.1f} d")
    plt.xlabel("x [m]")
    plt.ylabel("Free-gas saturation")
    plt.title("Gas saturation with phase switching")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "gas_saturation_profiles.png", dpi=180)

    plt.figure(figsize=(7.8, 4.8))
    plt.plot(df["time_days"], df["newton_iterations"], marker="o", label="Newton iterations")
    plt.plot(df["time_days"], df["linear_iterations_last"], marker="s", label="Last linear iterations")
    plt.xlabel("time [days]")
    plt.ylabel("iterations")
    plt.title("Sparse Newton / Krylov diagnostics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "solver_iterations.png", dpi=180)

    print(f"Example 08 finished. Outputs written to {out}")
    print(df.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
