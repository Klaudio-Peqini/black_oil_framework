"""Example 03: compressible dead-oil waterflood with tabulated PVT.

This is the next step after the basic fully implicit water-oil prototype. It
keeps the same primary variables, p_o and S_w, but now uses pressure-dependent
PVT properties from a table, rock compressibility, timestep rejection, and
explicit material-balance diagnostics.
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
    CoreyWaterOilRelPerm,
    State2P,
    RateWell,
    BHPWell,
    DeadOilSimulator,
    NewtonSolver,
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
                "p_min_bar": r.min_pressure / bar,
                "p_max_bar": r.max_pressure / bar,
                "sw_min": r.min_sw,
                "sw_max": r.max_sw,
                "oil_rate_m3_day": r.oil_rate * day,
                "water_prod_rate_m3_day": r.water_rate * day,
                "water_inj_rate_m3_day": r.water_injection_rate * day,
                "cumulative_oil_m3": r.cumulative_oil_produced,
                "cumulative_water_prod_m3": r.cumulative_water_produced,
                "cumulative_water_inj_m3": r.cumulative_water_injected,
                "recovery_factor": r.recovery_factor,
                "water_cut": r.water_rate / (r.water_rate + r.oil_rate + 1.0e-30),
                "oil_mb_error_m3": r.oil_material_balance_error,
                "water_mb_error_m3": r.water_material_balance_error,
                "oil_mb_error_relative": r.oil_material_balance_error_relative,
                "water_mb_error_relative": r.water_material_balance_error_relative,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    out = ensure_dir(ROOT / "outputs" / "example_03_dead_oil")

    grid = CartesianGrid1D(nx=40, length=900.0, area=1800.0)
    p0 = 260.0 * bar

    rock = Rock(
        porosity_ref=0.23,
        permeability=md_to_m2(95.0),
        compressibility=4.5e-10,
        p_ref=p0,
    )

    table = BlackOilPVTTable.from_csv(ROOT / "data" / "pvt" / "dead_oil_pvt.csv")
    water = TabulatedFluid(
        name="water",
        table=table,
        b_key="Bw",
        mu_key="muw_pa_s",
        rho_ref=1000.0,
        density_key="rhow_kg_m3",
    )
    oil = TabulatedFluid(
        name="oil",
        table=table,
        b_key="Bo",
        mu_key="muo_pa_s",
        rho_ref=820.0,
        density_key="rhoo_kg_m3",
    )

    relperm = CoreyWaterOilRelPerm(
        swc=0.18,
        sor=0.22,
        krw0=0.32,
        kro0=0.88,
        nw=2.4,
        no=2.1,
    )

    # Rate-controlled water injection and BHP-controlled production.
    wells = [
        RateWell(name="INJ_WATER", cell=0, phase="water", rate=2.0e-4),
        BHPWell(name="PROD_BHP", cell=grid.nx - 1, bhp=175.0 * bar, well_index=4.0e-13),
    ]

    state = State2P.constant(grid.nx, pressure=p0, sw=relperm.swc + 0.015)
    solver = NewtonSolver(tol=1.0e-7, max_iter=14, verbose=False)
    sim = DeadOilSimulator(grid, rock, water, oil, relperm, state, wells=wells, solver=solver)

    results = sim.run_adaptive(
        t_final=900.0 * day,
        dt_initial=10.0 * day,
        dt_min=0.25 * day,
        dt_max=35.0 * day,
        max_ds=0.075,
    )

    df = reports_to_frame(results["reports"])
    df.to_csv(out / "dead_oil_timestep_report.csv", index=False)
    np.savez(out / "dead_oil_results.npz", time=results["time"], pressure=results["pressure"], sw=results["sw"])

    x = grid.centers
    time_days = results["time"] / day
    pressure_bar = results["pressure"] / bar
    sw = results["sw"]

    sample_ids = np.unique(np.array([0, len(time_days) // 5, 2 * len(time_days) // 5, 3 * len(time_days) // 5, -1]))

    plt.figure(figsize=(7.5, 4.8))
    for idx in sample_ids:
        plt.plot(x, sw[idx], label=f"t = {time_days[idx]:.0f} d")
    plt.xlabel("x [m]")
    plt.ylabel("Water saturation")
    plt.title("Dead-oil waterflood: saturation profiles")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "water_saturation_profiles.png", dpi=180)

    plt.figure(figsize=(7.5, 4.8))
    for idx in sample_ids:
        plt.plot(x, pressure_bar[idx], label=f"t = {time_days[idx]:.0f} d")
    plt.xlabel("x [m]")
    plt.ylabel("Oil pressure [bar]")
    plt.title("Dead-oil waterflood: pressure profiles")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "pressure_profiles.png", dpi=180)

    plt.figure(figsize=(7.5, 4.8))
    plt.plot(df["time_days"], df["oil_rate_m3_day"], marker="o", label="Oil production")
    plt.plot(df["time_days"], df["water_prod_rate_m3_day"], marker="s", label="Water production")
    plt.plot(df["time_days"], df["water_inj_rate_m3_day"], linestyle="--", label="Water injection")
    plt.xlabel("Time [days]")
    plt.ylabel("Rate [m³/day]")
    plt.title("Dead-oil well rates")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "well_rates.png", dpi=180)

    plt.figure(figsize=(7.5, 4.8))
    plt.plot(df["time_days"], df["recovery_factor"], marker="o")
    plt.xlabel("Time [days]")
    plt.ylabel("Recovery factor")
    plt.title("Dead-oil recovery factor")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "recovery_factor.png", dpi=180)

    plt.figure(figsize=(7.5, 4.8))
    plt.semilogy(df["time_days"], np.abs(df["oil_mb_error_relative"]) + 1.0e-30, label="Oil")
    plt.semilogy(df["time_days"], np.abs(df["water_mb_error_relative"]) + 1.0e-30, label="Water")
    plt.xlabel("Time [days]")
    plt.ylabel("Absolute relative material-balance error")
    plt.title("Material-balance diagnostics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "material_balance_error.png", dpi=180)

    print(f"Example 03 finished. Results written to: {out}")
    print(df.tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
