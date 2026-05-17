"""Example 02: fully implicit two-phase oil-water displacement.

This is the practical first reservoir-simulation prototype: water is injected at
one side and fluids are produced from a BHP well at the other side.
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
    SlightlyCompressibleFluid,
    CoreyWaterOilRelPerm,
    State2P,
    RateWell,
    BHPWell,
    TwoPhaseOilWaterSimulator,
    NewtonSolver,
    day,
    cp,
    md_to_m2,
    bar,
)
from blackoil.io import ensure_dir


def main() -> None:
    out = ensure_dir(ROOT / "outputs" / "example_02_two_phase")

    grid = CartesianGrid1D(nx=35, length=700.0, area=1500.0)
    p0 = 240.0 * bar

    rock = Rock(
        porosity_ref=0.24,
        permeability=md_to_m2(120.0),
        compressibility=3.0e-10,
        p_ref=p0,
    )

    water = SlightlyCompressibleFluid(
        name="water",
        mu_ref=0.7 * cp,
        b_ref=1.0,
        c_b=4.0e-10,
        p_ref=p0,
    )
    oil = SlightlyCompressibleFluid(
        name="oil",
        mu_ref=4.0 * cp,
        b_ref=1.18,
        c_b=8.0e-10,
        p_ref=p0,
    )

    relperm = CoreyWaterOilRelPerm(swc=0.18, sor=0.20, krw0=0.35, kro0=0.9, nw=2.2, no=2.0)

    wells = [
        RateWell(name="WATER_INJ", cell=0, phase="water", rate=1.8e-4),
        BHPWell(name="PROD", cell=grid.nx - 1, bhp=190.0 * bar, well_index=3.0e-13),
    ]

    state = State2P.constant(grid.nx, pressure=p0, sw=relperm.swc + 0.02)
    solver = NewtonSolver(tol=1.0e-7, max_iter=12, verbose=False)
    sim = TwoPhaseOilWaterSimulator(grid, rock, water, oil, relperm, state, wells=wells, solver=solver)

    results = sim.run(t_final=600.0 * day, dt=20.0 * day)

    np.savez(out / "two_phase_results.npz", time=results["time"], pressure=results["pressure"], sw=results["sw"])

    rows = []
    for r in results["reports"]:
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
                "water_rate_m3_day": r.water_rate * day,
                "cumulative_oil_m3": r.cumulative_oil,
                "cumulative_water_m3": r.cumulative_water,
                "recovery_factor": r.recovery_factor,
                "water_cut": r.water_rate / (r.water_rate + r.oil_rate + 1.0e-30),
            }
        )
    pd.DataFrame(rows).to_csv(out / "two_phase_timestep_report.csv", index=False)

    x = grid.centers
    time_days = results["time"] / day
    pressure = results["pressure"] / bar
    sw = results["sw"]

    plt.figure(figsize=(7.5, 4.8))
    for idx in [0, len(time_days) // 4, len(time_days) // 2, -1]:
        plt.plot(x, sw[idx], label=f"t = {time_days[idx]:.0f} d")
    plt.xlabel("x [m]")
    plt.ylabel("Water saturation")
    plt.title("Fully implicit water-oil displacement")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "water_saturation_profiles.png", dpi=180)

    plt.figure(figsize=(7.5, 4.8))
    for idx in [0, len(time_days) // 4, len(time_days) // 2, -1]:
        plt.plot(x, pressure[idx], label=f"t = {time_days[idx]:.0f} d")
    plt.xlabel("x [m]")
    plt.ylabel("Pressure [bar]")
    plt.title("Pressure profiles")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "pressure_profiles.png", dpi=180)

    df = pd.DataFrame(rows)
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(df["time_days"], df["recovery_factor"], marker="o")
    plt.xlabel("Time [days]")
    plt.ylabel("Recovery factor")
    plt.title("Oil recovery factor")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "recovery_factor.png", dpi=180)

    plt.figure(figsize=(7.5, 4.8))
    plt.plot(df["time_days"], df["oil_rate_m3_day"], marker="o", label="Oil")
    plt.plot(df["time_days"], df["water_rate_m3_day"], marker="s", label="Water")
    plt.xlabel("Time [days]")
    plt.ylabel("Production rate [m³/day]")
    plt.title("Produced phase rates")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "production_rates.png", dpi=180)

    print(f"Example 02 finished. Results written to: {out}")


if __name__ == "__main__":
    main()
