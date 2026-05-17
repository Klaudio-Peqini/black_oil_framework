"""Example 01: single-phase pressure diffusion with one injector and one BHP producer."""

from pathlib import Path
import sys

# Allow running directly from the source tree without installation.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from blackoil import (
    CartesianGrid1D,
    Rock,
    SlightlyCompressibleFluid,
    State1P,
    RateWell,
    BHPWell,
    SinglePhaseSimulator,
    NewtonSolver,
    day,
    cp,
    md_to_m2,
    bar,
)
from blackoil.io import ensure_dir


def main() -> None:
    out = ensure_dir(ROOT / "outputs" / "example_01_single_phase")

    grid = CartesianGrid1D(nx=50, length=1000.0, area=2000.0)
    p0 = 220.0 * bar

    rock = Rock(
        porosity_ref=0.22,
        permeability=md_to_m2(150.0),
        compressibility=4.0e-10,
        p_ref=p0,
    )

    water = SlightlyCompressibleFluid(
        name="water",
        mu_ref=1.0 * cp,
        b_ref=1.0,
        c_b=4.0e-10,
        p_ref=p0,
    )

    wells = [
        RateWell(name="INJ", cell=0, phase="single", rate=2.5e-4),
        BHPWell(name="PROD", cell=grid.nx - 1, bhp=180.0 * bar, well_index=2.0e-13),
    ]

    state = State1P.constant(grid.nx, p0)
    solver = NewtonSolver(tol=1.0e-7, max_iter=10, verbose=False)
    sim = SinglePhaseSimulator(grid, rock, water, state, wells=wells, solver=solver)

    results = sim.run(t_final=200.0 * day, dt=10.0 * day)

    # Save arrays.
    np.savez(out / "single_phase_results.npz", time=results["time"], pressure=results["pressure"])

    report_rows = []
    for r in results["reports"]:
        report_rows.append(
            {
                "time_days": r.time / day,
                "dt_days": r.dt / day,
                "newton_iterations": r.newton.iterations,
                "residual_norm": r.newton.residual_norm,
                "p_min_bar": r.min_pressure / bar,
                "p_max_bar": r.max_pressure / bar,
            }
        )
    pd.DataFrame(report_rows).to_csv(out / "single_phase_timestep_report.csv", index=False)

    # Plot selected pressure profiles.
    x = grid.centers
    pressure = results["pressure"] / bar
    time_days = results["time"] / day

    plt.figure(figsize=(7.5, 4.8))
    for idx in [0, len(time_days) // 4, len(time_days) // 2, -1]:
        plt.plot(x, pressure[idx], label=f"t = {time_days[idx]:.0f} d")
    plt.xlabel("x [m]")
    plt.ylabel("Pressure [bar]")
    plt.title("Single-phase pressure diffusion")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "pressure_profiles.png", dpi=180)

    print(f"Example 01 finished. Results written to: {out}")


if __name__ == "__main__":
    main()
