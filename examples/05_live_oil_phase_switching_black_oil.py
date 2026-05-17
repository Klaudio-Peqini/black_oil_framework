"""Example 05: live-oil black-oil model with bubble-point phase switching.

This is Step 4B of the framework. The primary-variable interpretation changes
cell by cell:

* undersaturated oil: p, Sw, Rs with Sg = 0;
* saturated oil:     p, Sw, Sg with Rs = Rs_sat(p).

The example starts above the bubble point with undersaturated oil. Production at
a low bottom-hole pressure depletes the right side of the model, causing cells
to cross the bubble point and release a free gas phase.
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
    RateWell,
    BHPWell,
    LiveOilPhaseSwitchingSimulator,
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
                "sg_min": r.min_sg,
                "sg_max": r.max_sg,
                "so_min": r.min_so,
                "so_max": r.max_so,
                "rs_min_sm3_sm3": r.min_rs,
                "rs_max_sm3_sm3": r.max_rs,
                "saturated_cells": r.saturated_cells,
                "undersaturated_cells": r.undersaturated_cells,
                "switched_to_saturated": r.switched_to_saturated,
                "switched_to_undersaturated": r.switched_to_undersaturated,
                "oil_rate_m3_day": r.oil_rate * day,
                "water_prod_rate_m3_day": r.water_rate * day,
                "free_gas_rate_sm3_day": r.free_gas_rate * day,
                "gas_component_rate_sm3_day": r.gas_component_rate * day,
                "water_inj_rate_m3_day": r.water_injection_rate * day,
                "producing_gor_sm3_sm3": r.producing_gor,
                "cumulative_oil_m3": r.cumulative_oil_produced,
                "cumulative_water_prod_m3": r.cumulative_water_produced,
                "cumulative_free_gas_sm3": r.cumulative_free_gas_produced,
                "cumulative_gas_component_sm3": r.cumulative_gas_component_produced,
                "cumulative_water_inj_m3": r.cumulative_water_injected,
                "recovery_factor": r.recovery_factor,
                "water_cut": r.water_rate / (r.water_rate + r.oil_rate + 1.0e-30),
                "oil_mb_error_m3": r.oil_material_balance_error,
                "water_mb_error_m3": r.water_material_balance_error,
                "gas_mb_error_sm3": r.gas_material_balance_error,
                "oil_mb_error_relative": r.oil_material_balance_error_relative,
                "water_mb_error_relative": r.water_material_balance_error_relative,
                "gas_mb_error_relative": r.gas_material_balance_error_relative,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    out = ensure_dir(ROOT / "outputs" / "example_05_live_oil_phase_switching")

    grid = CartesianGrid1D(nx=12, length=600.0, area=1600.0)
    p0 = 280.0 * bar

    rock = Rock(
        porosity_ref=0.23,
        permeability=md_to_m2(135.0),
        compressibility=4.0e-10,
        p_ref=p0,
    )

    table = BlackOilPVTTable.from_csv(ROOT / "data" / "pvt" / "live_oil_pvt.csv")
    water = TabulatedFluid(
        name="water",
        table=table,
        b_key="Bw",
        mu_key="muw_pa_s",
        rho_ref=1000.0,
        density_key="rhow_kg_m3",
    )
    oil = TabulatedFluid(
        name="live_oil",
        table=table,
        b_key="Bo",
        mu_key="muo_pa_s",
        rho_ref=780.0,
        density_key="rhoo_kg_m3",
        rs_key="Rs_sm3_sm3",
    )
    gas = TabulatedFluid(
        name="gas",
        table=table,
        b_key="Bg",
        mu_key="mug_pa_s",
        rho_ref=1.2,
        density_key="rhog_kg_m3",
    )

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

    # The initial Rs is below Rs_sat(p0), so the whole reservoir starts
    # undersaturated. A low-BHP producer can then drive pressure below the
    # corresponding bubble point and create free gas near the producer.
    initial_rs = 155.0
    initial_pbub = float(oil.bubble_point_pressure(initial_rs))

    wells = [
        # Depletion-only case: a low-BHP producer drives the right side below
        # the bubble point, allowing the free gas phase to appear. Water
        # injection is intentionally omitted in this validation example so the
        # phase-state transition is easy to read.
        BHPWell(name="PROD_BHP", cell=grid.nx - 1, bhp=100.0 * bar, well_index=2.0e-14),
    ]

    state = StateBlackOil.constant_undersaturated(
        grid.nx,
        pressure=p0,
        sw=relperm.swc + 0.025,
        rs=initial_rs,
    )

    solver = NewtonSolver(tol=1.0e-7, max_iter=22, verbose=False, acceptable_tol=5.0e-3)
    sim = LiveOilPhaseSwitchingSimulator(
        grid,
        rock,
        water,
        oil,
        gas,
        relperm,
        state,
        wells=wells,
        solver=solver,
        sg_switch_tol=2.0e-5,
    )

    results = sim.run_adaptive(
        t_final=120.0 * day,
        dt_initial=1.0 * day,
        dt_min=0.005 * day,
        dt_max=6.0 * day,
        max_ds=0.050,
    )

    df = reports_to_frame(results["reports"])
    df.to_csv(out / "phase_switching_timestep_report.csv", index=False)
    np.savez(
        out / "phase_switching_results.npz",
        time=results["time"],
        pressure=results["pressure"],
        sw=results["sw"],
        sg=results["sg"],
        so=results["so"],
        rs=results["rs"],
        is_saturated=results["is_saturated"],
    )

    x = grid.centers
    time_days = results["time"] / day
    pressure_bar = results["pressure"] / bar
    sw = results["sw"]
    sg = results["sg"]
    so = results["so"]
    rs = results["rs"]
    sat_fraction = np.mean(results["is_saturated"], axis=1)
    sample_ids = np.unique(np.array([0, len(time_days) // 4, len(time_days) // 2, 3 * len(time_days) // 4, -1]))

    plt.figure(figsize=(7.8, 4.8))
    for idx in sample_ids:
        plt.plot(x, pressure_bar[idx], label=f"t = {time_days[idx]:.0f} d")
    plt.axhline(initial_pbub / bar, linestyle="--", linewidth=1.1, label=f"initial bubble point ≈ {initial_pbub / bar:.0f} bar")
    plt.xlabel("x [m]")
    plt.ylabel("Oil pressure [bar]")
    plt.title("Phase-switching black-oil: pressure profiles")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "pressure_profiles.png", dpi=180)

    plt.figure(figsize=(7.8, 4.8))
    for idx in sample_ids:
        plt.plot(x, sg[idx], label=f"Sg, t={time_days[idx]:.0f} d")
    plt.xlabel("x [m]")
    plt.ylabel("Free-gas saturation")
    plt.title("Free gas appears after bubble-point crossing")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "gas_saturation_profiles.png", dpi=180)

    plt.figure(figsize=(7.8, 4.8))
    for idx in sample_ids:
        plt.plot(x, rs[idx], label=f"Rs, t={time_days[idx]:.0f} d")
    plt.xlabel("x [m]")
    plt.ylabel("Rs [sm³/sm³]")
    plt.title("Solution gas-oil ratio with primary-variable switching")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "solution_gas_ratio_profiles.png", dpi=180)

    plt.figure(figsize=(7.8, 4.8))
    plt.plot(time_days, sat_fraction, marker="o")
    plt.xlabel("Time [days]")
    plt.ylabel("Fraction of saturated cells")
    plt.title("Bubble-point phase-state evolution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "saturated_fraction.png", dpi=180)

    plt.figure(figsize=(7.8, 4.8))
    plt.plot(df["time_days"], df["oil_rate_m3_day"], marker="o", label="Oil production")
    plt.plot(df["time_days"], df["water_prod_rate_m3_day"], marker="s", label="Water production")
    plt.plot(df["time_days"], df["water_inj_rate_m3_day"], linestyle="--", label="Water injection")
    plt.xlabel("Time [days]")
    plt.ylabel("Rate [m³/day]")
    plt.title("Phase-switching black-oil well liquid rates")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "liquid_well_rates.png", dpi=180)

    plt.figure(figsize=(7.8, 4.8))
    plt.plot(df["time_days"], df["free_gas_rate_sm3_day"], marker="o", label="Free-gas phase")
    plt.plot(df["time_days"], df["gas_component_rate_sm3_day"], marker="s", label="Total gas component")
    plt.xlabel("Time [days]")
    plt.ylabel("Gas rate [sm³/day]")
    plt.title("Gas production after phase appearance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "gas_well_rates.png", dpi=180)

    plt.figure(figsize=(7.8, 4.8))
    plt.plot(df["time_days"], df["producing_gor_sm3_sm3"], marker="o")
    plt.xlabel("Time [days]")
    plt.ylabel("Producing GOR [sm³/sm³]")
    plt.title("Producing gas-oil ratio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "producing_gor.png", dpi=180)

    plt.figure(figsize=(7.8, 4.8))
    plt.semilogy(df["time_days"], np.abs(df["oil_mb_error_relative"]) + 1.0e-30, label="Oil")
    plt.semilogy(df["time_days"], np.abs(df["water_mb_error_relative"]) + 1.0e-30, label="Water")
    plt.semilogy(df["time_days"], np.abs(df["gas_mb_error_relative"]) + 1.0e-30, label="Gas component")
    plt.xlabel("Time [days]")
    plt.ylabel("Absolute relative material-balance error")
    plt.title("Phase-switching material-balance diagnostics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "material_balance_error.png", dpi=180)

    print(f"Example 05 finished. Results written to: {out}")
    print(f"Initial Rs = {initial_rs:.1f} sm3/sm3; bubble point = {initial_pbub / bar:.1f} bar")
    print(df.tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
