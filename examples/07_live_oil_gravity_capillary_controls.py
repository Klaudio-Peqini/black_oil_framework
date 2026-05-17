"""Example 07: Step 5A black-oil model with gravity, capillarity, and controls.

This example extends the conservative Step 4C live-oil formulation by adding
phase-potential gravity terms, capillary-pressure phase pressures, and wells
whose requested rate controls can switch to BHP limits.
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
    NewtonSolverWithJacobian,
    AdvancedBlackOilSimulator5A,
    BrooksCoreyCapillaryPressure,
    day,
    md_to_m2,
    bar,
)
from blackoil.flux import phase_pressures_black_oil
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
                "water_injection_rate_m3_day": r.water_injection_rate * day,
                "gas_component_rate_sm3_day": r.gas_component_rate * day,
                "producing_gor_sm3_sm3": r.producing_gor,
                "recovery_factor": r.recovery_factor,
                "oil_mb_error_relative": r.oil_material_balance_error_relative,
                "water_mb_error_relative": r.water_material_balance_error_relative,
                "gas_mb_error_relative": r.gas_material_balance_error_relative,
            }
        )
    return pd.DataFrame(rows)


def controls_to_frame(control_log) -> pd.DataFrame:
    rows = []
    for k, entry in enumerate(control_log):
        for c in entry["controls"]:
            rows.append({"accepted_step": k + 1, "dt_days": entry["dt"] / day, **c})
    return pd.DataFrame(rows)


def main() -> None:
    out = ensure_dir(ROOT / "outputs" / "example_07_step5a_gravity_capillary_controls")

    nx = 7
    # An inclined 1D reservoir: depth is positive downward. The final cell is
    # deeper, so phase-potential flow includes a gravitational contribution.
    depth = np.linspace(1450.0, 1515.0, nx)
    grid = CartesianGrid1D(nx=nx, length=420.0, area=1400.0, depth=depth)
    p0 = 285.0 * bar

    rock = Rock(
        porosity_ref=0.24,
        permeability=np.linspace(md_to_m2(90.0), md_to_m2(160.0), nx),
        compressibility=4.0e-10,
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
        pe_w=2.0e4,
        pe_g=1.5e4,
        pcow_max=2.5e5,
        pcgo_max=2.0e5,
    )

    # A water injector is requested at a fixed rate, but with a maximum BHP.
    # A producer is requested at a liquid rate, but with a minimum BHP.
    wells = [
        ControlledWell(
            name="INJ_WATER_RATE_MAXBHP",
            cell=0,
            control="water_rate",
            target=10.0 / day,
            well_index=3.0e-15,
            max_bhp=360.0 * bar,
        ),
        ControlledWell(
            name="PROD_LIQ_RATE_MINBHP",
            cell=nx - 1,
            control="liquid_rate",
            target=-16.0 / day,
            well_index=4.0e-15,
            min_bhp=115.0 * bar,
        ),
    ]

    # Mixed initial phase state: most cells are undersaturated, while the
    # producer-side cells already contain a small amount of free gas.
    initial_rs = 145.0
    is_saturated = np.zeros(nx, dtype=bool)
    is_saturated[-3:] = True
    x_primary = np.full(nx, initial_rs, dtype=float)
    x_primary[is_saturated] = 0.028
    state = StateBlackOil(
        p=np.full(nx, p0, dtype=float),
        sw=np.full(nx, relperm.swc + 0.035, dtype=float),
        x=x_primary,
        is_saturated=is_saturated,
    )

    solver = NewtonSolverWithJacobian(
        tol=1.0e-7,
        max_iter=18,
        acceptable_tol=6.0e-3,
        acceptable_min_iterations=2,
    )
    sim = AdvancedBlackOilSimulator5A(
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
        t_final=10.0 * day,
        dt_initial=0.35 * day,
        dt_min=0.001 * day,
        dt_max=1.5 * day,
        max_ds=0.075,
    )

    df = reports_to_frame(results["reports"])
    cdf = controls_to_frame(sim.active_control_log)
    df.to_csv(out / "step5a_timestep_report.csv", index=False)
    cdf.to_csv(out / "step5a_control_log.csv", index=False)
    np.savez(
        out / "step5a_results.npz",
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
    sw = results["sw"]
    sg = results["sg"]
    rs = results["rs"]
    sat_fraction = np.mean(results["is_saturated"], axis=1)
    sample_ids = np.unique(np.array([0, len(time_days) // 3, 2 * len(time_days) // 3, -1]))

    plt.figure(figsize=(7.8, 4.8))
    for idx in sample_ids:
        plt.plot(x, pressure_bar[idx], label=f"t = {time_days[idx]:.1f} d")
    plt.xlabel("x [m]")
    plt.ylabel("Oil pressure [bar]")
    plt.title("Step 5A: oil pressure with gravity and capillary terms")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "pressure_profiles.png", dpi=180)

    plt.figure(figsize=(7.8, 4.8))
    for idx in sample_ids:
        plt.plot(x, sw[idx], label=f"t = {time_days[idx]:.1f} d")
    plt.xlabel("x [m]")
    plt.ylabel("Water saturation")
    plt.title("Water saturation under controlled injection")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "water_saturation_profiles.png", dpi=180)

    plt.figure(figsize=(7.8, 4.8))
    for idx in sample_ids:
        plt.plot(x, sg[idx], label=f"t = {time_days[idx]:.1f} d")
    plt.xlabel("x [m]")
    plt.ylabel("Free-gas saturation")
    plt.title("Free gas with phase-state switching")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "gas_saturation_profiles.png", dpi=180)

    plt.figure(figsize=(7.8, 4.8))
    for idx in sample_ids:
        plt.plot(x, rs[idx], label=f"t = {time_days[idx]:.1f} d")
    plt.xlabel("x [m]")
    plt.ylabel("Rs [sm³/sm³]")
    plt.title("Dissolved gas ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "solution_gas_ratio_profiles.png", dpi=180)

    # Show capillary pressure curves sampled over the simulated states.
    final_sw = sw[-1]
    final_sg = sg[-1]
    plt.figure(figsize=(7.8, 4.8))
    plt.plot(x, capillary.pcow(final_sw) / bar, marker="o", label="pcow")
    plt.plot(x, capillary.pcgo(final_sg) / bar, marker="s", label="pcgo")
    plt.xlabel("x [m]")
    plt.ylabel("Capillary pressure [bar]")
    plt.title("Final capillary-pressure fields")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "capillary_pressure_profiles.png", dpi=180)

    plt.figure(figsize=(7.8, 4.8))
    plt.plot(time_days, sat_fraction, marker="o")
    plt.xlabel("Time [days]")
    plt.ylabel("Fraction of saturated cells")
    plt.title("Phase-state evolution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "saturated_fraction.png", dpi=180)

    plt.figure(figsize=(7.8, 4.8))
    plt.plot(df["time_days"], df["oil_rate_m3_day"], marker="o", label="Oil produced")
    plt.plot(df["time_days"], df["water_prod_rate_m3_day"], marker="s", label="Water produced")
    plt.plot(df["time_days"], df["water_injection_rate_m3_day"], marker="^", label="Water injected")
    plt.xlabel("Time [days]")
    plt.ylabel("Rate [m³/day]")
    plt.title("Controlled well rates")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "controlled_well_rates.png", dpi=180)

    plt.figure(figsize=(7.8, 4.8))
    plt.plot(df["time_days"], df["producing_gor_sm3_sm3"], marker="o")
    plt.xlabel("Time [days]")
    plt.ylabel("Producing GOR [sm³/sm³]")
    plt.title("Gas-oil ratio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "producing_gor.png", dpi=180)

    plt.figure(figsize=(7.8, 4.8))
    plt.semilogy(df["time_days"], np.abs(df["oil_mb_error_relative"]) + 1e-30, marker="o", label="Oil")
    plt.semilogy(df["time_days"], np.abs(df["water_mb_error_relative"]) + 1e-30, marker="s", label="Water")
    plt.semilogy(df["time_days"], np.abs(df["gas_mb_error_relative"]) + 1e-30, marker="^", label="Gas component")
    plt.xlabel("Time [days]")
    plt.ylabel("Relative material-balance error")
    plt.title("Material-balance diagnostics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "material_balance_error.png", dpi=180)

    # A final diagnostic of actual phase pressures.
    pw, po, pg = phase_pressures_black_oil(results["pressure"][-1], sw[-1], sg[-1], capillary)
    plt.figure(figsize=(7.8, 4.8))
    plt.plot(x, pw / bar, marker="o", label="water pressure")
    plt.plot(x, po / bar, marker="s", label="oil pressure")
    plt.plot(x, pg / bar, marker="^", label="gas pressure")
    plt.xlabel("x [m]")
    plt.ylabel("Pressure [bar]")
    plt.title("Final phase pressures")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "final_phase_pressures.png", dpi=180)

    print(f"Step 5A example finished. Results written to {out}")


if __name__ == "__main__":
    main()
