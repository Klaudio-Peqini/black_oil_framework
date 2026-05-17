from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blackoil import (
    CartesianGrid1D,
    Rock,
    BlackOilPVTTable,
    TabulatedFluid,
    CoreyThreePhaseRelPerm,
    StateBlackOil,
    BHPWell,
    ControlledWell,
    NewtonSolverWithJacobian,
    AdvancedBlackOilSimulator5A,
    LinearCapillaryPressure,
    ZeroCapillaryPressure,
    residual_live_oil_5a_gravity_capillary,
    md_to_m2,
    bar,
    day,
)
from blackoil.flux import three_phase_black_oil_face_fluxes_with_rs_gravity_capillary


def _small_case(depth=None, capillary=None):
    grid = CartesianGrid1D(nx=4, length=200.0, area=500.0, depth=0.0 if depth is None else depth)
    p0 = 260.0 * bar
    rock = Rock(0.22, md_to_m2(100.0), compressibility=3.0e-10, p_ref=p0)
    table = BlackOilPVTTable.from_csv(ROOT / "data" / "pvt" / "live_oil_pvt.csv")
    water = TabulatedFluid("water", table, "Bw", "muw_pa_s", density_key="rhow_kg_m3")
    oil = TabulatedFluid("oil", table, "Bo", "muo_pa_s", density_key="rhoo_kg_m3", rs_key="Rs_sm3_sm3")
    gas = TabulatedFluid("gas", table, "Bg", "mug_pa_s", density_key="rhog_kg_m3")
    relperm = CoreyThreePhaseRelPerm()
    state = StateBlackOil.constant_saturated(grid.nx, p0, relperm.swc + 0.04, 0.035)
    wells = [BHPWell("P", grid.nx - 1, 150.0 * bar, 4.0e-15)]
    return grid, rock, water, oil, gas, relperm, state, wells, capillary or ZeroCapillaryPressure()


def test_linear_capillary_pressure_is_finite_and_monotone():
    cap = LinearCapillaryPressure(pcow_max=2.0e5, pcgo_max=1.0e5)
    sw = np.linspace(0.18, 0.75, 8)
    sg = np.linspace(0.02, 0.50, 8)
    pcw = cap.pcow(sw)
    pcg = cap.pcgo(sg)
    assert np.all(np.isfinite(pcw))
    assert np.all(np.isfinite(pcg))
    assert pcw[0] >= pcw[-1]
    assert pcg[0] >= pcg[-1]


def test_gravity_flux_vanishes_for_hydrostatic_oil_limit():
    depth = np.array([1000.0, 1010.0, 1020.0])
    grid = CartesianGrid1D(nx=3, length=30.0, area=100.0, depth=depth)
    table = BlackOilPVTTable.from_csv(ROOT / "data" / "pvt" / "live_oil_pvt.csv")
    water = TabulatedFluid("water", table, "Bw", "muw_pa_s", density_key="rhow_kg_m3")
    oil = TabulatedFluid("oil", table, "Bo", "muo_pa_s", density_key="rhoo_kg_m3", rs_key="Rs_sm3_sm3")
    gas = TabulatedFluid("gas", table, "Bg", "mug_pa_s", density_key="rhog_kg_m3")
    relperm = CoreyThreePhaseRelPerm()
    p_ref = 220.0 * bar
    rho_o = float(oil.density(p_ref))
    p = p_ref + rho_o * 9.80665 * (depth - depth[0])
    sw = np.full(grid.nx, relperm.swc + 0.03)
    sg = np.full(grid.nx, 0.0)
    rs = np.full(grid.nx, 120.0)
    fw, fo, fg, fgc = three_phase_black_oil_face_fluxes_with_rs_gravity_capillary(
        grid, md_to_m2(100.0), p, sw, sg, rs, relperm, water, oil, gas, ZeroCapillaryPressure()
    )
    assert np.linalg.norm(fo) < 5.0e-8


def test_controlled_well_switches_to_bhp_limit():
    grid, rock, water, oil, gas, relperm, state, _wells, cap = _small_case()
    # Large negative liquid rate should require too low a BHP and therefore switch.
    well = ControlledWell("PCTRL", grid.nx - 1, "liquid_rate", -5000.0 / day, 1.0e-15, min_bhp=180.0 * bar)
    sim = AdvancedBlackOilSimulator5A(grid, rock, water, oil, gas, relperm, state, wells=[well], capillary=cap)
    active, log = sim.resolve_well_controls(state)
    assert active[0].__class__.__name__ == "BHPWell"
    assert np.isclose(active[0].bhp, 180.0 * bar)
    assert log[0]["active_type"] == "BHPWell"


def test_step5a_residual_shape():
    grid, rock, water, oil, gas, relperm, state, wells, cap = _small_case(capillary=LinearCapillaryPressure())
    x = np.concatenate([state.p, state.sw, state.x])
    r = residual_live_oil_5a_gravity_capillary(
        x, state, 0.2 * day, grid, rock, water, oil, gas, relperm, wells, capillary=cap
    )
    assert r.shape == (3 * grid.nx,)
    assert np.all(np.isfinite(r))


def test_step5a_simulator_accepts_short_step():
    depth = np.linspace(1200.0, 1225.0, 4)
    grid, rock, water, oil, gas, relperm, state, _wells, cap = _small_case(depth=depth, capillary=LinearCapillaryPressure())
    well = ControlledWell("PCTRL", grid.nx - 1, "liquid_rate", -8.0 / day, 3.0e-15, min_bhp=150.0 * bar)
    solver = NewtonSolverWithJacobian(tol=1.0e-7, max_iter=14, acceptable_tol=1.0e-3)
    sim = AdvancedBlackOilSimulator5A(
        grid, rock, water, oil, gas, relperm, state, wells=[well], solver=solver, capillary=cap
    )
    ok, report, _chg, _to_sat, _to_unsat = sim.try_step(0.05 * day, max_ds=0.2)
    assert ok
    assert report.converged
    assert np.all(np.isfinite(sim.state.p))
