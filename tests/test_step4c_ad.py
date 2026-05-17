from pathlib import Path
import sys

import numpy as np
import pytest

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
    NewtonSolverWithJacobian,
    ConservativeLiveOilPhaseSwitchingSimulator,
    md_to_m2,
    bar,
    day,
)
from blackoil.black_oil_ad import JaxBlackOilADContext, residual_live_oil_phase_switching_conservative


def _small_case():
    grid = CartesianGrid1D(nx=4, length=200.0, area=500.0)
    p0 = 260.0 * bar
    rock = Rock(0.22, md_to_m2(100.0), compressibility=3.0e-10, p_ref=p0)
    table = BlackOilPVTTable.from_csv(ROOT / "data" / "pvt" / "live_oil_pvt.csv")
    water = TabulatedFluid("water", table, "Bw", "muw_pa_s", density_key="rhow_kg_m3")
    oil = TabulatedFluid("oil", table, "Bo", "muo_pa_s", density_key="rhoo_kg_m3", rs_key="Rs_sm3_sm3")
    gas = TabulatedFluid("gas", table, "Bg", "mug_pa_s", density_key="rhog_kg_m3")
    relperm = CoreyThreePhaseRelPerm()
    state = StateBlackOil.constant_undersaturated(grid.nx, p0, relperm.swc + 0.03, 140.0)
    wells = [BHPWell("P", grid.nx - 1, 150.0 * bar, 5.0e-15)]
    return grid, rock, water, oil, gas, relperm, state, wells


def test_conservative_residual_shape():
    grid, rock, water, oil, gas, relperm, state, wells = _small_case()
    x = np.concatenate([state.p, state.sw, state.x])
    r = residual_live_oil_phase_switching_conservative(x, state, 0.5 * day, grid, rock, water, oil, gas, relperm, wells)
    assert r.shape == (3 * grid.nx,)
    assert np.all(np.isfinite(r))


def test_jax_ad_jacobian_shape_and_finiteness():
    pytest.importorskip("jax")
    grid, rock, water, oil, gas, relperm, state, wells = _small_case()
    scales = np.concatenate([np.full(grid.nx, np.mean(state.p)), np.ones(grid.nx), np.full(grid.nx, oil.max_solution_gas_ratio())])
    ctx = JaxBlackOilADContext(grid, rock, water, oil, gas, relperm, state, wells, state.is_saturated.copy(), 0.5 * day, scales)
    y = np.concatenate([state.p, state.sw, state.x]) / scales
    j = ctx.jacobian(y)
    assert j.shape == (3 * grid.nx, 3 * grid.nx)
    assert np.all(np.isfinite(j))


def test_conservative_simulator_accepts_short_step():
    pytest.importorskip("jax")
    grid, rock, water, oil, gas, relperm, state, wells = _small_case()
    solver = NewtonSolverWithJacobian(tol=1.0e-7, max_iter=12, acceptable_tol=1.0e-2)
    sim = ConservativeLiveOilPhaseSwitchingSimulator(
        grid, rock, water, oil, gas, relperm, state, wells=wells, solver=solver, jacobian_mode="jax"
    )
    ok, report, _chg, _to_sat, _to_unsat = sim.try_step(0.1 * day, max_ds=0.2)
    assert ok
    assert report.converged
    assert np.all(np.isfinite(sim.state.p))
