from pathlib import Path
import numpy as np

from blackoil import (
    BlackOilPVTTable,
    TabulatedFluid,
    StateBlackOil,
    LiveOilPhaseSwitchingSimulator,
    CartesianGrid1D,
    Rock,
    CoreyThreePhaseRelPerm,
    NewtonSolver,
    bar,
    md_to_m2,
)

ROOT = Path(__file__).resolve().parents[1]


def _oil():
    table = BlackOilPVTTable.from_csv(ROOT / "data" / "pvt" / "live_oil_pvt.csv")
    return TabulatedFluid("oil", table, "Bo", "muo_pa_s", rs_key="Rs_sm3_sm3")


def test_bubble_point_inverse_is_monotone_reasonable():
    oil = _oil()
    pb = oil.bubble_point_pressure(np.array([58.0, 118.0, 155.0]))
    assert np.all(np.diff(pb) > 0.0)
    assert 1.8e7 < pb[1] < 2.0e7


def test_state_black_oil_interpretation():
    oil = _oil()
    st = StateBlackOil.constant_undersaturated(4, 280.0 * bar, 0.22, 120.0)
    sw, so, sg, rs = st.physical(oil)
    assert np.allclose(sw, 0.22)
    assert np.allclose(so, 0.78)
    assert np.allclose(sg, 0.0)
    assert np.allclose(rs, 120.0)

    st2 = StateBlackOil.constant_saturated(4, 220.0 * bar, 0.22, 0.05)
    _, so2, sg2, rs2 = st2.physical(oil)
    assert np.allclose(so2, 0.73)
    assert np.allclose(sg2, 0.05)
    assert np.allclose(rs2, oil.solution_gas_ratio(st2.p))


def test_apply_phase_switching_creates_free_gas_when_oversaturated():
    table = BlackOilPVTTable.from_csv(ROOT / "data" / "pvt" / "live_oil_pvt.csv")
    water = TabulatedFluid("water", table, "Bw", "muw_pa_s")
    oil = TabulatedFluid("oil", table, "Bo", "muo_pa_s", rs_key="Rs_sm3_sm3")
    gas = TabulatedFluid("gas", table, "Bg", "mug_pa_s")
    grid = CartesianGrid1D(nx=3, length=30.0, area=10.0)
    rock = Rock(0.2, md_to_m2(100.0), 1.0e-10, 200.0 * bar)
    relperm = CoreyThreePhaseRelPerm()
    state = StateBlackOil.constant_undersaturated(3, 150.0 * bar, 0.22, 140.0)
    sim = LiveOilPhaseSwitchingSimulator(
        grid, rock, water, oil, gas, relperm, state, solver=NewtonSolver()
    )
    switched, to_sat, to_unsat = sim.apply_phase_switching(state)
    assert to_sat == 3
    assert to_unsat == 0
    assert np.all(switched.is_saturated)
    assert np.all(switched.sg(oil) > 0.0)
