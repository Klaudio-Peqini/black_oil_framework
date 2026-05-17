import numpy as np

from blackoil import (
    CartesianGrid1D,
    Rock,
    BlackOilPVTTable,
    TabulatedFluid,
    CoreyThreePhaseRelPerm,
    State3P,
    BHPWell,
    RateWell,
    md_to_m2,
    bar,
)
from blackoil.residual import pack_three_phase, residual_live_oil_saturated


def make_live_oil_objects(nx=5):
    ptab = np.array([100.0, 200.0, 300.0]) * bar
    table = BlackOilPVTTable(
        ptab,
        Bw=np.array([1.02, 1.01, 1.00]),
        Bo=np.array([1.15, 1.25, 1.34]),
        Bg=np.array([0.010, 0.005, 0.0035]),
        muw_pa_s=np.array([0.00055, 0.00052, 0.00050]),
        muo_pa_s=np.array([0.0030, 0.0017, 0.0012]),
        mug_pa_s=np.array([0.000016, 0.000022, 0.000030]),
        Rs_sm3_sm3=np.array([60.0, 125.0, 170.0]),
    )
    water = TabulatedFluid("water", table, "Bw", "muw_pa_s")
    oil = TabulatedFluid("oil", table, "Bo", "muo_pa_s", rs_key="Rs_sm3_sm3")
    gas = TabulatedFluid("gas", table, "Bg", "mug_pa_s")
    grid = CartesianGrid1D(nx=nx, length=100.0, area=10.0)
    rock = Rock(0.22, md_to_m2(100.0), compressibility=3.0e-10, p_ref=200.0 * bar)
    relperm = CoreyThreePhaseRelPerm(swc=0.18, sor=0.20, sgc=0.02)
    return grid, rock, water, oil, gas, relperm


def test_solution_gas_ratio_interpolates():
    _, _, _, oil, _, _ = make_live_oil_objects()
    rs = oil.solution_gas_ratio(np.array([150.0, 250.0]) * bar)
    assert np.all(rs > 60.0)
    assert np.all(rs < 170.0)
    assert rs[1] > rs[0]


def test_three_phase_relperm_shapes_and_bounds():
    relperm = CoreyThreePhaseRelPerm(swc=0.18, sor=0.20, sgc=0.02)
    sw = np.array([0.20, 0.30, 0.40])
    sg = np.array([0.04, 0.08, 0.12])
    assert relperm.krw(sw, sg).shape == sw.shape
    assert relperm.kro(sw, sg).shape == sw.shape
    assert relperm.krg(sw, sg).shape == sw.shape
    assert np.all(relperm.krw(sw, sg) >= 0.0)
    assert np.all(relperm.kro(sw, sg) >= 0.0)
    assert np.all(relperm.krg(sw, sg) >= 0.0)


def test_live_oil_residual_shape():
    grid, rock, water, oil, gas, relperm = make_live_oil_objects(nx=4)
    old = State3P.constant(grid.nx, pressure=220.0 * bar, sw=0.22, sg=0.05)
    x = pack_three_phase(old.p, old.sw, old.sg)
    wells = [RateWell("INJ", 0, "water", 1.0e-5), BHPWell("PROD", grid.nx - 1, 150.0 * bar, 1.0e-13)]
    r = residual_live_oil_saturated(x, old, 2.0 * 86400.0, grid, rock, water, oil, gas, relperm, wells)
    assert r.shape == (3 * grid.nx,)
    assert np.all(np.isfinite(r))
