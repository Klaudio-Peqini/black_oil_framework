from blackoil import (
    CartesianGrid1D,
    Rock,
    SlightlyCompressibleFluid,
    CoreyWaterOilRelPerm,
    State1P,
    State2P,
    cp,
    md_to_m2,
    bar,
)
from blackoil.residual import residual_single_phase, residual_two_phase_oil_water, pack_two_phase


def test_residual_shapes():
    grid = CartesianGrid1D(nx=4, length=40.0)
    p0 = 200.0 * bar
    rock = Rock(0.2, md_to_m2(100.0), 1e-10, p0)
    water = SlightlyCompressibleFluid("water", 1.0 * cp, p_ref=p0)
    oil = SlightlyCompressibleFluid("oil", 3.0 * cp, b_ref=1.2, p_ref=p0)
    rp = CoreyWaterOilRelPerm()

    s1 = State1P.constant(grid.nx, p0)
    r1 = residual_single_phase(s1.p, s1, 86400.0, grid, rock, water)
    assert r1.shape == (grid.nx,)

    s2 = State2P.constant(grid.nx, p0, 0.25)
    x = pack_two_phase(s2.p, s2.sw)
    r2 = residual_two_phase_oil_water(x, s2, 86400.0, grid, rock, water, oil, rp)
    assert r2.shape == (2 * grid.nx,)
