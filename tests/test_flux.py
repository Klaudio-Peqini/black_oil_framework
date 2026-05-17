import numpy as np
from blackoil import CartesianGrid1D, md_to_m2, cp
from blackoil.flux import single_phase_face_flux


def test_single_phase_flux_direction():
    grid = CartesianGrid1D(nx=3, length=30.0, area=1.0)
    p = np.array([3.0e7, 2.0e7, 1.0e7])
    flux = single_phase_face_flux(grid, md_to_m2(100.0), p, mu=1.0 * cp, b=1.0)
    assert np.all(flux > 0.0)
