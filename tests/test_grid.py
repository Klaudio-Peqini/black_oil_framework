import numpy as np
from blackoil import CartesianGrid1D, md_to_m2


def test_grid_geometry():
    grid = CartesianGrid1D(nx=10, length=100.0, area=2.0)
    assert grid.dx == 10.0
    assert np.allclose(grid.volumes, 20.0)
    assert grid.neighbors.shape == (9, 2)


def test_transmissibility_positive():
    grid = CartesianGrid1D(nx=5, length=50.0, area=1.0)
    t = grid.geometric_transmissibility(md_to_m2(100.0))
    assert t.shape == (4,)
    assert np.all(t > 0.0)
