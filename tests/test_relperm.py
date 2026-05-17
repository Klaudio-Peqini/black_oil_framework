import numpy as np
from blackoil import CoreyWaterOilRelPerm


def test_corey_endpoints():
    rp = CoreyWaterOilRelPerm(swc=0.2, sor=0.2, krw0=0.3, kro0=0.9)
    assert np.isclose(rp.krw(0.2), 0.0)
    assert np.isclose(rp.kro(0.8), 0.0)
    assert np.isclose(rp.krw(0.8), 0.3)
    assert np.isclose(rp.kro(0.2), 0.9)
