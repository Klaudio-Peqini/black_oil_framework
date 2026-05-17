from pathlib import Path
import numpy as np

from blackoil import BlackOilPVTTable, TabulatedFluid


def test_tabulated_fluid_interpolation_and_clamping():
    pressure = np.array([1.0, 2.0, 3.0])
    table = BlackOilPVTTable(pressure=pressure, B=np.array([1.2, 1.1, 1.0]), mu=np.array([3.0, 4.0, 5.0]))
    fluid = TabulatedFluid("oil", table, b_key="B", mu_key="mu", rho_ref=800.0)

    assert np.isclose(fluid.formation_volume_factor(2.5), 1.05)
    assert np.isclose(fluid.viscosity(2.5), 4.5)
    assert np.isclose(fluid.formation_volume_factor(0.0), 1.2)
    assert np.isclose(fluid.formation_volume_factor(4.0), 1.0)


def test_dead_oil_csv_table_exists_and_loads():
    root = Path(__file__).resolve().parents[1]
    table = BlackOilPVTTable.from_csv(root / "data" / "pvt" / "dead_oil_pvt.csv")
    assert "Bo" in table.keys
    assert "Bw" in table.keys
    assert "muo_pa_s" in table.keys
    assert "muw_pa_s" in table.keys
