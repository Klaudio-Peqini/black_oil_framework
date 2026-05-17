from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SlightlyCompressibleFluid:
    """Slightly compressible phase model.

    Parameters
    ----------
    name:
        Phase name.
    mu_ref:
        Viscosity at reference pressure, Pa s.
    b_ref:
        Formation volume factor at reference pressure.
    c_b:
        Compressibility-like coefficient for formation volume factor, 1/Pa.
        The model uses B(p)=B_ref*exp(-c_b*(p-p_ref)).
    c_mu:
        Viscosity pressure coefficient, 1/Pa.
    rho_ref:
        Density at reference pressure, kg/m^3.
    p_ref:
        Reference pressure, Pa.
    """

    name: str
    mu_ref: float
    b_ref: float = 1.0
    c_b: float = 0.0
    c_mu: float = 0.0
    rho_ref: float = 1000.0
    p_ref: float = 0.0

    def formation_volume_factor(self, p: float | np.ndarray) -> np.ndarray:
        p_arr = np.asarray(p, dtype=float)
        return self.b_ref * np.exp(-self.c_b * (p_arr - self.p_ref))

    def viscosity(self, p: float | np.ndarray) -> np.ndarray:
        p_arr = np.asarray(p, dtype=float)
        return self.mu_ref * np.exp(self.c_mu * (p_arr - self.p_ref))

    def density(self, p: float | np.ndarray) -> np.ndarray:
        b = self.formation_volume_factor(p)
        return self.rho_ref / b


@dataclass
class TabulatedFluid:
    """Fluid property model backed by a pressure PVT table.

    This is the preferred fluid model for the dead-oil stage. The governing
    equations do not change relative to the two-phase oil-water simulator, but
    the closure laws become data driven:

        B_alpha = B_alpha(p),     mu_alpha = mu_alpha(p)

    Parameters
    ----------
    name:
        Fluid name, for example ``"water"`` or ``"oil"``.
    table:
        A :class:`BlackOilPVTTable` instance.
    b_key:
        Column name used for the formation volume factor.
    mu_key:
        Column name used for viscosity in Pa s.
    rho_ref:
        Surface/reference density in kg/m^3. The simple density relation used
        here is rho(p)=rho_ref/B(p), which is sufficient for the next dead-oil
        implementation stage. A dedicated density table can be added later.
    density_key:
        Optional table column for density. If provided, it overrides rho_ref/B.
    rs_key:
        Optional table column for solution gas-oil ratio. This is unused in the
        dead-oil model, but becomes central in the live-oil black-oil stage.
        The value is treated as stock-tank gas volume per stock-tank oil volume.
    """

    name: str
    table: "BlackOilPVTTable"
    b_key: str
    mu_key: str
    rho_ref: float = 1000.0
    density_key: str | None = None
    rs_key: str | None = None

    def formation_volume_factor(self, p: float | np.ndarray) -> np.ndarray:
        return self.table(self.b_key, p)

    def viscosity(self, p: float | np.ndarray) -> np.ndarray:
        return self.table(self.mu_key, p)

    def density(self, p: float | np.ndarray) -> np.ndarray:
        if self.density_key is not None:
            return self.table(self.density_key, p)
        return self.rho_ref / self.formation_volume_factor(p)

    def solution_gas_ratio(self, p: float | np.ndarray) -> np.ndarray:
        """Return Rs(p), or zero if the fluid has no dissolved-gas column.

        For live-oil black-oil calculations, this method is expected to be
        called on the oil object. Water and gas objects normally return zero.
        """
        p_arr = np.asarray(p, dtype=float)
        if self.rs_key is None:
            return np.zeros_like(p_arr, dtype=float)
        return self.table(self.rs_key, p_arr)

    def bubble_point_pressure(self, rs: float | np.ndarray) -> np.ndarray:
        """Return the bubble-point pressure associated with a dissolved gas ratio.

        This is the inverse of the saturated Rs(p) table. It is used by the
        Step 4B phase-state logic to determine whether an oil cell should be
        interpreted as undersaturated or saturated.
        """
        rs_arr = np.asarray(rs, dtype=float)
        if self.rs_key is None:
            return np.zeros_like(rs_arr, dtype=float)
        return self.table.pressure_for_solution_gas_ratio(self.rs_key, rs_arr)

    def max_solution_gas_ratio(self) -> float:
        """Return the largest tabulated saturated solution gas-oil ratio."""
        if self.rs_key is None:
            return 0.0
        return float(np.max(self.table.columns[self.rs_key]))


class BlackOilPVTTable:
    """Simple table interpolation for black-oil PVT data.

    This class is intentionally compact. It supports the future live-oil stage,
    where Bo, Bg, muo, mug, and Rs are pressure-table functions.
    """

    def __init__(self, pressure: np.ndarray, **columns: np.ndarray) -> None:
        self.pressure = np.asarray(pressure, dtype=float)
        if self.pressure.ndim != 1:
            raise ValueError("pressure must be one-dimensional")
        order = np.argsort(self.pressure)
        self.pressure = self.pressure[order]
        self.columns: dict[str, np.ndarray] = {}
        for key, value in columns.items():
            arr = np.asarray(value, dtype=float)[order]
            if arr.shape != self.pressure.shape:
                raise ValueError(f"column {key!r} has incompatible shape")
            self.columns[key] = arr

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self.columns.keys())

    @classmethod
    def from_csv(cls, filename: str, pressure_key: str = "pressure_pa") -> "BlackOilPVTTable":
        """Read a pressure PVT table from a CSV file.

        The CSV must contain one pressure column and any number of property
        columns, for example ``Bo``, ``Bw``, ``muo_pa_s`` and ``muw_pa_s``.
        """
        import pandas as pd

        df = pd.read_csv(filename)
        if pressure_key not in df.columns:
            raise KeyError(f"pressure column {pressure_key!r} was not found in {filename}")
        columns = {key: df[key].to_numpy(dtype=float) for key in df.columns if key != pressure_key}
        return cls(pressure=df[pressure_key].to_numpy(dtype=float), **columns)


    def pressure_for_solution_gas_ratio(self, key: str, rs: float | np.ndarray) -> np.ndarray:
        """Return saturation/bubble-point pressure corresponding to Rs.

        The method inverts a monotonic Rs column, which is adequate for the
        Step 4B phase-state logic. Values outside the table range are clamped
        to the pressure range, consistently with property interpolation.
        """
        if key not in self.columns:
            raise KeyError(f"PVT column {key!r} not available; available columns are {self.keys}")
        rs_arr = np.asarray(rs, dtype=float)
        values = np.asarray(self.columns[key], dtype=float)
        # np.interp requires increasing x-values. Most black-oil Rs tables are
        # increasing with pressure, but sorting by Rs makes the routine robust.
        order = np.argsort(values)
        values_sorted = values[order]
        pressure_sorted = self.pressure[order]
        rs_clip = np.clip(rs_arr, float(values_sorted[0]), float(values_sorted[-1]))
        return np.interp(rs_clip, values_sorted, pressure_sorted)

    def __call__(self, key: str, p: float | np.ndarray) -> np.ndarray:
        if key not in self.columns:
            raise KeyError(f"PVT column {key!r} not available; available columns are {self.keys}")
        p_arr = np.asarray(p, dtype=float)
        p_min = float(self.pressure[0])
        p_max = float(self.pressure[-1])
        # Clamp outside the table range. This is safer than silent linear
        # extrapolation for the early simulator versions and keeps Newton stable.
        p_clip = np.clip(p_arr, p_min, p_max)
        return np.interp(p_clip, self.pressure, self.columns[key])
