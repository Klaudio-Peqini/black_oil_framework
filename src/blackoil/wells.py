from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RateWell:
    """Fixed-rate source/sink well.

    Parameters
    ----------
    phase:
        "water", "oil", "gas", or "single".
    rate:
        Surface-volume source rate in m^3/s. Positive injects into the reservoir;
        negative produces from the reservoir.
    """

    name: str
    cell: int
    phase: str
    rate: float

    def single_phase_source(self, nx: int) -> np.ndarray:
        q = np.zeros(nx, dtype=float)
        if self.phase not in {"single", "water", "oil", "gas"}:
            raise ValueError(f"Unknown phase for rate well: {self.phase}")
        q[self.cell] += self.rate
        return q

    def two_phase_sources(self, nx: int) -> tuple[np.ndarray, np.ndarray]:
        qw = np.zeros(nx, dtype=float)
        qo = np.zeros(nx, dtype=float)
        if self.phase == "water":
            qw[self.cell] += self.rate
        elif self.phase == "oil":
            qo[self.cell] += self.rate
        elif self.phase == "single":
            qw[self.cell] += self.rate
        elif self.phase == "gas":
            raise ValueError("Gas rate wells require the live-oil three-phase model")
        else:
            raise ValueError(f"Unknown phase for rate well: {self.phase}")
        return qw, qo

    def three_phase_sources(self, nx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return water, oil, and free-gas surface-volume source arrays."""
        qw = np.zeros(nx, dtype=float)
        qo = np.zeros(nx, dtype=float)
        qg = np.zeros(nx, dtype=float)
        if self.phase == "water":
            qw[self.cell] += self.rate
        elif self.phase == "oil":
            qo[self.cell] += self.rate
        elif self.phase == "gas":
            qg[self.cell] += self.rate
        elif self.phase == "single":
            qw[self.cell] += self.rate
        else:
            raise ValueError(f"Unknown phase for rate well: {self.phase}")
        return qw, qo, qg


@dataclass
class BHPWell:
    """Bottom-hole-pressure well model.

    The sign convention follows the conservation equation source term:

        q > 0: injection into reservoir
        q < 0: production from reservoir

    For a producer, choose bhp lower than the reservoir pressure.
    """

    name: str
    cell: int
    bhp: float
    well_index: float

    def single_phase_source(self, p: np.ndarray, mu: np.ndarray | float, b: np.ndarray | float = 1.0) -> np.ndarray:
        q = np.zeros_like(p, dtype=float)
        mu_arr = np.asarray(mu, dtype=float)
        b_arr = np.asarray(b, dtype=float)
        mu_c = mu_arr[self.cell] if mu_arr.ndim else float(mu_arr)
        b_c = b_arr[self.cell] if b_arr.ndim else float(b_arr)
        q[self.cell] += self.well_index / (mu_c * b_c) * (self.bhp - p[self.cell])
        return q

    def two_phase_sources(
        self,
        p: np.ndarray,
        sw: np.ndarray,
        krw: np.ndarray,
        kro: np.ndarray,
        muw: np.ndarray,
        muo: np.ndarray,
        bw: np.ndarray,
        bo: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        qw = np.zeros_like(p, dtype=float)
        qo = np.zeros_like(p, dtype=float)
        c = self.cell
        drawdown = self.bhp - p[c]
        qw[c] += self.well_index * krw[c] / (muw[c] * bw[c]) * drawdown
        qo[c] += self.well_index * kro[c] / (muo[c] * bo[c]) * drawdown
        return qw, qo

    def three_phase_sources(
        self,
        p: np.ndarray,
        sw: np.ndarray,
        sg: np.ndarray,
        krw: np.ndarray,
        kro: np.ndarray,
        krg: np.ndarray,
        muw: np.ndarray,
        muo: np.ndarray,
        mug: np.ndarray,
        bw: np.ndarray,
        bo: np.ndarray,
        bg: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return water, oil, and free-gas surface-volume source arrays.

        Positive values inject into the reservoir; negative values produce.
        Dissolved gas associated with oil flow is added by the black-oil
        residual/simulator through Rs*q_o.
        """
        qw = np.zeros_like(p, dtype=float)
        qo = np.zeros_like(p, dtype=float)
        qg = np.zeros_like(p, dtype=float)
        c = self.cell
        drawdown = self.bhp - p[c]
        qw[c] += self.well_index * krw[c] / (muw[c] * bw[c]) * drawdown
        qo[c] += self.well_index * kro[c] / (muo[c] * bo[c]) * drawdown
        qg[c] += self.well_index * krg[c] / (mug[c] * bg[c]) * drawdown
        return qw, qo, qg


@dataclass
class MultiRateWell:
    """Fixed multi-phase surface-rate well.

    The sign convention is the same as for all wells in the framework:
    positive rates inject into the reservoir and negative rates produce from it.
    Rates are surface volumes in m^3/s for liquid phases and sm^3/s for gas.
    """

    name: str
    cell: int
    water_rate: float = 0.0
    oil_rate: float = 0.0
    gas_rate: float = 0.0

    def three_phase_sources(self, nx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        qw = np.zeros(nx, dtype=float)
        qo = np.zeros(nx, dtype=float)
        qg = np.zeros(nx, dtype=float)
        qw[self.cell] += self.water_rate
        qo[self.cell] += self.oil_rate
        qg[self.cell] += self.gas_rate
        return qw, qo, qg


@dataclass
class ControlledWell:
    """Reservoir well with rate/BHP switching.

    Parameters
    ----------
    control:
        One of ``"bhp"``, ``"water_rate"``, ``"oil_rate"``, ``"gas_rate"``,
        ``"liquid_rate"`` or ``"total_rate"``.
    target:
        Target value for the selected control. Rate targets use the framework
        sign convention: positive injection, negative production. BHP targets
        are pressures in Pa.
    min_bhp, max_bhp:
        Optional pressure limits. Production rate controls switch to ``min_bhp``
        if the requested rate would require a lower BHP. Injection controls
        switch to ``max_bhp`` if the requested rate would require a higher BHP.

    The active mode is resolved at the start of a timestep from the old state.
    This avoids introducing discontinuous control switching inside a Newton
    iteration, while still giving realistic operational behavior.
    """

    name: str
    cell: int
    control: str
    target: float
    well_index: float
    min_bhp: float | None = None
    max_bhp: float | None = None

    def _phase_mobilities(
        self,
        krw: np.ndarray,
        kro: np.ndarray,
        krg: np.ndarray,
        muw: np.ndarray,
        muo: np.ndarray,
        mug: np.ndarray,
        bw: np.ndarray,
        bo: np.ndarray,
        bg: np.ndarray,
    ) -> tuple[float, float, float]:
        c = self.cell
        mw = self.well_index * krw[c] / max(muw[c] * bw[c], 1.0e-30)
        mo = self.well_index * kro[c] / max(muo[c] * bo[c], 1.0e-30)
        mg = self.well_index * krg[c] / max(mug[c] * bg[c], 1.0e-30)
        return float(mw), float(mo), float(mg)

    def active_well(
        self,
        p_water: np.ndarray,
        p_oil: np.ndarray,
        p_gas: np.ndarray,
        krw: np.ndarray,
        kro: np.ndarray,
        krg: np.ndarray,
        muw: np.ndarray,
        muo: np.ndarray,
        mug: np.ndarray,
        bw: np.ndarray,
        bo: np.ndarray,
        bg: np.ndarray,
    ):
        ctrl = self.control.lower()
        c = self.cell
        if ctrl == "bhp":
            return BHPWell(self.name, c, self.target, self.well_index)

        mw, mo, mg = self._phase_mobilities(krw, kro, krg, muw, muo, mug, bw, bo, bg)
        target = float(self.target)

        if ctrl == "water_rate":
            bhp_est = p_water[c] + target / max(mw, 1.0e-30)
            active = RateWell(self.name, c, "water", target)
        elif ctrl == "oil_rate":
            bhp_est = p_oil[c] + target / max(mo, 1.0e-30)
            active = RateWell(self.name, c, "oil", target)
        elif ctrl == "gas_rate":
            bhp_est = p_gas[c] + target / max(mg, 1.0e-30)
            active = RateWell(self.name, c, "gas", target)
        elif ctrl == "liquid_rate":
            mt = max(mw + mo, 1.0e-30)
            bhp_est = (target + mw * p_water[c] + mo * p_oil[c]) / mt
            active = MultiRateWell(
                self.name,
                c,
                water_rate=target * mw / mt,
                oil_rate=target * mo / mt,
                gas_rate=0.0,
            )
        elif ctrl == "total_rate":
            mt = max(mw + mo + mg, 1.0e-30)
            bhp_est = (target + mw * p_water[c] + mo * p_oil[c] + mg * p_gas[c]) / mt
            active = MultiRateWell(
                self.name,
                c,
                water_rate=target * mw / mt,
                oil_rate=target * mo / mt,
                gas_rate=target * mg / mt,
            )
        else:
            raise ValueError(f"Unsupported well control {self.control!r}")

        if target < 0.0 and self.min_bhp is not None and bhp_est < self.min_bhp:
            return BHPWell(self.name, c, self.min_bhp, self.well_index)
        if target > 0.0 and self.max_bhp is not None and bhp_est > self.max_bhp:
            return BHPWell(self.name, c, self.max_bhp, self.well_index)
        return active
