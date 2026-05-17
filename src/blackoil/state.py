from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class State1P:
    """Single-phase pressure state."""

    p: np.ndarray

    @classmethod
    def constant(cls, nx: int, pressure: float) -> "State1P":
        return cls(p=np.full(nx, float(pressure), dtype=float))

    def copy(self) -> "State1P":
        return State1P(p=self.p.copy())


@dataclass
class State2P:
    """Two-phase oil-water state.

    Primary variables are oil pressure and water saturation.
    Oil saturation is computed as So=1-Sw.
    """

    p: np.ndarray
    sw: np.ndarray

    @classmethod
    def constant(cls, nx: int, pressure: float, sw: float) -> "State2P":
        return cls(
            p=np.full(nx, float(pressure), dtype=float),
            sw=np.full(nx, float(sw), dtype=float),
        )

    @property
    def so(self) -> np.ndarray:
        return 1.0 - self.sw

    def copy(self) -> "State2P":
        return State2P(p=self.p.copy(), sw=self.sw.copy())


@dataclass
class State3P:
    """Three-phase live-oil black-oil state.

    Primary variables are oil pressure, water saturation, and free-gas
    saturation. Oil saturation is computed from the saturation constraint
    So = 1 - Sw - Sg.
    """

    p: np.ndarray
    sw: np.ndarray
    sg: np.ndarray

    @classmethod
    def constant(cls, nx: int, pressure: float, sw: float, sg: float) -> "State3P":
        return cls(
            p=np.full(nx, float(pressure), dtype=float),
            sw=np.full(nx, float(sw), dtype=float),
            sg=np.full(nx, float(sg), dtype=float),
        )

    @property
    def so(self) -> np.ndarray:
        return 1.0 - self.sw - self.sg

    def copy(self) -> "State3P":
        return State3P(p=self.p.copy(), sw=self.sw.copy(), sg=self.sg.copy())


@dataclass
class StateBlackOil:
    """Primary-variable state for Step 4B black-oil phase switching.

    The third primary variable is interpreted cell-wise:

    * saturated cell:     x = Sg,       Rs = Rs_sat(p)
    * undersaturated cell: x = Rs,      Sg = 0

    The Boolean ``is_saturated`` array therefore carries the phase-state map.
    This is a standard primary-variable switching idea, implemented here in a
    deliberately compact form suitable for verification and future extension.
    """

    p: np.ndarray
    sw: np.ndarray
    x: np.ndarray
    is_saturated: np.ndarray

    @classmethod
    def constant_undersaturated(
        cls, nx: int, pressure: float, sw: float, rs: float
    ) -> "StateBlackOil":
        return cls(
            p=np.full(nx, float(pressure), dtype=float),
            sw=np.full(nx, float(sw), dtype=float),
            x=np.full(nx, float(rs), dtype=float),
            is_saturated=np.zeros(nx, dtype=bool),
        )

    @classmethod
    def constant_saturated(
        cls, nx: int, pressure: float, sw: float, sg: float
    ) -> "StateBlackOil":
        return cls(
            p=np.full(nx, float(pressure), dtype=float),
            sw=np.full(nx, float(sw), dtype=float),
            x=np.full(nx, float(sg), dtype=float),
            is_saturated=np.ones(nx, dtype=bool),
        )

    def copy(self) -> "StateBlackOil":
        return StateBlackOil(
            p=self.p.copy(),
            sw=self.sw.copy(),
            x=self.x.copy(),
            is_saturated=self.is_saturated.copy(),
        )

    def physical(self, oil) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return physical ``Sw, So, Sg, Rs`` arrays for the current state."""
        rs_sat = oil.solution_gas_ratio(self.p)
        sg = np.where(self.is_saturated, self.x, 0.0)
        rs = np.where(self.is_saturated, rs_sat, self.x)
        so = 1.0 - self.sw - sg
        return self.sw.copy(), so, sg, rs

    def sg(self, oil=None) -> np.ndarray:
        """Return free-gas saturation; ``oil`` is accepted for API symmetry."""
        return np.where(self.is_saturated, self.x, 0.0)

    def rs(self, oil) -> np.ndarray:
        """Return dissolved solution gas-oil ratio."""
        rs_sat = oil.solution_gas_ratio(self.p)
        return np.where(self.is_saturated, rs_sat, self.x)

    def so(self, oil=None) -> np.ndarray:
        """Return oil saturation using the current free-gas saturation."""
        return 1.0 - self.sw - self.sg(oil)

    def phase_label(self) -> np.ndarray:
        return np.where(self.is_saturated, "saturated", "undersaturated")
