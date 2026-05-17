from __future__ import annotations

from dataclasses import dataclass
import numpy as np


class ZeroCapillaryPressure:
    """Zero capillary-pressure closure.

    This is still useful as a verification limit: with horizontal depth and
    zero capillary pressure, the Step 5A fluxes reduce to the Step 4C fluxes.
    """

    def pcow(self, sw: float | np.ndarray) -> np.ndarray:
        return np.zeros_like(np.asarray(sw, dtype=float))

    def pcgo(self, sg: float | np.ndarray) -> np.ndarray:
        return np.zeros_like(np.asarray(sg, dtype=float))


@dataclass
class LinearCapillaryPressure:
    """Simple bounded linear capillary-pressure closure.

    The model is intended for code verification and smooth early tests rather
    than field-calibrated capillary curves.

    ``pcow`` decreases with water saturation, while ``pcgo`` decreases with gas
    saturation. Pressures are in Pa.
    """

    swc: float = 0.18
    sor: float = 0.20
    sgc: float = 0.02
    pcow_max: float = 2.0e5
    pcgo_max: float = 1.5e5

    def _clip01(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, 0.0, 1.0)

    def pcow(self, sw: float | np.ndarray) -> np.ndarray:
        sw_arr = np.asarray(sw, dtype=float)
        denom = max(1.0 - self.swc - self.sor, 1.0e-12)
        se = self._clip01((sw_arr - self.swc) / denom)
        return self.pcow_max * (1.0 - se)

    def pcgo(self, sg: float | np.ndarray) -> np.ndarray:
        sg_arr = np.asarray(sg, dtype=float)
        denom = max(1.0 - self.swc - self.sor - self.sgc, 1.0e-12)
        se = self._clip01((sg_arr - self.sgc) / denom)
        return self.pcgo_max * (1.0 - se)


@dataclass
class BrooksCoreyCapillaryPressure:
    """Bounded Brooks-Corey-like water-oil and gas-oil capillary curves.

    The expressions are deliberately guarded near connate saturations because
    the early dense-Jacobian research simulator is not yet a production-scale
    complementarity solver. For water-oil:

        pcow = pe_w * (Se_w^{-1/lambda_w} - 1),

    with analogous gas-oil behavior. Results are clipped to ``pc*_max``.
    """

    swc: float = 0.18
    sor: float = 0.20
    sgc: float = 0.02
    pe_w: float = 5.0e4
    pe_g: float = 3.5e4
    lambda_w: float = 2.0
    lambda_g: float = 2.0
    pcow_max: float = 6.0e5
    pcgo_max: float = 5.0e5
    eps: float = 1.0e-4

    def pcow(self, sw: float | np.ndarray) -> np.ndarray:
        sw_arr = np.asarray(sw, dtype=float)
        denom = max(1.0 - self.swc - self.sor, 1.0e-12)
        se = np.clip((sw_arr - self.swc) / denom, self.eps, 1.0)
        pc = self.pe_w * (se ** (-1.0 / self.lambda_w) - 1.0)
        return np.clip(pc, 0.0, self.pcow_max)

    def pcgo(self, sg: float | np.ndarray) -> np.ndarray:
        sg_arr = np.asarray(sg, dtype=float)
        denom = max(1.0 - self.swc - self.sor - self.sgc, 1.0e-12)
        se = np.clip((sg_arr - self.sgc) / denom, self.eps, 1.0)
        pc = self.pe_g * (se ** (-1.0 / self.lambda_g) - 1.0)
        return np.clip(pc, 0.0, self.pcgo_max)
