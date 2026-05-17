from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CoreyWaterOilRelPerm:
    """Corey-type water-oil relative permeability model."""

    swc: float = 0.2
    sor: float = 0.2
    krw0: float = 0.3
    kro0: float = 1.0
    nw: float = 2.0
    no: float = 2.0

    def effective_water_saturation(self, sw: float | np.ndarray) -> np.ndarray:
        sw_arr = np.asarray(sw, dtype=float)
        denom = 1.0 - self.swc - self.sor
        if denom <= 0.0:
            raise ValueError("Invalid residual saturations: swc + sor must be < 1")
        swe = (sw_arr - self.swc) / denom
        return np.clip(swe, 0.0, 1.0)

    def krw(self, sw: float | np.ndarray) -> np.ndarray:
        swe = self.effective_water_saturation(sw)
        return self.krw0 * swe**self.nw

    def kro(self, sw: float | np.ndarray) -> np.ndarray:
        swe = self.effective_water_saturation(sw)
        return self.kro0 * (1.0 - swe) ** self.no

    def fractional_flow_water(
        self,
        sw: float | np.ndarray,
        mu_w: float | np.ndarray,
        mu_o: float | np.ndarray,
    ) -> np.ndarray:
        lam_w = self.krw(sw) / np.asarray(mu_w, dtype=float)
        lam_o = self.kro(sw) / np.asarray(mu_o, dtype=float)
        denom = lam_w + lam_o
        return np.divide(lam_w, denom, out=np.zeros_like(lam_w), where=denom > 0.0)


@dataclass
class CoreyThreePhaseRelPerm:
    """Compact Corey-type three-phase relative-permeability model.

    This is deliberately simple and robust for Step 4A. It is not meant to
    replace Stone I/Stone II models, but it gives a clean first live-oil
    simulator with water, oil, and gas mobilities.
    """

    swc: float = 0.18
    sor: float = 0.20
    sgc: float = 0.02
    krw0: float = 0.32
    kro0: float = 0.85
    krg0: float = 0.70
    nw: float = 2.4
    no: float = 2.0
    ng: float = 2.0

    def _denom(self) -> float:
        denom = 1.0 - self.swc - self.sor - self.sgc
        if denom <= 0.0:
            raise ValueError("Invalid residual saturations: swc + sor + sgc must be < 1")
        return denom

    def effective_saturations(
        self, sw: float | np.ndarray, sg: float | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sw_arr = np.asarray(sw, dtype=float)
        sg_arr = np.asarray(sg, dtype=float)
        so_arr = 1.0 - sw_arr - sg_arr
        denom = self._denom()
        swe = np.clip((sw_arr - self.swc) / denom, 0.0, 1.0)
        sge = np.clip((sg_arr - self.sgc) / denom, 0.0, 1.0)
        soe = np.clip((so_arr - self.sor) / denom, 0.0, 1.0)
        return swe, soe, sge

    def krw(self, sw: float | np.ndarray, sg: float | np.ndarray | None = None) -> np.ndarray:
        if sg is None:
            sg = np.zeros_like(np.asarray(sw, dtype=float))
        swe, _, _ = self.effective_saturations(sw, sg)
        return self.krw0 * swe**self.nw

    def kro(self, sw: float | np.ndarray, sg: float | np.ndarray) -> np.ndarray:
        _, soe, _ = self.effective_saturations(sw, sg)
        return self.kro0 * soe**self.no

    def krg(self, sw: float | np.ndarray, sg: float | np.ndarray) -> np.ndarray:
        _, _, sge = self.effective_saturations(sw, sg)
        return self.krg0 * sge**self.ng
