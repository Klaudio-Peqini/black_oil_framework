from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdaptiveTimeStep:
    """Simple adaptive timestep controller."""

    dt: float
    dt_min: float
    dt_max: float
    growth: float = 1.4
    cut: float = 0.5

    def success(self, easy: bool = False) -> float:
        factor = self.growth if easy else min(1.15, self.growth)
        self.dt = min(self.dt_max, self.dt * factor)
        return self.dt

    def failure(self) -> float:
        self.dt = max(self.dt_min, self.dt * self.cut)
        return self.dt
