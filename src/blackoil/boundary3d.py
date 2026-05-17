from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class PressureBoundary3D:
    """Constant-state pressure boundary for structured 3D grids.

    The boundary is intentionally simple and reservoir-engineering oriented:
    it represents a ghost cell with prescribed oil pressure and saturations.
    Boundary fluxes are added as finite-volume outflow terms. Side names are
    ``left``, ``right``, ``front``, ``back``, ``top`` and ``bottom``.
    """

    side: str
    pressure: float
    sw: float = 0.2
    sg: float = 0.0
    rs: float | None = None

    def __post_init__(self) -> None:
        side = self.side.lower()
        if side not in {"left", "right", "front", "back", "top", "bottom"}:
            raise ValueError("3D pressure boundary side must be left/right/front/back/top/bottom")
        if self.pressure <= 0.0:
            raise ValueError("Boundary pressure must be positive")
        if not (0.0 <= self.sw <= 1.0 and 0.0 <= self.sg <= 1.0 and self.sw + self.sg <= 1.0):
            raise ValueError("Boundary saturations must be physically admissible")
        object.__setattr__(self, "side", side)


@dataclass
class BoundaryConditions3D:
    """Container for 3D finite-volume boundary conditions.

    No-flow is the default. Pressure boundaries can be used for aquifer-like
    tests or for verification against imposed external pressure states.
    """

    pressure_boundaries: list[PressureBoundary3D] = field(default_factory=list)

    @classmethod
    def no_flow(cls) -> "BoundaryConditions3D":
        return cls([])

    @classmethod
    def pressure(cls, boundaries: Iterable[PressureBoundary3D]) -> "BoundaryConditions3D":
        return cls(list(boundaries))

    def has_pressure_boundaries(self) -> bool:
        return bool(self.pressure_boundaries)
