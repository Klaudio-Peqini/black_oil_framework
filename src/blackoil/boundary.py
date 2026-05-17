from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class PressureBoundary:
    """Constant oil-pressure boundary for a 2D model.

    ``side`` is one of ``left``, ``right``, ``bottom`` or ``top``. The boundary
    state uses prescribed oil pressure and phase saturations. It is intentionally
    simple: it represents aquifer/pressure-support or pressure-depletion tests,
    while wells remain the preferred way to model field operations.
    """

    side: str
    pressure: float
    sw: float
    sg: float = 0.0
    rs: float | None = None


@dataclass
class BoundaryConditions2D:
    """Boundary-condition collection for Step 5C.

    The default is no-flow on all sides. Add ``PressureBoundary`` entries for
    pressure-support tests. Flux/rate boundaries can be added later using the
    same interface without changing the 2D residual.
    """

    pressure_boundaries: list[PressureBoundary] = field(default_factory=list)

    @classmethod
    def no_flow(cls) -> "BoundaryConditions2D":
        return cls([])

    def by_side(self) -> dict[str, list[PressureBoundary]]:
        out = {"left": [], "right": [], "bottom": [], "top": []}
        for bc in self.pressure_boundaries:
            side = bc.side.lower()
            if side not in out:
                raise ValueError(f"Unknown boundary side {bc.side!r}")
            out[side].append(bc)
        return out

    def has_pressure_boundaries(self) -> bool:
        return bool(self.pressure_boundaries)
