from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import numpy as np

from .state import StateBlackOil


CUMULATIVE_FIELDS = [
    "cumulative_oil_produced",
    "cumulative_water_produced",
    "cumulative_free_gas_produced",
    "cumulative_gas_component_produced",
    "cumulative_oil_injected",
    "cumulative_water_injected",
    "cumulative_free_gas_injected",
    "cumulative_gas_component_injected",
]


def save_black_oil_restart(path: str | Path, simulator, time: float, *, metadata: dict[str, Any] | None = None) -> Path:
    """Save the black-oil state, cumulative totals, and simple metadata.

    The restart is a compressed ``.npz`` file so it is easy to inspect from
    Python and portable across machines. Heavy objects such as grids, PVT tables
    and schedules are intentionally *not* serialized here; the user recreates the
    simulator configuration and then loads this dynamic state into it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = simulator.state
    metadata_json = json.dumps(metadata or {}, sort_keys=True)
    cums = {name: float(getattr(simulator, name, 0.0)) for name in CUMULATIVE_FIELDS}
    np.savez_compressed(
        path,
        time=np.asarray(float(time)),
        pressure=state.p,
        sw=state.sw,
        x=state.x,
        is_saturated=state.is_saturated.astype(np.int8),
        metadata_json=np.asarray(metadata_json),
        **{name: np.asarray(value) for name, value in cums.items()},
    )
    return path


def load_black_oil_restart(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        metadata_json = str(data["metadata_json"].item()) if "metadata_json" in data else "{}"
        out = {
            "time": float(data["time"]),
            "state": StateBlackOil(
                p=np.asarray(data["pressure"], dtype=float),
                sw=np.asarray(data["sw"], dtype=float),
                x=np.asarray(data["x"], dtype=float),
                is_saturated=np.asarray(data["is_saturated"], dtype=bool),
            ),
            "metadata": json.loads(metadata_json),
            "cumulatives": {},
        }
        for name in CUMULATIVE_FIELDS:
            out["cumulatives"][name] = float(data[name]) if name in data else 0.0
    return out


def apply_black_oil_restart(simulator, restart: dict[str, Any]) -> float:
    """Apply a restart dictionary returned by :func:`load_black_oil_restart`."""
    simulator.state = restart["state"].copy()
    for name, value in restart.get("cumulatives", {}).items():
        if hasattr(simulator, name):
            setattr(simulator, name, float(value))
    return float(restart["time"])
