from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy


@dataclass(frozen=True)
class FieldWellControlEvent3D:
    """Schedule event for a multi-completion 3D field well."""

    time: float
    name: str
    control: str | None = None
    target: float | None = None
    status: str | None = None
    min_bhp: float | None = None
    max_bhp: float | None = None

    def __post_init__(self) -> None:
        if self.time < -1.0e-12:
            raise ValueError("event time must be non-negative")
        if not self.name:
            raise ValueError("event requires a well name")
        if self.control is not None:
            ctrl = self.control.lower()
            if ctrl not in {"bhp", "water_rate", "oil_rate", "gas_rate", "liquid_rate", "total_rate", "shut"}:
                raise ValueError(f"unsupported 3D well control {self.control!r}")
            object.__setattr__(self, "control", ctrl)
        if self.status is not None:
            stat = self.status.lower()
            if stat not in {"open", "shut"}:
                raise ValueError("well status must be open or shut")
            object.__setattr__(self, "status", stat)


@dataclass
class FieldWellSchedule3D:
    """Piecewise-constant schedule for :class:`FieldWell3D` objects.

    The schedule stores a base well list and a sequence of events. At a given
    time, events are replayed to obtain the active field controls. This mirrors
    the Step 5D schedule logic while preserving multi-completion wells.
    """

    base_wells: list
    events: list[FieldWellControlEvent3D]
    report_times: list[float] | None = None

    def __post_init__(self) -> None:
        self.events = sorted(self.events, key=lambda e: (e.time, e.name))
        if self.report_times is not None:
            self.report_times = sorted(float(t) for t in self.report_times if t >= 0.0)

    @property
    def event_times(self) -> list[float]:
        return sorted({round(float(e.time), 12) for e in self.events})

    def milestone_times(self, t0: float, t1: float, include_report_times: bool = True) -> list[float]:
        times = [t for t in self.event_times if t0 + 1.0e-12 < t <= t1 + 1.0e-12]
        if include_report_times and self.report_times is not None:
            times += [t for t in self.report_times if t0 + 1.0e-12 < t <= t1 + 1.0e-12]
        return sorted(set(round(float(t), 12) for t in times))

    def next_milestone_after(self, t: float, t_final: float, include_report_times: bool = True) -> float:
        vals = self.milestone_times(t, t_final, include_report_times=include_report_times)
        return float(vals[0]) if vals else float(t_final)

    def wells_at(self, time: float) -> list:
        wells = deepcopy(self.base_wells)
        by_name = {w.name: w for w in wells}
        for ev in self.events:
            if ev.time > time + 1.0e-12:
                break
            if ev.name not in by_name:
                continue
            w = by_name[ev.name]
            if ev.control == "shut":
                w.status = "shut"
            elif ev.control is not None:
                w.control = ev.control
            if ev.target is not None:
                w.target = float(ev.target)
            if ev.status is not None:
                w.status = ev.status
            if ev.min_bhp is not None:
                w.min_bhp = float(ev.min_bhp)
            if ev.max_bhp is not None:
                w.max_bhp = float(ev.max_bhp)
        return wells

    def status_rows_at(self, time: float) -> list[dict]:
        rows = []
        for w in self.wells_at(time):
            rows.append({
                "time": float(time),
                "well": w.name,
                "type": w.well_type,
                "control": w.control,
                "target": float(w.target),
                "status": w.status,
                "min_bhp": float("nan") if w.min_bhp is None else float(w.min_bhp),
                "max_bhp": float("nan") if w.max_bhp is None else float(w.max_bhp),
                "open_completions": int(sum(c.is_open for c in w.completions)),
            })
        return rows
