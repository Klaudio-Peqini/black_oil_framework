from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import csv
import math
from typing import Iterable

import numpy as np

from .wells import ControlledWell
from .units import day, bar


@dataclass(frozen=True)
class WellControlEvent:
    """One scheduled operational event for a well.

    Parameters
    ----------
    time:
        Event time in seconds from the beginning of the simulation.
    name:
        Well name. Events with the same name update the same well.
    cell:
        Completed cell index. If omitted in later events, the previous cell is
        preserved.
    well_index:
        Productivity/injectivity index. If omitted in later events, the previous
        value is preserved.
    control:
        One of the controls supported by :class:`blackoil.wells.ControlledWell`:
        ``bhp``, ``water_rate``, ``oil_rate``, ``gas_rate``, ``liquid_rate`` or
        ``total_rate``. The special value ``shut`` closes the well.
    target:
        Control target in SI units. Rate targets follow the framework sign
        convention: positive injection and negative production. BHP targets are
        pressures in Pa.
    status:
        ``open`` or ``shut``. This provides an explicit status flag in addition
        to the special ``control='shut'`` convenience.
    group:
        Optional group name used only for diagnostics at this stage.
    """

    time: float
    name: str
    control: str
    target: float | None = None
    cell: int | None = None
    well_index: float | None = None
    min_bhp: float | None = None
    max_bhp: float | None = None
    status: str = "open"
    group: str | None = None

    def __post_init__(self):
        if self.time < -1.0e-12:
            raise ValueError("Schedule event time must be non-negative")
        if not self.name:
            raise ValueError("Schedule event requires a well name")
        status = self.status.lower()
        control = self.control.lower()
        if status not in {"open", "shut"}:
            raise ValueError("Well status must be 'open' or 'shut'")
        if control == "shut":
            object.__setattr__(self, "status", "shut")
        allowed = {"bhp", "water_rate", "oil_rate", "gas_rate", "liquid_rate", "total_rate", "shut"}
        if control not in allowed:
            raise ValueError(f"Unsupported scheduled control {self.control!r}")
        object.__setattr__(self, "control", control)
        object.__setattr__(self, "status", status if control != "shut" else "shut")


@dataclass
class ScheduledWellState:
    """Current schedule-resolved state of one well."""

    name: str
    cell: int
    well_index: float
    control: str
    target: float
    min_bhp: float | None = None
    max_bhp: float | None = None
    status: str = "open"
    group: str | None = None

    def to_controlled_well(self) -> ControlledWell | None:
        if self.status.lower() != "open" or self.control.lower() == "shut":
            return None
        return ControlledWell(
            name=self.name,
            cell=int(self.cell),
            control=self.control,
            target=float(self.target),
            well_index=float(self.well_index),
            min_bhp=self.min_bhp,
            max_bhp=self.max_bhp,
        )

    def apply(self, event: WellControlEvent) -> "ScheduledWellState":
        cell = self.cell if event.cell is None else int(event.cell)
        wi = self.well_index if event.well_index is None else float(event.well_index)
        control = self.control if event.control == "shut" else event.control
        target = self.target if event.target is None else float(event.target)
        min_bhp = self.min_bhp if event.min_bhp is None else float(event.min_bhp)
        max_bhp = self.max_bhp if event.max_bhp is None else float(event.max_bhp)
        group = self.group if event.group is None else event.group
        status = event.status
        return ScheduledWellState(
            name=self.name,
            cell=cell,
            well_index=wi,
            control=control,
            target=target,
            min_bhp=min_bhp,
            max_bhp=max_bhp,
            status=status,
            group=group,
        )


@dataclass
class WellSchedule:
    """Piecewise-constant multiwell production schedule.

    The schedule is intentionally minimal and transparent: at any time, events
    up to that time are replayed to obtain the active well list. This is more
    than fast enough for field-style research cases at the current scale and is
    convenient for restart/reproducibility.
    """

    events: list[WellControlEvent]
    report_times: list[float] | None = None

    def __post_init__(self):
        self.events = sorted(self.events, key=lambda e: (e.time, e.name))
        if self.report_times is not None:
            self.report_times = sorted(float(t) for t in self.report_times if t >= 0.0)

    @property
    def event_times(self) -> list[float]:
        vals = sorted({round(float(e.time), 12) for e in self.events})
        return [float(v) for v in vals]

    def milestone_times(self, t0: float, t1: float, include_report_times: bool = True) -> list[float]:
        times = [t for t in self.event_times if t0 + 1.0e-12 < t <= t1 + 1.0e-12]
        if include_report_times and self.report_times is not None:
            times += [t for t in self.report_times if t0 + 1.0e-12 < t <= t1 + 1.0e-12]
        return sorted(set(round(float(t), 12) for t in times))

    def next_milestone_after(self, t: float, t_final: float, include_report_times: bool = True) -> float:
        candidates = self.milestone_times(t, t_final, include_report_times=include_report_times)
        return float(candidates[0]) if candidates else float(t_final)

    def states_at(self, time: float) -> dict[str, ScheduledWellState]:
        states: dict[str, ScheduledWellState] = {}
        for ev in self.events:
            if ev.time > time + 1.0e-12:
                break
            if ev.name not in states:
                if ev.cell is None or ev.well_index is None:
                    raise ValueError(
                        f"First event for well {ev.name!r} must define cell and well_index"
                    )
                if ev.target is None and ev.status == "open" and ev.control != "shut":
                    raise ValueError(f"First open event for well {ev.name!r} must define a target")
                states[ev.name] = ScheduledWellState(
                    name=ev.name,
                    cell=int(ev.cell),
                    well_index=float(ev.well_index),
                    control="bhp" if ev.control == "shut" else ev.control,
                    target=float(0.0 if ev.target is None else ev.target),
                    min_bhp=ev.min_bhp,
                    max_bhp=ev.max_bhp,
                    status=ev.status,
                    group=ev.group,
                )
            else:
                states[ev.name] = states[ev.name].apply(ev)
        return states

    def wells_at(self, time: float) -> list[ControlledWell]:
        wells = []
        for st in self.states_at(time).values():
            w = st.to_controlled_well()
            if w is not None:
                wells.append(w)
        return wells

    def status_rows_at(self, time: float) -> list[dict]:
        rows = []
        for st in self.states_at(time).values():
            rows.append(
                {
                    "time": float(time),
                    "well": st.name,
                    "cell": int(st.cell),
                    "well_index": float(st.well_index),
                    "control": st.control,
                    "target": float(st.target),
                    "min_bhp": np.nan if st.min_bhp is None else float(st.min_bhp),
                    "max_bhp": np.nan if st.max_bhp is None else float(st.max_bhp),
                    "status": st.status,
                    "group": "" if st.group is None else st.group,
                }
            )
        return rows

    @classmethod
    def from_csv(cls, path: str | Path, *, time_unit: float = day, pressure_unit: float = bar) -> "WellSchedule":
        """Read a schedule CSV using reservoir-friendly units.

        Expected columns include ``time_days``, ``well``, ``control`` and
        ``target``. Optional columns are ``cell``, ``well_index``,
        ``min_bhp_bar``, ``max_bhp_bar``, ``status`` and ``group``. BHP targets
        and BHP limits are converted from bar to Pa; rates are converted from
        per-day to per-second.
        """
        path = Path(path)
        events: list[WellControlEvent] = []
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row or str(row.get("well", "")).strip().startswith("#"):
                    continue
                control = str(row.get("control", "")).strip().lower()
                target_raw = row.get("target", "")
                target = None if target_raw in (None, "") else float(target_raw)
                if target is not None:
                    target = target * pressure_unit if control == "bhp" else target / time_unit
                cell_raw = row.get("cell", "")
                wi_raw = row.get("well_index", "")
                min_raw = row.get("min_bhp_bar", "")
                max_raw = row.get("max_bhp_bar", "")
                events.append(
                    WellControlEvent(
                        time=float(row.get("time_days", 0.0)) * time_unit,
                        name=str(row.get("well", "")).strip(),
                        cell=None if cell_raw in (None, "") else int(float(cell_raw)),
                        well_index=None if wi_raw in (None, "") else float(wi_raw),
                        control=control,
                        target=target,
                        min_bhp=None if min_raw in (None, "") else float(min_raw) * pressure_unit,
                        max_bhp=None if max_raw in (None, "") else float(max_raw) * pressure_unit,
                        status=str(row.get("status", "open") or "open").strip().lower(),
                        group=(str(row.get("group", "")).strip() or None),
                    )
                )
        return cls(events=events)

    def to_csv(self, path: str | Path, *, time_unit: float = day, pressure_unit: float = bar) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "time_days",
            "well",
            "cell",
            "well_index",
            "control",
            "target",
            "min_bhp_bar",
            "max_bhp_bar",
            "status",
            "group",
        ]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for ev in self.events:
                target = ev.target
                target_out = "" if target is None else (target / pressure_unit if ev.control == "bhp" else target * time_unit)
                writer.writerow(
                    {
                        "time_days": ev.time / time_unit,
                        "well": ev.name,
                        "cell": "" if ev.cell is None else ev.cell,
                        "well_index": "" if ev.well_index is None else ev.well_index,
                        "control": ev.control,
                        "target": target_out,
                        "min_bhp_bar": "" if ev.min_bhp is None else ev.min_bhp / pressure_unit,
                        "max_bhp_bar": "" if ev.max_bhp is None else ev.max_bhp / pressure_unit,
                        "status": ev.status,
                        "group": "" if ev.group is None else ev.group,
                    }
                )
