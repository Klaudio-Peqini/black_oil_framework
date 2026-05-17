from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv

import numpy as np

from .black_oil_5c import HeterogeneousBlackOilSimulator5C, BlackOil2DStepInfo
from .schedule import WellSchedule
from .restart import save_black_oil_restart, load_black_oil_restart, apply_black_oil_restart


@dataclass
class ScheduledBlackOilSimulator5D(HeterogeneousBlackOilSimulator5C):
    """Step 5D simulator with field-style schedules and restart support.

    The flow model is inherited from Step 5C. Step 5D adds operational realism:
    well targets can change at scheduled times, wells can be opened or shut, the
    adaptive timestep is forced to land exactly on schedule/report milestones,
    active control histories are recorded, and restart files can be written and
    reloaded.
    """

    schedule: WellSchedule | None = None
    schedule_history: list[dict] = field(default_factory=list)
    current_time: float = 0.0

    def set_wells_from_schedule(self, time: float) -> None:
        if self.schedule is not None:
            self.wells = self.schedule.wells_at(time)
            for row in self.schedule.status_rows_at(time):
                self.schedule_history.append(row)

    def save_restart(self, path: str | Path, *, time: float | None = None, metadata: dict | None = None) -> Path:
        t = self.current_time if time is None else float(time)
        meta = {"stage": "5D", "current_time": t}
        if metadata:
            meta.update(metadata)
        return save_black_oil_restart(path, self, t, metadata=meta)

    def load_restart_into_self(self, path: str | Path) -> float:
        restart = load_black_oil_restart(path)
        t = apply_black_oil_restart(self, restart)
        self.current_time = t
        self.set_wells_from_schedule(t)
        return t

    def run_scheduled(
        self,
        t_final: float,
        dt_initial: float,
        dt_min: float,
        dt_max: float,
        *,
        t_start: float | None = None,
        max_ds: float = 0.06,
        growth: float = 1.25,
        cut: float = 0.5,
        restart_dir: str | Path | None = None,
        restart_interval: float | None = None,
        write_restart_at_report_times: bool = False,
    ) -> dict:
        t = self.current_time if t_start is None else float(t_start)
        self.current_time = t
        self.set_wells_from_schedule(t)

        times = [t]
        pressures = [self.state.p.copy()]
        sw0_state, so0_state, sg0_state, rs0_state = self.state.physical(self.oil)
        sw_values = [sw0_state.copy()]
        sg_values = [sg0_state.copy()]
        so_values = [so0_state.copy()]
        rs_values = [rs0_state.copy()]
        phase_values = [self.state.is_saturated.copy()]
        reports: list[BlackOil2DStepInfo] = []
        restart_files: list[str] = []
        schedule_events: list[dict] = []

        water0, oil0, gas0 = self.component_in_place()
        ooip0 = oil0
        dt = dt_initial
        next_restart_time = (t + restart_interval) if restart_interval is not None else None

        while t < t_final - 1.0e-12:
            self.set_wells_from_schedule(t)
            next_milestone = t_final
            if self.schedule is not None:
                next_milestone = min(next_milestone, self.schedule.next_milestone_after(t, t_final))
            if next_restart_time is not None:
                next_milestone = min(next_milestone, next_restart_time)
            dt = min(dt, dt_max, next_milestone - t)
            if dt <= 1.0e-14:
                # Landed on a milestone; refresh controls and advance the marker.
                self.set_wells_from_schedule(t)
                if next_restart_time is not None and abs(t - next_restart_time) < 1.0e-10:
                    next_restart_time += restart_interval
                continue

            old_state = self.state.copy()
            old_cums = tuple(float(getattr(self, name)) for name in [
                "cumulative_oil_produced",
                "cumulative_water_produced",
                "cumulative_free_gas_produced",
                "cumulative_gas_component_produced",
                "cumulative_oil_injected",
                "cumulative_water_injected",
                "cumulative_free_gas_injected",
                "cumulative_gas_component_injected",
            ])
            accepted, report, change_indicator, to_sat, to_unsat = self.try_step(dt, max_ds=max_ds)
            while not accepted:
                self.state = old_state.copy()
                for name, value in zip([
                    "cumulative_oil_produced",
                    "cumulative_water_produced",
                    "cumulative_free_gas_produced",
                    "cumulative_gas_component_produced",
                    "cumulative_oil_injected",
                    "cumulative_water_injected",
                    "cumulative_free_gas_injected",
                    "cumulative_gas_component_injected",
                ], old_cums):
                    setattr(self, name, value)
                dt *= cut
                if dt < dt_min:
                    raise RuntimeError(
                        "Scheduled 2D black-oil timestep failed below dt_min: "
                        f"dt={dt:.6e}, residual={report.residual_norm:.3e}, change={change_indicator:.3e}"
                    )
                accepted, report, change_indicator, to_sat, to_unsat = self.try_step(dt, max_ds=max_ds)

            qw, qo, qg_free, qg_component = self.well_sources(self.state)
            winj_rate, wprod_rate = self._split_rates(qw)
            oinj_rate, oprod_rate = self._split_rates(qo)
            gfree_inj_rate, gfree_prod_rate = self._split_rates(qg_free)
            gcomp_inj_rate, gcomp_prod_rate = self._split_rates(qg_component)
            self.cumulative_water_injected += winj_rate * dt
            self.cumulative_water_produced += wprod_rate * dt
            self.cumulative_oil_injected += oinj_rate * dt
            self.cumulative_oil_produced += oprod_rate * dt
            self.cumulative_free_gas_injected += gfree_inj_rate * dt
            self.cumulative_free_gas_produced += gfree_prod_rate * dt
            self.cumulative_gas_component_injected += gcomp_inj_rate * dt
            self.cumulative_gas_component_produced += gcomp_prod_rate * dt

            t += dt
            self.current_time = t
            self.set_wells_from_schedule(t)
            sw, so, sg, rs = self.state.physical(self.oil)
            water_now, oil_now, gas_now = self.component_in_place()
            water_balance = water_now - water0 - self.cumulative_water_injected + self.cumulative_water_produced
            oil_balance = oil_now - oil0 - self.cumulative_oil_injected + self.cumulative_oil_produced
            gas_balance = gas_now - gas0 - self.cumulative_gas_component_injected + self.cumulative_gas_component_produced
            water_scale = max(abs(water0), abs(water_now), self.cumulative_water_injected, self.cumulative_water_produced, 1.0)
            oil_scale = max(abs(oil0), abs(oil_now), self.cumulative_oil_injected, self.cumulative_oil_produced, 1.0)
            gas_scale = max(abs(gas0), abs(gas_now), self.cumulative_gas_component_injected, self.cumulative_gas_component_produced, 1.0)
            rf = self.cumulative_oil_produced / ooip0 if ooip0 > 0.0 else np.nan
            gor = gcomp_prod_rate / (oprod_rate + 1.0e-30)

            times.append(t)
            pressures.append(self.state.p.copy())
            sw_values.append(sw.copy())
            sg_values.append(sg.copy())
            so_values.append(so.copy())
            rs_values.append(rs.copy())
            phase_values.append(self.state.is_saturated.copy())
            reports.append(
                BlackOil2DStepInfo(
                    time=t,
                    dt=dt,
                    newton=report,
                    min_pressure=float(np.min(self.state.p)),
                    max_pressure=float(np.max(self.state.p)),
                    min_sw=float(np.min(sw)),
                    max_sw=float(np.max(sw)),
                    min_sg=float(np.min(sg)),
                    max_sg=float(np.max(sg)),
                    min_so=float(np.min(so)),
                    max_so=float(np.max(so)),
                    min_rs=float(np.min(rs)),
                    max_rs=float(np.max(rs)),
                    saturated_cells=int(np.count_nonzero(self.state.is_saturated)),
                    undersaturated_cells=int(self.n - np.count_nonzero(self.state.is_saturated)),
                    switched_to_saturated=to_sat,
                    switched_to_undersaturated=to_unsat,
                    oil_rate=oprod_rate,
                    water_rate=wprod_rate,
                    free_gas_rate=gfree_prod_rate,
                    gas_component_rate=gcomp_prod_rate,
                    water_injection_rate=winj_rate,
                    cumulative_oil_produced=self.cumulative_oil_produced,
                    cumulative_water_produced=self.cumulative_water_produced,
                    cumulative_free_gas_produced=self.cumulative_free_gas_produced,
                    cumulative_gas_component_produced=self.cumulative_gas_component_produced,
                    cumulative_water_injected=self.cumulative_water_injected,
                    recovery_factor=rf,
                    producing_gor=gor,
                    oil_material_balance_error=oil_balance,
                    water_material_balance_error=water_balance,
                    gas_material_balance_error=gas_balance,
                    oil_material_balance_error_relative=oil_balance / oil_scale,
                    water_material_balance_error_relative=water_balance / water_scale,
                    gas_material_balance_error_relative=gas_balance / gas_scale,
                    linear_iterations_total=getattr(report, "linear_iterations_total", 0),
                    jacobian_nnz=getattr(report, "jacobian_nnz_last", 0),
                    jacobian_colors=getattr(report, "jacobian_colors", 0),
                )
            )

            if restart_dir is not None and (
                (next_restart_time is not None and abs(t - next_restart_time) < 1.0e-8)
                or (write_restart_at_report_times and self.schedule is not None and any(abs(t - rt) < 1.0e-8 for rt in (self.schedule.report_times or [])))
            ):
                restart_path = Path(restart_dir) / f"restart_t{t:.6e}.npz"
                self.save_restart(restart_path, time=t, metadata={"reason": "scheduled_run"})
                restart_files.append(str(restart_path))
                if next_restart_time is not None and abs(t - next_restart_time) < 1.0e-8:
                    next_restart_time += restart_interval

            if self.schedule is not None:
                # Record compact schedule event diagnostics only at exact event times.
                if any(abs(t - et) < 1.0e-8 for et in self.schedule.event_times):
                    for row in self.schedule.status_rows_at(t):
                        schedule_events.append(row)

            if report.iterations <= max(2, self.solver.max_iter // 3) and change_indicator < 0.5 * max_ds:
                dt = min(dt * growth, dt_max)
            elif report.iterations > 0.75 * self.solver.max_iter or change_indicator > 0.8 * max_ds:
                dt = max(dt * cut, dt_min)

        return {
            "time": np.asarray(times),
            "pressure": np.asarray(pressures),
            "sw": np.asarray(sw_values),
            "sg": np.asarray(sg_values),
            "so": np.asarray(so_values),
            "rs": np.asarray(rs_values),
            "is_saturated": np.asarray(phase_values, dtype=bool),
            "reports": reports,
            "restart_files": restart_files,
            "schedule_events": schedule_events,
            "schedule_history": self.schedule_history,
        }
