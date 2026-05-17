from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .state import State1P, State2P
from .residual import residual_single_phase, residual_two_phase_oil_water, pack_two_phase, unpack_two_phase
from .nonlinear_solver import NewtonSolver, NewtonReport
from .wells import BHPWell


@dataclass
class SinglePhaseStepInfo:
    time: float
    dt: float
    newton: NewtonReport
    min_pressure: float
    max_pressure: float


@dataclass
class SinglePhaseSimulator:
    grid: object
    rock: object
    fluid: object
    state: State1P
    wells: list = field(default_factory=list)
    solver: NewtonSolver = field(default_factory=NewtonSolver)

    def step(self, dt: float) -> NewtonReport:
        old = self.state.copy()

        def func(x):
            return residual_single_phase(x, old, dt, self.grid, self.rock, self.fluid, self.wells)

        x_new, report = self.solver.solve(func, old.p)
        self.state = State1P(p=x_new)
        return report

    def run(self, t_final: float, dt: float) -> dict:
        times = [0.0]
        pressures = [self.state.p.copy()]
        reports: list[SinglePhaseStepInfo] = []
        t = 0.0
        while t < t_final - 1.0e-12:
            step_dt = min(dt, t_final - t)
            report = self.step(step_dt)
            t += step_dt
            times.append(t)
            pressures.append(self.state.p.copy())
            reports.append(
                SinglePhaseStepInfo(
                    time=t,
                    dt=step_dt,
                    newton=report,
                    min_pressure=float(np.min(self.state.p)),
                    max_pressure=float(np.max(self.state.p)),
                )
            )
            if not report.converged:
                raise RuntimeError(f"Newton failed at time {t:.6e}; residual={report.residual_norm:.3e}")
        return {"time": np.asarray(times), "pressure": np.asarray(pressures), "reports": reports}


@dataclass
class TwoPhaseOilWaterStepInfo:
    time: float
    dt: float
    newton: NewtonReport
    min_pressure: float
    max_pressure: float
    min_sw: float
    max_sw: float
    oil_rate: float
    water_rate: float
    cumulative_oil: float
    cumulative_water: float
    recovery_factor: float


@dataclass
class TwoPhaseOilWaterSimulator:
    grid: object
    rock: object
    water: object
    oil: object
    relperm: object
    state: State2P
    wells: list = field(default_factory=list)
    solver: NewtonSolver = field(default_factory=NewtonSolver)
    cumulative_oil: float = 0.0
    cumulative_water: float = 0.0

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        nx = self.grid.nx
        lower_p = np.full(nx, 1.0e5)
        upper_p = np.full(nx, 1.0e9)
        lower_sw = np.full(nx, self.relperm.swc + 1.0e-8)
        upper_sw = np.full(nx, 1.0 - self.relperm.sor - 1.0e-8)
        return np.concatenate([lower_p, lower_sw]), np.concatenate([upper_p, upper_sw])

    def step(self, dt: float) -> NewtonReport:
        old = self.state.copy()
        x0 = pack_two_phase(old.p, old.sw)
        lower, upper = self.bounds()

        def func(x):
            return residual_two_phase_oil_water(
                x, old, dt, self.grid, self.rock, self.water, self.oil, self.relperm, self.wells
            )

        x_new, report = self.solver.solve(func, x0, lower=lower, upper=upper)
        p, sw = unpack_two_phase(x_new, self.grid.nx)
        self.state = State2P(p=p, sw=sw)
        return report

    def bhp_well_rates(self) -> tuple[float, float]:
        """Return total stock-tank production rates from BHP wells.

        Positive returned values mean production out of the reservoir.
        """
        p = self.state.p
        sw = self.state.sw
        bw = self.water.formation_volume_factor(p)
        bo = self.oil.formation_volume_factor(p)
        muw = self.water.viscosity(p)
        muo = self.oil.viscosity(p)
        krw = self.relperm.krw(sw)
        kro = self.relperm.kro(sw)
        water_prod = 0.0
        oil_prod = 0.0
        for well in self.wells:
            if isinstance(well, BHPWell):
                qw, qo = well.two_phase_sources(p, sw, krw, kro, muw, muo, bw, bo)
                water_prod += max(0.0, -float(np.sum(qw)))
                oil_prod += max(0.0, -float(np.sum(qo)))
        return oil_prod, water_prod

    def initial_oil_in_place(self) -> float:
        pv = self.rock.pore_volume(self.grid.volumes, self.state.p)
        bo = self.oil.formation_volume_factor(self.state.p)
        return float(np.sum(pv * self.state.so / bo))

    def run(self, t_final: float, dt: float) -> dict:
        times = [0.0]
        pressures = [self.state.p.copy()]
        sw_values = [self.state.sw.copy()]
        reports: list[TwoPhaseOilWaterStepInfo] = []
        t = 0.0
        ooip0 = self.initial_oil_in_place()

        while t < t_final - 1.0e-12:
            step_dt = min(dt, t_final - t)
            report = self.step(step_dt)
            oil_rate, water_rate = self.bhp_well_rates()
            self.cumulative_oil += oil_rate * step_dt
            self.cumulative_water += water_rate * step_dt
            t += step_dt

            rf = self.cumulative_oil / ooip0 if ooip0 > 0.0 else np.nan
            times.append(t)
            pressures.append(self.state.p.copy())
            sw_values.append(self.state.sw.copy())
            reports.append(
                TwoPhaseOilWaterStepInfo(
                    time=t,
                    dt=step_dt,
                    newton=report,
                    min_pressure=float(np.min(self.state.p)),
                    max_pressure=float(np.max(self.state.p)),
                    min_sw=float(np.min(self.state.sw)),
                    max_sw=float(np.max(self.state.sw)),
                    oil_rate=oil_rate,
                    water_rate=water_rate,
                    cumulative_oil=self.cumulative_oil,
                    cumulative_water=self.cumulative_water,
                    recovery_factor=rf,
                )
            )
            if not report.converged:
                raise RuntimeError(f"Newton failed at time {t:.6e}; residual={report.residual_norm:.3e}")

        return {
            "time": np.asarray(times),
            "pressure": np.asarray(pressures),
            "sw": np.asarray(sw_values),
            "reports": reports,
        }
