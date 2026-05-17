from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .state import State2P
from .residual import residual_two_phase_oil_water, pack_two_phase, unpack_two_phase
from .nonlinear_solver import NewtonSolver, NewtonReport
from .wells import RateWell, BHPWell


@dataclass
class DeadOilStepInfo:
    """Diagnostics for one accepted dead-oil timestep."""

    time: float
    dt: float
    newton: NewtonReport
    min_pressure: float
    max_pressure: float
    min_sw: float
    max_sw: float
    oil_rate: float
    water_rate: float
    water_injection_rate: float
    cumulative_oil_produced: float
    cumulative_water_produced: float
    cumulative_water_injected: float
    recovery_factor: float
    oil_material_balance_error: float
    water_material_balance_error: float
    oil_material_balance_error_relative: float
    water_material_balance_error_relative: float


@dataclass
class DeadOilSimulator:
    """Fully implicit compressible dead-oil water-oil simulator.

    The primary variables are oil pressure and water saturation. The model is
    still two-phase, but unlike the pedagogical water-oil prototype it is meant
    to use pressure-dependent PVT data, rock compressibility, well controls,
    timestep rejection, and explicit material-balance diagnostics.
    """

    grid: object
    rock: object
    water: object
    oil: object
    relperm: object
    state: State2P
    wells: list = field(default_factory=list)
    solver: NewtonSolver = field(default_factory=NewtonSolver)
    cumulative_oil_produced: float = 0.0
    cumulative_water_produced: float = 0.0
    cumulative_oil_injected: float = 0.0
    cumulative_water_injected: float = 0.0

    def copy_state(self) -> State2P:
        return self.state.copy()

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        nx = self.grid.nx
        lower_p = np.full(nx, 1.0e5)
        upper_p = np.full(nx, 1.0e9)
        lower_sw = np.full(nx, self.relperm.swc + 1.0e-8)
        upper_sw = np.full(nx, 1.0 - self.relperm.sor - 1.0e-8)
        return np.concatenate([lower_p, lower_sw]), np.concatenate([upper_p, upper_sw])

    def component_in_place(self, state: State2P | None = None) -> tuple[float, float]:
        """Return stock-tank water and oil volumes in place."""
        st = self.state if state is None else state
        pv = self.rock.pore_volume(self.grid.volumes, st.p)
        bw = self.water.formation_volume_factor(st.p)
        bo = self.oil.formation_volume_factor(st.p)
        water = float(np.sum(pv * st.sw / bw))
        oil = float(np.sum(pv * st.so / bo))
        return water, oil

    def well_sources(self, state: State2P | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return phase source arrays at the supplied state.

        Positive values inject into the reservoir; negative values produce from
        the reservoir. Units are stock-tank m^3/s.
        """
        st = self.state if state is None else state
        p = st.p
        sw = st.sw
        bw = self.water.formation_volume_factor(p)
        bo = self.oil.formation_volume_factor(p)
        muw = self.water.viscosity(p)
        muo = self.oil.viscosity(p)
        krw = self.relperm.krw(sw)
        kro = self.relperm.kro(sw)
        qw = np.zeros(self.grid.nx, dtype=float)
        qo = np.zeros(self.grid.nx, dtype=float)
        for well in self.wells:
            if isinstance(well, RateWell):
                qwi, qoi = well.two_phase_sources(self.grid.nx)
            elif isinstance(well, BHPWell):
                qwi, qoi = well.two_phase_sources(p, sw, krw, kro, muw, muo, bw, bo)
            else:
                raise TypeError(f"Unsupported well type: {type(well)!r}")
            qw += qwi
            qo += qoi
        return qw, qo

    @staticmethod
    def _split_rates(q: np.ndarray) -> tuple[float, float]:
        injection = float(np.sum(np.maximum(q, 0.0)))
        production = float(np.sum(np.maximum(-q, 0.0)))
        return injection, production

    def try_step(self, dt: float, max_ds: float | None = None) -> tuple[bool, NewtonReport, float]:
        """Attempt one implicit step and update the state only on acceptance."""
        old = self.state.copy()
        x0 = pack_two_phase(old.p, old.sw)
        lower, upper = self.bounds()

        def func(x):
            return residual_two_phase_oil_water(
                x, old, dt, self.grid, self.rock, self.water, self.oil, self.relperm, self.wells
            )

        x_new, report = self.solver.solve(func, x0, lower=lower, upper=upper)
        p_new, sw_new = unpack_two_phase(x_new, self.grid.nx)
        ds_max = float(np.max(np.abs(sw_new - old.sw)))
        accepted = report.converged and (max_ds is None or ds_max <= max_ds)
        if accepted:
            self.state = State2P(p=p_new, sw=sw_new)
        return accepted, report, ds_max

    def run_adaptive(
        self,
        t_final: float,
        dt_initial: float,
        dt_min: float,
        dt_max: float,
        max_ds: float = 0.08,
        growth: float = 1.35,
        cut: float = 0.5,
    ) -> dict:
        """Run with timestep rejection and material-balance diagnostics."""
        times = [0.0]
        pressures = [self.state.p.copy()]
        sw_values = [self.state.sw.copy()]
        reports: list[DeadOilStepInfo] = []

        t = 0.0
        dt = dt_initial
        water0, oil0 = self.component_in_place()
        ooip0 = oil0

        while t < t_final - 1.0e-12:
            dt = min(dt, dt_max, t_final - t)
            old_state = self.state.copy()
            old_cums = (
                self.cumulative_oil_produced,
                self.cumulative_water_produced,
                self.cumulative_oil_injected,
                self.cumulative_water_injected,
            )

            accepted, report, ds_max = self.try_step(dt, max_ds=max_ds)
            while not accepted:
                self.state = old_state.copy()
                (
                    self.cumulative_oil_produced,
                    self.cumulative_water_produced,
                    self.cumulative_oil_injected,
                    self.cumulative_water_injected,
                ) = old_cums
                dt *= cut
                if dt < dt_min:
                    raise RuntimeError(
                        "Dead-oil timestep failed below dt_min: "
                        f"dt={dt:.6e}, residual={report.residual_norm:.3e}, ds_max={ds_max:.3e}"
                    )
                accepted, report, ds_max = self.try_step(dt, max_ds=max_ds)

            qw, qo = self.well_sources(self.state)
            winj_rate, wprod_rate = self._split_rates(qw)
            oinj_rate, oprod_rate = self._split_rates(qo)
            self.cumulative_water_injected += winj_rate * dt
            self.cumulative_water_produced += wprod_rate * dt
            self.cumulative_oil_injected += oinj_rate * dt
            self.cumulative_oil_produced += oprod_rate * dt

            t += dt
            water_now, oil_now = self.component_in_place()
            water_balance = water_now - water0 - self.cumulative_water_injected + self.cumulative_water_produced
            oil_balance = oil_now - oil0 - self.cumulative_oil_injected + self.cumulative_oil_produced
            water_scale = max(abs(water0), abs(water_now), self.cumulative_water_injected, self.cumulative_water_produced, 1.0)
            oil_scale = max(abs(oil0), abs(oil_now), self.cumulative_oil_injected, self.cumulative_oil_produced, 1.0)
            rf = self.cumulative_oil_produced / ooip0 if ooip0 > 0.0 else np.nan

            times.append(t)
            pressures.append(self.state.p.copy())
            sw_values.append(self.state.sw.copy())
            reports.append(
                DeadOilStepInfo(
                    time=t,
                    dt=dt,
                    newton=report,
                    min_pressure=float(np.min(self.state.p)),
                    max_pressure=float(np.max(self.state.p)),
                    min_sw=float(np.min(self.state.sw)),
                    max_sw=float(np.max(self.state.sw)),
                    oil_rate=oprod_rate,
                    water_rate=wprod_rate,
                    water_injection_rate=winj_rate,
                    cumulative_oil_produced=self.cumulative_oil_produced,
                    cumulative_water_produced=self.cumulative_water_produced,
                    cumulative_water_injected=self.cumulative_water_injected,
                    recovery_factor=rf,
                    oil_material_balance_error=oil_balance,
                    water_material_balance_error=water_balance,
                    oil_material_balance_error_relative=oil_balance / oil_scale,
                    water_material_balance_error_relative=water_balance / water_scale,
                )
            )

            # Increase only after robust Newton convergence and a gentle saturation move.
            if report.iterations <= max(2, self.solver.max_iter // 3) and ds_max < 0.5 * max_ds:
                dt = min(dt * growth, dt_max)
            elif report.iterations > 0.75 * self.solver.max_iter or ds_max > 0.8 * max_ds:
                dt = max(dt * cut, dt_min)

        return {
            "time": np.asarray(times),
            "pressure": np.asarray(pressures),
            "sw": np.asarray(sw_values),
            "reports": reports,
        }
