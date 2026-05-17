from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .state import State3P
from .residual import residual_live_oil_saturated, pack_three_phase, unpack_three_phase
from .nonlinear_solver import NewtonSolver, NewtonReport
from .wells import RateWell, BHPWell


@dataclass
class LiveOilStepInfo:
    """Diagnostics for one accepted saturated live-oil timestep."""

    time: float
    dt: float
    newton: NewtonReport
    min_pressure: float
    max_pressure: float
    min_sw: float
    max_sw: float
    min_sg: float
    max_sg: float
    min_so: float
    max_so: float
    oil_rate: float
    water_rate: float
    free_gas_rate: float
    gas_component_rate: float
    water_injection_rate: float
    cumulative_oil_produced: float
    cumulative_water_produced: float
    cumulative_free_gas_produced: float
    cumulative_gas_component_produced: float
    cumulative_water_injected: float
    recovery_factor: float
    producing_gor: float
    oil_material_balance_error: float
    water_material_balance_error: float
    gas_material_balance_error: float
    oil_material_balance_error_relative: float
    water_material_balance_error_relative: float
    gas_material_balance_error_relative: float


@dataclass
class LiveOilSaturatedSimulator:
    """Fully implicit saturated live-oil black-oil simulator.

    This is Step 4A of the framework. The primary variables are

        p_o, S_w, S_g,

    and the gas component equation contains both free gas and dissolved gas in
    oil through Rs(p). The model assumes the gas phase is present everywhere,
    therefore it does not yet implement bubble-point switching or phase
    disappearance. That more difficult complementarity logic should be added
    only after this saturated solver has been verified.
    """

    grid: object
    rock: object
    water: object
    oil: object
    gas: object
    relperm: object
    state: State3P
    wells: list = field(default_factory=list)
    solver: NewtonSolver = field(default_factory=NewtonSolver)
    cumulative_oil_produced: float = 0.0
    cumulative_water_produced: float = 0.0
    cumulative_free_gas_produced: float = 0.0
    cumulative_gas_component_produced: float = 0.0
    cumulative_oil_injected: float = 0.0
    cumulative_water_injected: float = 0.0
    cumulative_free_gas_injected: float = 0.0
    cumulative_gas_component_injected: float = 0.0

    def copy_state(self) -> State3P:
        return self.state.copy()

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        nx = self.grid.nx
        lower_p = np.full(nx, 1.0e5)
        upper_p = np.full(nx, 1.0e9)
        lower_sw = np.full(nx, self.relperm.swc + 1.0e-8)
        upper_sw = np.full(nx, 1.0 - self.relperm.sor - self.relperm.sgc - 1.0e-8)
        lower_sg = np.full(nx, self.relperm.sgc + 1.0e-8)
        upper_sg = np.full(nx, 1.0 - self.relperm.sor - self.relperm.swc - 1.0e-8)
        return (
            np.concatenate([lower_p, lower_sw, lower_sg]),
            np.concatenate([upper_p, upper_sw, upper_sg]),
        )

    def component_in_place(self, state: State3P | None = None) -> tuple[float, float, float]:
        """Return water, oil, and gas-component stock-tank volumes in place."""
        st = self.state if state is None else state
        pv = self.rock.pore_volume(self.grid.volumes, st.p)
        bw = self.water.formation_volume_factor(st.p)
        bo = self.oil.formation_volume_factor(st.p)
        bg = self.gas.formation_volume_factor(st.p)
        rs = self.oil.solution_gas_ratio(st.p)
        water = float(np.sum(pv * st.sw / bw))
        oil = float(np.sum(pv * st.so / bo))
        gas_component = float(np.sum(pv * (rs * st.so / bo + st.sg / bg)))
        return water, oil, gas_component

    def well_sources(self, state: State3P | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return water, oil, free-gas, and gas-component source arrays.

        Positive values inject into the reservoir; negative values produce from
        the reservoir. Units are stock-tank m^3/s for oil/water and stock-tank
        gas m^3/s for the gas component.
        """
        st = self.state if state is None else state
        p = st.p
        sw = st.sw
        sg = st.sg
        bw = self.water.formation_volume_factor(p)
        bo = self.oil.formation_volume_factor(p)
        bg = self.gas.formation_volume_factor(p)
        muw = self.water.viscosity(p)
        muo = self.oil.viscosity(p)
        mug = self.gas.viscosity(p)
        rs = self.oil.solution_gas_ratio(p)
        krw = self.relperm.krw(sw, sg)
        kro = self.relperm.kro(sw, sg)
        krg = self.relperm.krg(sw, sg)
        qw = np.zeros(self.grid.nx, dtype=float)
        qo = np.zeros(self.grid.nx, dtype=float)
        qg_free = np.zeros(self.grid.nx, dtype=float)
        for well in self.wells:
            if isinstance(well, RateWell):
                qwi, qoi, qgi = well.three_phase_sources(self.grid.nx)
            elif isinstance(well, BHPWell):
                qwi, qoi, qgi = well.three_phase_sources(
                    p, sw, sg, krw, kro, krg, muw, muo, mug, bw, bo, bg
                )
            else:
                raise TypeError(f"Unsupported well type: {type(well)!r}")
            qw += qwi
            qo += qoi
            qg_free += qgi
        qg_component = qg_free + rs * qo
        return qw, qo, qg_free, qg_component

    @staticmethod
    def _split_rates(q: np.ndarray) -> tuple[float, float]:
        injection = float(np.sum(np.maximum(q, 0.0)))
        production = float(np.sum(np.maximum(-q, 0.0)))
        return injection, production

    def _valid_saturation_state(self, sw: np.ndarray, sg: np.ndarray) -> bool:
        so = 1.0 - sw - sg
        return bool(
            np.all(sw >= self.relperm.swc - 1.0e-7)
            and np.all(sg >= self.relperm.sgc - 1.0e-7)
            and np.all(so >= self.relperm.sor - 1.0e-7)
        )

    def try_step(self, dt: float, max_ds: float | None = None) -> tuple[bool, NewtonReport, float]:
        """Attempt one implicit step and update the state only on acceptance."""
        old = self.state.copy()
        x0 = pack_three_phase(old.p, old.sw, old.sg)
        lower, upper = self.bounds()

        def func(x):
            return residual_live_oil_saturated(
                x, old, dt, self.grid, self.rock, self.water, self.oil, self.gas, self.relperm, self.wells
            )

        x_new, report = self.solver.solve(func, x0, lower=lower, upper=upper)
        p_new, sw_new, sg_new = unpack_three_phase(x_new, self.grid.nx)
        ds_max = float(max(np.max(np.abs(sw_new - old.sw)), np.max(np.abs(sg_new - old.sg))))
        accepted = (
            report.converged
            and self._valid_saturation_state(sw_new, sg_new)
            and (max_ds is None or ds_max <= max_ds)
        )
        if accepted:
            self.state = State3P(p=p_new, sw=sw_new, sg=sg_new)
        return accepted, report, ds_max

    def run_adaptive(
        self,
        t_final: float,
        dt_initial: float,
        dt_min: float,
        dt_max: float,
        max_ds: float = 0.06,
        growth: float = 1.25,
        cut: float = 0.5,
    ) -> dict:
        """Run with timestep rejection and three-component material balance."""
        times = [0.0]
        pressures = [self.state.p.copy()]
        sw_values = [self.state.sw.copy()]
        sg_values = [self.state.sg.copy()]
        so_values = [self.state.so.copy()]
        reports: list[LiveOilStepInfo] = []

        t = 0.0
        dt = dt_initial
        water0, oil0, gas0 = self.component_in_place()
        ooip0 = oil0

        while t < t_final - 1.0e-12:
            dt = min(dt, dt_max, t_final - t)
            old_state = self.state.copy()
            old_cums = (
                self.cumulative_oil_produced,
                self.cumulative_water_produced,
                self.cumulative_free_gas_produced,
                self.cumulative_gas_component_produced,
                self.cumulative_oil_injected,
                self.cumulative_water_injected,
                self.cumulative_free_gas_injected,
                self.cumulative_gas_component_injected,
            )

            accepted, report, ds_max = self.try_step(dt, max_ds=max_ds)
            while not accepted:
                self.state = old_state.copy()
                (
                    self.cumulative_oil_produced,
                    self.cumulative_water_produced,
                    self.cumulative_free_gas_produced,
                    self.cumulative_gas_component_produced,
                    self.cumulative_oil_injected,
                    self.cumulative_water_injected,
                    self.cumulative_free_gas_injected,
                    self.cumulative_gas_component_injected,
                ) = old_cums
                dt *= cut
                if dt < dt_min:
                    raise RuntimeError(
                        "Live-oil timestep failed below dt_min: "
                        f"dt={dt:.6e}, residual={report.residual_norm:.3e}, ds_max={ds_max:.3e}"
                    )
                accepted, report, ds_max = self.try_step(dt, max_ds=max_ds)

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
            sw_values.append(self.state.sw.copy())
            sg_values.append(self.state.sg.copy())
            so_values.append(self.state.so.copy())
            reports.append(
                LiveOilStepInfo(
                    time=t,
                    dt=dt,
                    newton=report,
                    min_pressure=float(np.min(self.state.p)),
                    max_pressure=float(np.max(self.state.p)),
                    min_sw=float(np.min(self.state.sw)),
                    max_sw=float(np.max(self.state.sw)),
                    min_sg=float(np.min(self.state.sg)),
                    max_sg=float(np.max(self.state.sg)),
                    min_so=float(np.min(self.state.so)),
                    max_so=float(np.max(self.state.so)),
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
                )
            )

            if report.iterations <= max(2, self.solver.max_iter // 3) and ds_max < 0.5 * max_ds:
                dt = min(dt * growth, dt_max)
            elif report.iterations > 0.75 * self.solver.max_iter or ds_max > 0.8 * max_ds:
                dt = max(dt * cut, dt_min)

        return {
            "time": np.asarray(times),
            "pressure": np.asarray(pressures),
            "sw": np.asarray(sw_values),
            "sg": np.asarray(sg_values),
            "so": np.asarray(so_values),
            "reports": reports,
        }
