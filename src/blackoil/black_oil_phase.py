from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .state import StateBlackOil
from .flux import three_phase_black_oil_face_fluxes_with_rs, divergence_from_face_flux
from .nonlinear_solver import NewtonSolver, NewtonReport
from .wells import RateWell, BHPWell


def pack_black_oil_primary(p: np.ndarray, sw: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(p, dtype=float), np.asarray(sw, dtype=float), np.asarray(x, dtype=float)])


def unpack_black_oil_primary(x: np.ndarray, nx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(x[:nx], dtype=float)
    sw = np.asarray(x[nx:2 * nx], dtype=float)
    third = np.asarray(x[2 * nx:], dtype=float)
    return p, sw, third


def interpret_black_oil_primary(
    p: np.ndarray,
    sw: np.ndarray,
    third: np.ndarray,
    is_saturated: np.ndarray,
    oil,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert primary variables to physical Sw, So, Sg, Rs fields.

    In saturated cells, ``third`` is free-gas saturation. In undersaturated
    cells, ``third`` is the dissolved solution gas-oil ratio Rs.
    """
    p = np.asarray(p, dtype=float)
    sw = np.asarray(sw, dtype=float)
    third = np.asarray(third, dtype=float)
    sat = np.asarray(is_saturated, dtype=bool)
    rs_sat = oil.solution_gas_ratio(p)
    sg = np.where(sat, third, 0.0)
    rs = np.where(sat, rs_sat, third)
    so = 1.0 - sw - sg
    return sw, so, sg, rs


def residual_live_oil_phase_switching(
    x: np.ndarray,
    old: StateBlackOil,
    dt: float,
    grid,
    rock,
    water,
    oil,
    gas,
    relperm,
    wells=None,
    active_saturated: np.ndarray | None = None,
) -> np.ndarray:
    """Fully implicit black-oil residual with fixed phase interpretation.

    This is the Step 4B residual. During one Newton solve, the active phase map
    is held fixed. In cells marked as saturated, the third unknown is Sg and
    Rs=Rs_sat(p). In undersaturated cells, the third unknown is Rs and Sg=0.
    After Newton convergence, the simulator applies a conservative phase-state
    update that may switch cells between these two interpretations.
    """
    wells = wells or []
    active_saturated = old.is_saturated if active_saturated is None else np.asarray(active_saturated, dtype=bool)
    p, sw, third = unpack_black_oil_primary(x, grid.nx)
    sw_new, so_new, sg_new, rs_new = interpret_black_oil_primary(p, sw, third, active_saturated, oil)
    sw_old, so_old, sg_old, rs_old = old.physical(oil)
    v = grid.volumes

    phi_new = rock.porosity(p)
    phi_old = rock.porosity(old.p)

    bw_new = water.formation_volume_factor(p)
    bo_new = oil.formation_volume_factor(p)
    bg_new = gas.formation_volume_factor(p)
    bw_old = water.formation_volume_factor(old.p)
    bo_old = oil.formation_volume_factor(old.p)
    bg_old = gas.formation_volume_factor(old.p)

    acc_w = v * (phi_new * sw_new / bw_new - phi_old * sw_old / bw_old) / dt
    acc_o = v * (phi_new * so_new / bo_new - phi_old * so_old / bo_old) / dt
    acc_g = v * (
        phi_new * (rs_new * so_new / bo_new + sg_new / bg_new)
        - phi_old * (rs_old * so_old / bo_old + sg_old / bg_old)
    ) / dt

    fw, fo, _fg_free, fg_comp = three_phase_black_oil_face_fluxes_with_rs(
        grid, rock.permeability, p, sw_new, sg_new, rs_new, relperm, water, oil, gas
    )
    div_w = divergence_from_face_flux(grid.nx, fw)
    div_o = divergence_from_face_flux(grid.nx, fo)
    div_g = divergence_from_face_flux(grid.nx, fg_comp)

    qw = np.zeros(grid.nx, dtype=float)
    qo = np.zeros(grid.nx, dtype=float)
    qg_free = np.zeros(grid.nx, dtype=float)

    krw = relperm.krw(sw_new, sg_new)
    kro = relperm.kro(sw_new, sg_new)
    krg = relperm.krg(sw_new, sg_new)
    muw = water.viscosity(p)
    muo = oil.viscosity(p)
    mug = gas.viscosity(p)

    for well in wells:
        if isinstance(well, RateWell):
            qwi, qoi, qgi = well.three_phase_sources(grid.nx)
        elif isinstance(well, BHPWell):
            qwi, qoi, qgi = well.three_phase_sources(
                p, sw_new, sg_new, krw, kro, krg, muw, muo, mug, bw_new, bo_new, bg_new
            )
        else:
            raise TypeError(f"Unsupported well type: {type(well)!r}")
        qw += qwi
        qo += qoi
        qg_free += qgi

    qg_component = qg_free + rs_new * qo

    # Row-scale the gas component equation. Its natural magnitude is roughly
    # Rs times the oil equation. In undersaturated cells with no free gas, the
    # gas equation may become nearly redundant with the oil equation; for this
    # Step 4B prototype we therefore hold Rs implicitly fixed in such cells.
    # Once a cell becomes saturated, the full gas-component conservation row is
    # activated and solves for Sg. This is the practical bridge toward a later
    # fully conservative variable-switching formulation with analytic Jacobians.
    gas_scale = max(float(oil.max_solution_gas_ratio()), 1.0)
    gas_component_residual = acc_g + div_g - qg_component
    rs_freeze_residual = (third - old.x) / gas_scale
    gas_row = np.where(active_saturated, gas_component_residual / gas_scale, rs_freeze_residual)
    return np.concatenate([
        acc_w + div_w - qw,
        acc_o + div_o - qo,
        gas_row,
    ])


@dataclass
class BlackOilStepInfo:
    """Diagnostics for one accepted Step 4B black-oil timestep."""

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
    min_rs: float
    max_rs: float
    saturated_cells: int
    undersaturated_cells: int
    switched_to_saturated: int
    switched_to_undersaturated: int
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
class LiveOilPhaseSwitchingSimulator:
    """Step 4B live-oil black-oil simulator with bubble-point logic.

    The simulator implements a primary-variable switching strategy:

    * undersaturated cell: primary unknowns are ``p, Sw, Rs`` and ``Sg = 0``;
    * saturated cell: primary unknowns are ``p, Sw, Sg`` and ``Rs = Rs_sat(p)``.

    The phase map is fixed during each Newton solve and updated after
    convergence. This is intentionally more stable and easier to verify than a
    first attempt at a fully coupled complementarity solver.
    """

    grid: object
    rock: object
    water: object
    oil: object
    gas: object
    relperm: object
    state: StateBlackOil
    wells: list = field(default_factory=list)
    solver: NewtonSolver = field(default_factory=NewtonSolver)
    sg_switch_tol: float = 1.0e-5
    saturation_eps: float = 1.0e-9
    cumulative_oil_produced: float = 0.0
    cumulative_water_produced: float = 0.0
    cumulative_free_gas_produced: float = 0.0
    cumulative_gas_component_produced: float = 0.0
    cumulative_oil_injected: float = 0.0
    cumulative_water_injected: float = 0.0
    cumulative_free_gas_injected: float = 0.0
    cumulative_gas_component_injected: float = 0.0

    def copy_state(self) -> StateBlackOil:
        return self.state.copy()

    def bounds(self, active_saturated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nx = self.grid.nx
        lower_p = np.full(nx, 1.0e5)
        upper_p = np.full(nx, 1.0e9)
        lower_sw = np.full(nx, self.relperm.swc + 1.0e-8)
        upper_sw = np.full(nx, 1.0 - self.relperm.sor - 1.0e-8)
        lower_x = np.zeros(nx, dtype=float)
        upper_x = np.where(
            active_saturated,
            1.0 - self.relperm.sor - self.relperm.swc - 1.0e-8,
            max(self.oil.max_solution_gas_ratio(), 1.0),
        )
        return (
            np.concatenate([lower_p, lower_sw, lower_x]),
            np.concatenate([upper_p, upper_sw, upper_x]),
        )

    def physical_arrays(self, state: StateBlackOil | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        st = self.state if state is None else state
        return st.physical(self.oil)

    def component_in_place(self, state: StateBlackOil | None = None) -> tuple[float, float, float]:
        st = self.state if state is None else state
        sw, so, sg, rs = st.physical(self.oil)
        pv = self.rock.pore_volume(self.grid.volumes, st.p)
        bw = self.water.formation_volume_factor(st.p)
        bo = self.oil.formation_volume_factor(st.p)
        bg = self.gas.formation_volume_factor(st.p)
        water = float(np.sum(pv * sw / bw))
        oil = float(np.sum(pv * so / bo))
        gas_component = float(np.sum(pv * (rs * so / bo + sg / bg)))
        return water, oil, gas_component

    def well_sources(self, state: StateBlackOil | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        st = self.state if state is None else state
        p = st.p
        sw, _so, sg, rs = st.physical(self.oil)
        bw = self.water.formation_volume_factor(p)
        bo = self.oil.formation_volume_factor(p)
        bg = self.gas.formation_volume_factor(p)
        muw = self.water.viscosity(p)
        muo = self.oil.viscosity(p)
        mug = self.gas.viscosity(p)
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

    def _valid_physical_state(self, state: StateBlackOil) -> bool:
        sw, so, sg, rs = state.physical(self.oil)
        rs_sat = self.oil.solution_gas_ratio(state.p)
        return bool(
            np.all(np.isfinite(state.p))
            and np.all(np.isfinite(sw))
            and np.all(np.isfinite(so))
            and np.all(np.isfinite(sg))
            and np.all(np.isfinite(rs))
            and np.all(sw >= self.relperm.swc - 1.0e-7)
            and np.all(so >= self.relperm.sor - 1.0e-7)
            and np.all(sg >= -1.0e-8)
            and np.all(rs >= -1.0e-8)
            and np.all(rs <= self.oil.max_solution_gas_ratio() + 1.0e-6)
            and np.all(rs[~state.is_saturated] <= rs_sat[~state.is_saturated] + 1.0e-5)
        )

    def apply_phase_switching(self, candidate: StateBlackOil) -> tuple[StateBlackOil, int, int]:
        """Apply conservative cell-wise phase switching after Newton convergence."""
        p = candidate.p.copy()
        sw = candidate.sw.copy()
        x = candidate.x.copy()
        sat = candidate.is_saturated.copy()
        rs_sat = self.oil.solution_gas_ratio(p)
        bo = self.oil.formation_volume_factor(p)
        bg = self.gas.formation_volume_factor(p)
        to_sat = 0
        to_unsat = 0

        for i in range(self.grid.nx):
            if not sat[i]:
                rs_u = max(float(x[i]), 0.0)
                if rs_u > rs_sat[i] * (1.0 + 1.0e-9) + 1.0e-9:
                    # Preserve gas component inventory while converting excess
                    # dissolved gas into a free-gas saturation.
                    so_unsat = max(1.0 - sw[i], self.relperm.sor + self.saturation_eps)
                    numerator = (rs_u - rs_sat[i]) * so_unsat / bo[i]
                    denominator = 1.0 / bg[i] - rs_sat[i] / bo[i]
                    if denominator <= 1.0e-20:
                        sg_new = self.saturation_eps
                    else:
                        sg_new = numerator / denominator
                    sg_upper = max(1.0 - sw[i] - self.relperm.sor - self.saturation_eps, self.saturation_eps)
                    x[i] = float(np.clip(sg_new, self.saturation_eps, sg_upper))
                    sat[i] = True
                    to_sat += 1
                else:
                    x[i] = float(np.clip(rs_u, 0.0, rs_sat[i]))
            else:
                sg_s = max(float(x[i]), 0.0)
                if sg_s <= self.sg_switch_tol:
                    so_sat = max(1.0 - sw[i] - sg_s, self.relperm.sor + self.saturation_eps)
                    # Convert the gas component inventory back into dissolved gas.
                    g_per_pv = rs_sat[i] * so_sat / bo[i] + sg_s / bg[i]
                    so_unsat = max(1.0 - sw[i], self.relperm.sor + self.saturation_eps)
                    rs_equiv = g_per_pv * bo[i] / so_unsat
                    if rs_equiv <= rs_sat[i] * (1.0 + 1.0e-9) + 1.0e-9:
                        x[i] = float(np.clip(rs_equiv, 0.0, rs_sat[i]))
                        sat[i] = False
                        to_unsat += 1
                    else:
                        sg_upper = max(1.0 - sw[i] - self.relperm.sor - self.saturation_eps, self.saturation_eps)
                        x[i] = float(np.clip(sg_s, self.saturation_eps, sg_upper))
                else:
                    sg_upper = max(1.0 - sw[i] - self.relperm.sor - self.saturation_eps, self.saturation_eps)
                    x[i] = float(np.clip(sg_s, self.saturation_eps, sg_upper))

        return StateBlackOil(p=p, sw=sw, x=x, is_saturated=sat), to_sat, to_unsat

    def try_step(self, dt: float, max_ds: float | None = None) -> tuple[bool, NewtonReport, float, int, int]:
        old = self.state.copy()
        active_saturated = old.is_saturated.copy()
        x0 = pack_black_oil_primary(old.p, old.sw, old.x)
        lower, upper = self.bounds(active_saturated)

        # Solve in scaled primary variables. Pressure is O(1e7 Pa), saturation
        # is O(1), and undersaturated Rs is O(1e2). Without column scaling the
        # dense finite-difference Jacobian becomes unnecessarily ill-conditioned
        # during undersaturated steps, where oil and gas equations are strongly
        # coupled.
        p_scale = max(float(np.mean(np.abs(old.p))), 1.0e5)
        rs_scale = max(float(self.oil.max_solution_gas_ratio()), 1.0)
        third_scale = np.where(active_saturated, 1.0, rs_scale)
        scales = np.concatenate([
            np.full(self.grid.nx, p_scale, dtype=float),
            np.ones(self.grid.nx, dtype=float),
            third_scale,
        ])
        y0 = x0 / scales
        lower_y = lower / scales
        upper_y = upper / scales

        def func_y(y):
            x = np.asarray(y, dtype=float) * scales
            return residual_live_oil_phase_switching(
                x,
                old,
                dt,
                self.grid,
                self.rock,
                self.water,
                self.oil,
                self.gas,
                self.relperm,
                self.wells,
                active_saturated=active_saturated,
            )

        y_new, report = self.solver.solve(func_y, y0, lower=lower_y, upper=upper_y)
        x_new = y_new * scales
        p_new, sw_new, third_new = unpack_black_oil_primary(x_new, self.grid.nx)
        raw_candidate = StateBlackOil(p=p_new, sw=sw_new, x=third_new, is_saturated=active_saturated)
        candidate, to_sat, to_unsat = self.apply_phase_switching(raw_candidate)

        _sw_old, _so_old, sg_old, rs_old = old.physical(self.oil)
        _sw_new, _so_new, sg_new, rs_new = candidate.physical(self.oil)
        ds_max = float(max(np.max(np.abs(sw_new - old.sw)), np.max(np.abs(sg_new - sg_old))))
        drs_scale = max(float(np.max(np.abs(rs_old))), 1.0)
        drs_rel = float(np.max(np.abs(rs_new - rs_old)) / drs_scale)
        change_indicator = max(ds_max, 0.05 * drs_rel)

        accepted = (
            report.converged
            and self._valid_physical_state(candidate)
            and (max_ds is None or change_indicator <= max_ds)
        )
        if accepted:
            self.state = candidate
        return accepted, report, change_indicator, to_sat, to_unsat

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
        times = [0.0]
        pressures = [self.state.p.copy()]
        sw0, so0, sg0, rs0 = self.state.physical(self.oil)
        sw_values = [sw0.copy()]
        sg_values = [sg0.copy()]
        so_values = [so0.copy()]
        rs_values = [rs0.copy()]
        phase_values = [self.state.is_saturated.copy()]
        reports: list[BlackOilStepInfo] = []

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

            accepted, report, change_indicator, to_sat, to_unsat = self.try_step(dt, max_ds=max_ds)
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
                        "Phase-switching black-oil timestep failed below dt_min: "
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
                BlackOilStepInfo(
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
                    undersaturated_cells=int(self.grid.nx - np.count_nonzero(self.state.is_saturated)),
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
                )
            )

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
        }
