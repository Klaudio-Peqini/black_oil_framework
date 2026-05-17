from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .state import StateBlackOil
from .black_oil_phase import (
    BlackOilStepInfo,
    pack_black_oil_primary,
    unpack_black_oil_primary,
    interpret_black_oil_primary,
)
from .black_oil_5a import _three_phase_well_sources_5a
from .capillary import ZeroCapillaryPressure
from .boundary import BoundaryConditions2D
from .flux import phase_pressures_black_oil
from .flux2d import (
    three_phase_black_oil_fluxes_2d,
    divergence_from_face_fluxes_2d,
    boundary_component_fluxes_2d,
)
from .sparse_jacobian import structured_grid_black_oil_sparsity
from .sparse_solver import SparseNewtonSolver
from .wells import ControlledWell, BHPWell


def residual_live_oil_5c_2d(
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
    capillary=None,
    gravity: float = 9.80665,
    boundaries: BoundaryConditions2D | None = None,
) -> np.ndarray:
    """Step 5C 2D black-oil residual for heterogeneous/anisotropic reservoirs.

    This is the Step 5A/5B physics generalized to a structured 2D stencil:
    conservative black-oil components, primary-variable switching, gravity,
    capillary pressure, controlled wells, anisotropic transmissibilities, and
    optional pressure boundaries. Arrays are flattened with x as the fast index.
    """
    wells = wells or []
    capillary = ZeroCapillaryPressure() if capillary is None else capillary
    boundaries = BoundaryConditions2D.no_flow() if boundaries is None else boundaries
    n = grid.n_cells
    active_saturated = old.is_saturated if active_saturated is None else np.asarray(active_saturated, dtype=bool)

    p, sw, third = unpack_black_oil_primary(x, n)
    sw_new, so_new, sg_new, rs_new = interpret_black_oil_primary(p, sw, third, active_saturated, oil)
    sw_old, so_old, sg_old, rs_old = old.physical(oil)
    v = grid.volumes

    pw_new, po_new, pg_new = phase_pressures_black_oil(p, sw_new, sg_new, capillary)
    pw_old, po_old, pg_old = phase_pressures_black_oil(old.p, sw_old, sg_old, capillary)

    phi_new = rock.porosity(po_new)
    phi_old = rock.porosity(po_old)
    bw_new = water.formation_volume_factor(pw_new)
    bo_new = oil.formation_volume_factor(po_new)
    bg_new = gas.formation_volume_factor(pg_new)
    bw_old = water.formation_volume_factor(pw_old)
    bo_old = oil.formation_volume_factor(po_old)
    bg_old = gas.formation_volume_factor(pg_old)

    acc_w = v * (phi_new * sw_new / bw_new - phi_old * sw_old / bw_old) / dt
    acc_o = v * (phi_new * so_new / bo_new - phi_old * so_old / bo_old) / dt
    acc_g = v * (
        phi_new * (rs_new * so_new / bo_new + sg_new / bg_new)
        - phi_old * (rs_old * so_old / bo_old + sg_old / bg_old)
    ) / dt

    fw_x, fw_y, fo_x, fo_y, _fg_x, _fg_y, fgc_x, fgc_y = three_phase_black_oil_fluxes_2d(
        grid,
        rock.permeability,
        p,
        sw_new,
        sg_new,
        rs_new,
        relperm,
        water,
        oil,
        gas,
        capillary=capillary,
        gravity=gravity,
    )
    div_w = divergence_from_face_fluxes_2d(grid, fw_x, fw_y)
    div_o = divergence_from_face_fluxes_2d(grid, fo_x, fo_y)
    div_g = divergence_from_face_fluxes_2d(grid, fgc_x, fgc_y)

    bc_w, bc_o, bc_g = boundary_component_fluxes_2d(
        grid,
        rock.permeability,
        p,
        sw_new,
        sg_new,
        rs_new,
        relperm,
        water,
        oil,
        gas,
        capillary=capillary,
        gravity=gravity,
        boundaries=boundaries,
    )
    div_w += bc_w
    div_o += bc_o
    div_g += bc_g

    qw = np.zeros(n, dtype=float)
    qo = np.zeros(n, dtype=float)
    qg_free = np.zeros(n, dtype=float)

    krw = relperm.krw(sw_new, sg_new)
    kro = relperm.kro(sw_new, sg_new)
    krg = relperm.krg(sw_new, sg_new)
    muw = water.viscosity(pw_new)
    muo = oil.viscosity(po_new)
    mug = gas.viscosity(pg_new)

    for well in wells:
        qwi, qoi, qgi = _three_phase_well_sources_5a(
            well,
            n,
            pw_new,
            po_new,
            pg_new,
            krw,
            kro,
            krg,
            muw,
            muo,
            mug,
            bw_new,
            bo_new,
            bg_new,
        )
        qw += qwi
        qo += qoi
        qg_free += qgi

    qg_component = qg_free + rs_new * qo
    gas_scale = max(float(oil.max_solution_gas_ratio()), 1.0)
    return np.concatenate([
        acc_w + div_w - qw,
        acc_o + div_o - qo,
        (acc_g + div_g - qg_component) / gas_scale,
    ])


@dataclass
class BlackOil2DStepInfo(BlackOilStepInfo):
    """Diagnostics for one accepted Step 5C timestep."""

    linear_iterations_total: int = 0
    jacobian_nnz: int = 0
    jacobian_colors: int = 0


@dataclass
class HeterogeneousBlackOilSimulator5C:
    """Step 5C simulator for 2D heterogeneous anisotropic reservoirs.

    The class deliberately mirrors the Step 5B sparse-Newton workflow but uses a
    2D finite-volume stencil and ``CartesianGrid2D``. It is still a research
    prototype: the grid is structured Cartesian, the wells are cell-centred, and
    pressure boundaries are simple constant-state boundaries. These choices are
    useful now because they verify 2D heterogeneity before introducing schedules
    and 3D corner-point geometry.
    """

    grid: object
    rock: object
    water: object
    oil: object
    gas: object
    relperm: object
    state: StateBlackOil
    wells: list = field(default_factory=list)
    capillary: object = field(default_factory=ZeroCapillaryPressure)
    gravity: float = 9.80665
    boundaries: BoundaryConditions2D = field(default_factory=BoundaryConditions2D.no_flow)
    solver: SparseNewtonSolver = field(
        default_factory=lambda: SparseNewtonSolver(
            tol=1.0e-8,
            max_iter=14,
            acceptable_tol=5.0e-4,
            acceptable_min_iterations=2,
            jacobian_strategy="sparse_fd",
            linear_solver="gmres",
            preconditioner="ilu",
            krylov_rtol=1.0e-6,
            krylov_maxiter=200,
            regularization=1.0e-18,
        )
    )
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
    active_control_log: list[dict] = field(default_factory=list)
    _sparsity_pattern: object | None = field(default=None, init=False, repr=False)

    @property
    def n(self) -> int:
        return self.grid.n_cells

    def sparsity_pattern(self):
        if self._sparsity_pattern is None:
            self._sparsity_pattern = structured_grid_black_oil_sparsity(self.grid, n_components=3)
        return self._sparsity_pattern

    def bounds(self, active_saturated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = self.n
        lower_p = np.full(n, 1.0e5)
        upper_p = np.full(n, 1.0e9)
        lower_sw = np.full(n, self.relperm.swc + 1.0e-8)
        upper_sw = np.full(n, 1.0 - self.relperm.sor - 1.0e-8)
        lower_x = np.zeros(n, dtype=float)
        upper_x = np.where(
            active_saturated,
            1.0 - self.relperm.sor - self.relperm.swc - 1.0e-8,
            max(self.oil.max_solution_gas_ratio(), 1.0),
        )
        return (
            np.concatenate([lower_p, lower_sw, lower_x]),
            np.concatenate([upper_p, upper_sw, upper_x]),
        )

    def _phase_pvt_relperm(self, state: StateBlackOil):
        sw, _so, sg, _rs = state.physical(self.oil)
        pw, po, pg = phase_pressures_black_oil(state.p, sw, sg, self.capillary)
        bw = self.water.formation_volume_factor(pw)
        bo = self.oil.formation_volume_factor(po)
        bg = self.gas.formation_volume_factor(pg)
        muw = self.water.viscosity(pw)
        muo = self.oil.viscosity(po)
        mug = self.gas.viscosity(pg)
        krw = self.relperm.krw(sw, sg)
        kro = self.relperm.kro(sw, sg)
        krg = self.relperm.krg(sw, sg)
        return pw, po, pg, krw, kro, krg, muw, muo, mug, bw, bo, bg

    def resolve_well_controls(self, state: StateBlackOil) -> tuple[list, list[dict]]:
        pw, po, pg, krw, kro, krg, muw, muo, mug, bw, bo, bg = self._phase_pvt_relperm(state)
        active = []
        log = []
        for well in self.wells:
            if isinstance(well, ControlledWell):
                active_well = well.active_well(pw, po, pg, krw, kro, krg, muw, muo, mug, bw, bo, bg)
                active.append(active_well)
                log.append(
                    {
                        "name": well.name,
                        "requested_control": well.control,
                        "target": well.target,
                        "active_type": type(active_well).__name__,
                        "active_control": getattr(active_well, "phase", "bhp" if isinstance(active_well, BHPWell) else "multi"),
                        "bhp": getattr(active_well, "bhp", np.nan),
                    }
                )
            else:
                active.append(well)
                log.append(
                    {
                        "name": getattr(well, "name", "well"),
                        "requested_control": type(well).__name__,
                        "target": np.nan,
                        "active_type": type(well).__name__,
                        "active_control": getattr(well, "phase", "bhp" if isinstance(well, BHPWell) else "multi"),
                        "bhp": getattr(well, "bhp", np.nan),
                    }
                )
        return active, log

    def _valid_physical_state(self, state: StateBlackOil) -> bool:
        sw, so, sg, rs = state.physical(self.oil)
        rs_sat = self.oil.solution_gas_ratio(state.p)
        return bool(
            np.all(np.isfinite(state.p))
            and np.all(np.isfinite(sw))
            and np.all(np.isfinite(sg))
            and np.all(state.p > 0.0)
            and np.all(sw >= self.relperm.swc - 1.0e-7)
            and np.all(so >= self.relperm.sor - 1.0e-7)
            and np.all(sg >= -1.0e-8)
            and np.all(rs >= -1.0e-8)
            and np.all(rs <= self.oil.max_solution_gas_ratio() + 1.0e-6)
            and np.all(rs[~state.is_saturated] <= rs_sat[~state.is_saturated] + 1.0e-5)
        )

    def apply_phase_switching(self, candidate: StateBlackOil) -> tuple[StateBlackOil, int, int]:
        p = candidate.p.copy()
        sw = candidate.sw.copy()
        x = candidate.x.copy()
        sat = candidate.is_saturated.copy()
        rs_sat = self.oil.solution_gas_ratio(p)
        bo = self.oil.formation_volume_factor(p)
        bg = self.gas.formation_volume_factor(p)
        to_sat = 0
        to_unsat = 0
        for i in range(self.n):
            if not sat[i]:
                rs_u = max(float(x[i]), 0.0)
                if rs_u > rs_sat[i] * (1.0 + 1.0e-9) + 1.0e-9:
                    so_unsat = max(1.0 - sw[i], self.relperm.sor + self.saturation_eps)
                    numerator = (rs_u - rs_sat[i]) * so_unsat / bo[i]
                    denominator = 1.0 / bg[i] - rs_sat[i] / bo[i]
                    sg_new = self.saturation_eps if denominator <= 1.0e-20 else numerator / denominator
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

    def component_in_place(self, state: StateBlackOil | None = None) -> tuple[float, float, float]:
        st = self.state if state is None else state
        sw, so, sg, rs = st.physical(self.oil)
        pw, po, pg = phase_pressures_black_oil(st.p, sw, sg, self.capillary)
        pv = self.rock.pore_volume(self.grid.volumes, po)
        bw = self.water.formation_volume_factor(pw)
        bo = self.oil.formation_volume_factor(po)
        bg = self.gas.formation_volume_factor(pg)
        water = float(np.sum(pv * sw / bw))
        oil = float(np.sum(pv * so / bo))
        gas_component = float(np.sum(pv * (rs * so / bo + sg / bg)))
        return water, oil, gas_component

    @staticmethod
    def _split_rates(q: np.ndarray) -> tuple[float, float]:
        inj = float(np.sum(np.maximum(q, 0.0)))
        prod = float(-np.sum(np.minimum(q, 0.0)))
        return inj, prod

    def well_sources(self, state: StateBlackOil | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        st = self.state if state is None else state
        sw, _so, sg, rs = st.physical(self.oil)
        pw, po, pg, krw, kro, krg, muw, muo, mug, bw, bo, bg = self._phase_pvt_relperm(st)
        qw = np.zeros(self.n, dtype=float)
        qo = np.zeros(self.n, dtype=float)
        qg_free = np.zeros(self.n, dtype=float)
        active_wells, _log = self.resolve_well_controls(st)
        for well in active_wells:
            qwi, qoi, qgi = _three_phase_well_sources_5a(
                well, self.n, pw, po, pg, krw, kro, krg, muw, muo, mug, bw, bo, bg
            )
            qw += qwi
            qo += qoi
            qg_free += qgi
        qg_component = qg_free + rs * qo
        return qw, qo, qg_free, qg_component

    def try_step(self, dt: float, max_ds: float | None = None):
        old = self.state.copy()
        active_saturated = old.is_saturated.copy()
        active_wells, control_log = self.resolve_well_controls(old)
        x0 = pack_black_oil_primary(old.p, old.sw, old.x)
        lower, upper = self.bounds(active_saturated)

        p_scale = max(float(np.mean(np.abs(old.p))), 1.0e5)
        rs_scale = max(float(self.oil.max_solution_gas_ratio()), 1.0)
        third_scale = np.where(active_saturated, 1.0, rs_scale)
        scales = np.concatenate([
            np.full(self.n, p_scale, dtype=float),
            np.ones(self.n, dtype=float),
            third_scale,
        ])
        y0 = x0 / scales
        lower_y = lower / scales
        upper_y = upper / scales

        def func_y(y):
            x = np.asarray(y, dtype=float) * scales
            return residual_live_oil_5c_2d(
                x,
                old,
                dt,
                self.grid,
                self.rock,
                self.water,
                self.oil,
                self.gas,
                self.relperm,
                active_wells,
                active_saturated=active_saturated,
                capillary=self.capillary,
                gravity=self.gravity,
                boundaries=self.boundaries,
            )

        y_new, report = self.solver.solve(
            func_y,
            y0,
            sparsity=self.sparsity_pattern(),
            lower=lower_y,
            upper=upper_y,
        )
        x_new = y_new * scales
        p_new, sw_new, third_new = unpack_black_oil_primary(x_new, self.n)
        raw_candidate = StateBlackOil(p=p_new, sw=sw_new, x=third_new, is_saturated=active_saturated)
        candidate, to_sat, to_unsat = self.apply_phase_switching(raw_candidate)

        _sw_old, _so_old, sg_old, rs_old = old.physical(self.oil)
        _sw_new, _so_new, sg_new, rs_new = candidate.physical(self.oil)
        ds_max = float(max(np.max(np.abs(candidate.sw - old.sw)), np.max(np.abs(sg_new - sg_old))))
        drs_scale = max(float(np.max(np.abs(rs_old))), 1.0)
        drs_rel = float(np.max(np.abs(rs_new - rs_old)) / drs_scale)
        change_indicator = max(ds_max, 0.05 * drs_rel)

        accepted = report.converged and self._valid_physical_state(candidate) and (max_ds is None or change_indicator <= max_ds)
        if accepted:
            self.state = candidate
            self.active_control_log.append({"dt": dt, "controls": control_log})
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
        reports: list[BlackOil2DStepInfo] = []

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
                        "2D black-oil timestep failed below dt_min: "
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
