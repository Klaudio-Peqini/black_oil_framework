from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .state import StateBlackOil
from .black_oil_phase import (
    BlackOilStepInfo,
    LiveOilPhaseSwitchingSimulator,
    pack_black_oil_primary,
    unpack_black_oil_primary,
    interpret_black_oil_primary,
)
from .black_oil_ad import ConservativeLiveOilPhaseSwitchingSimulator
from .flux import (
    divergence_from_face_flux,
    phase_pressures_black_oil,
    three_phase_black_oil_face_fluxes_with_rs_gravity_capillary,
)
from .wells import RateWell, BHPWell, MultiRateWell, ControlledWell
from .nonlinear_solver import NewtonSolver
from .ad_solver import NewtonSolverWithJacobian
from .capillary import ZeroCapillaryPressure


def _three_phase_well_sources_5a(
    well,
    nx: int,
    p_water: np.ndarray,
    p_oil: np.ndarray,
    p_gas: np.ndarray,
    krw: np.ndarray,
    kro: np.ndarray,
    krg: np.ndarray,
    muw: np.ndarray,
    muo: np.ndarray,
    mug: np.ndarray,
    bw: np.ndarray,
    bo: np.ndarray,
    bg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return phase/component source arrays for Step 5A well objects.

    BHP wells use the corresponding phase pressure at the completed cell. Rate
    wells use imposed surface rates. Controlled wells should normally be
    resolved to an active well before the nonlinear step, but the function also
    accepts them for post-step diagnostics.
    """
    if isinstance(well, ControlledWell):
        active = well.active_well(p_water, p_oil, p_gas, krw, kro, krg, muw, muo, mug, bw, bo, bg)
        return _three_phase_well_sources_5a(active, nx, p_water, p_oil, p_gas, krw, kro, krg, muw, muo, mug, bw, bo, bg)

    if isinstance(well, MultiRateWell):
        return well.three_phase_sources(nx)

    if isinstance(well, RateWell):
        return well.three_phase_sources(nx)

    if isinstance(well, BHPWell):
        qw = np.zeros(nx, dtype=float)
        qo = np.zeros(nx, dtype=float)
        qg = np.zeros(nx, dtype=float)
        c = well.cell
        qw[c] += well.well_index * krw[c] / (muw[c] * bw[c]) * (well.bhp - p_water[c])
        qo[c] += well.well_index * kro[c] / (muo[c] * bo[c]) * (well.bhp - p_oil[c])
        qg[c] += well.well_index * krg[c] / (mug[c] * bg[c]) * (well.bhp - p_gas[c])
        return qw, qo, qg

    raise TypeError(f"Unsupported well type: {type(well)!r}")


def residual_live_oil_5a_gravity_capillary(
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
) -> np.ndarray:
    """Step 5A conservative black-oil residual with gravity and capillarity.

    The primary-variable interpretation remains the Step 4C conservative
    switching formulation. The new features are:

    * phase pressures p_w = p_o - pcow(Sw), p_g = p_o + pcgo(Sg),
    * gravity potentials with depth positive downward,
    * PVT properties evaluated at phase pressures,
    * BHP wells using phase pressure drawdowns,
    * rate/BHP switching through resolved controlled wells.
    """
    wells = wells or []
    capillary = ZeroCapillaryPressure() if capillary is None else capillary
    active_saturated = old.is_saturated if active_saturated is None else np.asarray(active_saturated, dtype=bool)

    p, sw, third = unpack_black_oil_primary(x, grid.nx)
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

    fw, fo, _fg_free, fg_comp = three_phase_black_oil_face_fluxes_with_rs_gravity_capillary(
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
    div_w = divergence_from_face_flux(grid.nx, fw)
    div_o = divergence_from_face_flux(grid.nx, fo)
    div_g = divergence_from_face_flux(grid.nx, fg_comp)

    qw = np.zeros(grid.nx, dtype=float)
    qo = np.zeros(grid.nx, dtype=float)
    qg_free = np.zeros(grid.nx, dtype=float)

    krw = relperm.krw(sw_new, sg_new)
    kro = relperm.kro(sw_new, sg_new)
    krg = relperm.krg(sw_new, sg_new)
    muw = water.viscosity(pw_new)
    muo = oil.viscosity(po_new)
    mug = gas.viscosity(pg_new)

    for well in wells:
        qwi, qoi, qgi = _three_phase_well_sources_5a(
            well,
            grid.nx,
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
class AdvancedBlackOilSimulator5A(ConservativeLiveOilPhaseSwitchingSimulator):
    """Step 5A black-oil simulator with gravity, capillary pressure, and controls.

    This class deliberately reuses the Step 4C conservative primary-variable
    switching. The Newton step is still dense and research-scale. Sparse
    assembly and Newton-Krylov preconditioning are deferred to Step 5B.
    """

    capillary: object = field(default_factory=ZeroCapillaryPressure)
    gravity: float = 9.80665
    solver: NewtonSolverWithJacobian | NewtonSolver = field(default_factory=lambda: NewtonSolverWithJacobian(tol=1.0e-8, max_iter=18))
    jacobian_mode: str = "finite_difference"
    active_control_log: list[dict] = field(default_factory=list)

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
        """Resolve rate/BHP switching from a fixed state before Newton."""
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
                        "active_control": getattr(active_well, "phase", "multi" if isinstance(active_well, MultiRateWell) else "bhp"),
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

    def well_sources(self, state: StateBlackOil | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        st = self.state if state is None else state
        sw, _so, sg, rs = st.physical(self.oil)
        pw, po, pg, krw, kro, krg, muw, muo, mug, bw, bo, bg = self._phase_pvt_relperm(st)
        qw = np.zeros(self.grid.nx, dtype=float)
        qo = np.zeros(self.grid.nx, dtype=float)
        qg_free = np.zeros(self.grid.nx, dtype=float)
        # For diagnostics, controls are resolved on the current state.
        active_wells, _log = self.resolve_well_controls(st)
        for well in active_wells:
            qwi, qoi, qgi = _three_phase_well_sources_5a(
                well, self.grid.nx, pw, po, pg, krw, kro, krg, muw, muo, mug, bw, bo, bg
            )
            qw += qwi
            qo += qoi
            qg_free += qgi
        qg_component = qg_free + rs * qo
        return qw, qo, qg_free, qg_component

    def try_step(self, dt: float, max_ds: float | None = None) -> tuple[bool, object, float, int, int]:
        old = self.state.copy()
        active_saturated = old.is_saturated.copy()
        active_wells, control_log = self.resolve_well_controls(old)
        x0 = pack_black_oil_primary(old.p, old.sw, old.x)
        lower, upper = self.bounds(active_saturated)

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
            return residual_live_oil_5a_gravity_capillary(
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
            )

        if isinstance(self.solver, NewtonSolverWithJacobian):
            # Step 5A deliberately keeps the robust finite-difference fallback.
            # A sparse/AD assembly aware of capillary and control switching is
            # scheduled for Step 5B.
            y_new, report = self.solver.solve(func_y, y0, jac=None, lower=lower_y, upper=upper_y)
        else:
            y_new, report = self.solver.solve(func_y, y0, lower=lower_y, upper=upper_y)

        x_new = y_new * scales
        p_new, sw_new, third_new = unpack_black_oil_primary(x_new, self.grid.nx)
        raw_candidate = StateBlackOil(p=p_new, sw=sw_new, x=third_new, is_saturated=active_saturated)
        candidate, to_sat, to_unsat = self.apply_phase_switching(raw_candidate)

        _sw_old, _so_old, sg_old, rs_old = old.physical(self.oil)
        _sw_new, _so_new, sg_new, rs_new = candidate.physical(self.oil)
        ds_max = float(max(np.max(np.abs(candidate.sw - old.sw)), np.max(np.abs(sg_new - sg_old))))
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
            self.active_control_log.append({"dt": dt, "controls": control_log})
        return accepted, report, change_indicator, to_sat, to_unsat
