from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from .state import StateBlackOil
from .black_oil_phase import (
    BlackOilStepInfo,
    pack_black_oil_primary,
    unpack_black_oil_primary,
    interpret_black_oil_primary,
)
from .capillary import ZeroCapillaryPressure
from .flux import phase_pressures_black_oil
from .flux3d import (
    three_phase_black_oil_fluxes_3d,
    divergence_from_face_fluxes_3d,
    boundary_component_fluxes_3d,
)
from .boundary3d import BoundaryConditions3D
from .reservoir3d import ActiveCellMap3D
from .sparse_jacobian import structured_grid_black_oil_sparsity
from .sparse_solver import SparseNewtonSolver
from .restart import save_black_oil_restart, load_black_oil_restart, apply_black_oil_restart


@dataclass
class BlackOil3DStepInfo(BlackOilStepInfo):
    """Diagnostics for one accepted full-3D black-oil timestep."""

    active_cells: int = 0
    inactive_cells: int = 0
    linear_iterations_total: int = 0
    jacobian_nnz: int = 0
    jacobian_colors: int = 0
    active_controls: list[dict] = field(default_factory=list)


def _open_completion_data(well):
    comps = [c for c in well.completions if c.is_open]
    cells = np.asarray([c.cell for c in comps], dtype=int)
    wi = np.asarray([c.well_index for c in comps], dtype=float)
    return cells, wi


def _safe_normalized_weights(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    vals = np.maximum(vals, 0.0)
    total = float(np.sum(vals))
    if total <= 0.0:
        return np.full(vals.size, 1.0 / max(vals.size, 1), dtype=float)
    return vals / total


def _well_effective_control_3d(well, pw, po, pg, krw, kro, krg, muw, muo, mug, bw, bo, bg) -> tuple[str, float, dict]:
    """Return effective control after simple rate/BHP-limit switching."""
    ctrl = well.control.lower()
    target = float(well.target)
    cells, wi = _open_completion_data(well)
    log = {
        "name": well.name,
        "requested_control": ctrl,
        "requested_target": target,
        "active_control": ctrl,
        "active_target": target,
        "bhp_estimate": np.nan,
        "status": well.status,
        "open_completions": int(cells.size),
    }
    if cells.size == 0 or ctrl == "bhp" or well.status.lower() != "open":
        return ctrl, target, log

    lamw = krw[cells] / (muw[cells] * bw[cells])
    lamo = kro[cells] / (muo[cells] * bo[cells])
    lamg = krg[cells] / (mug[cells] * bg[cells])
    if ctrl == "water_rate":
        lam = lamw
        pref = pw[cells]
    elif ctrl == "oil_rate":
        lam = lamo
        pref = po[cells]
    elif ctrl == "gas_rate":
        lam = lamg
        pref = pg[cells]
    elif ctrl == "liquid_rate":
        lam = lamw + lamo
        pref = po[cells]
    elif ctrl == "total_rate":
        lam = lamw + lamo + lamg
        pref = po[cells]
    else:
        return ctrl, target, log
    productivity = float(np.sum(wi * np.maximum(lam, 0.0)))
    p_ref = float(np.average(pref, weights=np.maximum(wi, 1.0e-300)))
    if productivity > 0.0:
        bhp_est = p_ref + target / productivity
        log["bhp_estimate"] = float(bhp_est)
        if well.well_type == "producer" and well.min_bhp is not None and bhp_est < float(well.min_bhp):
            log["active_control"] = "bhp"
            log["active_target"] = float(well.min_bhp)
            return "bhp", float(well.min_bhp), log
        if well.well_type == "injector" and well.max_bhp is not None and bhp_est > float(well.max_bhp):
            log["active_control"] = "bhp"
            log["active_target"] = float(well.max_bhp)
            return "bhp", float(well.max_bhp), log
    return ctrl, target, log


def _three_phase_field_well_sources_3d(
    well,
    n_cells: int,
    pw,
    po,
    pg,
    krw,
    kro,
    krg,
    muw,
    muo,
    mug,
    bw,
    bo,
    bg,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return water/oil/free-gas source terms for a multi-completion 3D well."""
    qw = np.zeros(n_cells, dtype=float)
    qo = np.zeros(n_cells, dtype=float)
    qg = np.zeros(n_cells, dtype=float)
    cells, wi = _open_completion_data(well)
    ctrl, target, log = _well_effective_control_3d(well, pw, po, pg, krw, kro, krg, muw, muo, mug, bw, bo, bg)
    if well.status.lower() != "open" or cells.size == 0:
        return qw, qo, qg, log

    lamw = krw[cells] / (muw[cells] * bw[cells])
    lamo = kro[cells] / (muo[cells] * bo[cells])
    lamg = krg[cells] / (mug[cells] * bg[cells])

    if ctrl == "bhp":
        bhp = float(target)
        np.add.at(qw, cells, wi * lamw * (bhp - pw[cells]))
        np.add.at(qo, cells, wi * lamo * (bhp - po[cells]))
        np.add.at(qg, cells, wi * lamg * (bhp - pg[cells]))
    elif ctrl == "water_rate":
        weights = _safe_normalized_weights(wi * np.maximum(lamw, 1.0e-300))
        np.add.at(qw, cells, target * weights)
    elif ctrl == "oil_rate":
        weights = _safe_normalized_weights(wi * np.maximum(lamo, 1.0e-300))
        np.add.at(qo, cells, target * weights)
    elif ctrl == "gas_rate":
        weights = _safe_normalized_weights(wi * np.maximum(lamg, 1.0e-300))
        np.add.at(qg, cells, target * weights)
    elif ctrl == "liquid_rate":
        lam_liq = lamw + lamo
        weights = _safe_normalized_weights(wi * np.maximum(lam_liq, 1.0e-300))
        frac_w = np.divide(lamw, lam_liq, out=np.zeros_like(lamw), where=lam_liq > 0.0)
        frac_o = np.divide(lamo, lam_liq, out=np.zeros_like(lamo), where=lam_liq > 0.0)
        np.add.at(qw, cells, target * weights * frac_w)
        np.add.at(qo, cells, target * weights * frac_o)
    elif ctrl == "total_rate":
        lam_tot = lamw + lamo + lamg
        weights = _safe_normalized_weights(wi * np.maximum(lam_tot, 1.0e-300))
        frac_w = np.divide(lamw, lam_tot, out=np.zeros_like(lamw), where=lam_tot > 0.0)
        frac_o = np.divide(lamo, lam_tot, out=np.zeros_like(lamo), where=lam_tot > 0.0)
        frac_g = np.divide(lamg, lam_tot, out=np.zeros_like(lamg), where=lam_tot > 0.0)
        np.add.at(qw, cells, target * weights * frac_w)
        np.add.at(qo, cells, target * weights * frac_o)
        np.add.at(qg, cells, target * weights * frac_g)
    else:
        raise ValueError(f"Unsupported 3D well control {ctrl!r}")
    return qw, qo, qg, log


def residual_live_oil_7_3d(
    x: np.ndarray,
    old: StateBlackOil,
    dt: float,
    grid,
    rock,
    water,
    oil,
    gas,
    relperm,
    *,
    wells=None,
    active_saturated: np.ndarray | None = None,
    capillary=None,
    gravity: float = 9.80665,
    transmissibility_multipliers=None,
    active: ActiveCellMap3D | None = None,
    boundaries: BoundaryConditions3D | None = None,
) -> np.ndarray:
    """Integrated Step 7 full-3D black-oil residual.

    The residual combines conservative live-oil primary-variable switching,
    3D TPFA fluxes, gravity, capillary pressure, active/inactive cell handling,
    transmissibility multipliers/fault barriers, multi-completion wells, and
    optional pressure boundaries. Unknowns remain full-grid arrays ordered as
    ``[p_o cells, S_w cells, third-variable cells]``.
    """
    wells = wells or []
    cap = ZeroCapillaryPressure() if capillary is None else capillary
    boundaries = BoundaryConditions3D.no_flow() if boundaries is None else boundaries
    n = grid.n_cells
    mask = np.ones(n, dtype=bool) if active is None else active.active_mask
    active_saturated = old.is_saturated if active_saturated is None else np.asarray(active_saturated, dtype=bool)

    p, sw, third = unpack_black_oil_primary(x, n)
    sw_new, so_new, sg_new, rs_new = interpret_black_oil_primary(p, sw, third, active_saturated, oil)
    sw_old, so_old, sg_old, rs_old = old.physical(oil)
    volumes = grid.volumes

    pw_new, po_new, pg_new = phase_pressures_black_oil(p, sw_new, sg_new, cap)
    pw_old, po_old, pg_old = phase_pressures_black_oil(old.p, sw_old, sg_old, cap)
    phi_new = rock.porosity(po_new)
    phi_old = rock.porosity(po_old)
    bw_new = water.formation_volume_factor(pw_new); bo_new = oil.formation_volume_factor(po_new); bg_new = gas.formation_volume_factor(pg_new)
    bw_old = water.formation_volume_factor(pw_old); bo_old = oil.formation_volume_factor(po_old); bg_old = gas.formation_volume_factor(pg_old)

    acc_w = volumes * (phi_new * sw_new / bw_new - phi_old * sw_old / bw_old) / dt
    acc_o = volumes * (phi_new * so_new / bo_new - phi_old * so_old / bo_old) / dt
    acc_g = volumes * (
        phi_new * (rs_new * so_new / bo_new + sg_new / bg_new)
        - phi_old * (rs_old * so_old / bo_old + sg_old / bg_old)
    ) / dt

    fw_x, fw_y, fw_z, fo_x, fo_y, fo_z, _fg_x, _fg_y, _fg_z, fgc_x, fgc_y, fgc_z = three_phase_black_oil_fluxes_3d(
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
        capillary=cap,
        gravity=gravity,
        transmissibility_multipliers=transmissibility_multipliers,
    )
    div_w = divergence_from_face_fluxes_3d(grid, fw_x, fw_y, fw_z)
    div_o = divergence_from_face_fluxes_3d(grid, fo_x, fo_y, fo_z)
    div_g = divergence_from_face_fluxes_3d(grid, fgc_x, fgc_y, fgc_z)
    bc_w, bc_o, bc_g = boundary_component_fluxes_3d(
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
        capillary=cap,
        gravity=gravity,
        boundaries=boundaries,
    )
    div_w += bc_w; div_o += bc_o; div_g += bc_g

    krw = relperm.krw(sw_new, sg_new); kro = relperm.kro(sw_new, sg_new); krg = relperm.krg(sw_new, sg_new)
    muw = water.viscosity(pw_new); muo = oil.viscosity(po_new); mug = gas.viscosity(pg_new)
    qw = np.zeros(n, dtype=float); qo = np.zeros(n, dtype=float); qg_free = np.zeros(n, dtype=float)
    for well in wells:
        qwi, qoi, qgi, _log = _three_phase_field_well_sources_3d(
            well, n, pw_new, po_new, pg_new, krw, kro, krg, muw, muo, mug, bw_new, bo_new, bg_new
        )
        qw += qwi; qo += qoi; qg_free += qgi
    qg_component = qg_free + rs_new * qo

    rw = acc_w + div_w - qw
    ro = acc_o + div_o - qo
    gas_scale = max(float(oil.max_solution_gas_ratio()), 1.0)
    rg = (acc_g + div_g - qg_component) / gas_scale

    if not np.all(mask):
        inactive = ~mask
        # For inactive ACTNUM-like cells, enforce fixed state rather than mass
        # conservation. Multipliers already remove their face fluxes.
        p_scale = max(float(np.mean(np.abs(old.p[mask]))) if np.any(mask) else 1.0, 1.0)
        rw[inactive] = (p[inactive] - old.p[inactive]) / p_scale
        ro[inactive] = sw[inactive] - old.sw[inactive]
        rg[inactive] = third[inactive] - old.x[inactive]
    return np.concatenate([rw, ro, rg])


@dataclass
class FullBlackOilSimulator3D:
    """Step 7 integrated full-3D live-oil black-oil simulator.

    This class is the first end-to-end 3D simulator in the framework. It uses
    the 3D grid/property/fault/well infrastructure from Steps 6A--6C and the
    conservative phase-switching formulation developed in Steps 4B--5B.
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
    transmissibility_multipliers: object | None = None
    active: ActiveCellMap3D | None = None
    boundaries: BoundaryConditions3D = field(default_factory=BoundaryConditions3D.no_flow)
    solver: SparseNewtonSolver = field(default_factory=lambda: SparseNewtonSolver(
        tol=1.0e-8,
        acceptable_tol=2.0e-4,
        acceptable_min_iterations=2,
        max_iter=12,
        jacobian_strategy="sparse_fd",
        linear_solver="gmres",
        preconditioner="ilu",
        krylov_rtol=1.0e-6,
        krylov_maxiter=250,
        regularization=1.0e-20,
    ))
    schedule: object | None = None
    current_time: float = 0.0
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
    control_history: list[dict] = field(default_factory=list)
    _sparsity_pattern: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.active is None:
            self.active = ActiveCellMap3D(self.grid, np.ones(self.grid.n_cells, dtype=bool))
        if self.transmissibility_multipliers is not None:
            # Defensive: ensure inactive cells are barriers even if the caller
            # forgot to apply the active map when constructing multipliers.
            self.transmissibility_multipliers.apply_active_mask(self.active)

    @property
    def n(self) -> int:
        return self.grid.n_cells

    def set_wells_from_schedule(self, time: float) -> None:
        if self.schedule is not None:
            self.wells = self.schedule.wells_at(time)

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
        upper_x = np.where(active_saturated, 1.0 - self.relperm.sor - self.relperm.swc - 1.0e-8, max(self.oil.max_solution_gas_ratio(), 1.0))
        return np.concatenate([lower_p, lower_sw, lower_x]), np.concatenate([upper_p, upper_sw, upper_x])

    def _phase_pvt_relperm(self, state: StateBlackOil):
        sw, _so, sg, _rs = state.physical(self.oil)
        pw, po, pg = phase_pressures_black_oil(state.p, sw, sg, self.capillary)
        bw = self.water.formation_volume_factor(pw); bo = self.oil.formation_volume_factor(po); bg = self.gas.formation_volume_factor(pg)
        muw = self.water.viscosity(pw); muo = self.oil.viscosity(po); mug = self.gas.viscosity(pg)
        krw = self.relperm.krw(sw, sg); kro = self.relperm.kro(sw, sg); krg = self.relperm.krg(sw, sg)
        return pw, po, pg, krw, kro, krg, muw, muo, mug, bw, bo, bg

    def resolve_well_controls(self, state: StateBlackOil | None = None) -> tuple[list, list[dict]]:
        st = self.state if state is None else state
        pw, po, pg, krw, kro, krg, muw, muo, mug, bw, bo, bg = self._phase_pvt_relperm(st)
        logs = []
        for well in self.wells:
            _ctrl, _target, log = _well_effective_control_3d(well, pw, po, pg, krw, kro, krg, muw, muo, mug, bw, bo, bg)
            logs.append(log)
        return self.wells, logs

    def _valid_physical_state(self, state: StateBlackOil) -> bool:
        sw, so, sg, rs = state.physical(self.oil)
        rs_sat = self.oil.solution_gas_ratio(state.p)
        mask = self.active.active_mask if self.active is not None else np.ones(self.n, dtype=bool)
        return bool(
            np.all(np.isfinite(state.p)) and np.all(np.isfinite(sw)) and np.all(np.isfinite(sg))
            and np.all(state.p[mask] > 0.0)
            and np.all(sw[mask] >= self.relperm.swc - 1.0e-7)
            and np.all(so[mask] >= self.relperm.sor - 1.0e-7)
            and np.all(sg[mask] >= -1.0e-8)
            and np.all(rs[mask] >= -1.0e-8)
            and np.all(rs[mask] <= self.oil.max_solution_gas_ratio() + 1.0e-6)
            and np.all(rs[(~state.is_saturated) & mask] <= rs_sat[(~state.is_saturated) & mask] + 1.0e-5)
        )

    def apply_phase_switching(self, candidate: StateBlackOil) -> tuple[StateBlackOil, int, int]:
        p = candidate.p.copy(); sw = candidate.sw.copy(); x = candidate.x.copy(); sat = candidate.is_saturated.copy()
        rs_sat = self.oil.solution_gas_ratio(p)
        bo = self.oil.formation_volume_factor(p)
        bg = self.gas.formation_volume_factor(p)
        mask = self.active.active_mask if self.active is not None else np.ones(self.n, dtype=bool)
        to_sat = 0; to_unsat = 0
        for i in range(self.n):
            if not mask[i]:
                continue
            if not sat[i]:
                rs_u = max(float(x[i]), 0.0)
                if rs_u > rs_sat[i] * (1.0 + 1.0e-9) + 1.0e-9:
                    so_unsat = max(1.0 - sw[i], self.relperm.sor + self.saturation_eps)
                    numerator = (rs_u - rs_sat[i]) * so_unsat / bo[i]
                    denominator = 1.0 / bg[i] - rs_sat[i] / bo[i]
                    sg_new = self.saturation_eps if denominator <= 1.0e-20 else numerator / denominator
                    sg_upper = max(1.0 - sw[i] - self.relperm.sor - self.saturation_eps, self.saturation_eps)
                    x[i] = float(np.clip(sg_new, self.saturation_eps, sg_upper)); sat[i] = True; to_sat += 1
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
                        x[i] = float(np.clip(rs_equiv, 0.0, rs_sat[i])); sat[i] = False; to_unsat += 1
                    else:
                        sg_upper = max(1.0 - sw[i] - self.relperm.sor - self.saturation_eps, self.saturation_eps)
                        x[i] = float(np.clip(sg_s, self.saturation_eps, sg_upper))
                else:
                    sg_upper = max(1.0 - sw[i] - self.relperm.sor - self.saturation_eps, self.saturation_eps)
                    x[i] = float(np.clip(sg_s, self.saturation_eps, sg_upper))
        return StateBlackOil(p=p, sw=sw, x=x, is_saturated=sat), to_sat, to_unsat

    def component_in_place(self, state: StateBlackOil | None = None) -> tuple[float, float, float]:
        st = self.state if state is None else state
        mask = self.active.active_mask if self.active is not None else np.ones(self.n, dtype=bool)
        sw, so, sg, rs = st.physical(self.oil)
        pw, po, pg = phase_pressures_black_oil(st.p, sw, sg, self.capillary)
        pv = self.rock.pore_volume(self.grid.volumes, po)
        bw = self.water.formation_volume_factor(pw); bo = self.oil.formation_volume_factor(po); bg = self.gas.formation_volume_factor(pg)
        water = float(np.sum((pv * sw / bw)[mask]))
        oil = float(np.sum((pv * so / bo)[mask]))
        gas_component = float(np.sum((pv * (rs * so / bo + sg / bg))[mask]))
        return water, oil, gas_component

    @staticmethod
    def _split_rates(q: np.ndarray) -> tuple[float, float]:
        inj = float(np.sum(np.maximum(q, 0.0)))
        prod = float(-np.sum(np.minimum(q, 0.0)))
        return inj, prod

    def well_sources(self, state: StateBlackOil | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        st = self.state if state is None else state
        sw, _so, sg, rs = st.physical(self.oil)
        pw, po, pg, krw, kro, krg, muw, muo, mug, bw, bo, bg = self._phase_pvt_relperm(st)
        qw = np.zeros(self.n, dtype=float); qo = np.zeros(self.n, dtype=float); qg_free = np.zeros(self.n, dtype=float)
        logs = []
        for well in self.wells:
            qwi, qoi, qgi, log = _three_phase_field_well_sources_3d(well, self.n, pw, po, pg, krw, kro, krg, muw, muo, mug, bw, bo, bg)
            qw += qwi; qo += qoi; qg_free += qgi; logs.append(log)
        qg_component = qg_free + rs * qo
        return qw, qo, qg_free, qg_component, logs

    def try_step(self, dt: float, max_ds: float | None = None):
        old = self.state.copy()
        active_saturated = old.is_saturated.copy()
        x0 = pack_black_oil_primary(old.p, old.sw, old.x)
        lower, upper = self.bounds(active_saturated)

        def residual(y):
            return residual_live_oil_7_3d(
                y, old, dt, self.grid, self.rock, self.water, self.oil, self.gas, self.relperm,
                wells=self.wells,
                active_saturated=active_saturated,
                capillary=self.capillary,
                gravity=self.gravity,
                transmissibility_multipliers=self.transmissibility_multipliers,
                active=self.active,
                boundaries=self.boundaries,
            )

        x_new, report = self.solver.solve(residual, x0, sparsity=self.sparsity_pattern(), lower=lower, upper=upper)
        p_new, sw_new, third_new = unpack_black_oil_primary(x_new, self.n)
        candidate = StateBlackOil(p=p_new, sw=sw_new, x=third_new, is_saturated=active_saturated)
        switched, to_sat, to_unsat = self.apply_phase_switching(candidate)
        self.state = switched
        sw_old, _so_old, sg_old, _rs_old = old.physical(self.oil)
        sw_cur, _so_cur, sg_cur, _rs_cur = self.state.physical(self.oil)
        change_indicator = max(float(np.max(np.abs(sw_cur - sw_old))), float(np.max(np.abs(sg_cur - sg_old))))
        accepted = bool(report.converged and self._valid_physical_state(self.state))
        if max_ds is not None and change_indicator > max_ds:
            accepted = False
        return accepted, report, change_indicator, to_sat, to_unsat

    def save_restart(self, path: str | Path, *, time: float | None = None, metadata: dict | None = None) -> Path:
        t = self.current_time if time is None else float(time)
        meta = {"stage": "7", "current_time": t, "grid": "3D Cartesian"}
        if metadata:
            meta.update(metadata)
        return save_black_oil_restart(path, self, t, metadata=meta)

    def load_restart_into_self(self, path: str | Path) -> float:
        restart = load_black_oil_restart(path)
        t = apply_black_oil_restart(self, restart)
        self.current_time = t
        self.set_wells_from_schedule(t)
        return t

    def run(
        self,
        t_final: float,
        dt_initial: float,
        dt_min: float,
        dt_max: float,
        *,
        t_start: float | None = None,
        max_ds: float = 0.05,
        growth: float = 1.25,
        cut: float = 0.5,
        restart_dir: str | Path | None = None,
        restart_interval: float | None = None,
    ) -> dict:
        t = self.current_time if t_start is None else float(t_start)
        self.current_time = t
        self.set_wells_from_schedule(t)
        times = [t]
        sw0, so0, sg0, rs0 = self.state.physical(self.oil)
        pressures = [self.state.p.copy()]; sw_values = [sw0.copy()]; so_values = [so0.copy()]; sg_values = [sg0.copy()]; rs_values = [rs0.copy()]
        phase_values = [self.state.is_saturated.copy()]
        reports: list[BlackOil3DStepInfo] = []
        restart_files: list[str] = []
        schedule_rows: list[dict] = []
        water0, oil0, gas0 = self.component_in_place()
        ooip0 = oil0
        dt = dt_initial
        next_restart_time = (t + restart_interval) if restart_interval is not None else None
        if self.schedule is not None:
            schedule_rows.extend(self.schedule.status_rows_at(t))

        while t < t_final - 1.0e-12:
            self.set_wells_from_schedule(t)
            next_milestone = t_final
            if self.schedule is not None:
                next_milestone = min(next_milestone, self.schedule.next_milestone_after(t, t_final))
            if next_restart_time is not None:
                next_milestone = min(next_milestone, next_restart_time)
            dt = min(dt, dt_max, next_milestone - t)
            if dt <= 1.0e-14:
                if next_restart_time is not None and abs(t - next_restart_time) < 1.0e-8:
                    next_restart_time += restart_interval
                continue

            old_state = self.state.copy()
            old_cums = {name: float(getattr(self, name)) for name in [
                "cumulative_oil_produced", "cumulative_water_produced", "cumulative_free_gas_produced", "cumulative_gas_component_produced",
                "cumulative_oil_injected", "cumulative_water_injected", "cumulative_free_gas_injected", "cumulative_gas_component_injected",
            ]}
            accepted, nreport, change_indicator, to_sat, to_unsat = self.try_step(dt, max_ds=max_ds)
            while not accepted:
                self.state = old_state.copy()
                for key, value in old_cums.items():
                    setattr(self, key, value)
                dt *= cut
                if dt < dt_min:
                    raise RuntimeError(f"3D black-oil timestep failed below dt_min: dt={dt:.6e}, residual={nreport.residual_norm:.3e}")
                accepted, nreport, change_indicator, to_sat, to_unsat = self.try_step(dt, max_ds=max_ds)

            qw, qo, qg_free, qg_component, control_log = self.well_sources(self.state)
            winj_rate, wprod_rate = self._split_rates(qw)
            oinj_rate, oprod_rate = self._split_rates(qo)
            gfree_inj_rate, gfree_prod_rate = self._split_rates(qg_free)
            gcomp_inj_rate, gcomp_prod_rate = self._split_rates(qg_component)
            self.cumulative_water_injected += winj_rate * dt; self.cumulative_water_produced += wprod_rate * dt
            self.cumulative_oil_injected += oinj_rate * dt; self.cumulative_oil_produced += oprod_rate * dt
            self.cumulative_free_gas_injected += gfree_inj_rate * dt; self.cumulative_free_gas_produced += gfree_prod_rate * dt
            self.cumulative_gas_component_injected += gcomp_inj_rate * dt; self.cumulative_gas_component_produced += gcomp_prod_rate * dt

            t += dt
            self.current_time = t
            self.set_wells_from_schedule(t)
            if self.schedule is not None and any(abs(t - et) < 1.0e-8 for et in self.schedule.event_times):
                schedule_rows.extend(self.schedule.status_rows_at(t))

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
            mask = self.active.active_mask if self.active is not None else np.ones(self.n, dtype=bool)
            info = BlackOil3DStepInfo(
                time=t, dt=dt, newton=nreport,
                min_pressure=float(np.min(self.state.p[mask])), max_pressure=float(np.max(self.state.p[mask])),
                min_sw=float(np.min(sw[mask])), max_sw=float(np.max(sw[mask])),
                min_sg=float(np.min(sg[mask])), max_sg=float(np.max(sg[mask])),
                min_so=float(np.min(so[mask])), max_so=float(np.max(so[mask])),
                min_rs=float(np.min(rs[mask])), max_rs=float(np.max(rs[mask])),
                saturated_cells=int(np.count_nonzero(self.state.is_saturated & mask)),
                undersaturated_cells=int(np.count_nonzero((~self.state.is_saturated) & mask)),
                switched_to_saturated=to_sat, switched_to_undersaturated=to_unsat,
                oil_rate=oprod_rate, water_rate=wprod_rate, free_gas_rate=gfree_prod_rate, gas_component_rate=gcomp_prod_rate,
                water_injection_rate=winj_rate,
                cumulative_oil_produced=self.cumulative_oil_produced, cumulative_water_produced=self.cumulative_water_produced,
                cumulative_free_gas_produced=self.cumulative_free_gas_produced, cumulative_gas_component_produced=self.cumulative_gas_component_produced,
                cumulative_water_injected=self.cumulative_water_injected, recovery_factor=rf, producing_gor=gor,
                oil_material_balance_error=oil_balance, water_material_balance_error=water_balance, gas_material_balance_error=gas_balance,
                oil_material_balance_error_relative=oil_balance / oil_scale, water_material_balance_error_relative=water_balance / water_scale,
                gas_material_balance_error_relative=gas_balance / gas_scale,
                active_cells=int(np.count_nonzero(mask)), inactive_cells=int(self.n - np.count_nonzero(mask)),
                linear_iterations_total=getattr(nreport, "linear_iterations_total", 0),
                jacobian_nnz=getattr(nreport, "jacobian_nnz_last", 0), jacobian_colors=getattr(nreport, "jacobian_colors", 0),
                active_controls=control_log,
            )
            reports.append(info)
            self.control_history.append({"time": t, "dt": dt, "controls": control_log})
            times.append(t); pressures.append(self.state.p.copy()); sw_values.append(sw.copy()); so_values.append(so.copy()); sg_values.append(sg.copy()); rs_values.append(rs.copy()); phase_values.append(self.state.is_saturated.copy())

            if restart_dir is not None and next_restart_time is not None and abs(t - next_restart_time) < 1.0e-8:
                path = Path(restart_dir) / f"restart_t{t:.6e}.npz"
                self.save_restart(path, time=t, metadata={"reason": "step7_interval"})
                restart_files.append(str(path))
                next_restart_time += restart_interval

            if nreport.iterations <= max(2, self.solver.max_iter // 3) and change_indicator < 0.5 * max_ds:
                dt = min(dt * growth, dt_max)
            elif nreport.iterations > 0.75 * self.solver.max_iter or change_indicator > 0.8 * max_ds:
                dt = max(dt * cut, dt_min)

        return {
            "time": np.asarray(times),
            "pressure": np.asarray(pressures),
            "sw": np.asarray(sw_values),
            "so": np.asarray(so_values),
            "sg": np.asarray(sg_values),
            "rs": np.asarray(rs_values),
            "is_saturated": np.asarray(phase_values, dtype=bool),
            "reports": reports,
            "restart_files": restart_files,
            "schedule_history": schedule_rows,
            "control_history": self.control_history,
        }
