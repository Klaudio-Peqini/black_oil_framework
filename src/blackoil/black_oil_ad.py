from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .state import StateBlackOil
from .black_oil_phase import (
    LiveOilPhaseSwitchingSimulator,
    BlackOilStepInfo,
    pack_black_oil_primary,
    unpack_black_oil_primary,
    interpret_black_oil_primary,
)
from .flux import three_phase_black_oil_face_fluxes_with_rs, divergence_from_face_flux
from .wells import RateWell, BHPWell
from .nonlinear_solver import NewtonSolver
from .ad_solver import NewtonSolverWithJacobian


def residual_live_oil_phase_switching_conservative(
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
    """Fully conservative fixed-phase-map black-oil residual.

    This is the Step 4C residual. It differs from the Step 4B bridge residual in
    one decisive point: the gas-component conservation equation is active in
    every cell. In undersaturated cells the third primary variable is Rs and
    Sg=0; in saturated cells the third primary variable is Sg and Rs=Rs_sat(p).
    Hence the row for the third unknown is always the conservative gas-component
    equation, rather than an Rs-freezing equation.
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
    gas_scale = max(float(oil.max_solution_gas_ratio()), 1.0)
    return np.concatenate([
        acc_w + div_w - qw,
        acc_o + div_o - qo,
        (acc_g + div_g - qg_component) / gas_scale,
    ])


class JaxUnavailable(RuntimeError):
    pass


@dataclass
class JaxBlackOilADContext:
    """JAX automatic-differentiation context for the Step 4C residual.

    The context stores all scalar/grid/table data needed to evaluate the same
    conservative residual as ``residual_live_oil_phase_switching_conservative``.
    Its ``jacobian`` method returns dR/dy for the scaled Newton variables y.
    """

    grid: object
    rock: object
    water: object
    oil: object
    gas: object
    relperm: object
    old: StateBlackOil
    wells: list
    active_saturated: np.ndarray
    dt: float
    scales: np.ndarray

    def __post_init__(self) -> None:
        try:
            import jax  # noqa: F401
            jax.config.update("jax_enable_x64", True)
            import jax.numpy as jnp  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on local env
            raise JaxUnavailable("JAX is not installed; use finite-difference Jacobian mode") from exc
        self._jac_cache = None

    @staticmethod
    def _table_arrays(fluid):
        table = fluid.table
        return table.pressure, table.columns[fluid.b_key], table.columns[fluid.mu_key], fluid.density_key

    def _interp(self, jnp, p, table, values):
        return jnp.interp(jnp.clip(p, table[0], table[-1]), table, values)

    def residual_y(self, y):
        import jax.numpy as jnp

        nx = self.grid.nx
        x = y * jnp.asarray(self.scales)
        p = x[:nx]
        sw = x[nx:2 * nx]
        third = x[2 * nx:]
        sat = jnp.asarray(self.active_saturated)

        # PVT closures by table interpolation.
        ptab = jnp.asarray(self.oil.table.pressure)
        bwtab = jnp.asarray(self.water.table.columns[self.water.b_key])
        botab = jnp.asarray(self.oil.table.columns[self.oil.b_key])
        bgtab = jnp.asarray(self.gas.table.columns[self.gas.b_key])
        muwtab = jnp.asarray(self.water.table.columns[self.water.mu_key])
        muotab = jnp.asarray(self.oil.table.columns[self.oil.mu_key])
        mugtab = jnp.asarray(self.gas.table.columns[self.gas.mu_key])
        rsttab = jnp.asarray(self.oil.table.columns[self.oil.rs_key])

        bw = self._interp(jnp, p, ptab, bwtab)
        bo = self._interp(jnp, p, ptab, botab)
        bg = self._interp(jnp, p, ptab, bgtab)
        muw = self._interp(jnp, p, ptab, muwtab)
        muo = self._interp(jnp, p, ptab, muotab)
        mug = self._interp(jnp, p, ptab, mugtab)
        rs_sat = self._interp(jnp, p, ptab, rsttab)

        p_old = jnp.asarray(self.old.p)
        sw_old = jnp.asarray(self.old.sw)
        x_old = jnp.asarray(self.old.x)
        sat_old = jnp.asarray(self.old.is_saturated)
        rs_sat_old = self._interp(jnp, p_old, ptab, rsttab)
        sg_old = jnp.where(sat_old, x_old, 0.0)
        rs_old = jnp.where(sat_old, rs_sat_old, x_old)
        so_old = 1.0 - sw_old - sg_old

        sg = jnp.where(sat, third, 0.0)
        rs = jnp.where(sat, rs_sat, third)
        so = 1.0 - sw - sg

        phi = jnp.asarray(self.rock.porosity_ref) * jnp.exp(self.rock.compressibility * (p - self.rock.p_ref))
        phi_old = jnp.asarray(self.rock.porosity_ref) * jnp.exp(self.rock.compressibility * (p_old - self.rock.p_ref))
        bw_old = self._interp(jnp, p_old, ptab, bwtab)
        bo_old = self._interp(jnp, p_old, ptab, botab)
        bg_old = self._interp(jnp, p_old, ptab, bgtab)
        vol = jnp.asarray(self.grid.volumes)

        acc_w = vol * (phi * sw / bw - phi_old * sw_old / bw_old) / self.dt
        acc_o = vol * (phi * so / bo - phi_old * so_old / bo_old) / self.dt
        acc_g = vol * (phi * (rs * so / bo + sg / bg) - phi_old * (rs_old * so_old / bo_old + sg_old / bg_old)) / self.dt

        # Corey three-phase relperm. The clipping derivative is zero outside the
        # physical interval, matching the safeguarded residual used by the code.
        denom = 1.0 - self.relperm.swc - self.relperm.sor - self.relperm.sgc
        swe = jnp.clip((sw - self.relperm.swc) / denom, 0.0, 1.0)
        sge = jnp.clip((sg - self.relperm.sgc) / denom, 0.0, 1.0)
        soe = jnp.clip((so - self.relperm.sor) / denom, 0.0, 1.0)
        krw = self.relperm.krw0 * swe ** self.relperm.nw
        kro = self.relperm.kro0 * soe ** self.relperm.no
        krg = self.relperm.krg0 * sge ** self.relperm.ng

        tgeo = jnp.asarray(self.grid.geometric_transmissibility(self.rock.permeability))
        dp = p[1:] - p[:-1]
        use_left = dp <= 0.0
        idx_l = jnp.arange(nx - 1)
        idx_r = jnp.arange(1, nx)
        up = jnp.where(use_left, idx_l, idx_r)

        mob_w = krw[up] / (muw[up] * bw[up])
        mob_o = kro[up] / (muo[up] * bo[up])
        mob_g = krg[up] / (mug[up] * bg[up])
        fw = -tgeo * mob_w * dp
        fo = -tgeo * mob_o * dp
        fg_free = -tgeo * mob_g * dp
        fg_comp = rs[up] * fo + fg_free

        def div(face_flux):
            out = jnp.zeros(nx)
            out = out.at[:-1].add(face_flux)
            out = out.at[1:].add(-face_flux)
            return out

        div_w = div(fw)
        div_o = div(fo)
        div_g = div(fg_comp)

        qw = jnp.zeros(nx)
        qo = jnp.zeros(nx)
        qg_free = jnp.zeros(nx)
        for well in self.wells:
            c = int(well.cell)
            if isinstance(well, RateWell):
                if well.phase == "water" or well.phase == "single":
                    qw = qw.at[c].add(well.rate)
                elif well.phase == "oil":
                    qo = qo.at[c].add(well.rate)
                elif well.phase == "gas":
                    qg_free = qg_free.at[c].add(well.rate)
            elif isinstance(well, BHPWell):
                drawdown = well.bhp - p[c]
                qw = qw.at[c].add(well.well_index * krw[c] / (muw[c] * bw[c]) * drawdown)
                qo = qo.at[c].add(well.well_index * kro[c] / (muo[c] * bo[c]) * drawdown)
                qg_free = qg_free.at[c].add(well.well_index * krg[c] / (mug[c] * bg[c]) * drawdown)
            else:  # pragma: no cover
                raise TypeError(f"Unsupported well type: {type(well)!r}")

        qg_component = qg_free + rs * qo
        gas_scale = max(float(self.oil.max_solution_gas_ratio()), 1.0)
        return jnp.concatenate([acc_w + div_w - qw, acc_o + div_o - qo, (acc_g + div_g - qg_component) / gas_scale])

    def residual_numpy(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(self.residual_y(y), dtype=float)

    def jacobian(self, y: np.ndarray) -> np.ndarray:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        if self._jac_cache is None:
            self._jac_cache = jax.jacfwd(self.residual_y)
        return np.asarray(self._jac_cache(jnp.asarray(y, dtype=float)), dtype=float)


@dataclass
class ConservativeLiveOilPhaseSwitchingSimulator(LiveOilPhaseSwitchingSimulator):
    """Step 4C conservative phase-switching black-oil simulator.

    Compared with Step 4B, this class solves a conservative gas-component row
    in both undersaturated and saturated cells. The Newton matrix can be built
    by automatic differentiation through JAX. If JAX is unavailable or if
    ``jacobian_mode='finite_difference'``, the class falls back to the existing
    numerical Jacobian path.
    """

    solver: NewtonSolverWithJacobian | NewtonSolver = field(default_factory=lambda: NewtonSolverWithJacobian(tol=1.0e-8, max_iter=18))
    jacobian_mode: str = "jax"

    def try_step(self, dt: float, max_ds: float | None = None) -> tuple[bool, object, float, int, int]:
        old = self.state.copy()
        active_saturated = old.is_saturated.copy()
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
            return residual_live_oil_phase_switching_conservative(
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

        jac_y = None
        if self.jacobian_mode.lower() in {"jax", "ad", "auto", "automatic"}:
            try:
                ad_context = JaxBlackOilADContext(
                    grid=self.grid,
                    rock=self.rock,
                    water=self.water,
                    oil=self.oil,
                    gas=self.gas,
                    relperm=self.relperm,
                    old=old,
                    wells=self.wells,
                    active_saturated=active_saturated,
                    dt=dt,
                    scales=scales,
                )
                jac_y = ad_context.jacobian
            except Exception:
                jac_y = None

        if isinstance(self.solver, NewtonSolverWithJacobian):
            y_new, report = self.solver.solve(func_y, y0, jac=jac_y, lower=lower_y, upper=upper_y)
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
        return accepted, report, change_indicator, to_sat, to_unsat
