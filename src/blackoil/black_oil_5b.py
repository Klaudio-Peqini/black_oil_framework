from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .black_oil_5a import AdvancedBlackOilSimulator5A, residual_live_oil_5a_gravity_capillary
from .black_oil_phase import pack_black_oil_primary, unpack_black_oil_primary
from .state import StateBlackOil
from .sparse_jacobian import block_tridiagonal_black_oil_sparsity
from .sparse_solver import SparseNewtonSolver


@dataclass
class ScalableBlackOilSimulator5B(AdvancedBlackOilSimulator5A):
    """Step 5B simulator with sparse Newton and Newton-Krylov infrastructure.

    The governing equations and physics are inherited from Step 5A. The upgrade
    is numerical: the Newton matrix now uses the finite-volume sparsity pattern,
    sparse direct or Krylov linear solvers, and optional preconditioning. This is
    the bridge from small educational prototypes to 2D/3D research-scale grids.
    """

    solver: SparseNewtonSolver = field(
        default_factory=lambda: SparseNewtonSolver(
            tol=1.0e-8,
            max_iter=16,
            acceptable_tol=5.0e-4,
            acceptable_min_iterations=2,
            jacobian_strategy="sparse_fd",
            linear_solver="spsolve",
            preconditioner="none",
        )
    )
    _sparsity_pattern: object | None = field(default=None, init=False, repr=False)

    def sparsity_pattern(self):
        if self._sparsity_pattern is None:
            self._sparsity_pattern = block_tridiagonal_black_oil_sparsity(self.grid.nx, n_components=3)
        return self._sparsity_pattern

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

        y_new, report = self.solver.solve(
            func_y,
            y0,
            sparsity=self.sparsity_pattern(),
            lower=lower_y,
            upper=upper_y,
        )

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
