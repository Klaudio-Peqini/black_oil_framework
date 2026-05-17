from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve

from .jacobian import finite_difference_jacobian


@dataclass
class NewtonReport:
    converged: bool
    iterations: int
    residual_norm: float
    damping: float


@dataclass
class NewtonSolver:
    """Newton-Raphson solver with finite-difference Jacobian and damping."""

    tol: float = 1.0e-8
    max_iter: int = 12
    rel_jac_eps: float = 1.0e-7
    verbose: bool = False
    min_damping: float = 1.0e-4
    acceptable_tol: float | None = None
    acceptable_min_iterations: int = 1

    def solve(self, func, x0: np.ndarray, lower=None, upper=None) -> tuple[np.ndarray, NewtonReport]:
        x = np.asarray(x0, dtype=float).copy()
        lower_arr = None if lower is None else np.asarray(lower, dtype=float)
        upper_arr = None if upper is None else np.asarray(upper, dtype=float)

        last_damping = 1.0
        f = np.asarray(func(x), dtype=float)
        norm0 = float(np.linalg.norm(f, ord=2))
        if self.verbose:
            print(f"    Newton initial residual: {norm0:.6e}")
        if norm0 < self.tol:
            return x, NewtonReport(True, 0, norm0, 1.0)

        for iteration in range(1, self.max_iter + 1):
            jac = finite_difference_jacobian(func, x, f0=f, rel_eps=self.rel_jac_eps)
            try:
                dx = solve(jac, -f, assume_a="gen")
            except Exception as exc:  # pragma: no cover - rare fallback path
                raise RuntimeError(f"Linear solve failed during Newton iteration {iteration}: {exc}") from exc

            damping = 1.0
            accepted = False
            current_norm = float(np.linalg.norm(f, ord=2))

            while damping >= self.min_damping:
                candidate = x + damping * dx
                if lower_arr is not None:
                    candidate = np.maximum(candidate, lower_arr)
                if upper_arr is not None:
                    candidate = np.minimum(candidate, upper_arr)

                f_candidate = np.asarray(func(candidate), dtype=float)
                candidate_norm = float(np.linalg.norm(f_candidate, ord=2))

                if candidate_norm < current_norm or candidate_norm < self.tol:
                    x = candidate
                    f = f_candidate
                    last_damping = damping
                    accepted = True
                    break

                damping *= 0.5

            if not accepted:
                # Accept the smallest damped step rather than stopping abruptly.
                x = candidate
                f = f_candidate
                last_damping = damping

            norm = float(np.linalg.norm(f, ord=2))
            if self.verbose:
                print(f"    Newton {iteration:02d}: residual={norm:.6e}, damping={last_damping:.3e}")
            if norm < self.tol:
                return x, NewtonReport(True, iteration, norm, last_damping)
            if (
                self.acceptable_tol is not None
                and iteration >= self.acceptable_min_iterations
                and norm < self.acceptable_tol
            ):
                return x, NewtonReport(True, iteration, norm, last_damping)

        return x, NewtonReport(False, self.max_iter, float(np.linalg.norm(f, ord=2)), last_damping)
