from __future__ import annotations

from dataclasses import dataclass
import inspect
import numpy as np
from scipy.sparse import csr_matrix, eye, diags
from scipy.sparse.linalg import LinearOperator, spsolve, gmres, bicgstab, lgmres, spilu

from .nonlinear_solver import NewtonReport
from .sparse_jacobian import sparse_finite_difference_jacobian, greedy_column_coloring


@dataclass
class SparseNewtonReport(NewtonReport):
    """Newton diagnostics enriched with sparse-linear-solver information."""

    linear_solver: str = "spsolve"
    jacobian_strategy: str = "sparse_fd"
    preconditioner: str = "none"
    linear_iterations_total: int = 0
    linear_iterations_last: int = 0
    linear_info_last: int = 0
    jacobian_nnz_last: int = 0
    jacobian_colors: int = 0
    jacobian_evaluations_last: int = 0


def _call_krylov(solver, A, b, *, M=None, rtol=1.0e-6, atol=0.0, maxiter=200, restart=50):
    """Call SciPy Krylov solvers across minor API-version differences."""
    iters = {"count": 0}

    def callback(_arg):
        iters["count"] += 1

    kwargs = {"M": M, "maxiter": maxiter, "callback": callback}
    sig = inspect.signature(solver)
    if "rtol" in sig.parameters:
        kwargs["rtol"] = rtol
        kwargs["atol"] = atol
    else:  # pragma: no cover - for old SciPy releases
        kwargs["tol"] = rtol
    if solver is gmres and "restart" in sig.parameters:
        kwargs["restart"] = restart
    if solver is gmres and "callback_type" in sig.parameters:
        kwargs["callback_type"] = "legacy"
    dx, info = solver(A, b, **kwargs)
    return dx, int(info), int(iters["count"])


def _build_preconditioner(A: csr_matrix, kind: str):
    kind = (kind or "none").lower()
    n = A.shape[0]
    if kind in {"none", "identity"}:
        return None
    if kind == "jacobi":
        diag = A.diagonal().copy()
        small = np.abs(diag) < 1.0e-30
        diag[small] = 1.0
        inv_diag = 1.0 / diag
        return LinearOperator((n, n), matvec=lambda v: inv_diag * v, dtype=float)
    if kind == "ilu":
        # A very small diagonal shift makes early prototype Jacobians more
        # forgiving without materially changing the Newton correction.
        shifted = (A + 1.0e-20 * eye(n, format="csr")).tocsc()
        ilu = spilu(shifted, drop_tol=1.0e-4, fill_factor=10.0)
        return LinearOperator((n, n), matvec=ilu.solve, dtype=float)
    raise ValueError(f"Unknown preconditioner kind: {kind!r}")


def _approximate_diagonal_preconditioner(func, x: np.ndarray, f: np.ndarray, rel_eps: float):
    """Diagonal finite-difference preconditioner for matrix-free Krylov solves."""
    n = x.size
    diag = np.empty(n, dtype=float)
    for j in range(n):
        step = rel_eps * max(1.0, abs(x[j]))
        xp = x.copy()
        xp[j] += step
        fp = np.asarray(func(xp), dtype=float)
        diag[j] = (fp[j] - f[j]) / step
    small = np.abs(diag) < 1.0e-30
    diag[small] = 1.0
    inv_diag = 1.0 / diag
    return LinearOperator((n, n), matvec=lambda v: inv_diag * v, dtype=float)


@dataclass
class SparseNewtonSolver:
    """Damped Newton solver with sparse and Newton-Krylov linearizations.

    Parameters
    ----------
    jacobian_strategy:
        ``"sparse_fd"`` builds an explicit sparse finite-difference Jacobian
        using a supplied structural sparsity pattern. ``"matrix_free"`` forms a
        finite-difference Jacobian-vector product and solves the Newton system
        with a Krylov method.
    linear_solver:
        ``"spsolve"`` for direct sparse LU through SciPy, or ``"gmres"``,
        ``"bicgstab"`` or ``"lgmres"`` for Krylov solves.
    preconditioner:
        ``"none"``, ``"jacobi"`` or ``"ilu"`` for explicit sparse matrices.
        Matrix-free mode supports ``"none"`` and a diagonal finite-difference
        ``"jacobi"`` approximation.
    """

    tol: float = 1.0e-8
    max_iter: int = 14
    verbose: bool = False
    min_damping: float = 1.0e-4
    acceptable_tol: float | None = None
    acceptable_min_iterations: int = 1
    fd_rel_eps: float = 1.0e-7
    jacobian_strategy: str = "sparse_fd"
    linear_solver: str = "spsolve"
    preconditioner: str = "none"
    krylov_rtol: float = 1.0e-6
    krylov_atol: float = 0.0
    krylov_maxiter: int = 200
    gmres_restart: int = 50
    regularization: float = 0.0
    reuse_coloring: bool = True

    def __post_init__(self) -> None:
        self._colors_cache = None

    def _explicit_sparse_matrix(self, func, x, f, sparsity):
        if sparsity is None:
            raise ValueError("Sparse FD Newton requires a sparsity pattern")
        if self.reuse_coloring:
            if self._colors_cache is None:
                self._colors_cache = greedy_column_coloring(sparsity)
            colors = self._colors_cache
        else:
            colors = None
        A, info = sparse_finite_difference_jacobian(
            func, x, sparsity, f0=f, rel_eps=self.fd_rel_eps, colors=colors
        )
        if self.regularization > 0.0:
            A = A + self.regularization * eye(A.shape[0], format="csr")
        return A.tocsr(), info

    def _solve_linear(self, A, b, *, func=None, x=None, f=None):
        solver_name = self.linear_solver.lower()
        precond_name = self.preconditioner.lower()

        if solver_name == "spsolve":
            dx = spsolve(A, b)
            return np.asarray(dx, dtype=float), 0, 0

        if solver_name == "gmres":
            M = _build_preconditioner(A, precond_name)
            return _call_krylov(
                gmres,
                A,
                b,
                M=M,
                rtol=self.krylov_rtol,
                atol=self.krylov_atol,
                maxiter=self.krylov_maxiter,
                restart=self.gmres_restart,
            )
        if solver_name == "bicgstab":
            M = _build_preconditioner(A, precond_name)
            return _call_krylov(
                bicgstab,
                A,
                b,
                M=M,
                rtol=self.krylov_rtol,
                atol=self.krylov_atol,
                maxiter=self.krylov_maxiter,
            )
        if solver_name == "lgmres":
            M = _build_preconditioner(A, precond_name)
            return _call_krylov(
                lgmres,
                A,
                b,
                M=M,
                rtol=self.krylov_rtol,
                atol=self.krylov_atol,
                maxiter=self.krylov_maxiter,
            )
        raise ValueError(f"Unknown linear solver: {self.linear_solver!r}")

    def _matrix_free_operator(self, func, x: np.ndarray, f: np.ndarray):
        n = x.size
        x_norm = max(float(np.linalg.norm(x)), 1.0)

        def matvec(v):
            v = np.asarray(v, dtype=float)
            v_norm = max(float(np.linalg.norm(v)), 1.0e-30)
            step = self.fd_rel_eps * x_norm / v_norm
            return (np.asarray(func(x + step * v), dtype=float) - f) / step

        return LinearOperator((n, n), matvec=matvec, dtype=float)

    def _solve_matrix_free(self, func, x: np.ndarray, f: np.ndarray):
        Aop = self._matrix_free_operator(func, x, f)
        M = None
        if self.preconditioner.lower() == "jacobi":
            M = _approximate_diagonal_preconditioner(func, x, f, self.fd_rel_eps)
        elif self.preconditioner.lower() not in {"none", "identity"}:
            raise ValueError("Matrix-free mode currently supports only none or jacobi preconditioning")

        solver_name = self.linear_solver.lower()
        if solver_name == "spsolve":
            raise ValueError("matrix_free strategy requires a Krylov linear solver")
        if solver_name == "gmres":
            return _call_krylov(
                gmres,
                Aop,
                -f,
                M=M,
                rtol=self.krylov_rtol,
                atol=self.krylov_atol,
                maxiter=self.krylov_maxiter,
                restart=self.gmres_restart,
            )
        if solver_name == "bicgstab":
            return _call_krylov(
                bicgstab,
                Aop,
                -f,
                M=M,
                rtol=self.krylov_rtol,
                atol=self.krylov_atol,
                maxiter=self.krylov_maxiter,
            )
        if solver_name == "lgmres":
            return _call_krylov(
                lgmres,
                Aop,
                -f,
                M=M,
                rtol=self.krylov_rtol,
                atol=self.krylov_atol,
                maxiter=self.krylov_maxiter,
            )
        raise ValueError(f"Unknown matrix-free Krylov solver: {self.linear_solver!r}")

    def solve(self, func, x0: np.ndarray, *, sparsity: csr_matrix | None = None, lower=None, upper=None):
        x = np.asarray(x0, dtype=float).copy()
        lower_arr = None if lower is None else np.asarray(lower, dtype=float)
        upper_arr = None if upper is None else np.asarray(upper, dtype=float)

        f = np.asarray(func(x), dtype=float)
        norm0 = float(np.linalg.norm(f, ord=2))
        if self.verbose:
            print(f"    Sparse Newton initial residual: {norm0:.6e}")
        if norm0 < self.tol:
            return x, SparseNewtonReport(
                True,
                0,
                norm0,
                1.0,
                linear_solver=self.linear_solver,
                jacobian_strategy=self.jacobian_strategy,
                preconditioner=self.preconditioner,
            )

        last_damping = 1.0
        total_linear_iters = 0
        last_linear_iters = 0
        last_linear_info = 0
        last_nnz = 0
        last_colors = 0
        last_jac_evals = 0

        for iteration in range(1, self.max_iter + 1):
            strategy = self.jacobian_strategy.lower()
            if strategy == "sparse_fd":
                A, jac_info = self._explicit_sparse_matrix(func, x, f, sparsity)
                last_nnz = jac_info.nnz
                last_colors = jac_info.colors
                last_jac_evals = jac_info.residual_evaluations
                try:
                    dx, lin_info, lin_iters = self._solve_linear(A, -f)
                except Exception:
                    # Fallback to direct sparse solve if a Krylov/preconditioner
                    # choice is too optimistic for the current nonlinear state.
                    dx = spsolve(A, -f)
                    lin_info = 0
                    lin_iters = 0
            elif strategy == "matrix_free":
                dx, lin_info, lin_iters = self._solve_matrix_free(func, x, f)
            else:
                raise ValueError(f"Unknown Jacobian strategy: {self.jacobian_strategy!r}")

            last_linear_iters = int(lin_iters)
            last_linear_info = int(lin_info)
            total_linear_iters += int(lin_iters)

            if not np.all(np.isfinite(dx)):
                return x, SparseNewtonReport(
                    False,
                    iteration,
                    float(np.linalg.norm(f, ord=2)),
                    last_damping,
                    linear_solver=self.linear_solver,
                    jacobian_strategy=self.jacobian_strategy,
                    preconditioner=self.preconditioner,
                    linear_iterations_total=total_linear_iters,
                    linear_iterations_last=last_linear_iters,
                    linear_info_last=last_linear_info,
                    jacobian_nnz_last=last_nnz,
                    jacobian_colors=last_colors,
                    jacobian_evaluations_last=last_jac_evals,
                )

            damping = 1.0
            accepted = False
            current_norm = float(np.linalg.norm(f, ord=2))
            candidate = x.copy()
            f_candidate = f.copy()

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
                x = candidate
                f = f_candidate
                last_damping = damping

            norm = float(np.linalg.norm(f, ord=2))
            if self.verbose:
                print(
                    f"    Sparse Newton {iteration:02d}: residual={norm:.6e}, "
                    f"damping={last_damping:.3e}, lin_it={last_linear_iters}, nnz={last_nnz}"
                )
            if norm < self.tol:
                return x, SparseNewtonReport(
                    True,
                    iteration,
                    norm,
                    last_damping,
                    linear_solver=self.linear_solver,
                    jacobian_strategy=self.jacobian_strategy,
                    preconditioner=self.preconditioner,
                    linear_iterations_total=total_linear_iters,
                    linear_iterations_last=last_linear_iters,
                    linear_info_last=last_linear_info,
                    jacobian_nnz_last=last_nnz,
                    jacobian_colors=last_colors,
                    jacobian_evaluations_last=last_jac_evals,
                )
            if self.acceptable_tol is not None and iteration >= self.acceptable_min_iterations and norm < self.acceptable_tol:
                return x, SparseNewtonReport(
                    True,
                    iteration,
                    norm,
                    last_damping,
                    linear_solver=self.linear_solver,
                    jacobian_strategy=self.jacobian_strategy,
                    preconditioner=self.preconditioner,
                    linear_iterations_total=total_linear_iters,
                    linear_iterations_last=last_linear_iters,
                    linear_info_last=last_linear_info,
                    jacobian_nnz_last=last_nnz,
                    jacobian_colors=last_colors,
                    jacobian_evaluations_last=last_jac_evals,
                )

        return x, SparseNewtonReport(
            False,
            self.max_iter,
            float(np.linalg.norm(f, ord=2)),
            last_damping,
            linear_solver=self.linear_solver,
            jacobian_strategy=self.jacobian_strategy,
            preconditioner=self.preconditioner,
            linear_iterations_total=total_linear_iters,
            linear_iterations_last=last_linear_iters,
            linear_info_last=last_linear_info,
            jacobian_nnz_last=last_nnz,
            jacobian_colors=last_colors,
            jacobian_evaluations_last=last_jac_evals,
        )
