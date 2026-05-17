import numpy as np
from scipy.sparse import csr_matrix

from blackoil.sparse_jacobian import (
    block_tridiagonal_black_oil_sparsity,
    greedy_column_coloring,
    sparse_finite_difference_jacobian,
)
from blackoil.sparse_solver import SparseNewtonSolver, SparseNewtonReport


def test_block_tridiagonal_pattern_shape_and_boundaries():
    pattern = block_tridiagonal_black_oil_sparsity(nx=5, n_components=3)
    assert pattern.shape == (15, 15)
    # Boundary rows see 2 cells x 3 variables; interior rows see 3 cells x 3 variables.
    assert pattern.getrow(0).nnz == 6
    assert pattern.getrow(2).nnz == 9


def test_colored_sparse_fd_matches_dense_linear_operator_on_pattern():
    rng = np.random.default_rng(123)
    pattern = block_tridiagonal_black_oil_sparsity(nx=4, n_components=3)
    rows, cols = pattern.nonzero()
    values = rng.normal(size=len(rows))
    A = csr_matrix((values, (rows, cols)), shape=pattern.shape)
    x = rng.normal(size=A.shape[1])

    def func(z):
        return A @ z

    J, info = sparse_finite_difference_jacobian(func, x, pattern, rel_eps=1.0e-7)
    assert info.colors == len(greedy_column_coloring(pattern))
    assert info.residual_evaluations == info.colors
    np.testing.assert_allclose(J.toarray(), A.toarray(), rtol=1.0e-6, atol=1.0e-8)


def test_sparse_newton_solver_solves_small_nonlinear_system():
    def func(x):
        return np.array([x[0] ** 2 - 4.0, x[1] ** 2 - 9.0])

    pattern = csr_matrix(np.eye(2, dtype=bool))
    solver = SparseNewtonSolver(tol=1.0e-10, max_iter=10, linear_solver="spsolve")
    x, report = solver.solve(func, np.array([1.5, 2.5]), sparsity=pattern)
    assert isinstance(report, SparseNewtonReport)
    assert report.converged
    np.testing.assert_allclose(x, np.array([2.0, 3.0]), rtol=1.0e-8, atol=1.0e-8)


def test_sparse_newton_gmres_jacobi_path():
    A = csr_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    b = np.array([1.0, 2.0])

    def func(x):
        return A @ x - b

    pattern = csr_matrix(np.ones((2, 2), dtype=bool))
    solver = SparseNewtonSolver(
        tol=1.0e-10,
        max_iter=4,
        linear_solver="gmres",
        preconditioner="jacobi",
        krylov_rtol=1.0e-10,
    )
    x, report = solver.solve(func, np.zeros(2), sparsity=pattern)
    assert report.converged
    np.testing.assert_allclose(A @ x, b, rtol=1.0e-8, atol=1.0e-8)
