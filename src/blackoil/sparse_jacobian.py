from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix


@dataclass(frozen=True)
class SparseJacobianInfo:
    """Diagnostics returned together with a sparse finite-difference Jacobian."""

    nnz: int
    colors: int
    residual_evaluations: int


def block_tridiagonal_black_oil_sparsity(nx: int, n_components: int = 3) -> csr_matrix:
    """Return the structural Jacobian pattern for a 1D FV black-oil residual.

    The current black-oil residual is block ordered by component rows and block
    ordered by primary-variable columns:

    rows    = [water cells, oil cells, gas-component cells]
    columns = [pressure cells, water saturation cells, third variable cells]

    A cell residual depends on the primary variables in the same cell and in the
    immediate left/right neighbours because the TPFA finite-volume flux is a
    two-point stencil. Wells are local, so they are included by the cell centre
    part of the same pattern. The resulting matrix is sparse and block
    tridiagonal in cell space, although not literally block-tridiagonal in the
    block-ordered row/column numbering.
    """
    if nx <= 0:
        raise ValueError("nx must be positive")
    if n_components <= 0:
        raise ValueError("n_components must be positive")

    rows: list[int] = []
    cols: list[int] = []
    n = n_components * nx
    for eq_block in range(n_components):
        for cell in range(nx):
            row = eq_block * nx + cell
            for nb in (cell - 1, cell, cell + 1):
                if 0 <= nb < nx:
                    for var_block in range(n_components):
                        col = var_block * nx + nb
                        rows.append(row)
                        cols.append(col)
    data = np.ones(len(rows), dtype=bool)
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def greedy_column_coloring(pattern: csr_matrix) -> list[np.ndarray]:
    """Greedy distance-one column coloring for sparse finite differences.

    Columns with the same color must not share any residual row in the sparsity
    pattern. Such columns can be perturbed simultaneously; the individual
    Jacobian entries can then be recovered from the rows belonging to each
    column. This is a compact and transparent alternative to dense numerical
    Jacobians for early reservoir-simulation development.
    """
    pat = pattern.tocsc().astype(bool)
    ncols = pat.shape[1]
    row_to_cols = pattern.tocsr().astype(bool)

    col_colors = -np.ones(ncols, dtype=int)
    for col in range(ncols):
        rows = pat.indices[pat.indptr[col] : pat.indptr[col + 1]]
        used: set[int] = set()
        for row in rows:
            neighbours = row_to_cols.indices[row_to_cols.indptr[row] : row_to_cols.indptr[row + 1]]
            for nb_col in neighbours:
                color = col_colors[nb_col]
                if color >= 0:
                    used.add(int(color))
        color = 0
        while color in used:
            color += 1
        col_colors[col] = color

    return [np.where(col_colors == color)[0] for color in range(int(col_colors.max()) + 1)]


def sparse_finite_difference_jacobian(
    func,
    x: np.ndarray,
    pattern: csr_matrix,
    f0: np.ndarray | None = None,
    rel_eps: float = 1.0e-7,
    colors: list[np.ndarray] | None = None,
) -> tuple[csr_matrix, SparseJacobianInfo]:
    """Build a sparse finite-difference Jacobian by structural coloring.

    Parameters
    ----------
    func:
        Residual callback.
    x:
        Current Newton vector.
    pattern:
        Boolean structural sparsity pattern. Entries outside this pattern are
        assumed to be identically zero.
    f0:
        Optional residual at ``x``.
    rel_eps:
        Relative forward-difference perturbation.
    colors:
        Optional precomputed column-color groups.

    Notes
    -----
    This is not an analytic Jacobian. It is, however, much closer to the matrix
    structure used in professional finite-volume reservoir simulators than the
    previous dense finite-difference approach. The same pattern-building logic
    generalizes naturally to 2D/3D grids by replacing the neighbour list.
    """
    x = np.asarray(x, dtype=float)
    pat = pattern.tocsc().astype(bool)
    if pat.shape[0] != pat.shape[1] or pat.shape[1] != x.size:
        raise ValueError("sparsity pattern must be square and compatible with x")

    if f0 is None:
        f0 = np.asarray(func(x), dtype=float)
    else:
        f0 = np.asarray(f0, dtype=float)
    if f0.size != pat.shape[0]:
        raise ValueError("residual size is incompatible with sparsity pattern")

    if colors is None:
        colors = greedy_column_coloring(pattern)

    steps = rel_eps * np.maximum(1.0, np.abs(x))
    rows_out: list[np.ndarray] = []
    cols_out: list[np.ndarray] = []
    vals_out: list[np.ndarray] = []
    residual_evals = 0

    for group in colors:
        if len(group) == 0:
            continue
        x_pert = x.copy()
        x_pert[group] += steps[group]
        fp = np.asarray(func(x_pert), dtype=float)
        residual_evals += 1
        delta = fp - f0

        for col in group:
            start, end = pat.indptr[col], pat.indptr[col + 1]
            rows = pat.indices[start:end]
            if rows.size == 0:
                continue
            rows_out.append(rows)
            cols_out.append(np.full(rows.size, col, dtype=int))
            vals_out.append(delta[rows] / steps[col])

    if rows_out:
        rows_cat = np.concatenate(rows_out)
        cols_cat = np.concatenate(cols_out)
        vals_cat = np.concatenate(vals_out)
    else:  # pragma: no cover - impossible for the supplied patterns
        rows_cat = np.array([], dtype=int)
        cols_cat = np.array([], dtype=int)
        vals_cat = np.array([], dtype=float)

    jac = csr_matrix((vals_cat, (rows_cat, cols_cat)), shape=pat.shape)
    jac.eliminate_zeros()
    return jac, SparseJacobianInfo(nnz=int(jac.nnz), colors=len(colors), residual_evaluations=residual_evals)


def structured_grid_black_oil_sparsity(grid, n_components: int = 3) -> csr_matrix:
    """Return FV Jacobian pattern for a structured 1D/2D grid object.

    The grid must expose either ``n_cells`` and ``neighbors`` or the legacy 1D
    ``nx`` and ``neighbors`` attributes. Each residual row for a cell depends
    on that cell and all cells sharing a finite-volume face with it.
    """
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    n_cells = int(getattr(grid, "n_cells", getattr(grid, "nx")))
    if n_cells <= 0:
        raise ValueError("grid must contain at least one cell")
    adjacency: list[set[int]] = [set([i]) for i in range(n_cells)]
    pairs = np.asarray(grid.neighbors, dtype=int)
    if pairs.size:
        pairs = pairs.reshape((-1, 2))
        for a, b in pairs:
            adjacency[int(a)].add(int(b))
            adjacency[int(b)].add(int(a))

    rows: list[int] = []
    cols: list[int] = []
    n = n_components * n_cells
    for eq_block in range(n_components):
        for cell in range(n_cells):
            row = eq_block * n_cells + cell
            for nb in sorted(adjacency[cell]):
                for var_block in range(n_components):
                    rows.append(row)
                    cols.append(var_block * n_cells + nb)
    data = np.ones(len(rows), dtype=bool)
    return csr_matrix((data, (rows, cols)), shape=(n, n))
