# Step 5B — Sparse Jacobians, Sparse Linear Solvers, Newton-Krylov, and Preconditioning

Step 5B keeps the Step 5A physical model unchanged and upgrades the nonlinear
algebra. This is the first scalability-oriented stage of the black-oil framework.

## Motivation

The dense finite-difference Newton matrix used in the early prototypes is useful
for clarity but becomes inappropriate as soon as the mesh grows. A finite-volume
reservoir simulator has a naturally sparse Jacobian: each cell residual depends
mostly on the primary variables in the same cell and neighbouring cells. Step 5B
exploits this structure.

## Residual and primary variables

The conservative live-oil residual remains the Step 5A residual with primary
variables

\[
\mathbf{x}_i = (p_{o,i}, S_{w,i}, X_i),
\]

where \(X_i = R_{s,i}\) in undersaturated cells and \(X_i = S_{g,i}\) in
saturated cells. Gravity, capillary pressure, phase-state switching, and
controlled wells are preserved.

## Sparse structure

For the current 1D TPFA grid, cell \(i\) depends on \(i-1\), \(i\), and \(i+1\).
The structural Jacobian pattern is therefore block-tridiagonal in cell space.
The code implements this in

```text
src/blackoil/sparse_jacobian.py
```

using

```python
block_tridiagonal_black_oil_sparsity(nx, n_components=3)
```

The same idea will generalize to 2D and 3D by replacing the 1D neighbour list by
face-connectivity neighbour pairs.

## Colored sparse finite differences

The sparse Jacobian is built using structural column coloring. Columns that do
not influence the same residual rows can be perturbed simultaneously. This
reduces the number of residual evaluations from \(3N\) to a small number of
colors for structured grids.

This is not yet a hand-derived analytic Jacobian, but it is a major improvement
over dense finite differences and it preserves the finite-volume sparsity
structure needed for later large-scale simulation.

## Linear solvers

The new solver module is

```text
src/blackoil/sparse_solver.py
```

It supports:

- direct sparse solve with `spsolve`,
- Krylov GMRES,
- BiCGSTAB,
- LGMRES,
- matrix-free finite-difference Jacobian-vector products.

## Preconditioning

The implemented preconditioners are intentionally simple but useful:

- `none`,
- `jacobi`,
- `ilu` for explicit sparse matrices.

Matrix-free mode currently supports no preconditioner or a diagonal finite-
difference Jacobi approximation. For serious 2D/3D simulations, this layer
should later be extended with CPR-like pressure preconditioning.

## New simulator

The Step 5B simulator is

```python
ScalableBlackOilSimulator5B
```

It inherits the Step 5A physical residual and replaces the Newton solve by
`SparseNewtonSolver`.

## New example

Run:

```bash
python examples/08_live_oil_sparse_newton_krylov.py
```

The example uses sparse finite differences with GMRES and ILU preconditioning.
It writes solver diagnostics, pressure/saturation plots, and a timestep report to

```text
outputs/example_08_step5b_sparse_newton_krylov/
```

## Current approximations

- The Jacobian is sparse finite-difference, not fully analytic.
- The sparsity pattern is currently implemented for the 1D Cartesian grid.
- ILU is only a first preconditioner; CPR-style pressure preconditioning remains
  a future stage.
- Matrix-free Newton-Krylov is available but still prototype-level.

## Next step

The next planned stage is Step 5C: 2D heterogeneous reservoirs, anisotropic
permeability, boundary conditions, and validation cases. That step should also
start generalizing grid connectivity so the sparse pattern is no longer tied to a
1D grid.
