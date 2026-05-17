# Step 5C — 2D heterogeneous reservoirs, anisotropy, boundary conditions, and validation

Step 5C extends the previous 1D sparse black-oil framework into a structured 2D finite-volume reservoir model. The physical model is the same conservative live-oil formulation introduced in Steps 4C--5B, but the spatial discretization now uses a 2D Cartesian stencil and supports heterogeneous anisotropic permeability.

## New grid layer

The new class `CartesianGrid2D` stores a logically Cartesian grid with

\[
N = N_x N_y
\]

cell-centred unknowns. Cells are flattened according to

\[
I(i,j) = jN_x+i.
\]

The cell volume is

\[
V_{ij} = \Delta x\Delta y h,
\]

where \(h\) is the reservoir thickness.

## Heterogeneous anisotropic permeability

The rock permeability may now be supplied as

```python
permeability={"kx": Kx, "ky": Ky}
```

where `Kx` and `Ky` may be scalars, flattened arrays, or arrays of shape `(ny, nx)`.

The transmissibility across an x-face is

\[
T_x = \frac{A_x}{\frac{\Delta x}{2K_{x,L}}+\frac{\Delta x}{2K_{x,R}}},
\]

and the transmissibility across a y-face is

\[
T_y = \frac{A_y}{\frac{\Delta y}{2K_{y,B}}+\frac{\Delta y}{2K_{y,T}}}.
\]

This is the standard TPFA harmonic transmissibility for an orthogonal structured grid.

## 2D fluxes

For each phase \(\alpha\), the face flux is still written in potential form:

\[
F_{\alpha,f}
= -T_f \lambda_{\alpha,up}
\left[
(p_{\alpha,j}-p_{\alpha,i})
- \rho_{\alpha,up}g(D_j-D_i)
\right].
\]

The capillary-pressure definitions remain

\[
p_w = p_o-p_{cow}(S_w),
\]

\[
p_g = p_o+p_{cgo}(S_g).
\]

The gas-component flux is

\[
F_{G,f}
= R_{s,up(o)}F_{o,f}+F_{g,f}.
\]

## Boundary conditions

The default boundary condition is no-flow on all sides. A simple pressure boundary has been added through

```python
BoundaryConditions2D([
    PressureBoundary("left", pressure=..., sw=..., sg=..., rs=...),
])
```

Pressure boundaries are treated as external half-cells with prescribed oil pressure and phase saturations. They are useful for pressure-support and depletion tests. Flux/rate boundaries are intentionally deferred because field-scale development will mostly use wells and schedules.

## Wells

The Step 5A/5B well-control objects are retained:

- `ControlledWell(control="water_rate", max_bhp=...)`
- `ControlledWell(control="total_rate", min_bhp=...)`
- `BHPWell`
- `RateWell`
- `MultiRateWell`

For 2D, wells are completed in flattened cell indices, for example:

```python
inj = grid.cell_index(0, 0)
prod = grid.cell_index(grid.nx - 1, grid.ny - 1)
```

## Sparse Jacobian pattern

A new generic structured-grid sparsity pattern is provided:

```python
structured_grid_black_oil_sparsity(grid, n_components=3)
```

Each cell residual depends on the same cell and its face-neighbour cells. For a 2D Cartesian grid this gives a five-point finite-volume stencil. Since the unknowns are block ordered as

\[
[p_1,\ldots,p_N,S_{w,1},\ldots,S_{w,N},x_1,\ldots,x_N],
\]

the full Jacobian has a sparse block structure rather than a dense matrix.

## Validation examples

The main new runnable example is

```bash
python examples/09_2d_heterogeneous_black_oil.py
```

It includes:

- a 2D areal reservoir,
- a high-permeability channel,
- anisotropic permeability \(K_y < K_x\),
- corner injector/producer wells,
- capillary pressure,
- gravity support,
- conservative live-oil phase switching,
- sparse Newton/GMRES/ILU solution,
- 2D maps and VTK export.

The output directory is

```text
outputs/example_09_step5c_2d_heterogeneous/
```

and contains PNG maps, a CSV timestep report, an NPZ data archive, and

```text
final_state.vtk
```

which can be opened in ParaView.

## What remains approximate

Step 5C is still a structured-grid prototype. The important remaining limitations are:

1. The grid is orthogonal Cartesian, not yet corner-point or unstructured.
2. Wells are cell-centred and do not yet support multi-layer completions.
3. Boundary conditions are simple constant-pressure states.
4. Capillary and relative-permeability hysteresis are not included.
5. The Jacobian is still sparse finite-difference, not analytic sparse assembly.
6. 3D meshing and full visualization are still planned for Step 6.

## Next planned step

The next step should be **Step 5D: production schedules, rate/BHP switching histories, restart files, and multiwell field-style controls**.

After that, the framework should move to Step 6A, where the 3D mesh module begins.
