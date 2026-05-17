# Step 6A — Structured 3D mesh module and corner-point preparation

Step 6A introduces the mesh layer required before the framework can become a full 3D black-oil simulator. The physics modules from Steps 4C–5D are preserved. This stage focuses on geometry, indexing, transmissibility infrastructure, property mapping, sparse stencil preparation and visualization/export.

## 1. Structured Cartesian 3D grid

The new module is:

```text
src/blackoil/grid3d.py
```

The main class is:

```python
CartesianGrid3D(nx, ny, nz, lx, ly, lz, top_depth=0.0, depth=None)
```

Cells are flattened with x as the fastest index:

\[
\mathrm{cell}(i,j,k)=kN_xN_y+jN_x+i.
\]

The default cell-centre depth is positive downward:

\[
D_{ijk}=D_{top}+z_k.
\]

An explicit scalar or array-valued depth field can also be supplied, which allows structural dip to be represented before a true corner-point geometry is introduced.

## 2. 3D transmissibilities

The grid computes harmonic TPFA geometric transmissibilities in all three directions:

\[
T_x = \frac{A_x}{\frac{\Delta x}{2K_{x,L}}+\frac{\Delta x}{2K_{x,R}}},
\]

\[
T_y = \frac{A_y}{\frac{\Delta y}{2K_{y,L}}+\frac{\Delta y}{2K_{y,R}}},
\]

\[
T_z = \frac{A_z}{\frac{\Delta z}{2K_{z,L}}+\frac{\Delta z}{2K_{z,R}}}.
\]

The permeability input may be scalar, flat array, 3D array, tuple/list, or dictionary:

```python
permeability = {"kx": Kx, "ky": Ky, "kz": Kz}
```

This prepares the code for anisotropic 3D reservoir models.

## 3. Property mapping utilities

The new module

```text
src/blackoil/properties3d.py
```

contains utilities for synthetic validation fields:

- `layered_permeability_3d`,
- `lognormal_permeability_3d`,
- `gaussian_channel_permeability_3d`,
- `anisotropic_permeability_3d`,
- `porosity_from_permeability_3d`.

These functions are not intended to replace geological modelling. They provide controlled fields for solver development, visualization and regression testing.

## 4. Sparse stencil preparation

The existing sparse Jacobian infrastructure now accepts the 3D grid through the generic neighbor interface:

```python
structured_grid_black_oil_sparsity(grid, n_components=3)
```

For a 3D TPFA finite-volume model, each cell couples to itself and its face neighbors. Therefore the sparse pattern is the natural 7-point cell stencil, expanded by the number of primary variables and component equations.

## 5. VTK/ParaView export

The new module

```text
src/blackoil/visualization3d.py
```

exports legacy VTK rectilinear-grid files:

```python
write_vtk_rectilinear_grid_3d("state.vtk", grid, cell_data)
```

The output can be opened directly in ParaView and contains cell-centred scalar fields such as pressure, saturations, porosity, permeability and depth.

## 6. Corner-point/corner-grid preparation

The new module

```text
src/blackoil/cornerpoint.py
```

introduces a lightweight `CornerPointGridSpec` container with:

- `coord`,
- `zcorn`,
- `actnum`.

This is not yet a full GRDECL parser or corner-point flow grid. It defines the interface that later importers will use. A helper function

```python
make_cartesian_cornerpoint_spec(grid)
```

creates a corner-point-like representation of the Cartesian grid for testing and future interoperability.

## 7. New example

Run:

```bash
python examples/11_3d_mesh_property_visualization.py
```

The example writes:

```text
outputs/example_11_step6a_3d_mesh/
```

including:

- `step6a_3d_mesh_state.vtk`,
- permeability, porosity, pressure and gas-saturation slice figures,
- `step6a_mesh_summary.txt`.

## 8. What remains approximate

Step 6A does not yet solve the full 3D black-oil flow equations. It provides the mesh infrastructure required by that solver. Boundary-condition and well-completion logic in 3D is still limited to geometry helpers. Corner-point support is also only an interface and validation container at this stage.

## 9. Next step

The next natural step is **Step 6B: 3D property mapping, layered reservoirs, faults/transmissibility multipliers, inactive cells, and well trajectories/completions**. After that, Step 6C can focus on advanced 3D visualization, restart visualization and ParaView/PyVista workflows.
