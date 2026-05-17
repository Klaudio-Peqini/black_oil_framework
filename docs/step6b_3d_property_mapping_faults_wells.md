# Step 6B — 3D property mapping, inactive cells, faults and well completions

This stage extends the Step 6A structured 3D mesh into a reservoir-description layer that is suitable for a later full 3D black-oil flow simulator. The objective is not yet to solve the full 3D multiphase flow equations. Instead, Step 6B builds the infrastructure that a 3D finite-volume solver must have before it can be scientifically credible: active-cell indexing, geological property mapping, transmissibility multipliers, faults, and completion-level wells.

## 1. New modules

Step 6B adds three main modules.

```text
src/blackoil/reservoir3d.py
src/blackoil/faults3d.py
src/blackoil/wells3d.py
```

The modules are intentionally independent from the 1D/2D flow simulators. They can therefore be tested and refined without destabilizing the already validated black-oil physics in Steps 4C–5D.

## 2. Active-cell handling

Real field grids usually contain inactive cells: cells outside the reservoir outline, pinched-out cells, cells removed by `ACTNUM`, or cells made inactive during a geological import workflow. Step 6B introduces `ActiveCellMap3D`, which stores the full grid but exposes a compact active-cell representation.

The mapping is based on a Boolean array:

\[
A_i =
\begin{cases}
1, & \text{cell } i \text{ active},\\
0, & \text{cell } i \text{ inactive}.
\end{cases}
\]

The object provides:

- `active_cells`, the full-grid indices of active cells;
- `inactive_cells`, the full-grid indices of inactive cells;
- `full_to_active`, a map from full-grid index to active index;
- `compress(values)`, which extracts active-cell values;
- `expand(active_values)`, which restores an active vector to the full grid;
- `active_neighbor_pairs(face_neighbors)`, which filters finite-volume faces to active-active connections.

The helper functions are:

```python
active_mask_from_indices(grid, inactive_indices)
active_mask_from_boxes(grid, boxes)
```

The box convention is half-open:

```python
{"i0": 4, "i1": 8, "j0": 0, "j1": 5, "k0": 2, "k1": 4}
```

which deactivates:

\[
4 \le i < 8,\qquad 0 \le j < 5,\qquad 2 \le k < 4.
\]

## 3. Property mapping

The module `reservoir3d.py` also introduces utilities for mapping geological and petrophysical properties onto the Cartesian 3D grid.

### 3.1 Cell-array normalization

All property functions use the same accepted input convention:

- scalar value;
- flat array of size `n_cells`;
- 3D array of shape `(nz, ny, nx)`.

The returned array is always flat and follows the native grid ordering:

\[
\mathrm{cell}(i,j,k)=kN_xN_y+jN_x+i.
\]

The common helper is:

```python
as_cell_array_3d(grid, values, name="field")
```

### 3.2 Zone assignment

The framework supports zone generation by layer or by depth interval:

```python
zone_from_layers(grid, layer_ids)
zone_from_depth_intervals(grid, intervals, default=0)
```

A vertical zoning example is:

```python
zones = zone_from_layers(grid, [1, 1, 2, 2, 3, 3])
```

A depth-based zoning example is:

```python
zones = zone_from_depth_intervals(
    grid,
    [(1600, 1620, 1), (1620, 1650, 2), (1650, 1700, 3)],
)
```

### 3.3 Property assignment by zone

Zone values are mapped with:

```python
map_property_by_zone(grid, zone_ids, values_by_zone)
```

For example:

```python
kx_md = map_property_by_zone(
    grid,
    zones,
    {1: 400.0, 2: 150.0, 3: 40.0},
    name="Kx",
)
```

### 3.4 Region multipliers

Local baffles, streaks, or edited regions can be introduced using:

```python
apply_region_multiplier(grid, values, multiplier, box={...})
```

This is useful for synthetic validation cases and for building controlled heterogeneity before importing field data.

### 3.5 ReservoirPropertyModel3D

The central property container is:

```python
ReservoirPropertyModel3D
```

It stores:

- porosity;
- anisotropic permeability dictionary with `kx`, `ky`, `kz`;
- active-cell map;
- optional zone IDs;
- optional net-to-gross field.

The active pore volume is:

\[
PV_i = A_i\,\phi_i\,NTG_i\,V_i,
\]

where `A_i` is the active mask. Inactive cells therefore contribute zero pore volume.

## 4. Faults and transmissibility multipliers

Faults are implemented as face transmissibility multipliers. The base TPFA transmissibility is still computed from geometry and permeability:

\[
T_f = \frac{A_f}{d_L/K_L+d_R/K_R}.
\]

Step 6B then applies a multiplier:

\[
T_f^{eff}=m_fT_f.
\]

The multiplier may represent:

- sealing faults, `m_f = 0`;
- leaky faults, `0 < m_f < 1`;
- enhanced communication, `m_f > 1`;
- inactive-cell barriers, forced to `m_f = 0`.

### 4.1 FaultPlane3D

A structured fault plane is represented by:

```python
FaultPlane3D(name, axis, index, multiplier, i_range=None, j_range=None, k_range=None)
```

For `axis="x"`, the fault acts across x-normal faces between:

\[
i=\text{index}\quad\text{and}\quad i=\text{index}+1.
\]

For `axis="y"`, it acts between `j=index` and `j=index+1`. For `axis="z"`, it acts between `k=index` and `k=index+1`.

### 4.2 TransmissibilityMultipliers3D

The multiplier model stores three arrays:

```python
tx_multiplier
ty_multiplier
tz_multiplier
```

The model can be applied to base transmissibilities:

```python
txm, tym, tzm = multiplier_model.apply_to(tx, ty, tz)
```

It can also zero faces touching inactive cells:

```python
multiplier_model.apply_active_mask(active_map)
```

This prepares the finite-volume stencil needed by the future 3D black-oil residual.

## 5. Well trajectories and completions

Step 6B introduces completion-level well representation through `wells3d.py`.

The purpose is to move beyond the single-cell well representation used in earlier 1D/2D examples. In a field-scale 3D model, a well generally has a trajectory and several completed grid cells.

### 5.1 WellTrajectory3D

A well trajectory is represented as a polyline:

```python
WellTrajectory3D(name, points)
```

Convenience constructors are provided:

```python
WellTrajectory3D.vertical(name, x, y, z_top, z_bottom)
WellTrajectory3D.horizontal_x(name, x0, x1, y, z)
```

The trajectory is sampled along its length and mapped to the cells it crosses:

```python
cells = trajectory.cells_intersected(grid)
```

### 5.2 Completion3D

Each completion stores:

- completed cell;
- well index;
- open/shut status;
- skin;
- segment ID;
- optional label.

### 5.3 Peaceman well index

Step 6B includes a Cartesian Peaceman-type estimate:

\[
WI = \frac{2\pi h\sqrt{k_1k_2}}{\ln(r_e/r_w)+s}.
\]

Here:

- `h` is the completion length in the well direction;
- `k_1,k_2` are the two permeabilities perpendicular to the well axis;
- `r_w` is wellbore radius;
- `s` is skin;
- `r_e` is an anisotropic equivalent radius.

The function is:

```python
peaceman_well_index_3d(grid, permeability, cell, orientation="z")
```

### 5.4 FieldWell3D

A field-style well is represented by:

```python
FieldWell3D(
    name="PROD_A",
    well_type="producer",
    control="liquid_rate",
    target=-900.0,
    completions=completions,
    min_bhp=120e5,
)
```

Supported control names are aligned with earlier schedule/control work:

- `bhp`;
- `water_rate`;
- `oil_rate`;
- `gas_rate`;
- `liquid_rate`;
- `total_rate`.

The actual 3D well residual will be implemented later, but the data model is now ready.

## 6. Example 12

The new example is:

```bash
python examples/12_3d_property_fault_well_model.py
```

It creates:

- a `32 x 20 x 8` structured 3D reservoir;
- a dipping depth field;
- four vertical geological zones;
- heterogeneous anisotropic permeability;
- porosity from the permeability field;
- inactive-cell boxes;
- one sealing fault;
- one leaky fault;
- vertical injector and producer wells;
- one horizontal producer;
- completion-level well tables;
- a ParaView-compatible VTK file.

The outputs are written to:

```text
outputs/example_12_step6b_3d_reservoir_description/
```

Important files include:

```text
step6b_3d_reservoir_description.vtk
well_completions.csv
step6b_reservoir_description_summary.txt
active_cells_layer.png
zone_ids_layer.png
kx_layer_md.png
fault_indicator_layer.png
well_cells_layer.png
```

## 7. What remains approximate

Step 6B is a reservoir-description stage, not yet the final 3D flow simulator. The following items are still approximate or deferred:

1. The well-index formula is a Cartesian engineering approximation.
2. The trajectory-to-cell algorithm is sampling-based, not a full geometric line-cell intersection algorithm.
3. Faults are structured logical-plane multipliers, not yet arbitrary triangulated or corner-point fault surfaces.
4. Inactive cells are represented through active masks and zeroed transmissibility multipliers; the full reduced active-cell 3D residual will be built later.
5. Corner-point import is still prepared through interfaces, but a GRDECL parser and full corner geometry transmissibility calculation remain future work.

## 8. Why this step matters

The final 3D black-oil simulator will require a clean separation between:

- grid geometry;
- active cells;
- rock properties;
- transmissibility multipliers;
- faults;
- well completions;
- flow unknowns;
- nonlinear residual assembly.

Step 6B establishes that separation. The later 3D solver can now operate on a realistic reservoir description rather than on a purely synthetic uniform mesh.

## 9. Next step

The next planned step is:

**Step 6C — 3D visualization/export: VTK/PyVista/ParaView outputs, time-series state export, well-trajectory visualization, and reservoir-diagnostics plotting.**

That step should build directly on the active-cell, fault, and well objects introduced here.
