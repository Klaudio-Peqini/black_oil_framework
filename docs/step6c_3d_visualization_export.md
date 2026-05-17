# Step 6C — 3D Visualization, Export, Time-Series Output and Diagnostics

Step 6C adds the visualization and export layer required before the final full
3D black-oil simulator. The flow equations are not changed in this stage. The
objective is to ensure that once the 3D simulator is assembled, its state fields,
well geometry, completions, diagnostics and restart/report outputs can be
inspected with standard reservoir-engineering and scientific-visualization tools.

## Why this step is necessary

A 3D black-oil model is not useful if its results remain hidden inside arrays.
Pressure, saturations, faults, inactive cells, permeability channels and well
trajectories must be inspected spatially. Step 6C therefore creates the bridge
between numerical output and interpretation.

The implemented export layer supports:

- static reservoir-description VTK files;
- time-series VTK state files;
- ParaView `.pvd` collection files;
- well-trajectory `POLYDATA` export;
- completion-point `POLYDATA` export;
- compact NumPy `.npz` state archives;
- diagnostic CSV files;
- layer maps, histograms and time-history plots;
- optional PyVista conversion for richer notebooks and interactive workflows.

## Main new module

The main export module is:

```text
src/blackoil/visualization3d.py
```

It contains:

```python
write_vtk_rectilinear_grid_3d(...)
write_vtk_structured_points_3d(...)
write_pvd_collection(...)
VTKTimeSeriesWriter3D
save_state_time_series_npz(...)
cell_center_coordinates(...)
build_completion_cell_fields(...)
write_vtk_well_trajectories_3d(...)
write_vtk_completion_points_3d(...)
state_statistics(...)
to_pyvista_rectilinear_grid(...)
```

The dependency-free writer uses legacy ASCII VTK. This is deliberate: it works
on bare HPC login nodes, is easy to inspect, and can be opened directly in
ParaView.

## Diagnostic plotting module

A second module provides publication/report-oriented diagnostics:

```text
src/blackoil/diagnostics3d.py
```

It contains:

```python
plot_layer_map_3d(...)
plot_well_trajectories_3d(...)
plot_state_time_series(...)
plot_property_histograms(...)
plot_pore_volume_by_zone(...)
```

The functions use Matplotlib with a non-interactive backend, which makes them
suitable for batch jobs and remote servers.

## VTK state export

A typical state export looks like this:

```python
from blackoil.visualization3d import VTKTimeSeriesWriter3D

writer = VTKTimeSeriesWriter3D("outputs/vtk", grid, prefix="blackoil_state")

writer.write_state(
    time=0.0,
    cell_data={
        "pressure_bar": pressure / 1.0e5,
        "water_saturation": Sw,
        "gas_saturation": Sg,
        "oil_saturation": So,
        "kx_mD": Kx_mD,
        "active": active.astype(float),
    },
)

writer.write_pvd("blackoil_state_series.pvd")
```

ParaView can open the `.pvd` file as a time-dependent dataset.

## Well trajectories and completions

Well geometry is exported separately from reservoir cell data:

```python
write_vtk_well_trajectories_3d("well_trajectories.vtk", trajectories)
write_vtk_completion_points_3d("well_completions_points.vtk", grid, wells)
```

This keeps cell-centred reservoir fields and well geometry cleanly separated.
In ParaView, both files can be loaded and overlaid.

## Optional PyVista adapter

The function

```python
to_pyvista_rectilinear_grid(grid, cell_data)
```

returns a PyVista `RectilinearGrid` when PyVista is installed. PyVista is not a
mandatory dependency. It can be installed using:

```bash
pip install -e ".[visualization]"
```

The optional adapter is useful for notebooks, screenshots, clipping, thresholding
and VTK XML export.

## Example

The new example is:

```bash
python examples/13_3d_visualization_export_diagnostics.py
```

It writes files to:

```text
outputs/example_13_step6c_3d_visualization/
```

The most important files are:

```text
reservoir_static_description.vtk
well_trajectories.vtk
well_completions_points.vtk
vtk_timeseries/blackoil_state_series.pvd
vtk_timeseries/blackoil_state_00000.vtk
vtk_timeseries/blackoil_state_00001.vtk
...
blackoil_state_timeseries.npz
state_diagnostics.csv
final_pressure_layer.png
final_water_saturation_layer.png
final_gas_saturation_layer.png
static_kx_layer.png
well_trajectories_projection.png
property_histograms.png
pore_volume_by_zone.png
state_history_diagnostics.png
```

## How to inspect in ParaView

Open:

```text
vtk_timeseries/blackoil_state_series.pvd
```

Then optionally also open:

```text
well_trajectories.vtk
well_completions_points.vtk
reservoir_static_description.vtk
```

Useful ParaView operations include:

- threshold by `active > 0.5`;
- color by `pressure_bar`, `water_saturation`, `gas_saturation`, `kx_mD` or
  `zone_id`;
- use `Slice` or `Clip` to inspect internal layers;
- overlay `well_trajectories.vtk` with thicker lines;
- overlay `well_completions_points.vtk` as glyphs or spheres;
- animate through the `.pvd` time steps.

## Current approximation

Step 6C uses synthetic 3D pressure and saturation fields in the example. This is
intentional. The purpose of the step is to prove that a future 3D solver can
export states and diagnostics immediately. The final physical solver will replace
these synthetic arrays with computed black-oil states.

## Next step

The next and final major milestone is Step 7:

> Integrated full 3D black-oil framework.

Step 7 should combine the already-developed components:

- conservative black-oil residual;
- phase-state switching;
- gravity and capillary pressure;
- sparse Newton/Krylov solvers;
- schedules and restart files;
- 3D grids, faults, active cells and completions;
- Step 6C visualization/export.
