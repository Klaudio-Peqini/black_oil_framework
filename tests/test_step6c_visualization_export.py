from pathlib import Path

import numpy as np

from blackoil.grid3d import CartesianGrid3D
from blackoil.units import md
from blackoil.properties3d import anisotropic_permeability_3d
from blackoil.wells3d import FieldWell3D, WellTrajectory3D, completions_from_trajectory
from blackoil.visualization3d import (
    VTKTimeSeriesWriter3D,
    build_completion_cell_fields,
    save_state_time_series_npz,
    state_statistics,
    write_pvd_collection,
    write_vtk_completion_points_3d,
    write_vtk_rectilinear_grid_3d,
    write_vtk_well_trajectories_3d,
)
from blackoil.diagnostics3d import plot_layer_map_3d, plot_state_time_series, plot_pore_volume_by_zone


def test_vtk_time_series_writer_and_pvd(tmp_path: Path):
    grid = CartesianGrid3D(nx=4, ny=3, nz=2, lx=40.0, ly=30.0, lz=10.0)
    writer = VTKTimeSeriesWriter3D(tmp_path, grid, prefix="state")
    p0 = np.full(grid.n_cells, 200.0)
    p1 = np.full(grid.n_cells, 190.0)
    f0 = writer.write_state(0.0, {"pressure_bar": p0, "Sw": 0.2}, step=0)
    f1 = writer.write_state(10.0, {"pressure_bar": p1, "Sw": 0.25}, step=1)
    pvd = writer.write_pvd()
    manifest = writer.write_manifest()
    assert f0.exists() and f1.exists() and pvd.exists() and manifest.exists()
    text = pvd.read_text(encoding="utf-8")
    assert "DataSet" in text and "state_00000.vtk" in text and "state_00001.vtk" in text
    vtk_text = f0.read_text(encoding="utf-8")
    assert "RECTILINEAR_GRID" in vtk_text and "SCALARS pressure_bar" in vtk_text

    explicit = write_pvd_collection(tmp_path / "manual.pvd", [(0.0, f0), (1.0, f1)])
    assert explicit.exists()


def test_well_trajectory_and_completion_vtk_exports(tmp_path: Path):
    grid = CartesianGrid3D(nx=6, ny=5, nz=3, lx=60.0, ly=50.0, lz=15.0)
    kh = np.full(grid.n_cells, 120.0 * md)
    perm = anisotropic_permeability_3d(grid, kh, ky_kx=0.8, kvkh=0.05)
    traj = WellTrajectory3D.vertical("P1", x=25.0, y=25.0, z_top=0.0, z_bottom=15.0)
    comps = completions_from_trajectory(grid, traj, perm, orientation="z")
    well = FieldWell3D("P1", "producer", "bhp", 150e5, comps)

    traj_path = write_vtk_well_trajectories_3d(tmp_path / "trajectories.vtk", [traj])
    comp_path = write_vtk_completion_points_3d(tmp_path / "completions.vtk", grid, [well])
    fields = build_completion_cell_fields(grid, [well])

    assert traj_path.exists() and comp_path.exists()
    assert "POLYDATA" in traj_path.read_text(encoding="utf-8")
    assert "LINES 1" in traj_path.read_text(encoding="utf-8")
    assert "POINT_DATA" in comp_path.read_text(encoding="utf-8")
    assert fields["completion_count"].sum() == len(comps)
    assert fields["completion_wi"].sum() > 0.0


def test_state_statistics_npz_and_diagnostic_plots(tmp_path: Path):
    grid = CartesianGrid3D(nx=5, ny=4, nz=3, lx=50.0, ly=40.0, lz=12.0)
    p = np.linspace(180.0, 220.0, grid.n_cells)
    sw = np.linspace(0.15, 0.65, grid.n_cells)
    pv = np.full(grid.n_cells, 10.0)

    stats = state_statistics(grid, {"pressure_bar": p, "water saturation": sw}, pore_volume=pv)
    assert np.isclose(stats["pressure_bar_mean"], p.mean())
    assert np.isclose(stats["pressure_bar_pvmean"], p.mean())
    assert "water_saturation_max" in stats

    npz = save_state_time_series_npz(tmp_path / "states.npz", [0.0, 1.0], {"pressure": [p, p - 1.0], "Sw": [sw, sw + 0.01]})
    assert npz.exists()
    loaded = np.load(npz)
    assert loaded["pressure"].shape == (2, grid.n_cells)

    vtk = write_vtk_rectilinear_grid_3d(tmp_path / "state.vtk", grid, {"pressure": p})
    assert vtk.exists()

    layer_plot = plot_layer_map_3d(tmp_path / "layer.png", grid, p, title="Pressure")
    hist_plot = plot_state_time_series(tmp_path / "history.png", {"time_days": [0, 1, 2], "pressure": [220, 210, 200]})
    pv_plot = plot_pore_volume_by_zone(tmp_path / "pv_zone.png", np.repeat([1, 2, 3], grid.n_cells // 3 + 1)[: grid.n_cells], pv)
    assert layer_plot.exists() and hist_plot.exists() and pv_plot.exists()
