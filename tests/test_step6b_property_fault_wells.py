from pathlib import Path
import csv

import numpy as np

from blackoil.grid3d import CartesianGrid3D
from blackoil.units import md
from blackoil.reservoir3d import (
    ActiveCellMap3D,
    ReservoirPropertyModel3D,
    active_mask_from_boxes,
    active_mask_from_indices,
    apply_region_multiplier,
    map_property_by_zone,
    zone_from_depth_intervals,
    zone_from_layers,
)
from blackoil.faults3d import FaultPlane3D, TransmissibilityMultipliers3D, multipliers_from_faults
from blackoil.properties3d import anisotropic_permeability_3d, porosity_from_permeability_3d
from blackoil.wells3d import (
    Completion3D,
    FieldWell3D,
    WellTrajectory3D,
    completions_from_trajectory,
    peaceman_well_index_3d,
)


def test_active_cell_map_compress_expand_and_boxes():
    grid = CartesianGrid3D(nx=5, ny=4, nz=3, lx=50, ly=40, lz=15)
    mask = active_mask_from_boxes(grid, [{"i0": 1, "i1": 3, "j0": 1, "j1": 3, "k0": 1, "k1": 2}])
    active = ActiveCellMap3D(grid, mask)
    assert active.n_active == grid.n_cells - 4
    field = np.arange(grid.n_cells, dtype=float)
    compressed = active.compress(field)
    expanded = active.expand(compressed, fill_value=-1.0)
    assert compressed.size == active.n_active
    assert np.all(expanded[mask] == field[mask])
    assert np.all(expanded[~mask] == -1.0)

    mask2 = active_mask_from_indices(grid, [0, grid.n_cells - 1])
    assert mask2.sum() == grid.n_cells - 2


def test_zone_property_mapping_and_reservoir_property_model():
    grid = CartesianGrid3D(nx=4, ny=3, nz=3, lx=40, ly=30, lz=12, top_depth=1000)
    zones = zone_from_layers(grid, [10, 20, 30])
    kx_md = map_property_by_zone(grid, zones, {10: 300.0, 20: 80.0, 30: 30.0}, name="kx")
    assert np.isclose(kx_md[: grid.nx * grid.ny].mean(), 300.0)

    depth_zones = zone_from_depth_intervals(grid, [(1000, 1004, 1), (1004, 1008, 2), (1008, 2000, 3)])
    assert set(np.unique(depth_zones)) == {1, 2, 3}

    phi = porosity_from_permeability_3d(kx_md, phi_min=0.1, phi_max=0.25)
    perm = anisotropic_permeability_3d(grid, kx_md * md, ky_kx=0.7, kvkh=0.05)
    mask = active_mask_from_boxes(grid, [{"i0": 0, "i1": 1, "j0": 0, "j1": 1, "k0": 0, "k1": 3}])
    model = ReservoirPropertyModel3D.from_arrays(grid, porosity=phi, permeability=perm, active_mask=mask, zone_ids=zones)
    summary = model.summary()
    assert summary["n_active"] == grid.n_cells - 3
    assert model.pore_volume[~mask].sum() == 0.0
    assert model.active_property("kx").size == summary["n_active"]

    modified = apply_region_multiplier(grid, kx_md, 0.1, box={"i0": 0, "i1": 2, "j0": 0, "j1": 1, "k0": 2, "k1": 3})
    assert modified.min() < kx_md.min()


def test_fault_plane_and_transmissibility_multipliers():
    grid = CartesianGrid3D(nx=5, ny=4, nz=3, lx=50, ly=40, lz=15)
    tx, ty, tz = grid.geometric_transmissibility(100.0 * md)
    fault = FaultPlane3D(name="F1", axis="x", index=2, multiplier=0.0, j_range=(1, 3), k_range=(0, 2))
    faces = fault.affected_face_indices(grid)
    assert faces.size == 4

    inactive = active_mask_from_boxes(grid, [{"i0": 4, "i1": 5, "j0": 0, "j1": 4, "k0": 0, "k1": 3}])
    active = ActiveCellMap3D(grid, inactive)
    mult = multipliers_from_faults(grid, [fault], active=active)
    txm, tym, tzm = mult.apply_to(tx, ty, tz)
    assert np.count_nonzero(txm == 0.0) >= faces.size
    assert mult.to_cell_indicator().sum() > 0
    counts = mult.nonzero_counts()
    assert counts["x"] < tx.size


def test_well_trajectory_completions_and_field_well():
    grid = CartesianGrid3D(nx=8, ny=6, nz=4, lx=80, ly=60, lz=20)
    kh = np.full(grid.n_cells, 150.0 * md)
    perm = anisotropic_permeability_3d(grid, kh, ky_kx=0.8, kvkh=0.05)

    vertical = WellTrajectory3D.vertical("P1", x=45.0, y=25.0, z_top=0.0, z_bottom=20.0)
    cells = vertical.cells_intersected(grid)
    assert cells.size == grid.nz
    wi = peaceman_well_index_3d(grid, perm, int(cells[0]), well_radius=0.1, orientation="z")
    assert wi > 0.0

    completions = completions_from_trajectory(grid, vertical, perm, well_radius=0.1, orientation="z")
    assert len(completions) == grid.nz
    well = FieldWell3D("P1", "producer", "liquid_rate", -500.0, completions, min_bhp=120e5)
    assert well.is_open
    assert well.total_well_index > 0.0
    table = well.completion_table(grid)
    assert table[0]["well"] == "P1"
    assert {row["k"] for row in table} == set(range(grid.nz))

    horizontal = WellTrajectory3D.horizontal_x("I1", x0=5.0, x1=75.0, y=35.0, z=7.5)
    hcells = horizontal.cells_intersected(grid)
    assert hcells.size >= 4
    hcomp = completions_from_trajectory(grid, horizontal, perm, orientation="x")
    assert len(hcomp) == hcells.size
