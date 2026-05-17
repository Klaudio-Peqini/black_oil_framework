from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from blackoil.grid3d import CartesianGrid3D
from blackoil.flux3d import divergence_from_face_fluxes_3d, total_transmissibility_count_3d
from blackoil.properties3d import (
    anisotropic_permeability_3d,
    gaussian_channel_permeability_3d,
    layered_permeability_3d,
    lognormal_permeability_3d,
    porosity_from_permeability_3d,
)
from blackoil.visualization3d import write_vtk_rectilinear_grid_3d
from blackoil.cornerpoint import CornerPointGridSpec, make_cartesian_cornerpoint_spec
from blackoil.sparse_jacobian import structured_grid_black_oil_sparsity


def test_cartesian_grid3d_indexing_neighbors_and_volumes():
    grid = CartesianGrid3D(nx=3, ny=2, nz=2, lx=30.0, ly=20.0, lz=10.0, top_depth=1000.0)
    assert grid.n_cells == 12
    assert grid.cell_index(2, 1, 1) == 11
    assert grid.unravel_cell(11) == (2, 1, 1)
    assert np.allclose(grid.volumes, 500.0)
    assert len(grid.x_face_neighbors) == (3 - 1) * 2 * 2
    assert len(grid.y_face_neighbors) == 3 * (2 - 1) * 2
    assert len(grid.z_face_neighbors) == 3 * 2 * (2 - 1)
    assert total_transmissibility_count_3d(grid) == len(grid.neighbors)
    assert np.isclose(grid.depths.min(), 1002.5)
    assert np.isclose(grid.depths.max(), 1007.5)


def test_grid3d_transmissibility_for_homogeneous_cube():
    grid = CartesianGrid3D(nx=2, ny=2, nz=2, lx=2.0, ly=2.0, lz=2.0)
    tx, ty, tz = grid.geometric_transmissibility(10.0)
    assert tx.shape == (4,)
    assert ty.shape == (4,)
    assert tz.shape == (4,)
    assert np.allclose(tx, 10.0)
    assert np.allclose(ty, 10.0)
    assert np.allclose(tz, 10.0)


def test_grid3d_anisotropic_permeability_and_properties():
    grid = CartesianGrid3D(nx=4, ny=3, nz=2, lx=40.0, ly=30.0, lz=8.0)
    layered = layered_permeability_3d(grid, [100.0, 10.0], direction="z")
    assert layered.shape == (grid.n_cells,)
    assert np.isclose(layered[: grid.nx * grid.ny].mean(), 100.0)
    logk = lognormal_permeability_3d(grid, 50.0, sigma_log=0.2, seed=123)
    channel = gaussian_channel_permeability_3d(grid, 20.0, 200.0)
    kdict = anisotropic_permeability_3d(grid, logk, kvkh=0.05, ky_kx=0.8)
    assert set(kdict) == {"kx", "ky", "kz"}
    assert np.all(kdict["kz"] < kdict["kx"])
    phi = porosity_from_permeability_3d(channel)
    pv = grid.pore_volume(phi)
    assert np.all((phi > 0.0) & (phi < 1.0))
    assert np.isclose(pv.sum(), np.sum(phi * grid.volumes))


def test_divergence_from_face_fluxes_3d_conserves_internal_flux():
    grid = CartesianGrid3D(nx=3, ny=2, nz=2, lx=3.0, ly=2.0, lz=2.0)
    fx = np.ones(len(grid.x_face_neighbors))
    fy = 2.0 * np.ones(len(grid.y_face_neighbors))
    fz = -0.5 * np.ones(len(grid.z_face_neighbors))
    div = divergence_from_face_fluxes_3d(grid, fx, fy, fz)
    assert div.shape == (grid.n_cells,)
    assert abs(div.sum()) < 1.0e-12


def test_structured_sparsity_accepts_3d_grid():
    grid = CartesianGrid3D(nx=3, ny=2, nz=2, lx=3.0, ly=2.0, lz=2.0)
    pattern = structured_grid_black_oil_sparsity(grid, n_components=3)
    assert isinstance(pattern, csr_matrix)
    assert pattern.shape == (3 * grid.n_cells, 3 * grid.n_cells)
    assert pattern.nnz > 3 * grid.n_cells


def test_vtk_3d_writer_and_cornerpoint_spec(tmp_path: Path):
    grid = CartesianGrid3D(nx=2, ny=2, nz=2, lx=10.0, ly=10.0, lz=4.0, top_depth=900.0)
    pressure = np.linspace(200.0, 180.0, grid.n_cells)
    out = write_vtk_rectilinear_grid_3d(tmp_path / "state.vtk", grid, {"pressure": pressure})
    text = out.read_text(encoding="utf-8")
    assert "DATASET RECTILINEAR_GRID" in text
    assert "DIMENSIONS 3 3 3" in text
    assert "CELL_DATA 8" in text

    spec = make_cartesian_cornerpoint_spec(grid)
    assert isinstance(spec, CornerPointGridSpec)
    assert spec.coord.shape == ((grid.nx + 1) * (grid.ny + 1), 6)
    assert spec.zcorn.size == 8 * grid.n_cells
    assert spec.active_mask.all()
