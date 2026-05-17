import numpy as np
from blackoil import (
    CartesianGrid2D,
    Rock,
    BlackOilPVTTable,
    TabulatedFluid,
    CoreyThreePhaseRelPerm,
    StateBlackOil,
    BoundaryConditions2D,
    PressureBoundary,
    HeterogeneousBlackOilSimulator5C,
    SparseNewtonSolver,
    md_to_m2,
    bar,
)
from blackoil.flux2d import divergence_from_face_fluxes_2d, three_phase_black_oil_fluxes_2d, boundary_component_fluxes_2d
from blackoil.properties2d import layered_permeability_2d, gaussian_channel_permeability_2d
from blackoil.sparse_jacobian import structured_grid_black_oil_sparsity


def make_pvt():
    p = np.array([1.0e7, 2.0e7, 3.0e7])
    return BlackOilPVTTable(
        p,
        Bo=np.array([1.1, 1.2, 1.3]),
        Bw=np.array([1.02, 1.01, 1.00]),
        Bg=np.array([0.010, 0.007, 0.005]),
        muo_pa_s=np.array([3.0e-3, 2.4e-3, 2.0e-3]),
        muw_pa_s=np.array([5.5e-4, 5.3e-4, 5.1e-4]),
        mug_pa_s=np.array([1.5e-5, 1.8e-5, 2.0e-5]),
        rhoo_kg_m3=np.array([760.0, 730.0, 700.0]),
        rhow_kg_m3=np.array([1010.0, 1015.0, 1020.0]),
        rhog_kg_m3=np.array([90.0, 150.0, 210.0]),
        Rs_sm3_sm3=np.array([60.0, 110.0, 160.0]),
    )


def make_fluids():
    table = make_pvt()
    water = TabulatedFluid("water", table, "Bw", "muw_pa_s", density_key="rhow_kg_m3")
    oil = TabulatedFluid("oil", table, "Bo", "muo_pa_s", density_key="rhoo_kg_m3", rs_key="Rs_sm3_sm3")
    gas = TabulatedFluid("gas", table, "Bg", "mug_pa_s", density_key="rhog_kg_m3")
    return water, oil, gas


def test_cartesian_grid_2d_geometry_and_anisotropy():
    grid = CartesianGrid2D(nx=4, ny=3, lx=40.0, ly=30.0, thickness=5.0)
    assert grid.n_cells == 12
    assert grid.x_face_neighbors.shape == (9, 2)
    assert grid.y_face_neighbors.shape == (8, 2)
    kx = np.full(grid.n_cells, md_to_m2(100.0))
    ky = np.full(grid.n_cells, md_to_m2(25.0))
    tx, ty = grid.geometric_transmissibility({"kx": kx, "ky": ky})
    assert tx.size == 9
    assert ty.size == 8
    assert float(np.mean(tx)) > float(np.mean(ty))


def test_2d_divergence_conserves_interior_fluxes():
    grid = CartesianGrid2D(nx=3, ny=2, lx=30.0, ly=20.0, thickness=1.0)
    fx = np.ones(grid.x_face_neighbors.shape[0])
    fy = 2.0 * np.ones(grid.y_face_neighbors.shape[0])
    div = divergence_from_face_fluxes_2d(grid, fx, fy)
    assert abs(float(np.sum(div))) < 1.0e-14


def test_2d_zero_gradient_gives_zero_interior_flux():
    grid = CartesianGrid2D(nx=3, ny=2, lx=30.0, ly=20.0, thickness=2.0)
    water, oil, gas = make_fluids()
    relperm = CoreyThreePhaseRelPerm()
    n = grid.n_cells
    arrays = three_phase_black_oil_fluxes_2d(
        grid,
        {"kx": md_to_m2(100.0), "ky": md_to_m2(40.0)},
        p=np.full(n, 2.0e7),
        sw=np.full(n, relperm.swc + 0.05),
        sg=np.zeros(n),
        rs=np.full(n, 100.0),
        relperm=relperm,
        water=water,
        oil=oil,
        gas=gas,
        gravity=0.0,
    )
    for a in arrays:
        assert np.allclose(a, 0.0)


def test_pressure_boundary_generates_nonzero_flux():
    grid = CartesianGrid2D(nx=2, ny=2, lx=20.0, ly=20.0, thickness=1.0)
    water, oil, gas = make_fluids()
    relperm = CoreyThreePhaseRelPerm()
    n = grid.n_cells
    bc = BoundaryConditions2D([PressureBoundary("left", pressure=2.2e7, sw=0.3, sg=0.0, rs=100.0)])
    dw, do, dg = boundary_component_fluxes_2d(
        grid,
        {"kx": md_to_m2(100.0), "ky": md_to_m2(50.0)},
        p=np.full(n, 2.0e7),
        sw=np.full(n, 0.25),
        sg=np.zeros(n),
        rs=np.full(n, 100.0),
        relperm=relperm,
        water=water,
        oil=oil,
        gas=gas,
        boundaries=bc,
        gravity=0.0,
    )
    assert np.linalg.norm(dw) > 0.0
    assert np.linalg.norm(do) > 0.0
    assert np.linalg.norm(dg) > 0.0


def test_structured_grid_2d_sparsity_pattern():
    grid = CartesianGrid2D(nx=3, ny=2, lx=30.0, ly=20.0)
    pat = structured_grid_black_oil_sparsity(grid, n_components=3)
    assert pat.shape == (18, 18)
    assert pat.nnz > 3 * 3 * grid.n_cells


def test_heterogeneity_generators_have_correct_shape():
    grid = CartesianGrid2D(nx=5, ny=4, lx=50.0, ly=40.0)
    layers = layered_permeability_2d(grid, [md_to_m2(10.0), md_to_m2(100.0)], direction="y")
    channel = gaussian_channel_permeability_2d(grid, md_to_m2(10.0), md_to_m2(100.0))
    assert layers.shape == (grid.n_cells,)
    assert channel.shape == (grid.n_cells,)
    assert np.max(channel) > np.min(channel)


def test_step5c_single_timestep_runs_on_tiny_grid():
    grid = CartesianGrid2D(nx=2, ny=2, lx=50.0, ly=50.0, thickness=5.0)
    water, oil, gas = make_fluids()
    relperm = CoreyThreePhaseRelPerm(swc=0.18, sor=0.20, sgc=0.02)
    p0 = 2.5e7
    rock = Rock(0.22, {"kx": md_to_m2(80.0), "ky": md_to_m2(30.0)}, compressibility=1.0e-10, p_ref=p0)
    state = StateBlackOil.constant_undersaturated(grid.n_cells, p0, relperm.swc + 0.04, rs=100.0)
    solver = SparseNewtonSolver(
        tol=1.0e-7,
        max_iter=8,
        acceptable_tol=1.0e-3,
        acceptable_min_iterations=1,
        jacobian_strategy="sparse_fd",
        linear_solver="spsolve",
        preconditioner="none",
    )
    sim = HeterogeneousBlackOilSimulator5C(grid, rock, water, oil, gas, relperm, state, solver=solver)
    accepted, report, *_ = sim.try_step(0.1 * 86400.0, max_ds=0.2)
    assert accepted
    assert report.converged
