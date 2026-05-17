from pathlib import Path
import numpy as np

from blackoil.grid3d import CartesianGrid3D
from blackoil.units import md, bar, day
from blackoil.properties3d import anisotropic_permeability_3d, porosity_from_permeability_3d
from blackoil.reservoir3d import ReservoirPropertyModel3D, active_mask_from_boxes, zone_from_layers, map_property_by_zone
from blackoil.faults3d import FaultPlane3D, multipliers_from_faults
from blackoil.wells3d import FieldWell3D, WellTrajectory3D, completions_from_trajectory
from blackoil.pvt import BlackOilPVTTable, TabulatedFluid
from blackoil.relperm import CoreyThreePhaseRelPerm
from blackoil.capillary import LinearCapillaryPressure
from blackoil.rock import Rock
from blackoil.state import StateBlackOil
from blackoil.sparse_solver import SparseNewtonSolver
from blackoil.black_oil_phase import pack_black_oil_primary
from blackoil.black_oil_7 import FullBlackOilSimulator3D, residual_live_oil_7_3d
from blackoil.schedule3d import FieldWellControlEvent3D, FieldWellSchedule3D
from blackoil.flux3d import three_phase_black_oil_fluxes_3d, boundary_component_fluxes_3d
from blackoil.boundary3d import BoundaryConditions3D, PressureBoundary3D


def make_case(nx=3, ny=2, nz=2):
    root = Path(__file__).resolve().parents[1]
    grid = CartesianGrid3D(nx, ny, nz, 150.0, 100.0, 18.0, top_depth=1500.0)
    zones = zone_from_layers(grid, [1, 2])
    kx_md = map_property_by_zone(grid, zones, {1: 100.0, 2: 60.0})
    permeability = anisotropic_permeability_3d(grid, kx_md * md, ky_kx=0.8, kvkh=0.1)
    porosity = porosity_from_permeability_3d(kx_md, phi_min=0.18, phi_max=0.23)
    active_mask = active_mask_from_boxes(grid, [{"i0": 0, "i1": 1, "j0": 0, "j1": 1, "k0": 1, "k1": 2}])
    reservoir = ReservoirPropertyModel3D.from_arrays(grid, porosity=porosity, permeability=permeability, active_mask=active_mask, zone_ids=zones)
    mult = multipliers_from_faults(grid, [FaultPlane3D("F", axis="x", index=1, multiplier=0.5)], active=reservoir.active)
    table = BlackOilPVTTable.from_csv(root / "data" / "pvt" / "live_oil_pvt.csv")
    water = TabulatedFluid("water", table, "Bw", "muw_pa_s", density_key="rhow_kg_m3")
    oil = TabulatedFluid("oil", table, "Bo", "muo_pa_s", density_key="rhoo_kg_m3", rs_key="Rs_sm3_sm3")
    gas = TabulatedFluid("gas", table, "Bg", "mug_pa_s", density_key="rhog_kg_m3")
    relperm = CoreyThreePhaseRelPerm()
    p0 = 255.0 * bar
    rock = Rock(porosity_ref=reservoir.porosity, permeability=reservoir.permeability, compressibility=3.0e-10, p_ref=p0)
    state = StateBlackOil(
        p=np.full(grid.n_cells, p0),
        sw=np.full(grid.n_cells, relperm.swc + 0.04),
        x=np.full(grid.n_cells, float(oil.solution_gas_ratio(p0)) * 0.95),
        is_saturated=np.zeros(grid.n_cells, dtype=bool),
    )
    traj_i = WellTrajectory3D.vertical("I", 20.0, 20.0, 0.0, grid.lz)
    traj_p = WellTrajectory3D.vertical("P", grid.lx - 20.0, grid.ly - 20.0, 0.0, grid.lz)
    wells = [
        FieldWell3D("I", "injector", "water_rate", 1.0 / day, completions_from_trajectory(grid, traj_i, permeability, orientation="z", active=reservoir.active), max_bhp=340.0 * bar),
        FieldWell3D("P", "producer", "liquid_rate", -1.0 / day, completions_from_trajectory(grid, traj_p, permeability, orientation="z", active=reservoir.active), min_bhp=120.0 * bar),
    ]
    return grid, reservoir, mult, rock, water, oil, gas, relperm, state, wells


def test_step7_flux_shapes_and_boundary_fluxes():
    grid, reservoir, mult, rock, water, oil, gas, relperm, state, wells = make_case()
    sw, _so, sg, rs = state.physical(oil)
    fluxes = three_phase_black_oil_fluxes_3d(grid, rock.permeability, state.p, sw, sg, rs, relperm, water, oil, gas, capillary=LinearCapillaryPressure(), transmissibility_multipliers=mult)
    assert len(fluxes) == 12
    assert fluxes[0].shape[0] == len(grid.x_face_neighbors)
    assert fluxes[1].shape[0] == len(grid.y_face_neighbors)
    assert fluxes[2].shape[0] == len(grid.z_face_neighbors)
    bc = BoundaryConditions3D.pressure([PressureBoundary3D("right", 250.0 * bar, sw=0.22, sg=0.0)])
    bw, bo, bg = boundary_component_fluxes_3d(grid, rock.permeability, state.p, sw, sg, rs, relperm, water, oil, gas, boundaries=bc)
    assert bw.shape == bo.shape == bg.shape == (grid.n_cells,)
    assert np.all(np.isfinite(bw + bo + bg))


def test_step7_residual_is_finite_and_full_size():
    grid, reservoir, mult, rock, water, oil, gas, relperm, state, wells = make_case()
    x = pack_black_oil_primary(state.p, state.sw, state.x)
    r = residual_live_oil_7_3d(x, state, 2.0 * day, grid, rock, water, oil, gas, relperm, wells=wells, capillary=LinearCapillaryPressure(), transmissibility_multipliers=mult, active=reservoir.active)
    assert r.shape == (3 * grid.n_cells,)
    assert np.all(np.isfinite(r))


def test_step7_tiny_integrated_run_and_schedule():
    grid, reservoir, mult, rock, water, oil, gas, relperm, state, wells = make_case()
    schedule = FieldWellSchedule3D(
        base_wells=wells,
        events=[
            FieldWellControlEvent3D(0.0, "I", control="water_rate", target=1.0 / day),
            FieldWellControlEvent3D(1.0 * day, "P", control="liquid_rate", target=-1.2 / day),
        ],
        report_times=[0.0, 1.0 * day, 2.0 * day],
    )
    solver = SparseNewtonSolver(tol=1.0e-7, acceptable_tol=1.0e-2, acceptable_min_iterations=2, max_iter=8, jacobian_strategy="sparse_fd", linear_solver="spsolve", preconditioner="none")
    sim = FullBlackOilSimulator3D(grid, rock, water, oil, gas, relperm, state, wells=wells, schedule=schedule, transmissibility_multipliers=mult, active=reservoir.active, capillary=LinearCapillaryPressure(), solver=solver)
    results = sim.run(2.0 * day, dt_initial=1.0 * day, dt_min=0.05 * day, dt_max=1.0 * day, max_ds=0.08)
    assert len(results["reports"]) >= 2
    assert results["pressure"].shape[1] == grid.n_cells
    assert results["reports"][-1].active_cells == reservoir.active.n_active
    assert np.isfinite(results["reports"][-1].recovery_factor)
