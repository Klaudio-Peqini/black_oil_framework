"""Black-oil finite-volume reservoir-simulation framework."""

from .units import day, year, cp, md_to_m2, bar, psi
from .grid import CartesianGrid1D
from .grid2d import CartesianGrid2D
from .grid3d import CartesianGrid3D
from .rock import Rock
from .pvt import SlightlyCompressibleFluid, BlackOilPVTTable, TabulatedFluid
from .relperm import CoreyWaterOilRelPerm, CoreyThreePhaseRelPerm
from .wells import RateWell, BHPWell, MultiRateWell, ControlledWell
from .state import State1P, State2P, State3P, StateBlackOil
from .simulator import SinglePhaseSimulator, TwoPhaseOilWaterSimulator
from .nonlinear_solver import NewtonSolver, NewtonReport
from .ad_solver import NewtonSolverWithJacobian
from .dead_oil import DeadOilSimulator, DeadOilStepInfo
from .live_oil import LiveOilSaturatedSimulator, LiveOilStepInfo
from .black_oil_phase import LiveOilPhaseSwitchingSimulator, BlackOilStepInfo
from .black_oil_ad import ConservativeLiveOilPhaseSwitchingSimulator, residual_live_oil_phase_switching_conservative
from .capillary import ZeroCapillaryPressure, LinearCapillaryPressure, BrooksCoreyCapillaryPressure
from .black_oil_5a import AdvancedBlackOilSimulator5A, residual_live_oil_5a_gravity_capillary
from .sparse_jacobian import block_tridiagonal_black_oil_sparsity, sparse_finite_difference_jacobian
from .sparse_solver import SparseNewtonSolver, SparseNewtonReport
from .black_oil_5b import ScalableBlackOilSimulator5B
from .boundary import BoundaryConditions2D, PressureBoundary
from .boundary3d import BoundaryConditions3D, PressureBoundary3D
from .properties2d import layered_permeability_2d, gaussian_channel_permeability_2d, lognormal_permeability_2d
from .visualization2d import write_vtk_structured_grid_2d
from .flux3d import divergence_from_face_fluxes_3d, total_transmissibility_count_3d, three_phase_black_oil_fluxes_3d, boundary_component_fluxes_3d
from .properties3d import layered_permeability_3d, gaussian_channel_permeability_3d, lognormal_permeability_3d, anisotropic_permeability_3d, porosity_from_permeability_3d
from .visualization3d import (
    VTKTimeSeriesWriter3D,
    build_completion_cell_fields,
    cell_center_coordinates,
    save_state_time_series_npz,
    state_statistics,
    to_pyvista_rectilinear_grid,
    write_pvd_collection,
    write_vtk_completion_points_3d,
    write_vtk_rectilinear_grid_3d,
    write_vtk_structured_points_3d,
    write_vtk_well_trajectories_3d,
)
from .diagnostics3d import (
    plot_layer_map_3d,
    plot_pore_volume_by_zone,
    plot_property_histograms,
    plot_state_time_series,
    plot_well_trajectories_3d,
)
from .cornerpoint import CornerPointGridSpec, make_cartesian_cornerpoint_spec
from .black_oil_5c import HeterogeneousBlackOilSimulator5C, residual_live_oil_5c_2d
from .schedule import WellControlEvent, ScheduledWellState, WellSchedule
from .restart import save_black_oil_restart, load_black_oil_restart, apply_black_oil_restart
from .black_oil_5d import ScheduledBlackOilSimulator5D
from .schedule3d import FieldWellControlEvent3D, FieldWellSchedule3D
from .black_oil_7 import FullBlackOilSimulator3D, BlackOil3DStepInfo, residual_live_oil_7_3d

from .reservoir3d import (
    ActiveCellMap3D,
    ReservoirPropertyModel3D,
    active_mask_from_boxes,
    active_mask_from_indices,
    apply_region_multiplier,
    as_cell_array_3d,
    map_property_by_zone,
    zone_from_depth_intervals,
    zone_from_layers,
)
from .faults3d import FaultPlane3D, TransmissibilityMultipliers3D, multipliers_from_faults
from .wells3d import (
    Completion3D,
    FieldWell3D,
    WellTrajectory3D,
    completions_from_trajectory,
    peaceman_well_index_3d,
)

__all__ = [
    "day",
    "year",
    "cp",
    "md_to_m2",
    "bar",
    "psi",
    "CartesianGrid1D",
    "CartesianGrid2D",
    "CartesianGrid3D",
    "Rock",
    "SlightlyCompressibleFluid",
    "BlackOilPVTTable",
    "TabulatedFluid",
    "CoreyWaterOilRelPerm",
    "CoreyThreePhaseRelPerm",
    "RateWell",
    "BHPWell",
    "MultiRateWell",
    "ControlledWell",
    "State1P",
    "State2P",
    "State3P",
    "StateBlackOil",
    "SinglePhaseSimulator",
    "TwoPhaseOilWaterSimulator",
    "NewtonSolver",
    "NewtonReport",
    "NewtonSolverWithJacobian",
    "DeadOilSimulator",
    "DeadOilStepInfo",
    "LiveOilSaturatedSimulator",
    "LiveOilStepInfo",
    "LiveOilPhaseSwitchingSimulator",
    "BlackOilStepInfo",
    "ConservativeLiveOilPhaseSwitchingSimulator",
    "residual_live_oil_phase_switching_conservative",
    "ZeroCapillaryPressure",
    "LinearCapillaryPressure",
    "BrooksCoreyCapillaryPressure",
    "AdvancedBlackOilSimulator5A",
    "residual_live_oil_5a_gravity_capillary",
    "block_tridiagonal_black_oil_sparsity",
    "sparse_finite_difference_jacobian",
    "SparseNewtonSolver",
    "SparseNewtonReport",
    "ScalableBlackOilSimulator5B",
    "BoundaryConditions2D",
    "PressureBoundary",
    "layered_permeability_2d",
    "gaussian_channel_permeability_2d",
    "lognormal_permeability_2d",
    "write_vtk_structured_grid_2d",
    "divergence_from_face_fluxes_3d",
    "total_transmissibility_count_3d",
    "layered_permeability_3d",
    "gaussian_channel_permeability_3d",
    "lognormal_permeability_3d",
    "anisotropic_permeability_3d",
    "porosity_from_permeability_3d",
    "VTKTimeSeriesWriter3D",
    "build_completion_cell_fields",
    "cell_center_coordinates",
    "save_state_time_series_npz",
    "state_statistics",
    "to_pyvista_rectilinear_grid",
    "write_pvd_collection",
    "write_vtk_completion_points_3d",
    "write_vtk_rectilinear_grid_3d",
    "write_vtk_structured_points_3d",
    "write_vtk_well_trajectories_3d",
    "plot_layer_map_3d",
    "plot_pore_volume_by_zone",
    "plot_property_histograms",
    "plot_state_time_series",
    "plot_well_trajectories_3d",
    "CornerPointGridSpec",
    "make_cartesian_cornerpoint_spec",
    "HeterogeneousBlackOilSimulator5C",
    "residual_live_oil_5c_2d",
    "WellControlEvent",
    "ScheduledWellState",
    "WellSchedule",
    "save_black_oil_restart",
    "load_black_oil_restart",
    "apply_black_oil_restart",
    "ScheduledBlackOilSimulator5D",
    "ActiveCellMap3D",
    "ReservoirPropertyModel3D",
    "active_mask_from_boxes",
    "active_mask_from_indices",
    "apply_region_multiplier",
    "as_cell_array_3d",
    "map_property_by_zone",
    "zone_from_depth_intervals",
    "zone_from_layers",
    "FaultPlane3D",
    "TransmissibilityMultipliers3D",
    "multipliers_from_faults",
    "Completion3D",
    "FieldWell3D",
    "WellTrajectory3D",
    "completions_from_trajectory",
    "peaceman_well_index_3d",
    "BoundaryConditions3D",
    "PressureBoundary3D",
    "three_phase_black_oil_fluxes_3d",
    "boundary_component_fluxes_3d",
    "FieldWellControlEvent3D",
    "FieldWellSchedule3D",
    "FullBlackOilSimulator3D",
    "BlackOil3DStepInfo",
    "residual_live_oil_7_3d",
]
