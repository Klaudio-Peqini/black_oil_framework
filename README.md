 # A Fully Implicit Finite-Volume Framework for Multiphase Black-Oil Flow in Porous Reservoirs                                           
 
This repository is a staged research-and-development framework for black-oil reservoir simulation. It was built progressively from a minimal finite-volume prototype into an integrated 3D live-oil black-oil simulator with phase switching, solution gas, gravity, capillary pressure, sparse nonlinear solvers, schedules, restarts, 3D reservoir-description tools, and ParaView-compatible visualization.

The project is intentionally written as a clear scientific framework rather than as a closed commercial-style simulator. The goal is to expose the mathematical model, the numerical method, the code architecture, and the validation path in a way that can support future research, teaching, publication, and extension toward high-performance reservoir simulation.

---

## 1. What this framework currently contains

The current repository reaches **Step 7: integrated full 3D black-oil framework**.

It contains:

- fully implicit finite-volume formulations;
- single-phase pressure diffusion;
- two-phase water-oil displacement;
- compressible dead-oil model;
- saturated live-oil model;
- bubble-point / phase-state switching;
- conservative live-oil gas-component equation;
- sparse Newton infrastructure;
- 2D heterogeneous anisotropic reservoirs;
- field schedules and restart files;
- structured Cartesian 3D mesh infrastructure;
- 3D property mapping, inactive cells, faults, transmissibility multipliers;
- 3D well trajectories and completions;
- VTK/PVD visualization/export;
- an integrated full 3D black-oil simulator.

The principal final-stage simulator is:

```text
src/blackoil/black_oil_7.py
```

with:

```python
FullBlackOilSimulator3D
residual_live_oil_7_3d
BlackOil3DStepInfo
```

The main integrated example is:

```bash
python examples/14_full_3d_black_oil_integrated.py
```

---

## 2. Installation

The package uses SI units internally. The required Python dependencies are deliberately standard:

```text
numpy
scipy
matplotlib
pandas
pytest
```

Create an environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e .
pip install -r requirements.txt
```

Optional dependencies:

```bash
pip install -e ".[ad]"             # optional JAX support for earlier AD experiments
pip install -e ".[visualization]"  # optional PyVista support
```

Run the tests:

```bash
pytest -q
```

Run the stable examples:

```bash
python examples/run_all.py
```

Run all examples, including the experimental/integrated stages:

```bash
python examples/run_all.py --include-experimental
```

Run only the final integrated 3D model:

```bash
python examples/14_full_3d_black_oil_integrated.py
```

---

## 3. Repository structure

```text
black_oil_framework/
│
├── data/
│   └── pvt/
│       ├── dead_oil_pvt.csv
│       └── live_oil_pvt.csv
│
├── docs/
│   ├── mathematical_model.md
│   ├── numerical_method.md
│   ├── development_roadmap.md
│   ├── dead_oil_stage.md
│   ├── live_oil_saturated_stage.md
│   ├── live_oil_phase_switching_stage.md
│   ├── live_oil_conservative_ad_stage.md
│   ├── step5a_gravity_capillary_well_controls.md
│   ├── step5b_sparse_newton_krylov.md
│   ├── step5c_2d_heterogeneous_validation.md
│   ├── step5d_schedules_restarts_field_controls.md
│   ├── step6a_3d_mesh_cornerpoint_preparation.md
│   ├── step6b_3d_property_mapping_faults_wells.md
│   ├── step6c_3d_visualization_export.md
│   └── step7_integrated_full_3d_black_oil.md
│
├── examples/
│   ├── 01_single_phase_pressure_diffusion.py
│   ├── 02_two_phase_water_oil_fully_implicit.py
│   ├── 03_dead_oil_compressible.py
│   ├── 04_live_oil_saturated_black_oil.py
│   ├── 05_live_oil_phase_switching_black_oil.py
│   ├── 06_live_oil_conservative_ad_black_oil.py
│   ├── 07_live_oil_gravity_capillary_controls.py
│   ├── 08_live_oil_sparse_newton_krylov.py
│   ├── 09_2d_heterogeneous_black_oil.py
│   ├── 10_scheduled_field_controls_restart.py
│   ├── 11_3d_mesh_property_visualization.py
│   ├── 12_3d_property_fault_well_model.py
│   ├── 13_3d_visualization_export_diagnostics.py
│   ├── 14_full_3d_black_oil_integrated.py
│   └── run_all.py
│
├── src/blackoil/
│   ├── grid.py
│   ├── grid2d.py
│   ├── grid3d.py
│   ├── rock.py
│   ├── pvt.py
│   ├── relperm.py
│   ├── capillary.py
│   ├── state.py
│   ├── flux.py
│   ├── flux2d.py
│   ├── flux3d.py
│   ├── wells.py
│   ├── wells3d.py
│   ├── residual.py
│   ├── nonlinear_solver.py
│   ├── sparse_jacobian.py
│   ├── sparse_solver.py
│   ├── black_oil_5a.py
│   ├── black_oil_5b.py
│   ├── black_oil_5c.py
│   ├── black_oil_5d.py
│   ├── black_oil_7.py
│   ├── reservoir3d.py
│   ├── faults3d.py
│   ├── schedule.py
│   ├── schedule3d.py
│   ├── restart.py
│   ├── visualization2d.py
│   ├── visualization3d.py
│   └── diagnostics3d.py
│
└── tests/
    ├── test_*.py
    └── test_step7_full_3d_black_oil.py
```

---

## 4. Development history and staged logic

The framework was developed step by step. This is important because a black-oil simulator becomes fragile if all physics are introduced at once.

### Step 1 — Single-phase pressure diffusion

The first stage validates:

- Cartesian finite-volume grid;
- transmissibility calculation;
- accumulation term;
- pressure solve;
- simple source/sink terms.

Main example:

```bash
python examples/01_single_phase_pressure_diffusion.py
```

### Step 2 — Fully implicit water-oil flow

The second stage introduces two-phase flow with pressure and water saturation. It validates:

- nonlinear saturation transport;
- upwind mobility;
- relative permeability;
- fully implicit Newton solve;
- recovery factor.

Main example:

```bash
python examples/02_two_phase_water_oil_fully_implicit.py
```

### Step 3 — Compressible dead-oil model

This stage adds pressure-dependent PVT properties:

- $B_w(p)$;
- $B_o(p)$;
- $\mu_w(p)$;
- $\mu_o(p)$;
- rock compressibility;
- material-balance diagnostics.

Main example:

```bash
python examples/03_dead_oil_compressible.py
```

### Step 4A — Saturated live-oil black-oil model

This stage introduces the gas component and solution gas:

$$
\phi\left(R_s\frac{S_o}{B_o}+\frac{S_g}{B_g}\right).
$$

Main example:

```bash
python examples/04_live_oil_saturated_black_oil.py
```

### Step 4B — Bubble-point / phase-state switching

Cells can switch between:

```text
undersaturated: Sg = 0, primary variable is Rs
saturated:      Sg > 0, primary variable is Sg
```

Main example:

```bash
python examples/05_live_oil_phase_switching_black_oil.py
```

### Step 4C — Fully conservative switching and AD/sparse-preparation

The gas-component conservation equation is active in both saturated and undersaturated cells. The code also introduces the infrastructure needed for automatic or sparse Jacobian development.

Main example:

```bash
python examples/06_live_oil_conservative_ad_black_oil.py
```

### Step 5A — Gravity, capillary pressure and well controls

This stage adds:

- phase pressures;
- water-oil capillary pressure;
- gas-oil capillary pressure;
- gravity potential;
- water-rate, oil-rate, gas-rate, liquid-rate, total-rate and BHP controls;
- simple rate/BHP switching.

Main example:

```bash
python examples/07_live_oil_gravity_capillary_controls.py
```

### Step 5B — Sparse Newton and Krylov solvers

This stage adds:

- sparse finite-difference Jacobian;
- structured finite-volume sparsity pattern;
- GMRES, BiCGSTAB, LGMRES and sparse direct solves;
- Jacobi and ILU preconditioning;
- matrix-free Newton-Krylov prototype.

Main example:

```bash
python examples/08_live_oil_sparse_newton_krylov.py
```

### Step 5C — 2D heterogeneous reservoirs

This stage generalizes the model to 2D structured grids with:

- $K_x$, $K_y$ anisotropy;
- heterogeneous permeability;
- no-flow and pressure boundaries;
- 2D validation cases;
- VTK export.

Main example:

```bash
python examples/09_2d_heterogeneous_black_oil.py
```

### Step 5D — Schedules and restarts

This stage introduces field-style operational realism:

- well opening and shutting;
- scheduled control changes;
- timestep landing on schedule/report/restart times;
- restart save/load.

Main example:

```bash
python examples/10_scheduled_field_controls_restart.py
```

### Step 6A — 3D mesh module

This stage adds:

- structured Cartesian 3D grid;
- 3D cell indexing;
- x/y/z internal faces;
- 3D transmissibilities;
- preliminary corner-point-grid interface.

Main example:

```bash
python examples/11_3d_mesh_property_visualization.py
```

### Step 6B — 3D reservoir description

This stage adds:

- active/inactive cells;
- property mapping by zones/layers/depth intervals;
- transmissibility multipliers;
- faults;
- well trajectories;
- 3D completions and Peaceman well-index estimates.

Main example:

```bash
python examples/12_3d_property_fault_well_model.py
```

### Step 6C — 3D visualization and export

This stage adds:

- VTK rectilinear-grid export;
- PVD time-series export;
- well-trajectory VTK output;
- completion-point VTK output;
- state-history diagnostics;
- optional PyVista conversion.

Main example:

```bash
python examples/13_3d_visualization_export_diagnostics.py
```

### Step 7 — Integrated full 3D black-oil framework

The final stage combines all previous pieces into one integrated simulator:

```bash
python examples/14_full_3d_black_oil_integrated.py
```

---

## 5. Mathematical model

The black-oil model describes water, oil and gas phases. The main phase saturations satisfy:

$$
S_w + S_o + S_g = 1.
$$

The primary pressure is oil pressure $p_o$. The phase pressures are:

$$
p_w = p_o - p_{cow}(S_w),
$$

$$
p_g = p_o + p_{cgo}(S_g).
$$

The phase Darcy velocity is:

$$
\mathbf{u}_\alpha = -\mathbf{K}\frac{k_{r\alpha}}{\mu_\alpha}\left(\nabla p_\alpha - \rho_\alpha g \nabla D\right),
$$

where $D$ is positive-downward depth.

The component equations are:

### Water

$$
\frac{\partial}{\partial t}\left(\phi\frac{S_w}{B_w}\right) + \nabla\cdot\left(\frac{\mathbf{u}_w}{B_w}\right) = q_w.
$$

### Oil

$$
\frac{\partial}{\partial t}\left(\phi\frac{S_o}{B_o}\right) + \nabla\cdot\left(\frac{\mathbf{u}_o}{B_o}\right) = q_o.
$$

### Gas component

$$
\frac{\partial}{\partial t}\left[ \phi\left(R_s\frac{S_o}{B_o}+\frac{S_g}{B_g}\right) \right] + \nabla\cdot\left[ R_s\frac{\mathbf{u}_o}{B_o}+\frac{\mathbf{u}_g}{B_g} \right] = q_g.
$$

The gas equation is conservative with respect to both dissolved gas and free gas.

---

## 6. Primary-variable switching

The Step 7 simulator uses one pressure variable, one water saturation variable and one phase-state-dependent third variable per cell:

$$
(p_o, S_w, x).
$$

For undersaturated oil cells:

$$
S_g = 0, \qquad x=R_s.
$$

For saturated cells:

$$
S_g > 0, \qquad x=S_g, \qquad R_s = R_s^{sat}(p_o).
$$

The phase map is held fixed during one Newton solve and updated after convergence. This design is robust for a research framework and avoids introducing a full nonlinear complementarity formulation too early.

---

## 7. Numerical method

The numerical method is:

```text
finite volume in space + fully implicit time integration + Newton-Raphson nonlinear solve
```

For each component and cell:

$$
R_i^{n+1} = \frac{A_i^{n+1}-A_i^n}{\Delta t} + \sum_j F_{ij}^{n+1} - Q_i^{n+1}.
$$

The nonlinear system is:

$$
\mathbf{R}(\mathbf{x}^{n+1}) = 0.
$$

The Newton correction solves:

$$
\mathbf{J}\Delta\mathbf{x} = -\mathbf{R}.
$$

The Step 7 implementation uses sparse finite-difference Jacobians with finite-volume stencil coloring. This is not as efficient as analytic assembly, but it exposes the correct structure and prepares the code for analytic or automatic-differentiated sparse assembly later.

---

## 8. 3D grid and permeability

The 3D grid is structured Cartesian. The flattened index is:

$$
cell(i,j,k)=kN_xN_y+jN_x+i.
$$

The permeability field is anisotropic:

```python
permeability = {
    "kx": Kx,
    "ky": Ky,
    "kz": Kz,
}
```

Each component can be a scalar, a flattened array of length `n_cells`, or an array of shape `(nz, ny, nx)`.

The code supports:

- layered properties;
- zone-based property mapping;
- depth-interval property mapping;
- Gaussian channel fields;
- lognormal fields;
- porosity derived from permeability;
- inactive-cell maps.

---

## 9. Faults and transmissibility multipliers

Faults are represented by face transmissibility multipliers. A fault may act on x-normal, y-normal or z-normal faces:

```python
FaultPlane3D(
    name="F_MAIN",
    axis="x",
    index=10,
    multiplier=0.0,
)
```

Typical meanings:

```text
multiplier = 0.0     sealing fault
0 < multiplier < 1   partially sealing/leaky fault
multiplier = 1.0     no transmissibility modification
multiplier > 1.0     enhanced communication
```

Inactive cells are also converted into transmissibility barriers. Faces touching inactive cells receive zero multiplier.

---

## 10. Wells and completions

Step 7 uses `FieldWell3D`, which contains multiple `Completion3D` objects. Completions can be created from a trajectory:

```python
trajectory = WellTrajectory3D.vertical("PROD", x=800.0, y=400.0, z_top=0.0, z_bottom=60.0)
completions = completions_from_trajectory(grid, trajectory, permeability, orientation="z", active=active_map)
well = FieldWell3D("PROD", "producer", "liquid_rate", -500.0/day, completions, min_bhp=120*bar)
```

Supported controls are:

```text
bhp
water_rate
oil_rate
gas_rate
liquid_rate
total_rate
```

The current well model is completion-based but not yet segmented. For BHP wells, each completion computes phase rates using phase pressure drawdown. For rate wells, target rates are distributed over completions using mobility-weighted productivity.

---

## 11. Field schedules

The 3D schedule module is:

```text
src/blackoil/schedule3d.py
```

It contains:

```python
FieldWellControlEvent3D
FieldWellSchedule3D
```

Example:

```python
schedule = FieldWellSchedule3D(
    base_wells=wells,
    events=[
        FieldWellControlEvent3D(0.0, "INJ", control="water_rate", target=300.0/day),
        FieldWellControlEvent3D(100.0*day, "PROD", control="bhp", target=150.0*bar),
    ],
    report_times=[0.0, 100.0*day, 200.0*day],
)
```

The simulator lands exactly on schedule milestones and restart milestones.

---

## 12. Restarts

Restart support is provided by:

```text
src/blackoil/restart.py
```

A restart file stores:

- simulation time;
- oil pressure;
- water saturation;
- third primary variable;
- saturated/undersaturated phase map;
- cumulative production and injection quantities;
- metadata.

Restart files are compressed `.npz` files. They intentionally do not store static objects such as grids, PVT tables or schedules. A user recreates the simulator configuration and then loads the restart into it.

---

## 13. Visualization and ParaView workflow

The 3D visualization module is:

```text
src/blackoil/visualization3d.py
```

The final example writes:

```text
outputs/example_14_step7_full_3d_black_oil/
├── step7_static_reservoir.vtk
├── step7_well_trajectories.vtk
├── step7_well_completions.vtk
├── vtk_timeseries/
│   ├── step7_blackoil_state_series.pvd
│   ├── step7_blackoil_state_00000.vtk
│   ├── step7_blackoil_state_00001.vtk
│   └── ...
├── step7_timestep_report.csv
├── step7_control_history.csv
├── step7_state_statistics.csv
├── step7_state_timeseries.npz
└── diagnostic plots
```

To inspect the 3D result in ParaView:

1. Open:

```text
outputs/example_14_step7_full_3d_black_oil/vtk_timeseries/step7_blackoil_state_series.pvd
```

2. Also open:

```text
outputs/example_14_step7_full_3d_black_oil/step7_static_reservoir.vtk
outputs/example_14_step7_full_3d_black_oil/step7_well_trajectories.vtk
outputs/example_14_step7_full_3d_black_oil/step7_well_completions.vtk
```

3. Use ParaView filters such as:

```text
Threshold
Slice
Clip
Contour
Glyph
Temporal Statistics
```

Recommended fields to visualize:

```text
pressure_bar
water_saturation
gas_saturation
oil_saturation
Rs_sm3_sm3
phase_state_saturated
kx_mD
porosity
fault_indicator
well_index_sum
```

---

## 14. Main Step 7 example

Run:

```bash
python examples/14_full_3d_black_oil_integrated.py
```

This example creates a small but complete 3D case with:

- `5 x 4 x 3 = 60` cells;
- active/inactive cells;
- three permeability zones;
- a Gaussian high-permeability channel contribution;
- two faults/transmissibility multiplier surfaces;
- three wells;
- vertical and horizontal trajectories;
- multiple completions;
- live-oil PVT table;
- primary-variable switching;
- capillary pressure;
- gravity;
- GMRES/ILU sparse Newton solve;
- scheduled well-control changes;
- restart files;
- VTK/PVD export;
- CSV diagnostics;
- publication-style plots.

The model is intentionally small so it runs quickly. It is not intended as a field-scale benchmark, but as a complete integration and regression case.

---

## 15. Testing

Run all tests:

```bash
pytest -q
```

The test suite covers:

- grid construction;
- transmissibility and flux shapes;
- relative permeability;
- PVT interpolation;
- dead-oil residuals;
- live-oil residuals;
- phase switching;
- sparse Jacobian infrastructure;
- 2D grid and flux tools;
- schedules and restarts;
- 3D mesh tools;
- property mapping;
- faults;
- wells;
- visualization export;
- integrated Step 7 3D black-oil run.

---

## 16. Units

The framework uses SI units internally:

```text
pressure:       Pa
length:         m
volume:         m^3
time:           s
permeability:   m^2
viscosity:      Pa s
rates:          m^3/s
```

Convenience constants are available:

```python
from blackoil.units import day, year, bar, cp, md, md_to_m2
```

Examples are written using readable engineering units but convert them to SI before solving.

---

## 17. Important limitations

This is a research framework, not a production reservoir simulator. The most important limitations are:

1. **Grid geometry**: full flow currently uses structured Cartesian 3D grids. Corner-point geometry is prepared but not yet implemented as a flow grid.
2. **Jacobian**: the Step 7 Jacobian is sparse finite difference, not analytic.
3. **Preconditioning**: ILU/Jacobi are available, but CPR-like pressure preconditioning is not yet implemented.
4. **Phase switching**: phase state is fixed during one Newton solve and updated after convergence.
5. **Wells**: wells are multi-completion but not segmented.
6. **Schedules**: schedule events change well controls but do not yet include group/facility constraints.
7. **Relative permeability**: Corey-type relative permeability is used; Stone models and endpoint scaling can be added later.
8. **Capillarity**: smooth verification capillary closures are included, but hysteresis is not.
9. **Parallelism**: the current implementation is serial Python/SciPy.
10. **Field scale**: large 3D cases will require analytic Jacobians, stronger preconditioning and possibly parallel assembly.

These limitations are deliberate. The framework prioritizes transparency, extensibility and stepwise verification.

---

## 18. Recommended next upgrades

The natural next scientific/software upgrades are:

### 18.1 Analytic sparse Jacobian

Replace sparse finite-difference Jacobian assembly with analytic block assembly. This will improve speed, robustness and scalability.

### 18.2 Automatic differentiation on active-cell variables

The current vector uses full-grid variables. A more advanced implementation would compress active cells and use AD only over active variables.

### 18.3 CPR preconditioning

A constrained-pressure-residual preconditioner would be the natural reservoir-simulation preconditioner for large black-oil systems.

### 18.4 Corner-point grid flow support

The corner-point preparation layer should evolve into:

```text
COORD/ZCORN/ACTNUM import
corner-point cell geometry
non-neighbour connections
fault transmissibility multipliers
```

### 18.5 More realistic relative permeability

Add:

- Stone I;
- Stone II;
- endpoint scaling;
- hysteresis;
- region-dependent tables.

### 18.6 Well and facility models

Add:

- segmented wells;
- wellbore pressure drop;
- group constraints;
- separator conditions;
- surface-network constraints.

### 18.7 HPC pathway

For HPC-scale simulations, the framework should move toward:

- sparse block assembly;
- domain decomposition;
- PETSc or Trilinos backends;
- parallel VTK output;
- restart partitioning;
- batch validation cases.

---

## 19. Minimal Step 7 usage sketch

A compact Step 7 workflow looks like this:

```python
from blackoil.grid3d import CartesianGrid3D
from blackoil.rock import Rock
from blackoil.pvt import BlackOilPVTTable, TabulatedFluid
from blackoil.relperm import CoreyThreePhaseRelPerm
from blackoil.state import StateBlackOil
from blackoil.black_oil_7 import FullBlackOilSimulator3D

# Create grid, rock, PVT, relperm, wells, faults and state.
# Then:

sim = FullBlackOilSimulator3D(
    grid=grid,
    rock=rock,
    water=water,
    oil=oil,
    gas=gas,
    relperm=relperm,
    state=state,
    wells=wells,
    schedule=schedule,
    capillary=capillary,
    transmissibility_multipliers=multipliers,
    active=active_map,
)

results = sim.run(
    t_final=20*day,
    dt_initial=5*day,
    dt_min=0.1*day,
    dt_max=5*day,
)
```

For a complete version, inspect:

```text
examples/14_full_3d_black_oil_integrated.py
```

---

## 20. Suggested citation-style description

A suitable technical description of the framework is:

> A staged, fully implicit finite-volume framework for multiphase black-oil reservoir simulation, including conservative live-oil component equations, primary-variable phase switching, solution gas, gravity, capillary pressure, sparse Newton-Krylov solvers, schedules, restarts, heterogeneous 2D/3D reservoirs, inactive cells, transmissibility multipliers, multi-completion wells and VTK/PVD visualization.

---

## 21. Final note

The most important design principle of this repository is controlled complexity. Each physical and numerical feature was introduced only after the previous one became testable. This makes the final 3D simulator easier to understand, easier to debug, and easier to extend toward real field-scale reservoir simulation.
