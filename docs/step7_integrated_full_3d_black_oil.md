# Step 7 — Integrated Full 3D Black-Oil Framework

Step 7 is the first end-to-end 3D reservoir-simulation layer in the staged black-oil framework. It combines the physical and numerical elements developed in the previous steps into a single integrated simulator:

- conservative live-oil black-oil component equations;
- primary-variable switching between undersaturated and saturated oil cells;
- solution gas ratio \(R_s\);
- free gas saturation \(S_g\);
- 3D finite-volume TPFA fluxes;
- gravity through phase potentials;
- capillary-pressure phase-pressure relations;
- heterogeneous anisotropic permeability fields;
- inactive cells / ACTNUM-style masks;
- transmissibility multipliers and faults;
- multi-completion 3D wells;
- field-style production schedules;
- sparse Newton solver with sparse finite-difference Jacobian, Krylov methods, and preconditioning;
- restart files;
- VTK/PVD export for ParaView and optional PyVista workflows.

The main implementation is:

```text
src/blackoil/black_oil_7.py
```

with the central class:

```python
FullBlackOilSimulator3D
```

and residual function:

```python
residual_live_oil_7_3d
```

The main runnable demonstration is:

```bash
python examples/14_full_3d_black_oil_integrated.py
```

---

## 1. Primary variables and phase state

The simulator uses the same primary-variable switching strategy developed in Steps 4B and 4C. The unknown vector is ordered as:

\[
\mathbf{x} = [p_o, S_w, x]^T,
\]

where the meaning of \(x\) depends on the phase state of each cell.

For undersaturated cells:

\[
S_g = 0, \qquad x = R_s.
\]

For saturated cells:

\[
S_g > 0, \qquad x = S_g, \qquad R_s = R_s^{sat}(p_o).
\]

This allows the gas component to be conserved across both regimes without using a separate complementarity solver at this stage.

---

## 2. Conservative component equations

The full black-oil component equations are solved in finite-volume form.

### Water component

\[
\frac{\partial}{\partial t}\left(\phi \frac{S_w}{B_w}\right)
+ \nabla\cdot\left(\frac{\mathbf{u}_w}{B_w}\right)
= q_w.
\]

### Oil component

\[
\frac{\partial}{\partial t}\left(\phi \frac{S_o}{B_o}\right)
+ \nabla\cdot\left(\frac{\mathbf{u}_o}{B_o}\right)
= q_o.
\]

### Gas component

\[
\frac{\partial}{\partial t}\left[
\phi\left(R_s\frac{S_o}{B_o}+\frac{S_g}{B_g}\right)\right]
+ \nabla\cdot\left[
R_s\frac{\mathbf{u}_o}{B_o}+\frac{\mathbf{u}_g}{B_g}\right]
= q_g.
\]

The gas component includes both dissolved gas and free gas. This is the key feature that distinguishes the current simulator from the earlier dead-oil and immiscible prototypes.

---

## 3. Gravity and capillary pressure

The phase pressures are computed from oil pressure and capillary closures:

\[
p_w = p_o - p_{cow}(S_w),
\]

\[
p_g = p_o + p_{cgo}(S_g).
\]

The phase potential difference across a face uses positive-downward depth \(D\):

\[
\Delta\Phi_{\alpha,ij}
= (p_{\alpha,j}-p_{\alpha,i})
- \rho_{\alpha,up} g (D_j-D_i).
\]

The phase flux is then:

\[
F_{\alpha,ij}
= -T_{ij}\frac{k_{r\alpha,up}}{\mu_{\alpha,up}B_{\alpha,up}}\Delta\Phi_{\alpha,ij}.
\]

The gas-component flux uses upwind oil flux for dissolved gas:

\[
F_{g,comp,ij} = R_{s,up(o)}F_{o,ij} + F_{g,ij}.
\]

---

## 4. 3D grid, transmissibility, faults, and inactive cells

The simulator uses `CartesianGrid3D` as the first full 3D grid type. The flattened cell index is:

\[
cell(i,j,k) = kN_xN_y + jN_x + i.
\]

The three geometric transmissibility sets are:

\[
T_x, \qquad T_y, \qquad T_z.
\]

Anisotropic permeability is passed as:

```python
permeability = {"kx": Kx, "ky": Ky, "kz": Kz}
```

Faults and inactive cells are represented by face transmissibility multipliers. A sealing fault uses multiplier zero; a leaky fault uses a value between zero and one. Inactive cells are handled in two ways:

1. faces touching inactive cells are given zero transmissibility;
2. inactive cells are constrained to keep their old primary variables in the residual.

This keeps the full logical grid intact while preventing inactive cells from participating in flow.

---

## 5. Multi-completion 3D well model

Step 7 uses `FieldWell3D` objects from `wells3d.py`. A well contains a list of `Completion3D` entries, each with:

- cell index;
- well index;
- open/shut status;
- skin;
- optional segment number.

The current well controls are:

```text
bhp
water_rate
oil_rate
gas_rate
liquid_rate
total_rate
```

For BHP wells, each completion contributes:

\[
q_{\alpha,c} = WI_c\frac{k_{r\alpha,c}}{\mu_{\alpha,c}B_{\alpha,c}}(p_{bhp}-p_{\alpha,c}).
\]

For rate-controlled wells, the requested rate is distributed over open completions according to mobility-weighted completion productivity. Producer wells can switch to a minimum BHP limit; injector wells can switch to a maximum BHP limit.

This is still a compact well model. Future refinements may add explicit wellbore pressure drops, segmented wells, group controls, and facility constraints.

---

## 6. Sparse nonlinear solve

The nonlinear system is solved by a damped Newton method using `SparseNewtonSolver`. The default Step 7 route is:

```text
sparse finite-difference Jacobian + GMRES + ILU
```

The sparse Jacobian structure is generated from the structured 3D finite-volume stencil. Each residual cell depends on itself and its face-neighbouring cells. This produces a sparse block matrix with three equation blocks and three variable blocks.

Available linear-solver options include:

```text
spsolve
gmres
bicgstab
lgmres
```

Available preconditioners include:

```text
none
jacobi
ilu
```

The Jacobian is still finite-difference-based. For production-scale simulation, the next major numerical upgrade would be analytic or automatic-differentiated sparse Jacobian assembly.

---

## 7. Schedules and restarts

Step 7 introduces `FieldWellSchedule3D`, which is a multi-completion schedule layer for 3D field wells. It supports piecewise-constant changes in:

- control type;
- target value;
- open/shut status;
- BHP limits.

The simulator lands exactly on schedule milestones and restart milestones. Restart files are saved as compressed `.npz` files containing:

- pressure;
- water saturation;
- third primary variable;
- saturated/undersaturated phase map;
- cumulative production/injection quantities;
- metadata.

Restart files intentionally do not serialize the entire grid and PVT setup. The user recreates the static model and loads the dynamic state into it.

---

## 8. Visualization and diagnostics

The Step 7 example exports:

```text
step7_static_reservoir.vtk
step7_well_trajectories.vtk
step7_well_completions.vtk
vtk_timeseries/step7_blackoil_state_series.pvd
vtk_timeseries/step7_blackoil_state_*.vtk
step7_state_timeseries.npz
step7_timestep_report.csv
step7_control_history.csv
step7_state_statistics.csv
```

The `.pvd` file is the recommended ParaView entry point. It loads all time steps as a time series. The static reservoir, well trajectories, and completion points can be opened alongside it.

---

## 9. Current approximations

Step 7 is an integrated research simulator, not a commercial-grade reservoir simulator. The main approximations are:

1. The grid is structured Cartesian. Corner-point geometry is prepared but not yet used for full flow.
2. The well model is completion-based but not yet segmented.
3. The Jacobian is sparse finite difference, not analytic.
4. Phase switching is fixed during a Newton step and updated after convergence.
5. Rate wells distribute phase rates by local mobility-weighted completion productivity.
6. Boundary-condition support is intentionally simple.
7. Capillary models are smooth verification closures, not field-calibrated hysteretic curves.

These approximations are acceptable for a scientific development framework. They keep the code readable, verifiable, and extensible.

---

## 10. Recommended next research/development upgrades

A realistic continuation would include:

1. analytic sparse Jacobian assembly;
2. automatic differentiation on active-cell compressed variables;
3. CPR-like pressure preconditioning;
4. corner-point grid import from GRDECL-like data;
5. endpoint-scaled Stone I/Stone II relative permeability;
6. capillary hysteresis and drainage/imbibition logic;
7. segmented wells and group controls;
8. aquifer models;
9. larger validation problems such as SPE-style benchmarks;
10. MPI/domain decomposition or parallel assembly for HPC use.

Step 7 provides the first complete skeleton into which these refinements can be inserted.
