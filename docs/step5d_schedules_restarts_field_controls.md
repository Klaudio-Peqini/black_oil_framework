# Step 5D — Production schedules, control histories, restarts, and field-style multiwell controls

Step 5D adds an operational layer on top of the Step 5C two-dimensional heterogeneous black-oil simulator. The underlying flow model is unchanged: the simulator still solves the conservative live-oil black-oil equations with phase-state switching, capillary pressure, gravity, sparse Newton iterations, and heterogeneous anisotropic finite-volume transmissibilities. The new contribution is that well controls are no longer fixed for the whole simulation.

## 1. Schedule model

The schedule is represented by `WellSchedule`, a replayable list of `WellControlEvent` objects. Each event is piecewise constant from its event time until another event changes the same well. Events may:

- open a new well;
- change control mode;
- change a rate or BHP target;
- change BHP limits;
- shut a well;
- associate wells with diagnostic groups.

The supported controls are inherited from `ControlledWell`:

\[
\texttt{bhp},\quad \texttt{water\_rate},\quad \texttt{oil\_rate},\quad \texttt{gas\_rate},\quad \texttt{liquid\_rate},\quad \texttt{total\_rate}.
\]

The sign convention remains unchanged. Positive rates inject into the reservoir and negative rates produce from it.

## 2. Rate/BHP switching histories

At the beginning of a timestep, the schedule-resolved wells are converted into active `ControlledWell` objects. The existing rate/BHP limit logic then decides whether a well remains on rate control or switches to a BHP limit. This produces two histories:

1. `schedule_history`: the user-requested field schedule state;
2. `active_control_log`: the actually active solver control after rate/BHP switching.

This distinction is important. A well may be requested as a rate-controlled producer, but if the requested rate would require a bottom-hole pressure below the specified minimum, the active well becomes a BHP well for that timestep.

## 3. Schedule milestones and timestep control

The adaptive timestepper is forced to land exactly on:

- schedule event times;
- requested report times;
- optional restart times.

This avoids changing a well target inside a nonlinear solve. It also makes output histories and restart files easier to interpret.

## 4. Restart files

Restart files are stored as compressed `.npz` files. A restart contains:

- simulation time;
- oil pressure;
- water saturation;
- primary third variable, either \(R_s\) or \(S_g\);
- saturated/undersaturated cell flag;
- cumulative produced and injected quantities;
- simple JSON metadata.

The grid, rock model, PVT tables, relative permeability, capillary model, solver settings, and schedule are intentionally not stored in the restart. They should be recreated by the case script, after which the dynamic state is loaded with:

```python
restart = load_black_oil_restart("restart_t1.npz")
apply_black_oil_restart(simulator, restart)
```

This is a transparent research-code design. It avoids opaque binary case files while still supporting reproducible continuation.

## 5. New files

The main new modules are:

```text
src/blackoil/schedule.py
src/blackoil/restart.py
src/blackoil/black_oil_5d.py
```

The main example is:

```text
examples/10_scheduled_field_controls_restart.py
```

The example writes:

```text
outputs/example_10_step5d_schedules_restart/field_schedule.csv
outputs/example_10_step5d_schedules_restart/step5d_timestep_report.csv
outputs/example_10_step5d_schedules_restart/step5d_schedule_history.csv
outputs/example_10_step5d_schedules_restart/step5d_active_control_history.csv
outputs/example_10_step5d_schedules_restart/restart/*.npz
outputs/example_10_step5d_schedules_restart/final_state.vtk
```

## 6. Scientific status

Step 5D is now much closer to field-style reservoir simulation because the model can describe changing operational strategy. The simulator can run depletion/injection histories, open and shut wells, impose BHP limits, and restart from intermediate states.

Remaining approximations include:

- cell-centred wells rather than full well trajectories;
- no group-level optimization or strict facility constraints yet;
- no perforation-level completion model;
- no wellbore hydraulics;
- schedule format is intentionally simple and CSV-friendly rather than a full Eclipse-style deck.

The next step should be Step 6A: a 3D mesh module, beginning with structured Cartesian 3D grids and then preparing the design for corner-point/corner-grid import.
