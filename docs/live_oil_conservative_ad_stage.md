# Step 4C: conservative primary-variable switching with automatic-differentiated Jacobian

This stage upgrades the Step 4B phase-switching live-oil black-oil model. The
main purpose is to remove the temporary gas-row simplification used in Step 4B
and to prepare the nonlinear solver for larger and more realistic problems.

## Primary variables

The simulator still uses a cell-wise primary-variable interpretation:

- undersaturated cell: `(p_o, S_w, R_s)` with `S_g = 0`;
- saturated cell: `(p_o, S_w, S_g)` with `R_s = R_s^sat(p_o)`.

The phase map is held fixed during each Newton solve and updated after
convergence. This avoids an unstable phase-appearance decision inside a single
Newton iteration while still allowing conservative phase-state changes from one
accepted timestep to the next.

## Conservative residual

Step 4C activates the gas-component conservation equation in both phase states:

```text
R_g = d/dt [ phi ( R_s S_o/B_o + S_g/B_g ) ]
      + div [ R_s u_o/B_o + u_g/B_g ]
      - q_g^component.
```

In undersaturated cells `S_g = 0` and `R_s` is solved as the third primary
unknown. In saturated cells `R_s = R_s^sat(p_o)` and `S_g` is solved as the
third primary unknown. This is more conservative than the Step 4B bridge, where
`R_s` was frozen in undersaturated cells.

## Automatic-differentiated Jacobian

The new module `blackoil.black_oil_ad` contains a JAX-based residual context
that differentiates the fixed-phase-map residual with respect to the scaled
Newton variables. The Newton solver now accepts a Jacobian callback:

```python
NewtonSolverWithJacobian.solve(func, x0, jac=jacobian_callback)
```

If JAX is unavailable, the simulator can fall back to the existing dense
finite-difference Jacobian by setting:

```python
jacobian_mode="finite_difference"
```

## New files

```text
src/blackoil/ad_solver.py
src/blackoil/black_oil_ad.py
examples/06_live_oil_conservative_ad_black_oil.py
tests/test_step4c_ad.py
```

## Current limitations

This is still a one-dimensional TPFA framework. Gravity, capillary pressure,
three-dimensional grids, well completions across multiple layers, and sparse
linear algebra are deferred to later stages. The JAX Jacobian is dense in this
prototype; for realistic 3D grids the next step must introduce sparse assembly
or matrix-free Newton-Krylov methods.
