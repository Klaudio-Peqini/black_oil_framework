# Step 5A — Gravity, capillary pressure, and realistic well controls

This stage upgrades the Step 4C conservative live-oil black-oil simulator with three physically important mechanisms:

1. phase-potential gravity terms,
2. water-oil and gas-oil capillary-pressure relations,
3. operational well controls with rate/BHP switching.

The primary-variable switching logic from Step 4C is preserved. In an undersaturated cell the primary variables are

\[
(p_o, S_w, R_s), \qquad S_g = 0,
\]

whereas in a saturated cell the variables are

\[
(p_o, S_w, S_g), \qquad R_s = R_s^{sat}(p_o).
\]

The gas-component conservation equation remains active in both regimes.

## Phase pressures

The primary pressure is the oil pressure. Capillary pressure defines the other phase pressures:

\[
p_w = p_o - p_{cow}(S_w),
\]

\[
p_g = p_o + p_{cgo}(S_g).
\]

The implementation provides three capillary closures:

- `ZeroCapillaryPressure`,
- `LinearCapillaryPressure`,
- `BrooksCoreyCapillaryPressure`.

The Brooks-Corey-like model is bounded near residual saturations for numerical robustness in this research-stage implementation.

## Gravity and phase potentials

Depth is positive downward. For phase \(\alpha\), the finite-volume flux between two neighboring cells is computed from the phase potential difference

\[
\Delta \Phi_{\alpha,ij}
=
(p_{\alpha,j}-p_{\alpha,i})
-
\rho_{\alpha,up} g (D_j-D_i),
\]

where \(D\) is depth. A hydrostatic profile satisfying

\[
\frac{dp_\alpha}{dD} = \rho_\alpha g
\]

therefore gives approximately zero flux. First-order upwinding is based on the phase-potential direction.

The gas-component face flux is

\[
F_{g,c,ij} = R_{s,up_o}F_{o,ij}+F_{g,free,ij},
\]

where \(R_s\) is upwinded with the oil phase.

## Component accumulation

PVT properties are now evaluated at phase pressures:

\[
B_w = B_w(p_w), \qquad B_o = B_o(p_o), \qquad B_g = B_g(p_g).
\]

The gas-component accumulation remains

\[
\phi\left(R_s\frac{S_o}{B_o} + \frac{S_g}{B_g}\right).
\]

## Well controls

Step 5A introduces `ControlledWell`, which supports:

- `bhp`,
- `water_rate`,
- `oil_rate`,
- `gas_rate`,
- `liquid_rate`,
- `total_rate`.

The sign convention is unchanged:

- positive rate means injection,
- negative rate means production.

A rate-controlled producer can switch to a minimum BHP limit, while an injector can switch to a maximum BHP limit. The active control is resolved at the beginning of each timestep and kept fixed during Newton iteration. This avoids discontinuous control changes inside the nonlinear solve.

For BHP wells, Step 5A uses phase-pressure drawdowns:

\[
q_\alpha = WI\frac{k_{r\alpha}}{\mu_\alpha B_\alpha}(p_{bhp}-p_\alpha).
\]

## New files

- `src/blackoil/black_oil_5a.py`
- updated `src/blackoil/capillary.py`
- updated `src/blackoil/flux.py`
- updated `src/blackoil/wells.py`
- `examples/07_live_oil_gravity_capillary_controls.py`
- `tests/test_step5a_gravity_capillary_wells.py`

## Example

Run:

```bash
python examples/07_live_oil_gravity_capillary_controls.py
```

The output folder is:

```text
outputs/example_07_step5a_gravity_capillary_controls/
```

It contains plots for pressure, water saturation, gas saturation, solution gas ratio, capillary pressure, controlled well rates, producing GOR, final phase pressures, and material-balance error.

## Current limitations

This remains a research-scale simulator. In Step 5A the nonlinear Jacobian is still dense and is assembled by finite differences for the gravity/capillary/control residual. This is acceptable for small verification cases, but not for 2D/3D field-scale simulation.

The next step should therefore be:

**Step 5B — sparse Jacobian structure, sparse linear solvers, Newton-Krylov options, and preconditioning.**
