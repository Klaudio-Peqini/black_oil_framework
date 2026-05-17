# Step 4B — Bubble-point phase-state logic

This stage adds the first phase-state layer to the live-oil black-oil framework.
The simulator can now distinguish between undersaturated and saturated oil cells.

## Primary-variable interpretation

The simulator stores three primary variables per cell:

\[
(p_o, S_w, x).
\]

The meaning of \(x\) depends on the phase state:

- undersaturated oil cell:

\[
x = R_s, \qquad S_g = 0;
\]

- saturated oil cell:

\[
x = S_g, \qquad R_s = R_s^{sat}(p_o).
\]

The Boolean array `is_saturated` carries the cell-wise phase map.

## Bubble-point condition

For an undersaturated cell, the current dissolved gas ratio is compared with the
saturated value from the PVT table:

\[
R_s > R_s^{sat}(p_o).
\]

If this occurs after a Newton step, the cell is switched to saturated state and
the excess gas is converted into free-gas saturation. The implemented conversion
approximately preserves the gas-component inventory:

\[
R_s^{old}\frac{1-S_w}{B_o}
=
R_s^{sat}(p_o)\frac{1-S_w-S_g}{B_o}
+
\frac{S_g}{B_g}.
\]

Solving for \(S_g\) gives

\[
S_g =
\frac{\left(R_s^{old}-R_s^{sat}\right)(1-S_w)/B_o}
{1/B_g - R_s^{sat}/B_o}.
\]

When a saturated cell has negligible gas saturation, it can switch back to an
undersaturated state by converting the gas-component inventory into an equivalent
\(R_s\).

## Numerical strategy

The phase map is fixed during one Newton solve. After convergence, a conservative
phase-state update is applied. This avoids the discontinuous active-set change
inside the Newton iteration and is much easier to verify in the present dense
finite-difference Jacobian prototype.

For undersaturated cells in this Step 4B implementation, \(R_s\) is held fixed
inside the implicit residual unless the phase-state update detects bubble-point
crossing. This avoids a near-singular gas equation in depletion examples where
there is no independent gas injection. The saturated cells still use the full gas
component conservation equation.

This is a transitional model. The next version should replace this practical
stabilization with a fully conservative primary-variable switching formulation
using either analytic derivatives, automatic differentiation, or a sparse
nonlinear solver.

## New example

Run:

```bash
python examples/05_live_oil_phase_switching_black_oil.py
```

The example starts with undersaturated live oil and uses a low-BHP producer to
cross the bubble point locally. It outputs pressure, gas saturation, \(R_s\),
saturated-cell fraction, well rates, producing GOR, and material-balance plots.
