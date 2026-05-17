# Step 4A: Saturated live-oil black-oil model

This stage upgrades the dead-oil simulator to a first true black-oil formulation.
The model is intentionally restricted to the saturated case, where a free-gas
phase is present in every active cell. This avoids phase-appearance logic while
still introducing the essential black-oil coupling through the solution gas-oil
ratio \(R_s(p)\).

## Primary variables

For each finite-volume cell the primary variables are

\[
\mathbf{x}_i = \left[p_{o,i}, S_{w,i}, S_{g,i}\right]^T,
\]

with

\[
S_o = 1 - S_w - S_g.
\]

The model enforces admissible saturation bounds during timestep acceptance:

\[
S_w \ge S_{wc}, \qquad S_g \ge S_{gc}, \qquad S_o \ge S_{or}.
\]

## Component equations

The water component is

\[
\frac{\partial}{\partial t}\left(\phi \frac{S_w}{B_w}\right)
+ \nabla\cdot\left(\frac{\mathbf{u}_w}{B_w}\right) = q_w.
\]

The oil component is

\[
\frac{\partial}{\partial t}\left(\phi \frac{S_o}{B_o}\right)
+ \nabla\cdot\left(\frac{\mathbf{u}_o}{B_o}\right) = q_o.
\]

The gas component contains both dissolved gas in oil and free gas:

\[
\frac{\partial}{\partial t}
\left[\phi\left(R_s\frac{S_o}{B_o}+\frac{S_g}{B_g}\right)\right]
+ \nabla\cdot
\left(R_s\frac{\mathbf{u}_o}{B_o}+\frac{\mathbf{u}_g}{B_g}\right)
= q_g.
\]

## Phase fluxes

For Step 4A, the model uses no capillary pressure and no gravity. Therefore all
phases share the oil pressure. The stock-tank phase fluxes are discretized by
TPFA and first-order upwinding:

\[
F_{\alpha,ij} = -T_{ij}\frac{k_{r\alpha}^{up}}{\mu_\alpha^{up}B_\alpha^{up}}(p_j-p_i),
\qquad \alpha\in\{w,o,g\}.
\]

The gas component flux is

\[
F_{G,ij} = R_s^{up} F_{o,ij} + F_{g,ij}.
\]

## PVT data

The live-oil example uses a CSV table containing

- pressure,
- \(B_o, B_w, B_g\),
- \(\mu_o, \mu_w, \mu_g\),
- phase densities,
- \(R_s(p)\).

The current interpolation is linear with pressure clamping outside the tabulated
range. This keeps early Newton solves stable and avoids silent extrapolation.

## Well model

The stage supports

- rate water injection,
- rate oil/gas sources,
- BHP-controlled three-phase production.

For a BHP well the phase source terms are

\[
q_\alpha = WI\frac{k_{r\alpha}}{\mu_\alpha B_\alpha}(p_{bhp}-p_{cell}).
\]

The gas component produced through the oil phase is added as

\[
q_G = q_g + R_s q_o.
\]

## Current limitations

This stage does not yet include:

- undersaturated oil,
- bubble-point switching,
- gas phase disappearance,
- capillary pressure,
- gravity,
- 2D/3D grids,
- Stone I/Stone II three-phase relative permeability,
- analytical Jacobian.

The next step should add bubble-point and phase-state logic while preserving the
same component-residual structure.
