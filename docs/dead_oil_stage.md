# Dead-oil stage

This stage upgrades the initial water-oil prototype into a compressible dead-oil simulator. The gas component is still absent, but the implementation now follows the structure needed by a black-oil code:

- pressure-dependent rock porosity,
- pressure-dependent water and oil formation volume factors,
- pressure-dependent water and oil viscosities,
- tabulated PVT interpolation,
- rate and BHP well controls,
- fully implicit finite-volume residuals,
- timestep rejection,
- material-balance diagnostics.

## Primary variables

The primary unknowns per grid cell are

\[
p_o, \qquad S_w .
\]

The oil saturation is

\[
S_o = 1 - S_w .
\]

## Component equations

The water component equation is

\[
\frac{\partial}{\partial t}\left(\phi\frac{S_w}{B_w}\right)
+ \nabla\cdot\left(\frac{\mathbf u_w}{B_w}\right)
= q_w .
\]

The oil component equation is

\[
\frac{\partial}{\partial t}\left(\phi\frac{S_o}{B_o}\right)
+ \nabla\cdot\left(\frac{\mathbf u_o}{B_o}\right)
= q_o .
\]

The phase velocities are evaluated using Darcy flow and upstream mobilities.
For the current 1D implementation, gravity and capillary pressure are still disabled.

## Why this is the correct step before live oil

The live-oil model will add the gas component and the pressure-dependent solution gas-oil ratio \(R_s(p)\). However, the most important numerical machinery is already present in the dead-oil stage: fully implicit coupling, pressure-dependent accumulation, PVT interpolation, well source terms, and adaptive timestep control. This means the next extension can focus on phase-state logic and the gas conservation equation rather than rebuilding the numerical architecture.
