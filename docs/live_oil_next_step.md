# Proposed next step after the dead-oil model

The next development stage should be a live-oil black-oil model with solution gas, but it should be introduced in two controlled sub-stages.

## Stage 4A: undersaturated live-oil model

Use primary variables

\[
p_o, \qquad S_w, \qquad R_s .
\]

In this sub-stage, there is no free gas phase:

\[
S_g = 0, \qquad S_o = 1-S_w .
\]

The gas component is present only as dissolved gas in oil. The gas conservation equation becomes

\[
\frac{\partial}{\partial t}\left(\phi \frac{R_s S_o}{B_o}\right)
+
\nabla\cdot\left(\frac{R_s \mathbf u_o}{B_o}\right)
=
q_g .
\]

This is the safest bridge from dead-oil to full black-oil because it adds the gas component without introducing free-gas mobility and saturation immediately.

## Stage 4B: saturated live-oil model with free gas

Use primary variables

\[
p_o, \qquad S_w, \qquad S_g .
\]

The oil saturation is

\[
S_o = 1-S_w-S_g .
\]

The solution gas-oil ratio is constrained by the saturated PVT relation

\[
R_s = R_s^{sat}(p_o) .
\]

The gas conservation equation is then

\[
\frac{\partial}{\partial t}
\left[
\phi\left(
\frac{R_s S_o}{B_o}+
\frac{S_g}{B_g}
\right)
\right]
+
\nabla\cdot
\left[
\frac{R_s \mathbf u_o}{B_o}+
\frac{\mathbf u_g}{B_g}
\right]
=
q_g .
\]

## Phase-state switching

The main difficulty is not the gas equation itself, but variable switching:

- undersaturated cell: primary variable is \(R_s\), with \(S_g=0\),
- saturated cell: primary variable is \(S_g\), with \(R_s=R_s^{sat}(p_o)\).

A robust first implementation should avoid complicated local switching by beginning with a globally saturated case. After that, local switching can be introduced.

## Recommended next coding task

The immediate next coding task should be:

1. Add `StateLiveOil` with `p`, `sw`, `sg`, and derived `so`.
2. Add gas PVT columns: `Bg`, `mug`, and `Rs`.
3. Add a Corey gas-oil relative permeability model.
4. Add three-phase fluxes: water, oil, gas.
5. Add the gas residual while keeping all cells saturated.
6. Validate on pressure depletion below bubble point.

Only after this is stable should local phase appearance/disappearance be added.
