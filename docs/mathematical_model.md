# Mathematical model

## Current implemented stages

The current repository implements the first two stages of the black-oil development path.

## Single-phase slightly compressible flow

The conservation equation is

```math
\frac{\partial}{\partial t}\left(\frac{\phi}{B}\right)
+ \nabla\cdot\left(\frac{\mathbf u}{B}\right) = q,
```

with Darcy velocity

```math
\mathbf u = -\frac{K}{\mu}\nabla p.
```

## Two-phase oil-water flow

The primary variables are oil pressure and water saturation:

```math
x_i = (p_i, S_{w,i}).
```

Water equation:

```math
\frac{\partial}{\partial t}\left(\phi \frac{S_w}{B_w}\right)
+ \nabla\cdot\left(\frac{\mathbf u_w}{B_w}\right)=q_w.
```

Oil equation:

```math
\frac{\partial}{\partial t}\left(\phi \frac{S_o}{B_o}\right)
+ \nabla\cdot\left(\frac{\mathbf u_o}{B_o}\right)=q_o,
\qquad S_o=1-S_w.
```

Phase velocities are

```math
\mathbf u_\alpha=-K\frac{k_{r\alpha}}{\mu_\alpha}\nabla p,
\qquad \alpha\in\{w,o\}.
```

The first implementation assumes no capillary pressure and no gravity. These are intentionally left as clean extension points.
