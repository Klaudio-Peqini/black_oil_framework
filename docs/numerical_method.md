# Numerical method

## Finite-volume discretization

The code uses a cell-centered finite-volume method. On a 1D Cartesian grid, the flux through the face between cells `i` and `j=i+1` is positive from left to right.

For a phase `alpha`, the stock-tank flux is approximated as

```math
F_{\alpha,ij} = -T_{ij}\frac{k_{r\alpha}^{up}}{\mu_\alpha^{up}B_\alpha^{up}}
(p_j-p_i).
```

The transmissibility is

```math
T_{ij}=\frac{A}{\Delta x_i/(2K_i)+\Delta x_j/(2K_j)}.
```

## Fully implicit time stepping

For each cell and component, the residual has the form

```math
R_i^{n+1}=\frac{A_i^{n+1}-A_i^n}{\Delta t}
+\sum_f F_{i,f}^{n+1}-Q_i^{n+1}=0.
```

The full nonlinear system is solved by Newton-Raphson:

```math
J(x^k)\Delta x^k=-R(x^k),
\qquad x^{k+1}=x^k+\Delta x^k.
```

The current framework uses a dense finite-difference Jacobian for readability. For larger grids, this should be replaced with sparse analytic derivatives or automatic differentiation.
