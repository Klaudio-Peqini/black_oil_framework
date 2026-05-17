from __future__ import annotations

import numpy as np


def finite_difference_jacobian(func, x: np.ndarray, f0: np.ndarray | None = None, rel_eps: float = 1.0e-7) -> np.ndarray:
    """Dense forward finite-difference Jacobian.

    This is deliberately simple and transparent. It is appropriate for early
    prototypes and small validation cases. For large grids, replace this module
    with analytic derivatives, automatic differentiation, or sparse coloring.
    """
    x = np.asarray(x, dtype=float)
    if f0 is None:
        f0 = np.asarray(func(x), dtype=float)
    else:
        f0 = np.asarray(f0, dtype=float)

    n = x.size
    m = f0.size
    jac = np.empty((m, n), dtype=float)

    for j in range(n):
        step = rel_eps * max(1.0, abs(x[j]))
        x_pert = x.copy()
        x_pert[j] += step
        fp = np.asarray(func(x_pert), dtype=float)
        jac[:, j] = (fp - f0) / step

    return jac
