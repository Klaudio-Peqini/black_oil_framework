from __future__ import annotations

import numpy as np


def layered_permeability_2d(grid, layer_values, direction: str = "y") -> np.ndarray:
    """Create a piecewise-layered permeability field for validation cases.

    ``layer_values`` is a sequence of permeability values. For ``direction='y'``
    the values are assigned to horizontal layers from bottom to top. For
    ``direction='x'`` they are assigned from left to right.
    """
    values = np.asarray(layer_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("layer_values must be a non-empty 1D sequence")
    if np.any(values <= 0.0):
        raise ValueError("layer permeabilities must be positive")
    field = np.empty((grid.ny, grid.nx), dtype=float)
    direction = direction.lower()
    if direction == "y":
        bins = np.linspace(0, grid.ny, values.size + 1, dtype=int)
        for k, value in enumerate(values):
            field[bins[k] : bins[k + 1], :] = value
    elif direction == "x":
        bins = np.linspace(0, grid.nx, values.size + 1, dtype=int)
        for k, value in enumerate(values):
            field[:, bins[k] : bins[k + 1]] = value
    else:
        raise ValueError("direction must be 'x' or 'y'")
    return field.ravel()


def gaussian_channel_permeability_2d(
    grid,
    k_background: float,
    k_channel: float,
    y_center_fraction: float = 0.5,
    width_fraction: float = 0.12,
) -> np.ndarray:
    """Smooth high-permeability channel used in 2D validation examples."""
    if k_background <= 0.0 or k_channel <= 0.0:
        raise ValueError("permeabilities must be positive")
    _x, y = grid.centers
    y0 = y_center_fraction * grid.ly
    sigma = max(width_fraction * grid.ly, 1.0e-30)
    weight = np.exp(-0.5 * ((y - y0) / sigma) ** 2)
    return k_background + (k_channel - k_background) * weight


def lognormal_permeability_2d(
    grid,
    geometric_mean: float,
    sigma_log: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a reproducible lognormal permeability field."""
    if geometric_mean <= 0.0:
        raise ValueError("geometric_mean must be positive")
    rng = np.random.default_rng(seed)
    return geometric_mean * np.exp(sigma_log * rng.standard_normal(grid.n_cells))
