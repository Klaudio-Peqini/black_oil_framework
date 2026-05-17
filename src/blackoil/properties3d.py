from __future__ import annotations

import numpy as np


def _validate_positive(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must be positive")
    return arr


def layered_permeability_3d(grid, layer_values, direction: str = "z") -> np.ndarray:
    """Create a piecewise-layered 3D permeability field.

    For ``direction='z'`` values are assigned from shallow to deep. For
    ``direction='y'`` values are assigned from front to back, and for
    ``direction='x'`` values are assigned from left to right.
    """
    values = _validate_positive(np.asarray(layer_values, dtype=float), "layer_values")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("layer_values must be a non-empty 1D sequence")
    field = np.empty((grid.nz, grid.ny, grid.nx), dtype=float)
    direction = direction.lower()
    if direction == "z":
        bins = np.linspace(0, grid.nz, values.size + 1, dtype=int)
        for k, value in enumerate(values):
            field[bins[k] : bins[k + 1], :, :] = value
    elif direction == "y":
        bins = np.linspace(0, grid.ny, values.size + 1, dtype=int)
        for j, value in enumerate(values):
            field[:, bins[j] : bins[j + 1], :] = value
    elif direction == "x":
        bins = np.linspace(0, grid.nx, values.size + 1, dtype=int)
        for i, value in enumerate(values):
            field[:, :, bins[i] : bins[i + 1]] = value
    else:
        raise ValueError("direction must be 'x', 'y' or 'z'")
    return field.ravel()


def lognormal_permeability_3d(grid, geometric_mean: float, sigma_log: float = 1.0, seed: int | None = None) -> np.ndarray:
    """Generate a reproducible lognormal 3D permeability field."""
    if geometric_mean <= 0.0:
        raise ValueError("geometric_mean must be positive")
    if sigma_log < 0.0:
        raise ValueError("sigma_log must be non-negative")
    rng = np.random.default_rng(seed)
    return geometric_mean * np.exp(sigma_log * rng.standard_normal(grid.n_cells))


def gaussian_channel_permeability_3d(
    grid,
    k_background: float,
    k_channel: float,
    y_center_fraction: float = 0.5,
    z_center_fraction: float = 0.5,
    width_y_fraction: float = 0.15,
    width_z_fraction: float = 0.25,
) -> np.ndarray:
    """Smooth high-permeability channel aligned with the x direction."""
    if k_background <= 0.0 or k_channel <= 0.0:
        raise ValueError("permeabilities must be positive")
    _x, y, z = grid.centers
    y0 = y_center_fraction * grid.ly
    z0 = z_center_fraction * grid.lz
    sig_y = max(width_y_fraction * grid.ly, 1.0e-30)
    sig_z = max(width_z_fraction * grid.lz, 1.0e-30)
    weight = np.exp(-0.5 * ((y - y0) / sig_y) ** 2 - 0.5 * ((z - z0) / sig_z) ** 2)
    return k_background + (k_channel - k_background) * weight


def anisotropic_permeability_3d(grid, horizontal_k, kvkh: float = 0.1, ky_kx: float = 1.0) -> dict[str, np.ndarray]:
    """Create an anisotropic permeability dictionary from a horizontal field.

    ``horizontal_k`` may be scalar or a 3D/flat array. ``ky_kx`` controls areal
    anisotropy and ``kvkh`` controls vertical-to-horizontal permeability ratio.
    """
    if kvkh <= 0.0 or ky_kx <= 0.0:
        raise ValueError("anisotropy ratios must be positive")
    h = np.asarray(horizontal_k, dtype=float)
    if h.ndim == 0:
        kx = np.full(grid.n_cells, float(h), dtype=float)
    elif h.shape == (grid.nz, grid.ny, grid.nx):
        kx = h.ravel()
    elif h.shape == (grid.n_cells,):
        kx = h.copy()
    else:
        raise ValueError("horizontal_k must be scalar, (nz,ny,nx), or (n_cells,)")
    _validate_positive(kx, "horizontal_k")
    return {"kx": kx, "ky": ky_kx * kx, "kz": kvkh * kx}


def porosity_from_permeability_3d(k, phi_min: float = 0.08, phi_max: float = 0.28) -> np.ndarray:
    """Simple monotone porosity proxy useful for synthetic mesh examples.

    This is not a rock-physics model. It maps log-permeability to a bounded
    porosity field for testing property mapping, pore volumes and visualization.
    """
    if not (0.0 < phi_min < phi_max < 1.0):
        raise ValueError("require 0 < phi_min < phi_max < 1")
    k = _validate_positive(np.asarray(k, dtype=float), "permeability")
    logk = np.log(k)
    lo, hi = float(logk.min()), float(logk.max())
    if hi - lo < 1.0e-30:
        return np.full(k.size, 0.5 * (phi_min + phi_max), dtype=float)
    scaled = (logk - lo) / (hi - lo)
    return phi_min + (phi_max - phi_min) * scaled
