from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def _mpl():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def _as_cell_array(grid, values, *, name="field") -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.full(grid.n_cells, float(arr), dtype=float)
    if arr.shape == (grid.nz, grid.ny, grid.nx):
        return arr.ravel()
    if arr.size == grid.n_cells:
        return arr.reshape(-1).astype(float, copy=True)
    raise ValueError(f"{name!r} must be scalar, shape (nz,ny,nx), or size n_cells")


def plot_layer_map_3d(
    path,
    grid,
    values,
    *,
    layer: int | None = None,
    title: str | None = None,
    label: str | None = None,
    cmap: str = "viridis",
    wells=None,
    fault_indicator=None,
) -> Path:
    """Plot a horizontal layer map for a 3D cell-centred scalar field."""
    plt = _mpl()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if layer is None:
        layer = grid.nz // 2
    if not (0 <= layer < grid.nz):
        raise ValueError("layer index out of range")
    arr = grid.reshape(_as_cell_array(grid, values)).astype(float)[layer]
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    im = ax.imshow(arr, origin="lower", extent=[0, grid.lx, 0, grid.ly], aspect="auto", cmap=cmap)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title or f"Layer {layer}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(label or title or "value")

    if fault_indicator is not None:
        fi = grid.reshape(_as_cell_array(grid, fault_indicator, name="fault_indicator"))[layer]
        yy, xx = np.where(fi > 0.5)
        if xx.size:
            ax.scatter((xx + 0.5) * grid.dx, (yy + 0.5) * grid.dy, marker="s", s=8, facecolors="none", edgecolors="k", linewidths=0.35, label="fault/barrier")
    if wells is not None:
        for well in wells:
            xs, ys = [], []
            for comp in well.completions:
                i, j, k = grid.unravel_cell(comp.cell)
                if k == layer:
                    xs.append((i + 0.5) * grid.dx)
                    ys.append((j + 0.5) * grid.dy)
            if xs:
                ax.scatter(xs, ys, s=28, marker="o", label=well.name)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_well_trajectories_3d(path, grid, trajectories: Sequence[object], wells=None) -> Path:
    """Plot well trajectories in XY and XZ projections for reports."""
    plt = _mpl()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    ax_xy, ax_xz = axes
    for n, trajectory in enumerate(trajectories, start=1):
        pts = np.asarray(trajectory.points, dtype=float)
        label = getattr(trajectory, "name", f"WELL_{n}")
        ax_xy.plot(pts[:, 0], pts[:, 1], marker="o", label=label)
        ax_xz.plot(pts[:, 0], pts[:, 2], marker="o", label=label)
    ax_xy.set_xlim(0, grid.lx); ax_xy.set_ylim(0, grid.ly)
    ax_xz.set_xlim(0, grid.lx); ax_xz.set_ylim(grid.lz, 0)
    ax_xy.set_xlabel("x [m]"); ax_xy.set_ylabel("y [m]"); ax_xy.set_title("Well trajectories: map view")
    ax_xz.set_xlabel("x [m]"); ax_xz.set_ylabel("z [m]"); ax_xz.set_title("Well trajectories: vertical projection")
    ax_xy.grid(True, alpha=0.25); ax_xz.grid(True, alpha=0.25)
    ax_xy.legend(fontsize=8); ax_xz.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_state_time_series(path, history: Mapping[str, Sequence[float]], *, title: str = "3D state diagnostics") -> Path:
    """Plot scalar diagnostic histories such as means, maxima and rates."""
    plt = _mpl()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if "time_days" in history:
        time = np.asarray(history["time_days"], dtype=float)
        xlabel = "time [days]"
    elif "time" in history:
        time = np.asarray(history["time"], dtype=float)
        xlabel = "time"
    else:
        first = next(iter(history.values()))
        time = np.arange(len(first), dtype=float)
        xlabel = "step"
    keys = [k for k in history.keys() if k not in {"time", "time_days"}]
    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    for key in keys:
        arr = np.asarray(history[key], dtype=float)
        if arr.size == time.size:
            ax.plot(time, arr, marker="o", linewidth=1.5, label=key)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("diagnostic value")
    ax.grid(True, alpha=0.25)
    if keys:
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_property_histograms(path, fields: Mapping[str, Sequence[float] | np.ndarray], *, bins: int = 40) -> Path:
    """Plot compact histograms of selected reservoir properties."""
    plt = _mpl()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(fields)
    if n == 0:
        raise ValueError("at least one field is required")
    fig, axes = plt.subplots(n, 1, figsize=(7.4, max(3.0, 2.4 * n)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (name, values) in zip(axes, fields.items()):
        arr = np.asarray(values, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        ax.hist(arr, bins=bins)
        ax.set_title(str(name))
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("value")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_pore_volume_by_zone(path, zone_ids, pore_volume) -> Path:
    """Plot pore-volume distribution by integer zone identifier."""
    plt = _mpl()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    zones = np.asarray(zone_ids, dtype=int).ravel()
    pv = np.asarray(pore_volume, dtype=float).ravel()
    if zones.size != pv.size:
        raise ValueError("zone_ids and pore_volume must have the same size")
    unique = np.unique(zones)
    totals = np.asarray([pv[zones == z].sum() for z in unique], dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    ax.bar([str(z) for z in unique], totals)
    ax.set_xlabel("zone id")
    ax.set_ylabel("pore volume [m³]")
    ax.set_title("Pore-volume distribution by zone")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
