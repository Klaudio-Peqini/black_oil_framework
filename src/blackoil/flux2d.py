from __future__ import annotations

import numpy as np

from .capillary import ZeroCapillaryPressure
from .flux import phase_pressures_black_oil
from .boundary import BoundaryConditions2D, PressureBoundary


def divergence_from_face_fluxes_2d(grid, flux_x: np.ndarray, flux_y: np.ndarray) -> np.ndarray:
    """Convert 2D interior face fluxes into finite-volume net outflow.

    ``flux_x`` is positive from west to east across vertical faces.
    ``flux_y`` is positive from lower logical row to upper logical row.
    """
    div = np.zeros(grid.n_cells, dtype=float)
    x_pairs = grid.x_face_neighbors
    y_pairs = grid.y_face_neighbors
    if len(x_pairs):
        left, right = x_pairs[:, 0], x_pairs[:, 1]
        div[left] += flux_x
        div[right] -= flux_x
    if len(y_pairs):
        south, north = y_pairs[:, 0], y_pairs[:, 1]
        div[south] += flux_y
        div[north] -= flux_y
    return div


def _phase_flux_2d_faces(grid, permeability, phase_pressure, density, kr, mu, b, gravity):
    tx, ty = grid.geometric_transmissibility(permeability)
    depth = grid.depths

    def compute(pairs, tgeo):
        if len(pairs) == 0:
            return np.empty(0), np.empty(0, dtype=int)
        i = pairs[:, 0]
        j = pairs[:, 1]
        dp = phase_pressure[j] - phase_pressure[i]
        dz = depth[j] - depth[i]
        rho_avg = 0.5 * (density[i] + density[j])
        dpot_direction = dp - rho_avg * gravity * dz
        use_i = dpot_direction <= 0.0
        up = np.where(use_i, i, j)
        dpot = dp - density[up] * gravity * dz
        mob = kr[up] / (mu[up] * b[up])
        return -tgeo * mob * dpot, up

    fx, upx = compute(grid.x_face_neighbors, tx)
    fy, upy = compute(grid.y_face_neighbors, ty)
    return fx, fy, upx, upy


def three_phase_black_oil_fluxes_2d(
    grid,
    permeability,
    p,
    sw,
    sg,
    rs,
    relperm,
    water,
    oil,
    gas,
    capillary=None,
    gravity: float = 9.80665,
):
    """Interior 2D black-oil component fluxes with gravity and capillarity.

    Returns eight arrays: water x/y, oil x/y, free-gas x/y, gas-component x/y.
    """
    cap = ZeroCapillaryPressure() if capillary is None else capillary
    p = np.asarray(p, dtype=float)
    sw = np.asarray(sw, dtype=float)
    sg = np.asarray(sg, dtype=float)
    rs = np.asarray(rs, dtype=float)
    pw, po, pg = phase_pressures_black_oil(p, sw, sg, cap)

    bw = water.formation_volume_factor(pw)
    bo = oil.formation_volume_factor(po)
    bg = gas.formation_volume_factor(pg)
    muw = water.viscosity(pw)
    muo = oil.viscosity(po)
    mug = gas.viscosity(pg)
    rhow = water.density(pw)
    rhoo = oil.density(po)
    rhog = gas.density(pg)
    krw = relperm.krw(sw, sg)
    kro = relperm.kro(sw, sg)
    krg = relperm.krg(sw, sg)

    fw_x, fw_y, _upw_x, _upw_y = _phase_flux_2d_faces(grid, permeability, pw, rhow, krw, muw, bw, gravity)
    fo_x, fo_y, upo_x, upo_y = _phase_flux_2d_faces(grid, permeability, po, rhoo, kro, muo, bo, gravity)
    fg_x, fg_y, _upg_x, _upg_y = _phase_flux_2d_faces(grid, permeability, pg, rhog, krg, mug, bg, gravity)
    fgc_x = rs[upo_x] * fo_x + fg_x
    fgc_y = rs[upo_y] * fo_y + fg_y
    return fw_x, fw_y, fo_x, fo_y, fg_x, fg_y, fgc_x, fgc_y


def _boundary_cells(grid, side: str):
    side = side.lower()
    if side == "left":
        cells = np.array([grid.cell_index(0, j) for j in range(grid.ny)], dtype=int)
        distance = 0.5 * grid.dx
        area = grid.dy * grid.thickness
    elif side == "right":
        cells = np.array([grid.cell_index(grid.nx - 1, j) for j in range(grid.ny)], dtype=int)
        distance = 0.5 * grid.dx
        area = grid.dy * grid.thickness
    elif side == "bottom":
        cells = np.array([grid.cell_index(i, 0) for i in range(grid.nx)], dtype=int)
        distance = 0.5 * grid.dy
        area = grid.dx * grid.thickness
    elif side == "top":
        cells = np.array([grid.cell_index(i, grid.ny - 1) for i in range(grid.nx)], dtype=int)
        distance = 0.5 * grid.dy
        area = grid.dx * grid.thickness
    else:
        raise ValueError(f"Unknown boundary side {side!r}")
    return cells, distance, area


def boundary_component_fluxes_2d(
    grid,
    permeability,
    p,
    sw,
    sg,
    rs,
    relperm,
    water,
    oil,
    gas,
    capillary=None,
    gravity: float = 9.80665,
    boundaries: BoundaryConditions2D | None = None,
):
    """Return net boundary outflow contributions for water/oil/gas components.

    The result is a tuple of three arrays with shape ``(n_cells,)``. Positive
    values are net outflow from the domain, consistent with the residual
    convention ``accumulation + divergence - source``.
    """
    if boundaries is None or not boundaries.has_pressure_boundaries():
        z = np.zeros(grid.n_cells, dtype=float)
        return z.copy(), z.copy(), z.copy()

    cap = ZeroCapillaryPressure() if capillary is None else capillary
    p = np.asarray(p, dtype=float)
    sw = np.asarray(sw, dtype=float)
    sg = np.asarray(sg, dtype=float)
    rs = np.asarray(rs, dtype=float)
    pw, po, pg = phase_pressures_black_oil(p, sw, sg, cap)
    krw = relperm.krw(sw, sg)
    kro = relperm.kro(sw, sg)
    krg = relperm.krg(sw, sg)
    bw = water.formation_volume_factor(pw)
    bo = oil.formation_volume_factor(po)
    bg = gas.formation_volume_factor(pg)
    muw = water.viscosity(pw)
    muo = oil.viscosity(po)
    mug = gas.viscosity(pg)
    rhow = water.density(pw)
    rhoo = oil.density(po)
    rhog = gas.density(pg)
    depth = grid.depths
    kx, ky = grid._permeability_components(permeability)

    div_w = np.zeros(grid.n_cells, dtype=float)
    div_o = np.zeros(grid.n_cells, dtype=float)
    div_g = np.zeros(grid.n_cells, dtype=float)

    for bc in boundaries.pressure_boundaries:
        cells, dist, area = _boundary_cells(grid, bc.side)
        side = bc.side.lower()
        kcomp = kx[cells] if side in {"left", "right"} else ky[cells]
        tgeo = area * kcomp / dist
        rs_bc = oil.solution_gas_ratio(np.full_like(cells, float(bc.pressure), dtype=float)) if bc.rs is None else np.full(cells.size, float(bc.rs))
        sw_bc = np.full(cells.size, float(bc.sw), dtype=float)
        sg_bc = np.full(cells.size, float(bc.sg), dtype=float)
        p_bc = np.full(cells.size, float(bc.pressure), dtype=float)
        pwb, pob, pgb = phase_pressures_black_oil(p_bc, sw_bc, sg_bc, cap)
        # Boundary depth is taken equal to adjacent cell-centre depth for this
        # first implementation. Structural boundary depths can be added later.
        dz = np.zeros(cells.size, dtype=float)

        def one_phase(p_cell, p_bound, rho_cell, rho_bound, kr_cell, mu_cell, b_cell, kr_bound, mu_bound, b_bound):
            dp = p_bound - p_cell
            rho_avg = 0.5 * (rho_cell + rho_bound)
            dpot_dir = dp - rho_avg * gravity * dz
            use_cell = dpot_dir <= 0.0  # positive cell-to-boundary outflow
            rho_up = np.where(use_cell, rho_cell, rho_bound)
            dpot = dp - rho_up * gravity * dz
            mob = np.where(use_cell, kr_cell / (mu_cell * b_cell), kr_bound / (mu_bound * b_bound))
            return -tgeo * mob * dpot, use_cell

        bwb = water.formation_volume_factor(pwb); bob = oil.formation_volume_factor(pob); bgb = gas.formation_volume_factor(pgb)
        muwb = water.viscosity(pwb); muob = oil.viscosity(pob); mugb = gas.viscosity(pgb)
        rhowb = water.density(pwb); rhoob = oil.density(pob); rhogb = gas.density(pgb)
        krwb = relperm.krw(sw_bc, sg_bc); krob = relperm.kro(sw_bc, sg_bc); krgb = relperm.krg(sw_bc, sg_bc)

        fw, _ = one_phase(pw[cells], pwb, rhow[cells], rhowb, krw[cells], muw[cells], bw[cells], krwb, muwb, bwb)
        fo, oil_from_cell = one_phase(po[cells], pob, rhoo[cells], rhoob, kro[cells], muo[cells], bo[cells], krob, muob, bob)
        fg, _ = one_phase(pg[cells], pgb, rhog[cells], rhogb, krg[cells], mug[cells], bg[cells], krgb, mugb, bgb)
        rs_up = np.where(oil_from_cell, rs[cells], rs_bc)
        fgc = rs_up * fo + fg
        div_w[cells] += fw
        div_o[cells] += fo
        div_g[cells] += fgc

    return div_w, div_o, div_g
