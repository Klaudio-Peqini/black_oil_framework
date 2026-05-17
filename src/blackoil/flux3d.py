from __future__ import annotations

import numpy as np

from .capillary import ZeroCapillaryPressure
from .flux import phase_pressures_black_oil
from .boundary3d import BoundaryConditions3D


def divergence_from_face_fluxes_3d(grid, flux_x: np.ndarray, flux_y: np.ndarray, flux_z: np.ndarray) -> np.ndarray:
    """Convert 3D interior face fluxes into finite-volume net outflow.

    ``flux_x`` is positive west-to-east, ``flux_y`` is positive front-to-back,
    and ``flux_z`` is positive shallow-to-deep. The returned array has one net
    outflow value per cell and follows the residual convention used throughout
    the framework: accumulation + divergence - source = 0.
    """
    div = np.zeros(grid.n_cells, dtype=float)
    for pairs, flux in (
        (grid.x_face_neighbors, np.asarray(flux_x, dtype=float)),
        (grid.y_face_neighbors, np.asarray(flux_y, dtype=float)),
        (grid.z_face_neighbors, np.asarray(flux_z, dtype=float)),
    ):
        if len(pairs):
            if flux.size != len(pairs):
                raise ValueError("face-flux array size is incompatible with grid")
            left, right = pairs[:, 0], pairs[:, 1]
            div[left] += flux
            div[right] -= flux
    return div


def total_transmissibility_count_3d(grid) -> int:
    """Return the number of internal faces/transmissibilities in a 3D grid."""
    return len(grid.x_face_neighbors) + len(grid.y_face_neighbors) + len(grid.z_face_neighbors)


def _transmissibilities_with_multipliers(grid, permeability, transmissibility_multipliers=None):
    tx, ty, tz = grid.geometric_transmissibility(permeability)
    if transmissibility_multipliers is None:
        return tx, ty, tz
    return transmissibility_multipliers.apply_to(tx, ty, tz)


def _phase_flux_3d_faces(grid, permeability, phase_pressure, density, kr, mu, b, gravity, transmissibility_multipliers=None):
    tx, ty, tz = _transmissibilities_with_multipliers(grid, permeability, transmissibility_multipliers)
    depth = grid.depths

    def compute(pairs, tgeo):
        if len(pairs) == 0:
            return np.empty(0), np.empty(0, dtype=int)
        left = pairs[:, 0]
        right = pairs[:, 1]
        dp = phase_pressure[right] - phase_pressure[left]
        dz = depth[right] - depth[left]
        rho_avg = 0.5 * (density[left] + density[right])
        # Directional potential only decides upwind. The final potential uses
        # the density of the selected upwind phase state.
        dpot_direction = dp - rho_avg * gravity * dz
        use_left = dpot_direction <= 0.0
        up = np.where(use_left, left, right)
        dpot = dp - density[up] * gravity * dz
        mob = kr[up] / (mu[up] * b[up])
        return -tgeo * mob * dpot, up

    fx, upx = compute(grid.x_face_neighbors, tx)
    fy, upy = compute(grid.y_face_neighbors, ty)
    fz, upz = compute(grid.z_face_neighbors, tz)
    return fx, fy, fz, upx, upy, upz


def three_phase_black_oil_fluxes_3d(
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
    *,
    capillary=None,
    gravity: float = 9.80665,
    transmissibility_multipliers=None,
):
    """Interior 3D black-oil component fluxes with gravity and capillarity.

    Returns twelve arrays: water x/y/z, oil x/y/z, free-gas x/y/z, and
    gas-component x/y/z. Flux signs follow the grid face orientation.
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

    fw_x, fw_y, fw_z, _upw_x, _upw_y, _upw_z = _phase_flux_3d_faces(
        grid, permeability, pw, rhow, krw, muw, bw, gravity, transmissibility_multipliers
    )
    fo_x, fo_y, fo_z, upo_x, upo_y, upo_z = _phase_flux_3d_faces(
        grid, permeability, po, rhoo, kro, muo, bo, gravity, transmissibility_multipliers
    )
    fg_x, fg_y, fg_z, _upg_x, _upg_y, _upg_z = _phase_flux_3d_faces(
        grid, permeability, pg, rhog, krg, mug, bg, gravity, transmissibility_multipliers
    )
    fgc_x = rs[upo_x] * fo_x + fg_x
    fgc_y = rs[upo_y] * fo_y + fg_y
    fgc_z = rs[upo_z] * fo_z + fg_z
    return fw_x, fw_y, fw_z, fo_x, fo_y, fo_z, fg_x, fg_y, fg_z, fgc_x, fgc_y, fgc_z


def boundary_component_fluxes_3d(
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
    *,
    capillary=None,
    gravity: float = 9.80665,
    boundaries: BoundaryConditions3D | None = None,
):
    """Return net boundary outflow contributions for 3D pressure boundaries."""
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
    bw = water.formation_volume_factor(pw); bo = oil.formation_volume_factor(po); bg = gas.formation_volume_factor(pg)
    muw = water.viscosity(pw); muo = oil.viscosity(po); mug = gas.viscosity(pg)
    rhow = water.density(pw); rhoo = oil.density(po); rhog = gas.density(pg)
    depth = grid.depths
    kx, ky, kz = grid._permeability_components(permeability)

    div_w = np.zeros(grid.n_cells, dtype=float)
    div_o = np.zeros(grid.n_cells, dtype=float)
    div_g = np.zeros(grid.n_cells, dtype=float)

    for bc in boundaries.pressure_boundaries:
        cells, dist, area, normal = grid.boundary_cells(bc.side)
        if normal == "x":
            kcomp = kx[cells]
        elif normal == "y":
            kcomp = ky[cells]
        else:
            kcomp = kz[cells]
        tgeo = area * kcomp / dist
        p_bc = np.full(cells.size, float(bc.pressure), dtype=float)
        sw_bc = np.full(cells.size, float(bc.sw), dtype=float)
        sg_bc = np.full(cells.size, float(bc.sg), dtype=float)
        rs_bc = oil.solution_gas_ratio(p_bc) if bc.rs is None else np.full(cells.size, float(bc.rs))
        pwb, pob, pgb = phase_pressures_black_oil(p_bc, sw_bc, sg_bc, cap)
        # Boundary depth is currently cell-centre depth. This avoids imposing a
        # structural boundary model before corner-point support is introduced.
        dz = np.zeros(cells.size, dtype=float)

        def one_phase(p_cell, p_bound, rho_cell, rho_bound, kr_cell, mu_cell, b_cell, kr_bound, mu_bound, b_bound):
            dp = p_bound - p_cell
            rho_avg = 0.5 * (rho_cell + rho_bound)
            dpot_dir = dp - rho_avg * gravity * dz
            use_cell = dpot_dir <= 0.0
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
        fgc = np.where(oil_from_cell, rs[cells], rs_bc) * fo + fg
        div_w[cells] += fw
        div_o[cells] += fo
        div_g[cells] += fgc

    return div_w, div_o, div_g
