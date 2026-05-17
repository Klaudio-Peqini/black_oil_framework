from __future__ import annotations

import numpy as np
from .grid import CartesianGrid1D


def divergence_from_face_flux(nx: int, face_flux: np.ndarray) -> np.ndarray:
    """Convert 1D interior face fluxes into finite-volume divergence.

    face_flux[k] is positive from cell k to cell k+1. The returned vector is the
    net outflow from each cell.
    """
    div = np.zeros(nx, dtype=float)
    if nx <= 1:
        return div
    div[:-1] += face_flux
    div[1:] -= face_flux
    return div


def single_phase_face_flux(
    grid: CartesianGrid1D,
    permeability: float | np.ndarray,
    p: np.ndarray,
    mu: float | np.ndarray,
    b: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Single-phase stock-tank-volume face flux.

    The flux is positive from left to right.
    """
    tgeo = grid.geometric_transmissibility(permeability)
    p = np.asarray(p, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if mu_arr.ndim == 0:
        mu_arr = np.full(grid.nx, float(mu_arr))
    if b_arr.ndim == 0:
        b_arr = np.full(grid.nx, float(b_arr))

    dp = p[1:] - p[:-1]
    # Upwind according to reservoir-volume flux direction. With no gravity,
    # positive flow left-to-right occurs when p_left > p_right.
    use_left = dp <= 0.0
    upwind = np.where(use_left, np.arange(grid.nx - 1), np.arange(1, grid.nx))
    mobility = 1.0 / (mu_arr[upwind] * b_arr[upwind])
    return -tgeo * mobility * dp


def two_phase_face_fluxes(
    grid: CartesianGrid1D,
    permeability: float | np.ndarray,
    p: np.ndarray,
    sw: np.ndarray,
    relperm,
    water,
    oil,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute water and oil component fluxes at interior faces.

    The phase pressure is assumed common in this first implementation stage:
    no gravity and no capillary pressure.
    """
    tgeo = grid.geometric_transmissibility(permeability)
    p = np.asarray(p, dtype=float)
    sw = np.asarray(sw, dtype=float)

    bw = water.formation_volume_factor(p)
    bo = oil.formation_volume_factor(p)
    muw = water.viscosity(p)
    muo = oil.viscosity(p)
    krw = relperm.krw(sw)
    kro = relperm.kro(sw)

    dp = p[1:] - p[:-1]
    use_left = dp <= 0.0
    upwind = np.where(use_left, np.arange(grid.nx - 1), np.arange(1, grid.nx))

    mob_w = krw[upwind] / (muw[upwind] * bw[upwind])
    mob_o = kro[upwind] / (muo[upwind] * bo[upwind])

    fw = -tgeo * mob_w * dp
    fo = -tgeo * mob_o * dp
    return fw, fo


def three_phase_black_oil_face_fluxes(
    grid: CartesianGrid1D,
    permeability: float | np.ndarray,
    p: np.ndarray,
    sw: np.ndarray,
    sg: np.ndarray,
    relperm,
    water,
    oil,
    gas,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute black-oil component fluxes at interior faces.

    The implementation is Step 4A: no gravity, no capillary pressure, common
    oil/water/gas pressure, and first-order upwind mobilities. The returned
    arrays are stock-tank component fluxes, positive from cell i to i+1:

    * water component flux: u_w / B_w
    * oil component flux:   u_o / B_o
    * free-gas phase flux:  u_g / B_g
    * gas component flux:   Rs_up * u_o / B_o + u_g / B_g
    """
    tgeo = grid.geometric_transmissibility(permeability)
    p = np.asarray(p, dtype=float)
    sw = np.asarray(sw, dtype=float)
    sg = np.asarray(sg, dtype=float)

    bw = water.formation_volume_factor(p)
    bo = oil.formation_volume_factor(p)
    bg = gas.formation_volume_factor(p)
    muw = water.viscosity(p)
    muo = oil.viscosity(p)
    mug = gas.viscosity(p)
    rs = oil.solution_gas_ratio(p)

    krw = relperm.krw(sw, sg)
    kro = relperm.kro(sw, sg)
    krg = relperm.krg(sw, sg)

    dp = p[1:] - p[:-1]
    use_left = dp <= 0.0
    upwind = np.where(use_left, np.arange(grid.nx - 1), np.arange(1, grid.nx))

    mob_w = krw[upwind] / (muw[upwind] * bw[upwind])
    mob_o = kro[upwind] / (muo[upwind] * bo[upwind])
    mob_g = krg[upwind] / (mug[upwind] * bg[upwind])

    fw = -tgeo * mob_w * dp
    fo = -tgeo * mob_o * dp
    fg_free = -tgeo * mob_g * dp
    fg_component = rs[upwind] * fo + fg_free
    return fw, fo, fg_free, fg_component


def three_phase_black_oil_face_fluxes_with_rs(
    grid: CartesianGrid1D,
    permeability: float | np.ndarray,
    p: np.ndarray,
    sw: np.ndarray,
    sg: np.ndarray,
    rs: np.ndarray,
    relperm,
    water,
    oil,
    gas,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Three-phase black-oil face fluxes using an explicitly supplied Rs field.

    Step 4A used Rs_sat(p) everywhere. Step 4B needs a cell-wise Rs field
    because undersaturated cells use Rs as a primary variable, while saturated
    cells still use Rs_sat(p). The supplied ``rs`` array is therefore upwinded
    consistently with the oil flux.
    """
    tgeo = grid.geometric_transmissibility(permeability)
    p = np.asarray(p, dtype=float)
    sw = np.asarray(sw, dtype=float)
    sg = np.asarray(sg, dtype=float)
    rs = np.asarray(rs, dtype=float)

    bw = water.formation_volume_factor(p)
    bo = oil.formation_volume_factor(p)
    bg = gas.formation_volume_factor(p)
    muw = water.viscosity(p)
    muo = oil.viscosity(p)
    mug = gas.viscosity(p)

    krw = relperm.krw(sw, sg)
    kro = relperm.kro(sw, sg)
    krg = relperm.krg(sw, sg)

    dp = p[1:] - p[:-1]
    use_left = dp <= 0.0
    upwind = np.where(use_left, np.arange(grid.nx - 1), np.arange(1, grid.nx))

    mob_w = krw[upwind] / (muw[upwind] * bw[upwind])
    mob_o = kro[upwind] / (muo[upwind] * bo[upwind])
    mob_g = krg[upwind] / (mug[upwind] * bg[upwind])

    fw = -tgeo * mob_w * dp
    fo = -tgeo * mob_o * dp
    fg_free = -tgeo * mob_g * dp
    fg_component = rs[upwind] * fo + fg_free
    return fw, fo, fg_free, fg_component



def phase_pressures_black_oil(
    p_o: np.ndarray,
    sw: np.ndarray,
    sg: np.ndarray,
    capillary=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return water, oil, and gas phase pressures from primary oil pressure.

    The capillary convention is

        pcow = p_o - p_w,
        pcgo = p_g - p_o.

    Therefore p_w = p_o - pcow(Sw) and p_g = p_o + pcgo(Sg).
    """
    from .capillary import ZeroCapillaryPressure

    cap = ZeroCapillaryPressure() if capillary is None else capillary
    po = np.asarray(p_o, dtype=float)
    sw_arr = np.asarray(sw, dtype=float)
    sg_arr = np.asarray(sg, dtype=float)
    pw = po - cap.pcow(sw_arr)
    pg = po + cap.pcgo(sg_arr)
    return pw, po, pg


def _phase_flux_with_potential(
    grid: CartesianGrid1D,
    permeability: float | np.ndarray,
    phase_pressure: np.ndarray,
    density: np.ndarray,
    kr: np.ndarray,
    mu: np.ndarray,
    b: np.ndarray,
    gravity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return face fluxes, upwind indices, and potential differences.

    Depth is positive downward. A hydrostatic profile with
    dp/d(depth)=rho*g therefore gives zero potential difference.
    """
    tgeo = grid.geometric_transmissibility(permeability)
    p = np.asarray(phase_pressure, dtype=float)
    rho = np.asarray(density, dtype=float)
    kr = np.asarray(kr, dtype=float)
    mu = np.asarray(mu, dtype=float)
    b = np.asarray(b, dtype=float)
    depth = grid.depths
    dz = depth[1:] - depth[:-1]
    dp = p[1:] - p[:-1]

    rho_avg = 0.5 * (rho[:-1] + rho[1:])
    dpot_for_direction = dp - rho_avg * gravity * dz
    use_left = dpot_for_direction <= 0.0
    upwind = np.where(use_left, np.arange(grid.nx - 1), np.arange(1, grid.nx))

    dpot = dp - rho[upwind] * gravity * dz
    mobility = kr[upwind] / (mu[upwind] * b[upwind])
    flux = -tgeo * mobility * dpot
    return flux, upwind, dpot


def three_phase_black_oil_face_fluxes_with_rs_gravity_capillary(
    grid: CartesianGrid1D,
    permeability: float | np.ndarray,
    p: np.ndarray,
    sw: np.ndarray,
    sg: np.ndarray,
    rs: np.ndarray,
    relperm,
    water,
    oil,
    gas,
    capillary=None,
    gravity: float = 9.80665,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Three-phase black-oil fluxes with gravity and capillary pressure.

    Returns stock-tank component fluxes, positive from cell i to i+1:

    * water component flux: u_w / B_w(p_w)
    * oil component flux:   u_o / B_o(p_o)
    * free-gas phase flux:  u_g / B_g(p_g)
    * gas component flux:   Rs_up_o * u_o / B_o + u_g / B_g

    Upwinding is phase-potential based. Rs is upwinded with the oil-phase flux.
    """
    p = np.asarray(p, dtype=float)
    sw = np.asarray(sw, dtype=float)
    sg = np.asarray(sg, dtype=float)
    rs = np.asarray(rs, dtype=float)

    pw, po, pg = phase_pressures_black_oil(p, sw, sg, capillary)

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

    fw, _up_w, _dpot_w = _phase_flux_with_potential(
        grid, permeability, pw, rhow, krw, muw, bw, gravity
    )
    fo, up_o, _dpot_o = _phase_flux_with_potential(
        grid, permeability, po, rhoo, kro, muo, bo, gravity
    )
    fg_free, _up_g, _dpot_g = _phase_flux_with_potential(
        grid, permeability, pg, rhog, krg, mug, bg, gravity
    )
    fg_component = rs[up_o] * fo + fg_free
    return fw, fo, fg_free, fg_component
