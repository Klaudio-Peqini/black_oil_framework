from __future__ import annotations

import numpy as np

from .state import State1P, State2P, State3P
from .wells import RateWell, BHPWell
from .flux import (
    single_phase_face_flux,
    two_phase_face_fluxes,
    three_phase_black_oil_face_fluxes,
    divergence_from_face_flux,
)


def residual_single_phase(x: np.ndarray, old: State1P, dt: float, grid, rock, fluid, wells=None) -> np.ndarray:
    """Fully implicit residual for single-phase slightly compressible flow."""
    wells = wells or []
    p = np.asarray(x, dtype=float)
    v = grid.volumes

    phi_new = rock.porosity(p)
    phi_old = rock.porosity(old.p)
    b_new = fluid.formation_volume_factor(p)
    b_old = fluid.formation_volume_factor(old.p)
    mu_new = fluid.viscosity(p)

    accum = v * (phi_new / b_new - phi_old / b_old) / dt
    face_flux = single_phase_face_flux(grid, rock.permeability, p, mu_new, b_new)
    div = divergence_from_face_flux(grid.nx, face_flux)

    q = np.zeros(grid.nx, dtype=float)
    for well in wells:
        if isinstance(well, RateWell):
            q += well.single_phase_source(grid.nx)
        elif isinstance(well, BHPWell):
            q += well.single_phase_source(p, mu_new, b_new)
        else:
            raise TypeError(f"Unsupported well type: {type(well)!r}")

    return accum + div - q


def unpack_two_phase(x: np.ndarray, nx: int) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(x[:nx], dtype=float)
    sw = np.asarray(x[nx:], dtype=float)
    return p, sw


def pack_two_phase(p: np.ndarray, sw: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(p, dtype=float), np.asarray(sw, dtype=float)])


def residual_two_phase_oil_water(
    x: np.ndarray,
    old: State2P,
    dt: float,
    grid,
    rock,
    water,
    oil,
    relperm,
    wells=None,
) -> np.ndarray:
    """Fully implicit residual for two-phase oil-water flow.

    Primary variables: p_o and S_w.

    Equations:
        d/dt(phi Sw/Bw) + div(uw/Bw) = qw
        d/dt(phi So/Bo) + div(uo/Bo) = qo
    """
    wells = wells or []
    p, sw = unpack_two_phase(x, grid.nx)
    so = 1.0 - sw
    v = grid.volumes

    phi_new = rock.porosity(p)
    phi_old = rock.porosity(old.p)

    bw_new = water.formation_volume_factor(p)
    bo_new = oil.formation_volume_factor(p)
    bw_old = water.formation_volume_factor(old.p)
    bo_old = oil.formation_volume_factor(old.p)

    acc_w = v * (phi_new * sw / bw_new - phi_old * old.sw / bw_old) / dt
    acc_o = v * (phi_new * so / bo_new - phi_old * old.so / bo_old) / dt

    fw, fo = two_phase_face_fluxes(grid, rock.permeability, p, sw, relperm, water, oil)
    div_w = divergence_from_face_flux(grid.nx, fw)
    div_o = divergence_from_face_flux(grid.nx, fo)

    qw = np.zeros(grid.nx, dtype=float)
    qo = np.zeros(grid.nx, dtype=float)

    krw = relperm.krw(sw)
    kro = relperm.kro(sw)
    muw = water.viscosity(p)
    muo = oil.viscosity(p)

    for well in wells:
        if isinstance(well, RateWell):
            qwi, qoi = well.two_phase_sources(grid.nx)
        elif isinstance(well, BHPWell):
            qwi, qoi = well.two_phase_sources(p, sw, krw, kro, muw, muo, bw_new, bo_new)
        else:
            raise TypeError(f"Unsupported well type: {type(well)!r}")
        qw += qwi
        qo += qoi

    rw = acc_w + div_w - qw
    ro = acc_o + div_o - qo
    return np.concatenate([rw, ro])


def unpack_three_phase(x: np.ndarray, nx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(x[:nx], dtype=float)
    sw = np.asarray(x[nx:2 * nx], dtype=float)
    sg = np.asarray(x[2 * nx:], dtype=float)
    return p, sw, sg


def pack_three_phase(p: np.ndarray, sw: np.ndarray, sg: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(p, dtype=float), np.asarray(sw, dtype=float), np.asarray(sg, dtype=float)])


def residual_live_oil_saturated(
    x: np.ndarray,
    old: State3P,
    dt: float,
    grid,
    rock,
    water,
    oil,
    gas,
    relperm,
    wells=None,
) -> np.ndarray:
    """Fully implicit saturated live-oil black-oil residual.

    Primary variables are oil pressure, water saturation, and free-gas
    saturation. The saturation state is assumed saturated with a mobile/free gas
    phase present; phase appearance/disappearance is deliberately postponed to
    the next development step.

    Component equations:
        d/dt(phi Sw/Bw) + div(uw/Bw) = qw
        d/dt(phi So/Bo) + div(uo/Bo) = qo
        d/dt(phi [Rs So/Bo + Sg/Bg])
            + div(Rs uo/Bo + ug/Bg) = qg_component
    """
    wells = wells or []
    p, sw, sg = unpack_three_phase(x, grid.nx)
    so = 1.0 - sw - sg
    v = grid.volumes

    phi_new = rock.porosity(p)
    phi_old = rock.porosity(old.p)

    bw_new = water.formation_volume_factor(p)
    bo_new = oil.formation_volume_factor(p)
    bg_new = gas.formation_volume_factor(p)
    bw_old = water.formation_volume_factor(old.p)
    bo_old = oil.formation_volume_factor(old.p)
    bg_old = gas.formation_volume_factor(old.p)

    rs_new = oil.solution_gas_ratio(p)
    rs_old = oil.solution_gas_ratio(old.p)

    acc_w = v * (phi_new * sw / bw_new - phi_old * old.sw / bw_old) / dt
    acc_o = v * (phi_new * so / bo_new - phi_old * old.so / bo_old) / dt
    acc_g = v * (
        phi_new * (rs_new * so / bo_new + sg / bg_new)
        - phi_old * (rs_old * old.so / bo_old + old.sg / bg_old)
    ) / dt

    fw, fo, _fg_free, fg_comp = three_phase_black_oil_face_fluxes(
        grid, rock.permeability, p, sw, sg, relperm, water, oil, gas
    )
    div_w = divergence_from_face_flux(grid.nx, fw)
    div_o = divergence_from_face_flux(grid.nx, fo)
    div_g = divergence_from_face_flux(grid.nx, fg_comp)

    qw = np.zeros(grid.nx, dtype=float)
    qo = np.zeros(grid.nx, dtype=float)
    qg_free = np.zeros(grid.nx, dtype=float)

    krw = relperm.krw(sw, sg)
    kro = relperm.kro(sw, sg)
    krg = relperm.krg(sw, sg)
    muw = water.viscosity(p)
    muo = oil.viscosity(p)
    mug = gas.viscosity(p)

    for well in wells:
        if isinstance(well, RateWell):
            qwi, qoi, qgi = well.three_phase_sources(grid.nx)
        elif isinstance(well, BHPWell):
            qwi, qoi, qgi = well.three_phase_sources(
                p, sw, sg, krw, kro, krg, muw, muo, mug, bw_new, bo_new, bg_new
            )
        else:
            raise TypeError(f"Unsupported well type: {type(well)!r}")
        qw += qwi
        qo += qoi
        qg_free += qgi

    # The gas component source contains free-gas flow plus gas dissolved in oil.
    qg_component = qg_free + rs_new * qo

    rw = acc_w + div_w - qw
    ro = acc_o + div_o - qo
    rg = acc_g + div_g - qg_component
    return np.concatenate([rw, ro, rg])
