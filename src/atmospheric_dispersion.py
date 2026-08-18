"""Software atmospheric dispersion correction (--fix-atmospheric-dispersion).

Every real ADC (atmospheric dispersion corrector) found in the research
pass behind this feature is a physical rotating-double-prism device -- no
established post-capture software equivalent exists to port, unlike this
codebase's other additions. This is derived from first principles (the
classical refraction formula) rather than translating a known reference
algorithm, which is a genuinely different, higher-risk kind of work than
the rest of this codebase's ports: flagged as experimental, opt-in, and
NOT wired into --auto.

Physics: atmospheric refraction bends starlight toward the zenith by an
angle depending on wavelength (air's refractive index is chromatic), so a
star's blue and red images land at very slightly different positions on
the detector -- worse at low altitude (high zenith angle). This corrects
it by shifting each colour channel back toward its reference wavelength's
position by the calculated angular separation.

Scope, stated plainly: the refractive-index model here is Filippenko
(1982)'s parameterization of the classical Cauchy-type dispersion formula
for STANDARD atmospheric conditions (T=15C, P=760mmHg, dry air) -- it does
not correct for the actual site pressure/temperature/humidity on the
observation night the way real ADC hardware (or a full Edlen/Ciddor model)
would. The zenith angle and parallactic angle (the on-detector direction
of "toward zenith", which depends on site latitude, target RA/Dec, and
observation time) are required CLI parameters here, not auto-derived --
getting that geometry wrong would shift colour channels in the wrong
direction, actively worsening the image, so this only ever acts on
explicitly-supplied values rather than silently guessing.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _refractive_index_air_minus_one(wavelength_nm) -> np.ndarray:
    """``n - 1`` for standard air, Filippenko (1982)'s parameterization of
    the classical Cauchy-type dispersion formula. Sanity-checked (not just
    trusted from memory) against the well-known ballpark fact that air's
    refractive index at visible wavelengths is approximately 1.00027 --
    see ``tests/test_atmospheric_dispersion.py``.
    """
    um = np.asarray(wavelength_nm, dtype=np.float64) / 1000.0
    inv_um2 = 1.0 / (um ** 2)
    n_minus_1_e6 = 64.328 + 29498.1 / (146.0 - inv_um2) + 255.4 / (41.0 - inv_um2)
    return n_minus_1_e6 * 1e-6


def differential_refraction_arcsec(wavelength_nm: float, reference_nm: float,
                                   zenith_angle_deg: float) -> float:
    """Angular separation (arcsec) atmospheric refraction introduces
    between ``wavelength_nm`` and ``reference_nm`` at the given zenith
    angle, via the classical low-order refraction formula ``R = (n-1) *
    tan(z)``. Valid away from the horizon (refraction -- and this
    approximation -- diverges as zenith angle approaches 90deg); not
    meaningful much past ~70deg.
    """
    n1 = float(_refractive_index_air_minus_one(np.array([wavelength_nm]))[0])
    n2 = float(_refractive_index_air_minus_one(np.array([reference_nm]))[0])
    z_rad = np.deg2rad(zenith_angle_deg)
    delta_rad = (n1 - n2) * np.tan(z_rad)
    return float(np.degrees(delta_rad) * 3600.0)


def correct_atmospheric_dispersion(
    img: np.ndarray, plate_scale_arcsec_px: float, zenith_angle_deg: float,
    parallactic_angle_deg: float,
    wavelengths_nm: Tuple[float, float, float] = (650.0, 550.0, 450.0),
    reference_index: int = 1,
) -> np.ndarray:
    """Shift each RGB channel of ``img`` back toward its reference
    wavelength's position to undo atmospheric dispersion.

    ``wavelengths_nm``: effective wavelength of the R/G/B channels
    (defaults are broad-band OSC-ish estimates -- pass real filter/QE
    effective wavelengths for a narrowband or filtered setup).
    ``reference_index``: which channel (0=R, 1=G, 2=B) is treated as the
    non-shifted reference; default green (the least displaced of the
    three under normal atmospheric dispersion, since it sits between red
    and blue).
    ``parallactic_angle_deg``: on-detector direction of "toward zenith"
    (0 = up/+y, 90 = right/+x, standard image-plane convention) -- this
    depends on site latitude, target RA/Dec, and observation time; this
    function does not derive it, the caller must supply it.

    Returns a float32 array the same shape as ``img``.
    """
    from scipy.ndimage import shift as _ndi_shift

    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError("correct_atmospheric_dispersion expects an (H, W, 3) image")

    ref_wl = wavelengths_nm[reference_index]
    theta = np.deg2rad(parallactic_angle_deg)
    # Image-plane unit vector "toward zenith": (dy, dx) with the stated
    # convention (0deg = +y/up, 90deg = +x/right).
    dir_y, dir_x = np.cos(theta), np.sin(theta)

    out = np.empty_like(img, dtype=np.float32)
    for c in range(3):
        if c == reference_index:
            out[:, :, c] = img[:, :, c]
            continue
        sep_arcsec = differential_refraction_arcsec(
            wavelengths_nm[c], ref_wl, zenith_angle_deg)
        sep_px = sep_arcsec / max(plate_scale_arcsec_px, 1e-9)
        # Shift this channel by -separation along the zenith direction to
        # move it back onto the reference channel's position (refraction
        # displaces shorter wavelengths further toward zenith than longer
        # ones, so undoing it moves each channel the opposite way).
        shift_vec = (-sep_px * dir_y, -sep_px * dir_x)
        out[:, :, c] = _ndi_shift(img[:, :, c].astype(np.float64), shift_vec,
                                  order=3, mode='nearest')

    return out
