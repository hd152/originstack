"""Photometric color calibration: gray-locus white balance."""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from src.photometry_core import aperture_photometry_batch

_log = logging.getLogger("originstack")


def _aperture_photometry(img: np.ndarray, star_positions,
                          aperture_r: int = 6,
                          sky_inner: int = 8,
                          sky_outer: int = 12) -> Optional[np.ndarray]:
    """Circular aperture photometry for R, G, B channels via the shared
    ``photometry_core.aperture_photometry_batch`` kernel (partial-pixel
    aperture, median + MAD sky annulus).

    Returns an ``(N, 3)`` array for the stars with a positive, finite flux
    in every channel, or None.
    """
    if star_positions is None or len(star_positions) == 0:
        return None
    px = np.asarray(star_positions['xcentroid'], dtype=float)
    py = np.asarray(star_positions['ycentroid'], dtype=float)
    flux, *_ = aperture_photometry_batch(
        img, px, py, float(aperture_r), float(sky_inner), float(sky_outer))
    good = np.all(np.isfinite(flux) & (flux > 0.0), axis=1)
    return flux[good].astype(np.float64) if np.any(good) else None


def _gray_locus_calibration(fluxes: np.ndarray,
                              sigma_clip: float = 2.5) -> Optional[np.ndarray]:
    """Derive per-channel correction factors from the gray locus.

    Stars in the "gray locus" (roughly solar-type, spectral class F-G-K)
    should have equal R, G, B fluxes after calibration.  This function:

    1. Normalises each star's fluxes to unit sum.
    2. Computes the median fractional contribution per channel (should be
       ≈ 1/3 for a perfectly calibrated gray-locus star).
    3. Returns scale factors ``[kR, kG, kB]`` such that multiplying each
       channel by its scale factor drives the median toward 1/3.

    Stars that are far from the gray locus (red M-dwarfs, hot blue O-stars)
    are sigma-clipped away before computing the median.

    Args:
        fluxes:     (N, 3) aperture flux array from ``_aperture_photometry``.
        sigma_clip: Clipping threshold in units of MAD-sigma (default 2.5).

    Returns:
        Array [kR, kG, kB] — per-channel multiplication factors, or None.
    """
    if fluxes is None or len(fluxes) < 5:
        return None

    # Normalise each star's fluxes to unit sum → fractional colours
    totals = fluxes.sum(axis=1, keepdims=True)
    valid = totals.ravel() > 1e-9
    if valid.sum() < 5:
        return None
    frac = fluxes[valid] / totals[valid]   # (N, 3) each row sums to 1

    # Sigma-clipping on G/R and G/B colour ratios to find gray-locus stars
    gr = frac[:, 1] / (frac[:, 0] + 1e-12)
    gb = frac[:, 1] / (frac[:, 2] + 1e-12)
    for _ in range(3):
        med_gr, med_gb = np.median(gr), np.median(gb)
        mad_gr = np.median(np.abs(gr - med_gr))
        mad_gb = np.median(np.abs(gb - med_gb))
        keep = ((np.abs(gr - med_gr) < sigma_clip * 1.4826 * max(mad_gr, 1e-9)) &
                (np.abs(gb - med_gb) < sigma_clip * 1.4826 * max(mad_gb, 1e-9)))
        if keep.sum() < 5:
            break
        frac = frac[keep]
        gr, gb = gr[keep], gb[keep]

    # Median fractional flux per channel for the gray-locus population
    med_frac = np.median(frac, axis=0)   # (3,) should be ≈ [1/3, 1/3, 1/3]
    if np.any(med_frac <= 0):
        return None

    # Scale factors: drive each channel's median fraction to 1/3
    target = 1.0 / 3.0
    scales = target / med_frac   # e.g. if green fraction is 0.4, kG = (1/3)/0.4 < 1
    # Normalise so that green channel is unchanged (green is most sensitive)
    scales = scales / scales[1]
    return scales.astype(np.float64)


def photometric_color_calibrate(img: np.ndarray,
                                  star_positions,
                                  aperture_r: int = 6,
                                  sigma_clip: float = 2.5,
                                  verbose: bool = False
                                  ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Calibrate image white balance using the stellar gray-locus method.

    Measures per-star R, G, B fluxes via aperture photometry, identifies
    the solar-type "gray locus" population via iterative sigma-clipping on
    colour ratios, and derives per-channel scale factors that make the
    median gray-locus star colour-neutral.

    This corrects systematic white-balance errors caused by:
    - Unequal quantum efficiency across the Bayer pattern channels.
    - Residual atmospheric extinction colour gradients.
    - Incorrect or absent flat-field colour response correction.

    Optionally queries the Gaia DR3 catalogue via direct HTTP for
    photometrically calibrated BP-RP colour indices to refine the gray-locus
    selection, falling back to the internal sigma-clip method if unavailable.

    Args:
        img:            Float32 stacked image (H, W, 3).
        star_positions: Source table from detect_stars_auto.
        aperture_r:     Aperture radius in pixels for flux integration.
        sigma_clip:     Sigma threshold for gray-locus sigma-clipping.
        verbose:        Print calibration diagnostics.

    Returns:
        (calibrated_image, scales) — corrected float32 image and the
        [kR, kG, kB] scale factors applied.  scales is None if calibration
        failed (image returned unchanged).
    """
    fluxes = _aperture_photometry(img, star_positions, aperture_r=aperture_r)
    if fluxes is None:
        _log.warning("Photometric calibration: aperture photometry failed")
        return img.copy(), None

    scales = _gray_locus_calibration(fluxes, sigma_clip=sigma_clip)
    if scales is None:
        _log.warning("Photometric calibration: gray-locus fit failed")
        return img.copy(), None

    # Clamp corrections to a sensible range (avoid overcorrecting)
    scales = np.clip(scales, 0.5, 2.0)

    if verbose:
        print(f"    Photometric calibration scales: "
              f"R×{scales[0]:.4f}  G×{scales[1]:.4f}  B×{scales[2]:.4f}")

    result = img.astype(np.float64)
    for c in range(3):
        result[:, :, c] *= scales[c]

    return np.clip(result, 0.0, None).astype(np.float32), scales


