"""Photometric color calibration: gray-locus and Gaia-based white balance."""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

_log = logging.getLogger(__name__)


def _aperture_photometry(img: np.ndarray, star_positions,
                          aperture_r: int = 6,
                          sky_inner: int = 8,
                          sky_outer: int = 12) -> Optional[np.ndarray]:
    """Simple circular aperture photometry for R, G, B channels.

    For each star, integrates flux in a circle of radius ``aperture_r``
    after subtracting the median annulus sky value.

    Returns an (N, 3) array of flux measurements or None on failure.
    """
    if star_positions is None or len(star_positions) == 0:
        return None
    H, W = img.shape[:2]
    fluxes = []

    for star in star_positions:
        yc = int(round(float(star['ycentroid'])))
        xc = int(round(float(star['xcentroid'])))
        if (yc < sky_outer + 1 or yc >= H - sky_outer - 1 or
                xc < sky_outer + 1 or xc >= W - sky_outer - 1):
            continue

        yy, xx = np.mgrid[yc - sky_outer:yc + sky_outer + 1,
                           xc - sky_outer:xc + sky_outer + 1]
        r2 = (yy - yc) ** 2 + (xx - xc) ** 2
        ap_mask = r2 <= aperture_r ** 2
        sky_mask = (r2 >= sky_inner ** 2) & (r2 <= sky_outer ** 2)

        star_flux = []
        skip = False
        for c in range(3):
            patch = img[yc - sky_outer:yc + sky_outer + 1,
                        xc - sky_outer:xc + sky_outer + 1, c]
            sky_pix = patch[sky_mask]
            if sky_pix.size < 4:
                skip = True
                break
            sky_bg = float(np.median(sky_pix))
            ap_pix = patch[ap_mask] - sky_bg
            flux = float(ap_pix.sum())
            if flux <= 0:
                skip = True
                break
            star_flux.append(flux)
        if not skip and len(star_flux) == 3:
            fluxes.append(star_flux)

    return np.array(fluxes, dtype=np.float64) if fluxes else None


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

    Optionally queries the Gaia DR3 catalogue via ``astroquery`` for
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


def try_gaia_calibration(img: np.ndarray,
                          star_positions,
                          wcs=None,
                          verbose: bool = False
                          ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Attempt Gaia DR3-based photometric calibration (requires astroquery + WCS).

    Queries the Gaia DR3 catalogue for sources in the image field, matches
    them to detected stars, and uses the Gaia BP-RP colour index to predict
    expected R/G/B ratios for each star.  Computes the per-channel scale
    factor that minimises the residual between observed and predicted colours.

    Falls back to the gray-locus method if:
    - ``astroquery`` is not installed.
    - The image has no WCS information (no plate solve was run).
    - Fewer than 10 Gaia matches are found.

    Args:
        img:            Float32 stacked image (H, W, 3).
        star_positions: Source table from detect_stars_auto.
        wcs:            Astropy WCS object from plate solve (or None).
        verbose:        Verbosity flag.

    Returns:
        (calibrated_image, scales) — same convention as
        ``photometric_color_calibrate``.
    """
    if wcs is None:
        _log.debug("Gaia calibration: no WCS — falling back to gray locus")
        return photometric_color_calibrate(img, star_positions, verbose=verbose)

    try:
        from astroquery.gaia import Gaia
        from astropy.coordinates import SkyCoord
        import astropy.units as u
    except ImportError:
        _log.debug("Gaia calibration: astroquery not installed — falling back")
        return photometric_color_calibrate(img, star_positions, verbose=verbose)

    try:
        H, W = img.shape[:2]
        # Field centre and radius
        centre_sky = wcs.pixel_to_world(W / 2, H / 2)
        corner_sky = wcs.pixel_to_world(0.0, 0.0)
        radius_deg = float(centre_sky.separation(corner_sky).deg) * 1.1

        Gaia.ROW_LIMIT = 2000
        result = Gaia.cone_search_async(
            centre_sky, radius=u.Quantity(radius_deg, u.deg),
            columns=['source_id', 'ra', 'dec', 'phot_bp_mean_mag',
                     'phot_rp_mean_mag', 'phot_g_mean_mag']).get_results()

        if len(result) < 10:
            _log.debug("Gaia calibration: only %d sources — falling back", len(result))
            return photometric_color_calibrate(img, star_positions, verbose=verbose)

        # Match Gaia to detected stars via pixel coordinates
        gaia_sky = SkyCoord(ra=result['ra'], dec=result['dec'], unit='deg')
        gaia_pix = np.column_stack(wcs.world_to_pixel(gaia_sky))   # (N_gaia, 2)

        if star_positions is None or len(star_positions) == 0:
            return photometric_color_calibrate(img, star_positions, verbose=verbose)

        det_x = np.array([float(s['xcentroid']) for s in star_positions])
        det_y = np.array([float(s['ycentroid']) for s in star_positions])

        # Nearest-neighbour match
        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack([det_x, det_y]))
        dists, idx = tree.query(gaia_pix, k=1, distance_upper_bound=5.0)
        matched = dists < 5.0

        if matched.sum() < 10:
            _log.debug("Gaia calibration: too few matches (%d) — falling back",
                       matched.sum())
            return photometric_color_calibrate(img, star_positions, verbose=verbose)

        # Compute aperture photometry for matched stars
        matched_sources = star_positions[idx[matched]]
        fluxes = _aperture_photometry(img, matched_sources)
        if fluxes is None or len(fluxes) < 10:
            return photometric_color_calibrate(img, star_positions, verbose=verbose)

        # BP-RP to R/G/B mapping (approximate for broad-band DSLR/OSC filters)
        # Empirical transformation from Gaia to Johnson-Cousins:
        #   R - G_Gaia ≈ -0.049 + 0.278*(BP-RP)
        #   B - G_Gaia ≈ +0.033 - 0.565*(BP-RP)
        bp_rp = np.array(result['phot_bp_mean_mag'] - result['phot_rp_mean_mag'])[matched]
        bp_rp = bp_rp[~np.isnan(bp_rp) & (np.abs(bp_rp) < 5)]
        if len(bp_rp) < 10:
            return photometric_color_calibrate(img, star_positions, verbose=verbose)

        expected_r_g = -0.049 + 0.278 * bp_rp
        expected_b_g = 0.033 - 0.565 * bp_rp

        # Observed R/G and B/G
        obs_rg = fluxes[:len(bp_rp), 0] / (fluxes[:len(bp_rp), 1] + 1e-9)
        obs_bg = fluxes[:len(bp_rp), 2] / (fluxes[:len(bp_rp), 1] + 1e-9)
        # Convert Gaia magnitude differences to flux ratios
        exp_rg_flux = 10.0 ** (-0.4 * expected_r_g)
        exp_bg_flux = 10.0 ** (-0.4 * expected_b_g)

        scale_r = float(np.median(exp_rg_flux / (obs_rg + 1e-12)))
        scale_b = float(np.median(exp_bg_flux / (obs_bg + 1e-12)))
        scales = np.clip(np.array([scale_r, 1.0, scale_b]), 0.5, 2.0)

        if verbose:
            print(f"    Gaia calibration ({matched.sum()} matches): "
                  f"R×{scales[0]:.4f}  G×1.0000  B×{scales[2]:.4f}")

        result_img = img.astype(np.float64)
        for c in range(3):
            result_img[:, :, c] *= scales[c]
        return np.clip(result_img, 0.0, None).astype(np.float32), scales

    except Exception as exc:
        _log.warning("Gaia calibration failed (%s) — falling back to gray locus", exc)
        return photometric_color_calibrate(img, star_positions, verbose=verbose)
