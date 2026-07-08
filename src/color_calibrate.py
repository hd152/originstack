"""Photometric colour calibration via Gaia/2MASS star colours.

Requires:
  * The stacked image to have been plate-solved (WCS keywords in header).
  * astroquery  (``pip install astroquery``)
  * astropy     (already a core dependency)

Workflow
--------
1. Parse WCS from the FITS header.
2. Query the Gaia DR3 source catalogue for stars within the field.
3. Extract per-channel instrumental fluxes for each Gaia star via aperture
   photometry on the stacked image.
4. Compute a linear scale factor for each channel such that the per-star
   colour ratios match the Gaia G_BP − G_RP → B−V relationship.
5. Apply the scale factors multiplicatively to the image.

If plate solving failed, astroquery is unavailable, or too few stars are
matched, a warning is printed and the image is returned unchanged.
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    HAS_ASTROPY_WCS = True
except Exception:
    HAS_ASTROPY_WCS = False

try:
    from astropy.stats import sigma_clipped_stats
    HAS_SIGMA_CLIP = True
except Exception:
    HAS_SIGMA_CLIP = False

try:
    from astroquery.gaia import Gaia
    HAS_GAIA = True
except Exception:
    HAS_GAIA = False

try:
    from astroquery.vizier import Vizier
    HAS_VIZIER = True
except Exception:
    HAS_VIZIER = False


# ---------------------------------------------------------------------------
# Catalog query
# ---------------------------------------------------------------------------

def _field_radius_deg(header) -> float:
    """Estimate the field-of-view radius in degrees from WCS."""
    try:
        wcs = WCS(header)
        naxis1 = int(header.get("NAXIS1", 0))
        naxis2 = int(header.get("NAXIS2", 0))
        if naxis1 == 0 or naxis2 == 0:
            return 0.5
        corners = wcs.all_pix2world(
            [[0, 0], [naxis1, 0], [0, naxis2], [naxis1, naxis2]], 0
        )
        ra_c, dec_c = float(header.get("CRVAL1", 0)), float(header.get("CRVAL2", 0))
        dists = [
            np.sqrt((c[0] - ra_c) ** 2 + (c[1] - dec_c) ** 2)
            for c in corners
        ]
        return float(np.max(dists)) * 1.1  # 10 % margin
    except Exception:
        return 0.5


def query_gaia_stars(header, max_stars: int = 500):
    """Query Gaia DR3 for stars within the image field.

    Returns an astropy Table with columns: ra, dec, phot_g_mean_mag,
    phot_bp_mean_mag, phot_rp_mean_mag.  Returns None on failure.
    """
    if not HAS_GAIA or not HAS_ASTROPY_WCS:
        return None
    if "CRVAL1" not in header or "CRVAL2" not in header:
        return None

    ra  = float(header["CRVAL1"])
    dec = float(header["CRVAL2"])
    radius_deg = _field_radius_deg(header)

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    radius = u.Quantity(radius_deg, unit=u.deg)

    try:
        Gaia.ROW_LIMIT = max_stars
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            job = Gaia.cone_search_async(
                coord, radius,
                columns=["ra", "dec",
                         "phot_g_mean_mag",
                         "phot_bp_mean_mag",
                         "phot_rp_mean_mag"],
                verbose=False,
            )
            table = job.get_results()
        if table is None or len(table) == 0:
            return None
        # Filter out stars without colour info
        mask = (table["phot_bp_mean_mag"].mask == False if hasattr(table["phot_bp_mean_mag"], "mask")
                else np.ones(len(table), dtype=bool))
        table = table[mask]
        return table if len(table) >= 10 else None
    except Exception:
        return None


def query_2mass_stars(header, max_stars: int = 500):
    """Fallback: query 2MASS PSC via VizieR for J/H/K magnitudes."""
    if not HAS_VIZIER or not HAS_ASTROPY_WCS:
        return None
    if "CRVAL1" not in header or "CRVAL2" not in header:
        return None

    ra  = float(header["CRVAL1"])
    dec = float(header["CRVAL2"])
    radius_deg = _field_radius_deg(header)

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

    try:
        viz = Vizier(columns=["RAJ2000", "DEJ2000", "Jmag", "Hmag", "Kmag"],
                     row_limit=max_stars)
        result = viz.query_region(coord, radius=radius_deg * u.deg,
                                  catalog="II/246/out")
        if not result or len(result) == 0:
            return None
        table = result[0]
        return table if len(table) >= 10 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Aperture photometry
# ---------------------------------------------------------------------------

def _pixel_coords(table, header) -> Optional[np.ndarray]:
    """Convert catalogue RA/Dec to image pixel coordinates.

    Returns (N, 2) array of (x, y) pixel coordinates, or None.
    """
    if not HAS_ASTROPY_WCS:
        return None
    try:
        wcs = WCS(header)
        if "ra" in table.colnames:
            ra  = np.array(table["ra"],  dtype=float)
            dec = np.array(table["dec"], dtype=float)
        else:
            ra  = np.array(table["RAJ2000"], dtype=float)
            dec = np.array(table["DEJ2000"], dtype=float)
        coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        x, y = wcs.all_world2pix(ra, dec, 0)
        return np.column_stack([x, y])
    except Exception:
        return None


def _aperture_flux(img: np.ndarray, px: np.ndarray, py: np.ndarray,
                   radius: int = 5, sky_annulus: int = 3) -> np.ndarray:
    """Simple circular aperture photometry.

    Returns (N, 3) array of per-star, per-channel background-subtracted flux.
    Stars outside the image or with negative sky are excluded (set to NaN).
    """
    H, W = img.shape[:2]
    C = img.shape[2] if img.ndim == 3 else 1
    fluxes = np.full((len(px), C), np.nan, dtype=np.float64)

    r_in  = radius
    r_out = radius + sky_annulus

    yy, xx = np.mgrid[-r_out:r_out + 1, -r_out:r_out + 1]
    ap_mask  = xx ** 2 + yy ** 2 <= r_in  ** 2
    sky_mask = (xx ** 2 + yy ** 2 >  r_in  ** 2) & \
               (xx ** 2 + yy ** 2 <= r_out ** 2)

    for i, (cx, cy) in enumerate(zip(px, py)):
        cx_int = int(round(cx))
        cy_int = int(round(cy))
        y0 = cy_int - r_out
        y1 = cy_int + r_out + 1
        x0 = cx_int - r_out
        x1 = cx_int + r_out + 1
        if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
            continue
        patch = img[y0:y1, x0:x1]  # (2*r_out+1, 2*r_out+1, C)
        if patch.shape[:2] != (2 * r_out + 1, 2 * r_out + 1):
            continue
        for c in range(C):
            ch = patch[:, :, c] if img.ndim == 3 else patch
            sky_vals = ch[sky_mask]
            if HAS_SIGMA_CLIP and len(sky_vals) > 5:
                try:
                    _, sky_med, _ = sigma_clipped_stats(sky_vals, sigma=3.0)
                    sky_med = float(sky_med)
                except Exception:
                    sky_med = float(np.median(sky_vals))
            else:
                sky_med = float(np.median(sky_vals))
            flux = float(np.sum(ch[ap_mask])) - sky_med * int(ap_mask.sum())
            fluxes[i, c] = max(flux, 0.0)

    return fluxes


# ---------------------------------------------------------------------------
# Scale-factor fitting
# ---------------------------------------------------------------------------

def _bp_rp_to_bv(bp_rp: np.ndarray) -> np.ndarray:
    """Approximate conversion: Gaia BP-RP → Johnson B-V.

    Polynomial fit to Jordi et al. 2010 Table 3.
    Valid roughly for BP-RP in [0.0, 3.0].
    """
    return 0.0895 + 0.5289 * bp_rp - 0.0991 * bp_rp ** 2


def fit_channel_scales(img: np.ndarray, header,
                       catalog,
                       catalog_type: str = "gaia",
                       verbose: bool = False) -> Tuple[float, float, float]:
    """Fit per-channel multiplicative scale factors to match catalogue colours.

    Returns (scale_R, scale_G, scale_B).  Values close to 1.0 indicate the
    channel is already well-calibrated.  Returns (1.0, 1.0, 1.0) on failure.
    """
    pixel_coords = _pixel_coords(catalog, header)
    if pixel_coords is None:
        return 1.0, 1.0, 1.0

    px = pixel_coords[:, 0]
    py = pixel_coords[:, 1]

    # Use aperture radius scaled to image size
    ap_radius = max(3, min(8, int(min(img.shape[:2]) / 200)))
    fluxes = _aperture_flux(img, px, py, radius=ap_radius)

    valid = np.all(np.isfinite(fluxes) & (fluxes > 0), axis=1)
    if valid.sum() < 10:
        if verbose:
            print(f"  [colour cal] Too few valid stars ({valid.sum()}) — skipping")
        return 1.0, 1.0, 1.0

    fluxes = fluxes[valid]

    # Expected colour ratios from catalogue
    if catalog_type == "gaia":
        try:
            bp = np.array(catalog["phot_bp_mean_mag"][valid], dtype=float)
            rp = np.array(catalog["phot_rp_mean_mag"][valid], dtype=float)
            g  = np.array(catalog["phot_g_mean_mag"][valid], dtype=float)
        except Exception:
            return 1.0, 1.0, 1.0
        bp_rp = bp - rp
        bv = _bp_rp_to_bv(bp_rp)
        # Expected relative flux ratios (normalised to G-band as proxy for green)
        # Use simplified colour-index relationships:
        #   f_B ∝ 10^(-0.4 * B) ,  f_V ∝ 10^(-0.4 * V)
        # B-V → expected R/G and B/G ratios
        #   V = g (Gaia G ≈ broad-V)
        #   B = g + bv
        b_mag = g + bv
        r_mag = g - 0.5 * bv       # rough R ≈ V − 0.5*(B−V)
        flux_b_expected = 10.0 ** (-0.4 * b_mag)
        flux_g_expected = 10.0 ** (-0.4 * g)
        flux_r_expected = 10.0 ** (-0.4 * r_mag)
    else:
        # 2MASS J/H/K — coarser proxy
        try:
            j = np.array(catalog["Jmag"][valid], dtype=float)
            h = np.array(catalog["Hmag"][valid], dtype=float)
            k = np.array(catalog["Kmag"][valid], dtype=float)
        except Exception:
            return 1.0, 1.0, 1.0
        flux_r_expected = 10.0 ** (-0.4 * j)
        flux_g_expected = 10.0 ** (-0.4 * h)
        flux_b_expected = 10.0 ** (-0.4 * k)

    # Compute per-star measured ratios vs expected ratios
    # scale_R = median(flux_R_measured / flux_R_expected) normalised to G
    def _robust_ratio(meas: np.ndarray, expected: np.ndarray) -> float:
        ratio = meas / np.maximum(expected, 1e-30)
        ratio = ratio / np.median(ratio)  # normalise so green ≈ 1
        return float(np.median(ratio))

    scale_r = _robust_ratio(fluxes[:, 0], flux_r_expected)
    scale_g = _robust_ratio(fluxes[:, 1], flux_g_expected)
    scale_b = _robust_ratio(fluxes[:, 2], flux_b_expected)

    # Normalise so that mean(scale) = 1 (preserve overall brightness)
    mean_scale = (scale_r + scale_g + scale_b) / 3.0
    if mean_scale > 0:
        scale_r /= mean_scale
        scale_g /= mean_scale
        scale_b /= mean_scale

    # Clamp to a reasonable range to guard against bad fits
    lo, hi = 0.5, 2.0
    scale_r = float(np.clip(scale_r, lo, hi))
    scale_g = float(np.clip(scale_g, lo, hi))
    scale_b = float(np.clip(scale_b, lo, hi))

    if verbose:
        n_used = valid.sum()
        print(f"  [colour cal] {n_used} stars used; "
              f"scales R={scale_r:.4f} G={scale_g:.4f} B={scale_b:.4f}")

    return scale_r, scale_g, scale_b


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_photometric_calibration(img: np.ndarray,
                                   scales: Tuple[float, float, float]) -> np.ndarray:
    """Apply per-channel multiplicative scale factors to the image.

    Args:
        img:    (H, W, 3) float32 image.
        scales: (scale_R, scale_G, scale_B).

    Returns:
        Calibrated (H, W, 3) float32 image.
    """
    result = img.astype(np.float32, copy=True)
    for c, s in enumerate(scales):
        result[:, :, c] *= s
    return result


def run_photometric_calibration(img: np.ndarray, header,
                                 verbose: bool = False
                                 ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Full pipeline: query catalogue → fit scales → apply.

    Returns (calibrated_img, (scale_R, scale_G, scale_B)).
    On failure returns (original_img, (1.0, 1.0, 1.0)).
    """
    _has_wcs = header.get("PLTSOLVD", False) or (
        'CTYPE1' in header and 'CRVAL1' in header and 'CRPIX1' in header)
    if not _has_wcs:
        if verbose:
            print("  [colour cal] No usable WCS — skipping colour calibration")
        return img, (1.0, 1.0, 1.0)

    catalog = None
    catalog_type = "gaia"

    if HAS_GAIA:
        catalog = query_gaia_stars(header)
        catalog_type = "gaia"

    if catalog is None and HAS_VIZIER:
        if verbose:
            print("  [colour cal] Gaia query failed — trying 2MASS via VizieR")
        catalog = query_2mass_stars(header)
        catalog_type = "2mass"

    if catalog is None:
        if verbose:
            print("  [colour cal] No catalogue available — skipping colour calibration")
        return img, (1.0, 1.0, 1.0)

    if verbose:
        print(f"  [colour cal] Queried {len(catalog)} stars from "
              f"{'Gaia DR3' if catalog_type == 'gaia' else '2MASS'}")

    scales = fit_channel_scales(img, header, catalog,
                                catalog_type=catalog_type,
                                verbose=verbose)
    calibrated = apply_photometric_calibration(img, scales)
    return calibrated, scales
