"""Photometric colour calibration via Gaia/2MASS star colours.

Requires:
  * The stacked image to have been plate-solved (WCS keywords in header).
  * astropy     (already a core dependency)

Gaia/2MASS catalogue access is direct HTTP (src/net_query.py, stdlib
urllib) against the Gaia and VizieR TAP services -- no astroquery
dependency.

Workflow
--------
1. Parse WCS from the FITS header.
2. Query the Gaia DR3 source catalogue for stars within the field.
3. Extract per-channel instrumental fluxes for each Gaia star via aperture
   photometry on the stacked image.
4. Compute a linear scale factor for each channel such that the per-star
   colour ratios match the Gaia G_BP − G_RP → B−V relationship.
5. Apply the scale factors multiplicatively to the image.

If plate solving failed, the catalogue query failed, or too few stars are
matched, a warning is printed and the image is returned unchanged.
"""
from __future__ import annotations

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

from src import net_query


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
    phot_bp_mean_mag, phot_rp_mean_mag, teff_gspphot.  Returns None on
    failure. ``teff_gspphot`` (GSP-Phot effective temperature estimate) is
    deliberately not in ``require_not_null`` -- most stars in a typical
    field lack it, and ``fit_channel_scales_spcc`` falls back to the
    colour-index relation per-star when it's NaN rather than losing those
    stars from the fit entirely.
    """
    if not HAS_ASTROPY_WCS:
        return None
    if "CRVAL1" not in header or "CRVAL2" not in header:
        return None

    ra  = float(header["CRVAL1"])
    dec = float(header["CRVAL2"])
    radius_deg = _field_radius_deg(header)

    table = net_query.gaia_cone_search(
        ra, dec, radius_deg,
        columns=["ra", "dec", "phot_g_mean_mag",
                 "phot_bp_mean_mag", "phot_rp_mean_mag", "teff_gspphot"],
        max_rows=max_stars,
        require_not_null=["phot_bp_mean_mag", "phot_rp_mean_mag"])
    if table is None or len(table) == 0:
        return None
    return table if len(table) >= 10 else None


def query_2mass_stars(header, max_stars: int = 500):
    """Fallback: query 2MASS PSC via VizieR for J/H/K magnitudes."""
    if not HAS_ASTROPY_WCS:
        return None
    if "CRVAL1" not in header or "CRVAL2" not in header:
        return None

    ra  = float(header["CRVAL1"])
    dec = float(header["CRVAL2"])
    radius_deg = _field_radius_deg(header)

    table = net_query.vizier_cone_search(
        ra, dec, radius_deg, catalog="II/246/out",
        columns=["RAJ2000", "DEJ2000", "Jmag", "Hmag", "Kmag"],
        max_rows=max_stars)
    if table is None or len(table) == 0:
        return None
    return table if len(table) >= 10 else None


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


# ---------------------------------------------------------------------------
# Spectrophotometric calibration (SPCC-style): physically integrate a
# per-star spectral proxy against actual channel response curves, instead of
# fit_channel_scales' fixed "B = G+(B-V), R = G-0.5*(B-V)" colour-index
# formula. The real differentiator of SPCC-style tools (PixInsight, Siril)
# over simple colour-index matching is this integration step -- reproducing
# each channel's actual spectral response instead of assuming one fixed
# conversion works for every camera/filter combination.
# ---------------------------------------------------------------------------

def _blackbody_spectrum(teff_k: np.ndarray, wavelengths_nm: np.ndarray) -> np.ndarray:
    """Planck blackbody spectral radiance (arbitrary units -- only used in
    ratios, so the physical constants' units don't need to be tracked).
    ``teff_k`` broadcasts against ``wavelengths_nm`` (either can be scalar).
    """
    h_planck = 6.62607015e-34
    c_light = 2.99792458e8
    k_boltz = 1.380649e-23
    wl_m = np.asarray(wavelengths_nm, dtype=np.float64) * 1e-9
    teff = np.asarray(teff_k, dtype=np.float64)
    with np.errstate(over='ignore', divide='ignore'):
        exponent = (h_planck * c_light) / (wl_m * k_boltz * teff)
        radiance = (2 * h_planck * c_light ** 2) / (wl_m ** 5 * np.expm1(exponent))
    return np.nan_to_num(radiance, nan=0.0, posinf=0.0, neginf=0.0)


# Generic per-channel response curves (Gaussian proxies for a typical OSC
# Bayer sensor's R/G/B response) -- NOT a measured QE/filter curve for any
# specific camera. This is the honest, stated fallback used when the caller
# doesn't supply real curves; it is what makes this "SPCC-style" rather than
# a claim of matching PixInsight/Siril's own curve libraries exactly.
_DEFAULT_BAND_CENTERS_NM = {'R': 620.0, 'G': 540.0, 'B': 460.0}
_DEFAULT_BAND_SIGMA_NM = 45.0
_SPCC_WAVELENGTHS_NM = np.linspace(350.0, 950.0, 300)


def _default_channel_response(channel: str, wavelengths_nm: np.ndarray) -> np.ndarray:
    center = _DEFAULT_BAND_CENTERS_NM[channel]
    return np.exp(-0.5 * ((wavelengths_nm - center) / _DEFAULT_BAND_SIGMA_NM) ** 2)


def synthetic_channel_flux(teff_k: float, channel_response=None) -> Tuple[float, float, float]:
    """Integrate a blackbody spectrum at ``teff_k`` against R/G/B channel
    response curves. ``channel_response(channel: str, wavelengths_nm) ->
    array`` lets a caller supply real sensor QE x filter transmission
    curves; defaults to ``_default_channel_response``.

    Returns (flux_R, flux_G, flux_B) in arbitrary but mutually-comparable
    units (only ratios between channels are ever used downstream).
    """
    wl = _SPCC_WAVELENGTHS_NM
    spec = _blackbody_spectrum(float(teff_k), wl)
    resp_fn = channel_response or _default_channel_response
    _integrate = getattr(np, 'trapezoid', np.trapz)  # trapezoid: numpy >= 2.0
    fluxes = []
    for ch in ('R', 'G', 'B'):
        resp = resp_fn(ch, wl)
        fluxes.append(float(_integrate(spec * resp, wl)))
    return tuple(fluxes)


def _synthetic_channel_flux_batch(teff_k: np.ndarray, channel_response=None
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized form of ``synthetic_channel_flux`` for many stars at
    once. ``fit_channel_scales_spcc`` used to call the scalar version once
    per matched star (up to ~500 per Gaia query) in a Python loop; the
    blackbody spectrum and its integration against each channel response
    both broadcast cleanly across stars, so this replaces that loop with 4
    vectorized calls total (1 spectrum + 3 channel integrations) regardless
    of star count. ``synthetic_channel_flux`` itself is kept as the
    scalar, single-star API (used directly by callers that only have one
    temperature and by its own tests) rather than folded into this.
    """
    wl = _SPCC_WAVELENGTHS_NM
    teff_arr = np.asarray(teff_k, dtype=np.float64)
    spec = _blackbody_spectrum(teff_arr[:, None], wl[None, :])  # (n_stars, n_wl)
    resp_fn = channel_response or _default_channel_response
    _integrate = getattr(np, 'trapezoid', np.trapz)
    fluxes = []
    for ch in ('R', 'G', 'B'):
        resp = resp_fn(ch, wl)  # (n_wl,)
        fluxes.append(_integrate(spec * resp[None, :], wl, axis=1))  # (n_stars,)
    return fluxes[0], fluxes[1], fluxes[2]


def fit_channel_scales_spcc(img: np.ndarray, header, catalog,
                            channel_response=None,
                            verbose: bool = False) -> Tuple[float, float, float]:
    """Spectrophotometric variant of ``fit_channel_scales``: per-star
    expected channel flux ratios come from integrating a blackbody spectrum
    at the star's Gaia ``teff_gspphot`` against channel response curves,
    instead of a single fixed colour-index formula. Falls back to
    ``_bp_rp_to_bv``'s colour-index relation per-star when ``teff_gspphot``
    is unavailable (most GSP-Phot estimates are missing for a random field
    -- typically a minority of matched stars have one), so coverage doesn't
    collapse to only the stars with a temperature estimate.

    Returns (scale_R, scale_G, scale_B); (1.0, 1.0, 1.0) on failure --
    same failure contract as ``fit_channel_scales``.
    """
    pixel_coords = _pixel_coords(catalog, header)
    if pixel_coords is None:
        return 1.0, 1.0, 1.0

    px, py = pixel_coords[:, 0], pixel_coords[:, 1]
    ap_radius = max(3, min(8, int(min(img.shape[:2]) / 200)))
    fluxes = _aperture_flux(img, px, py, radius=ap_radius)

    valid = np.all(np.isfinite(fluxes) & (fluxes > 0), axis=1)
    if valid.sum() < 10:
        if verbose:
            print(f"  [SPCC] Too few valid stars ({valid.sum()}) — skipping")
        return 1.0, 1.0, 1.0
    fluxes = fluxes[valid]

    try:
        bp = np.array(catalog["phot_bp_mean_mag"][valid], dtype=float)
        rp = np.array(catalog["phot_rp_mean_mag"][valid], dtype=float)
        g_mag = np.array(catalog["phot_g_mean_mag"][valid], dtype=float)
    except Exception:
        return 1.0, 1.0, 1.0
    if "teff_gspphot" in catalog.colnames:
        teff = np.array(catalog["teff_gspphot"][valid], dtype=float)
    else:
        teff = np.full(valid.sum(), np.nan)

    bv = _bp_rp_to_bv(bp - rp)
    fallback_b = g_mag + bv
    fallback_r = g_mag - 0.5 * bv
    flux_g_expected = 10.0 ** (-0.4 * g_mag)

    n = len(teff)
    flux_r_expected = np.empty(n)
    flux_b_expected = np.empty(n)

    # Vectorized over all stars at once (previously a per-star Python loop
    # calling synthetic_channel_flux individually -- up to ~500 iterations
    # per Gaia query): _synthetic_channel_flux_batch computes the blackbody
    # spectrum and its 3 channel integrations for every candidate star in 4
    # calls total, independent of star count.
    has_teff = np.isfinite(teff) & (teff >= 2000.0) & (teff <= 50000.0)
    use_bb = np.zeros(n, dtype=bool)
    if np.any(has_teff):
        idx = np.flatnonzero(has_teff)
        fr, fg, fb = _synthetic_channel_flux_batch(teff[idx], channel_response)
        fg_ok = fg > 0
        good = idx[fg_ok]
        # Normalise the synthetic G-band flux to each star's actual measured
        # Gaia G magnitude, so units match flux_g_expected's photometric
        # scale -- fr/fg/fb are otherwise on an arbitrary blackbody-radiance
        # scale, only their ratios are physical.
        scale = flux_g_expected[good] / fg[fg_ok]
        flux_r_expected[good] = fr[fg_ok] * scale
        flux_b_expected[good] = fb[fg_ok] * scale
        use_bb[good] = True
    n_bb = int(np.sum(use_bb))

    # Fallback: same colour-index approximation fit_channel_scales uses.
    fallback_mask = ~use_bb
    flux_r_expected[fallback_mask] = 10.0 ** (-0.4 * fallback_r[fallback_mask])
    flux_b_expected[fallback_mask] = 10.0 ** (-0.4 * fallback_b[fallback_mask])

    def _robust_ratio(meas: np.ndarray, expected: np.ndarray) -> float:
        ratio = meas / np.maximum(expected, 1e-30)
        ratio = ratio / np.median(ratio)
        return float(np.median(ratio))

    scale_r = _robust_ratio(fluxes[:, 0], flux_r_expected)
    scale_g = _robust_ratio(fluxes[:, 1], flux_g_expected)
    scale_b = _robust_ratio(fluxes[:, 2], flux_b_expected)

    mean_scale = (scale_r + scale_g + scale_b) / 3.0
    if mean_scale > 0:
        scale_r /= mean_scale
        scale_g /= mean_scale
        scale_b /= mean_scale

    lo, hi = 0.5, 2.0
    scale_r = float(np.clip(scale_r, lo, hi))
    scale_g = float(np.clip(scale_g, lo, hi))
    scale_b = float(np.clip(scale_b, lo, hi))

    if verbose:
        print(f"  [SPCC] {n} stars used ({n_bb} via blackbody Teff, "
              f"{n - n_bb} via colour-index fallback); "
              f"scales R={scale_r:.4f} G={scale_g:.4f} B={scale_b:.4f}")

    return scale_r, scale_g, scale_b


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
                                 verbose: bool = False,
                                 method: str = 'colorindex'
                                 ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Full pipeline: query catalogue → fit scales → apply.

    ``method``: 'colorindex' (default -- ``fit_channel_scales``'s fixed
    B-V-derived formula) or 'spcc' (``fit_channel_scales_spcc``'s
    blackbody-spectrum integration against channel response curves, falling
    back to the colour-index formula per-star when a Gaia Teff estimate
    isn't available). 'spcc' needs 2MASS's catalogue skipped -- it's a
    Gaia-only method (2MASS carries no Teff column) -- so on a Gaia-query
    failure it degrades to 'colorindex' via 2MASS rather than returning
    unscaled.

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

    catalog = query_gaia_stars(header)
    catalog_type = "gaia"

    if catalog is None:
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

    if method == 'spcc' and catalog_type == 'gaia':
        scales = fit_channel_scales_spcc(img, header, catalog, verbose=verbose)
    else:
        scales = fit_channel_scales(img, header, catalog,
                                    catalog_type=catalog_type,
                                    verbose=verbose)
    calibrated = apply_photometric_calibration(img, scales)
    return calibrated, scales
