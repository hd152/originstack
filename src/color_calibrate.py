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

from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    from astropy.wcs import WCS  # noqa: F401  (kept for HAS_ASTROPY_WCS gate)
    HAS_ASTROPY_WCS = True
except Exception:
    HAS_ASTROPY_WCS = False

from src import net_query
from src.photometry_core import _pixel_coords, aperture_photometry_batch

# ---------------------------------------------------------------------------
# Catalog query
# ---------------------------------------------------------------------------

def _field_radius_deg(header) -> float:
    """Estimate the field-of-view radius in degrees from WCS."""
    try:
        wcs = WCS(header).celestial  # (3, H, W) cube header -> 2-axis
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
        max_rows=max_stars, order_by="Jmag")
    if table is None or len(table) == 0:
        return None
    return table if len(table) >= 10 else None


# ---------------------------------------------------------------------------
# Aperture photometry
# ---------------------------------------------------------------------------

def _aperture_flux(img: np.ndarray, px: np.ndarray, py: np.ndarray,
                   radius: int = 5, sky_annulus: int = 3) -> np.ndarray:
    """Circular aperture photometry -> (N, 3) per-channel background-
    subtracted flux (negatives clamped to 0; NaN for stars whose aperture
    leaves the frame).

    Thin wrapper over ``photometry_core.aperture_photometry_batch`` -- the
    shared partial-pixel / robust-annulus kernel. Colour-cal only uses the
    per-star flux ratios, so the small numeric shift from the previous
    integer-mask + sigma-clipped-annulus implementation (sub-percent, and
    the scales are clamped to [0.5, 2.0] anyway) is not material.
    """
    flux, *_ = aperture_photometry_batch(
        img, np.asarray(px, dtype=float), np.asarray(py, dtype=float),
        float(radius), float(radius), float(radius + sky_annulus))
    return np.where(np.isfinite(flux), np.maximum(flux, 0.0), np.nan)


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
# hasattr, not getattr's default arg -- that eagerly evaluates np.trapz, which some numpy builds have removed entirely.
_TRAPZ = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz


def _default_channel_response(channel: str, wavelengths_nm: np.ndarray) -> np.ndarray:
    center = _DEFAULT_BAND_CENTERS_NM[channel]
    return np.exp(-0.5 * ((wavelengths_nm - center) / _DEFAULT_BAND_SIGMA_NM) ** 2)


def _synthetic_channel_flux_batch(teff_k: np.ndarray, channel_response=None
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate a blackbody spectrum at each ``teff_k`` against R/G/B
    channel response curves, for many stars at once (the spectrum and its
    integration broadcast across stars). ``channel_response(channel: str,
    wavelengths_nm) -> array`` lets a caller supply real sensor QE x filter
    transmission curves; defaults to ``_default_channel_response``.

    Returns (flux_R, flux_G, flux_B) arrays in arbitrary but mutually
    comparable units (only ratios between channels are used downstream).
    """
    wl = _SPCC_WAVELENGTHS_NM
    teff_arr = np.asarray(teff_k, dtype=np.float64)
    spec = _blackbody_spectrum(teff_arr[:, None], wl[None, :])  # (n_stars, n_wl)
    resp_fn = channel_response or _default_channel_response
    fluxes = []
    for ch in ('R', 'G', 'B'):
        resp = resp_fn(ch, wl)  # (n_wl,)
        fluxes.append(_TRAPZ(spec * resp[None, :], wl, axis=1))  # (n_stars,)
    return fluxes[0], fluxes[1], fluxes[2]


def _channel_correction(meas: np.ndarray, expected: np.ndarray) -> float:
    """Multiplicative factor that brings a channel's measured star fluxes to
    the catalogue's expected ones: median(expected / measured).

    ``apply_photometric_calibration`` multiplies the image by this, so a
    channel reading too bright gets a factor below 1. Units (ADU vs. the
    catalogue's 10^(-0.4 m)) are common to all three channels and cancel in
    the callers' mean normalisation. This used to divide each star's ratio by
    the channel's own median ratio before taking the median -- always 1.0 --
    so --color-calibrate never changed the image.
    """
    ratio = np.maximum(expected, 1e-30) / np.maximum(meas, 1e-30)
    return float(np.median(ratio))


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

    # Vectorized over all stars at once: _synthetic_channel_flux_batch
    # computes the blackbody spectrum and its 3 channel integrations for
    # every candidate star in 4 calls total, independent of star count.
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

    scale_r = _channel_correction(fluxes[:, 0], flux_r_expected)
    scale_g = _channel_correction(fluxes[:, 1], flux_g_expected)
    scale_b = _channel_correction(fluxes[:, 2], flux_b_expected)

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
    # Per-channel correction = median(expected / measured); mean-normalised below.
    scale_r = _channel_correction(fluxes[:, 0], flux_r_expected)
    scale_g = _channel_correction(fluxes[:, 1], flux_g_expected)
    scale_b = _channel_correction(fluxes[:, 2], flux_b_expected)

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


# Gaia BP-RP of the Sun (Casagrande & VandenBerg 2018: 0.82). A star of this
# colour is rendered white by ``fit_channel_scales_solar``.
SOLAR_BP_RP = 0.82


def fit_channel_scales_solar(img: np.ndarray, header, verbose: bool = False,
                             min_stars: int = 15, slope_prior=None,
                             min_prior_stars: int = 6):
    """Per-channel scales that render a solar-colour (G2V) star white.

    For every detected, isolated, unsaturated star matched to Gaia DR3
    (``photometry.match_gaia_field``), the instrumental colour indices
    ``-2.5 log10(B/R)`` and ``-2.5 log10(G/R)`` are measured by aperture
    photometry with a local sky annulus and fitted against Gaia BP-RP with a
    Theil-Sen line. The fitted indices at BP-RP = 0.82 are the colour this
    stack gives the Sun; the scales cancel them, so a G2V star comes out with
    B = G = R -- the white reference PixInsight's (S)PCC uses by default.

    Unlike ``fit_channel_scales`` it fits a colour *relation* rather than
    assuming one (no B-V transformation, no guessed R band), uses only stars
    that are actually detected and well measured, and sizes the aperture from
    the measured FWHM.

    ``slope_prior`` = (slope_BR, slope_GR): the camera's instrumental colour slopes
    against BP-RP from its profile (src/camera_profile.py ``colour_slopes``). They
    are a property of the sensor's spectral response -- per-channel scales (white
    balance, flat, transparency) move only the intercepts -- so with fewer than
    ``min_stars`` good stars, but at least ``min_prior_stars``, the fit keeps the
    profile's slopes and fits only the intercepts (a median). A free fit is still
    used whenever there are enough stars.

    Returns ``((s_R, s_G, s_B), info)`` with G fixed at 1, or ``None`` when too
    few good stars are found or the result is implausible (a channel scale
    outside 0.5-2).
    """
    from scipy import stats

    from src.photometry import match_gaia_field
    gm = match_gaia_field(img, header, verbose=verbose)
    if gm is None:
        return None
    lum = np.asarray(img, dtype=np.float64).mean(axis=2)
    sat = 0.6 * float(np.percentile(lum, 99.99))
    sky = float(np.median(lum[::4, ::4]))
    ceil = sky + 0.7 * (float(np.nanmax(lum)) - sky)
    need = min_stars if slope_prior is None else min(min_stars, min_prior_stars)

    def select(gm):
        bp_rp = gm.bp - gm.rp
        xs, ys = gm.x, gm.y
        d2 = (xs[:, None] - xs[None, :]) ** 2 + (ys[:, None] - ys[None, :]) ** 2
        np.fill_diagonal(d2, np.inf)
        isolated = d2.min(axis=1) > (2.0 * gm.r_out) ** 2 if len(xs) > 1 else np.ones(len(xs), bool)
        base = np.isfinite(bp_rp) & (bp_rp > -0.3) & (bp_rp < 2.5) & isolated
        ok = base & (gm.det_peak < sat)
        if ok.sum() < need:
            # On a sparse field the 99.99th percentile is the sky itself, so 0.6x it
            # sits below every star's peak (which includes the sky) and nothing
            # passes. Then judge saturation against the data ceiling, above the sky.
            ok = base & (gm.det_peak < ceil)
        return bp_rp, base, ok

    bp_rp, base, ok = select(gm)
    saturated = base & ~ok
    if ok.sum() < need and saturated.sum() >= max(need, base.sum() // 2):
        # A rich field: the catalogue's brightest rows (it is queried brightest-first)
        # are all saturated in the subs -- M37 matched 620 stars at G 10.8-12.9, all
        # clipped. Query again from just fainter than the faintest saturated one.
        g_min = float(np.nanmax(gm.g[saturated]))
        deeper = match_gaia_field(img, header, verbose=verbose, min_g=g_min)
        if deeper is not None:
            if verbose:
                from src.utils import safe_print
                safe_print(f"  [colour cal] {int(saturated.sum())} matched stars saturated; "
                      f"re-queried Gaia from G > {g_min:.1f}")
            gm = deeper
            bp_rp, base, ok = select(gm)
    xs, ys = gm.x, gm.y
    if ok.sum() < need:
        if verbose:
            print(f"  [colour cal] {int(ok.sum())} usable Gaia stars (< {need})")
        return None
    flux, _sky, sky_sigma, _peak, area = aperture_photometry_batch(
        np.ascontiguousarray(img, dtype=np.float32), xs[ok], ys[ok],
        float(gm.ap_radius), float(gm.r_in), float(gm.r_out))
    snr = flux / np.maximum(sky_sigma * np.sqrt(area)[:, None], 1e-12)
    good = np.all(np.isfinite(flux) & (flux > 0) & (snr > 20.0), axis=1)
    if good.sum() < need:
        if verbose:
            print(f"  [colour cal] {int(good.sum())} stars with SNR > 20 (< {need})")
        return None
    f = flux[good]
    c = bp_rp[ok][good]
    ci_br = -2.5 * np.log10(f[:, 2] / f[:, 0])
    ci_gr = -2.5 * np.log10(f[:, 1] / f[:, 0])
    prior_used = good.sum() < min_stars
    if prior_used:
        s_br, s_gr = (float(v) for v in slope_prior)
        i_br = float(np.median(ci_br - s_br * c))
        i_gr = float(np.median(ci_gr - s_gr * c))
    else:
        s_br, i_br = stats.theilslopes(ci_br, c)[:2]
        s_gr, i_gr = stats.theilslopes(ci_gr, c)[:2]
    br_sun = i_br + s_br * SOLAR_BP_RP
    gr_sun = i_gr + s_gr * SOLAR_BP_RP
    # Multiply B by 10^(0.4 br_sun) and G by 10^(0.4 gr_sun) relative to R so
    # the Sun's B/R and G/R become 1, then normalise G to 1.
    s_r, s_g, s_b = 1.0, 10.0 ** (0.4 * gr_sun), 10.0 ** (0.4 * br_sun)
    s_r, s_b, s_g = s_r / s_g, s_b / s_g, 1.0
    resid = ci_br - (i_br + s_br * c)
    info = dict(n=int(good.sum()), slope_br=float(s_br), slope_gr=float(s_gr),
                prior=bool(prior_used),
                scatter_br=float(1.4826 * np.median(np.abs(resid - np.median(resid)))))
    if not all(0.5 <= v <= 2.0 for v in (s_r, s_b)):
        if verbose:
            print(f"  [colour cal] implausible scales R={s_r:.3f} B={s_b:.3f} -- not applied")
        return None
    return (float(s_r), float(s_g), float(s_b)), info


def calibrate_linear_stack(img: np.ndarray, header, method: str = 'solar',
                           verbose: bool = False, slope_prior=None):
    """Colour-calibrate a *linear* stack in place before Phase 4.

    Returns ``(scales, info)`` (``info`` is a short description for the log)
    or ``None`` when no calibration was applied. ``method`` 'solar' is
    ``fit_channel_scales_solar``; 'colorindex' and 'spcc' are the older fits
    in ``run_photometric_calibration``.
    """
    if method == 'solar':
        fit = fit_channel_scales_solar(img, header, verbose=verbose, slope_prior=slope_prior)
        if fit is None:
            return None
        scales, d = fit
        info = (f"{d['n']} Gaia stars, white = G2V, B-R scatter "
                f"{d['scatter_br']:.3f} mag"
                + (", colour slopes from the camera profile" if d.get('prior') else ""))
    else:
        _, scales = run_photometric_calibration(img, header, verbose=verbose, method=method)
        if scales == (1.0, 1.0, 1.0):
            return None
        info = f"method {method}"
    apply_scales_inplace(img, scales)
    return scales, info


def _clipped_weight(img: np.ndarray):
    """Per pixel, 0..1: how close it is to a clipped plateau, or None if the
    stack has none. A channel has a plateau when ``Config.CLIP_PLATEAU_MIN_STARS``
    separate regions sit within 2% of its maximum: saturated star cores stack to
    nearly one level, while a smooth unclipped galaxy core is a single region
    (counting pixels instead would neutralise it). Ramps from 0 at 80% of the
    plateau to 1 at it, like the white balance step's
    ``debayer._desaturate_near_clipped_highlights``."""
    from scipy import ndimage as ndi

    from src.models import Config
    frac = None
    for c in range(img.shape[2]):
        ch = img[:, :, c]
        top = float(np.nanmax(ch))
        if not np.isfinite(top) or top <= 0:
            continue
        _, n_regions = ndi.label(ch >= 0.98 * top)
        if n_regions < Config.CLIP_PLATEAU_MIN_STARS:
            continue
        f = ch / np.float32(top)
        frac = f if frac is None else np.maximum(frac, f)
    if frac is None:
        return None
    return np.clip((frac - 0.8) / 0.2, 0.0, 1.0)


def apply_scales_inplace(img: np.ndarray, scales) -> np.ndarray:
    """Multiply each channel by its colour-calibration scale, keeping clipped
    star cores neutral.

    A saturated core is clipped equally in R/G/B (white balance already made it
    neutral), so scaling it per channel turns it a colour it never had: with a
    Gaia fit of R x0.72, B x1.09 every bright star became a blue disc, and the
    colour-preserving stretch kept that blue. Pixels near a clipped plateau are
    blended towards their brightest scaled channel instead."""
    weight = _clipped_weight(img)
    for c, s in enumerate(scales):
        img[:, :, c] *= np.float32(s)
    if weight is not None:
        sel = weight > 0
        w = weight[sel][:, None].astype(np.float32)
        px = img[sel]
        neutral = px.max(axis=1, keepdims=True)
        img[sel] = px * (1 - w) + neutral * w
    return img


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
    return apply_scales_inplace(img.astype(np.float32, copy=True), scales)


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
