"""Absolute aperture photometry on the linear stack (--photometry).

Turns the linear pre-post-processing stack into a calibrated stellar
photometry catalogue:

  1. Detect stars (matched filter, shared with quality analysis).
  2. Cone-search Gaia DR3 around the field and project the catalogue onto
     the image via the header WCS (session info.json solve from a Celestron
     Origin, or --plate-solve).
  3. Cross-match detected stars to Gaia by pixel position.
  4. Circular-aperture photometry per channel (reuses color_calibrate's
     `_aperture_flux`: aperture + sigma-clipped sky annulus).
  5. Fit a per-channel photometric zero point
        m_cal = m_inst - k * X + ZP
     against the Gaia G/BP/RP magnitudes (RP->R, G->G, BP->B -- a coarse
     OSC mapping, stated plainly, not a filter-matched transform), with a
     robust sigma-clipped median and an airmass term X derived from the
     site location (info.json GPS) + field centre + observation time.
  6. Write `<output>_photometry.csv` and MAGZP_* header keywords.

Scope for this first cut: single stacked frame -> one catalogue + one
zero point per channel. No per-sub time series / light curves (that needs
a per-frame photometry pass the streaming stacker does not currently keep
frames around for) and no photon-transfer gain estimation from the
calibration frames -- the Poisson error term is only included when the
header carries a real GAIN/EGAIN (e-/ADU). Everything here is derivable
from the lights + info.json + a Gaia query; absolute all-sky accuracy is
bounded by the OSC channel<->passband mismatch and the single-airmass fit
(extinction k defaults to nominal per-band values, override with
--photometry-extinction-k).
"""
from __future__ import annotations

import csv
import logging
import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import astro_native as _NATIVE
except Exception:  # pragma: no cover - native module is optional
    _NATIVE = None

_log = logging.getLogger("originstack")

_APB_SUBPIX = 4  # aperture-edge supersampling factor (native + numpy paths)

# Nominal broadband extinction coefficients (mag / airmass) for a typical
# sea-level-ish site, used when the fit only has one airmass to work with
# (a single stacked frame). Override all three with --photometry-extinction-k.
_NOMINAL_K = {"R": 0.09, "G": 0.15, "B": 0.23}

_CHANNELS = ("R", "G", "B")
# Gaia band feeding each OSC channel's catalogue magnitude. Coarse by
# construction -- Gaia G is a very broad white-ish band, BP/RP are half-band
# blue/red -- but good enough to anchor a zero point to ~0.05 mag.
_GAIA_BAND_FOR_CHANNEL = {"R": "phot_rp_mean_mag", "G": "phot_g_mean_mag",
                          "B": "phot_bp_mean_mag"}


def _field_centre_and_radius(header, shape):
    """(ra_deg, dec_deg, search_radius_deg, plate_scale_arcsec) from the WCS."""
    try:
        from astropy.wcs import WCS
        from astropy.wcs.utils import proj_plane_pixel_scales
    except Exception:
        return None
    try:
        w = WCS(header).celestial
        if not w.has_celestial:
            return None
        H, W = shape[:2]
        sky = w.pixel_to_world(W / 2.0, H / 2.0)
        ra0 = float(sky.ra.deg)
        dec0 = float(sky.dec.deg)
        scales_deg = proj_plane_pixel_scales(w)  # deg/pixel, (x, y)
        scale_arcsec = float(np.mean(scales_deg)) * 3600.0
        half_diag_deg = 0.5 * math.hypot(W * scales_deg[0], H * scales_deg[1])
        return ra0, dec0, half_diag_deg * 1.15, scale_arcsec
    except Exception:
        return None


def _airmass(header, session_info, ra_deg, dec_deg, when=None):
    """Airmass at the field centre for a given observation time.

    Needs site lat/long (info.json GPS) and a timestamp: ``when`` if given,
    else header DATE-OBS, else info.json dateTime. Returns None when either
    is missing or astropy can't parse the time -- the caller then drops the
    k*X term and lets the zero point absorb the mean extinction.
    """
    if session_info is None or not getattr(session_info, "has_gps", False):
        return None
    time_str = str(when) if when else None
    if time_str is None and header is not None:
        for key in ("DATE-OBS", "DATE_OBS", "DATEOBS"):
            if header.get(key):
                time_str = str(header[key])
                break
    if time_str is None:
        time_str = getattr(session_info, "date_time", None)
    if not time_str:
        return None
    try:
        from astropy.coordinates import AltAz, EarthLocation, SkyCoord
        from astropy.time import Time
        import astropy.units as u
    except Exception:
        return None
    try:
        loc = EarthLocation(lat=session_info.latitude * u.deg,
                            lon=session_info.longitude * u.deg,
                            height=(session_info.altitude or 0.0) * u.m)
        t = Time(time_str)
        altaz = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg).transform_to(
            AltAz(obstime=t, location=loc))
        alt_deg = float(altaz.alt.deg)
        if alt_deg <= 3.0:
            return None
        # Kasten & Young (1989) -- stays finite and accurate toward the horizon.
        z = 90.0 - alt_deg
        X = 1.0 / (math.cos(math.radians(z))
                   + 0.50572 * (96.07995 - z) ** (-1.6364))
        return float(X) if np.isfinite(X) and X >= 1.0 else None
    except Exception:
        return None


def _match_catalog_to_detections(cat_xy, det_xy, radius_px):
    """Nearest-neighbour match, one detected star per catalogue star.

    Returns (cat_idx, det_idx) int arrays of accepted pairs.
    """
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return np.zeros(0, int), np.zeros(0, int)
    if len(cat_xy) == 0 or len(det_xy) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    tree = cKDTree(det_xy)
    dist, idx = tree.query(cat_xy, k=1, distance_upper_bound=radius_px)
    ok = np.isfinite(dist)
    cat_idx = np.flatnonzero(ok)
    det_idx = idx[ok]
    # Drop catalogue stars that collided onto the same detection (blends).
    seen = {}
    keep_cat, keep_det = [], []
    for ci, di in zip(cat_idx, det_idx):
        if di in seen:
            continue
        seen[di] = ci
        keep_cat.append(ci)
        keep_det.append(di)
    return np.asarray(keep_cat, int), np.asarray(keep_det, int)


def _fit_zeropoint_colorterm(resid, color, ref_color, fit_ct=False,
                             sigma=2.5, iters=3):
    """Robust fit of  resid_i = ZP + CT * (color_i - ref_color).

    ``resid`` is per-star ``m_cat - m_inst + k*X``; ``color`` is the Gaia
    BP-RP colour. With ``fit_ct=False`` the colour term is forced to zero
    and this reduces to a sigma-clipped median (the default). Returns
    ``(zp, zp_err, ct, n_used)`` or ``None``.
    """
    r = np.asarray(resid, dtype=np.float64)
    col = np.asarray(color, dtype=np.float64) - float(ref_color)
    good = np.isfinite(r) & np.isfinite(col)
    r, col = r[good], col[good]
    if r.size < 5:
        return None
    keep = np.ones(r.size, bool)
    zp, ct = float(np.median(r)), 0.0
    for _ in range(iters):
        rr, cc = r[keep], col[keep]
        if fit_ct and rr.size >= 8:
            A = np.column_stack([np.ones_like(cc), cc])
            coef, *_ = np.linalg.lstsq(A, rr, rcond=None)
            zp, ct = float(coef[0]), float(coef[1])
        else:
            zp, ct = float(np.median(rr)), 0.0
        res = r - (zp + ct * col)
        centre = np.median(res[keep])
        scale = 1.4826 * np.median(np.abs(res[keep] - centre))
        if scale <= 0:
            break
        nk = np.abs(res - centre) <= sigma * scale
        if nk.sum() < 5 or nk.sum() == keep.sum():
            if nk.sum() >= 5:
                keep = nk
            break
        keep = nk
    res_final = r[keep] - (zp + ct * col[keep])
    zp_err = (1.4826 * np.median(np.abs(res_final - np.median(res_final)))
              / math.sqrt(max(int(keep.sum()), 1)))
    return zp, float(zp_err), ct, int(keep.sum())


# ---------------------------------------------------------------------------
# Batch aperture photometry (native Rust kernel + numpy mirror)
# ---------------------------------------------------------------------------

def aperture_photometry_batch(img, xs, ys, r_ap, r_in, r_out,
                              subpix: int = _APB_SUBPIX):
    """Partial-pixel circular-aperture photometry for many centres at once.

    Returns ``(flux, sky, sky_sigma, peak, area)`` -- ``flux``/``sky``/
    ``sky_sigma``/``peak`` are ``(N, C)`` (background-subtracted flux,
    robust sky median, ``1.4826*MAD`` sky sigma, and the raw max pixel
    value inside the aperture per channel), ``area`` is ``(N,)`` (effective
    aperture pixel area). A star whose full ``r_out`` disk is not inside
    the frame gets an all-NaN row. Integer pixel coordinates are pixel
    centres (matches ``star_detect`` / ``_aperture_flux``).

    Dispatches to the native ``aperture_photometry_batch`` kernel when
    built; ``_aperture_photometry_batch_numpy`` otherwise.
    """
    img_c = np.ascontiguousarray(img, dtype=np.float32)
    xs_c = np.ascontiguousarray(xs, dtype=np.float64)
    ys_c = np.ascontiguousarray(ys, dtype=np.float64)
    if _NATIVE is not None and hasattr(_NATIVE, "aperture_photometry_batch"):
        return _NATIVE.aperture_photometry_batch(
            img_c, xs_c, ys_c, float(r_ap), float(r_in), float(r_out),
            int(subpix))
    return _aperture_photometry_batch_numpy(
        img_c, xs_c, ys_c, float(r_ap), float(r_in), float(r_out), int(subpix))


def _aperture_photometry_batch_numpy(img, xs, ys, r_ap, r_in, r_out,
                                     subpix=_APB_SUBPIX):
    """Numpy reference for :func:`aperture_photometry_batch` (parity-tested
    against the native kernel in ``tests/test_native.py``)."""
    a = np.asarray(img, dtype=np.float64)
    H, W, C = a.shape
    N = len(xs)
    flux = np.full((N, C), np.nan)
    sky = np.full((N, C), np.nan)
    sig = np.full((N, C), np.nan)
    peak = np.full((N, C), np.nan)
    area = np.full(N, np.nan)

    sub_off = (np.arange(subpix) + 0.5) / subpix - 0.5
    oy, ox = np.meshgrid(sub_off, sub_off, indexing="ij")
    half_diag = math.sqrt(0.5)
    full_in = (r_ap - half_diag) ** 2 if r_ap > half_diag else -1.0
    full_out = (r_ap + half_diag) ** 2
    r_ap2, r_in2, r_out2 = r_ap ** 2, r_in ** 2, r_out ** 2

    for i in range(N):
        cx, cy = float(xs[i]), float(ys[i])
        if (not np.isfinite(cx) or not np.isfinite(cy)
                or cx - r_out < 0 or cx + r_out >= W - 1
                or cy - r_out < 0 or cy + r_out >= H - 1):
            continue
        x0 = max(int(math.floor(cx - r_out)) - 1, 0)
        y0 = max(int(math.floor(cy - r_out)) - 1, 0)
        x1 = min(int(math.ceil(cx + r_out)) + 1, W - 1)
        y1 = min(int(math.ceil(cy + r_out)) + 1, H - 1)
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        dy = yy - cy
        dx = xx - cx
        d2 = dx * dx + dy * dy

        frac = np.zeros(d2.shape, dtype=np.float64)
        if full_in > 0:
            frac[d2 <= full_in] = 1.0
        edge = (d2 > (full_in if full_in > 0 else -1.0)) & (d2 < full_out)
        if np.any(edge):
            de_y = dy[edge][:, None, None] + oy[None]
            de_x = dx[edge][:, None, None] + ox[None]
            frac[edge] = (de_x ** 2 + de_y ** 2 <= r_ap2).mean(axis=(1, 2))

        patch = a[y0:y1 + 1, x0:x1 + 1, :]
        ap_area = float(frac.sum())
        area[i] = ap_area
        ap_sum = (patch * frac[..., None]).sum(axis=(0, 1))
        ap_mask = frac > 0.0
        ann = (d2 > r_in2) & (d2 <= r_out2)
        for c in range(C):
            if np.any(ap_mask):
                peak[i, c] = float(patch[ap_mask, c].max())
            vals = patch[ann, c]
            if vals.size < 4:
                continue
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            sky[i, c] = med
            sig[i, c] = 1.4826 * mad
            flux[i, c] = ap_sum[c] - med * ap_area
    return flux, sky, sig, peak, area


# ---------------------------------------------------------------------------
# Gaia field cross-match (shared by single-frame and time-series photometry)
# ---------------------------------------------------------------------------

@dataclass
class GaiaMatch:
    source_id: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    x: np.ndarray          # Gaia-projected pixel position on the input grid
    y: np.ndarray
    g: np.ndarray
    bp: np.ndarray
    rp: np.ndarray
    det_peak: np.ndarray   # peak of the matched detection
    fwhm: float
    ap_radius: int
    r_in: float
    r_out: float
    field_ra: float
    field_dec: float
    plate_scale: float


def match_gaia_field(img_rgb, header, *, max_rows: int = 1200,
                     verbose: bool = False) -> Optional[GaiaMatch]:
    """Detect stars, cone-search Gaia DR3, project it via the header WCS,
    and cross-match by pixel position. Returns a :class:`GaiaMatch` or
    ``None`` (with a logged reason) if any step fails."""
    a = np.ascontiguousarray(np.asarray(img_rgb, dtype=np.float64))
    if a.ndim != 3 or a.shape[2] != 3:
        _log.warning("Photometry: needs an (H, W, 3) RGB image -- skipping")
        return None
    H, W = a.shape[:2]
    luma = a.mean(axis=2)

    centre = _field_centre_and_radius(header, a.shape)
    if centre is None:
        _log.warning("Photometry: no usable WCS (needs --plate-solve or a "
                     "session info.json solve) -- skipping")
        return None
    ra0, dec0, search_radius_deg, plate_scale = centre

    from src.star_detect import detect_stars_matched_filter
    sources = detect_stars_matched_filter(luma)
    if sources is None or len(sources) < 8:
        _log.warning("Photometry: only %d stars detected -- skipping",
                     0 if sources is None else len(sources))
        return None
    det_xy = np.column_stack([np.asarray(sources["xcentroid"], float),
                              np.asarray(sources["ycentroid"], float)])

    try:
        from src.quality import measure_fwhm
        fwhm = float(measure_fwhm(luma, sources))
    except Exception:
        fwhm = 0.0
    if not (1.0 <= fwhm <= 25.0):
        fwhm = 3.5
    ap_radius = int(max(3, round(1.6 * fwhm)))
    r_in = float(max(ap_radius + 3, round(2.2 * fwhm)))
    r_out = float(r_in + max(5.0, round(2.0 * fwhm)))

    from src.net_query import gaia_cone_search
    catalog = gaia_cone_search(
        ra0, dec0, min(search_radius_deg, 2.0),
        ["source_id", "ra", "dec", "phot_g_mean_mag",
         "phot_bp_mean_mag", "phot_rp_mean_mag"],
        max_rows=max_rows,
        require_not_null=["phot_g_mean_mag", "phot_bp_mean_mag",
                          "phot_rp_mean_mag"])
    if catalog is None or len(catalog) < 8:
        _log.warning("Photometry: Gaia query returned %s stars -- skipping",
                     "no" if catalog is None else len(catalog))
        return None

    from src.color_calibrate import _pixel_coords
    cat_pix = _pixel_coords(catalog, header)
    if cat_pix is None:
        _log.warning("Photometry: could not project Gaia onto the image -- skipping")
        return None

    edge = r_out + 1.0
    inb = ((cat_pix[:, 0] >= edge) & (cat_pix[:, 0] < W - edge) &
           (cat_pix[:, 1] >= edge) & (cat_pix[:, 1] < H - edge))
    cat_rows_in = np.flatnonzero(inb)
    if len(cat_rows_in) < 8:
        _log.warning("Photometry: only %d Gaia stars land inside the frame -- skipping",
                     len(cat_rows_in))
        return None

    match_radius = max(3.0, 2.0 * fwhm)
    m_local, det_idx = _match_catalog_to_detections(
        cat_pix[cat_rows_in], det_xy, match_radius)
    if len(m_local) < 6:
        _log.warning("Photometry: only %d Gaia<->detection matches -- skipping",
                     len(m_local))
        return None
    rows = cat_rows_in[m_local]

    def col(name):
        return np.asarray(catalog[name], float)[rows]

    try:
        source_id = np.asarray(catalog["source_id"])[rows]
    except Exception:
        source_id = np.arange(len(rows))

    if verbose:
        _log.info("Photometry: %d Gaia matches, FWHM %.2f px, aperture r=%d px",
                  len(rows), fwhm, ap_radius)

    return GaiaMatch(
        source_id=source_id, ra=col("ra"), dec=col("dec"),
        x=cat_pix[rows, 0], y=cat_pix[rows, 1],
        g=col("phot_g_mean_mag"), bp=col("phot_bp_mean_mag"),
        rp=col("phot_rp_mean_mag"),
        det_peak=np.asarray(sources["peak"], float)[det_idx],
        fwhm=fwhm, ap_radius=ap_radius, r_in=r_in, r_out=r_out,
        field_ra=ra0, field_dec=dec0, plate_scale=plate_scale)


def _resolve_extinction(args, airmass):
    """(k dict, X) from --photometry-extinction-k / nominal + the airmass."""
    k_override = getattr(args, "photometry_extinction_k", None)
    if k_override is not None:
        k = {c: float(k_override) for c in _CHANNELS}
    else:
        k = dict(_NOMINAL_K)
    return k, (airmass if airmass is not None else 0.0)


def _read_gain(header, args=None):
    """Sensor gain (e-/ADU): --photometry-gain override, else a PTC estimate
    stashed by the calibration builder, else the FITS header."""
    override = getattr(args, "photometry_gain", None) if args is not None else None
    if override is not None:
        try:
            if float(override) > 0:
                return float(override)
        except (TypeError, ValueError):
            pass
    ptc = getattr(args, "_ptc_gain_e_per_adu", None) if args is not None else None
    if ptc:
        try:
            if float(ptc) > 0:
                return float(ptc)
        except (TypeError, ValueError):
            pass
    for key in ("EGAIN", "GAIN"):
        v = header.get(key) if hasattr(header, "get") else None
        try:
            if v is not None and float(v) > 0:
                return float(v)
        except (TypeError, ValueError):
            pass
    return None


def _photometer_matched(gm, img_rgb, header, args, session_info):
    """Aperture-photometer a matched Gaia field on one image and fit the
    per-channel zero point (+ optional colour term). Returns a dict with
    per-star rows and the fit, or None."""
    fit_ct = bool(getattr(args, "photometry_color_terms", False))
    airmass = _airmass(header, session_info, gm.field_ra, gm.field_dec)
    k, X = _resolve_extinction(args, airmass)
    gain = _read_gain(header, args)

    flux, sky, sky_sig, ap_peak, area = aperture_photometry_batch(
        img_rgb, gm.x, gm.y, float(gm.ap_radius), gm.r_in, gm.r_out)

    sat_level = None
    for key in ("SATURATE", "DATAMAX"):
        try:
            if header.get(key) is not None:
                sat_level = float(header[key])
                break
        except (TypeError, ValueError):
            pass
    if sat_level is None:
        sat_level = 0.98 * float(np.nanmax(np.asarray(img_rgb)))

    color = gm.bp - gm.rp
    cat_channel_mag = {"R": gm.rp, "G": gm.g, "B": gm.bp}
    n = len(gm.x)
    star_peak = row_nanmax(ap_peak, gm.det_peak)
    saturated = star_peak >= sat_level

    m_inst = {c: np.full(n, np.nan) for c in _CHANNELS}
    magerr_flux = {c: np.full(n, np.nan) for c in _CHANNELS}
    snr = {c: np.full(n, np.nan) for c in _CHANNELS}
    resid = {c: np.full(n, np.nan) for c in _CHANNELS}
    for ci, ch in enumerate(_CHANNELS):
        f = flux[:, ci]
        ok = np.isfinite(f) & (f > 0)
        mi = np.where(ok, -2.5 * np.log10(np.where(ok, f, 1.0)), np.nan)
        m_inst[ch] = mi
        var = area * (sky_sig[:, ci] ** 2)
        if gain is not None:
            var = var + np.clip(f, 0, None) / gain
        se = np.sqrt(np.clip(var, 1e-12, None))
        s = np.where(ok, f / se, np.nan)
        snr[ch] = s
        magerr_flux[ch] = 1.0857 / np.clip(s, 1e-6, None)
        r = cat_channel_mag[ch] - mi + k[ch] * X
        resid[ch] = np.where(ok & ~saturated, r, np.nan)

    finite_color = color[np.isfinite(color)]
    ref_color = float(np.median(finite_color)) if finite_color.size else 0.0

    fit = {}
    for ch in _CHANNELS:
        out = _fit_zeropoint_colorterm(resid[ch], color, ref_color, fit_ct=fit_ct)
        if out is not None:
            fit[ch] = {"zp": out[0], "zp_err": out[1], "ct": out[2], "n": out[3]}
    if not fit:
        return None

    rows = []
    for i in range(n):
        row = {
            "source_id": _id_str(gm.source_id[i]),
            "x": round(float(gm.x[i]), 3), "y": round(float(gm.y[i]), 3),
            "ra_deg": round(float(gm.ra[i]), 7),
            "dec_deg": round(float(gm.dec[i]), 7),
            "gaia_g": round(float(gm.g[i]), 4),
            "gaia_bp": round(float(gm.bp[i]), 4),
            "gaia_rp": round(float(gm.rp[i]), 4),
            "saturated": int(bool(saturated[i])),
        }
        for ci, ch in enumerate(_CHANNELS):
            lc = ch.lower()
            f = flux[i, ci]
            row[f"flux_{lc}"] = round(float(f), 4) if np.isfinite(f) else ""
            if ch not in fit or not np.isfinite(m_inst[ch][i]):
                row[f"mag_{lc}"] = ""
                row[f"magerr_{lc}"] = ""
                row[f"snr_{lc}"] = ""
                continue
            zp = fit[ch]
            m_cal = (m_inst[ch][i] - k[ch] * X + zp["zp"]
                     + zp["ct"] * (float(color[i]) - ref_color))
            row[f"mag_{lc}"] = round(float(m_cal), 4)
            row[f"magerr_{lc}"] = round(
                float(math.hypot(magerr_flux[ch][i], zp["zp_err"])), 4)
            row[f"snr_{lc}"] = round(float(snr[ch][i]), 2)
        rows.append(row)

    return {
        "rows": rows, "fit": fit, "k": k, "X": X, "airmass": airmass,
        "ref_color": ref_color, "fit_ct": fit_ct, "gain": gain,
        "m_inst": m_inst, "magerr_flux": magerr_flux, "snr": snr,
        "flux": flux, "saturated": saturated,
    }


def _id_str(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return str(v)


def row_nanmax(a, fallback):
    """Per-row max ignoring NaN, without numpy's All-NaN-slice warning;
    rows that are entirely NaN take the matching ``fallback`` value."""
    a = np.asarray(a, dtype=np.float64)
    all_nan = ~np.any(np.isfinite(a), axis=1)
    out = np.where(np.isfinite(a), a, -np.inf).max(axis=1)
    return np.where(all_nan, np.asarray(fallback, dtype=np.float64), out)


_PHOT_FIELDNAMES = (["source_id", "x", "y", "ra_deg", "dec_deg",
                     "gaia_g", "gaia_bp", "gaia_rp", "saturated"]
                    + [f"{p}_{c}" for c in ("r", "g", "b")
                       for p in ("flux", "mag", "magerr", "snr")])


def run_photometry(linear_img: np.ndarray, header, args, session_info,
                   output_path: str) -> Optional[dict]:
    """Photometer the linear stack against Gaia DR3. Returns a summary dict
    (also used to stamp header keywords) or None if photometry could not run.
    """
    if linear_img.ndim != 3 or linear_img.shape[2] != 3:
        _log.warning("Photometry: needs an (H, W, 3) RGB stack -- skipping")
        return None

    gm = match_gaia_field(linear_img, header,
                          verbose=getattr(args, "verbose", False))
    if gm is None:
        return None

    phot = _photometer_matched(gm, linear_img, header, args, session_info)
    if phot is None:
        _log.warning("Photometry: zero-point fit failed on every channel -- skipping")
        return None

    csv_path = os.path.splitext(output_path)[0] + "_photometry.csv"
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_PHOT_FIELDNAMES,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(phot["rows"])
    except Exception as exc:
        _log.warning("Photometry: could not write %s: %s", csv_path, exc)
        csv_path = None

    return {
        "csv_path": csv_path,
        "n_matched": len(phot["rows"]),
        "fwhm_px": round(gm.fwhm, 2),
        "aperture_px": gm.ap_radius,
        "airmass": round(phot["airmass"], 3) if phot["airmass"] is not None else None,
        "extinction_k": {c: round(phot["k"][c], 3) for c in _CHANNELS},
        "plate_scale_arcsec": round(gm.plate_scale, 3),
        "color_terms_fitted": phot["fit_ct"],
        "ref_color": round(phot["ref_color"], 4),
        "gain_e_per_adu": round(phot["gain"], 4) if phot["gain"] else None,
        "zeropoints": {c: {"zp": round(v["zp"], 4),
                           "zp_err": round(v["zp_err"], 4),
                           "ct": round(v["ct"], 4),
                           "n": v["n"]}
                      for c, v in phot["fit"].items()},
    }


def format_photometry_summary(summary: dict) -> str:
    """One compact multi-line block for the pipeline log."""
    lines = [f"  Photometry: {summary['n_matched']} Gaia matches, "
             f"FWHM {summary['fwhm_px']}px, aperture r={summary['aperture_px']}px"]
    if summary["airmass"] is not None:
        lines.append(f"    airmass X={summary['airmass']}  "
                     f"k(R/G/B)={summary['extinction_k']['R']}/"
                     f"{summary['extinction_k']['G']}/{summary['extinction_k']['B']}")
    else:
        lines.append("    airmass: unknown (no site GPS / timestamp) -- "
                     "extinction folded into the zero point")
    if summary.get("gain_e_per_adu"):
        lines.append(f"    gain {summary['gain_e_per_adu']} e-/ADU -> Poisson "
                     "term included in per-star errors")
    ct_on = summary.get("color_terms_fitted")
    for ch in _CHANNELS:
        zp = summary["zeropoints"].get(ch)
        if not zp:
            continue
        msg = (f"    ZP_{ch} = {zp['zp']:+.4f} +/- {zp['zp_err']:.4f}  "
               f"(n={zp['n']})")
        if ct_on:
            msg += (f"   CT_{ch} = {zp['ct']:+.4f} /mag "
                    f"(ref BP-RP {summary['ref_color']})")
        lines.append(msg)
    if summary.get("csv_path"):
        lines.append(f"    catalogue: {os.path.basename(summary['csv_path'])}")
    return "\n".join(lines)
