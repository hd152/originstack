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
from typing import Optional

import numpy as np

_log = logging.getLogger("originstack")

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


def _airmass(header, session_info, ra_deg, dec_deg):
    """Airmass at the field centre for the stack's mean observation time.

    Needs site lat/long (info.json GPS) and a timestamp (header DATE-OBS,
    else info.json dateTime). Returns None when either is missing or
    astropy can't parse the time -- the caller then drops the k*X term and
    lets the zero point absorb the mean extinction.
    """
    if session_info is None or not getattr(session_info, "has_gps", False):
        return None
    time_str = None
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


def _robust_zeropoint(residuals, sigma=2.5, iters=3):
    """Sigma-clipped median of per-star (m_cat - m_inst + k*X) values."""
    r = np.asarray(residuals, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return None
    keep = np.ones(r.size, bool)
    for _ in range(iters):
        med = np.median(r[keep])
        mad = np.median(np.abs(r[keep] - med))
        scale = 1.4826 * mad
        if scale <= 0:
            break
        new_keep = np.abs(r - med) <= sigma * scale
        if new_keep.sum() < 5 or new_keep.sum() == keep.sum():
            keep = new_keep if new_keep.sum() >= 5 else keep
            break
        keep = new_keep
    used = r[keep]
    zp = float(np.median(used))
    mad = float(np.median(np.abs(used - zp)))
    zp_err = 1.4826 * mad / math.sqrt(max(used.size, 1))
    return zp, zp_err, int(used.size)


def run_photometry(linear_img: np.ndarray, header, args, session_info,
                   output_path: str) -> Optional[dict]:
    """Photometer the linear stack against Gaia DR3. Returns a summary dict
    (also used to stamp header keywords) or None if photometry could not run.
    """
    if linear_img.ndim != 3 or linear_img.shape[2] != 3:
        _log.warning("Photometry: needs an (H, W, 3) RGB stack -- skipping")
        return None

    img = np.ascontiguousarray(linear_img.astype(np.float64))
    H, W = img.shape[:2]
    luma = img.mean(axis=2)

    centre = _field_centre_and_radius(header, img.shape)
    if centre is None:
        _log.warning("Photometry: no usable WCS in the output header "
                     "(needs --plate-solve or a session info.json solve) -- skipping")
        return None
    ra0, dec0, search_radius_deg, plate_scale = centre

    from src.star_detect import detect_stars_matched_filter
    sources = detect_stars_matched_filter(luma)
    if sources is None or len(sources) < 8:
        _log.warning("Photometry: only %d stars detected -- skipping",
                     0 if sources is None else len(sources))
        return None
    det_xy = np.column_stack([
        np.asarray(sources["xcentroid"], float),
        np.asarray(sources["ycentroid"], float)])

    # FWHM for aperture sizing (fall back to a sane default).
    try:
        from src.quality import measure_fwhm
        fwhm = float(measure_fwhm(luma, sources))
    except Exception:
        fwhm = 0.0
    if not (1.0 <= fwhm <= 25.0):
        fwhm = 3.5
    ap_radius = int(max(3, round(1.6 * fwhm)))
    sky_annulus = int(max(4, round(2.0 * fwhm)))

    from src.net_query import gaia_cone_search
    gaia_cols = ["source_id", "ra", "dec",
                 "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"]
    catalog = gaia_cone_search(
        ra0, dec0, min(search_radius_deg, 2.0), gaia_cols, max_rows=1200,
        require_not_null=["phot_g_mean_mag", "phot_bp_mean_mag",
                          "phot_rp_mean_mag"])
    if catalog is None or len(catalog) < 8:
        _log.warning("Photometry: Gaia query returned %s stars -- skipping",
                     "no" if catalog is None else len(catalog))
        return None

    from src.color_calibrate import _pixel_coords, _aperture_flux
    cat_pix = _pixel_coords(catalog, header)
    if cat_pix is None:
        _log.warning("Photometry: could not project Gaia onto the image -- skipping")
        return None

    inb = ((cat_pix[:, 0] >= ap_radius + sky_annulus) &
           (cat_pix[:, 0] < W - ap_radius - sky_annulus) &
           (cat_pix[:, 1] >= ap_radius + sky_annulus) &
           (cat_pix[:, 1] < H - ap_radius - sky_annulus))
    cat_pix_in = cat_pix[inb]
    cat_rows_in = np.flatnonzero(inb)
    if len(cat_pix_in) < 8:
        _log.warning("Photometry: only %d Gaia stars land inside the frame -- skipping",
                     len(cat_pix_in))
        return None

    match_radius = max(3.0, 2.0 * fwhm)
    m_local, det_idx = _match_catalog_to_detections(cat_pix_in, det_xy, match_radius)
    if len(m_local) < 6:
        _log.warning("Photometry: only %d Gaia<->detection matches -- skipping",
                     len(m_local))
        return None
    cat_rows = cat_rows_in[m_local]
    px = cat_pix[cat_rows, 0]
    py = cat_pix[cat_rows, 1]

    fluxes = _aperture_flux(img, px, py, radius=ap_radius, sky_annulus=sky_annulus)
    peaks = np.asarray(sources["peak"], float)[det_idx]

    g_mag = np.asarray(catalog["phot_g_mean_mag"], float)[cat_rows]
    bp_mag = np.asarray(catalog["phot_bp_mean_mag"], float)[cat_rows]
    rp_mag = np.asarray(catalog["phot_rp_mean_mag"], float)[cat_rows]
    try:
        source_id = np.asarray(catalog["source_id"])[cat_rows]
    except Exception:
        source_id = np.arange(len(cat_rows))
    cat_ra = np.asarray(catalog["ra"], float)[cat_rows]
    cat_dec = np.asarray(catalog["dec"], float)[cat_rows]
    cat_channel_mag = {"R": rp_mag, "G": g_mag, "B": bp_mag}

    # Airmass + extinction coefficients.
    airmass = _airmass(header, session_info, ra0, dec0)
    k_override = getattr(args, "photometry_extinction_k", None)
    if k_override is not None:
        k = {c: float(k_override) for c in _CHANNELS}
    else:
        k = dict(_NOMINAL_K)
    X = airmass if airmass is not None else 0.0

    # Optional Poisson term: only when the header gives a real gain (e-/ADU).
    gain = None
    for key in ("EGAIN", "GAIN"):
        v = header.get(key)
        try:
            if v is not None and float(v) > 0:
                gain = float(v)
                break
        except (TypeError, ValueError):
            pass

    n_ap_pix = math.pi * ap_radius ** 2
    sat_level = None
    for key in ("SATURATE", "DATAMAX"):
        try:
            if header.get(key) is not None:
                sat_level = float(header[key])
                break
        except (TypeError, ValueError):
            pass
    if sat_level is None:
        sat_level = 0.98 * float(np.nanmax(img))

    result_rows = []
    per_channel_resid = {c: [] for c in _CHANNELS}
    for i in range(len(cat_rows)):
        row = {
            "source_id": int(source_id[i]) if np.isscalar(source_id[i]) or
            isinstance(source_id[i], (np.integer,)) else str(source_id[i]),
            "x": round(float(px[i]), 3), "y": round(float(py[i]), 3),
            "ra_deg": round(float(cat_ra[i]), 7), "dec_deg": round(float(cat_dec[i]), 7),
            "gaia_g": round(float(g_mag[i]), 4),
            "gaia_bp": round(float(bp_mag[i]), 4),
            "gaia_rp": round(float(rp_mag[i]), 4),
            "saturated": int(bool(peaks[i] >= sat_level)),
        }
        for ci, ch in enumerate(_CHANNELS):
            flux = float(fluxes[i, ci]) if np.isfinite(fluxes[i, ci]) else np.nan
            row[f"flux_{ch.lower()}"] = (round(flux, 4)
                                        if np.isfinite(flux) else "")
            if not np.isfinite(flux) or flux <= 0:
                row[f"mag_{ch.lower()}"] = ""
                row[f"magerr_{ch.lower()}"] = ""
                row[f"snr_{ch.lower()}"] = ""
                continue
            m_inst = -2.5 * math.log10(flux)
            # Residual toward the zero point:  ZP = m_cat - m_inst + k*X
            resid = float(cat_channel_mag[ch][i]) - m_inst + k[ch] * X
            if np.isfinite(resid) and not row["saturated"]:
                per_channel_resid[ch].append(resid)
            # Flux uncertainty: sky shot noise over the aperture (+ Poisson
            # when a real gain is known).
            var = n_ap_pix * _sky_variance(img[..., ci], px[i], py[i],
                                           ap_radius, sky_annulus)
            if gain is not None:
                var += flux / gain
            flux_err = math.sqrt(max(var, 1e-12))
            snr = flux / flux_err
            row[f"snr_{ch.lower()}"] = round(float(snr), 2)
            row["_m_inst_" + ch] = m_inst
            row["_magerr_flux_" + ch] = 1.0857 / max(snr, 1e-6)
        result_rows.append(row)

    zeropoints = {}
    for ch in _CHANNELS:
        zp = _robust_zeropoint(per_channel_resid[ch])
        if zp is not None:
            zeropoints[ch] = {"zp": zp[0], "zp_err": zp[1], "n": zp[2]}

    if not zeropoints:
        _log.warning("Photometry: zero-point fit failed on every channel -- skipping")
        return None

    # Second pass: calibrated magnitudes now that ZP is known.
    for row in result_rows:
        for ch in _CHANNELS:
            key_inst = "_m_inst_" + ch
            if key_inst not in row or ch not in zeropoints:
                row.setdefault(f"mag_{ch.lower()}", "")
                row.setdefault(f"magerr_{ch.lower()}", "")
                continue
            zp = zeropoints[ch]
            m_cal = row[key_inst] - k[ch] * X + zp["zp"]
            magerr = math.hypot(row.get("_magerr_flux_" + ch, 0.0), zp["zp_err"])
            row[f"mag_{ch.lower()}"] = round(float(m_cal), 4)
            row[f"magerr_{ch.lower()}"] = round(float(magerr), 4)

    # Strip private scratch keys before writing.
    clean_rows = []
    for row in result_rows:
        clean_rows.append({key: val for key, val in row.items()
                           if not key.startswith("_")})

    csv_path = os.path.splitext(output_path)[0] + "_photometry.csv"
    fieldnames = ["source_id", "x", "y", "ra_deg", "dec_deg",
                  "gaia_g", "gaia_bp", "gaia_rp", "saturated"]
    for ch in ("r", "g", "b"):
        fieldnames += [f"flux_{ch}", f"mag_{ch}", f"magerr_{ch}", f"snr_{ch}"]
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in clean_rows:
                writer.writerow(row)
    except Exception as exc:
        _log.warning("Photometry: could not write %s: %s", csv_path, exc)
        csv_path = None

    summary = {
        "csv_path": csv_path,
        "n_matched": len(clean_rows),
        "fwhm_px": round(fwhm, 2),
        "aperture_px": ap_radius,
        "airmass": round(airmass, 3) if airmass is not None else None,
        "extinction_k": {c: round(k[c], 3) for c in _CHANNELS},
        "plate_scale_arcsec": round(plate_scale, 3),
        "zeropoints": {c: {"zp": round(v["zp"], 4),
                            "zp_err": round(v["zp_err"], 4),
                            "n": v["n"]}
                       for c, v in zeropoints.items()},
    }
    return summary


def _sky_variance(channel, cx, cy, r_in, annulus_width):
    """Per-pixel sky variance from a robust MAD over the sky annulus."""
    r_out = r_in + annulus_width
    x0 = int(round(cx)) - r_out
    y0 = int(round(cy)) - r_out
    patch = channel[max(y0, 0):y0 + 2 * r_out + 1, max(x0, 0):x0 + 2 * r_out + 1]
    if patch.size == 0:
        return 1.0
    yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    cyl = cy - max(y0, 0)
    cxl = cx - max(x0, 0)
    rr = (yy - cyl) ** 2 + (xx - cxl) ** 2
    sky = patch[(rr > r_in ** 2) & (rr <= r_out ** 2)]
    if sky.size < 8:
        return float(np.var(patch)) if patch.size else 1.0
    med = np.median(sky)
    mad = np.median(np.abs(sky - med))
    sigma = 1.4826 * mad
    return float(sigma ** 2) if sigma > 0 else float(np.var(sky))


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
    for ch in _CHANNELS:
        zp = summary["zeropoints"].get(ch)
        if zp:
            lines.append(f"    ZP_{ch} = {zp['zp']:+.4f} +/- {zp['zp_err']:.4f}  "
                         f"(n={zp['n']})")
    if summary.get("csv_path"):
        lines.append(f"    catalogue: {os.path.basename(summary['csv_path'])}")
    return "\n".join(lines)
