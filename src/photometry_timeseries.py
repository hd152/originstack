"""Per-frame differential photometry / light curves (--photometry-timeseries).

The streaming stacker collapses the time axis; this runs a separate pass
over the same registered frames (still in memory right after Phase 3),
aperture-photometering a fixed Gaia-matched star list on every sub, then
ensemble-differential-calibrating so a per-frame zero point removes
transparency / airmass drift. Output is two CSVs:

  <output>_lightcurves.csv       long format, one row per (frame, star)
  <output>_lightcurve_stats.csv  one row per star: mean, rms, reduced chi2,
                                 a crude variability flag

Needs a WCS on the stacked grid to build the star list. Only the session
info.json solve (Celestron Origin) is available this early in the pipeline
-- --plate-solve runs after Phase 4 -- so this is skipped with a message
when there is no session WCS. Differential only: no absolute zero point,
extinction cancels in the ensemble (same field), airmass is recorded but
not applied.
"""
from __future__ import annotations

import csv
import logging
import math
import os
import warnings
from typing import Optional

import numpy as np

from src.photometry import _GAIA_BAND_FOR_CHANNEL, _airmass, _read_gain, match_gaia_field
from src.photometry_core import _id_str, aperture_photometry_batch, row_nanmax
from src.utils import header_get_first

_log = logging.getLogger("originstack")
_CH = ("r", "g", "b")


def _cropped_session_wcs_header(session_info, shape_hw, left, top):
    """FITS header carrying the session (Origin) WCS shifted onto the
    cropped stacked grid, or None if the session provides no WCS."""
    if session_info is None or not getattr(session_info, "has_wcs", False):
        return None
    try:
        from astropy.io import fits

        from src.session_info import build_wcs_keywords
        kw = build_wcs_keywords(session_info)
        if not kw:
            return None
        h = fits.Header()
        H, W = shape_hw[:2]
        h["NAXIS"] = 2
        h["NAXIS1"] = int(W)
        h["NAXIS2"] = int(H)
        for key, val in kw.items():
            h[key] = val
        if "CRPIX1" in h:
            h["CRPIX1"] = float(h["CRPIX1"]) - float(left)
        if "CRPIX2" in h:
            h["CRPIX2"] = float(h["CRPIX2"]) - float(top)
        return h
    except Exception as exc:
        _log.debug("Time-series photometry: WCS header build failed: %s", exc)
        return None


def _frame_time_iso(frame, session_info, j, n):
    """ISO UTC timestamp for sub *j*: the frame's own DATE-OBS if present,
    else interpolated from the session start + total duration."""
    hdr = getattr(frame, "header", {}) or {}
    own = header_get_first(hdr, ("DATE-OBS", "DATE_OBS", "DATEOBS"), cast=str)
    if own:
        return own
    start = getattr(session_info, "date_time", None) if session_info else None
    dur_ms = getattr(session_info, "total_duration_ms", None) if session_info else None
    if start and dur_ms and n > 1:
        try:
            from astropy.time import Time, TimeDelta
            t0 = Time(str(start))
            dt = TimeDelta((dur_ms / 1000.0) * (j / (n - 1)), format="sec")
            return (t0 + dt).isot
        except Exception:
            return start
    return start


def _to_mjd(iso):
    if not iso:
        return np.nan
    try:
        from astropy.time import Time
        return float(Time(str(iso)).mjd)
    except Exception:
        return np.nan


def _parse_target(spec, gm, header):
    """Resolve --photometry-target to an index into the matched star list.

    Accepts 'RA,DEC' in degrees, or 'px:X,Y' in cropped-stack pixels.
    Returns the nearest matched-star index within the match tolerance, or
    None.
    """
    if not spec:
        return None
    try:
        s = str(spec).strip()
        if s.lower().startswith("px:"):
            xs, ys = s[3:].split(",")
            tx, ty = float(xs), float(ys)
        else:
            ra_s, dec_s = s.replace(";", ",").split(",")
            ra, dec = float(ra_s), float(dec_s)
            from astropy.wcs import WCS
            w = WCS(header).celestial
            tx, ty = [float(v) for v in w.all_world2pix(ra, dec, 0)]
    except Exception as exc:
        _log.warning("Time-series photometry: could not parse --photometry-target "
                     "%r (%s)", spec, exc)
        return None
    d2 = (gm.x - tx) ** 2 + (gm.y - ty) ** 2
    i = int(np.argmin(d2))
    tol = max(3.0, 2.0 * gm.fwhm)
    if math.sqrt(d2[i]) > tol:
        _log.warning("Time-series photometry: --photometry-target has no Gaia "
                     "match within %.1f px", tol)
        return None
    return i


def run_timeseries_photometry(final, final_indices, mem_rgb, shifts, transforms,
                              displacement_fields, crop, stacked_linear,
                              session_info, args, output_path) -> Optional[dict]:
    top, bottom, left, right = crop
    header = _cropped_session_wcs_header(session_info, stacked_linear.shape, left, top)
    if header is None:
        _log.info("Time-series photometry: no session info.json WCS -- skipping "
                  "(--plate-solve runs too late in the pipeline for per-frame "
                  "photometry)")
        return None

    gm = match_gaia_field(stacked_linear, header,
                          verbose=getattr(args, "verbose", False))
    if gm is None:
        return None

    from src.registration import apply_transform

    n_frames = len(final)
    n_star = len(gm.x)
    gaia_mag = {c: getattr(gm, _GAIA_BAND_FOR_CHANNEL[c.upper()]) for c in _CH}
    gain = _read_gain(header, args)

    flux = np.full((n_frames, n_star, 3), np.nan)
    ferr = np.full((n_frames, n_star, 3), np.nan)
    fpeak = np.full((n_frames, n_star), np.nan)
    mjd = np.full(n_frames, np.nan)
    airmass = np.full(n_frames, np.nan)
    fnames = []
    obs_ceiling = 0.0     # max pixel value actually seen across the subs

    for j in range(n_frames):
        idx = final_indices[j]
        frame = np.ascontiguousarray(np.asarray(mem_rgb[idx], dtype=np.float32))
        lf = None
        if displacement_fields is not None and j < len(displacement_fields):
            lf = displacement_fields[j]
        aligned = apply_transform(frame, shift=shifts[j],
                                  transform=transforms[j], local_field=lf)
        sub = np.ascontiguousarray(aligned[top:bottom, left:right])
        obs_ceiling = max(obs_ceiling, float(np.nanmax(sub)))
        f, _sky, sig, pk, area = aperture_photometry_batch(
            sub, gm.x, gm.y, float(gm.ap_radius), gm.r_in, gm.r_out)
        flux[j] = f
        var = area[:, None] * (sig ** 2)
        if gain is not None:
            var = var + np.clip(f, 0.0, None) / gain
        ferr[j] = np.sqrt(np.clip(var, 1e-12, None))
        fpeak[j] = row_nanmax(pk, np.zeros(n_star))
        iso = _frame_time_iso(final[j], session_info, j, n_frames)
        mjd[j] = _to_mjd(iso)
        airmass[j] = _airmass(header, session_info, gm.field_ra, gm.field_dec,
                              when=iso) or np.nan
        fnames.append(os.path.basename(getattr(final[j], "path", f"frame{j}")))

    # Instrumental magnitudes.
    with np.errstate(invalid="ignore", divide="ignore"):
        m_inst = np.where(flux > 0, -2.5 * np.log10(np.where(flux > 0, flux, 1.0)),
                          np.nan)
        magerr = 1.0857 * ferr / np.where(flux > 0, flux, np.nan)

    # Only a pixel at the actual observed ceiling counts as saturated -- a
    # bright star's per-frame peak routinely exceeds a fraction of the
    # (fainter) mean-stack max, so a stack-derived threshold would wrongly
    # exclude the best comparison stars.
    _hdr_sat = header_get_first(header, ("SATURATE", "DATAMAX"), cast=float)
    sat_level = _hdr_sat if _hdr_sat is not None else obs_ceiling * (1.0 - 1e-6)
    ever_sat = np.any(fpeak >= sat_level, axis=0)
    finite_frac = np.mean(np.isfinite(m_inst[:, :, 1]), axis=0)

    # Ensemble comparison stars: well-detected, unsaturated, mid-brightness.
    g_lo, g_hi = np.nanpercentile(gm.g, [15, 85])
    ensemble = (finite_frac >= 0.8) & (~ever_sat) & (gm.g >= g_lo) & (gm.g <= g_hi)
    target_idx = _parse_target(getattr(args, "photometry_target", None), gm, header)
    if target_idx is not None:
        ensemble[target_idx] = False
    if ensemble.sum() < 3:
        ensemble = (finite_frac >= 0.8) & (~ever_sat)
        if target_idx is not None:
            ensemble[target_idx] = False
    if ensemble.sum() < 3:
        _log.warning("Time-series photometry: only %d usable comparison stars "
                     "-- light curves will be noisy", int(ensemble.sum()))

    # Iterative ensemble differential zero point (per channel, per frame).
    # A frame or star with no finite ensemble residual makes nanmedian/nanstd
    # emit an "All-NaN slice" RuntimeWarning; the NaN it returns is handled
    # downstream as a missing point, so silence just this block.
    zp = np.zeros((n_frames, 3))
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(4):
            for c in range(3):
                diff = gaia_mag[_CH[c]][None, :] - m_inst[:, :, c]
                zp[:, c] = np.nanmedian(
                    np.where(ensemble[None, :], diff, np.nan), axis=1)
            m_corr = m_inst + zp[:, None, :]
            rms_g = np.nanstd(m_corr[:, :, 1], axis=0)
            if ensemble.sum() <= 3:
                break
            thr = np.nanmedian(rms_g[ensemble]) + 2.0 * 1.4826 * np.nanmedian(
                np.abs(rms_g[ensemble] - np.nanmedian(rms_g[ensemble])))
            new_ens = ensemble & (rms_g <= max(thr, 1e-6))
            if new_ens.sum() < 3 or new_ens.sum() == ensemble.sum():
                ensemble = new_ens if new_ens.sum() >= 3 else ensemble
                break
            ensemble = new_ens

        m_corr = m_inst + zp[:, None, :]
        # per-frame zero-point scatter folds into every star's error
        zp_err = np.zeros((n_frames, 3))
        for c in range(3):
            r = np.where(ensemble[None, :],
                         gaia_mag[_CH[c]][None, :] - m_corr[:, :, c], np.nan)
            n_ens = max(int(ensemble.sum()), 1)
            zp_err[:, c] = 1.4826 * np.nanmedian(
                np.abs(r - np.nanmedian(r, axis=1, keepdims=True)),
                axis=1) / math.sqrt(n_ens)
    zp_err = np.nan_to_num(zp_err, nan=0.0)
    tot_err = np.sqrt(magerr ** 2 + (zp_err[:, None, :]) ** 2)

    lc_path = os.path.splitext(output_path)[0] + "_lightcurves.csv"
    stats_path = os.path.splitext(output_path)[0] + "_lightcurve_stats.csv"

    lc_fields = (["frame", "filename", "mjd", "airmass", "source_id", "x", "y",
                  "gaia_g", "is_target"]
                 + [f"{p}_{c}" for c in _CH for p in ("mag", "magerr")]
                 + ["flag"])
    try:
        with open(lc_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=lc_fields, extrasaction="ignore")
            w.writeheader()
            for j in range(n_frames):
                for i in range(n_star):
                    flag = ""
                    if fpeak[j, i] >= sat_level:
                        flag = "s"
                    elif not np.isfinite(m_inst[j, i, 1]):
                        flag = "e"
                    row = {
                        "frame": j, "filename": fnames[j],
                        "mjd": round(float(mjd[j]), 8) if np.isfinite(mjd[j]) else "",
                        "airmass": round(float(airmass[j]), 4) if np.isfinite(airmass[j]) else "",
                        "source_id": _id_str(gm.source_id[i]),
                        "x": round(float(gm.x[i]), 3), "y": round(float(gm.y[i]), 3),
                        "gaia_g": round(float(gm.g[i]), 4),
                        "is_target": int(i == target_idx),
                        "flag": flag,
                    }
                    for c, ch in enumerate(_CH):
                        mv = m_corr[j, i, c]
                        row[f"mag_{ch}"] = round(float(mv), 4) if np.isfinite(mv) else ""
                        ev = tot_err[j, i, c]
                        row[f"magerr_{ch}"] = round(float(ev), 4) if np.isfinite(ev) else ""
                    w.writerow(row)
    except Exception as exc:
        _log.warning("Time-series photometry: could not write %s: %s", lc_path, exc)
        lc_path = None

    # Per-star summary.
    n_variable = 0
    st_fields = ["source_id", "x", "y", "gaia_g", "gaia_bp_rp", "is_target",
                 "is_ensemble", "n_points", "n_saturated",
                 "mean_mag_g", "rms_g", "mad_g", "ptp_g", "chi2red_g",
                 "mean_mag_r", "rms_r", "mean_mag_b", "rms_b", "variable"]
    stats_rows = []
    for i in range(n_star):
        g = m_corr[:, i, 1]
        ok = np.isfinite(g)
        npts = int(ok.sum())
        row = {
            "source_id": _id_str(gm.source_id[i]),
            "x": round(float(gm.x[i]), 3), "y": round(float(gm.y[i]), 3),
            "gaia_g": round(float(gm.g[i]), 4),
            "gaia_bp_rp": round(float(gm.bp[i] - gm.rp[i]), 4),
            "is_target": int(i == target_idx),
            "is_ensemble": int(bool(ensemble[i])),
            "n_points": npts,
            "n_saturated": int(np.nansum(fpeak[:, i] >= sat_level)),
        }
        if npts >= 3:
            gm_mean = float(np.nanmean(g))
            gm_rms = float(np.nanstd(g))
            gm_mad = float(1.4826 * np.nanmedian(np.abs(g[ok] - np.nanmedian(g[ok]))))
            gm_ptp = float(np.nanmax(g[ok]) - np.nanmin(g[ok]))
            e = tot_err[:, i, 1]
            e_ok = e[ok & np.isfinite(e)]
            g_ok = g[ok & np.isfinite(e)]
            if e_ok.size >= 3:
                chi2red = float(np.sum(((g_ok - gm_mean) / e_ok) ** 2) / (e_ok.size - 1))
                med_e = float(np.median(e_ok))
            else:
                chi2red, med_e = float("nan"), float("nan")
            variable = int(np.isfinite(chi2red) and chi2red > 3.0
                           and gm_rms > 3.0 * med_e
                           and row["n_saturated"] == 0
                           and not bool(ensemble[i]))
            n_variable += variable
            row.update({
                "mean_mag_g": round(gm_mean, 4), "rms_g": round(gm_rms, 4),
                "mad_g": round(gm_mad, 4), "ptp_g": round(gm_ptp, 4),
                "chi2red_g": round(chi2red, 3) if np.isfinite(chi2red) else "",
                "mean_mag_r": round(float(np.nanmean(m_corr[:, i, 0])), 4),
                "rms_r": round(float(np.nanstd(m_corr[:, i, 0])), 4),
                "mean_mag_b": round(float(np.nanmean(m_corr[:, i, 2])), 4),
                "rms_b": round(float(np.nanstd(m_corr[:, i, 2])), 4),
                "variable": variable,
            })
        else:
            row.update({k: "" for k in st_fields[9:]})
            row["variable"] = 0
        stats_rows.append(row)

    try:
        with open(stats_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=st_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(stats_rows)
    except Exception as exc:
        _log.warning("Time-series photometry: could not write %s: %s", stats_path, exc)
        stats_path = None

    valid_mjd = mjd[np.isfinite(mjd)]
    span_min = (float(valid_mjd.max() - valid_mjd.min()) * 1440.0
                if valid_mjd.size >= 2 else None)

    target_stats = None
    if target_idx is not None and stats_rows[target_idx].get("mean_mag_g") != "":
        tr = stats_rows[target_idx]
        target_stats = {"source_id": tr["source_id"], "mean_mag_g": tr["mean_mag_g"],
                        "rms_g": tr["rms_g"], "chi2red_g": tr["chi2red_g"],
                        "n_points": tr["n_points"]}

    return {
        "lightcurves_csv": lc_path,
        "stats_csv": stats_path,
        "n_frames": n_frames,
        "n_stars": n_star,
        "n_ensemble": int(ensemble.sum()),
        "n_variable_candidates": int(n_variable),
        "span_minutes": round(span_min, 1) if span_min else None,
        "median_zp_scatter_g": round(float(np.nanmedian(zp_err[:, 1])), 4),
        "gain_e_per_adu": round(gain, 4) if gain else None,
        "target": target_stats,
    }


def format_timeseries_summary(s: dict) -> str:
    lines = [f"  Time-series photometry: {s['n_stars']} stars x {s['n_frames']} "
             f"frames, {s['n_ensemble']} comparison stars"]
    if s["span_minutes"]:
        lines.append(f"    baseline {s['span_minutes']} min, "
                     f"ensemble zp scatter {s['median_zp_scatter_g']} mag (G)")
    lines.append(f"    variability candidates: {s['n_variable_candidates']}")
    if s.get("target"):
        t = s["target"]
        lines.append(f"    target {t['source_id']}: <G>={t['mean_mag_g']} "
                     f"rms={t['rms_g']} chi2red={t['chi2red_g']} (n={t['n_points']})")
    for key in ("lightcurves_csv", "stats_csv"):
        if s.get(key):
            lines.append(f"    {os.path.basename(s[key])}")
    return "\n".join(lines)
