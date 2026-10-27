"""AAVSO variable-star reports from a time-series run (``--aavso-report``).

Turns ``--photometry-timeseries``'s per-frame light curves into a file in the AAVSO
Extended Format, ready for WebObs upload:

- **Comparison stars from AAVSO's own sequence** (VSP chart, ``net_query.aavso_vsp_chart``),
  not Gaia: AAVSO wants standard V-band comparison stars and the chart ID. Each VSP star is
  found among the measured stars through the stack WCS.
- **Ensemble, re-zeroed per frame**: the target's magnitude is its green-channel
  instrumental magnitude plus the per-frame median of (V - instrumental) over the
  comparison stars (``CNAME=ENSEMBLE``). The green channel of a colour camera is AAVSO's
  ``TG`` band; no transformation is applied (``TRANS=NO``).
- **A check star**: the comparison star closest in brightness to the target is held out of
  the ensemble and its measured magnitude reported (``KNAME``/``KMAG``) -- how well the
  calibration reproduces a known star is the reviewer's first test, and the summary prints it.
- **Times** are mid-exposure JD (UTC), the frame's DATE-OBS plus half the exposure.
- **Binning** (``--aavso-bin`` minutes): single short subs are noisy; frames are averaged
  in time bins (inverse-variance weighted; the error is the larger of the propagated error
  and the bin's scatter over sqrt(n)).

The light-curve CSV's magnitudes already carry a Gaia ensemble zero point per frame; the
VSP re-zeroing replaces it, so the result is on the comparison stars' V scale.
"""
from __future__ import annotations

import csv
import logging
import math
from typing import Dict, Optional

import numpy as np

_log = logging.getLogger("originstack")

MIN_COMPS = 2          # ensemble stars needed in a frame (plus the check star)
MATCH_PX = 2.5         # VSP star to measured star, px


def _sexa_to_deg(ra: str, dec: str):
    import astropy.units as u
    from astropy.coordinates import Angle
    return float(Angle(ra, unit=u.hourangle).deg), float(Angle(dec, unit=u.deg).deg)


def parse_vsp(chart: dict) -> Optional[dict]:
    """``{"chartid", "star", "auid", "ra", "dec", "comps": [{auid, label, ra, dec, v, v_err, b}]}``
    (degrees) from a VSP chart, keeping comparison stars with a V magnitude."""
    if not chart or not isinstance(chart.get("photometry"), list):
        return None
    comps = []
    for p in chart["photometry"]:
        bands = {b.get("band"): b for b in p.get("bands") or []}
        v = bands.get("V")
        if not v or v.get("mag") is None:
            continue
        try:
            ra, dec = _sexa_to_deg(p["ra"], p["dec"])
        except Exception:
            continue
        b = bands.get("B")
        comps.append({"auid": p.get("auid") or "", "label": str(p.get("label") or ""), "ra": ra, "dec": dec,
                      "v": float(v["mag"]), "v_err": float(v.get("error") or 0.0),
                      "b": float(b["mag"]) if b and b.get("mag") is not None else None})
    out = {"chartid": chart.get("chartid") or "na", "star": chart.get("star"), "auid": chart.get("auid"),
           "ra": None, "dec": None, "comps": comps}
    if chart.get("ra") and chart.get("dec"):
        try:
            out["ra"], out["dec"] = _sexa_to_deg(chart["ra"], chart["dec"])
        except Exception:
            pass
    return out


def read_lightcurves(path: str):
    """``(frames, stars)`` from ``<output>_lightcurves.csv``: frames[j] = {mjd, airmass};
    stars[source_id] = {x, y, mag (n_frames,), err (n_frames,)} for the green channel, NaN
    where missing or saturated."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    n = 1 + max(int(r["frame"]) for r in rows)
    frames = [{"mjd": math.nan, "airmass": math.nan} for _ in range(n)]
    stars: Dict[str, dict] = {}

    def num(s):
        try:
            return float(s)
        except (TypeError, ValueError):
            return math.nan

    for r in rows:
        j = int(r["frame"])
        frames[j]["mjd"] = num(r["mjd"])
        frames[j]["airmass"] = num(r["airmass"])
        s = stars.get(r["source_id"])
        if s is None:
            s = stars[r["source_id"]] = {"x": num(r["x"]), "y": num(r["y"]),
                                         "mag": np.full(n, np.nan), "err": np.full(n, np.nan)}
        if r.get("flag") in ("s", "e"):
            continue
        s["mag"][j], s["err"][j] = num(r["mag_g"]), num(r["magerr_g"])
    return frames, stars


def _nearest(stars: dict, x: float, y: float):
    best, d = None, math.inf
    for sid, s in stars.items():
        dd = math.hypot(s["x"] - x, s["y"] - y)
        if dd < d:
            best, d = sid, dd
    return (best, d) if d <= MATCH_PX else (None, d)


def _reject_outlier_comps(comps: list, stars: dict, k: float = 3.0, floor: float = 0.1):
    """Drop comparison stars whose offset ``V - instrumental`` (median over frames, after the
    per-frame ensemble zero point) stands out from the others: saturated, blended or sitting
    on bright nebulosity. On the Orion Nebula, V 2.8-9.0 comparison stars read 0.6-4 mag off
    and, left in, pulled every zero point by ~0.1 mag. Iterates; returns (kept, rejected)."""
    kept, rejected = list(comps), []
    while len(kept) > MIN_COMPS + 1:
        m = np.array([stars[c["sid"]]["mag"] for c in kept])            # (n_comp, n_frames)
        v = np.array([c["v"] for c in kept])[:, None]
        zp = np.nanmedian(v - m, axis=0)
        off = np.nanmedian(v - m - zp[None, :], axis=1)
        dev = np.abs(off - np.nanmedian(off))
        lim = max(k * 1.4826 * float(np.nanmedian(dev)), floor)
        worst = int(np.nanargmax(dev))
        if not dev[worst] > lim:
            break
        rejected.append(kept.pop(worst))
    return kept, rejected


def build_report(lc_csv: str, wcs_header, vsp: dict, *, target_ra: Optional[float] = None,
                 target_dec: Optional[float] = None, star_name: Optional[str] = None,
                 obscode: str = "", exptime_s: float = 0.0, bin_minutes: float = 0.0,
                 software: str = "OriginStack") -> dict:
    """The report text and a summary dict, or ``{"error": reason}``."""
    from astropy.wcs import WCS
    frames, stars = read_lightcurves(lc_csv)
    w = WCS(wcs_header).celestial
    ra = target_ra if target_ra is not None else vsp.get("ra")
    dec = target_dec if target_dec is not None else vsp.get("dec")
    if ra is None or dec is None:
        return {"error": "no target position (give --aavso-star or --photometry-target RA,DEC)"}
    tx, ty = (float(v) for v in w.all_world2pix(ra, dec, 0))
    tid, td = _nearest(stars, tx, ty)
    if tid is None:
        return {"error": f"the target is not among the measured stars (nearest {td:.1f} px; it may be "
                         "too faint, saturated or outside the stacked field)"}
    comps = []
    for c in vsp["comps"]:
        cx, cy = (float(v) for v in w.all_world2pix(c["ra"], c["dec"], 0))
        sid, _ = _nearest(stars, cx, cy)
        if sid is not None and sid != tid and np.isfinite(stars[sid]["mag"]).sum() >= 3:
            comps.append(dict(c, sid=sid))
    comps, rejected = _reject_outlier_comps(comps, stars)
    if len(comps) < MIN_COMPS + 1:
        return {"error": f"only {len(comps)} usable VSP comparison stars measured (need {MIN_COMPS + 1}: "
                         "an ensemble and a check star); try a wider --aavso-fov or a fainter --aavso-maglimit"}
    t = stars[tid]
    t_mean = float(np.nanmedian(t["mag"]))
    # check star: closest in instrumental brightness to the target
    inst = [float(np.nanmedian(stars[c["sid"]]["mag"])) for c in comps]
    k = int(np.argmin([abs(m - t_mean) for m in inst]))
    check, ens = comps[k], comps[:k] + comps[k + 1:]
    n = len(frames)
    zp = np.full(n, np.nan)
    zp_err = np.full(n, np.nan)
    n_used = np.zeros(n, int)
    for j in range(n):
        d = np.array([c["v"] - stars[c["sid"]]["mag"][j] for c in ens])
        d = d[np.isfinite(d)]
        if d.size >= MIN_COMPS:
            zp[j] = float(np.median(d))
            mad = 1.4826 * float(np.median(np.abs(d - zp[j])))
            zp_err[j] = (mad if mad > 0 else float(np.std(d))) / math.sqrt(d.size)
            n_used[j] = d.size
    mag = t["mag"] + zp
    err = np.sqrt(t["err"] ** 2 + zp_err ** 2)
    kmag = stars[check["sid"]]["mag"] + zp
    mjd = np.array([f["mjd"] for f in frames]) + exptime_s / 2.0 / 86400.0
    am = np.array([f["airmass"] for f in frames])
    ok = np.isfinite(mag) & np.isfinite(err) & (err > 0) & np.isfinite(mjd)
    if ok.sum() == 0:
        return {"error": "no frame had the target and enough comparison stars measured"}
    idx = np.flatnonzero(ok)
    if bin_minutes and bin_minutes > 0:
        key = np.floor((mjd[idx] - mjd[idx].min()) * 1440.0 / bin_minutes).astype(int)
        groups = [idx[key == g] for g in np.unique(key)]
    else:
        groups = [np.array([i]) for i in idx]
    rows = []
    for g in groups:
        wts = 1.0 / err[g] ** 2
        m = float(np.sum(wts * mag[g]) / np.sum(wts))
        e = math.sqrt(1.0 / float(np.sum(wts)))
        if g.size > 1:
            e = max(e, float(np.std(mag[g], ddof=1)) / math.sqrt(g.size))
        kk = kmag[g][np.isfinite(kmag[g])]
        rows.append({"jd": float(np.mean(mjd[g])) + 2400000.5, "mag": m, "err": e,
                     "kmag": float(np.mean(kk)) if kk.size else math.nan,
                     "airmass": float(np.nanmean(am[g])) if np.isfinite(am[g]).any() else math.nan,
                     "n": int(g.size), "n_comp": int(np.median(n_used[g]))})
    name = star_name or vsp.get("star") or f"{ra:.5f} {dec:+.5f}"
    kname = check["auid"] or check["label"]
    lines = ["#TYPE=EXTENDED", f"#OBSCODE={obscode or 'CHANGEME'}", f"#SOFTWARE={software}",
             "#DELIM=,", "#DATE=JD", "#OBSTYPE=DSLR",
             "#NAME,DATE,MAG,MERR,FILT,TRANS,MTYPE,CNAME,CMAG,KNAME,KMAG,AMASS,GROUP,CHART,NOTES"]
    for r in rows:
        note = (f"ensemble of {r['n_comp']} VSP comps|{r['n']} x {exptime_s:g}s"
                if r["n"] > 1 else f"ensemble of {r['n_comp']} VSP comps")
        lines.append(",".join([
            name.replace(",", " "), f"{r['jd']:.5f}", f"{r['mag']:.3f}", f"{r['err']:.3f}", "TG", "NO", "STD",
            "ENSEMBLE", "na", kname, f"{r['kmag']:.3f}" if np.isfinite(r["kmag"]) else "na",
            f"{r['airmass']:.3f}" if np.isfinite(r["airmass"]) else "na", "na", vsp.get("chartid") or "na",
            note]))
    kk = kmag[ok][np.isfinite(kmag[ok])]
    return {"text": "\n".join(lines) + "\n", "n_points": len(rows), "n_frames": int(ok.sum()),
            "n_comps": len(ens), "rejected": [c["label"] for c in rejected],
            "chartid": vsp.get("chartid"), "star": name,
            "target_mean": float(np.mean([r["mag"] for r in rows])),
            "target_rms": float(np.std([r["mag"] for r in rows])) if len(rows) > 1 else 0.0,
            "median_err": float(np.median([r["err"] for r in rows])),
            "check": {"name": kname, "label": check["label"], "v": check["v"],
                      "measured": float(np.mean(kk)) if kk.size else math.nan,
                      "rms": float(np.std(kk)) if kk.size > 1 else math.nan},
            "obscode_missing": not obscode, "name_missing": not (star_name or vsp.get("star"))}


def format_summary(s: dict, path: Optional[str] = None) -> str:
    if "error" in s:
        return f"  AAVSO report: not written -- {s['error']}"
    c = s["check"]
    lines = [f"  AAVSO report: {s['star']}, {s['n_points']} points from {s['n_frames']} frames, "
             f"ensemble of {s['n_comps']} comparison stars (chart {s['chartid']})"
             + (f", {len(s['rejected'])} rejected as outliers ({', '.join(s['rejected'])})" if s.get("rejected") else ""),
             f"    target TG {s['target_mean']:.3f} (rms {s['target_rms']:.3f}, median error {s['median_err']:.3f})",
             f"    check star {c['label']} ({c['name']}): measured {c['measured']:.3f} vs V {c['v']:.3f} "
             f"(diff {c['measured'] - c['v']:+.3f}, rms {c['rms']:.3f})"]
    if s.get("name_missing"):
        lines.append("    star named by its position (no --aavso-star): AAVSO needs its designation in NAME")
    if s.get("obscode_missing"):
        lines.append("    OBSCODE not set (--aavso-obscode): replace CHANGEME in the file before uploading")
    if path:
        lines.append(f"    {path}")
    return "\n".join(lines)


def write_aavso_report(args, output_path: str, wcs_header, exptime_s: float) -> Optional[dict]:
    """Pipeline entry: query VSP, build the report from ``<output>_lightcurves.csv``, write
    ``<output>_aavso.txt``. Returns the summary (with ``path``) or ``{"error": ...}``."""
    import os

    from src import net_query
    from src.utils import read_version
    base = os.path.splitext(output_path)[0]
    lc = base + "_lightcurves.csv"
    if not os.path.exists(lc):
        return {"error": "no light curves (needs --photometry-timeseries)"}
    star = getattr(args, "aavso_star", None)
    tra = tdec = None
    spec = getattr(args, "photometry_target", None)
    if spec and not str(spec).lower().startswith("px:"):
        try:
            tra, tdec = (float(v) for v in str(spec).replace(";", ",").split(","))
        except ValueError:
            tra = tdec = None
    if star is None and tra is None:
        return {"error": "no target: give --aavso-star NAME or --photometry-target RA,DEC"}
    fov = float(getattr(args, "aavso_fov", 60.0) or 60.0)
    maglim = float(getattr(args, "aavso_maglimit", 15.0) or 15.0)
    chart = (net_query.aavso_vsp_chart(star=star, fov_arcmin=fov, maglimit=maglim) if star
             else net_query.aavso_vsp_chart(ra_deg=tra, dec_deg=tdec, fov_arcmin=fov, maglimit=maglim))
    vsp = parse_vsp(chart)
    if vsp is None:
        return {"error": "AAVSO VSP did not return a chart" + (f" for {star!r}" if star else "")}
    s = build_report(lc, wcs_header, vsp, target_ra=tra, target_dec=tdec, star_name=star,
                     obscode=getattr(args, "aavso_obscode", None) or "", exptime_s=exptime_s,
                     bin_minutes=float(getattr(args, "aavso_bin", 0.0) or 0.0),
                     software=f"OriginStack {read_version()} (ensemble, VSP chart {vsp['chartid']})")
    if "error" in s:
        return s
    path = base + "_aavso.txt"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(s.pop("text"))
    s["path"] = path
    return s

