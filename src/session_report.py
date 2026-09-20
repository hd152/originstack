"""Session diagnostics report (``--session-report``).

The numbers that explain how a night went are all measured during a run -- frame
quality, registration shifts and rotations, residuals, transparency -- but only
their averages ever reached the log. Laid out against time they answer the
questions that otherwise need a spreadsheet: is the mount drifting and at what
rate, is there a periodic tracking error, how fast is the field rotating (the
number that decides whether frames can be pooled), did the focus wander with
temperature, and where did the sky go bad.

Writes ``<output>_session.png`` (a panel of plots, drawn with PIL only) and
``<output>_session.csv`` (one row per stacked frame), and prints a text summary.
"""
from __future__ import annotations

import csv
import logging
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_log = logging.getLogger("originstack")

_TEMP_KEYS = ('CCD-TEMP', 'CCDTEMP', 'TEMPERAT', 'SENSOR-TEMP', 'SET-TEMP')


def _frame_time_min(frame, j: int) -> float:
    """Observation time of a frame in minutes (absolute, from DATE-OBS), else its
    index scaled to one frame per minute so the plots still order correctly."""
    from src.utils import parse_timestamp
    hdr = getattr(frame, 'header', None) or {}
    for key in ('DATE-OBS', 'DATE'):
        v = hdr.get(key)
        if v:
            try:
                dt = parse_timestamp(str(v))
            except Exception:
                dt = None
            if dt is not None:
                return dt.timestamp() / 60.0 if hasattr(dt, 'timestamp') else float(j)
    return float(j)


def _temperature(frame) -> float:
    hdr = getattr(frame, 'header', None) or {}
    for k in _TEMP_KEYS:
        try:
            v = float(hdr.get(k))
            if math.isfinite(v):
                return v
        except (TypeError, ValueError):
            continue
    return float('nan')


def _frame_shape(final: Sequence[Any]) -> Optional[Tuple[int, int]]:
    for f in final:
        hdr = getattr(f, 'header', None) or {}
        try:
            h, w = int(hdr['NAXIS2']), int(hdr['NAXIS1'])
            if h > 0 and w > 0:
                return h, w
        except (KeyError, TypeError, ValueError):
            continue
    return None


def collect_session_table(final: Sequence[Any], shifts: Sequence[Any],
                          transforms: Sequence[Any],
                          shape_hw: Optional[Tuple[int, int]] = None) -> Dict[str, np.ndarray]:
    """Per-frame arrays, in acquisition order.

    ``dy``/``dx`` are where the frame's *centre* lands in the reference, minus
    the centre itself. A transform's raw translation is the displacement of the
    frame's (0, 0) corner, and under field rotation that corner sweeps an arc
    of radius ~the frame diagonal: on a real session rotating 0.5 deg/min the raw
    translation read as 23 px/min of 'drift' that was almost all rotation.
    Referenced to the centre, rotation about the centre contributes nothing and
    what is left is the pointing drift. ``shape_hw`` defaults to the frames'
    NAXIS keywords; with neither, the corner is used (old behaviour)."""
    from src.transparency import to_aligned_yx
    shape_hw = shape_hw or _frame_shape(final)
    centre = np.array([[(shape_hw[0] - 1) / 2.0, (shape_hw[1] - 1) / 2.0]]) if shape_hw \
        else np.zeros((1, 2))
    n = len(final)
    t = np.array([_frame_time_min(f, j) for j, f in enumerate(final)], dtype=float)
    order = np.argsort(t, kind='stable')

    def metric(key):
        vals = []
        for f in final:
            v = (getattr(f, 'metrics', None) or {}).get(key)
            try:
                vals.append(float(v) if v is not None else np.nan)
            except (TypeError, ValueError):
                vals.append(np.nan)
        return np.array(vals, dtype=float)

    dy = np.zeros(n)
    dx = np.zeros(n)
    rot = np.zeros(n)
    for j in range(n):
        tf = transforms[j] if j < len(transforms) else None
        sh = shifts[j] if j < len(shifts) and shifts[j] is not None else None
        moved = to_aligned_yx(centre, tf, sh)[0] - centre[0]
        dy[j], dx[j] = float(moved[0]), float(moved[1])
        if tf is not None:
            rot[j] = math.degrees(math.atan2(tf.params[1, 0], tf.params[0, 0]))
    tab = {
        't_min': t - t[order[0]] if n else t,
        'fwhm': metric('fwhm'), 'ellipticity': metric('ellipticity'), 'snr': metric('snr'),
        'background': metric('background'), 'noise': metric('noise'),
        'star_count': metric('star_count'), 'score': metric('score'),
        'transparency': metric('transparency'),
        # the residual check samples ~20% of a large session and leaves the rest
        # at exactly 0.0; those are unmeasured, not perfectly registered
        'residual_px': np.where(metric('reg_residual_px') == 0.0, np.nan,
                                metric('reg_residual_px')),
        'dy': dy, 'dx': dx, 'rot_deg': rot,
        'temp_c': np.array([_temperature(f) for f in final], dtype=float),
        'name': np.array([os.path.basename(getattr(f, 'path', str(j)))
                          for j, f in enumerate(final)]),
    }
    return {k: v[order] for k, v in tab.items()}


def _linfit(t: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """(slope, intercept, rms residual), ignoring non-finite points."""
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(t[ok]) <= 0:
        return float('nan'), float('nan'), float('nan')
    p = np.polyfit(t[ok], y[ok], 1)
    return float(p[0]), float(p[1]), float(np.sqrt(np.mean((y[ok] - np.polyval(p, t[ok])) ** 2)))


def analyze_tracking(t_min: np.ndarray, dy: np.ndarray, dx: np.ndarray,
                     rot_deg: np.ndarray) -> Dict[str, float]:
    """Drift rate, its direction, periodic residual, and field-rotation rate.

    Drift is the linear trend of the registration shift (rate and direction);
    what a smooth quadratic leaves behind is the tracking error, searched for a
    period with a Lomb-Scargle scan (reported only below a 1% false-alarm
    probability). Rotation is unwrapped first so a
    field that crosses +-180 degrees is not read as a jump."""
    out: Dict[str, float] = {}
    if len(t_min) < 5:
        return out
    sy, _, _ = _linfit(t_min, dy)
    sx, _, _ = _linfit(t_min, dx)
    out['drift_dy_px_per_min'], out['drift_dx_px_per_min'] = sy, sx
    # An alt-az mount's pointing drift curves, so scatter about a straight line
    # mostly measures that curvature (9.3 px on a real session whose path is smooth
    # to a pixel or two). The rate above stays the linear one; the *scatter* is
    # about a smooth quadratic, and so is the periodic-error search below.
    def _quad_resid(y):
        ok = np.isfinite(y)
        if ok.sum() < 6:
            return np.full_like(y, np.nan, dtype=float), float('nan')
        p = np.polyfit(t_min[ok], y[ok], 2)
        res = y - np.polyval(p, t_min)
        return res, float(np.sqrt(np.nanmean(res ** 2)))
    res_y, ry = _quad_resid(dy)
    res_x, rx = _quad_resid(dx)
    out['drift_px_per_min'] = float(math.hypot(sy, sx))
    out['drift_direction_deg'] = float(math.degrees(math.atan2(sy, sx)))
    out['drift_residual_rms_px'] = float(math.hypot(ry, rx))
    unwrapped = np.degrees(np.unwrap(np.radians(rot_deg)))
    sr, _, rr = _linfit(t_min, unwrapped)
    out['rotation_deg_per_min'] = sr
    out['rotation_total_deg'] = float(unwrapped[-1] - unwrapped[0])
    out['rotation_residual_rms_deg'] = rr
    # periodic tracking error in what the linear fits leave behind
    try:
        if len(t_min) >= 30 and np.ptp(t_min) > 6:
            from astropy.timeseries import LombScargle
            best = None
            for name, res in (('dy', res_y), ('dx', res_x)):
                if not np.isfinite(res).all() or np.std(res) <= 0:
                    continue
                ls = LombScargle(t_min, res)
                lo, hi = 1.0 / (np.ptp(t_min) / 2.0), 1.0 / max(2.5 * float(np.median(np.diff(t_min))), 0.2)
                if hi <= lo:
                    continue
                f, pw = ls.autopower(minimum_frequency=lo, maximum_frequency=hi,
                                     samples_per_peak=10)
                k = int(np.argmax(pw))
                fap = float(ls.false_alarm_probability(pw[k], minimum_frequency=lo,
                                                       maximum_frequency=hi))
                mp = ls.model_parameters(f[k])
                amp = float(np.hypot(mp[1], mp[2]))
                if best is None or fap < best[0]:
                    best = (fap, 1.0 / float(f[k]), amp, name)
            if best is not None:
                out['periodic_fap'], out['periodic_period_min'] = best[0], best[1]
                out['periodic_amplitude_px'] = best[2]
                out['periodic_axis'] = best[3]
    except Exception as exc:
        _log.debug("periodic tracking search failed: %s", exc)
    return out


def analyze_trends(t_min: np.ndarray, fwhm: np.ndarray, temp_c: np.ndarray,
                   transparency: np.ndarray) -> Dict[str, float]:
    """Focus drift (FWHM slope, and its correlation with temperature) and the
    transparency range."""
    out: Dict[str, float] = {}
    s, _, _ = _linfit(t_min / 60.0, fwhm)
    out['fwhm_slope_px_per_hour'] = s
    ok = np.isfinite(fwhm) & np.isfinite(temp_c)
    if ok.sum() >= 8 and np.ptp(temp_c[ok]) > 0.3 and np.std(fwhm[ok]) > 0:
        out['fwhm_temp_corr'] = float(np.corrcoef(fwhm[ok], temp_c[ok])[0, 1])
        out['temp_range_c'] = float(np.ptp(temp_c[ok]))
    tr = transparency[np.isfinite(transparency)]
    if tr.size:
        out['transparency_min'] = float(tr.min())
        out['transparency_p10'] = float(np.percentile(tr, 10))
    return out


def summary_lines(tab: Dict[str, np.ndarray], trk: Dict[str, float],
                  trend: Dict[str, float], n_rejected: int = 0) -> List[str]:
    n = len(tab['t_min'])
    span = float(tab['t_min'][-1]) if n else 0.0
    lines = [f"{n} frames over {span:.0f} min"
             + (f" ({n_rejected} rejected before stacking)" if n_rejected else "")]
    if 'drift_px_per_min' in trk:
        lines.append(f"Drift {trk['drift_px_per_min']:.2f} px/min toward "
                     f"{trk['drift_direction_deg']:+.0f} deg, tracking scatter "
                     f"{trk['drift_residual_rms_px']:.2f} px rms")
    if 'periodic_fap' in trk and trk['periodic_fap'] < 0.01:
        lines.append(f"Periodic tracking error: {trk['periodic_period_min']:.1f} min period, "
                     f"{trk['periodic_amplitude_px']:.2f} px amplitude in {trk['periodic_axis']} "
                     f"(FAP {trk['periodic_fap']:.1e})")
    if 'rotation_deg_per_min' in trk:
        lines.append(f"Field rotation {trk['rotation_deg_per_min'] * 60:+.2f} deg/hour, "
                     f"{trk['rotation_total_deg']:+.1f} deg over the session")
    fw = tab['fwhm'][np.isfinite(tab['fwhm'])]
    if fw.size:
        lines.append(f"FWHM median {np.median(fw):.2f} px, range {fw.min():.2f}-{fw.max():.2f}"
                     + (f", trend {trend['fwhm_slope_px_per_hour']:+.2f} px/hour"
                        if np.isfinite(trend.get('fwhm_slope_px_per_hour', np.nan)) else ""))
    if 'fwhm_temp_corr' in trend and abs(trend['fwhm_temp_corr']) > 0.5:
        lines.append(f"FWHM tracks sensor temperature (r = {trend['fwhm_temp_corr']:+.2f} over "
                     f"{trend['temp_range_c']:.1f} C): likely focus drift")
    if 'transparency_min' in trend:
        lines.append(f"Transparency min {trend['transparency_min']:.2f}, "
                     f"10th percentile {trend['transparency_p10']:.2f}")
    return lines


# --------------------------------------------------------------------------
# Plotting (PIL only)
# --------------------------------------------------------------------------

_BG, _FG, _GRID = (255, 255, 255), (30, 30, 30), (220, 220, 220)
_COLORS = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (255, 127, 14)]


def _nice_ticks(lo: float, hi: float, n: int = 5) -> List[float]:
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo]
    raw = (hi - lo) / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=raw)
    first = math.ceil(lo / step) * step
    ticks, v = [], first
    while v <= hi + 1e-9:
        ticks.append(v)
        v += step
    return ticks


def _panel(draw, box, title: str, xs: np.ndarray, series: List[Tuple[str, np.ndarray]],
           font, ylabel: str = '') -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=_FG)
    draw.text((x0 + 8, y0 + 4), title, fill=_FG, font=font)
    px0, py0, px1, py1 = x0 + 62, y0 + 26, x1 - 14, y1 - 30
    allv = np.concatenate([s[1][np.isfinite(s[1])] for s in series]) if series else np.array([])
    fin_x = xs[np.isfinite(xs)]
    if allv.size == 0 or fin_x.size == 0:
        draw.text((px0 + 10, (py0 + py1) // 2), "no data", fill=(150, 150, 150), font=font)
        return
    lo, hi = float(allv.min()), float(allv.max())
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5
    pad = 0.06 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    xlo, xhi = float(fin_x.min()), float(fin_x.max())
    if xhi <= xlo:
        xhi = xlo + 1.0

    def X(v):
        return px0 + (v - xlo) / (xhi - xlo) * (px1 - px0)

    def Y(v):
        return py1 - (v - lo) / (hi - lo) * (py1 - py0)

    for tv in _nice_ticks(lo, hi, 4):
        yy = Y(tv)
        draw.line([(px0, yy), (px1, yy)], fill=_GRID)
        draw.text((x0 + 4, yy - 6), f"{tv:.3g}", fill=_FG, font=font)
    for tv in _nice_ticks(xlo, xhi, 5):
        xx = X(tv)
        draw.line([(xx, py0), (xx, py1)], fill=_GRID)
        draw.text((xx - 10, py1 + 4), f"{tv:.0f}", fill=_FG, font=font)
    draw.text((px1 - 60, y1 - 16), "minutes", fill=(120, 120, 120), font=font)
    if ylabel:
        draw.text((x0 + 8, y0 + 16), ylabel, fill=(120, 120, 120), font=font)
    draw.rectangle((px0, py0, px1, py1), outline=(150, 150, 150))
    for k, (label, ys) in enumerate(series):
        col = _COLORS[k % len(_COLORS)]
        ok = np.isfinite(xs) & np.isfinite(ys)
        pts = [(X(a), Y(b)) for a, b in zip(xs[ok], ys[ok])]
        if len(pts) > 1:
            draw.line(pts, fill=col, width=1)
        for (a, b) in pts:
            draw.ellipse((a - 1.6, b - 1.6, a + 1.6, b + 1.6), fill=col)
        if len(series) > 1:
            draw.text((px0 + 6 + 60 * k, py0 + 2), label, fill=col, font=font)


def render_report_png(tab: Dict[str, np.ndarray], lines: List[str], path: str) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return False
    W = 1800
    H = 3 * 400 + 40 + 20 * (len(lines) + 1) + 10
    img = Image.new('RGB', (W, H), _BG)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:                       # older Pillow: fixed default size
        font = ImageFont.load_default()
    t = tab['t_min']
    cw, ch = W // 3, 400
    panels = [
        ("FWHM (px)", [('FWHM', tab['fwhm'])], ''),
        ("Transparency (session median = 1)", [('transp.', tab['transparency'])], ''),
        ("SNR", [('SNR', tab['snr'])], ''),
        ("Sky background (ADU)", [('bkg', tab['background'])], ''),
        ("Registration shift (px)", [('dx', tab['dx'] - tab['dx'][0]),
                                      ('dy', tab['dy'] - tab['dy'][0])], ''),
        ("Field rotation (deg)", [('rot', np.degrees(np.unwrap(np.radians(tab['rot_deg']))))], ''),
        ("PSF ellipticity", [('ell.', tab['ellipticity'])], ''),
        ("Registration residual (px)", [('resid', tab['residual_px'])], ''),
        ("Sensor temperature (C)", [('temp', tab['temp_c'])], ''),
    ]
    for i, (title, series, yl) in enumerate(panels):
        r, c = divmod(i, 3)
        box = (c * cw + 6, r * ch + 6, (c + 1) * cw - 6, (r + 1) * ch - 6)
        _panel(d, box, title, t, series, font, yl)
    y = 3 * ch + 14
    d.text((14, y), "Session summary", fill=_FG, font=font)
    for k, line in enumerate(lines):
        d.text((14, y + 22 + 20 * k), line, fill=_FG, font=font)
    img.save(path, format='PNG')
    return True


_CSV_FIELDS = ['name', 't_min', 'fwhm', 'ellipticity', 'snr', 'background', 'noise',
               'star_count', 'score', 'transparency', 'residual_px', 'dx', 'dy', 'rot_deg',
               'temp_c']


def write_session_report(final: Sequence[Any], shifts: Sequence[Any], transforms: Sequence[Any],
                         output_path: str, n_rejected: int = 0,
                         shape_hw: Optional[Tuple[int, int]] = None) -> Optional[Dict[str, Any]]:
    """Build the table, analysis, PNG and CSV. Returns a summary dict, or None
    when there is nothing to report (fewer than 3 frames)."""
    if len(final) < 3:
        return None
    tab = collect_session_table(final, shifts, transforms, shape_hw)
    trk = analyze_tracking(tab['t_min'], tab['dy'], tab['dx'], tab['rot_deg'])
    trend = analyze_trends(tab['t_min'], tab['fwhm'], tab['temp_c'], tab['transparency'])
    lines = summary_lines(tab, trk, trend, n_rejected)
    base = os.path.splitext(output_path)[0]
    png_path, csv_path = base + '_session.png', base + '_session.csv'
    ok_png = render_report_png(tab, lines, png_path)
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(_CSV_FIELDS)
            for j in range(len(tab['t_min'])):
                w.writerow([tab['name'][j]] + [
                    ('' if not np.isfinite(tab[k][j]) else round(float(tab[k][j]), 4))
                    for k in _CSV_FIELDS[1:]])
    except Exception as exc:
        _log.warning("session report: could not write %s: %s", csv_path, exc)
        csv_path = None
    return {'lines': lines, 'tracking': trk, 'trends': trend,
            'png': png_path if ok_png else None, 'csv': csv_path}
