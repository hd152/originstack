"""Period search and transit fitting on ``--photometry-timeseries`` light curves.

``photometry_timeseries`` writes one row per (frame, star) and flags variables by
reduced chi-squared, but stops there: it says a star varies, not how. This adds

  * a Lomb-Scargle period search (astropy's implementation, with its
    false-alarm probability) for periodic variables, and
  * a transit search + fit for the marked target or any dimming candidate: a
    box-least-squares scan for the best period/epoch/duration, then a
    trapezoid (finite ingress/egress) least-squares fit of the dip itself,
    with parameter errors from the fit's Jacobian and a BIC comparison against
    a flat light curve.

Everything is in magnitudes as written by the timeseries step (``mag_g`` with
``magerr_g``); transit depths are quoted in mmag and converted to flux internally.
A session is a few hours long, so periods are searched only up to half the
baseline (fewer than two cycles cannot be told from a trend), and a "transit"
is one dip: BLS is used to place it, the trapezoid fit to measure it.
"""
from __future__ import annotations

import csv
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

_log = logging.getLogger("originstack")

_MIN_POINTS = 20
_MIN_BASELINE_H = 0.5
_MAX_STARS = 60


def load_lightcurves(csv_path: str, channel: str = 'g') -> Dict[str, Dict[str, np.ndarray]]:
    """Read ``<output>_lightcurves.csv`` into {source_id: {mjd, mag, err, target}}.

    Rows flagged saturated/error, or with no magnitude, are dropped."""
    out: Dict[str, Dict[str, Any]] = {}
    with open(csv_path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            if row.get('flag'):
                continue
            try:
                t = float(row['mjd'])
                m = float(row[f'mag_{channel}'])
                e = float(row[f'magerr_{channel}'])
            except (KeyError, TypeError, ValueError):
                continue
            if not (np.isfinite(t) and np.isfinite(m) and np.isfinite(e)) or e <= 0:
                continue
            d = out.setdefault(row['source_id'], {'mjd': [], 'mag': [], 'err': [],
                                                   'target': int(row.get('is_target') or 0)})
            d['mjd'].append(t)
            d['mag'].append(m)
            d['err'].append(e)
    result = {}
    for sid, d in out.items():
        order = np.argsort(d['mjd'])
        result[sid] = {'mjd': np.asarray(d['mjd'])[order], 'mag': np.asarray(d['mag'])[order],
                       'err': np.asarray(d['err'])[order], 'target': d['target']}
    return result


def lomb_scargle(t_days: np.ndarray, mag: np.ndarray, err: np.ndarray,
                 min_period_h: Optional[float] = None) -> Dict[str, float]:
    """Best period (hours), its FAP, and the semi-amplitude (mag) of the sinusoid.

    Searches from the median cadence's Nyquist-ish limit (or ``min_period_h``)
    up to half the baseline."""
    from astropy.timeseries import LombScargle
    t = np.asarray(t_days, float)
    baseline = float(t.max() - t.min())
    dt = float(np.median(np.diff(t))) if len(t) > 2 else baseline
    pmin = max((min_period_h or 0.0) / 24.0, 4.0 * dt)
    pmax = baseline / 2.0
    if pmax <= pmin:
        return {'period_h': float('nan'), 'fap': 1.0, 'amplitude': float('nan'), 'power': 0.0}
    ls = LombScargle(t, mag, err)
    freq, power = ls.autopower(minimum_frequency=1.0 / pmax, maximum_frequency=1.0 / pmin,
                               samples_per_peak=10)
    k = int(np.argmax(power))
    fap = float(ls.false_alarm_probability(power[k], minimum_frequency=1.0 / pmax,
                                           maximum_frequency=1.0 / pmin))
    model = ls.model_parameters(freq[k])              # offset, sin, cos coefficients
    amp = float(np.hypot(model[1], model[2])) if len(model) >= 3 else float('nan')
    return {'period_h': 24.0 / float(freq[k]), 'fap': fap, 'amplitude': amp,
            'power': float(power[k])}


def _trapezoid(t, t0, depth, dur, ingress):
    """Flux dip: 1 - depth inside a trapezoid centred on t0, total duration
    ``dur`` (first to last contact), with ingress/egress each ``ingress*dur``."""
    ing = np.clip(ingress, 1e-3, 0.5) * dur
    x = np.abs(t - t0)
    half = dur / 2.0
    y = np.ones_like(t, dtype=float)
    flat = x <= half - ing
    y[flat] = 1.0 - depth
    ramp = (x > half - ing) & (x < half)
    y[ramp] = 1.0 - depth * (half - x[ramp]) / ing
    return y


def fit_transit(t_days: np.ndarray, mag: np.ndarray, err: np.ndarray,
                t0_guess: Optional[float] = None, dur_guess_d: Optional[float] = None,
                depth_guess: Optional[float] = None) -> Dict[str, float]:
    """Trapezoid fit of a single dip to a magnitude light curve.

    Returns t0 (MJD), depth (mmag), duration (h), ingress fraction, their 1-sigma
    errors, reduced chi-squared, and ``delta_bic`` = BIC(flat) - BIC(transit)
    (positive favours a transit; > 10 is strong)."""
    from scipy.optimize import least_squares
    t = np.asarray(t_days, float)
    flux = 10.0 ** (-0.4 * (np.asarray(mag, float) - np.median(mag)))
    ferr = flux * np.asarray(err, float) * 0.4 * np.log(10.0)
    n = len(t)
    span = float(t.max() - t.min())
    if t0_guess is None:
        t0_guess = float(t[np.argmin(np.convolve(flux, np.ones(5) / 5.0, mode='same'))])
    dur_guess_d = dur_guess_d or span / 6.0
    depth_guess = depth_guess or max(float(1.0 - np.percentile(flux, 5)), 1e-3)

    def resid(p):
        return (flux - p[4] * _trapezoid(t, p[0], p[1], p[2], p[3])) / ferr

    p0 = np.array([t0_guess, depth_guess, dur_guess_d, 0.2, 1.0])
    lo = np.array([t.min(), 0.0, span / 200.0, 0.01, 0.9])
    hi = np.array([t.max(), 0.5, span, 0.5, 1.1])
    try:
        sol = least_squares(resid, np.clip(p0, lo + 1e-9, hi - 1e-9), bounds=(lo, hi))
    except Exception as exc:  # pragma: no cover
        _log.debug("transit fit failed: %s", exc)
        return {}
    chi2 = float(np.sum(sol.fun ** 2))
    dof = max(n - 5, 1)
    chi2_flat = float(np.sum(((flux - np.median(flux)) / ferr) ** 2))
    dbic = (chi2_flat + 1 * np.log(n)) - (chi2 + 5 * np.log(n))
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * max(chi2 / dof, 1.0)
        sig = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        sig = np.full(5, float('nan'))
    depth_flux = float(sol.x[1] * sol.x[4])
    return {
        't0_mjd': float(sol.x[0]), 't0_err_h': float(sig[0] * 24.0),
        'depth_mmag': float(-2500.0 * np.log10(max(1.0 - depth_flux, 1e-9))),
        'depth_err_mmag': float(1085.7 * sig[1]),
        'duration_h': float(sol.x[2] * 24.0), 'duration_err_h': float(sig[2] * 24.0),
        'ingress_frac': float(sol.x[3]),
        'chi2_red': chi2 / dof, 'delta_bic': float(dbic),
    }


def box_search(t_days: np.ndarray, mag: np.ndarray, err: np.ndarray) -> Dict[str, float]:
    """BLS scan for a periodic dip; SNR is astropy's depth signal-to-noise."""
    from astropy.timeseries import BoxLeastSquares
    t = np.asarray(t_days, float)
    flux = 10.0 ** (-0.4 * (np.asarray(mag, float) - np.median(mag)))
    ferr = flux * np.asarray(err, float) * 0.4 * np.log(10.0)
    span = float(t.max() - t.min())
    bls = BoxLeastSquares(t, flux, dy=ferr)
    periods = np.linspace(span / 2.0, span * 3.0, 300)      # >= 2 events or a single one
    durations = np.linspace(span / 40.0, span / 4.0, 8)
    res = bls.power(periods, durations, objective='snr')
    k = int(np.argmax(res.power))
    return {'period_d': float(res.period[k]), 't0_mjd': float(res.transit_time[k]),
            'duration_d': float(res.duration[k]), 'depth': float(res.depth[k]),
            'snr': float(res.depth_snr[k])}


def analyze_star(mjd: np.ndarray, mag: np.ndarray, err: np.ndarray) -> Dict[str, Any]:
    """LS period search + transit search/fit for one light curve."""
    out: Dict[str, Any] = {'n_points': int(len(mjd))}
    if len(mjd) < _MIN_POINTS:
        return out
    baseline_h = float((mjd.max() - mjd.min()) * 24.0)
    out['baseline_h'] = baseline_h
    if baseline_h < _MIN_BASELINE_H:
        return out
    try:
        ls = lomb_scargle(mjd, mag, err)
        out.update({'ls_period_h': ls['period_h'], 'ls_fap': ls['fap'],
                    'ls_amplitude_mag': ls['amplitude']})
    except Exception as exc:
        _log.debug("lomb-scargle failed: %s", exc)
    try:
        b = box_search(mjd, mag, err)
        fit = fit_transit(mjd, mag, err, t0_guess=b['t0_mjd'],
                          dur_guess_d=b['duration_d'], depth_guess=max(b['depth'], 1e-3))
        out.update({'bls_snr': b['snr'], 'bls_depth_mmag': 1085.7 * b['depth']})
        out.update({f'transit_{k}': v for k, v in fit.items()})
    except Exception as exc:
        _log.debug("transit search failed: %s", exc)
    return out


_FIELDS = ['source_id', 'is_target', 'n_points', 'baseline_h', 'ls_period_h', 'ls_fap',
           'ls_amplitude_mag', 'bls_snr', 'bls_depth_mmag', 'transit_t0_mjd',
           'transit_t0_err_h', 'transit_depth_mmag', 'transit_depth_err_mmag',
           'transit_duration_h', 'transit_duration_err_h', 'transit_ingress_frac',
           'transit_chi2_red', 'transit_delta_bic']


def _stat_rank(stats_csv: Optional[str]) -> Dict[str, float]:
    """source_id -> reduced chi-squared, to rank which stars deserve a search."""
    rank: Dict[str, float] = {}
    if not stats_csv or not os.path.exists(stats_csv):
        return rank
    with open(stats_csv, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            for key in ('chi2red_g', 'chi2_red_g', 'chi2_red'):
                if row.get(key):
                    try:
                        rank[row['source_id']] = float(row[key])
                    except ValueError:
                        pass
                    break
    return rank


def analyze_lightcurves(lc_csv: str, stats_csv: Optional[str], out_path: str,
                        max_stars: int = _MAX_STARS) -> Optional[Dict[str, Any]]:
    """Analyse the target and the most variable stars; write ``out_path`` (CSV).

    Returns a summary dict (rows written, best periodic candidate, best transit
    candidate) or None if there was nothing to analyse."""
    curves = load_lightcurves(lc_csv)
    if not curves:
        return None
    rank = _stat_rank(stats_csv)
    ids = sorted(curves, key=lambda s: (-curves[s]['target'], -rank.get(s, 0.0)))
    ids = [s for s in ids if len(curves[s]['mjd']) >= _MIN_POINTS][:max_stars]
    rows: List[Dict[str, Any]] = []
    for sid in ids:
        c = curves[sid]
        r = analyze_star(c['mjd'], c['mag'], c['err'])
        r.update({'source_id': sid, 'is_target': c['target']})
        rows.append(r)
    if not rows:
        return None
    with open(out_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    periodic = [r for r in rows if r.get('ls_fap', 1.0) < 0.01]
    transit = [r for r in rows if r.get('transit_delta_bic', -1) > 10 and r.get('bls_snr', 0) > 5]
    best_p = min(periodic, key=lambda r: r['ls_fap']) if periodic else None
    best_t = max(transit, key=lambda r: r['transit_delta_bic']) if transit else None
    return {'stars': len(rows), 'periodic': len(periodic), 'transit': len(transit),
            'best_periodic': best_p, 'best_transit': best_t, 'path': out_path}


def format_analysis_summary(s: Dict[str, Any]) -> str:
    lines = [f"  Light-curve analysis: {s['stars']} star(s) searched, "
             f"{s['periodic']} periodic (FAP < 1%), {s['transit']} transit-like"]
    p = s.get('best_periodic')
    if p:
        lines.append(f"    Best period: source {p['source_id']}  P = {p['ls_period_h']:.3f} h  "
                     f"amp {p['ls_amplitude_mag'] * 1000:.1f} mmag  FAP {p['ls_fap']:.1e}")
    t = s.get('best_transit')
    if t:
        lines.append(f"    Best dip: source {t['source_id']}  depth "
                     f"{t['transit_depth_mmag']:.1f}+-{t['transit_depth_err_mmag']:.1f} mmag  "
                     f"duration {t['transit_duration_h']:.2f} h  "
                     f"dBIC {t['transit_delta_bic']:.0f}")
    lines.append(f"    -> {os.path.basename(s['path'])}")
    return "\n".join(lines)
