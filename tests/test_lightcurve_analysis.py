"""Light-curve period search and transit fitting."""
import csv

import numpy as np

from src.lightcurve_analysis import (
    analyze_lightcurves,
    analyze_star,
    box_search,
    fit_transit,
    load_lightcurves,
    lomb_scargle,
)


def _times(n=160, hours=4.0, seed=0, t0=60000.0):
    rng = np.random.default_rng(seed)
    return t0 + np.sort(rng.uniform(0, hours / 24.0, n))


def test_recovers_a_sinusoidal_period_and_amplitude():
    t = _times()
    rng = np.random.default_rng(1)
    period_h, amp = 0.8, 0.05
    mag = 14.0 + amp * np.sin(2 * np.pi * (t - t[0]) * 24.0 / period_h + 0.7)
    mag += rng.normal(0, 0.008, len(t))
    r = lomb_scargle(t, mag, np.full(len(t), 0.008))
    assert abs(r['period_h'] - period_h) / period_h < 0.03
    assert abs(r['amplitude'] - amp) < 0.008
    assert r['fap'] < 1e-6


def test_pure_noise_has_a_high_false_alarm_probability():
    t = _times(seed=2)
    mag = 14.0 + np.random.default_rng(3).normal(0, 0.01, len(t))
    r = lomb_scargle(t, mag, np.full(len(t), 0.01))
    assert r['fap'] > 0.05


def _transit_curve(depth_mmag=25.0, dur_h=0.9, t0_frac=0.55, hours=4.0, noise=0.004, seed=4):
    t = _times(hours=hours, seed=seed)
    span = t[-1] - t[0]
    t0 = t[0] + t0_frac * span
    f = np.ones_like(t)
    half = dur_h / 48.0
    ing = 0.15 * dur_h / 24.0
    x = np.abs(t - t0)
    depth = 1 - 10 ** (-0.4 * depth_mmag / 1000.0)
    f[x <= half - ing] = 1 - depth
    ramp = (x > half - ing) & (x < half)
    f[ramp] = 1 - depth * (half - x[ramp]) / ing
    mag = 14.0 - 2.5 * np.log10(f) + np.random.default_rng(seed + 1).normal(0, noise, len(t))
    return t, mag, np.full(len(t), noise), t0


def test_transit_fit_recovers_depth_duration_and_epoch():
    t, mag, err, t0 = _transit_curve()
    b = box_search(t, mag, err)
    assert b['snr'] > 8
    fit = fit_transit(t, mag, err, t0_guess=b['t0_mjd'], dur_guess_d=b['duration_d'])
    assert abs(fit['t0_mjd'] - t0) * 24 < 0.15                 # within ~9 minutes
    assert abs(fit['depth_mmag'] - 25.0) < 4.0
    assert abs(fit['duration_h'] - 0.9) < 0.2
    assert fit['delta_bic'] > 30
    assert 0 < fit['depth_err_mmag'] < 5 and fit['duration_err_h'] > 0


def test_flat_star_gives_no_significant_transit():
    t = _times(seed=6)
    mag = 14.0 + np.random.default_rng(7).normal(0, 0.005, len(t))
    out = analyze_star(t, mag, np.full(len(t), 0.005))
    assert out.get('transit_delta_bic', 0) < 10 or out.get('bls_snr', 0) < 5


def test_short_or_sparse_series_are_skipped():
    assert analyze_star(np.arange(10.0), np.zeros(10), np.ones(10)) == {'n_points': 10}
    t = 60000 + np.linspace(0, 0.01, 40)             # 14 minutes: too short to say anything
    assert 'ls_period_h' not in analyze_star(t, np.zeros(40), np.full(40, 0.01))


def _write_csv(path, curves):
    fields = ['frame', 'filename', 'mjd', 'airmass', 'source_id', 'x', 'y', 'gaia_g', 'is_target',
              'mag_r', 'magerr_r', 'mag_g', 'magerr_g', 'mag_b', 'magerr_b', 'flag']
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for sid, (t, mag, err, target) in curves.items():
            for j in range(len(t)):
                w.writerow({'frame': j, 'source_id': sid, 'mjd': t[j], 'mag_g': mag[j],
                            'magerr_g': err[j], 'is_target': target,
                            'flag': 's' if j == 3 else ''})


def test_end_to_end_csv_roundtrip(tmp_path):
    t, mag, err, _ = _transit_curve()
    ts = _times(seed=8)
    sine = 13.0 + 0.04 * np.sin(2 * np.pi * (ts - ts[0]) * 24.0 / 0.7) + np.random.default_rng(9).normal(0, 0.006, len(ts))
    flat = 15.0 + np.random.default_rng(10).normal(0, 0.006, len(ts))
    p = tmp_path / 'x_lightcurves.csv'
    _write_csv(p, {'111': (t, mag, err, 1), '222': (ts, sine, np.full(len(ts), 0.006), 0),
                   '333': (ts, flat, np.full(len(ts), 0.006), 0)})
    curves = load_lightcurves(str(p))
    assert set(curves) == {'111', '222', '333'}
    assert len(curves['111']['mjd']) == len(t) - 1            # the flagged row was dropped
    out = tmp_path / 'x_lightcurve_analysis.csv'
    s = analyze_lightcurves(str(p), None, str(out))
    assert s['stars'] == 3 and out.exists()
    assert s['best_periodic']['source_id'] == '222'
    assert abs(s['best_periodic']['ls_period_h'] - 0.7) < 0.03
    assert s['best_transit']['source_id'] == '111'
    rows = list(csv.DictReader(open(out)))
    assert rows[0]['source_id'] == '111'                       # the target is analysed first


def test_nothing_to_analyse_returns_none(tmp_path):
    p = tmp_path / 'e.csv'
    _write_csv(p, {'1': (np.arange(5.0) + 60000, np.zeros(5), np.ones(5), 0)})
    assert analyze_lightcurves(str(p), None, str(tmp_path / 'o.csv')) is None
