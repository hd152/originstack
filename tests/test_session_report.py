"""Session diagnostics: drift, periodic tracking error, rotation, focus trend, plots."""
import csv
from types import SimpleNamespace

import numpy as np

from src.affine_fit import RigidTransform
from src.session_report import (
    analyze_tracking,
    analyze_trends,
    collect_session_table,
    summary_lines,
    write_session_report,
)


def _t(n=120, minutes=30.0, seed=0):
    return np.sort(np.random.default_rng(seed).uniform(0, minutes, n))


def test_drift_rate_direction_and_rotation_rate():
    t = _t()
    rng = np.random.default_rng(1)
    dy = 1.5 + 0.20 * t + rng.normal(0, 0.15, len(t))
    dx = -3.0 + 0.10 * t + rng.normal(0, 0.15, len(t))
    rot = 0.5 + 0.075 * t                                     # deg, like the real session
    r = analyze_tracking(t, dy, dx, rot)
    assert abs(r['drift_dy_px_per_min'] - 0.20) < 0.01
    assert abs(r['drift_dx_px_per_min'] - 0.10) < 0.01
    assert abs(r['drift_px_per_min'] - np.hypot(0.2, 0.1)) < 0.01
    assert abs(r['rotation_deg_per_min'] - 0.075) < 0.002
    assert abs(r['rotation_total_deg'] - 0.075 * (t[-1] - t[0])) < 0.05
    assert r['drift_residual_rms_px'] < 0.4


def test_rotation_across_the_180_degree_wrap_is_not_a_jump():
    t = np.linspace(0, 30, 60)
    rot = ((178.0 + 0.2 * t) + 180.0) % 360.0 - 180.0         # wraps to -180 partway
    r = analyze_tracking(t, np.zeros(60), np.zeros(60), rot)
    assert abs(r['rotation_deg_per_min'] - 0.2) < 0.005


def test_a_periodic_tracking_error_is_found_and_noise_is_not():
    t = _t(n=200, minutes=40.0, seed=2)
    rng = np.random.default_rng(3)
    dx = 0.05 * t + 0.8 * np.sin(2 * np.pi * t / 8.0) + rng.normal(0, 0.1, len(t))
    dy = 0.02 * t + rng.normal(0, 0.1, len(t))
    r = analyze_tracking(t, dy, dx, np.zeros(len(t)))
    assert r['periodic_fap'] < 1e-6 and abs(r['periodic_period_min'] - 8.0) < 0.3
    assert r['periodic_axis'] == 'dx' and abs(r['periodic_amplitude_px'] - 0.8) < 0.15
    quiet = analyze_tracking(t, dy, 0.05 * t + rng.normal(0, 0.1, len(t)), np.zeros(len(t)))
    assert quiet.get('periodic_fap', 1.0) > 0.01


def test_focus_drift_follows_temperature():
    t = np.linspace(0, 120, 80)
    temp = 20.0 - 0.05 * t
    fwhm = 4.0 + 0.6 * (20.0 - temp) / 6.0 + np.random.default_rng(4).normal(0, 0.03, 80)
    r = analyze_trends(t, fwhm, temp, np.ones(80))
    assert r['fwhm_temp_corr'] < -0.9 and r['fwhm_slope_px_per_hour'] > 0.1
    assert r['temp_range_c'] > 5


def _frames(n=40, seed=5):
    rng = np.random.default_rng(seed)
    frames, shifts, tfs = [], [], []
    for j in range(n):
        f = SimpleNamespace(
            path=f'Light{j:05d}.fits',
            header={'DATE-OBS': f'2025-09-28T20:{j // 2:02d}:{(j % 2) * 30:02d}', 'CCD-TEMP': 28.0 - 0.02 * j},
            metrics={'fwhm': 4.5 + 0.01 * j + rng.normal(0, 0.05), 'snr': 1.7, 'background': 9400.0,
                     'ellipticity': 0.1, 'transparency': 1.0 + rng.normal(0, 0.02),
                     'reg_residual_px': 1.8, 'star_count': 80, 'noise': 400, 'score': 0.9})
        frames.append(f)
        tfs.append(RigidTransform.from_rotation_translation(np.deg2rad(0.05 * j), (0.3 * j, -0.1 * j)))
        shifts.append((0.0, 0.0))
    return frames, shifts, tfs


def test_table_is_time_ordered_and_reads_transforms():
    frames, shifts, tfs = _frames()
    frames, shifts, tfs = frames[::-1], shifts[::-1], tfs[::-1]      # scrambled on purpose
    tab = collect_session_table(frames, shifts, tfs)
    assert np.all(np.diff(tab['t_min']) >= 0) and abs(tab['t_min'][-1] - 19.5) < 0.6
    assert tab['name'][0] == 'Light00000.fits'
    assert abs(tab['rot_deg'][-1] - 0.05 * 39) < 0.01
    assert np.isfinite(tab['temp_c']).all() and np.isfinite(tab['transparency']).all()


def test_report_writes_png_and_csv(tmp_path):
    frames, shifts, tfs = _frames()
    out = str(tmp_path / 'stack.fits')
    r = write_session_report(frames, shifts, tfs, out, n_rejected=3)
    assert r is not None
    from PIL import Image
    im = Image.open(r['png'])
    assert im.size[0] == 1800 and im.size[1] >= 1290
    arr = np.asarray(im.convert('L'))
    assert (arr < 200).mean() > 0.01                    # something was actually drawn
    rows = list(csv.DictReader(open(r['csv'])))
    assert len(rows) == len(frames) and rows[0]['name'] == 'Light00000.fits'
    assert any('rejected before stacking' in ln for ln in r['lines'])
    assert any('Field rotation' in ln for ln in r['lines'])
    assert any('Drift' in ln for ln in r['lines'])


def test_missing_fields_and_tiny_sessions_do_not_break():
    f = SimpleNamespace(path='a.fits', header={}, metrics={})
    assert write_session_report([f, f], [(0, 0)] * 2, [None] * 2, 'x.fits') is None
    frames = [SimpleNamespace(path=f'{j}.fits', header={}, metrics={}) for j in range(6)]
    tab = collect_session_table(frames, [(0.0, 0.0)] * 6, [None] * 6)
    trk = analyze_tracking(tab['t_min'], tab['dy'], tab['dx'], tab['rot_deg'])
    assert summary_lines(tab, trk, analyze_trends(tab['t_min'], tab['fwhm'], tab['temp_c'],
                                                  tab['transparency']))[0].startswith('6 frames')


def test_rotation_about_the_centre_is_not_read_as_drift():
    """A transform's raw translation is the corner's displacement, which under field
    rotation sweeps a large arc. Referenced to the frame centre, a pure rotation about
    the centre must read as zero drift."""
    H, W = 1000, 1400
    c = np.array([(H - 1) / 2.0, (W - 1) / 2.0])
    frames, tfs = [], []
    for j in range(30):
        th = np.radians(0.5 * j)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        t = c - R @ c                                   # rotate about the centre
        m = np.eye(3)
        m[:2, :2] = R.T
        m[1, 2], m[0, 2] = t[0], t[1]
        tfs.append(SimpleNamespace(params=m))
        frames.append(SimpleNamespace(path=f'{j}.fits', header={'DATE-OBS': f'2025-09-28T20:{j:02d}:00',
                                                                'NAXIS1': W, 'NAXIS2': H}, metrics={}))
    raw = np.array([np.hypot(tf.params[1, 2], tf.params[0, 2]) for tf in tfs])
    assert raw.max() > 100                              # the corner really does sweep an arc
    tab = collect_session_table(frames, [(0.0, 0.0)] * 30, tfs)      # shape from NAXIS
    assert np.abs(tab['dy']).max() < 1e-6 and np.abs(tab['dx']).max() < 1e-6
    assert abs(abs(tab['rot_deg'][-1]) - 0.5 * 29) < 0.01        # sign is the inverse's
    r = analyze_tracking(tab['t_min'], tab['dy'], tab['dx'], tab['rot_deg'])
    assert r['drift_px_per_min'] < 1e-6 and abs(abs(r['rotation_deg_per_min']) - 0.5) < 0.01


def test_unmeasured_residuals_are_not_plotted_as_zero(tmp_path):
    frames, shifts, tfs = _frames(n=10)
    for j, f in enumerate(frames):
        f.metrics['reg_residual_px'] = 0.0 if j % 2 else 1.8
    tab = collect_session_table(frames, shifts, tfs)
    assert np.isnan(tab['residual_px']).sum() == 5 and np.nanmin(tab['residual_px']) > 1.0


def test_png_is_tall_enough_for_every_summary_line(tmp_path):
    frames, shifts, tfs = _frames()
    r = write_session_report(frames, shifts, tfs, str(tmp_path / 's.fits'))
    from PIL import Image
    assert Image.open(r['png']).size[1] >= 3 * 400 + 20 * len(r['lines'])
