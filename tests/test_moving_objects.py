"""Moving-object detection, velocity-space linking and tracked stacking."""
import csv

import numpy as np

from src.moving_objects import (
    collect_detections,
    detect_in_residual,
    find_moving_objects,
    find_tracks,
    format_summary,
    star_mask,
    tracked_stack,
    write_outputs,
)

_H, _W, _N, _SPAN = 220, 320, 40, 30.0


def _scene(mover=True, v=(0.55, -0.30), amp=200.0, sigma=8.0, seed=0, n=_N, hot_pixel=True):
    """N aligned frames: fixed stars (with a little per-frame seeing variation), noise,
    optionally one point source moving at v px/min, optionally a stationary hot pixel."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:_H, :_W].astype(np.float32)
    sr = np.random.default_rng(99)
    stars = [(sr.uniform(15, _H - 15), sr.uniform(15, _W - 15), sr.uniform(2000, 12000))
             for _ in range(25)]
    t = np.sort(rng.uniform(0, _SPAN, n))
    t -= t[0]
    frames = np.empty((n, _H, _W, 3), np.float32)
    x0, y0 = 60.0, 150.0
    for j in range(n):
        img = np.full((_H, _W), 500.0, np.float32)
        fw = 1.8 * (1.0 + 0.05 * rng.normal())                     # seeing wobble
        for (cy, cx, a) in stars:
            img += a * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * fw ** 2))
        if mover:
            mx, my = x0 + v[0] * t[j], y0 + v[1] * t[j]
            img += amp * np.exp(-((yy - my) ** 2 + (xx - mx) ** 2) / (2 * 1.8 ** 2))
        if hot_pixel:
            img[100:102, 200:202] += 300.0
        img += rng.normal(0, sigma, img.shape)
        frames[j] = np.repeat(img[:, :, None], 3, axis=2)
    return frames, t, (x0, y0, v)


def _reference(frames):
    return np.median(frames, axis=0).astype(np.float32)


def test_finds_the_mover_and_its_velocity():
    frames, t, (x0, y0, v) = _scene()
    res = find_moving_objects(frames, t, _reference(frames), fwhm=4.2)
    assert len(res['tracks']) == 1
    trk = res['tracks'][0]
    assert abs(trk['vx'] - v[0]) < 0.03 and abs(trk['vy'] - v[1]) < 0.03
    assert trk['n_frames'] >= 0.75 * _N
    assert trk['rms_px'] < 1.0 and trk['mean_snr'] > 5.0
    tmid = trk['t_mid']
    assert abs(trk['x0'] - (x0 + v[0] * tmid)) < 1.5 and abs(trk['y0'] - (y0 + v[1] * tmid)) < 1.5


def test_no_mover_means_no_tracks_despite_stars_and_a_hot_pixel():
    frames, t, _ = _scene(mover=False)
    res = find_moving_objects(frames, t, _reference(frames), fwhm=4.2)
    assert res['tracks'] == []


def test_a_stationary_source_is_not_reported():
    frames, t, _ = _scene(mover=True, v=(0.02, 0.01))          # ~0.6 px in 30 min: < 2 FWHM
    res = find_moving_objects(frames, t, _reference(frames), fwhm=4.2)
    assert res['tracks'] == []


def test_a_faster_mover_in_another_direction():
    frames, t, (x0, y0, v) = _scene(v=(-0.9, 0.5), seed=3)
    res = find_moving_objects(frames, t, _reference(frames), fwhm=4.2)
    assert len(res['tracks']) == 1
    assert abs(res['tracks'][0]['vx'] - v[0]) < 0.04 and abs(res['tracks'][0]['vy'] - v[1]) < 0.04


def test_two_movers_are_both_found():
    a, t, _ = _scene(v=(0.55, -0.30), seed=1)
    # second mover starts elsewhere: shift its source by re-adding a displaced copy
    yy, xx = np.mgrid[:_H, :_W].astype(np.float32)
    frames = a.copy()
    for j in range(_N):
        mx, my = 250.0 + (-0.4) * t[j], 40.0 + 0.6 * t[j]
        frames[j] += (200.0 * np.exp(-((yy - my) ** 2 + (xx - mx) ** 2) / (2 * 1.8 ** 2)))[:, :, None]
    res = find_moving_objects(frames, t, _reference(frames), fwhm=4.2)
    assert len(res['tracks']) == 2
    found = sorted((round(tr['vx'], 2), round(tr['vy'], 2)) for tr in res['tracks'])
    (ax_, ay_), (bx_, by_) = found                     # sorted by vx: the -0.4 mover first
    assert abs(ax_ + 0.4) < 0.04 and abs(ay_ - 0.6) < 0.04
    assert abs(bx_ - 0.55) < 0.04 and abs(by_ + 0.30) < 0.04


def test_star_mask_covers_stars_more_for_bright_ones():
    n = 6
    yx = np.zeros(n, dtype=[('xcentroid', 'f8'), ('ycentroid', 'f8'), ('flux', 'f8')])
    yx['xcentroid'] = [40.0, 10.0, 80.0, 60.0, 150.0, 120.0]
    yx['ycentroid'] = [50.0, 20.0, 80.0, 20.0, 80.0, 50.0]
    yx['flux'] = [1000.0] * 5 + [300000.0]              # five ordinary stars, one very bright
    m = star_mask(np.zeros((100, 200), np.float32), yx, 4.0)
    assert m[50, 40] and m[50, 120]
    assert m[:, 100:140].sum() > 3 * m[:, 25:55].sum()


def test_detect_in_residual_respects_threshold_and_mask():
    rng = np.random.default_rng(0)
    d = rng.normal(0, 5, (100, 100)).astype(np.float32)
    yy, xx = np.mgrid[:100, :100]
    d += 60 * np.exp(-((yy - 30) ** 2 + (xx - 40) ** 2) / (2 * 1.8 ** 2))
    det = detect_in_residual(d, 4.2, 5.0)
    assert any(abs(x - 40) < 2 and abs(y - 30) < 2 for x, y, _ in det)
    mask = np.zeros((100, 100), bool)
    mask[20:40, 30:50] = True
    det2 = detect_in_residual(d, 4.2, 5.0, mask)
    assert not any(abs(x - 40) < 3 and abs(y - 30) < 3 for x, y, _ in det2)
    assert len(detect_in_residual(np.zeros((50, 50), np.float32), 4.0, 5.0)) == 0


def test_tracked_stack_brings_a_faint_mover_out_of_the_noise(tmp_path):
    frames, t, (x0, y0, v) = _scene(amp=25.0, sigma=8.0, seed=5)   # ~1 sigma per frame
    trk = {'x0': x0 + v[0] * np.median(t), 'y0': y0 + v[1] * np.median(t), 'vx': v[0], 'vy': v[1],
           't_mid': float(np.median(t))}
    img = tracked_stack(frames, t, trk, half=25)
    assert img is not None and img.shape == (51, 51, 3)
    centre = img[22:29, 22:29, 1].mean() - np.median(img[:, :, 1])
    noise = 1.4826 * np.median(np.abs(img[:, :, 1] - np.median(img[:, :, 1])))
    assert centre > 4 * noise                                   # invisible per frame, clear stacked
    # off-frame tracks are skipped, and too few windows gives None
    far = dict(trk, x0=1e4)
    assert tracked_stack(frames, t, far, half=25) is None


def test_outputs_and_summary(tmp_path):
    frames, t, _ = _scene(seed=7)
    res = find_moving_objects(frames, t, _reference(frames), fwhm=4.2)
    paths = write_outputs(res, frames, t, str(tmp_path / 'run'), stack_tracks=True)
    assert any(p.endswith('_moving_objects.csv') for p in paths)
    assert any(p.endswith('_moving_1.fits') for p in paths)
    rows = list(csv.DictReader(open(paths[0])))
    assert len(rows) == len(res['tracks']) and float(rows[0]['speed_px_per_min']) > 0.4
    assert 'linked track' in format_summary(res)
    assert write_outputs({'tracks': []}, frames, t, str(tmp_path / 'none')) == []


def test_link_step_alone_ignores_random_clutter():
    rng = np.random.default_rng(11)
    det = np.column_stack([rng.integers(0, 40, 800), rng.uniform(0, 30, 800),
                           rng.uniform(0, _W, 800), rng.uniform(0, _H, 800),
                           rng.uniform(5, 7, 800)])
    assert find_tracks(det, (_H, _W), 30.0, 4.0, 40) == []
    assert collect_detections(np.zeros((2, 20, 20, 3), np.float32), np.array([0.0, 1.0]),
                              np.zeros((20, 20), np.float32), 4.0, 5.0, None).shape[1] == 5
