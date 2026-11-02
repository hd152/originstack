"""tools/bench_lsb_injection.py: the injected field, the transfer function, aperture curves."""
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import bench_lsb_injection as bl  # noqa: E402

H, W = 512, 640


def _noise_stack(seed=0, smooth=0.0):
    rng = np.random.default_rng(seed)
    ch = []
    for _ in range(3):
        n = rng.normal(0, 10.0, (H, W))
        if smooth:
            n = ndimage.gaussian_filter(n, smooth) * (smooth * 3.5)
        ch.append(1000.0 + n)
    return np.stack(ch, axis=2).astype(np.float32)


def test_cirrus_field_statistics_and_spectrum():
    f = bl.cirrus_field((H, W), np.random.default_rng(3))
    assert abs(f.mean()) < 0.05 and abs(f.std() - 1.0) < 0.01
    fy, fx = np.fft.fftfreq(H)[:, None], np.fft.fftfreq(W)[None, :]
    k = np.hypot(fy, fx)
    p = np.abs(np.fft.fft2(f)) ** 2
    edges = np.geomspace(1 / 200, 1 / 24, 9)
    mid, pw = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (k >= lo) & (k < hi)
        mid.append(np.sqrt(lo * hi))
        pw.append(p[s].mean())
    slope = np.polyfit(np.log(mid), np.log(pw), 1)[0]
    assert -3.4 < slope < -2.4
    assert p[k > 1 / 10].sum() < 1e-4 * p.sum()                # nothing below ~10 px


def test_transfer_function_identity_blur_and_background_removal():
    rng = np.random.default_rng(4)
    field = bl.cirrus_field((H, W), rng, min_period=16, max_period=600)
    ident = bl.transfer_function(field.copy(), field)
    assert all(abs(r['T'] - 1.0) < 1e-6 for r in ident)
    blurred = bl.transfer_function(ndimage.gaussian_filter(field, 8.0), field)
    t = {r['period_lo']: r['T'] for r in blurred}
    assert t[16.0] < 0.4 and t[256.0] > 0.9                    # fine scales smoothed, large kept
    flat = bl.transfer_function(field - ndimage.gaussian_filter(field, 25.0), field)
    t = {r['period_lo']: r['T'] for r in flat}
    assert t[256.0] < 0.2 and t[16.0] > 0.9                    # a background step eats the large scales


def test_aperture_curves_see_correlated_noise():
    white = bl.luminance_of(_noise_stack(1))
    corr = bl.luminance_of(_noise_stack(1, smooth=2.0))
    rw = bl.aperture_curves(white, np.ones(white.shape, bool), (4, 8))
    rc = bl.aperture_curves(corr, np.ones(corr.shape, bool), (4, 8))
    assert all(0.8 < r['ratio'] < 1.25 for r in rw)
    assert rc[0]['ratio'] > 1.5 * rw[0]['ratio']


def test_aperture_curves_skip_masked_regions():
    lum = bl.luminance_of(_noise_stack(2))
    mask = np.ones(lum.shape, bool)
    mask[:, : W // 2] = False
    rows = bl.aperture_curves(lum, mask, (8,))
    assert rows[0]['n'] >= 8
    lum[:, W // 2:] = 0.0                                       # the sky half is silent; unmasked half is not used
    assert bl.aperture_curves(lum, mask, (8,))[0]['scatter'] < 1e-6


def test_bench_end_to_end_with_a_smoothing_pipeline():
    stack = _noise_stack(5)
    res = bl.bench(stack, [0.5, 2.0], run=lambda a: ndimage.gaussian_filter(a, (6.0, 6.0, 0)),
                   seed=2)
    assert set(res['amplitudes']) == {'0.5', '2.0'}
    t = {r['period_lo']: r['T'] for r in res['amplitudes']['2.0']}
    assert t[16.0] < 0.5 and t[128.0] > 0.9
    text = bl.format_report(res)
    assert 'transfer function' in text and 'aperture' in text
    assert res['apertures_control'][0]['ratio'] > res['apertures_input'][0]['ratio']
