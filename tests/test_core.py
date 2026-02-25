import os
import tempfile
import numpy as np
from astropy.io import fits

from astro_stack import debayer_bilinear, calculate_shift, compute_quality_metrics


def make_star_image(shape=(64, 64), centers=[(32, 32)], amp=1000.0):
    im = np.zeros(shape, dtype=np.float32)
    for y, x in centers:
        yy, xx = np.indices(shape)
        r2 = (yy - y) ** 2 + (xx - x) ** 2
        im += amp * np.exp(-r2 / (2.0 * 2.0 ** 2))
    im += 100.0
    return im


def test_debayer_bilinear_shape():
    raw = np.zeros((10, 10), dtype=np.float32)
    raw[0::2, 0::2] = 100
    rgb = debayer_bilinear(raw, pattern='RGGB')
    assert rgb.shape == (10, 10, 3)


def test_calculate_shift_recovery():
    im = make_star_image((64, 64), centers=[(20, 20)])
    im2 = np.roll(np.roll(im, 3, axis=0), -2, axis=1)
    sy, sx = calculate_shift(im, im2, upsample=1)
    # im2 was rolled down 3 and left 2; expected shift to align is (-3, +2)
    assert abs(sy + 3) < 1.0
    assert abs(sx - 2) < 1.0


def test_quality_metrics_counts_stars():
    im = make_star_image((64, 64), centers=[(20, 20), (40, 10)])
    metrics = compute_quality_metrics(im)
    assert metrics['brightness'] > 0
    assert metrics['contrast'] > 0
    assert metrics['score'] > 0
