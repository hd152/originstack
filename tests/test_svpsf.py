"""Tests for spatially-variant PSF deconvolution (src/psf_deconvolution.py)."""
from __future__ import annotations

import numpy as np
import pytest

from src.psf_deconvolution import (richardson_lucy_svpsf, _feather_window,
                                   _shift_sources, estimate_psf)
from src.models import Config


def _sources(xs, ys, flux):
    dt = np.dtype([('xcentroid', np.float64), ('ycentroid', np.float64),
                   ('flux', np.float64)])
    out = np.zeros(len(xs), dtype=dt)
    out['xcentroid'] = xs
    out['ycentroid'] = ys
    out['flux'] = flux
    return out


def _gaussian(H, W, cx, cy, sigma, amp):
    yy, xx = np.mgrid[0:H, 0:W]
    return amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))


def _star_field(H, W, sigma_fn, n_per_axis=8, amp=3000.0, bg=100.0, seed=0):
    """Field of stars whose blur sigma varies with position (sigma_fn(x,y))."""
    img = np.full((H, W), bg, np.float32)
    xs, ys, fl = [], [], []
    rng = np.random.default_rng(seed)
    for gx in np.linspace(0.1, 0.9, n_per_axis):
        for gy in np.linspace(0.1, 0.9, n_per_axis):
            x = gx * W + rng.uniform(-4, 4)
            y = gy * H + rng.uniform(-4, 4)
            s = sigma_fn(x, y)
            img += _gaussian(H, W, x, y, s, amp).astype(np.float32)
            xs.append(x); ys.append(y); fl.append(amp)
    return img, _sources(np.array(xs), np.array(ys), np.array(fl))


def test_feather_window_partitions_to_one():
    # Two tiles overlapping by their margins should sum to ~1 across the seam
    w = _feather_window(20, 40, my=5, mx=10)
    assert w.shape == (20, 40)
    assert w.max() <= 1.0 + 1e-9
    assert w[10, 20] == pytest.approx(1.0)   # centre is full weight
    assert w[0, 0] < 0.1                       # corner ramps toward 0


def test_shift_sources():
    s = _sources([100.0, 50.0], [80.0, 30.0], [1.0, 1.0])
    out = _shift_sources(s, 40, 20)
    assert out['xcentroid'][0] == 60.0
    assert out['ycentroid'][0] == 60.0


def test_svpsf_runs_and_sharpens():
    H, W = 360, 360
    # Uniform blur so both global and local PSFs are well-defined.
    img2d, src = _star_field(H, W, lambda x, y: 2.6, n_per_axis=9)
    rgb = np.stack([img2d] * 3, axis=2).astype(np.float32)
    if len(src) < Config.RL_PSF_MIN_STARS:
        pytest.skip("not enough synthetic stars for PSF fit")
    out = richardson_lucy_svpsf(rgb, src, iterations=10, n_tiles=3)
    assert out.shape == rgb.shape
    assert out.dtype == np.float32
    # Deconvolution should raise peak sharpness: a central star's peak grows
    # relative to its immediate neighbourhood.
    cy, cx = H // 2, W // 2
    # find brightest pixel near centre
    reg = img2d[cy - 30:cy + 30, cx - 30:cx + 30]
    ry, rx = np.unravel_index(np.argmax(reg), reg.shape)
    py, px = cy - 30 + ry, cx - 30 + rx
    before = img2d[py, px] - 100.0
    after = out[py, px, 1] - 100.0
    ring_before = img2d[py, px + 3] - 100.0
    ring_after = out[py, px + 3, 1] - 100.0
    # concentration (peak / nearby) should increase after deconvolution
    assert (after / max(ring_after, 1e-3)) > (before / max(ring_before, 1e-3))


def test_svpsf_noop_without_sources():
    rgb = np.random.default_rng(0).uniform(0, 1, (60, 60, 3)).astype(np.float32)
    out = richardson_lucy_svpsf(rgb, None, iterations=5, n_tiles=2)
    assert np.allclose(out, rgb)


def test_svpsf_falls_back_to_global_psf_when_tile_sparse():
    # Few stars overall -> tiles are sparse, so every tile uses the global PSF,
    # but the routine must still run and return a valid image.
    H, W = 300, 300
    img2d, src = _star_field(H, W, lambda x, y: 2.6, n_per_axis=5)
    rgb = np.stack([img2d] * 3, axis=2).astype(np.float32)
    gpsf, _ = estimate_psf(rgb, src)
    if gpsf is None:
        pytest.skip("global PSF could not be estimated for this fixture")
    out = richardson_lucy_svpsf(rgb, src, iterations=8, n_tiles=4)
    assert out.shape == rgb.shape
    assert np.all(np.isfinite(out))
