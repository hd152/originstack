"""Tests for src/star_removal.py — inpaint stars with local background."""
from __future__ import annotations

import numpy as np

from src.quality import _SOURCES_DTYPE
from src.star_removal import build_star_mask, remove_stars


def _sources(entries):
    """entries: list of (xcentroid, ycentroid, flux, peak)."""
    out = np.zeros(len(entries), dtype=_SOURCES_DTYPE)
    for i, (x, y, flux, peak) in enumerate(entries):
        out[i] = (x, y, flux, peak, 0.0, 0.0, 1.0, 2.0, 2.0, 0.0)
    return out


def _star_image(H, W, cx, cy, amp, sigma, bg=100.0):
    yy, xx = np.mgrid[0:H, 0:W]
    star = amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return (bg + star).astype(np.float32)


def test_removes_bright_star():
    H, W = 150, 150
    lum = _star_image(H, W, 75, 75, amp=20000.0, sigma=2.0)
    rgb = np.stack([lum] * 3, axis=2)
    sources = _sources([(75.0, 75.0, 20000.0, 20100.0)])

    out, n = remove_stars(rgb, sources, fwhm=4.0, verbose=False)
    assert n == 1
    # Star core pulled down close to the flat background level.
    assert out[75, 75, 1] < rgb[75, 75, 1] - 5000.0
    assert abs(out[75, 75, 1] - 100.0) < 500.0


def test_background_far_from_star_untouched():
    H, W = 150, 150
    lum = _star_image(H, W, 75, 75, amp=20000.0, sigma=2.0)
    rgb = np.stack([lum] * 3, axis=2)
    sources = _sources([(75.0, 75.0, 20000.0, 20100.0)])

    out, n = remove_stars(rgb, sources, fwhm=4.0, verbose=False)
    assert n == 1
    # Corner far from the star's footprint is untouched.
    assert np.allclose(out[5, 5, :], rgb[5, 5, :])


def test_no_sources_is_noop():
    H, W = 60, 60
    rgb = np.full((H, W, 3), 100.0, dtype=np.float32)
    out, n = remove_stars(rgb, None, fwhm=4.0)
    assert n == 0
    assert np.array_equal(out, rgb)
    assert out is not rgb  # copy, not the same array


def test_fainter_star_gets_smaller_disk():
    H, W = 80, 80
    shape = (H, W)
    bright = _sources([(40.0, 40.0, 50000.0, 50000.0), (40.0, 40.0, 500.0, 500.0)])
    faint_only = _sources([(40.0, 40.0, 500.0, 500.0)])
    mask_bright, r_bright = build_star_mask(shape, bright, fwhm=4.0)
    mask_faint, r_faint = build_star_mask(shape, faint_only, fwhm=4.0)
    # A star far brighter than the field median gets a bigger disk.
    assert r_bright >= r_faint


def test_shape_and_dtype_preserved():
    H, W = 100, 100
    rgb = _star_image(H, W, 50, 50, amp=10000.0, sigma=2.0)
    rgb = np.stack([rgb] * 3, axis=2).astype(np.float32)
    sources = _sources([(50.0, 50.0, 10000.0, 10100.0)])
    out, n = remove_stars(rgb, sources, fwhm=4.0)
    assert out.shape == rgb.shape
    assert out.dtype == np.float32


def test_mask_area_safety_cap_aborts():
    # A dense grid of "stars" covering nearly the whole frame should trip the
    # area-fraction guard and return the image unchanged (n=0).
    H, W = 60, 60
    ys, xs = np.mgrid[2:H:4, 2:W:4]
    entries = [(float(x), float(y), 5000.0, 5000.0) for y, x in zip(ys.ravel(), xs.ravel())]
    sources = _sources(entries)
    rgb = np.full((H, W, 3), 100.0, dtype=np.float32)
    mask, _ = build_star_mask((H, W), sources, fwhm=4.0, radius_scale=3.0)
    assert mask is None
    out, n = remove_stars(rgb, sources, fwhm=4.0, radius_scale=3.0)
    assert n == 0
    assert np.array_equal(out, rgb)
