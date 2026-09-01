"""Tests for src/star_repair.py — saturated star core reconstruction."""
from __future__ import annotations

import numpy as np

from src.star_repair import _fit_moffat_wing, _moffat, repair_saturated_stars


def _moffat_star(H, W, cx, cy, amp, alpha, beta, bg=100.0):
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(xx - cx, yy - cy)
    return (bg + _moffat(r, amp, alpha, beta)).astype(np.float32)


def test_fit_recovers_moffat_params():
    yy, xx = np.mgrid[0:40, 0:40]
    r = np.hypot(xx - 20, yy - 20).ravel()
    v = _moffat(r, amp=5000.0, alpha=3.0, beta=2.5)
    fit = _fit_moffat_wing(r, v)
    assert fit is not None
    amp, alpha, beta = fit
    assert abs(alpha - 3.0) < 0.5
    assert abs(beta - 2.5) < 1.0


def test_repairs_clipped_core():
    H, W = 120, 120
    # True bright star, then clip it to simulate saturation.
    true = _moffat_star(H, W, 60, 60, amp=60000.0, alpha=4.0, beta=2.6)
    sat_level = 30000.0
    clipped = np.minimum(true, sat_level)
    rgb = np.stack([clipped] * 3, axis=2).astype(np.float32)

    peak_before = float(rgb[60, 60, 1])
    out = repair_saturated_stars(rgb, verbose=False)
    peak_after = float(out[60, 60, 1])

    # Core peak should be lifted back well above the clip level, toward truth.
    assert peak_after > peak_before + 5000.0
    assert peak_after > sat_level
    # Unsaturated wing pixels must be untouched.
    assert np.allclose(out[60, 90, 1], rgb[60, 90, 1])
    # Never overshoots wildly past the true peak.
    assert peak_after < true[60, 60] * 1.6


def test_preserves_star_color():
    H, W = 120, 120
    # A red-dominant star: red wing brighter than blue.
    base = _moffat_star(H, W, 60, 60, amp=60000.0, alpha=4.0, beta=2.6, bg=100.0)
    blue = _moffat_star(H, W, 60, 60, amp=30000.0, alpha=4.0, beta=2.6, bg=100.0)
    sat = 25000.0
    rgb = np.stack([np.minimum(base, sat),
                    np.minimum((base + blue) / 2, sat),
                    np.minimum(blue, sat)], axis=2).astype(np.float32)
    out = repair_saturated_stars(rgb)
    # After repair the reconstructed core keeps red > blue (colour preserved).
    assert out[60, 60, 0] > out[60, 60, 2]


def test_no_saturation_is_noop():
    H, W = 80, 80
    rgb = _moffat_star(H, W, 40, 40, amp=5000.0, alpha=4.0, beta=2.6)
    rgb = np.stack([rgb] * 3, axis=2).astype(np.float32)
    # Peak well below max*sat_frac after adding a brighter isolated pixel elsewhere
    rgb[5, 5, :] = rgb.max() * 2  # a single hot pixel is < min_core, skipped
    out = repair_saturated_stars(rgb)
    assert np.allclose(out, rgb)


def test_shape_and_dtype_preserved():
    H, W = 100, 100
    true = _moffat_star(H, W, 50, 50, amp=60000.0, alpha=4.0, beta=2.6)
    rgb = np.stack([np.minimum(true, 30000.0)] * 3, axis=2).astype(np.float32)
    out = repair_saturated_stars(rgb)
    assert out.shape == rgb.shape
    assert out.dtype == np.float32
