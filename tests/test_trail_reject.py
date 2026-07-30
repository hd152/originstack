"""Tests for src/trail_reject.py — satellite/aircraft trail rejection."""
from __future__ import annotations

import numpy as np
import pytest

from src.trail_reject import detect_trail_mask, reject_trails, _bresenham_line


def _sky(H, W, level=100.0, noise=3.0, seed=0):
    rng = np.random.default_rng(seed)
    return (level + rng.standard_normal((H, W)) * noise).astype(np.float32)


def _add_stars(img, n=40, seed=1):
    rng = np.random.default_rng(seed)
    H, W = img.shape
    yy, xx = np.mgrid[-3:4, -3:4]
    g = np.exp(-(xx * xx + yy * yy) / (2 * 1.2 ** 2))
    for _ in range(n):
        y = rng.integers(5, H - 5)
        x = rng.integers(5, W - 5)
        img[y - 3:y + 4, x - 3:x + 4] += (2000 * g).astype(np.float32)
    return img


def _add_trail(img, amp=1500.0):
    """Draw a bright diagonal streak across the frame."""
    H, W = img.shape
    xs = np.arange(W)
    ys = (0.4 * xs + 40).astype(int)
    ok = (ys >= 1) & (ys < H - 1)
    for dy in (-1, 0, 1):
        img[np.clip(ys[ok] + dy, 0, H - 1), xs[ok]] += amp
    return img


def test_detects_trail():
    img = _add_trail(_add_stars(_sky(300, 400)))
    mask, n = detect_trail_mask(img)
    assert mask is not None and n >= 1
    assert mask.sum() > 50


def test_no_false_positive_on_starfield():
    img = _add_stars(_sky(300, 400), n=60)
    mask, n = detect_trail_mask(img)
    # Compact stars must not be reported as a trail.
    assert mask is None or n == 0


def test_reject_removes_streak_flux():
    H, W = 300, 400
    base = _add_stars(_sky(H, W))
    trailed = _add_trail(base.copy(), amp=1500.0)
    rgb = np.stack([trailed] * 3, axis=2).astype(np.float32)
    out, n = reject_trails(rgb)
    assert n >= 1
    lum_after = 0.299 * out[:, :, 0] + 0.587 * out[:, :, 1] + 0.114 * out[:, :, 2]
    # Sample the known trail path (avoiding star pixels): mean flux there must
    # fall back toward the ~100 ADU sky, not stay at ~1600.
    xs = np.arange(W)
    ys = np.clip((0.4 * xs + 40).astype(int), 0, H - 1)
    before_path = trailed[ys, xs].mean()
    after_path = lum_after[ys, xs].mean()
    assert before_path > 1000.0            # trail was bright to begin with
    assert after_path < before_path * 0.4  # trail flux largely erased


def test_reject_preserves_shape_and_noop_when_clean():
    rgb = np.stack([_add_stars(_sky(200, 200))] * 3, axis=2).astype(np.float32)
    out, n = reject_trails(rgb)
    assert out.shape == rgb.shape
    assert out.dtype == np.float32
    if n == 0:
        assert np.allclose(out, rgb)


def test_area_guard_aborts_on_huge_detection():
    # A frame that is almost entirely "bright" should not be gutted: the
    # max-area guard returns no mask.
    img = _sky(200, 200, level=100.0, noise=1.0)
    img[:] += 5000.0  # everything bright
    mask, n = detect_trail_mask(img, max_area_frac=0.08)
    assert mask is None


class TestBresenhamVsSkimage:
    """_bresenham_line replaced skimage.draw.line -- verify bit-exact pixel
    agreement against the real implementation wherever it's installed."""

    @pytest.mark.parametrize("r0,c0,r1,c1", [
        (0, 0, 0, 10), (0, 0, 10, 0), (0, 0, 10, 10), (0, 0, 10, 3),
        (0, 0, 3, 10), (5, 5, 5, 5), (10, 3, 0, 17), (20, 40, 22, 5),
        (-5, -5, 5, 5), (0, 0, -8, -3),
    ])
    def test_matches_skimage_draw_line(self, r0, c0, r1, c1):
        skdraw = pytest.importorskip("skimage.draw")
        rr_sk, cc_sk = skdraw.line(r0, c0, r1, c1)
        rr, cc = _bresenham_line(r0, c0, r1, c1)
        np.testing.assert_array_equal(rr, rr_sk)
        np.testing.assert_array_equal(cc, cc_sk)
