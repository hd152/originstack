"""Tests for src/local_normalize.py — per-frame additive background matching."""
from __future__ import annotations

import numpy as np
import pytest

from src.local_normalize import _HAS_SCIPY, _coarse_background, _upsample, local_normalize_stack

pytestmark = pytest.mark.skipif(not _HAS_SCIPY, reason="scipy not installed")


def _plane(H, W, gx, gy, c0):
    """Linear gradient plane."""
    yy, xx = np.mgrid[0:H, 0:W]
    return (c0 + gx * (xx / W) + gy * (yy / H)).astype(np.float32)


def test_coarse_background_rejects_stars():
    H, W = 120, 120
    frame = np.full((H, W, 1), 100.0, np.float32)
    # add bright stars — must NOT pull the low-percentile background up
    frame[30, 30, 0] = 5000
    frame[60, 90, 0] = 8000
    bg = _coarse_background(frame, grid=8, pct=30.0)
    assert np.allclose(bg, 100.0, atol=1.0)


def test_upsample_shape():
    g = np.random.default_rng(0).uniform(0, 1, (6, 6, 3)).astype(np.float32)
    up = _upsample(g, 100, 80)
    assert up.shape == (100, 80, 3)


def test_removes_per_frame_gradient_preserves_common_signal():
    N, H, W, C = 6, 120, 160, 3
    rng = np.random.default_rng(3)
    # Common signal: a fixed nebulosity bump present in EVERY frame.
    yy, xx = np.mgrid[0:H, 0:W]
    nebula = (400.0 * np.exp(-(((xx - 80) / 40) ** 2 + ((yy - 60) / 30) ** 2))).astype(np.float32)

    stack = np.empty((N, H, W, C), np.float32)
    for i in range(N):
        # Each frame: constant sky + a DIFFERENT gradient + common nebula + noise
        gx, gy = rng.uniform(-300, 300), rng.uniform(-300, 300)
        base = _plane(H, W, gx, gy, 100.0)
        for c in range(C):
            stack[i, :, :, c] = base + nebula + rng.standard_normal((H, W)) * 2.0

    # Sample a high-lever background point (far from the gradient origin) where
    # the per-frame gradient genuinely differs frame-to-frame.
    py, px = 100, 140
    spread_before = stack[:, py, px, 0].std()
    assert spread_before > 100.0  # the per-frame gradients really do differ here

    n = local_normalize_stack(stack, grid=16)
    assert n == N

    spread_after = stack[:, py, px, 0].std()
    # Per-frame background spread should collapse dramatically.
    assert spread_after < spread_before * 0.1

    # Common nebula signal must survive: the bump still stands well above sky.
    center = stack[:, 60, 80, 0].mean()
    corner = stack[:, 5, 5, 0].mean()
    assert center - corner > 250.0  # nebula (~400) largely preserved


def test_noop_single_frame():
    stack = np.random.default_rng(0).uniform(90, 110, (1, 40, 40, 3)).astype(np.float32)
    before = stack.copy()
    assert local_normalize_stack(stack) == 0
    assert np.allclose(stack, before)


def test_flat_frames_unchanged():
    # Identical flat frames -> deviations are ~0 -> no meaningful change.
    stack = np.full((4, 60, 60, 3), 100.0, np.float32)
    local_normalize_stack(stack, grid=12)
    assert np.allclose(stack, 100.0, atol=0.5)
