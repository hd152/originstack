"""Tests for autodetect_bayer_orientation (src/debayer.py).

Regression coverage for a real bug found via --stream on a Celestron Origin
Trifid Nebula session: BAYERPAT declared 'GBRG' in the FITS header, but the
green sub-pixels were actually laid out on the OTHER diagonal (matching
'RGGB'/'BGGR' instead) -- a single-axis row-orientation mismatch between the
capture software's header and the pixel data as loaded. Demosaicing with the
declared-but-wrong pattern produced a severe (~25% of signal) 2x2-periodic
bias surviving into every output channel.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.debayer import autodetect_bayer_orientation, debayer, green_equalize


def _mosaic(rgb: np.ndarray, pattern: str) -> np.ndarray:
    """Inverse of debayer: sample rgb at each Bayer-tile position's own
    channel, matching src.debayer._PATTERN_OFFSETS' (row, col) convention."""
    from src.debayer import _PATTERN_OFFSETS
    offsets = _PATTERN_OFFSETS[pattern]
    ch_for_offset = {offsets[0]: 0, offsets[1]: 1, offsets[2]: 1, offsets[3]: 2}
    H, W, _ = rgb.shape
    out = np.zeros((H, W), dtype=np.float32)
    for (r, c), ch in ch_for_offset.items():
        out[r::2, c::2] = rgb[r::2, c::2, ch]
    return out


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
def test_autodetect_keeps_correctly_labeled_pattern(pattern):
    """Real sensor data with only a small (<20%) G1/G2 sensitivity mismatch
    -- what green_equalize is designed to correct -- must NOT be swapped."""
    rng = np.random.default_rng(0)
    H, W = 200, 200
    rgb = np.empty((H, W, 3), dtype=np.float32)
    for ch, mean in enumerate((1200.0, 2000.0, 3400.0)):  # distinct per-channel
        rgb[:, :, ch] = mean + rng.standard_normal((H, W)).astype(np.float32) * 30.0
    raw = _mosaic(rgb, pattern)
    detected = autodetect_bayer_orientation(raw, pattern)
    assert detected == pattern


@pytest.mark.parametrize("declared,actual", [
    ("GBRG", "RGGB"), ("RGGB", "GBRG"), ("GRBG", "BGGR"), ("BGGR", "GRBG"),
])
def test_autodetect_corrects_row_flip_mismatch(declared, actual):
    """Data actually laid out as `actual` but mislabeled `declared` (its
    row-flip counterpart) must be detected and corrected back to `actual`."""
    rng = np.random.default_rng(1)
    H, W = 200, 200
    rgb = np.empty((H, W, 3), dtype=np.float32)
    for ch, mean in enumerate((1200.0, 2000.0, 3400.0)):  # distinct per-channel
        rgb[:, :, ch] = mean + rng.standard_normal((H, W)).astype(np.float32) * 30.0
    raw = _mosaic(rgb, actual)  # data genuinely follows `actual`'s layout
    detected = autodetect_bayer_orientation(raw, declared)
    assert detected == actual


def test_autodetect_reproduces_trifid_bug_fix_end_to_end():
    """End-to-end: mosaic a smooth synthetic image as RGGB (matching the real
    Trifid data's true layout), declare it GBRG (matching the file's actual,
    wrong header), and confirm debayering with the auto-corrected pattern
    removes the phase-dependent checkerboard the raw declared pattern causes.
    """
    rng = np.random.default_rng(2)
    H, W = 300, 300
    rgb = np.empty((H, W, 3), dtype=np.float32)
    for ch, mean in enumerate((15000.0, 19000.0, 26000.0)):  # ~matches real Trifid levels
        rgb[:, :, ch] = mean + rng.standard_normal((H, W)).astype(np.float32) * 400.0
    raw = _mosaic(rgb, "RGGB")  # true layout, matches the real bug report

    declared = "GBRG"  # wrong, matches the real file's header

    # Without the fix: debayering with the declared (wrong) pattern.
    eq_wrong = green_equalize(raw, pattern=declared)
    rgb_wrong = debayer(eq_wrong, pattern=declared, method="malvar")
    crop = rgb_wrong[20:-20, 20:-20, 1].astype(np.float64)  # green channel
    spread_wrong = max(
        crop[0::2, 0::2].mean(), crop[0::2, 1::2].mean(),
        crop[1::2, 0::2].mean(), crop[1::2, 1::2].mean(),
    ) - min(
        crop[0::2, 0::2].mean(), crop[0::2, 1::2].mean(),
        crop[1::2, 0::2].mean(), crop[1::2, 1::2].mean(),
    )
    assert spread_wrong > 1000.0  # reproduces the real ~4230 ADU-scale bug

    # With the fix: auto-corrected pattern.
    corrected = autodetect_bayer_orientation(raw, declared)
    assert corrected == "RGGB"
    eq_fixed = green_equalize(raw, pattern=corrected)
    rgb_fixed = debayer(eq_fixed, pattern=corrected, method="malvar")
    crop2 = rgb_fixed[20:-20, 20:-20, 1].astype(np.float64)
    spread_fixed = max(
        crop2[0::2, 0::2].mean(), crop2[0::2, 1::2].mean(),
        crop2[1::2, 0::2].mean(), crop2[1::2, 1::2].mean(),
    ) - min(
        crop2[0::2, 0::2].mean(), crop2[0::2, 1::2].mean(),
        crop2[1::2, 0::2].mean(), crop2[1::2, 1::2].mean(),
    )
    assert spread_fixed < 50.0  # clean, matches noise floor


def test_autodetect_handles_unknown_pattern_gracefully():
    raw = np.full((50, 50), 1000.0, dtype=np.float32)
    assert autodetect_bayer_orientation(raw, "NOTAPATTERN") == "NOTAPATTERN"


def test_autodetect_handles_near_zero_green_gracefully():
    """Near-black data (e.g. a bias frame) must not divide by ~0."""
    raw = np.zeros((50, 50), dtype=np.float32)
    result = autodetect_bayer_orientation(raw, "GBRG")
    assert result == "GBRG"
