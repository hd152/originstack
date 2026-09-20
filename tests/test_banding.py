"""--banding-removal: injected row/column offsets go; stars, gradients, clean frames stay."""
import argparse

import numpy as np

from src.banding import banding_strength, remove_banding_2d, remove_banding_bayer, remove_banding_rgb
from src.frame_processor import _banding_cfg


def _mosaic(H=512, W=640, seed=0, row_sigma=0.0, col_sigma=0.0, stars=True):
    """RGGB-ish mosaic: per-colour levels, a sky gradient, noise, optional stars,
    plus per-row / per-column offsets shared by every colour in that row/column."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :W]
    level = np.zeros((H, W))
    for (py, px), lv in {(0, 0): 1000.0, (0, 1): 1500.0, (1, 0): 1500.0, (1, 1): 800.0}.items():
        level[py::2, px::2] = lv
    clean = level + 0.02 * yy + 0.01 * xx + rng.normal(0, 10, (H, W))
    if stars:
        for _ in range(60):
            cy, cx = rng.uniform(10, H - 10), rng.uniform(10, W - 10)
            clean += rng.uniform(800, 5000) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 4.0)
    rows = rng.normal(0, row_sigma, H) if row_sigma else np.zeros(H)
    cols = rng.normal(0, col_sigma, W) if col_sigma else np.zeros(W)
    return clean.astype(np.float32), (clean + rows[:, None] + cols[None, :]).astype(np.float32), rows, cols


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))


def test_row_and_column_offsets_are_removed_from_a_bayer_mosaic():
    clean, bad, rows, cols = _mosaic(row_sigma=6.0, col_sigma=3.0)
    out = remove_banding_bayer(bad)
    row_err = out.mean(axis=1) - clean.mean(axis=1)
    col_err = out.mean(axis=0) - clean.mean(axis=0)
    row_before = bad.mean(axis=1) - clean.mean(axis=1)
    col_before = bad.mean(axis=0) - clean.mean(axis=0)
    assert _rms(row_err) < 0.30 * _rms(row_before)
    assert _rms(col_err) < 0.40 * _rms(col_before)


def test_stars_and_the_sky_gradient_survive():
    clean, bad, _, _ = _mosaic(row_sigma=6.0)
    out = remove_banding_bayer(bad)
    star = np.unravel_index(np.argmax(clean), clean.shape)
    assert abs(out[star] - clean[star]) < 0.03 * clean[star]
    # large-scale gradient (top-to-bottom mean difference) is not flattened
    g_clean = clean[-100:].mean() - clean[:100].mean()
    g_out = out[-100:].mean() - out[:100].mean()
    assert abs(g_out - g_clean) < 1.0


def test_a_clean_frame_is_left_essentially_untouched():
    clean, _, _, _ = _mosaic()
    out = remove_banding_bayer(clean)
    assert float(np.abs(out - clean).max()) < 5.0
    assert _rms(out - clean) < 0.6      # against per-pixel noise of 10


def test_amount_scales_the_correction_and_zero_is_a_noop():
    clean, bad, _, _ = _mosaic(row_sigma=6.0)
    full = remove_banding_bayer(bad, amount=1.0)
    half = remove_banding_bayer(bad, amount=0.5)
    e = lambda x: _rms(x.mean(axis=1) - clean.mean(axis=1))
    assert e(full) < e(half) < e(bad)
    np.testing.assert_array_equal(remove_banding_bayer(bad, amount=0.0), bad)


def _plane(H=400, W=500, seed=0, row_sigma=0.0):
    """A single (non-mosaic) plane: uniform level, gentle gradient, noise, row offsets."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :W]
    clean = 1000 + 0.02 * yy + rng.normal(0, 10, (H, W))
    rows = rng.normal(0, row_sigma, H) if row_sigma else np.zeros(H)
    return clean.astype(np.float32), (clean + rows[:, None]).astype(np.float32), rows


def test_strength_reads_high_on_banded_and_near_zero_on_clean():
    clean, bad, _ = _plane(row_sigma=8.0)
    rr_bad, _, noise = banding_strength(bad)
    rr_clean, _, _ = banding_strength(clean)
    assert rr_bad > 3 * rr_clean and rr_bad > 4.0
    assert noise > 0


def test_mono_and_rgb_paths():
    clean, bad, _ = _plane(row_sigma=6.0)
    out = remove_banding_2d(bad, cols=False)
    assert _rms(out.mean(axis=1) - clean.mean(axis=1)) < 0.4 * _rms(bad.mean(axis=1) - clean.mean(axis=1))
    rgb = np.stack([bad, bad * 0.9, bad * 1.1], axis=2)
    o = remove_banding_rgb(rgb, cols=False)
    assert o.shape == rgb.shape and o.dtype == np.float32
    np.testing.assert_array_equal(remove_banding_2d(bad, amount=0), bad)


def test_config_helper():
    assert _banding_cfg(argparse.Namespace(banding_removal=False)) is None
    assert _banding_cfg(argparse.Namespace(banding_removal=True)) == (1.0, 3.0)
    assert _banding_cfg(argparse.Namespace(banding_removal=True, banding_amount=0.5,
                                           banding_sigma=2.0)) == (0.5, 2.0)
