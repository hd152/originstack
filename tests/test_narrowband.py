"""Tests for narrowband palette enhancements in src/channel_combine.py."""
from __future__ import annotations

import numpy as np

from src.channel_combine import fix_narrowband_stars, narrowband_combine, scnr_green


def test_sho_palette_mapping():
    H, W = 60, 60
    ha = np.full((H, W), 0.8, np.float32)
    oiii = np.full((H, W), 0.3, np.float32)
    sii = np.full((H, W), 0.5, np.float32)
    rgb = narrowband_combine("sho", ha=ha, oiii=oiii, sii=sii)
    assert rgb.shape == (H, W, 3)
    # SHO: R<-SII, G<-Ha, B<-OIII. After per-channel normalise a constant frame
    # maps to 0, so use a gradient instead to check ordering.
    ha = np.tile(np.linspace(0, 1, W, dtype=np.float32), (H, 1))
    rgb = narrowband_combine("sho", ha=ha, oiii=oiii, sii=sii)
    # Green (Ha) should vary across columns; R/B (constant) should be flat.
    assert rgb[:, :, 1].std() > 0.1
    assert rgb[:, 0, 1].mean() < rgb[:, -1, 1].mean()


def test_scnr_removes_green_cast():
    # Pure green field -> SCNR should pull green down toward (R+B)/2 = 0.
    rgb = np.zeros((40, 40, 3), np.float32)
    rgb[:, :, 1] = 0.9
    out = scnr_green(rgb, amount=1.0)
    assert out[:, :, 1].max() < 0.05        # green suppressed
    # Where green is legitimately below neutral, it is untouched.
    rgb2 = np.dstack([np.full((10, 10), 0.8, np.float32),
                      np.full((10, 10), 0.3, np.float32),
                      np.full((10, 10), 0.8, np.float32)])
    out2 = scnr_green(rgb2, amount=1.0)
    assert np.allclose(out2[:, :, 1], 0.3)  # G < (R+B)/2 -> unchanged


def test_scnr_amount_zero_is_noop():
    rng = np.random.default_rng(0)
    rgb = rng.uniform(0, 1, (20, 20, 3)).astype(np.float32)
    assert np.allclose(scnr_green(rgb, 0.0), np.clip(rgb, 0, 1))


def _star_image(H, W, cx, cy, color):
    yy, xx = np.mgrid[0:H, 0:W]
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 3.0 ** 2)).astype(np.float32)
    return np.stack([g * color[0], g * color[1], g * color[2]], axis=2) + 0.01


def test_desaturate_neutralises_star():
    H, W = 80, 80
    # magenta star (R,B high, G low) on dark bg
    rgb = _star_image(H, W, 40, 40, (1.0, 0.1, 1.0))
    rgb = np.clip(rgb, 0, 1).astype(np.float32)
    before = rgb[40, 40].copy()
    out = fix_narrowband_stars(rgb, mode="desaturate", strength=1.0)
    after = out[40, 40]
    # channel spread (saturation proxy) should shrink at the star core
    assert (after.max() - after.min()) < (before.max() - before.min())


def test_rgb_recolor_transplants_color():
    H, W = 80, 80
    nb = np.clip(_star_image(H, W, 40, 40, (1.0, 0.1, 1.0)), 0, 1).astype(np.float32)
    # broadband reference: same star but blue-white
    ref = np.clip(_star_image(H, W, 40, 40, (0.6, 0.7, 1.0)), 0, 1).astype(np.float32)
    out = fix_narrowband_stars(nb, mode="rgb", rgb_stars=ref, strength=1.0)
    # after recolor the star core should lean blue (ref colour), not magenta
    assert out[40, 40, 2] >= out[40, 40, 0]


def test_recolor_noop_when_no_stars():
    rgb = np.full((30, 30, 3), 0.2, np.float32)  # flat, no stars
    out = fix_narrowband_stars(rgb, mode="desaturate")
    assert np.allclose(out, rgb)
