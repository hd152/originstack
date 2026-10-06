"""Finished-image fixes found by comparing against the Celestron Origin's own stack
(tools/bench_vs_origin.py): the preview white point on a small target, and clipped
star cores through the Gaia colour calibration."""
import numpy as np

from src.color_calibrate import apply_scales_inplace
from src.io_fits import _preview_white, render_preview_float
from src.models import Config


def _small_galaxy_field(seed=0):
    """Sky 1000 +- 10 with a faint galaxy covering ~0.3% of the frame."""
    rng = np.random.default_rng(seed)
    h, w = 400, 600
    img = 1000 + rng.normal(0, 10, (h, w, 3)).astype(np.float32)
    yy, xx = np.mgrid[:h, :w]
    galaxy = 3000 * np.exp(-(((yy - 200) / 8.0) ** 2 + ((xx - 300) / 14.0) ** 2))
    img += galaxy[..., None].astype(np.float32)
    return img, galaxy > 30


def test_white_point_never_hugs_the_sky():
    lum = np.random.default_rng(1).normal(100.0, 1.0, 200_000)
    white, sp_scale = _preview_white(lum, 100.0, 1.0, 99.5, 101.0)
    assert white >= 100.0 + Config.PREVIEW_WHITE_MIN_SIGMA
    assert sp_scale == 1.0 / 3.0          # the percentile was ~3 sigma: capped at 3x


def test_white_point_keeps_a_higher_percentile():
    lum = np.r_[np.random.default_rng(2).normal(100.0, 1.0, 1000), np.full(100, 5000.0)]
    white, sp_scale = _preview_white(lum, 100.0, 1.0, 99.5, 101.0)
    assert white == np.percentile(lum, 99.5) and sp_scale == 1.0


def test_symmetry_point_follows_a_small_white_point_raise():
    # percentile at sky + 40 sigma, floor at 50: SP scales by (40 - 1) / (50 - 1)
    lum = np.r_[np.full(990, 100.0), np.full(10, 140.0)]
    white, sp_scale = _preview_white(lum, 100.0, 1.0, 99.5, 101.0)
    assert white == 150.0
    assert abs(sp_scale - (np.percentile(lum, 99.5) - 101.0) / 49.0) < 1e-12


def test_small_target_does_not_turn_the_sky_to_snow():
    # Before: white = 99.5th percentile ~ sky + 5 sigma here, so sky noise spanned
    # the display range and a large share of sky pixels rendered bright.
    img, target = _small_galaxy_field()
    out = render_preview_float(img, stretch='ghs', black_sigma=1.0, color='preserve')
    sky = out[~target].mean(-1)
    assert np.mean(sky > 0.5) < 0.001
    assert out[target].mean() > 4 * sky.mean()     # the galaxy still stands out


def _clipped_star_frame():
    """Three clipped (equal R=G=B) star cores and an unclipped coloured patch."""
    img = np.full((60, 60, 3), 100.0, np.float32)
    for y, x in ((10, 10), (10, 40), (45, 12)):
        img[y:y + 3, x:x + 3] = 5000.0             # clipped plateaus, neutral
    img[30:33, 30:33] = (900.0, 1000.0, 1100.0)    # coloured, well below the plateau
    return img


def test_clipped_cores_stay_neutral_through_the_gains():
    img = _clipped_star_frame()
    apply_scales_inplace(img, (0.72, 1.0, 1.09))
    for y, x in ((11, 11), (11, 41), (46, 13)):
        np.testing.assert_allclose(img[y, x], img[y, x].max(), rtol=1e-6)


def test_unclipped_pixels_keep_the_calibrated_colour():
    img = _clipped_star_frame()
    apply_scales_inplace(img, (0.72, 1.0, 1.09))
    np.testing.assert_allclose(img[31, 31], (900 * 0.72, 1000.0, 1100 * 1.09), rtol=1e-6)
    np.testing.assert_allclose(img[0, 0], (72.0, 100.0, 109.0), rtol=1e-6)


def test_unclipped_bright_core_keeps_its_colour():
    # A smooth, unclipped galaxy core: hundreds of pixels within 2% of the
    # maximum, but one region -- and stars whose peaks differ.
    yy, xx = np.mgrid[:200, :200]
    core = np.exp(-((yy - 100) ** 2 + (xx - 100) ** 2) / (2 * 40.0 ** 2))
    img = (100 + core[..., None] * np.array([4000.0, 3000.0, 2000.0])).astype(np.float32)
    for (y, x), peak in zip(((20, 20), (20, 180), (180, 30)), (1500.0, 2200.0, 2900.0)):
        img[y - 1:y + 2, x - 1:x + 2] += peak
    expected = img * np.array([0.72, 1.0, 1.09], np.float32)
    apply_scales_inplace(img, (0.72, 1.0, 1.09))
    np.testing.assert_array_equal(img, expected)
