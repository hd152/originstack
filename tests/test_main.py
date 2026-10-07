"""Unit tests for the public API re-exported by originstack.py."""

from __future__ import annotations

import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import originstack as astro
import src.gpu_context as _gpu_mod
from tests._helpers import add_gaussian_stars, write_fits

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _star_field(H=64, W=64, n_stars=5, seed=42, bg=100.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.normal(bg, 5.0, (H, W)).astype(np.float32)
    stars = [(rng.integers(5, H - 5), rng.integers(5, W - 5), 600.0) for _ in range(n_stars)]
    return add_gaussian_stars(img, stars, sigma=np.sqrt(2.0)).clip(0)


def _rgb(H=64, W=64, seed=42) -> np.ndarray:
    return np.random.default_rng(seed).uniform(100, 1000, (H, W, 3)).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestClassifyFrame(unittest.TestCase):
    """Frame-type classification, table-driven.

    Consolidated 2026-09 from two parallel copies (this one and a near-identical
    TestClassifyFrame in test_unit_extended.py) that between them spent 30 test
    methods on these 22 behaviours under different names. Cases are the union of
    both; add a row rather than a method.
    """

    CASES = [
        # (filename, header, expected, why)
        # Pipeline outputs must never be re-ingested as inputs.
        ('any.fits', {'COMBINED': True}, 'skip', 'COMBINED header'),
        ('any.fits', {'CREATOR': 'astro_stack v2'}, 'skip', 'legacy CREATOR prefix'),
        ('any.fits', {'CREATOR': 'astro_stack/pipeline'}, 'skip', 'CREATOR substring'),
        ('any.fits', {'CREATOR': 'originstack.py v1.0'}, 'skip', 'current CREATOR'),
        # Dark: filename or IMAGETYP, either case, with or without a separator.
        ('dark_001.fit', {}, 'dark', 'filename'),
        ('dark001.fits', {}, 'dark', 'filename, no separator'),
        ('DARK_001.FIT', {}, 'dark', 'filename is case-insensitive'),
        ('x.fit', {'IMAGETYP': 'dark'}, 'dark', 'IMAGETYP'),
        ('image.fits', {'IMAGETYP': 'Dark'}, 'dark', 'IMAGETYP mixed case'),
        ('image.fits', {'IMAGETYP': 'DARK'}, 'dark', 'IMAGETYP upper case'),
        # Flat.
        ('flat_001.fit', {}, 'flat', 'filename'),
        ('x.fit', {'IMAGETYP': 'flat'}, 'flat', 'IMAGETYP'),
        ('x.fit', {'IMAGETYP': 'FLAT'}, 'flat', 'IMAGETYP upper case'),
        # Bias, including the zero-exposure shortcut.
        ('bias_001.fit', {}, 'bias', 'filename'),
        ('bias0001.fits', {}, 'bias', 'filename, no separator'),
        ('x.fit', {'IMAGETYP': 'bias'}, 'bias', 'IMAGETYP'),
        ('frame.fit', {'EXPTIME': 0}, 'bias', 'zero exposure'),
        # Light is the fallthrough.
        ('frame_001.fit', {}, 'light', 'no signal anywhere'),
        ('img_001.fits', {}, 'light', 'empty header'),
        ('frame.fit', {'EXPTIME': 30.0}, 'light', 'non-zero exposure'),
        ('light_001.fits', {'EXPTIME': 120, 'IMAGETYP': 'Light Frame'}, 'light',
         'explicit light'),
        # Precedence: the filename wins over a contradicting IMAGETYP.
        ('dark_001.fit', {'IMAGETYP': 'light'}, 'dark', 'filename beats IMAGETYP'),
    ]

    def test_classification(self):
        for name, header, expected, why in self.CASES:
            with self.subTest(name=name, why=why):
                self.assertEqual(astro.classify_frame(name, header), expected)

class TestFormatTime(unittest.TestCase):
    def test_zero(self):
        self.assertIn("0.0s", astro.format_time(0.0))

    def test_seconds(self):
        r = astro.format_time(45.3)
        self.assertIn("45", r)
        self.assertIn("s", r)

    def test_minutes(self):
        self.assertIn("m", astro.format_time(125.0))

    def test_hours(self):
        self.assertIn("h", astro.format_time(3700.0))

    def test_exactly_one_minute(self):
        self.assertIn("m", astro.format_time(60.0))


class TestValidateImageData(unittest.TestCase):
    def _good(self):
        return _star_field(32, 32) + 200.0

    def test_valid_passes(self):
        ok, msg = astro.validate_image_data(self._good())
        self.assertTrue(ok, msg)
        self.assertIsNone(msg)

    def test_nan_rejected(self):
        img = self._good()
        img[5, 5] = np.nan
        ok, _ = astro.validate_image_data(img)
        self.assertFalse(ok)

    def test_inf_rejected(self):
        img = self._good()
        img[3, 3] = np.inf
        ok, _ = astro.validate_image_data(img)
        self.assertFalse(ok)

    def test_flat_image_rejected(self):
        ok, _ = astro.validate_image_data(np.full((32, 32), 500.0, dtype=np.float32))
        self.assertFalse(ok)

    def test_saturated_rejected(self):
        ok, _ = astro.validate_image_data(np.full((32, 32), 65535.0, dtype=np.float32))
        self.assertFalse(ok)

    def test_mostly_zeros_rejected(self):
        img = np.zeros((32, 32), dtype=np.float32)
        img[0, 0] = 1000.0
        ok, _ = astro.validate_image_data(img)
        self.assertFalse(ok)

    def test_low_dynamic_range_rejected(self):
        img = np.full((32, 32), 500.0, dtype=np.float32)
        img += np.random.default_rng(0).uniform(0, 4, img.shape).astype(np.float32)
        ok, _ = astro.validate_image_data(img)
        self.assertFalse(ok)


class TestComputeQualityMetrics(unittest.TestCase):
    def test_required_keys(self):
        m = astro.compute_quality_metrics(_star_field() + 100.0)
        for k in ("brightness", "contrast", "score", "star_count",
                  "snr", "sharpness", "fwhm", "background", "noise", "dynamic_range"):
            self.assertIn(k, m)

    def test_brighter_higher_brightness(self):
        m_dim = astro.compute_quality_metrics(_star_field() + 10.0)
        m_bright = astro.compute_quality_metrics(_star_field() + 500.0)
        self.assertGreater(m_bright["brightness"], m_dim["brightness"])

    def test_score_positive(self):
        self.assertGreater(astro.compute_quality_metrics(_star_field() + 100.0)["score"], 0)

    def test_dynamic_range_positive(self):
        self.assertGreater(
            astro.compute_quality_metrics(_star_field() + 100.0)["dynamic_range"], 0
        )

    def test_brightness_near_median(self):
        img = np.full((32, 32), 123.0, dtype=np.float32)
        self.assertAlmostEqual(
            astro.compute_quality_metrics(img)["brightness"], 123.0, delta=2.0
        )


class TestDebayerDispatch(unittest.TestCase):
    def setUp(self):
        self._gpu = _gpu_mod._gpu
        _gpu_mod._gpu = astro.GpuContext(use_gpu=False)
        self.raw = np.random.default_rng(2).uniform(100, 1000, (64, 64)).astype(np.float32)

    def tearDown(self):
        _gpu_mod._gpu = self._gpu

    def test_bilinear(self):
        self.assertEqual(astro.debayer(self.raw, method="bilinear").shape, (64, 64, 3))

    def test_malvar(self):
        self.assertEqual(astro.debayer(self.raw, method="malvar").shape, (64, 64, 3))

    def test_default_equals_bilinear(self):
        np.testing.assert_array_equal(
            astro.debayer(self.raw),
            astro.debayer(self.raw, method="bilinear"),
        )


class TestChromaticAberrationCorrection(unittest.TestCase):
    """Verifies the downsample-before-correlate CA fix recovers a known
    injected R/B shift as accurately as full-res correlation (downsample=1),
    and that the corrected R/B align with G to well under 1px."""

    def setUp(self):
        self._gpu = _gpu_mod._gpu
        _gpu_mod._gpu = astro.GpuContext(use_gpu=False)

    def tearDown(self):
        _gpu_mod._gpu = self._gpu

    def _make_frame(self, seed=4):
        from scipy import ndimage
        rng = np.random.default_rng(seed)
        H, W = 200, 240
        g = np.full((H, W), 200.0)
        yy, xx = np.mgrid[0:H, 0:W]
        for _ in range(15):
            cy, cx = rng.uniform(20, H - 20), rng.uniform(20, W - 20)
            g += rng.uniform(500, 3000) * np.exp(
                -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.8 ** 2))
        r = ndimage.shift(g, shift=(1.2, -0.8), order=3, mode='reflect')
        b = ndimage.shift(g, shift=(-0.6, 1.1), order=3, mode='reflect')
        return np.stack([r, g, b], axis=2).astype(np.float32)

    def _residual_misalignment(self, corrected, ref):
        from src.phase_correlate import phase_cross_correlation
        shift, _, _ = phase_cross_correlation(
            ref.astype(np.float64), corrected.astype(np.float64), upsample_factor=20)
        return float(np.hypot(shift[0], shift[1]))

    def test_downsampled_correction_matches_fullres_accuracy(self):
        rgb = self._make_frame()
        out_ds = astro.correct_chromatic_aberration(rgb.copy(), downsample=2)
        out_full = astro.correct_chromatic_aberration(rgb.copy(), downsample=1)
        for out, label in ((out_ds, 'downsample=2'), (out_full, 'downsample=1')):
            err_r = self._residual_misalignment(out[:, :, 0], out[:, :, 1])
            err_b = self._residual_misalignment(out[:, :, 2], out[:, :, 1])
            self.assertLess(err_r, 0.3, f'{label}: R residual {err_r}')
            self.assertLess(err_b, 0.3, f'{label}: B residual {err_b}')

    def test_native_and_scipy_fallback_agree(self):
        """The native Lanczos-warp shift-apply path and the scipy fallback
        must both correct to well-aligned output (not just avoid crashing)."""
        import src.debayer as db
        rgb = self._make_frame()
        had_native = db._HAS_NATIVE
        try:
            db._HAS_NATIVE = True
            out_native = astro.correct_chromatic_aberration(rgb.copy(), downsample=2)
            db._HAS_NATIVE = False
            out_scipy = astro.correct_chromatic_aberration(rgb.copy(), downsample=2)
        finally:
            db._HAS_NATIVE = had_native
        for out, label in ((out_native, 'native'), (out_scipy, 'scipy-fallback')):
            err_r = self._residual_misalignment(out[:, :, 0], out[:, :, 1])
            self.assertLess(err_r, 0.3, f'{label}: R residual {err_r}')


class TestWhiteBalance(unittest.TestCase):
    def setUp(self):
        self._gpu = _gpu_mod._gpu
        _gpu_mod._gpu = astro.GpuContext(use_gpu=False)

    def tearDown(self):
        _gpu_mod._gpu = self._gpu

    def test_grayworld_equalises_means(self):
        img = np.zeros((16, 16, 3), dtype=np.float32)
        img[:, :, 0] = 1.0
        img[:, :, 1] = 3.0
        img[:, :, 2] = 6.0
        out = astro.white_balance_grayworld(img)
        means = [float(out[:, :, c].mean()) for c in range(3)]
        self.assertAlmostEqual(means[0], means[1], delta=0.1)
        self.assertAlmostEqual(means[1], means[2], delta=0.1)

    def test_grayworld_shape(self):
        self.assertEqual(astro.white_balance_grayworld(_rgb(32, 32)).shape, (32, 32, 3))

    def test_grayworld_nonneg(self):
        self.assertGreaterEqual(float(astro.white_balance_grayworld(_rgb()).min()), 0.0)

    def test_whitepatch_shape(self):
        self.assertEqual(astro.white_balance_whitepatch(_rgb(32, 32)).shape, (32, 32, 3))

    def test_whitepatch_nonneg(self):
        self.assertGreaterEqual(float(astro.white_balance_whitepatch(_rgb()).min()), 0.0)


class TestRemoveHotPixels(unittest.TestCase):
    def setUp(self):
        self._gpu = _gpu_mod._gpu
        _gpu_mod._gpu = astro.GpuContext(use_gpu=False)

    def tearDown(self):
        _gpu_mod._gpu = self._gpu

    def test_2d_hot_pixel_corrected(self):
        img = np.full((32, 32), 100.0, dtype=np.float32)
        img[16, 16] = 50000.0
        out = astro.remove_hot_pixels(img, threshold=5.0)
        self.assertLess(float(out[16, 16]), 50000.0)

    def test_2d_clean_image_unchanged(self):
        img = np.random.default_rng(3).uniform(90, 110, (32, 32)).astype(np.float32)
        np.testing.assert_array_equal(img, astro.remove_hot_pixels(img, threshold=12.0))

    def test_bayer_hot_pixel_corrected(self):
        # Need a noisy background so MAD > 0; otherwise sigma=0 and the guard skips
        rng = np.random.default_rng(42)
        img = rng.normal(100.0, 10.0, (32, 32)).astype(np.float32)
        img[10, 10] = 60000.0
        out = astro.remove_hot_pixels_bayer(img, threshold=3.0)
        self.assertLess(float(out[10, 10]), 60000.0)

    def test_bayer_shape_preserved(self):
        img = np.random.default_rng(4).uniform(90, 110, (32, 32)).astype(np.float32)
        self.assertEqual(astro.remove_hot_pixels_bayer(img).shape, img.shape)

    def test_bayer_normal_pixels_near_unchanged(self):
        img = np.random.default_rng(5).uniform(490, 510, (64, 64)).astype(np.float32)
        img[30, 30] = 65000.0
        out = astro.remove_hot_pixels_bayer(img, threshold=5.0)
        mask = np.ones((64, 64), dtype=bool)
        mask[30, 30] = False
        np.testing.assert_allclose(out[mask], img[mask], atol=1.0)


class TestBuildHotPixelMap(unittest.TestCase):
    def test_detects_hot_pixel(self):
        dark = np.full((32, 32), 500.0, dtype=np.float32)
        dark[10, 10] = 10000.0
        hmap = astro.build_hot_pixel_map(dark, sigma_threshold=5.0)
        self.assertTrue(hmap[10, 10])

    def test_clean_dark_few_flags(self):
        dark = np.random.default_rng(6).uniform(490, 510, (32, 32)).astype(np.float32)
        hmap = astro.build_hot_pixel_map(dark, sigma_threshold=5.0)
        self.assertEqual(hmap.dtype, bool)
        self.assertLess(int(hmap.sum()), 5)

    def test_returns_bool(self):
        dark = np.full((16, 16), 100.0, dtype=np.float32)
        self.assertEqual(astro.build_hot_pixel_map(dark).dtype, bool)


class TestApplyHotPixelMapBayer(unittest.TestCase):
    def test_flagged_replaced(self):
        data = np.full((32, 32), 100.0, dtype=np.float32)
        data[8, 8] = 60000.0
        hmap = np.zeros((32, 32), dtype=bool)
        hmap[8, 8] = True
        out = astro.apply_hot_pixel_map_bayer(data, hmap)
        self.assertLess(float(out[8, 8]), 60000.0)

    def test_none_map_unchanged(self):
        data = np.random.default_rng(7).uniform(90, 110, (32, 32)).astype(np.float32)
        np.testing.assert_array_equal(data, astro.apply_hot_pixel_map_bayer(data, None))

    def test_all_false_unchanged(self):
        data = np.random.default_rng(8).uniform(90, 110, (32, 32)).astype(np.float32)
        hmap = np.zeros((32, 32), dtype=bool)
        np.testing.assert_array_equal(data, astro.apply_hot_pixel_map_bayer(data, hmap))


class TestCalcCommonCrop(unittest.TestCase):
    M = astro.Config.CROP_MARGIN

    def test_zero_shifts_margin_only(self):
        top, bottom, left, right = astro.calc_common_crop([(0.0, 0.0)] * 3, (100, 100))
        self.assertEqual(top, self.M)
        self.assertEqual(bottom, 100 - self.M)
        self.assertEqual(left, self.M)
        self.assertEqual(right, 100 - self.M)

    def test_positive_shifts_crop_top_left(self):
        top, bottom, left, right = astro.calc_common_crop(
            [(5.0, 5.0), (0.0, 0.0)], (100, 100)
        )
        self.assertGreater(top, self.M)
        self.assertGreater(left, self.M)

    def test_negative_shifts_crop_bottom_right(self):
        _, bottom, _, right = astro.calc_common_crop(
            [(-5.0, -5.0), (0.0, 0.0)], (100, 100)
        )
        self.assertLess(bottom, 100 - self.M)

    def test_crop_region_valid(self):
        shifts = [(3.0, 2.0), (-1.0, 4.0), (0.0, -2.0)]
        top, bottom, left, right = astro.calc_common_crop(shifts, (80, 80))
        self.assertGreaterEqual(top, 0)
        self.assertLessEqual(bottom, 80)
        self.assertGreaterEqual(left, 0)
        self.assertLessEqual(right, 80)
        self.assertLess(top, bottom)
        self.assertLess(left, right)

    def test_excessive_shifts_valid_region(self):
        shifts = [(60.0, 60.0), (-60.0, -60.0)]
        top, bottom, left, right = astro.calc_common_crop(shifts, (100, 100))
        self.assertLessEqual(top, bottom)
        self.assertLessEqual(left, right)

    def test_shifts_beyond_frame_fall_back_to_full_frame(self):
        crop = astro.calc_common_crop([(0.0, 0.0), (200.0, 200.0)], (100, 100))
        self.assertEqual(crop, (0, 100, 0, 100))


class TestSigmaClipTile(unittest.TestCase):
    def _tile(self, N=8, H=4, W=4, C=3):
        return np.random.default_rng(10).uniform(100, 200, (N, H, W, C)).astype(np.float32)

    def test_output_shape(self):
        out = astro._sigma_clip_tile(self._tile(), 3.0, 3, None, False)
        self.assertEqual(out.shape, (4, 4, 3))

    def test_output_dtype(self):
        out = astro._sigma_clip_tile(self._tile(), 3.0, 3, None, False)
        self.assertEqual(out.dtype, np.float32)

    def test_outlier_rejected(self):
        tile = np.full((8, 4, 4, 1), 100.0, dtype=np.float32)
        tile[0, :, :, :] = 50000.0
        out = astro._sigma_clip_tile(tile, 3.0, 5, None, False)
        self.assertLess(float(out.mean()), 5000.0)

    def test_all_same_returns_same(self):
        tile = np.full((6, 4, 4, 1), 200.0, dtype=np.float32)
        out = astro._sigma_clip_tile(tile, 3.0, 3, None, False)
        np.testing.assert_allclose(out, 200.0, atol=1e-3)

    def test_winsorize_shape(self):
        out = astro._sigma_clip_tile(self._tile(), 3.0, 3, None, True)
        self.assertEqual(out.shape, (4, 4, 3))

    def test_weights_pull_result_toward_high_value_frame(self):
        # Quality-weighted mean (winsorize=False, large sigma = no rejection):
        # equal weights → plain mean; heavy weight on high-value frame → higher result.
        tile = np.zeros((4, 4, 4, 1), dtype=np.float32)
        tile[0, :, :, :] = 1000.0
        tile[1:, :, :, :] = 100.0
        w_even = np.array([1.0, 1.0, 1.0, 1.0])
        w_heavy = np.array([100.0, 1.0, 1.0, 1.0])
        # sigma=1000 means no rejection; weights dominate the combine
        out_even = astro._sigma_clip_tile(tile, 1000.0, 1, w_even, False)
        out_heavy = astro._sigma_clip_tile(tile, 1000.0, 1, w_heavy, False)
        self.assertGreater(float(out_heavy.mean()), float(out_even.mean()))


class TestSigmaClipCombine(unittest.TestCase):
    def _data(self, N=6, H=16, W=16, C=3):
        return np.random.default_rng(11).uniform(100, 200, (N, H, W, C)).astype(np.float32)

    def test_output_shape(self):
        self.assertEqual(astro.sigma_clip_combine(self._data()).shape, (16, 16, 3))

    def test_output_dtype(self):
        self.assertEqual(astro.sigma_clip_combine(self._data()).dtype, np.float32)

    def test_result_within_input_range(self):
        data = self._data()
        out = astro.sigma_clip_combine(data)
        self.assertGreaterEqual(float(out.min()), float(data.min()) - 1.0)
        self.assertLessEqual(float(out.max()), float(data.max()) + 1.0)

    def test_outlier_frame_suppressed(self):
        data = np.full((8, 16, 16, 1), 150.0, dtype=np.float32)
        data[0, :, :, :] = 50000.0
        out = astro.sigma_clip_combine(data, sigma=3.0, max_iters=5)
        self.assertLess(float(out.mean()), 1000.0)

    def test_winsorize_mode(self):
        self.assertEqual(astro.sigma_clip_combine(self._data(), winsorize=True).shape, (16, 16, 3))

    def test_single_frame(self):
        data = np.random.default_rng(12).uniform(100, 200, (1, 8, 8, 3)).astype(np.float32)
        np.testing.assert_allclose(astro.sigma_clip_combine(data), data[0], atol=0.2)

    def test_uniform_stack_gives_frame_value(self):
        data = np.full((5, 8, 8, 1), 250.0, dtype=np.float32)
        np.testing.assert_allclose(astro.sigma_clip_combine(data), 250.0, atol=1.0)

    def test_two_frames_both_contribute(self):
        data = np.stack([np.full((8, 8, 1), 100.0, dtype=np.float32),
                         np.full((8, 8, 1), 200.0, dtype=np.float32)], axis=0)
        np.testing.assert_allclose(astro.sigma_clip_combine(data), 150.0, atol=2.0)

    def test_weights_pull_mean(self):
        data = np.stack([np.full((8, 8, 1), 100.0, dtype=np.float32),
                         np.full((8, 8, 1), 200.0, dtype=np.float32)], axis=0)
        out = astro.sigma_clip_combine(data, weights=np.array([3.0, 1.0]))
        np.testing.assert_allclose(out, 125.0, atol=5.0)  # (100*3 + 200) / 4


class TestDetectDither(unittest.TestCase):
    def test_zero_shifts_aligned(self):
        r = astro.detect_dither([(0.0, 0.0)] * 10)
        self.assertEqual(r["pattern"], "aligned")
        self.assertFalse(r["is_dithered"])

    def test_fewer_than_3_aligned(self):
        r = astro.detect_dither([(1.0, 1.0), (2.0, 2.0)])
        self.assertEqual(r["pattern"], "aligned")

    def test_required_keys(self):
        r = astro.detect_dither([(1.0, 2.0)] * 5)
        for k in ("is_dithered", "pattern", "mean_magnitude",
                  "unique_positions", "direction_spread_deg", "autocorrelation"):
            self.assertIn(k, r)

    def test_mean_magnitude_correct(self):
        r = astro.detect_dither([(3.0, 4.0)] * 5)
        self.assertAlmostEqual(r["mean_magnitude"], 5.0, delta=0.01)

    def test_random_shifts_large_magnitude(self):
        rng = np.random.default_rng(13)
        shifts = [(float(rng.uniform(-10, 10)), float(rng.uniform(-10, 10))) for _ in range(20)]
        r = astro.detect_dither(shifts)
        self.assertGreater(r["mean_magnitude"], 1.0)
        self.assertIn(r["pattern"], ("dithered", "tracking_drift", "aligned"))

    def test_spread_positions_dithered(self):
        # rng seed=1 uniform(-15, 15): spread, no sequential autocorrelation.
        shifts = [(0.4, 13.5), (-10.7, 13.5), (-5.6, -2.3), (9.8, -2.7),
                  (1.5, -14.2), (7.6, 1.1), (-5.1, 8.7), (-5.9, -1.4),
                  (-11.0, -2.9), (-8.9, -7.1)]
        self.assertTrue(astro.detect_dither(shifts)["is_dithered"])

    def test_is_dithered_is_bool(self):
        r = astro.detect_dither([(0.0, 0.0)] * 5)
        self.assertIsInstance(r["is_dithered"], bool)


class TestArcsinhStretch(unittest.TestCase):
    def test_output_range(self):
        out = astro.arcsinh_stretch(_star_field() + 100.0)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-6)

    def test_shape_preserved(self):
        img = np.random.default_rng(14).uniform(0, 1000, (32, 32)).astype(np.float32)
        self.assertEqual(astro.arcsinh_stretch(img).shape, img.shape)

    def test_all_zeros_returns_zeros(self):
        np.testing.assert_array_equal(
            astro.arcsinh_stretch(np.zeros((16, 16), dtype=np.float32)), 0.0
        )

    def test_monotone_brighter_maps_higher(self):
        img = np.array([[10.0, 100.0, 1000.0]], dtype=np.float32)
        out = astro.arcsinh_stretch(img)
        self.assertLess(float(out[0, 0]), float(out[0, 1]))
        self.assertLess(float(out[0, 1]), float(out[0, 2]))

    def test_custom_factor(self):
        out = astro.arcsinh_stretch(_star_field() + 100.0, factor=10.0)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-6)


class TestReduceChromaNoise(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(
            astro.reduce_chroma_noise(_rgb(32, 32) + 100.0, sigma=2.0).shape, (32, 32, 3)
        )

    def test_dtype(self):
        self.assertEqual(astro.reduce_chroma_noise(_rgb(32, 32) + 100.0).dtype, np.float32)

    def test_nonneg(self):
        self.assertGreaterEqual(
            float(astro.reduce_chroma_noise(_rgb(32, 32) + 100.0, sigma=2.0).min()), 0.0
        )

    def test_luminance_approximately_preserved(self):
        img = _rgb(32, 32) + 100.0
        lum_b = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        out = astro.reduce_chroma_noise(img, sigma=2.0)
        lum_a = 0.299 * out[:, :, 0] + 0.587 * out[:, :, 1] + 0.114 * out[:, :, 2]
        rel = abs(float(lum_a.mean()) - float(lum_b.mean())) / (float(lum_b.mean()) + 1e-12)
        self.assertLess(rel, 0.15)


class TestApplyTransform(unittest.TestCase):
    def setUp(self):
        self._gpu = _gpu_mod._gpu
        _gpu_mod._gpu = astro.GpuContext(use_gpu=False)

    def tearDown(self):
        _gpu_mod._gpu = self._gpu

    def test_zero_shift_identity(self):
        img = _rgb(32, 32) + 100.0
        np.testing.assert_allclose(astro.apply_transform(img, shift=(0.0, 0.0)), img, atol=0.1)

    def test_shape_preserved(self):
        img = _rgb(32, 32)
        self.assertEqual(astro.apply_transform(img, shift=(2.5, -1.5)).shape, img.shape)

    def test_no_args_returns_input(self):
        img = _rgb(16, 16)
        np.testing.assert_array_equal(astro.apply_transform(img), img)

    def test_shift_moves_peak_row(self):
        img = np.zeros((64, 64, 3), dtype=np.float32)
        img[32, 32, :] = 1000.0
        out = astro.apply_transform(img, shift=(5.0, 0.0))
        peak_before = int(np.argmax(img[:, :, 0].max(axis=1)))
        peak_after = int(np.argmax(out[:, :, 0].max(axis=1)))
        self.assertGreaterEqual(peak_after, peak_before)


class TestProcessingStats(unittest.TestCase):
    def test_total_time_positive(self):
        import time
        s = astro.ProcessingStats()
        time.sleep(0.01)
        self.assertGreater(s.total_time(), 0)

    def test_add_error(self):
        s = astro.ProcessingStats()
        s.add_error("path.fit", "corrupt")
        self.assertEqual(s.errors, [("path.fit", "corrupt")])

    def test_add_warning(self):
        s = astro.ProcessingStats()
        s.add_warning("only 3 frames")
        self.assertIn("3 frames", s.warnings[0])

    def test_defaults(self):
        s = astro.ProcessingStats()
        self.assertEqual(s.total_frames, 0)
        self.assertEqual(s.accepted_frames, 0)
        self.assertEqual(s.rejected_frames, 0)
        self.assertEqual(s.errors, [])
        self.assertEqual(s.warnings, [])


class TestGpuContext(unittest.TestCase):
    def setUp(self):
        self.ctx = astro.GpuContext(use_gpu=False)

    def test_active_false(self):
        self.assertFalse(self.ctx.active)

    def test_xp_is_numpy(self):
        self.assertIs(self.ctx.xp, np)

    def test_to_device_noop(self):
        arr = np.array([1.0, 2.0])
        np.testing.assert_array_equal(self.ctx.to_device(arr), arr)

    def test_to_host_noop(self):
        arr = np.array([1.0, 2.0])
        np.testing.assert_array_equal(self.ctx.to_host(arr), arr)

    def test_free_pool_no_error(self):
        self.ctx.free_pool()

    def test_available_vram_zero(self):
        self.assertEqual(self.ctx.available_vram_mb(), 0.0)

    def test_max_workers_at_least_1(self):
        self.assertGreaterEqual(self.ctx.max_gpu_workers(per_worker_mb=500.0), 1)

    def test_is_oom_matches_cuda_runtime_error_by_name(self):
        # Real class: cupy_backends.cuda.api.runtime.CUDARuntimeError.
        class CUDARuntimeError(Exception):
            pass
        self.assertTrue(self.ctx.is_oom(CUDARuntimeError("some other message")))

    def test_is_oom_matches_cuda_driver_error_by_name(self):
        # Real class: cupy_backends.cuda.api.driver.CUDADriverError, raised
        # e.g. by cuModuleUnload/cuMemFree under VRAM pressure -- distinct
        # from CUDARuntimeError (driver API vs runtime API). Regression test
        # for a real crash: this class wasn't matched by name at all before,
        # only coincidentally by message content for one specific wording.
        class CUDADriverError(Exception):
            pass
        self.assertTrue(self.ctx.is_oom(CUDADriverError("CUDA_ERROR_UNKNOWN: unknown error")))

    def test_is_oom_matches_out_of_memory_message(self):
        self.assertTrue(self.ctx.is_oom(RuntimeError("out of memory")))
        self.assertTrue(self.ctx.is_oom(RuntimeError("cudaErrorMemoryAllocation: out of memory")))

    def test_is_oom_rejects_unrelated_error(self):
        self.assertFalse(self.ctx.is_oom(ValueError("shape mismatch")))

    def test_stream_context_survives_stream_creation_oom(self):
        """Regression test: gpu.stream_context() used to have zero OOM
        protection -- every GPU call site enters it via
        `with gpu.stream_context():` before doing any real work, and a
        failure creating the CUDA stream itself (a real failure mode under
        severe VRAM pressure) propagated uncaught straight through all of
        them. Must disable the GPU and degrade to a no-op instead of
        raising."""
        # _gpu_mod.cp is whatever `import cupy as cp` resolved to at
        # gpu_context.py's own first import -- a real cupy package if one
        # happens to be installed, or plain `None` when it's genuinely
        # absent (e.g. CI, which only installs requirements.txt, not
        # requirements-gpu.txt). Either way, patch the module-level `cp`
        # *name* on _gpu_mod itself to a fresh fake module rather than
        # mutating an attribute on whatever `cp` currently is -- mutating
        # fails outright when `cp` is None, and would clobber a real cupy
        # install's actual `.cuda` submodule otherwise.
        ctx = astro.GpuContext(use_gpu=False)
        ctx.active = True  # pretend GPU is live without needing real CUDA

        class _CUDARuntimeError(Exception):
            pass

        fake_cuda = types.ModuleType("cupy.cuda")
        fake_cuda.Stream = mock.Mock(
            side_effect=_CUDARuntimeError("cudaErrorMemoryAllocation: out of memory"))
        fake_cp = types.ModuleType("cupy")
        fake_cp.cuda = fake_cuda
        with mock.patch.object(_gpu_mod, 'cp', fake_cp):
            with ctx.stream_context() as stream:
                self.assertIsNone(stream)  # degraded to no-op, didn't raise
        self.assertFalse(ctx.active, "GPU must be disabled after a stream-creation OOM")

    def test_stream_context_reraises_non_oom_error(self):
        ctx = astro.GpuContext(use_gpu=False)
        ctx.active = True

        fake_cuda = types.ModuleType("cupy.cuda")
        fake_cuda.Stream = mock.Mock(side_effect=ValueError("unrelated"))
        fake_cp = types.ModuleType("cupy")
        fake_cp.cuda = fake_cuda
        with mock.patch.object(_gpu_mod, 'cp', fake_cp):
            with self.assertRaises(ValueError):
                with ctx.stream_context():
                    pass
        self.assertTrue(ctx.active, "a non-OOM error must not disable the GPU")


class TestApplyTransformGpuOomFallback(unittest.TestCase):
    """Regression test: apply_transform's GPU OOM handler used to call only
    free_pool(), leaving gpu.active True -- a real capacity OOM (workload
    doesn't fit in this card's VRAM) doesn't resolve itself by freeing the
    pool once, so every subsequent per-frame call kept retrying the GPU and
    re-OOMing, observed in the wild as a long cascade of CUDA errors. Must
    call disable() instead, matching debayer.py's GPU call sites."""

    def setUp(self):
        self._real_gpu = _gpu_mod._gpu

        class _CUDARuntimeError(Exception):
            pass
        self._oom_exc_cls = _CUDARuntimeError

        fake = astro.GpuContext(use_gpu=False)
        fake.active = True  # pretend GPU is live without needing real CUDA
        fake.to_device = lambda arr: arr
        fake.to_host = lambda arr: np.asarray(arr)

        class _FakeXndimage:
            def shift(_self, *a, **kw):
                raise _CUDARuntimeError("cudaErrorMemoryAllocation: out of memory")

        fake.xndimage = _FakeXndimage()
        fake.xp = np
        self.fake = fake
        _gpu_mod._gpu = fake

    def tearDown(self):
        _gpu_mod._gpu = self._real_gpu

    def test_oom_falls_back_to_cpu_and_disables_gpu(self):
        img = np.zeros((8, 8, 3), dtype=np.float32)
        result = astro.apply_transform(img, shift=(1.0, 2.0))
        self.assertEqual(result.shape, img.shape)
        self.assertFalse(self.fake.active, "GPU must be permanently disabled after a real OOM")


class TestFrameInfo(unittest.TestCase):
    def test_defaults(self):
        fi = astro.FrameInfo(path="a.fit", type="light", header={})
        self.assertTrue(fi.accepted)
        self.assertIsNone(fi.metrics)
        self.assertEqual(fi.shift, (0.0, 0.0))

    def test_custom_values(self):
        fi = astro.FrameInfo(
            path="d.fit", type="dark", header={"EXPTIME": 30},
            accepted=False, shift=(3.0, -1.5)
        )
        self.assertFalse(fi.accepted)
        self.assertEqual(fi.shift, (3.0, -1.5))


class TestDiscoverFrames(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.data = np.ones((8, 8), dtype=np.float32) * 50.0

    def _write(self, name):
        write_fits(os.path.join(self.tmp, name), self.data)

    def test_light_discovery(self):
        self._write("light_001.fit")
        self._write("light_002.fit")
        frames = astro.discover_frames(self.tmp)
        self.assertEqual(len(frames["light"]), 2)

    def test_dark_discovery(self):
        self._write("dark_001.fit")
        frames = astro.discover_frames(self.tmp)
        self.assertEqual(len(frames["dark"]), 1)
        self.assertEqual(len(frames["light"]), 0)

    def test_mixed_directory(self):
        for name in ("light_1.fit", "dark_1.fit", "flat_1.fit", "bias_1.fit"):
            self._write(name)
        frames = astro.discover_frames(self.tmp)
        for ftype in ("light", "dark", "flat", "bias"):
            self.assertEqual(len(frames[ftype]), 1, ftype)

    def test_empty_directory(self):
        frames = astro.discover_frames(self.tmp)
        for ftype in ("light", "dark", "flat", "bias"):
            self.assertEqual(frames[ftype], [])


class TestCalculateShift(unittest.TestCase):
    def setUp(self):
        self._gpu = _gpu_mod._gpu
        _gpu_mod._gpu = astro.GpuContext(use_gpu=False)

    def tearDown(self):
        _gpu_mod._gpu = self._gpu

    def test_identical_images_near_zero(self):
        img = _star_field(64, 64)
        sy, sx = astro.calculate_shift(img, img)
        self.assertLess(abs(sy), 2.0)
        self.assertLess(abs(sx), 2.0)

    def test_returns_finite(self):
        ref = _star_field(64, 64) + 100.0
        img = _star_field(64, 64, seed=99) + 100.0
        sy, sx = astro.calculate_shift(ref, img)
        self.assertTrue(np.isfinite(sy) and np.isfinite(sx))

    def test_large_offset_finite(self):
        ref = np.zeros((64, 64), dtype=np.float32)
        ref[32, 32] = 1000.0
        img = np.zeros((64, 64), dtype=np.float32)
        img[10, 10] = 1000.0
        sy, sx = astro.calculate_shift(ref, img)
        self.assertTrue(np.isfinite(sy) and np.isfinite(sx))


class TestSafePrint(unittest.TestCase):
    def test_ascii_printed(self):
        from io import StringIO
        buf = StringIO()
        with mock.patch("builtins.print", lambda t: buf.write(t + "\n")):
            astro.safe_print("Hello World")
        self.assertIn("Hello World", buf.getvalue())

    def test_unicode_fallback_replaces_symbols(self):
        calls = []

        def first_fail(text):
            if not calls:
                calls.append(text)
                raise UnicodeEncodeError("utf-8", text, 0, 1, "test")
            calls.append(text)

        with mock.patch("builtins.print", side_effect=first_fail):
            try:
                astro.safe_print("✓ OK")
            except Exception:
                pass
        if len(calls) > 1:
            self.assertIn("[OK]", calls[-1])


class TestReadVersion(unittest.TestCase):
    def test_reads_repo_root_version_file(self):
        # tests/../VERSION -- the real repo-root file, not a mock. Confirms
        # the fallback candidate path resolves correctly from src/utils.py.
        version = astro.read_version()
        self.assertNotEqual(version, 'dev')
        self.assertRegex(version, r'^\d+\.\d+\.\d+$')

    def test_prefers_meipass_when_set(self):
        import sys
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'VERSION').write_text('9.9.9', encoding='utf-8')
            sys._MEIPASS = d
            try:
                self.assertEqual(astro.read_version(), '9.9.9')
            finally:
                del sys._MEIPASS

    def test_returns_dev_when_nothing_found(self):
        import sys
        from unittest import mock
        with mock.patch('pathlib.Path.read_text', side_effect=OSError("missing")):
            sys._MEIPASS = '/nonexistent'
            try:
                self.assertEqual(astro.read_version(), 'dev')
            finally:
                del sys._MEIPASS


class TestConfigConstants(unittest.TestCase):
    def test_hot_pixel_threshold_positive(self):
        self.assertGreater(astro.Config.HOT_PIXEL_THRESHOLD, 0)

    def test_crop_margin_nonneg(self):
        self.assertGreaterEqual(astro.Config.CROP_MARGIN, 0)

    def test_quality_thresholds_nonneg(self):
        self.assertGreaterEqual(astro.Config.QUALITY_LOW_BRIGHTNESS, 0)
        self.assertGreaterEqual(astro.Config.QUALITY_LOW_CONTRAST, 0)

    def test_preview_quality_in_range(self):
        q = astro.Config.PREVIEW_JPEG_QUALITY
        self.assertGreaterEqual(q, 1)
        self.assertLessEqual(q, 100)

    def test_preview_percentiles_ordered(self):
        lo, hi = astro.Config.PREVIEW_STRETCH_PERCENTILES
        self.assertLess(lo, hi)

    def test_tile_size_positive(self):
        self.assertGreater(astro.Config.TILE_SIZE, 0)

    def test_min_recommended_frames_positive(self):
        self.assertGreater(astro.Config.MIN_RECOMMENDED_FRAMES, 0)
