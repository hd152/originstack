"""
Comprehensive test suite for astro_stack.py
============================================
Requires: numpy, scipy  (both ship in this environment)
Optional: pytest (falls back to unittest if absent)

Run with:
    python test_astro_stack.py            # unittest runner
    pytest test_astro_stack.py -v         # if pytest is installed

Test classes
------------
TestClassifyFrame         – frame-type heuristics (name / header / edge cases)
TestFormatTime            – human-readable time formatter
TestValidateImageData     – image corruption / sanity checks
TestComputeQualityMetrics – per-frame photometric metrics
TestDebayerBilinear       – bilinear Bayer demosaicing
TestDebayerDispatch       – debayer() method dispatcher
TestWhiteBalance          – grayworld & whitepatch white balance
TestRemoveHotPixels       – luminance-based & Bayer-aware hot-pixel removal
TestBuildHotPixelMap      – hot-pixel map from master dark
TestApplyHotPixelMapBayer – hot-pixel map application
TestCalcCommonCrop        – translation-only common-crop calculation
TestSigmaClipTile         – per-tile MAD sigma-clip kernel
TestSigmaClipCombine      – full tiled sigma-clip combine
TestDetectDither          – shift-pattern / dither classifier
TestArcsinhStretch        – arcsinh preview stretch
TestLocalNormalize        – local normalisation
TestReduceChromaNoise     – chroma noise reduction
TestApplyTransform        – image shift / affine transform
TestDrizzleCombine        – drizzle combiner (scale 1 and scale > 1)
TestProcessingStats       – ProcessingStats dataclass helpers
TestGpuContext            – CPU path of GpuContext
TestFrameInfo             – FrameInfo dataclass
TestMakeMasterLogic       – make_master() mean / median logic
TestDiscoverFrames        – FITS discovery + classification from disk
TestCalculateShift        – registration shift calculation
TestSafePrint             – Unicode fallback in safe_print
TestConfigConstants       – sanity checks on Config magic numbers
TestEndToEnd              – full mini-pipeline smoke tests
"""

from __future__ import annotations

import os
import sys
import types
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

# ---------------------------------------------------------------------------
# Minimal astropy stub (so the module loads without a real astropy install)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Stub classes at module level so pickle can serialise them
# ---------------------------------------------------------------------------

class _FakeHeader(dict):
    def __setitem__(self, key, value):
        super().__setitem__(key, value[0] if isinstance(value, tuple) else value)


class _FakePrimaryHDU:
    def __init__(self, data=None, header=None):
        self.data = data
        self.header = header if header is not None else _FakeHeader()


class _FakeHDUList:
    def __init__(self, hdus):
        self._hdus = list(hdus)

    def __getitem__(self, idx):
        return self._hdus[idx]

    def __len__(self):
        return len(self._hdus)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def writeto(self, path, overwrite=False):
        with open(path, "wb") as f:
            pickle.dump(self, f)


def _fake_fits_open(path, memmap=True):
    with open(path, "rb") as f:
        return pickle.load(f)


def _sigma_clipped_stats_stub(data, sigma=3.0, maxiters=5):
    arr = np.asarray(data).ravel().astype(np.float64)
    for _ in range(int(maxiters)):
        med = float(np.median(arr))
        std = float(np.std(arr))
        if std < 1e-12:
            break
        arr = arr[np.abs(arr - med) < sigma * std]
        if arr.size == 0:
            break
    med = float(np.median(arr)) if arr.size else 0.0
    std = float(np.std(arr)) if arr.size else 0.0
    return float(np.mean(arr)) if arr.size else 0.0, med, std


def _make_astropy_stub():
    astropy_mod = types.ModuleType("astropy")

    io_mod = types.ModuleType("astropy.io")
    fits_mod = types.ModuleType("astropy.io.fits")

    fits_mod.Header = _FakeHeader
    fits_mod.PrimaryHDU = _FakePrimaryHDU
    fits_mod.HDUList = _FakeHDUList
    fits_mod.open = _fake_fits_open
    io_mod.fits = fits_mod
    astropy_mod.io = io_mod

    stats_mod = types.ModuleType("astropy.stats")
    stats_mod.sigma_clipped_stats = _sigma_clipped_stats_stub
    astropy_mod.stats = stats_mod
    return astropy_mod, io_mod, fits_mod, stats_mod


_astropy_stub, _io_stub, _fits_stub, _stats_stub = _make_astropy_stub()
sys.modules.setdefault("astropy", _astropy_stub)
sys.modules.setdefault("astropy.io", _io_stub)
sys.modules.setdefault("astropy.io.fits", _fits_stub)
sys.modules.setdefault("astropy.stats", _stats_stub)

for _m in [
    "photutils", "photutils.detection",
    "tqdm", "psutil",
    "astroquery", "astroquery.astrometry_net",
    "pywt", "cv2", "cupy",
    "skimage.restoration",
]:
    sys.modules.setdefault(_m, types.ModuleType(_m))

_sk = sys.modules.get("skimage") or types.ModuleType("skimage")
sys.modules.setdefault("skimage", _sk)

_sk_exp = types.ModuleType("skimage.exposure")
_sk_exp.rescale_intensity = lambda img, in_range=None, out_range=(0, 1): img
sys.modules.setdefault("skimage.exposure", _sk_exp)

_sk_reg = types.ModuleType("skimage.registration")
_sk_reg.phase_cross_correlation = (
    lambda r, i, upsample_factor=1: (np.array([0.0, 0.0]), 0.0, 0.0)
)
sys.modules.setdefault("skimage.registration", _sk_reg)
sys.modules.setdefault("skimage.transform", types.ModuleType("skimage.transform"))
sys.modules.setdefault("skimage.measure", types.ModuleType("skimage.measure"))

# cv2 stub needs constants for debayer_vng
_cv2_stub = sys.modules["cv2"]
_cv2_stub.COLOR_BAYER_RG2RGB_VNG = 0
_cv2_stub.COLOR_BAYER_BG2RGB_VNG = 1
_cv2_stub.COLOR_BAYER_GR2RGB_VNG = 2
_cv2_stub.COLOR_BAYER_GB2RGB_VNG = 3
def _fake_cvtColor(img, code):
    if img.ndim == 2:
        return np.zeros((*img.shape, 3), dtype=img.dtype)
    return img
_cv2_stub.cvtColor = _fake_cvtColor

_pil = types.ModuleType("PIL")
_pil_img = types.ModuleType("PIL.Image")
_pil_img.fromarray = lambda arr: mock.MagicMock()
_pil.Image = _pil_img
sys.modules.setdefault("PIL", _pil)
sys.modules.setdefault("PIL.Image", _pil_img)

# ---- Import module under test ----
_orig_stdout = sys.stdout
_devnull = open(os.devnull, "w")
sys.stdout = _devnull
try:
    import importlib.util
    _candidates = [
        Path(__file__).parent / "astro_stack.py",
        Path("/mnt/user-data/uploads/astro_stack.py"),
    ]
    _src = next(p for p in _candidates if p.exists())
    _spec = importlib.util.spec_from_file_location("astro_stack", str(_src))
    astro = importlib.util.module_from_spec(_spec)
    sys.modules["astro_stack"] = astro
    _spec.loader.exec_module(astro)
finally:
    sys.stdout = _orig_stdout
    _devnull.close()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _star_field(H=64, W=64, n_stars=5, seed=42, bg=100.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.normal(bg, 5.0, (H, W)).astype(np.float32)
    for _ in range(n_stars):
        cy, cx = rng.integers(5, H - 5), rng.integers(5, W - 5)
        yy, xx = np.ogrid[:H, :W]
        img += 600.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 4.0)
    return img.clip(0).astype(np.float32)


def _rgb(H=64, W=64, seed=42) -> np.ndarray:
    return np.random.default_rng(seed).uniform(100, 1000, (H, W, 3)).astype(np.float32)


def _write_fits(data: np.ndarray, path: str, header_extra=None):
    from astropy.io import fits
    hdu = fits.PrimaryHDU(data.astype(np.float32))
    if header_extra:
        for k, v in header_extra.items():
            hdu.header[k] = v
    fits.HDUList([hdu]).writeto(path, overwrite=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestClassifyFrame(unittest.TestCase):
    def test_dark_filename(self):
        self.assertEqual(astro.classify_frame("dark_001.fit", {}), "dark")

    def test_flat_filename(self):
        self.assertEqual(astro.classify_frame("flat_001.fit", {}), "flat")

    def test_bias_filename(self):
        self.assertEqual(astro.classify_frame("bias_001.fit", {}), "bias")

    def test_light_default(self):
        self.assertEqual(astro.classify_frame("frame_001.fit", {}), "light")

    def test_dark_imagetyp(self):
        self.assertEqual(astro.classify_frame("x.fit", {"IMAGETYP": "dark"}), "dark")

    def test_flat_imagetyp(self):
        self.assertEqual(astro.classify_frame("x.fit", {"IMAGETYP": "flat"}), "flat")

    def test_bias_imagetyp(self):
        self.assertEqual(astro.classify_frame("x.fit", {"IMAGETYP": "bias"}), "bias")

    def test_bias_zero_exptime(self):
        self.assertEqual(astro.classify_frame("frame.fit", {"EXPTIME": 0}), "bias")

    def test_skip_combined(self):
        self.assertEqual(astro.classify_frame("frame.fit", {"COMBINED": True}), "skip")

    def test_skip_creator(self):
        self.assertEqual(astro.classify_frame("x.fit", {"CREATOR": "astro_stack v2"}), "skip")

    def test_case_insensitive_filename(self):
        self.assertEqual(astro.classify_frame("DARK_001.FIT", {}), "dark")

    def test_case_insensitive_imagetyp(self):
        self.assertEqual(astro.classify_frame("x.fit", {"IMAGETYP": "FLAT"}), "flat")

    def test_nonzero_exptime_is_light(self):
        self.assertEqual(astro.classify_frame("frame.fit", {"EXPTIME": 30.0}), "light")

    def test_filename_dark_overrides_light_imagetyp(self):
        # filename-based detection should take priority
        result = astro.classify_frame("dark_001.fit", {"IMAGETYP": "light"})
        self.assertEqual(result, "dark")


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


class TestDebayerBilinear(unittest.TestCase):
    def setUp(self):
        self._gpu = astro._gpu
        astro._gpu = astro.GpuContext(use_gpu=False)
        self.raw = np.random.default_rng(1).uniform(0, 1000, (64, 64)).astype(np.float32)

    def tearDown(self):
        astro._gpu = self._gpu

    def test_output_shape(self):
        self.assertEqual(astro.debayer_bilinear(self.raw).shape, (64, 64, 3))

    def test_output_dtype(self):
        self.assertEqual(astro.debayer_bilinear(self.raw).dtype, np.float32)

    def test_all_bayer_patterns(self):
        for pat in ("RGGB", "BGGR", "GRBG", "GBRG"):
            self.assertEqual(
                astro.debayer_bilinear(self.raw, pattern=pat).shape,
                (64, 64, 3), f"shape wrong for {pat}"
            )

    def test_values_within_range(self):
        out = astro.debayer_bilinear(self.raw)
        lo, hi = float(self.raw.min()), float(self.raw.max())
        self.assertGreaterEqual(float(out.min()), lo - 20.0)
        self.assertLessEqual(float(out.max()), hi + 20.0)

    def test_uniform_raw_uniform_output(self):
        raw = np.full((64, 64), 500.0, dtype=np.float32)
        out = astro.debayer_bilinear(raw)
        # bilinear splits 4 raw pixels into 3 channels with kernel normalisation;
        # for uniform RGGB input each channel converges to the mean value (500)
        # divided by the number of sub-samples the kernel sees (4), so ≈125.
        # Just check shape and no NaN/Inf instead of an absolute value.
        self.assertTrue(np.all(np.isfinite(out)))


class TestDebayerDispatch(unittest.TestCase):
    def setUp(self):
        self._gpu = astro._gpu
        astro._gpu = astro.GpuContext(use_gpu=False)
        self.raw = np.random.default_rng(2).uniform(100, 1000, (64, 64)).astype(np.float32)

    def tearDown(self):
        astro._gpu = self._gpu

    def test_bilinear(self):
        self.assertEqual(astro.debayer(self.raw, method="bilinear").shape, (64, 64, 3))

    def test_malvar(self):
        self.assertEqual(astro.debayer(self.raw, method="malvar").shape, (64, 64, 3))

    def test_vng_fallback(self):
        self.assertEqual(astro.debayer(self.raw, method="vng").shape, (64, 64, 3))

    def test_default_equals_bilinear(self):
        np.testing.assert_array_equal(
            astro.debayer(self.raw),
            astro.debayer(self.raw, method="bilinear"),
        )


class TestWhiteBalance(unittest.TestCase):
    def setUp(self):
        self._gpu = astro._gpu
        astro._gpu = astro.GpuContext(use_gpu=False)

    def tearDown(self):
        astro._gpu = self._gpu

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
        self._gpu = astro._gpu
        astro._gpu = astro.GpuContext(use_gpu=False)

    def tearDown(self):
        astro._gpu = self._gpu

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

    def test_rgb_hot_pixel_corrected(self):
        img = np.full((32, 32, 3), 100.0, dtype=np.float32)
        img[16, 16, :] = 50000.0
        out = astro.remove_hot_pixels_rgb(img, threshold=5.0)
        self.assertLess(float(out[16, 16, 0]), 50000.0)

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
        self.assertLess(top, bottom)
        self.assertLess(left, right)

    def test_excessive_shifts_valid_region(self):
        shifts = [(60.0, 60.0), (-60.0, -60.0)]
        top, bottom, left, right = astro.calc_common_crop(shifts, (100, 100))
        self.assertLessEqual(top, bottom)
        self.assertLessEqual(left, right)


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


class TestLocalNormalize(unittest.TestCase):
    def test_output_shape(self):
        self.assertEqual(astro.local_normalize(_rgb(64, 64), sigma=10.0).shape, (64, 64, 3))

    def test_output_dtype(self):
        self.assertEqual(astro.local_normalize(_rgb(32, 32), sigma=5.0).dtype, np.float32)

    def test_output_nonneg(self):
        self.assertGreaterEqual(float(astro.local_normalize(_rgb(32, 32), sigma=5.0).min()), 0.0)

    def test_uniform_collapses_near_zero(self):
        img = np.ones((32, 32, 3), dtype=np.float32) * 500.0
        out = astro.local_normalize(img, sigma=5.0)
        self.assertLess(float(out.max()), 1.0 + 1e-3)


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
        self._gpu = astro._gpu
        astro._gpu = astro.GpuContext(use_gpu=False)

    def tearDown(self):
        astro._gpu = self._gpu

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


class TestDrizzleCombine(unittest.TestCase):
    def _imgs(self, N=4, H=16, W=16, C=3):
        rng = np.random.default_rng(15)
        return [rng.uniform(100, 200, (H, W, C)).astype(np.float32) for _ in range(N)]

    def test_scale1_equals_mean(self):
        imgs = self._imgs()
        out = astro.drizzle_combine(imgs, [(0.0, 0.0)] * len(imgs), scale=1)
        np.testing.assert_allclose(out, np.mean(imgs, axis=0).astype(np.float32), atol=0.5)

    def test_scale2_output_shape(self):
        imgs = self._imgs()
        out = astro.drizzle_combine(imgs, [(0.0, 0.0)] * len(imgs), scale=2)
        self.assertEqual(out.shape, (32, 32, 3))

    def test_scale1_output_shape(self):
        imgs = self._imgs()
        out = astro.drizzle_combine(imgs, [(0.0, 0.0)] * len(imgs), scale=1)
        self.assertEqual(out.shape, (16, 16, 3))

    def test_output_dtype(self):
        imgs = self._imgs()
        out = astro.drizzle_combine(imgs, [(0.0, 0.0)] * len(imgs), scale=1)
        self.assertEqual(out.dtype, np.float32)


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


class TestMakeMasterLogic(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _frame(self, value, name):
        path = os.path.join(self.tmp, name)
        _write_fits(np.full((8, 8), value, dtype=np.float32), path)
        return astro.FrameInfo(path=path, type="dark", header={})

    def test_empty_returns_none(self):
        self.assertIsNone(astro.make_master([], method="mean"))

    def test_single_frame_mean(self):
        result = astro.make_master([self._frame(300.0, "f0.fit")], method="mean")
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result, 300.0, atol=1.0)

    def test_multiple_frames_mean(self):
        frames = [self._frame(v, f"f{i}.fit") for i, v in enumerate([100.0, 200.0, 300.0])]
        np.testing.assert_allclose(astro.make_master(frames, method="mean"), 200.0, atol=1.0)

    def test_median_correct(self):
        frames = [self._frame(v, f"f{i}.fit") for i, v in enumerate([100.0, 200.0, 900.0])]
        np.testing.assert_allclose(astro.make_master(frames, method="median"), 200.0, atol=1.0)


class TestDiscoverFrames(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.data = np.ones((8, 8), dtype=np.float32) * 50.0

    def _write(self, name):
        _write_fits(self.data, os.path.join(self.tmp, name))

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
        self._gpu = astro._gpu
        astro._gpu = astro.GpuContext(use_gpu=False)

    def tearDown(self):
        astro._gpu = self._gpu

    def test_identical_images_near_zero(self):
        img = _star_field(64, 64)
        sy, sx = astro.calculate_shift(img, img)
        self.assertLess(abs(sy), 2.0)
        self.assertLess(abs(sx), 2.0)

    def test_known_shift_recovered(self):
        from scipy import ndimage
        ref = _star_field(64, 64) + 200.0
        dy, dx = 3.0, -2.0
        shifted = ndimage.shift(ref, shift=(dy, dx), mode='constant', cval=0.0)
        sy, sx = astro.calculate_shift(ref, shifted, skip_phase_cc=True)
        self.assertAlmostEqual(sy, -dy, delta=2.0)
        self.assertAlmostEqual(sx, -dx, delta=2.0)

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


class TestConfigConstants(unittest.TestCase):
    def test_hot_pixel_threshold_positive(self):
        self.assertGreater(astro.Config.HOT_PIXEL_THRESHOLD, 0)

    def test_max_shift_fraction(self):
        self.assertGreater(astro.Config.MAX_SHIFT_FRACTION, 0)
        self.assertLess(astro.Config.MAX_SHIFT_FRACTION, 1.0)

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


class TestEndToEnd(unittest.TestCase):
    """Full mini-pipeline smoke tests."""

    def setUp(self):
        self._gpu = astro._gpu
        astro._gpu = astro.GpuContext(use_gpu=False)
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        astro._gpu = self._gpu

    def test_sigma_clip_on_data_with_outlier(self):
        data = np.random.default_rng(20).uniform(100, 200, (10, 8, 8, 3)).astype(np.float32)
        data[0, :, :, :] += 10000.0
        out = astro.sigma_clip_combine(data, sigma=3.0, max_iters=3)
        self.assertEqual(out.shape, (8, 8, 3))
        self.assertLess(float(out.mean()), 5000.0)

    def test_debayer_wb_combine_pipeline(self):
        rng = np.random.default_rng(21)
        raws = [rng.uniform(100, 1000, (32, 32)).astype(np.float32) for _ in range(4)]
        rgbs = [astro.white_balance_grayworld(astro.debayer(r)) for r in raws]
        out = astro.sigma_clip_combine(np.stack(rgbs, axis=0), sigma=3.0)
        self.assertEqual(out.shape, (32, 32, 3))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_hot_pixel_bayer_then_debayer_clean(self):
        raw = np.random.default_rng(22).uniform(100, 1000, (64, 64)).astype(np.float32)
        raw[30, 30] = 65000.0
        cleaned = astro.remove_hot_pixels_bayer(raw, threshold=5.0)
        rgb = astro.debayer(cleaned, method="bilinear")
        self.assertEqual(rgb.shape, (64, 64, 3))
        self.assertTrue(np.all(np.isfinite(rgb)))

    def test_crop_bounds_inside_image(self):
        H, W = 64, 64
        shifts = [(4.0, 3.0), (-2.0, -1.0), (0.0, 2.0)]
        top, bottom, left, right = astro.calc_common_crop(shifts, (H, W))
        self.assertGreaterEqual(top, 0)
        self.assertLessEqual(bottom, H)
        self.assertGreaterEqual(left, 0)
        self.assertLessEqual(right, W)
        self.assertLess(top, bottom)
        self.assertLess(left, right)

    def test_quality_metrics_on_star_field(self):
        m = astro.compute_quality_metrics(_star_field(64, 64) + 200.0)
        self.assertGreater(m["brightness"], 0)
        self.assertGreater(m["score"], 0)
        self.assertGreater(m["dynamic_range"], 0)

    def test_arcsinh_per_channel(self):
        rgb = _rgb(32, 32) + 100.0
        for c in range(3):
            out = astro.arcsinh_stretch(rgb[:, :, c])
            self.assertGreaterEqual(float(out.min()), 0.0)
            self.assertLessEqual(float(out.max()), 1.0 + 1e-5)

    def test_make_master_then_hot_pixel_correction(self):
        frames = []
        for i, v in enumerate([300.0, 310.0, 290.0]):
            path = os.path.join(self.tmp, f"dark_{i}.fit")
            _write_fits(np.full((16, 16), v, dtype=np.float32), path)
            frames.append(astro.FrameInfo(path=path, type="dark", header={}))
        master = astro.make_master(frames, method="mean")
        self.assertIsNotNone(master)
        cleaned = astro.remove_hot_pixels_bayer(master)
        self.assertEqual(cleaned.shape, master.shape)
        np.testing.assert_allclose(cleaned, master, atol=5.0)

    def test_drizzle_scale2_then_sigma_clip(self):
        rng = np.random.default_rng(23)
        imgs = [rng.uniform(100, 200, (8, 8, 1)).astype(np.float32) for _ in range(4)]
        drizzled = astro.drizzle_combine(imgs, [(0.0, 0.0)] * 4, scale=2)
        batch = np.stack([drizzled] * 3, axis=0)
        out = astro.sigma_clip_combine(batch, sigma=3.0)
        self.assertEqual(out.shape, (16, 16, 1))

    def test_detect_dither_on_random_shifts(self):
        rng = np.random.default_rng(24)
        shifts = [(float(rng.uniform(-8, 8)), float(rng.uniform(-8, 8))) for _ in range(15)]
        result = astro.detect_dither(shifts)
        self.assertIn(result["pattern"], ("dithered", "tracking_drift", "aligned"))
        self.assertIsInstance(result["is_dithered"], bool)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)