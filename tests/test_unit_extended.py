"""Extended unit tests covering frame_discovery, io_fits, debayer,
registration helpers, and stacking algorithms."""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
from astropy.io import fits

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_fits(path: str, data: np.ndarray, header: dict | None = None) -> None:
    hdr = fits.Header()
    if header:
        for k, v in header.items():
            hdr[k] = v
    fits.writeto(path, data.astype(np.float32), header=hdr, overwrite=True)


def _make_bayer(shape=(32, 32), pattern='RGGB', r=1000.0, g=800.0, b=600.0) -> np.ndarray:
    """Synthetic Bayer frame with distinct channel values."""
    raw = np.zeros(shape, dtype=np.float32)
    offsets = {'RGGB': {'R': (0, 0), 'G1': (0, 1), 'G2': (1, 0), 'B': (1, 1)},
               'BGGR': {'R': (1, 1), 'G1': (0, 1), 'G2': (1, 0), 'B': (0, 0)},
               'GRBG': {'R': (0, 1), 'G1': (0, 0), 'G2': (1, 1), 'B': (1, 0)},
               'GBRG': {'R': (1, 0), 'G1': (0, 0), 'G2': (1, 1), 'B': (0, 1)}}
    o = offsets[pattern]
    raw[o['R'][0]::2, o['R'][1]::2] = r
    raw[o['G1'][0]::2, o['G1'][1]::2] = g
    raw[o['G2'][0]::2, o['G2'][1]::2] = g
    raw[o['B'][0]::2, o['B'][1]::2] = b
    return raw


# ===========================================================================
# 1. classify_frame (src/frame_discovery.py)
# ===========================================================================


class TestLoadFits(unittest.TestCase):

    def setUp(self):
        from src.io_fits import load_fits
        self.load_fits = load_fits

    def test_round_trip_float32(self):
        """Data written as float32 should be read back identically."""
        data = np.random.default_rng(0).uniform(100, 5000, (64, 64)).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
            path = f.name
        try:
            _write_fits(path, data)
            loaded, _ = self.load_fits(path)
            np.testing.assert_allclose(loaded, data, rtol=1e-5)
        finally:
            os.unlink(path)

    def test_header_preserved(self):
        data = np.ones((16, 16), dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
            path = f.name
        try:
            _write_fits(path, data, {'EXPTIME': 120.0, 'BAYERPAT': 'RGGB'})
            _, hdr = self.load_fits(path)
            self.assertAlmostEqual(float(hdr['EXPTIME']), 120.0)
            self.assertEqual(hdr.get('BAYERPAT'), 'RGGB')
        finally:
            os.unlink(path)

    def test_output_is_float32(self):
        data = np.ones((8, 8), dtype=np.uint16) * 3000
        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
            path = f.name
        try:
            fits.writeto(path, data, overwrite=True)
            loaded, _ = self.load_fits(path)
            self.assertEqual(loaded.dtype, np.float32)
        finally:
            os.unlink(path)


class TestMakeMaster(unittest.TestCase):

    def setUp(self):
        from src.io_fits import make_master
        from src.models import FrameInfo
        self.make_master = make_master
        self.FrameInfo = FrameInfo

    def _write_frame(self, tmpdir: str, name: str, data: np.ndarray):
        path = os.path.join(tmpdir, name)
        _write_fits(path, data)
        return self.FrameInfo(path=path, type='dark', header={})

    def test_returns_none_for_empty_list(self):
        self.assertIsNone(self.make_master([]))

    def test_median_of_three_frames(self):
        """Median master should equal the middle value at each pixel."""
        with tempfile.TemporaryDirectory() as d:
            f1 = self._write_frame(d, 'a.fits', np.full((16, 16), 100.0, dtype=np.float32))
            f2 = self._write_frame(d, 'b.fits', np.full((16, 16), 200.0, dtype=np.float32))
            f3 = self._write_frame(d, 'c.fits', np.full((16, 16), 300.0, dtype=np.float32))
            master = self.make_master([f1, f2, f3], method='median')
            self.assertIsNotNone(master)
            np.testing.assert_allclose(master, 200.0, atol=1.0)

    def test_mean_of_two_frames(self):
        """Mean master should average the two frames exactly."""
        with tempfile.TemporaryDirectory() as d:
            f1 = self._write_frame(d, 'a.fits', np.full((16, 16), 100.0, dtype=np.float32))
            f2 = self._write_frame(d, 'b.fits', np.full((16, 16), 300.0, dtype=np.float32))
            master = self.make_master([f1, f2], method='mean')
            self.assertIsNotNone(master)
            np.testing.assert_allclose(master, 200.0, atol=1.0)

    def test_single_frame_returned_as_is(self):
        with tempfile.TemporaryDirectory() as d:
            data = np.random.default_rng(1).uniform(50, 200, (16, 16)).astype(np.float32)
            f = self._write_frame(d, 'a.fits', data)
            master = self.make_master([f], method='median')
            self.assertIsNotNone(master)
            np.testing.assert_allclose(master, data, atol=1.0)

    def test_outlier_rejected_by_median(self):
        """With 5 frames where one is a huge outlier, median should be unaffected."""
        with tempfile.TemporaryDirectory() as d:
            frames = []
            for i in range(4):
                frames.append(self._write_frame(d, f'{i}.fits',
                                                np.full((16, 16), 100.0, dtype=np.float32)))
            frames.append(self._write_frame(d, 'hot.fits',
                                            np.full((16, 16), 50000.0, dtype=np.float32)))
            master = self.make_master(frames, method='median')
            self.assertIsNotNone(master)
            self.assertLess(float(np.max(master)), 200.0)


# ===========================================================================
# 3. Debayering and calibration helpers (src/debayer.py)
# ===========================================================================

class TestDebayerPatterns(unittest.TestCase):

    def setUp(self):
        from src.debayer import debayer_bilinear
        self.debayer = debayer_bilinear

    def _channel_means(self, rgb: np.ndarray):
        return float(rgb[:, :, 0].mean()), float(rgb[:, :, 1].mean()), float(rgb[:, :, 2].mean())

    def test_rggb_channel_ordering(self):
        """RGGB: R>G>B when raw R pixels >> G >> B."""
        raw = _make_bayer((32, 32), 'RGGB', r=3000, g=1500, b=500)
        rgb = self.debayer(raw, pattern='RGGB')
        r, g, b = self._channel_means(rgb)
        self.assertGreater(r, g)
        self.assertGreater(g, b)

    def test_bggr_channel_ordering(self):
        """BGGR: swapped R/B relative to RGGB."""
        raw = _make_bayer((32, 32), 'BGGR', r=3000, g=1500, b=500)
        rgb = self.debayer(raw, pattern='BGGR')
        r, g, b = self._channel_means(rgb)
        # R (from red sub-channel) should still be reconstructed as red
        self.assertGreater(r, b)

    def test_output_shape_is_h_w_3(self):
        raw = np.zeros((64, 64), dtype=np.float32)
        rgb = self.debayer(raw, pattern='RGGB')
        self.assertEqual(rgb.shape, (64, 64, 3))

    def test_uniform_bayer_gives_uniform_rgb(self):
        """All-constant raw → all channels equal after debayer."""
        raw = np.full((32, 32), 1000.0, dtype=np.float32)
        rgb = self.debayer(raw, pattern='RGGB')
        np.testing.assert_allclose(rgb[:, :, 0], rgb[:, :, 1], atol=50.0)
        np.testing.assert_allclose(rgb[:, :, 1], rgb[:, :, 2], atol=50.0)

    def test_odd_sized_input_handled(self):
        """Pipeline crops, but debayer should not crash on odd sizes."""
        raw = np.zeros((33, 33), dtype=np.float32)
        try:
            rgb = self.debayer(raw, pattern='RGGB')
            self.assertEqual(rgb.ndim, 3)
        except Exception as e:
            self.fail(f"debayer_bilinear crashed on odd-sized input: {e}")


class TestWhiteBalanceEdgeCases(unittest.TestCase):

    def setUp(self):
        from src.debayer import white_balance_grayworld, white_balance_whitepatch
        self.grayworld = white_balance_grayworld
        self.whitepatch = white_balance_whitepatch

    def _unbalanced_rgb(self, r=3000.0, g=1000.0, b=500.0, shape=(32, 32)) -> np.ndarray:
        rgb = np.zeros((*shape, 3), dtype=np.float32)
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        return rgb

    def test_grayworld_dtype_float32(self):
        rgb = self._unbalanced_rgb()
        result = self.grayworld(rgb)
        self.assertEqual(result.dtype, np.float32)

    def test_grayworld_saturated_star_stays_white(self):
        """A clipped (equal-valued) star core must not be scaled into a colour cast.

        Reproduces a real-world bug: a background with a non-neutral raw
        colour cast (needing per-channel gains > 1x apart) plus a small
        fully-saturated (equal R=G=B) star region. Grayworld's gain is fit
        to the whole-frame mean (background-dominated), and without
        highlight-aware handling that gain gets applied uniformly to the
        already-clipped star too, scaling its equal channels apart into a
        magenta/purple halo even though the background comes out neutral.
        """
        shape = (256, 256)
        rgb = np.zeros((*shape, 3), dtype=np.float32)
        rgb[:, :, 0] = 3000.0
        rgb[:, :, 1] = 1000.0
        rgb[:, :, 2] = 1500.0
        rgb[124:128, 124:128, :] = 50000.0  # fully clipped "star": equal in all channels,
        # a realistic (small) area fraction of the frame -- same as a real bright star
        result = self.grayworld(rgb)
        star = result[124:128, 124:128, :]
        r_mean, g_mean, b_mean = star[:, :, 0].mean(), star[:, :, 1].mean(), star[:, :, 2].mean()
        self.assertLess(abs(r_mean - g_mean) / g_mean, 0.15)
        self.assertLess(abs(b_mean - g_mean) / g_mean, 0.15)
        # Background must still be properly white-balanced (the fix must not
        # regress normal, unsaturated colour-cast correction).
        bg = result[0:8, 0:8, :]
        bg_r, bg_g, bg_b = bg[:, :, 0].mean(), bg[:, :, 1].mean(), bg[:, :, 2].mean()
        self.assertLess(abs(bg_r - bg_g) / bg_g, 0.10)
        self.assertLess(abs(bg_b - bg_g) / bg_g, 0.10)

    def test_whitepatch_max_is_close_to_reference(self):
        """White patch should scale so that channel maximums are similar."""
        rgb = self._unbalanced_rgb()
        result = self.whitepatch(rgb)
        maxes = [result[:, :, c].max() for c in range(3)]
        # After white-patch WB the maxes should be closer together
        before_spread = max(3000.0, 1000.0, 500.0) - min(3000.0, 1000.0, 500.0)
        after_spread = max(maxes) - min(maxes)
        self.assertLess(after_spread, before_spread)

    def test_grayworld_zero_channel_no_crash(self):
        """All-zero green channel should not cause division by zero."""
        rgb = np.zeros((16, 16, 3), dtype=np.float32)
        rgb[:, :, 0] = 1000.0
        try:
            result = self.grayworld(rgb)
            self.assertEqual(result.shape, rgb.shape)
        except Exception as e:
            self.fail(f"grayworld crashed on zero channel: {e}")


class TestHotPixels(unittest.TestCase):

    def setUp(self):
        from src.debayer import build_hot_pixel_map, fix_hot_pixels
        self.build_map = build_hot_pixel_map
        self.fix = fix_hot_pixels

    def test_build_map_flags_bright_pixels(self):
        """Hot pixels well above the noise floor should be flagged."""
        dark = np.random.default_rng(0).normal(50, 2, (64, 64)).astype(np.float32)
        dark[10, 10] = 5000.0  # hot pixel
        hot_map = self.build_map(dark, sigma_threshold=5.0)
        self.assertTrue(bool(hot_map[10, 10]))

    def test_build_map_does_not_flag_normal_pixels(self):
        """Normal pixels should not be flagged."""
        dark = np.random.default_rng(1).normal(50, 2, (64, 64)).astype(np.float32)
        hot_map = self.build_map(dark, sigma_threshold=8.0)
        # At sigma=8 effectively nothing normal should be flagged
        self.assertLess(int(hot_map.sum()), 5)

    def test_fix_hot_pixels_reduces_spike(self):
        """A single hot pixel should be significantly reduced after fix."""
        img = np.full((32, 32, 3), 200.0, dtype=np.float32)
        img[16, 16, 0] = 60000.0
        result = self.fix(img, mode='rgb')
        self.assertLess(float(result[16, 16, 0]), 60000.0 * 0.5)

    def test_fix_hot_pixels_preserves_clean_pixels(self):
        """Clean pixels should remain close to their original value."""
        img = np.full((32, 32, 3), 500.0, dtype=np.float32)
        img[5, 5, :] = 40000.0  # isolated spike
        result = self.fix(img, mode='rgb')
        # Interior of clean region should be essentially unchanged
        region = result[10:20, 10:20, :]
        np.testing.assert_allclose(region, 500.0, atol=5.0)

    def test_fix_preserves_shape_and_dtype(self):
        img = np.random.default_rng(2).uniform(100, 300, (32, 32, 3)).astype(np.float32)
        result = self.fix(img, mode='rgb')
        self.assertEqual(result.shape, img.shape)
        self.assertEqual(result.dtype, np.float32)


# ===========================================================================
# 4. Registration helpers (src/registration.py)
# ===========================================================================

class TestApplyShift(unittest.TestCase):

    def setUp(self):
        from src.registration import apply_shift
        self.apply_shift = apply_shift

    def _star_lum(self, shape=(64, 64), cy=32, cx=32, amp=1000.0) -> np.ndarray:
        yy, xx = np.indices(shape)
        return (amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 8.0)).astype(np.float32)

    def test_shifted_image_peak_moves_correctly(self):
        """Shifting by (dy, dx) should move the star centroid by that amount."""
        img = self._star_lum((64, 64), cy=32, cx=32)
        dy, dx = 5.0, -4.0
        shifted = self.apply_shift(img, (dy, dx))
        peak_y, peak_x = np.unravel_index(np.argmax(shifted), shifted.shape)
        self.assertAlmostEqual(float(peak_y), 32 + dy, delta=1.5)
        self.assertAlmostEqual(float(peak_x), 32 + dx, delta=1.5)

    def test_zero_shift_returns_same(self):
        img = self._star_lum()
        result = self.apply_shift(img, (0.0, 0.0))
        np.testing.assert_allclose(result, img, atol=0.1)

    def test_output_shape_preserved(self):
        img = np.random.rand(48, 56).astype(np.float32)
        result = self.apply_shift(img, (3.0, -2.0))
        self.assertEqual(result.shape, img.shape)


class TestCalcCommonCropFallback(unittest.TestCase):

    def setUp(self):
        from src.registration import calc_common_crop
        self.calc_crop = calc_common_crop

    def test_large_shifts_fallback_to_full_frame(self):
        """When shifts exceed the image, fallback to (0, H, 0, W)."""
        shifts = [(0.0, 0.0), (200.0, 200.0)]  # shift > image dims
        top, bot, left, right = self.calc_crop(shifts, (100, 100))
        # Fallback: full frame
        self.assertEqual((top, bot, left, right), (0, 100, 0, 100))


class TestDetectDitherSpread(unittest.TestCase):

    def setUp(self):
        from src.registration import detect_dither
        self.detect_dither = detect_dither

    def test_spread_positions_dithered(self):
        """10 spread positions with no sequential autocorrelation should be detected as dithered."""
        # Generated with rng seed=1 uniform(-15,15) — verified to pass all detection criteria
        shifts = [(0.4, 13.5), (-10.7, 13.5), (-5.6, -2.3), (9.8, -2.7),
                  (1.5, -14.2), (7.6, 1.1), (-5.1, 8.7), (-5.9, -1.4),
                  (-11.0, -2.9), (-8.9, -7.1)]
        result = self.detect_dither(shifts)
        self.assertTrue(result['is_dithered'])

class TestSigmaClipCombineNumerics(unittest.TestCase):

    def setUp(self):
        from src.stacking import sigma_clip_combine
        self.combine = sigma_clip_combine

    def _stack(self, *frames):
        """Stack a list of 2D arrays into a (N, H, W, 1) data cube."""
        return np.stack([f[:, :, np.newaxis] for f in frames], axis=0)

    def test_uniform_stack_gives_correct_mean(self):
        """All identical frames should return the frame value."""
        data = np.full((5, 8, 8, 1), 250.0, dtype=np.float32)
        result = self.combine(data)
        np.testing.assert_allclose(result, 250.0, atol=1.0)

    def test_two_frames_no_rejection(self):
        """Two frames that differ should both contribute (not enough to sigma-clip)."""
        a = np.full((8, 8, 1), 100.0, dtype=np.float32)
        b = np.full((8, 8, 1), 200.0, dtype=np.float32)
        data = np.stack([a, b], axis=0)
        result = self.combine(data)
        np.testing.assert_allclose(result, 150.0, atol=2.0)

    def test_weighted_combine(self):
        """Higher-weight frames should pull the mean closer to their values."""
        a = np.full((8, 8, 1), 100.0, dtype=np.float32)
        b = np.full((8, 8, 1), 200.0, dtype=np.float32)
        data = np.stack([a, b], axis=0)
        weights = np.array([3.0, 1.0])
        result = self.combine(data, weights=weights)
        # Weighted mean = (100*3 + 200*1) / 4 = 125
        np.testing.assert_allclose(result, 125.0, atol=5.0)

class TestPercentileClipCombine(unittest.TestCase):

    def setUp(self):
        from src.stacking import percentile_clip_combine
        self.combine = percentile_clip_combine

    def test_output_shape(self):
        data = np.random.default_rng(3).uniform(80, 120, (6, 16, 16, 3)).astype(np.float32)
        result = self.combine(data, low=20, high=80)
        self.assertEqual(result.shape, (16, 16, 3))

    def test_rejects_extremes(self):
        """Percentile clip should exclude the top and bottom frames."""
        rng = np.random.default_rng(4)
        data = rng.normal(100, 1, (10, 8, 8, 1)).astype(np.float32)
        data[0, :, :, :] = 5000.0   # top outlier
        data[-1, :, :, :] = -500.0  # bottom outlier
        result = self.combine(data, low=10, high=90)
        np.testing.assert_allclose(result, 100.0, atol=10.0)

    def test_uniform_stack(self):
        data = np.full((5, 8, 8, 1), 300.0, dtype=np.float32)
        result = self.combine(data)
        np.testing.assert_allclose(result, 300.0, atol=1.0)


class TestESDCombine(unittest.TestCase):

    def setUp(self):
        from src.stacking import esd_combine
        self.combine = esd_combine

    def test_output_shape(self):
        data = np.random.default_rng(5).uniform(80, 120, (5, 8, 8, 3)).astype(np.float32)
        result = self.combine(data)
        self.assertEqual(result.shape, (8, 8, 3))

    def test_rejects_single_outlier(self):
        """ESD combine should be robust against a single very bright pixel."""
        rng = np.random.default_rng(6)
        data = rng.normal(100, 1, (8, 8, 8, 1)).astype(np.float32)
        data[0, 4, 4, 0] = 8000.0
        result = self.combine(data, max_outliers=2)
        self.assertLess(float(result[4, 4, 0]), 300.0)

    def test_two_frame_minimum(self):
        """ESD should handle as few as 2 frames without crashing."""
        data = np.full((2, 4, 4, 1), 150.0, dtype=np.float32)
        result = self.combine(data)
        self.assertEqual(result.shape, (4, 4, 1))


# ===========================================================================
# 6. Registration shift calculation (src/registration.py)
# ===========================================================================

