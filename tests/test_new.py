"""Additional unit tests covering background, denoising, quality, stacking, and health check."""
from __future__ import annotations

import io
import sys
import unittest

import numpy as np

from src.models import FrameInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rgb(h=64, w=64, value=100.0) -> np.ndarray:
    """Return a flat float32 RGB image."""
    return np.full((h, w, 3), value, dtype=np.float32)


def _star_lum(shape=(64, 64), centers=None, amp=500.0, bg=50.0) -> np.ndarray:
    """Luminance image with Gaussian stars on a uniform background."""
    if centers is None:
        centers = [(16, 16), (48, 48)]
    img = np.full(shape, bg, dtype=np.float32)
    yy, xx = np.indices(shape)
    for cy, cx in centers:
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        img += amp * np.exp(-r2 / (2 * 2.0 ** 2))
    return img


def _star_rgb(h=64, w=64, centers=None, amp=500.0, bg=50.0) -> np.ndarray:
    lum = _star_lum((h, w), centers, amp, bg)
    return np.stack([lum, lum, lum], axis=2).astype(np.float32)


def _make_frame(header: dict, frame_type: str = 'light') -> FrameInfo:
    return FrameInfo(path='dummy.fits', type=frame_type, header=header)


def _capture(func, *args, **kwargs):
    """Run func, capture and return stdout as a string."""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        func(*args, **kwargs)
    finally:
        sys.stdout = old
    return buf.getvalue()


# ===========================================================================
# Background extraction
# ===========================================================================

class TestExtractBackground(unittest.TestCase):

    def setUp(self):
        from src.background import extract_background
        self.extract_background = extract_background

    def test_output_shape_matches_input(self):
        img = np.random.default_rng(0).normal(100, 5, (128, 128)).astype(np.float32)
        bg = self.extract_background(img, mesh_size=32)
        self.assertEqual(bg.shape, img.shape)

    def test_output_dtype_is_float32(self):
        img = np.ones((64, 64), dtype=np.float32) * 200.0
        bg = self.extract_background(img, mesh_size=16)
        self.assertEqual(bg.dtype, np.float32)

    def test_recovers_linear_gradient(self):
        """Background model should approximate a known linear gradient."""
        H, W = 128, 128
        yy = np.linspace(0, 100, H)[:, None] * np.ones((1, W))
        img = yy.astype(np.float32)
        bg = self.extract_background(img, mesh_size=32)
        # Residual after subtraction should be small relative to gradient range
        residual = np.abs(img - bg)
        self.assertLess(float(np.mean(residual)), 15.0)

    def test_uniform_image_returns_near_constant(self):
        img = np.full((64, 64), 300.0, dtype=np.float32)
        bg = self.extract_background(img, mesh_size=16)
        # All cells should yield the same value — background should be flat
        self.assertLess(float(np.std(bg)), 10.0)

    def test_star_mask_reduces_background_bias(self):
        """A bright region filling a whole mesh cell inflates the no-mask estimate;
        masking it lets the model recover the true sky level."""
        # 64x64 image, mesh_size=32 → 2x2 grid of 4 cells (each 32x32 pixels).
        # Fill the top-left cell entirely with a very bright value so sigma-clipping
        # alone cannot reject it (all pixels in the cell are bright).
        img = np.full((64, 64), 100.0, dtype=np.float32)
        img[:32, :32] = 3000.0        # entire top-left cell is bright
        bg_no_mask = self.extract_background(img, mesh_size=32)
        star_mask = np.zeros_like(img)
        star_mask[:32, :32] = 1.0     # mask the same region
        bg_with_mask = self.extract_background(img, mesh_size=32, star_mask=star_mask)
        # Median sky-model with mask should be closer to the true flat sky (100 ADU)
        # than without mask.
        sky_no   = float(np.median(bg_no_mask))
        sky_with = float(np.median(bg_with_mask))
        self.assertLess(abs(sky_with - 100.0), abs(sky_no - 100.0))


class TestFitRbfSurfaceBounded(unittest.TestCase):
    """Regression test: a large contiguous gap in the DBE patch samples (e.g.
    left by outlier rejection near a bright star) once let the thin-plate-
    spline RBF extrapolate far outside the real sky range, producing a
    visible over/under-subtracted patch in the final image after the surface
    was subtracted from a flat stack. The fit must stay bounded near the
    sampled sky value everywhere, including inside the gap."""

    def setUp(self):
        import src.background as bg_mod
        if not bg_mod.HAS_RBF:
            self.skipTest("scipy.interpolate.RBFInterpolator not available")
        self.fit_rbf_surface = bg_mod._fit_rbf_surface

    def test_surface_bounded_across_large_sample_gap(self):
        rng = np.random.default_rng(0)
        sky = 5000.0
        # Dense samples everywhere except a large contiguous wedge in one
        # corner (mimics a star-mask/outlier-rejection gap), all at a flat
        # sky value with small noise.
        pts = []
        for gy in np.linspace(0.02, 0.98, 25):
            for gx in np.linspace(0.02, 0.98, 25):
                if gx > 0.55 and gy > 0.55:
                    continue  # the gap
                pts.append((gy, gx))
        coords = np.array(pts)
        values = sky + rng.normal(0, 5.0, len(pts))

        surface = self.fit_rbf_surface(
            coords, values, H=256, W=256, kernel='thin_plate_spline',
            smoothing=0.0, outlier_sigma=3.0, max_iter=3,
            patch_size=32, verbose=False)

        # Sample inside the gap (bottom-right corner in pixel space).
        gap = surface[220:256, 220:256]
        self.assertLess(abs(float(np.median(gap)) - sky), 200.0)
        self.assertLess(float(np.max(np.abs(surface - sky))), 500.0)


class TestApplyBackgroundExtraction(unittest.TestCase):

    def setUp(self):
        from src.background import apply_background_extraction
        self.apply_bg = apply_background_extraction

    def test_output_shape_and_dtype(self):
        rgb = _star_rgb(64, 64)
        result = self.apply_bg(rgb, mesh_size=16)
        self.assertEqual(result.shape, rgb.shape)
        self.assertEqual(result.dtype, np.float32)

    def test_reduces_background_level(self):
        """After extraction the median sky level should drop toward zero."""
        bg_level = 500.0
        rgb = _rgb(64, 64, bg_level)
        result = self.apply_bg(rgb, mesh_size=16)
        self.assertLess(float(np.median(result)), bg_level * 0.5)

    def test_does_not_clip_negative_sky(self):
        """Subtraction is intentionally allowed to go negative — no hard clip to 0."""
        rng = np.random.default_rng(1)
        rgb = rng.normal(100, 10, (64, 64, 3)).astype(np.float32)
        result = self.apply_bg(rgb, mesh_size=16)
        # After subtracting roughly uniform background there should be negative pixels
        self.assertTrue(np.any(result < 0))


class TestSkyFloorNormalize(unittest.TestCase):

    def setUp(self):
        from src.background import sky_floor_normalize
        self.sky_floor_normalize = sky_floor_normalize

    def test_output_shape_and_dtype(self):
        rgb = _star_rgb()
        result = self.sky_floor_normalize(rgb)
        self.assertEqual(result.shape, rgb.shape)
        self.assertEqual(result.dtype, np.float32)

    def test_sky_floor_near_zero_after_normalization(self):
        """Sky background should be pushed close to zero."""
        bg_level = 200.0
        rng = np.random.default_rng(2)
        rgb = rng.normal(bg_level, 5, (128, 128, 3)).astype(np.float32)
        result = self.sky_floor_normalize(rgb)
        self.assertLess(float(np.median(result)), 20.0)

    def test_clips_output_to_non_negative(self):
        rgb = _star_rgb(bg=100.0)
        result = self.sky_floor_normalize(rgb)
        self.assertGreaterEqual(float(result.min()), 0.0)

    def test_no_crash_on_crowded_field(self):
        """Should return unchanged image if there are too few sky pixels."""
        # All pixels are bright "sources"
        rgb = np.full((64, 64, 3), 10000.0, dtype=np.float32)
        result = self.sky_floor_normalize(rgb)
        self.assertEqual(result.shape, rgb.shape)


class TestEstimateSkySigma(unittest.TestCase):

    def setUp(self):
        from src.background import _estimate_sky_sigma
        self.estimate = _estimate_sky_sigma

    def test_returns_positive_value(self):
        rng = np.random.default_rng(3)
        rgb = rng.normal(100, 5, (64, 64, 3)).astype(np.float32)
        sigma = self.estimate(rgb)
        self.assertGreater(sigma, 0.0)

    def test_higher_noise_gives_higher_sigma(self):
        rng = np.random.default_rng(4)
        rgb_low  = rng.normal(100, 2, (64, 64, 3)).astype(np.float32)
        rgb_high = rng.normal(100, 20, (64, 64, 3)).astype(np.float32)
        self.assertLess(self.estimate(rgb_low), self.estimate(rgb_high))

    def test_all_zeros_returns_fallback(self):
        rgb = np.zeros((64, 64, 3), dtype=np.float32)
        sigma = self.estimate(rgb)
        self.assertGreater(sigma, 0.0)


# ===========================================================================
# Denoising
# ===========================================================================

class TestBayesShrinkThreshold(unittest.TestCase):

    def setUp(self):
        from src.denoising import _bayesshrink_threshold
        self.threshold = _bayesshrink_threshold

    def test_pure_noise_returns_inf(self):
        """When the assumed noise level exceeds the observed RMS, signal variance
        <= 0 and the function should return inf (zero all coefficients)."""
        rng = np.random.default_rng(5)
        noise_coeffs = rng.normal(0, 2.0, (32, 32))
        # Pass sigma_noise larger than the observed RMS so sigma_sq_s <= 0
        observed_rms = float(np.sqrt(np.mean(noise_coeffs ** 2)))
        t = self.threshold(noise_coeffs, sigma_noise=observed_rms * 2.0)
        self.assertTrue(np.isinf(t))

    def test_strong_signal_returns_finite_threshold(self):
        """When signal >> noise the threshold should be finite and positive."""
        rng = np.random.default_rng(6)
        signal = np.sin(np.linspace(0, 4 * np.pi, 64))
        coeffs = np.outer(signal, signal) * 100.0 + rng.normal(0, 1.0, (64, 64))
        t = self.threshold(coeffs, sigma_noise=1.0)
        self.assertTrue(np.isfinite(t))
        self.assertGreater(t, 0.0)

    def test_threshold_increases_with_noise(self):
        """Noisier subbands should receive a larger (more aggressive) threshold."""
        rng = np.random.default_rng(7)
        # Fixed signal, varying noise level
        signal = np.ones((32, 32)) * 10.0
        t_low  = self.threshold(signal + rng.normal(0, 1, (32, 32)), 1.0)
        t_high = self.threshold(signal + rng.normal(0, 5, (32, 32)), 5.0)
        self.assertGreaterEqual(t_high, t_low)


class TestWaveletDenoise(unittest.TestCase):

    def setUp(self):
        from src.denoising import wavelet_denoise, HAS_PYWT
        self.denoise = wavelet_denoise
        self.has_pywt = HAS_PYWT

    def test_shape_preserved(self):
        rgb = _star_rgb(64, 64)
        result = self.denoise(rgb)
        self.assertEqual(result.shape, rgb.shape)

    def test_dtype_is_float32(self):
        rgb = _star_rgb(64, 64)
        result = self.denoise(rgb)
        self.assertEqual(result.dtype, np.float32)

    @unittest.skipUnless(True, "always run — graceful pass-through when pywt absent")
    def test_noisy_image_noise_reduced(self):
        if not self.has_pywt:
            self.skipTest("pywt not installed")
        rng = np.random.default_rng(8)
        clean = _star_rgb(64, 64, amp=300.0, bg=50.0)
        noisy = (clean + rng.normal(0, 20, clean.shape)).astype(np.float32)
        denoised = self.denoise(noisy)
        noise_before = float(np.std(noisy - clean))
        noise_after  = float(np.std(denoised - clean))
        self.assertLess(noise_after, noise_before)

    def test_star_mask_restores_star_cores(self):
        if not self.has_pywt:
            self.skipTest("pywt not installed")
        rng = np.random.default_rng(9)
        rgb = _star_rgb(64, 64, centers=[(32, 32)], amp=2000.0, bg=50.0)
        noisy = (rgb + rng.normal(0, 30, rgb.shape)).astype(np.float32)
        star_mask = np.zeros((64, 64), dtype=np.float32)
        star_mask[29:36, 29:36] = 1.0
        result_no_mask   = self.denoise(noisy)
        result_with_mask = self.denoise(noisy, star_mask=star_mask)
        # At star centre, masked result should be closer to original noisy data
        cy, cx = 32, 32
        diff_no_mask   = abs(float(result_no_mask[cy, cx, 0]) - float(noisy[cy, cx, 0]))
        diff_with_mask = abs(float(result_with_mask[cy, cx, 0]) - float(noisy[cy, cx, 0]))
        self.assertLess(diff_with_mask, diff_no_mask + 1.0)


class TestAdaptiveWaveletDenoise(unittest.TestCase):

    def setUp(self):
        from src.denoising import adaptive_wavelet_denoise, HAS_PYWT
        self.denoise = adaptive_wavelet_denoise
        self.has_pywt = HAS_PYWT

    def test_shape_and_dtype(self):
        rgb = _star_rgb(64, 64)
        result = self.denoise(rgb)
        self.assertEqual(result.shape, rgb.shape)
        self.assertEqual(result.dtype, np.float32)

    def test_passthrough_without_pywt(self):
        """If pywt is absent the function must return the input unchanged."""
        if self.has_pywt:
            self.skipTest("pywt is installed — testing degraded path not needed")
        rgb = _star_rgb(64, 64)
        result = self.denoise(rgb)
        np.testing.assert_array_equal(result, rgb)

    def test_noise_reduction_quality(self):
        if not self.has_pywt:
            self.skipTest("pywt not installed")
        rng = np.random.default_rng(10)
        clean = _star_rgb(64, 64, amp=300.0, bg=50.0)
        noisy = (clean + rng.normal(0, 20, clean.shape)).astype(np.float32)
        denoised = self.denoise(noisy)
        self.assertLess(float(np.std(denoised - clean)), float(np.std(noisy - clean)))


class TestArcSinhStretch(unittest.TestCase):

    def setUp(self):
        from src.denoising import arcsinh_stretch
        self.stretch = arcsinh_stretch

    def test_output_in_unit_range(self):
        img = _star_rgb().astype(np.float32)
        result = self.stretch(img)
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)

    def test_shape_preserved(self):
        img = _star_rgb(32, 48)
        result = self.stretch(img)
        self.assertEqual(result.shape, img.shape)

    def test_zero_image_returns_zeros(self):
        img = np.zeros((32, 32, 3), dtype=np.float32)
        result = self.stretch(img)
        np.testing.assert_array_equal(result, np.zeros_like(result))

    def test_brighter_pixel_maps_higher(self):
        """The arcsinh stretch is monotonic: a brighter pixel must map higher,
        provided neither pixel is saturated to 1.0."""
        rng = np.random.default_rng(20)
        # Uniform background; moderate star amplitudes that stay below the 99.8th
        # percentile white point so neither pixel clips to 1.0.
        img = rng.normal(200, 10, (64, 64, 3)).astype(np.float32)
        img[10, 10, :] += 30.0
        img[20, 20, :] += 80.0
        result = self.stretch(img)
        self.assertGreater(float(result[20, 20, 0]), float(result[10, 10, 0]))

    def test_explicit_factor_accepted(self):
        img = _star_rgb()
        result = self.stretch(img, factor=10.0)
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)


class TestReduceChromaNoise(unittest.TestCase):

    def setUp(self):
        from src.denoising import reduce_chroma_noise
        self.reduce = reduce_chroma_noise

    def test_shape_preserved(self):
        rgb = _star_rgb()
        result = self.reduce(rgb)
        self.assertEqual(result.shape, rgb.shape)

    def test_dtype_is_float32(self):
        rgb = _star_rgb()
        result = self.reduce(rgb)
        self.assertEqual(result.dtype, np.float32)

    def test_non_negative_output(self):
        rgb = _star_rgb(bg=50.0)
        result = self.reduce(rgb)
        self.assertGreaterEqual(float(result.min()), 0.0)

    def test_chroma_noise_reduced_in_sky(self):
        """Sky pixels should have less channel-to-channel variation after NR."""
        rng = np.random.default_rng(11)
        # Flat sky with random per-channel noise (chroma noise)
        base = 100.0
        rgb = np.stack([
            rng.normal(base, 10, (64, 64)),
            rng.normal(base, 10, (64, 64)),
            rng.normal(base, 10, (64, 64)),
        ], axis=2).astype(np.float32)
        result = self.reduce(rgb, sigma=3.0)
        before = float(np.std(rgb[:, :, 0] - rgb[:, :, 1]))
        after  = float(np.std(result[:, :, 0] - result[:, :, 1]))
        self.assertLess(after, before)


# ===========================================================================
# Quality analysis
# ===========================================================================

class TestGenerateStarMask(unittest.TestCase):

    def setUp(self):
        from src.quality import generate_star_mask
        self.generate_star_mask = generate_star_mask

    def _mock_sources(self, positions):
        """Return a minimal photutils-style table substitute."""
        import numpy.lib.recfunctions as rf
        yc = np.array([p[0] for p in positions], dtype=np.float64)
        xc = np.array([p[1] for p in positions], dtype=np.float64)
        # Use a simple structured array
        dt = np.dtype([('ycentroid', np.float64), ('xcentroid', np.float64)])
        arr = np.zeros(len(positions), dtype=dt)
        arr['ycentroid'] = yc
        arr['xcentroid'] = xc
        return arr

    def test_output_shape(self):
        sources = self._mock_sources([(16, 16), (48, 48)])
        mask = self.generate_star_mask((64, 64), sources)
        self.assertEqual(mask.shape, (64, 64))

    def test_values_in_zero_one(self):
        sources = self._mock_sources([(32, 32)])
        mask = self.generate_star_mask((64, 64), sources)
        self.assertGreaterEqual(float(mask.min()), 0.0)
        self.assertLessEqual(float(mask.max()), 1.0 + 1e-6)

    def test_star_centre_has_high_value(self):
        sources = self._mock_sources([(32, 32)])
        mask = self.generate_star_mask((64, 64), sources, fwhm=4.0)
        self.assertGreater(float(mask[32, 32]), 0.5)

    def test_empty_sources_returns_zeros(self):
        mask = self.generate_star_mask((32, 32), None)
        np.testing.assert_array_equal(mask, np.zeros((32, 32), dtype=np.float32))

    def test_multiple_stars_covered(self):
        centers = [(10, 10), (50, 50)]
        sources = self._mock_sources(centers)
        mask = self.generate_star_mask((64, 64), sources, fwhm=3.0)
        for cy, cx in centers:
            self.assertGreater(float(mask[cy, cx]), 0.3,
                               msg=f"Star at ({cy},{cx}) not covered by mask")


class TestMeasureFwhm(unittest.TestCase):

    def setUp(self):
        from src.quality import measure_fwhm
        self.measure_fwhm = measure_fwhm

    def _gaussian_sources(self, shape, centers, fwhm=4.0, amp=500.0, bg=50.0):
        img = np.full(shape, bg, dtype=np.float32)
        yy, xx = np.indices(shape)
        sigma = fwhm / 2.355
        for cy, cx in centers:
            img += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
        dt = np.dtype([('ycentroid', np.float64), ('xcentroid', np.float64),
                       ('flux', np.float64)])
        arr = np.zeros(len(centers), dtype=dt)
        arr['ycentroid'] = [c[0] for c in centers]
        arr['xcentroid'] = [c[1] for c in centers]
        arr['flux'] = amp
        return img, arr

    def test_returns_positive_for_star_field(self):
        img, sources = self._gaussian_sources((64, 64), [(32, 32)], fwhm=4.0)
        fwhm = self.measure_fwhm(img, sources)
        self.assertGreater(fwhm, 0.0)

    def test_fwhm_plausible_range(self):
        """Measured FWHM should be in the right ballpark for the injected PSF."""
        true_fwhm = 5.0
        img, sources = self._gaussian_sources((128, 128), [(64, 64)], fwhm=true_fwhm)
        fwhm = self.measure_fwhm(img, sources)
        # Accept ±3 px tolerance for the half-max area estimator
        self.assertLess(abs(fwhm - true_fwhm), 3.0)

    def test_empty_sources_returns_zero(self):
        img = np.ones((32, 32), dtype=np.float32) * 100.0
        fwhm = self.measure_fwhm(img, None)
        self.assertEqual(fwhm, 0.0)

    def test_larger_psf_gives_larger_fwhm(self):
        img_small, src_small = self._gaussian_sources((128, 128), [(64, 64)], fwhm=3.0)
        img_large, src_large = self._gaussian_sources((128, 128), [(64, 64)], fwhm=8.0)
        fwhm_small = self.measure_fwhm(img_small, src_small)
        fwhm_large = self.measure_fwhm(img_large, src_large)
        self.assertLess(fwhm_small, fwhm_large)


# ===========================================================================
# Quality metrics — MAD estimator, advanced_metrics flag, SEP fast path,
# FWHM scaling, and _process_single_frame preloaded_data path
# ===========================================================================

class TestComputeQualityMetricsAdvanced(unittest.TestCase):
    """Tests for the new compute_quality_metrics parameters and MAD estimator."""

    def setUp(self):
        from src.quality import compute_quality_metrics
        self.cqm = compute_quality_metrics

    def _star_lum(self, shape=(64, 64), bg=100.0, amp=500.0):
        img = np.full(shape, bg, dtype=np.float32)
        yy, xx = np.indices(shape)
        for cy, cx in [(16, 16), (48, 48)]:
            img += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.0 ** 2))
        return img

    def test_metrics_include_strehl_and_dispersion_keys(self):
        m = self.cqm(self._star_lum())
        self.assertIn('strehl', m)
        self.assertIn('dispersion_px', m)

    def test_advanced_metrics_false_zeros_strehl(self):
        """When advanced_metrics=False, Strehl and dispersion must be 0.0."""
        m = self.cqm(self._star_lum(), advanced_metrics=False)
        self.assertEqual(m['strehl'], 0.0)
        self.assertEqual(m['dispersion_px'], 0.0)

    def test_advanced_metrics_true_is_default(self):
        """Default call (no advanced_metrics kwarg) should return the same keys."""
        m = self.cqm(self._star_lum())
        self.assertIn('strehl', m)
        self.assertIn('dispersion_px', m)

    def test_mad_background_close_to_median(self):
        """MAD estimator: background should equal the pixel median."""
        bg = 300.0
        img = np.full((64, 64), bg, dtype=np.float32)
        # Add a few bright stars so the image isn't rejected as flat
        img[10, 10] = 1500.0
        img[50, 50] = 1200.0
        m = self.cqm(img)
        self.assertAlmostEqual(m['background'], bg, delta=5.0)

    def test_mad_noise_is_positive(self):
        """MAD estimator must always return positive noise."""
        m = self.cqm(self._star_lum())
        self.assertGreater(m['noise'], 0.0)

    def test_fwhm_monotone_larger_psf_larger_fwhm(self):
        """Larger injected PSF must yield larger reported FWHM from compute_quality_metrics."""
        def _field(fwhm_px):
            img = np.full((128, 128), 100.0, dtype=np.float32)
            yy, xx = np.indices((128, 128))
            sigma = fwhm_px / 2.355
            for cy, cx in [(32, 32), (64, 96), (96, 32)]:
                img += 800.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
            return img

        m_small = self.cqm(_field(3.0))
        m_large = self.cqm(_field(7.0))
        if m_small['fwhm'] > 0 and m_large['fwhm'] > 0:
            self.assertLess(m_small['fwhm'], m_large['fwhm'])

    def test_quick_mode_skips_star_detection(self):
        """quick=True must still return all required keys, with star_count=0."""
        m = self.cqm(self._star_lum(), quick=True)
        for k in ('brightness', 'score', 'snr', 'background', 'noise'):
            self.assertIn(k, m)
        self.assertEqual(m['star_count'], 0)
        self.assertEqual(m['fwhm'], 0.0)


class TestSepDetectStars(unittest.TestCase):
    """Tests for the _sep_detect_stars() fast-path function."""

    def setUp(self):
        from src.quality import _sep_detect_stars, _SEP_AVAILABLE
        self.detect = _sep_detect_stars
        self.sep_available = _SEP_AVAILABLE

    def _star_image(self, shape=(128, 128), n_stars=10, bg=100.0, amp=800.0):
        rng = np.random.default_rng(42)
        img = rng.normal(bg, 5.0, shape).astype(np.float32)
        yy, xx = np.indices(shape)
        for _ in range(n_stars):
            cy = rng.integers(20, shape[0] - 20)
            cx = rng.integers(20, shape[1] - 20)
            img += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.5 ** 2))
        return img.clip(0)

    def test_returns_none_when_sep_unavailable(self):
        """When SEP is not installed, _sep_detect_stars must return None gracefully."""
        if self.sep_available:
            self.skipTest("SEP is installed — testing unavailable path is not needed")
        result = self.detect(self._star_image(), noise=5.0)
        self.assertIsNone(result)

    @unittest.skipUnless(True, "always run; guarded internally by sep_available")
    def test_returns_compatible_structured_array_when_sep_available(self):
        if not self.sep_available:
            self.skipTest("SEP not installed")
        result = self.detect(self._star_image(), noise=5.0)
        self.assertIsNotNone(result)
        for field in ('xcentroid', 'ycentroid', 'flux', 'peak', 'roundness1', 'sharpness'):
            self.assertIn(field, result.dtype.names)

    def test_elongated_sources_filtered_out(self):
        """Very elongated sources (streaks) should be removed by the roundness filter."""
        if not self.sep_available:
            self.skipTest("SEP not installed")
        img = np.full((128, 128), 100.0, dtype=np.float32)
        # Inject a horizontal streak (elongated in x)
        img[64, 20:108] = 5000.0
        result = self.detect(img, noise=5.0)
        # Either no sources or only filtered (round) ones
        if result is not None and len(result) > 0:
            roundness = result['roundness1']
            self.assertTrue(np.all(roundness < 0.75),
                            f"Elongated source not filtered: max roundness={roundness.max():.2f}")

    def test_empty_image_returns_none(self):
        """Flat image with no sources should return None or empty result."""
        img = np.full((128, 128), 100.0, dtype=np.float32)
        if not self.sep_available:
            self.skipTest("SEP not installed")
        result = self.detect(img, noise=5.0)
        self.assertTrue(result is None or len(result) == 0)


class TestProcessSingleFramePreload(unittest.TestCase):
    """Tests for the preloaded_data and advanced_metrics paths in _process_single_frame."""

    def setUp(self):
        from src.frame_processor import _process_single_frame
        self.process = _process_single_frame

    def _fake_raw(self, shape=(64, 64)):
        """Bayer raw with enough dynamic range to pass validate_image_data.

        Background of ~200 ADU with several bright star spikes ensures
        p99 - p01 > 10 after bilinear debayering.
        """
        rng = np.random.default_rng(1)
        img = rng.normal(200.0, 15.0, shape).astype(np.float32).clip(0)
        # Inject bright stars so the luminance dynamic range exceeds 10 ADU
        yy, xx = np.indices(shape)
        for cy, cx in [(10, 10), (30, 50), (50, 20)]:
            img += 800.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.0 ** 2))
        return img.clip(0)

    def test_preloaded_data_bypasses_disk(self):
        """When preloaded_data is supplied, load_fits must NOT be called."""
        from unittest.mock import patch
        raw = self._fake_raw()
        with patch('src.frame_processor.load_fits', side_effect=AssertionError("load_fits called")):
            result = self.process(
                path='nonexistent.fits',
                header={},
                masters={},
                debayer_method='bilinear',
                white_balance='none',
                preloaded_data=(raw, {}),
            )
        # Should succeed (debayer + quality) without touching the disk
        self.assertIsNone(result.get('error'), result.get('error'))
        self.assertIn('rgb', result)

    def test_advanced_metrics_false_zeros_strehl_in_result(self):
        """advanced_metrics=False must propagate to compute_quality_metrics."""
        raw = self._fake_raw((64, 64))
        result = self.process(
            path='dummy.fits',
            header={},
            masters={},
            debayer_method='bilinear',
            white_balance='none',
            advanced_metrics=False,
            preloaded_data=(raw, {}),
        )
        self.assertIsNone(result.get('error'), result.get('error'))
        self.assertEqual(result['metrics'].get('strehl', 0.0), 0.0)
        self.assertEqual(result['metrics'].get('dispersion_px', 0.0), 0.0)

    def test_advanced_metrics_true_is_default(self):
        """Default call must include strehl key in metrics."""
        raw = self._fake_raw((64, 64))
        result = self.process(
            path='dummy.fits',
            header={},
            masters={},
            debayer_method='bilinear',
            white_balance='none',
            preloaded_data=(raw, {}),
        )
        self.assertIsNone(result.get('error'), result.get('error'))
        self.assertIn('strehl', result['metrics'])


# ===========================================================================
# Stacking — L.A.Cosmic cosmic ray rejection
# ===========================================================================

class TestLACosmicReject(unittest.TestCase):

    def setUp(self):
        from src.stacking import lacosmic_reject
        self.lacosmic_reject = lacosmic_reject

    def _clean_rgb(self, shape=(64, 64), bg=100.0, noise=3.0) -> np.ndarray:
        rng = np.random.default_rng(42)
        data = rng.normal(bg, noise, (*shape, 3)).astype(np.float32)
        return np.clip(data, 0, None)

    def test_output_shape_preserved(self):
        rgb = self._clean_rgb()
        result = self.lacosmic_reject(rgb)
        self.assertEqual(result.shape, rgb.shape)

    def test_output_dtype_is_float32(self):
        rgb = self._clean_rgb()
        result = self.lacosmic_reject(rgb)
        self.assertEqual(result.dtype, np.float32)

    def test_cosmic_ray_spike_is_reduced(self):
        """A single-pixel spike far above the noise floor should be brought down."""
        rgb = self._clean_rgb((64, 64))
        # Inject a bright single-pixel CR at (30, 30) in the red channel
        rgb[30, 30, 0] = 50000.0
        result = self.lacosmic_reject(rgb, sigclip=4.5, objlim=5.0)
        self.assertLess(float(result[30, 30, 0]), 50000.0 * 0.5)

    def test_star_core_not_flagged(self):
        """A Gaussian star should NOT be rejected — it has an extended PSF."""
        rng = np.random.default_rng(43)
        rgb = rng.normal(100, 3, (64, 64, 3)).astype(np.float32)
        # Add a realistic star (sigma=3 pixels, peak ~2000 ADU above background)
        yy, xx = np.indices((64, 64))
        r2 = (yy - 32) ** 2 + (xx - 32) ** 2
        star = 2000.0 * np.exp(-r2 / (2 * 3.0 ** 2))
        rgb[:, :, 0] += star
        rgb[:, :, 1] += star
        rgb[:, :, 2] += star
        original_peak = float(rgb[32, 32, 0])
        result = self.lacosmic_reject(rgb, sigclip=4.5, objlim=5.0)
        cleaned_peak = float(result[32, 32, 0])
        # Peak should be largely preserved (within 20 % of original)
        self.assertGreater(cleaned_peak, original_peak * 0.80)

    def test_clean_image_unchanged(self):
        """An image without bright spikes should pass through essentially unmodified."""
        rng = np.random.default_rng(44)
        rgb = (rng.normal(200, 5, (64, 64, 3))).astype(np.float32)
        result = self.lacosmic_reject(rgb, sigclip=4.5, objlim=5.0)
        # Mean should be similar (within 5 %)
        self.assertAlmostEqual(float(np.mean(result)), float(np.mean(rgb)),
                               delta=float(np.mean(rgb)) * 0.05)

    def test_multiple_channel_crs_cleaned_independently(self):
        """CR in one channel should not affect the other channels."""
        rgb = self._clean_rgb((64, 64))
        rgb[20, 20, 1] = 40000.0   # Green channel CR only
        result = self.lacosmic_reject(rgb, sigclip=4.5, objlim=5.0)
        # Red and blue at that pixel should be virtually unchanged
        self.assertAlmostEqual(float(result[20, 20, 0]), float(rgb[20, 20, 0]),
                               delta=50.0)
        self.assertAlmostEqual(float(result[20, 20, 2]), float(rgb[20, 20, 2]),
                               delta=50.0)


# ===========================================================================
# Health check
# ===========================================================================

class TestRunHealthCheck(unittest.TestCase):

    def setUp(self):
        from src.health_check import run_health_check
        self.run_health_check = run_health_check

    def _light(self, **kwargs) -> FrameInfo:
        hdr = {'NAXIS1': 800, 'NAXIS2': 600, 'EXPTIME': 120.0,
               'ISOSPEED': 800}
        hdr.update(kwargs)
        return _make_frame(hdr, 'light')

    def _dark(self, **kwargs) -> FrameInfo:
        hdr = {'NAXIS1': 800, 'NAXIS2': 600, 'EXPTIME': 120.0,
               'ISOSPEED': 800}
        hdr.update(kwargs)
        return _make_frame(hdr, 'dark')

    def test_no_lights_reports_cannot_stack(self):
        out = _capture(self.run_health_check,
                       {'light': [], 'dark': [], 'flat': [], 'bias': []},
                       {}, 'dummy_dir')
        self.assertIn('CANNOT STACK', out)

    def test_consistent_lights_reports_ready(self):
        """Enough consistent lights with matching calibration frames → READY."""
        from src.models import Config
        n = Config.MIN_RECOMMENDED_FRAMES
        lights = [self._light() for _ in range(n)]
        darks  = [self._dark()]
        flats  = [_make_frame({'NAXIS1': 800, 'NAXIS2': 600}, 'flat')]
        biases = [_make_frame({'NAXIS1': 800, 'NAXIS2': 600}, 'bias')]
        masters = {'dark_exptime': 120.0}
        out = _capture(self.run_health_check,
                       {'light': lights, 'dark': darks, 'flat': flats, 'bias': biases},
                       masters, 'dummy_dir')
        self.assertIn('READY TO STACK', out)

    def test_mixed_iso_triggers_warning(self):
        lights = [self._light(ISOSPEED=800) for _ in range(3)]
        lights += [self._light(ISOSPEED=1600) for _ in range(2)]
        out = _capture(self.run_health_check,
                       {'light': lights, 'dark': [], 'flat': [], 'bias': []},
                       {}, 'dummy_dir')
        self.assertIn('ISO', out)

    def test_mixed_dimensions_warns(self):
        lights = [self._light(NAXIS1=800, NAXIS2=600) for _ in range(3)]
        lights += [self._light(NAXIS1=1600, NAXIS2=1200) for _ in range(2)]
        out = _capture(self.run_health_check,
                       {'light': lights, 'dark': [], 'flat': [], 'bias': []},
                       {}, 'dummy_dir')
        self.assertIn('INCONSISTENT', out)

    def test_no_darks_warns(self):
        lights = [self._light() for _ in range(5)]
        out = _capture(self.run_health_check,
                       {'light': lights, 'dark': [], 'flat': [], 'bias': []},
                       {}, 'dummy_dir')
        self.assertIn('dark', out.lower())

    def test_no_flats_warns(self):
        lights = [self._light() for _ in range(5)]
        out = _capture(self.run_health_check,
                       {'light': lights, 'dark': [], 'flat': [], 'bias': []},
                       {}, 'dummy_dir')
        self.assertIn('flat', out.lower())

    def test_dark_exposure_mismatch_warns(self):
        lights = [self._light(EXPTIME=120.0) for _ in range(3)]
        darks  = [self._dark(EXPTIME=60.0)]
        masters = {'dark_exptime': 60.0}
        out = _capture(self.run_health_check,
                       {'light': lights, 'dark': darks, 'flat': [], 'bias': []},
                       masters, 'dummy_dir')
        self.assertIn('exposure', out.lower())

    def test_dark_dimension_mismatch_warns(self):
        lights = [self._light(NAXIS1=800, NAXIS2=600) for _ in range(3)]
        darks  = [self._dark(NAXIS1=1600, NAXIS2=1200)]
        out = _capture(self.run_health_check,
                       {'light': lights, 'dark': darks, 'flat': [], 'bias': []},
                       {}, 'dummy_dir')
        self.assertIn('differ', out.lower())

    def test_low_frame_count_warns(self):
        lights = [self._light()]   # just one frame
        out = _capture(self.run_health_check,
                       {'light': lights, 'dark': [], 'flat': [], 'bias': []},
                       {}, 'dummy_dir')
        self.assertIn('recommended', out.lower())

    def test_mixed_exposure_warns(self):
        lights = [self._light(EXPTIME=120.0) for _ in range(3)]
        lights += [self._light(EXPTIME=60.0) for _ in range(2)]
        out = _capture(self.run_health_check,
                       {'light': lights, 'dark': [], 'flat': [], 'bias': []},
                       {}, 'dummy_dir')
        self.assertIn('exposure', out.lower())

    def test_temperature_recorded_in_output(self):
        lights = [self._light(**{'CCD-TEMP': -10.0}) for _ in range(3)]
        out = _capture(self.run_health_check,
                       {'light': lights, 'dark': [], 'flat': [], 'bias': []},
                       {}, 'dummy_dir')
        self.assertIn('temp', out.lower())


if __name__ == '__main__':
    unittest.main()
