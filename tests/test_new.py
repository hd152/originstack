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


class TestFitBackgroundSurface(unittest.TestCase):
    """Behavioural tests for DBE's robust local-regression surface fit.

    The original thin-plate-spline RBF fit was unbounded: a large contiguous
    gap in the patch samples (left by outlier rejection near a bright star)
    let it extrapolate far outside the real sky range, producing a visible
    over/under-subtracted wedge after the surface was subtracted from a flat
    stack. The robust local fit must stay bounded near the sampled sky value
    everywhere, follow genuine gradients, and shrug off contaminated patches.
    """

    def setUp(self):
        import src.background as bg_mod
        self.fit_surface = bg_mod._fit_background_surface

    def _fit(self, coords, values, H=256, W=256, patch_size=32):
        return self.fit_surface(coords, values, H=H, W=W,
                                outlier_sigma=2.5, max_iter=3,
                                patch_size=patch_size, verbose=False)

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

        surface = self._fit(coords, values)

        # Sample inside the gap (bottom-right corner in pixel space).
        gap = surface[220:256, 220:256]
        self.assertLess(abs(float(np.median(gap)) - sky), 200.0)
        self.assertLess(float(np.max(np.abs(surface - sky))), 500.0)

    def test_recovers_linear_gradient(self):
        """A light-pollution-style linear gradient must be followed, not
        flattened, including at the image edges."""
        rng = np.random.default_rng(1)
        pts, vals = [], []
        for gy in np.linspace(0.02, 0.98, 25):
            for gx in np.linspace(0.02, 0.98, 25):
                pts.append((gy, gx))
                vals.append(5000.0 + 400.0 * gy + 250.0 * gx
                            + rng.normal(0, 5.0))
        surface = self._fit(np.array(pts), np.array(vals))

        H = W = 256
        gy, gx = np.mgrid[0:H, 0:W]
        truth = 5000.0 + 400.0 * (gy / H) + 250.0 * (gx / W)
        err = np.abs(surface - truth)
        # Interior must track closely; edges may show slight boundary bias.
        self.assertLess(float(np.median(err)), 15.0)
        self.assertLess(float(err.max()), 80.0)

    def test_contaminated_patches_downweighted(self):
        """A cluster of patches inflated by a bright object must not drag
        the fitted surface up: IRLS should suppress them."""
        rng = np.random.default_rng(2)
        pts, vals = [], []
        for gy in np.linspace(0.02, 0.98, 25):
            for gx in np.linspace(0.02, 0.98, 25):
                pts.append((gy, gx))
                v = 5000.0 + rng.normal(0, 5.0)
                if 0.4 < gy < 0.6 and 0.4 < gx < 0.6:
                    v += 800.0  # contaminated cluster
                vals.append(v)
        surface = self._fit(np.array(pts), np.array(vals))
        center = surface[118:138, 118:138]
        # Without downweighting the center would sit ~800 above sky.
        self.assertLess(abs(float(np.median(center)) - 5000.0), 120.0)


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
        from src.denoising import wavelet_denoise
        self.denoise = wavelet_denoise

    def test_shape_preserved(self):
        rgb = _star_rgb(64, 64)
        result = self.denoise(rgb)
        self.assertEqual(result.shape, rgb.shape)

    def test_dtype_is_float32(self):
        rgb = _star_rgb(64, 64)
        result = self.denoise(rgb)
        self.assertEqual(result.dtype, np.float32)

    def test_noisy_image_noise_reduced(self):
        rng = np.random.default_rng(8)
        clean = _star_rgb(64, 64, amp=300.0, bg=50.0)
        noisy = (clean + rng.normal(0, 20, clean.shape)).astype(np.float32)
        denoised = self.denoise(noisy)
        noise_before = float(np.std(noisy - clean))
        noise_after  = float(np.std(denoised - clean))
        self.assertLess(noise_after, noise_before)

    def test_star_mask_restores_star_cores(self):
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
        from src.denoising import adaptive_wavelet_denoise
        self.denoise = adaptive_wavelet_denoise

    def test_shape_and_dtype(self):
        rgb = _star_rgb(64, 64)
        result = self.denoise(rgb)
        self.assertEqual(result.shape, rgb.shape)
        self.assertEqual(result.dtype, np.float32)

    def test_noise_reduction_quality(self):
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
        """Return a minimal _SOURCES_DTYPE-compatible structured array."""
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


class TestPreviewBlackSigmaDepthScaling(unittest.TestCase):
    """The auto-advisor softens the preview black-point clip on shallow
    stacks: residual mid-scale background structure modulates which sky
    pixels cross the clip threshold, rendering as soft black splotches in
    the preview JPEG when integration is short. Deep stacks keep the
    preset's aggressive clip; already-low presets pass through untouched."""

    def _run(self, n_frames, preset):
        import argparse
        from src.auto_settings import _apply_quality_settings
        args = argparse.Namespace(preview_black_sigma=preset,
                                  stack_method='sigma_clip',
                                  auto_denoise_strength=True)
        sig = {'n_frames': n_frames, 'snr': 1.7, 'fwhm': 5.4,
               'star_count': 25, 'strehl': 0.3, 'dispersion': 0.5,
               'median_ellipticity': 0.1, 'dynamic_range': 100,
               'concentration': 5, 'median_filling': 0.1,
               'diffuse_excess': 0.5, 'peak_excess': 5}
        _apply_quality_settings(sig, args, 'galaxy')
        return args.preview_black_sigma

    def test_very_shallow_caps_to_one(self):
        self.assertEqual(self._run(12, 3.0), 1.0)

    def test_shallow_caps_to_two(self):
        self.assertEqual(self._run(35, 3.0), 2.0)

    def test_deep_stack_untouched(self):
        self.assertEqual(self._run(205, 3.0), 3.0)

    def test_low_preset_untouched(self):
        self.assertEqual(self._run(12, 1.0), 1.0)

    def test_negative_preset_untouched(self):
        self.assertEqual(self._run(12, -0.5), -0.5)


class TestPatchScoresPhase1Split(unittest.TestCase):
    """compute_patch_scores (Phase 1) + patch_scores_to_map (Phase 2) must
    reproduce compute_patch_quality_map exactly for an unshifted frame, and
    place weight correctly after a coarse-grid shift. The old path warped the
    full-res frame with cval=0 before scoring, so the sky->0 border step
    inflated border-patch Brenner scores and poisoned the per-frame max
    normalisation for strongly dithered frames; the split path scores before
    shifting and avoids that entirely."""

    def test_zero_shift_matches_legacy_exactly(self):
        from src.registration import (compute_patch_quality_map,
                                      compute_patch_scores, patch_scores_to_map)
        rng = np.random.default_rng(0)
        lum = rng.normal(1000, 50, (512, 768)).astype(np.float32)
        legacy = compute_patch_quality_map(lum)
        split = patch_scores_to_map(compute_patch_scores(lum), 512, 768)
        np.testing.assert_allclose(split, legacy, rtol=0, atol=1e-6)

    def test_shifted_grid_moves_weight(self):
        from scipy import ndimage
        from src.registration import (compute_patch_scores,
                                      patch_scores_to_map,
                                      _patch_grid_geometry)
        rng = np.random.default_rng(1)
        H, W = 512, 768
        lum = rng.normal(1000, 5, (H, W)).astype(np.float32)
        # One sharp textured block -> one dominant patch
        lum[64:128, 64:128] += rng.normal(0, 400, (64, 64)).astype(np.float32)
        grid = compute_patch_scores(lum)
        ph, pw, _, _ = _patch_grid_geometry(H, W)
        # Shift by exactly one patch down/right
        g = ndimage.shift(grid.astype(np.float32), shift=(1.0, 1.0),
                          order=1, mode='nearest')
        m = patch_scores_to_map(g, H, W)
        iy, ix = np.unravel_index(np.argmax(grid), grid.shape)
        peak = np.unravel_index(np.argmax(m), m.shape)
        # Peak of the full-res map should sit ~one patch below/right of the
        # original patch center.
        self.assertAlmostEqual(peak[0] / ph, iy + 1, delta=1.0)
        self.assertAlmostEqual(peak[1] / pw, ix + 1, delta=1.0)


class TestSinglePrimaryLumaDenoiser(unittest.TestCase):
    """Rule 14: the advisor must not layer multiple full-frame luma
    denoisers. Precedence BM3D > MMT > wavelet > ACDNR; chroma-only steps
    are unaffected."""

    def _run(self, **flags):
        import argparse
        from src.auto_settings import _apply_quality_settings
        defaults = dict(preview_black_sigma=0.0, stack_method='sigma_clip',
                        auto_denoise_strength=True, denoise=False,
                        denoise_mmt=False, denoise_acdnr=False,
                        denoise_bm3d=False)
        defaults.update(flags)
        args = argparse.Namespace(**defaults)
        # snr below the BM3D auto-enable threshold so BM3D stays out of play
        sig = {'n_frames': 35, 'snr': 7.0, 'fwhm': 5.4, 'star_count': 25,
               'strehl': 0.3, 'dispersion': 0.5, 'median_ellipticity': 0.1,
               'dynamic_range': 100, 'concentration': 5, 'median_filling': 0.1,
               'diffuse_excess': 0.5, 'peak_excess': 5}
        _apply_quality_settings(sig, args, 'galaxy')
        return args

    def test_mmt_wins_over_acdnr_and_wavelet(self):
        a = self._run(denoise_mmt=True, denoise_acdnr=True, denoise=True)
        self.assertTrue(a.denoise_mmt)
        self.assertFalse(a.denoise_acdnr)
        self.assertFalse(a.denoise)

    def test_wavelet_wins_over_acdnr(self):
        a = self._run(denoise=True, denoise_acdnr=True)
        self.assertTrue(a.denoise)
        self.assertFalse(a.denoise_acdnr)

    def test_acdnr_alone_survives(self):
        a = self._run(denoise_acdnr=True)
        self.assertTrue(a.denoise_acdnr)

    def test_low_snr_acdnr_not_added_when_mmt_active(self):
        import argparse
        from src.auto_settings import _apply_quality_settings
        args = argparse.Namespace(preview_black_sigma=0.0,
                                  stack_method='sigma_clip',
                                  auto_denoise_strength=True, denoise=False,
                                  denoise_mmt=True, denoise_acdnr=False,
                                  denoise_bm3d=False)
        sig = {'n_frames': 35, 'snr': 2.0, 'fwhm': 5.4, 'star_count': 25,
               'strehl': 0.3, 'dispersion': 0.5, 'median_ellipticity': 0.1,
               'dynamic_range': 100, 'concentration': 5, 'median_filling': 0.1,
               'diffuse_excess': 0.5, 'peak_excess': 5}
        _apply_quality_settings(sig, args, 'galaxy')
        self.assertTrue(args.denoise_mmt)
        self.assertFalse(args.denoise_acdnr)
