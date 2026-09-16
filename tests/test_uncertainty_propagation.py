"""Uncertainty propagation through the Phase 4 chain.

The Monte Carlo propagator is validated against chains whose variance
transformation is known analytically -- identity, a pure scale, and a box
blur -- so a regression shows up as a number that disagrees with theory, not
just as a crash. The MC estimator's own standard error on a standard
deviation is ~1/sqrt(2K), so tolerances here are set from that (K is raised
where a tighter check is wanted) rather than picked by eye.
"""

import argparse
import unittest

import numpy as np

from src.uncertainty import (
    _quiet_args,
    confidence_map,
    error_aware_black_point,
    propagate_uncertainty,
    summarize_confidence,
)


def _args(**kw):
    ns = argparse.Namespace(remove_stars=True, nmf_separate=True,
                            photometric_calibration=True, verbose=True)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


class TestQuietArgs(unittest.TestCase):
    def test_disables_side_effecting_steps_without_touching_the_original(self):
        original = _args()
        quiet = _quiet_args(original)

        for name in ('remove_stars', 'nmf_separate', 'photometric_calibration', 'verbose'):
            self.assertFalse(getattr(quiet, name), f"{name} should be off in realizations")
            self.assertTrue(getattr(original, name), f"{name} must survive on the original")

        self.assertIsNone(quiet._diagnostic_dir)
        self.assertTrue(quiet._uncertainty_realization)

    def test_tolerates_args_missing_the_optional_flags(self):
        # Namespaces built by tests/older configs may not carry every flag.
        quiet = _quiet_args(argparse.Namespace())
        self.assertIsNone(quiet._diagnostic_dir)


class TestPropagateUncertainty(unittest.TestCase):
    """Chains with a known variance transformation."""

    def setUp(self):
        self.h, self.w, self.c = 24, 24, 3
        self.stacked = np.full((self.h, self.w, self.c), 100.0, dtype=np.float32)
        self.args = _args()

    def test_identity_chain_recovers_the_input_sigma(self):
        sigma_in = 4.0
        sigma_map = np.full((self.h, self.w), sigma_in, dtype=np.float32)

        sigma_post, mean_post = propagate_uncertainty(
            self.stacked, sigma_map, self.args, [], None,
            postprocess_fn=lambda img, a, f, s: img,
            n_realizations=64, seed=1)

        self.assertEqual(sigma_post.shape, (self.h, self.w))
        # Pooled over 24x24x3 pixels the mean is far tighter than per-pixel 1/sqrt(2K).
        self.assertAlmostEqual(float(sigma_post.mean()), sigma_in, delta=0.25)
        self.assertAlmostEqual(float(mean_post.mean()), 100.0, delta=0.5)

    def test_linear_scale_scales_sigma_by_the_same_factor(self):
        sigma_map = np.full((self.h, self.w), 4.0, dtype=np.float32)
        gain = 3.0

        sigma_post, _ = propagate_uncertainty(
            self.stacked, sigma_map, self.args, [], None,
            postprocess_fn=lambda img, a, f, s: img * gain,
            n_realizations=64, seed=2)

        self.assertAlmostEqual(float(sigma_post.mean()), 4.0 * gain, delta=0.8)

    def test_averaging_chain_reduces_sigma_like_sqrt_n(self):
        """A 3x3 box blur over independent noise divides sigma by 3 (sqrt of 9)."""
        sigma_map = np.full((self.h, self.w), 9.0, dtype=np.float32)

        def box_blur(img, a, f, s):
            from scipy.ndimage import uniform_filter
            return uniform_filter(img, size=(3, 3, 1), mode='reflect')

        sigma_post, _ = propagate_uncertainty(
            self.stacked, sigma_map, self.args, [], None,
            postprocess_fn=box_blur, n_realizations=64, seed=3)

        # Interior only: the reflect boundary reuses samples, so edge pixels
        # average fewer independent values and sit above the 1/3 prediction.
        interior = sigma_post[4:-4, 4:-4]
        self.assertAlmostEqual(float(interior.mean()), 9.0 / 3.0, delta=0.4)

    def test_spatially_varying_input_sigma_is_preserved(self):
        sigma_map = np.zeros((self.h, self.w), dtype=np.float32)
        sigma_map[:, :self.w // 2] = 2.0
        sigma_map[:, self.w // 2:] = 8.0

        sigma_post, _ = propagate_uncertainty(
            self.stacked, sigma_map, self.args, [], None,
            postprocess_fn=lambda img, a, f, s: img,
            n_realizations=64, seed=4)

        self.assertAlmostEqual(float(sigma_post[:, :self.w // 2].mean()), 2.0, delta=0.3)
        self.assertAlmostEqual(float(sigma_post[:, self.w // 2:].mean()), 8.0, delta=0.9)

    def test_is_reproducible_for_a_fixed_seed(self):
        sigma_map = np.full((self.h, self.w), 3.0, dtype=np.float32)
        kw = dict(postprocess_fn=lambda img, a, f, s: img, n_realizations=8, seed=7)

        a, _ = propagate_uncertainty(self.stacked, sigma_map, self.args, [], None, **kw)
        b, _ = propagate_uncertainty(self.stacked, sigma_map, self.args, [], None, **kw)

        np.testing.assert_array_equal(a, b)

    def test_accepts_a_per_channel_input_sigma(self):
        sigma_map = np.full((self.h, self.w, self.c), 5.0, dtype=np.float32)
        sigma_post, _ = propagate_uncertainty(
            self.stacked, sigma_map, self.args, [], None,
            postprocess_fn=lambda img, a, f, s: img, n_realizations=32, seed=5)
        self.assertAlmostEqual(float(sigma_post.mean()), 5.0, delta=0.5)

    def test_survives_a_minority_of_failing_realizations(self):
        sigma_map = np.full((self.h, self.w), 4.0, dtype=np.float32)
        state = {'n': 0}

        def flaky(img, a, f, s):
            state['n'] += 1
            if state['n'] % 4 == 0:
                raise RuntimeError("simulated post-processing failure")
            return img

        sigma_post, _ = propagate_uncertainty(
            self.stacked, sigma_map, self.args, [], None,
            postprocess_fn=flaky, n_realizations=32, seed=6)

        self.assertAlmostEqual(float(sigma_post.mean()), 4.0, delta=0.5)

    def test_raises_when_too_few_realizations_survive(self):
        sigma_map = np.full((self.h, self.w), 4.0, dtype=np.float32)

        def always_fails(img, a, f, s):
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            propagate_uncertainty(self.stacked, sigma_map, self.args, [], None,
                                  postprocess_fn=always_fails, n_realizations=8, seed=8)

    def test_rejects_a_chain_that_changes_geometry(self):
        """A crop makes realizations non-comparable -- fail loudly, don't misalign."""
        sigma_map = np.full((self.h, self.w), 4.0, dtype=np.float32)

        with self.assertRaises(ValueError):
            propagate_uncertainty(self.stacked, sigma_map, self.args, [], None,
                                  postprocess_fn=lambda img, a, f, s: img[:-2, :-2],
                                  n_realizations=4, seed=9)


class TestConfidenceMap(unittest.TestCase):
    def test_snr_is_signal_above_sky_in_sigma_units(self):
        img = np.full((8, 8, 3), 10.0, dtype=np.float32)
        img[4, 4] = 40.0                       # +30 above sky
        sigma = np.full((8, 8), 3.0, dtype=np.float32)

        snr = confidence_map(img, sigma)

        self.assertAlmostEqual(float(snr[4, 4]), 10.0, places=4)   # 30/3
        self.assertAlmostEqual(float(snr[0, 0]), 0.0, places=4)

    def test_explicit_background_overrides_the_median(self):
        img = np.full((8, 8, 3), 10.0, dtype=np.float32)
        sigma = np.full((8, 8), 2.0, dtype=np.float32)

        snr = confidence_map(img, sigma, background=6.0)

        self.assertAlmostEqual(float(snr[0, 0]), 2.0, places=4)     # (10-6)/2

    def test_zero_sigma_pixels_are_undefined_not_infinitely_confident(self):
        """Post-processing clamps some pixels to a constant -> sigma 0.

        Those carry no measurement, so confidence must be NaN rather than
        signal/epsilon (which would rank the pipeline's own floor artifacts
        as the most confident pixels in the frame).
        """
        img = np.full((4, 4, 3), 5.0, dtype=np.float32)
        img[0, 0] = 9.0
        sigma = np.full((4, 4), 2.0, dtype=np.float32)
        sigma[0, 0] = 0.0          # clamped by the chain

        snr = confidence_map(img, sigma)

        self.assertTrue(np.isnan(snr[0, 0]))
        self.assertTrue(np.isfinite(snr[1, 1]))

    def test_clamped_pixels_are_excluded_from_the_black_point(self):
        img = np.zeros((4, 4, 3), dtype=np.float32)
        img[0, 0] = 1000.0         # brightest, but clamped -> NaN confidence
        img[1, 1] = 50.0
        sigma = np.full((4, 4), 2.0, dtype=np.float32)
        sigma[0, 0] = 0.0
        snr = confidence_map(img, sigma, background=0.0)

        floor = error_aware_black_point(snr, img, n_sigma=3.0)

        self.assertIsNotNone(floor)
        self.assertNotAlmostEqual(floor, 1000.0, places=3)

    def test_summary_reports_fractions_above_each_threshold(self):
        snr = np.zeros((10, 10), dtype=np.float32)
        snr[:5] = 6.0          # half the pixels clear both 3 and 5
        text = summarize_confidence(snr)
        self.assertIn("50.0%", text)
        self.assertNotIn("clamped", text)

    def test_summary_reports_the_clamped_fraction_separately(self):
        snr = np.zeros((10, 10), dtype=np.float32)
        snr[:2] = np.nan       # 20% clamped
        snr[2:6] = 6.0         # 40 of 80 measured pixels clear both thresholds
        text = summarize_confidence(snr)
        self.assertIn("20.0% clamped", text)
        self.assertIn(">3sigma 50.0%", text)   # of the measured pixels, not of all

    def test_summary_handles_an_entirely_clamped_frame(self):
        snr = np.full((4, 4), np.nan, dtype=np.float32)
        self.assertIn("no measurable pixels", summarize_confidence(snr))


class TestErrorAwareBlackPoint(unittest.TestCase):
    def test_returns_the_faintest_pixel_clearing_the_threshold(self):
        img = np.zeros((4, 4, 3), dtype=np.float32)
        img[0, 0] = 100.0
        img[1, 1] = 50.0
        img[2, 2] = 10.0
        snr = np.zeros((4, 4), dtype=np.float32)
        snr[0, 0] = 9.0
        snr[1, 1] = 4.0        # clears 3 sigma, and is the faintest that does
        snr[2, 2] = 1.0        # below threshold

        floor = error_aware_black_point(snr, img, n_sigma=3.0)

        self.assertAlmostEqual(floor, 50.0, places=4)

    def test_returns_none_when_nothing_clears_the_threshold(self):
        img = np.ones((4, 4, 3), dtype=np.float32)
        snr = np.full((4, 4), 0.5, dtype=np.float32)
        self.assertIsNone(error_aware_black_point(snr, img, n_sigma=3.0))


if __name__ == '__main__':
    unittest.main()
