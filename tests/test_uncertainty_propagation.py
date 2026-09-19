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
    _QUIET_OFF,
    _quiet_args,
    confidence_map,
    error_aware_black_point,
    propagate_uncertainty,
    summarize_confidence,
)


def _args(**kw):
    ns = argparse.Namespace(**{name: True for name in _QUIET_OFF})
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


class TestQuietArgs(unittest.TestCase):
    def test_disables_side_effecting_steps_without_touching_the_original(self):
        original = _args()
        quiet = _quiet_args(original)

        for name in _QUIET_OFF:
            self.assertFalse(getattr(quiet, name), f"{name} should be off in realizations")
            self.assertTrue(getattr(original, name), f"{name} must survive on the original")

        self.assertIsNone(quiet._diagnostic_dir)

    def test_every_phase4_file_writing_flag_is_quieted(self):
        """The list is hand-maintained, so pin what it must contain.

        Each of these makes postprocess_stack write a sidecar. Left enabled,
        it is rewritten once per realization under a swallowed stdout, and the
        file left on disk is the last *noise realization* rather than the real
        image -- silently, since the "Saved:" line goes into the buffer too.
        """
        for name in ('remove_stars', 'nmf_separate', 'aberration_report',
                     'diagnostic', 'export_masks', 'keep_intermediates',
                     'comet_radial_renorm', 'comet_larson_sekanina'):
            self.assertIn(name, _QUIET_OFF,
                          f"{name} writes a sidecar and must not run K times")

    def test_network_steps_are_quieted(self):
        # Gaia/VizieR queries (K of them) -- they don't shape the noise field.
        self.assertIn('photometric_calibration', _QUIET_OFF)
        self.assertIn('annotate', _QUIET_OFF)

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

        sigma_post, mean_post, _ = propagate_uncertainty(
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

        sigma_post, _, _ = propagate_uncertainty(
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

        sigma_post, _, _ = propagate_uncertainty(
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

        sigma_post, _, _ = propagate_uncertainty(
            self.stacked, sigma_map, self.args, [], None,
            postprocess_fn=lambda img, a, f, s: img,
            n_realizations=64, seed=4)

        self.assertAlmostEqual(float(sigma_post[:, :self.w // 2].mean()), 2.0, delta=0.3)
        self.assertAlmostEqual(float(sigma_post[:, self.w // 2:].mean()), 8.0, delta=0.9)

    def test_is_reproducible_for_a_fixed_seed(self):
        sigma_map = np.full((self.h, self.w), 3.0, dtype=np.float32)
        kw = dict(postprocess_fn=lambda img, a, f, s: img, n_realizations=8, seed=7)

        a, _, _ = propagate_uncertainty(self.stacked, sigma_map, self.args, [], None, **kw)
        b, _, _ = propagate_uncertainty(self.stacked, sigma_map, self.args, [], None, **kw)

        np.testing.assert_array_equal(a, b)

    def test_accepts_a_per_channel_input_sigma(self):
        sigma_map = np.full((self.h, self.w, self.c), 5.0, dtype=np.float32)
        sigma_post, _, _ = propagate_uncertainty(
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

        sigma_post, _, _ = propagate_uncertainty(
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


class TestNoiseAdaptiveChainBias(unittest.TestCase):
    """The case the three analytic chains above structurally cannot catch.

    Identity, a pure scale and a box blur are all *linear*, so the fact that a
    realization carries sqrt(2)x the real noise (``stacked`` already holds
    ~sigma of its own, and propagation adds another sigma on top) cancels
    exactly and the estimator looks exact. Most of this project's Phase 4 is
    not linear in that sense: BayesShrink reads its threshold off each
    subband's measured noise, DBE and the sky-floor passes measure sky sigma,
    ``estimate_denoise_strength`` keys off SNR. Handed an inflated
    realization, each denoises *harder* than it did on the real image, and the
    spread that comes back understates the truth.

    These tests pin the bias against this project's *real* wavelet denoiser,
    not a toy, because the size of the effect is easy to get wrong from first
    principles -- see ``test_bias_against_the_real_wavelet_denoiser``.
    """

    def setUp(self):
        self.h, self.w, self.c = 24, 24, 3
        self.stacked_clean = np.full((self.h, self.w, self.c), 100.0, dtype=np.float32)
        self.sigma_in = 4.0
        self.args = _args()

    @staticmethod
    def _structured_field(h, w, c):
        """Sky gradient + gaussian blobs -- content a denoiser reacts to.

        A flat field is the wrong fixture here: at a 3-sigma threshold almost
        every pixel clips to the median, both spreads collapse toward zero,
        and the test passes while measuring nothing.
        """
        yy, xx = np.mgrid[0:h, 0:w]
        img = (100.0 + 0.05 * xx + 0.03 * yy).astype(np.float32)
        for cy, cx, amp, sd in [(8, 8, 60, 3), (16, 18, 35, 4)]:
            img = img + amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sd * sd))
        return np.repeat(img[:, :, None], c, axis=2).astype(np.float32)

    def test_bias_against_the_real_wavelet_denoiser(self):
        """Propagated vs true output sigma for an actually-adaptive chain.

        Ground truth is the spread of the chain's output at the *real*
        operating point -- independent noisy copies of the clean image, each
        carrying exactly sigma -- against propagation's realizations, which
        sit on top of an already-noisy stack and so carry sqrt(2)*sigma.

        The measured ratio when this was written was 0.93-1.03 across sigma in
        {1, 4, 12} and threshold_factor in {2, 3, 5}. The assertion is
        deliberately loose around that: the point is to catch a *mechanism*
        change (a denoiser whose adaptive response is strong enough to bias
        the estimate seriously), not to pin one machine's arithmetic.
        """
        from src.denoising import directional_wavelet_denoise

        h = w = 48
        clean = self._structured_field(h, w, self.c)
        rng = np.random.default_rng(11)

        def chain(img, a=None, f=None, s=None):
            return directional_wavelet_denoise(np.asarray(img, dtype=np.float32),
                                               levels=3)

        stacked = clean + rng.standard_normal(clean.shape).astype(np.float32) * self.sigma_in
        sigma_map = np.full((h, w), self.sigma_in, dtype=np.float32)

        propagated, _, adaptivity = propagate_uncertainty(
            stacked, sigma_map, self.args, [], None,
            postprocess_fn=chain, n_realizations=24, seed=12,
            probe_realizations=0)

        truth = float(np.stack([
            chain(clean + rng.standard_normal(clean.shape).astype(np.float32) * self.sigma_in)
            for _ in range(24)]).std(axis=0).mean())

        ratio = float(propagated.mean()) / truth
        self.assertGreater(ratio, 0.80,
                           f"propagated sigma understates by more than the measured "
                           f"few percent (ratio {ratio:.3f}); an adaptive step's "
                           f"response has changed and the module docstring's "
                           f"accuracy claim needs re-measuring")
        self.assertLess(ratio, 1.20,
                        f"propagated sigma now OVER-states (ratio {ratio:.3f}); "
                        f"the documented bias direction has flipped")

    def test_adaptivity_probe_reports_two_for_a_scale_invariant_chain(self):
        """A linear chain must NOT be flagged -- otherwise the warning is noise."""
        sigma_map = np.full((self.h, self.w), self.sigma_in, dtype=np.float32)

        _, _, adaptivity = propagate_uncertainty(
            self.stacked_clean, sigma_map, self.args, [], None,
            postprocess_fn=lambda img, a, f, s: img * 3.0,
            n_realizations=32, seed=13, probe_realizations=16)

        self.assertIsNotNone(adaptivity)
        self.assertAlmostEqual(adaptivity, 2.0, delta=0.25)
        # ... and therefore sits above the threshold pipeline.py warns at, so
        # a linear chain never produces the "adapts strongly" note.
        self.assertGreater(adaptivity, 1.5)

    def test_probe_can_be_disabled(self):
        sigma_map = np.full((self.h, self.w), self.sigma_in, dtype=np.float32)
        _, _, adaptivity = propagate_uncertainty(
            self.stacked_clean, sigma_map, self.args, [], None,
            postprocess_fn=lambda img, a, f, s: img,
            n_realizations=4, seed=14, probe_realizations=0)
        self.assertIsNone(adaptivity)


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


class TestFlagValidation(unittest.TestCase):
    """Arguments whose bad values used to be accepted and acted on silently."""

    def _parse(self, *extra):
        from src.cli import build_parser
        return build_parser().parse_args(['-d', 'somewhere', *extra])

    def _rejects(self, *extra):
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self._parse(*extra)

    def test_defaults_are_unchanged(self):
        args = self._parse()
        self.assertEqual(args.uncertainty_realizations, 8)
        self.assertEqual(args.transient_threshold, 5.0)

    def test_a_valid_realization_count_is_accepted(self):
        self.assertEqual(self._parse('--uncertainty-realizations', '16').uncertainty_realizations, 16)

    def test_fewer_than_two_realizations_is_rejected(self):
        """propagate_uncertainty used to clamp these to 2 without saying so, so
        `--uncertainty-realizations 0` quietly ran two passes."""
        for bad in ('0', '1', '-4', 'many', '2.5'):
            with self.subTest(value=bad):
                self._rejects('--uncertainty-realizations', bad)

    def test_a_positive_threshold_is_accepted(self):
        self.assertEqual(self._parse('--transient-threshold', '3.5').transient_threshold, 3.5)

    def test_a_non_positive_threshold_is_rejected(self):
        """Zero admits every pixel and reports the 500 noisiest as detections."""
        for bad in ('0', '0.0', '-2', 'nan', 'high'):
            with self.subTest(value=bad):
                self._rejects('--transient-threshold', bad)

    def test_the_help_text_states_the_measured_caveat_and_the_memory_cost(self):
        from src.cli import build_parser
        text = ' '.join(next(a for a in build_parser()._actions
                            if '--uncertainty-propagate' in a.option_strings).help.split())
        self.assertNotIn('exact for the nonlinear', text,
                         "the 'exact' claim was corrected -- see uncertainty.py")
        self.assertIn('biased', text)
        self.assertIn('GB', text)
