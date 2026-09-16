"""Tests for the optional generalized-Anscombe VST path in wavelet denoising.

generalized_anscombe_forward/_inverse stabilize Poisson(shot)+Gaussian(read)
noise variance before/after BayesShrink thresholding, only when a detector
``gain`` is supplied to wavelet_denoise/adaptive_wavelet_denoise -- these
tests check the transform round-trips closely on a clean (noise-free) signal,
that gain=None (default) is unaffected, and that the transform actually
delivers the property it exists for: noise variance in the transformed
domain is approximately level-independent, unlike the raw ADU domain where
Poisson noise scales with signal brightness. (An end-to-end "lower MSE
against ground truth" claim was tried and dropped: VST trades noise-variance
uniformity for some compression of the signal's own gradient structure, and
for smooth extended content -- as opposed to point-like/sparse content --
that can be a wash or net worse by plain MSE even though the noise model is
now correctly calibrated. The variance-stabilization property itself is the
reliable, always-true thing to assert.)
"""
from __future__ import annotations

import unittest

import numpy as np

from src.denoising import (
    generalized_anscombe_forward,
    generalized_anscombe_inverse,
    wavelet_denoise,
)


class TestGeneralizedAnscombeRoundtrip(unittest.TestCase):

    def test_roundtrip_close_on_clean_signal(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(1.0, 1000.0, 2000)
        gain, read_noise = 1.5, 4.0
        z = generalized_anscombe_forward(x, gain, read_noise)
        x_hat = generalized_anscombe_inverse(z, gain, read_noise)
        # The "unbiased" inverse is tuned for noisy z, not an exact algebraic
        # inverse of a deterministic value -- allow a small relative slack.
        np.testing.assert_allclose(x_hat, x, rtol=0.02, atol=0.5)

    def test_forward_nonnegative_and_finite(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(-50.0, 1000.0, 2000)  # includes negative (chroma-plane-like) values
        z = generalized_anscombe_forward(x, gain=2.0, read_noise=5.0)
        self.assertTrue(np.all(np.isfinite(z)))
        self.assertTrue(np.all(z >= 0.0))


class TestWaveletDenoiseVstDefaultUnaffected(unittest.TestCase):

    def test_gain_none_matches_pre_vst_call_signature(self):
        rng = np.random.default_rng(2)
        img = rng.uniform(0, 500, (32, 32, 3)).astype(np.float32)
        out_default = wavelet_denoise(img, threshold_factor=3.0, chroma_factor=2.0)
        out_explicit_none = wavelet_denoise(img, threshold_factor=3.0, chroma_factor=2.0,
                                            gain=None, read_noise=5.0)
        np.testing.assert_array_equal(out_default, out_explicit_none)


class TestGeneralizedAnscombeStabilizesVariance(unittest.TestCase):
    """The actual guarantee GAT provides: post-transform noise variance is
    approximately constant across signal levels, unlike raw ADU where
    Poisson shot noise scales with brightness. This is what lets a single
    global (or per-subband) sigma estimate be valid everywhere, which is the
    real justification for wiring it into wavelet_denoise/
    adaptive_wavelet_denoise -- independent of whether any particular scene's
    end-to-end MSE improves."""

    def test_variance_uniform_across_brightness_levels_with_read_noise(self):
        rng = np.random.default_rng(3)
        gain, read_noise = 2.0, 5.0
        n = 100_000
        variances_y = []
        variances_z = []
        for true_adu in (30.0, 100.0, 430.0):
            electrons = gain * true_adu
            noisy_e = rng.poisson(electrons, n).astype(np.float64)
            y = noisy_e / gain + rng.normal(0.0, read_noise / gain, n)
            z = generalized_anscombe_forward(y, gain, read_noise)
            variances_y.append(np.var(y))
            variances_z.append(np.var(z))

        # Raw ADU variance should scale strongly with brightness (Poisson);
        # GAT variance should stay close to 1 regardless of brightness.
        self.assertGreater(max(variances_y) / min(variances_y), 5.0)
        for v in variances_z:
            self.assertAlmostEqual(v, 1.0, delta=0.05)


if __name__ == '__main__':
    unittest.main()
