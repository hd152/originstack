"""Tests for the Bortle-9-motivated gradient-anomaly quality signal:
- estimate_background_gradient (src/quality.py): per-frame gradient
  magnitude from a coarse robust plane fit.
- quality_gate's statistical-outlier stage (src/frame_processor.py): now
  also flags a frame with an anomalously *strong* gradient relative to the
  rest of the session (a cloud reflecting light-pollution glow, a light
  cycling on mid-session) -- one-sided (only above-average is suspicious,
  unlike SNR/star-count/contrast).
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np

from src.quality import estimate_background_gradient
from src.frame_processor import quality_gate


class TestEstimateBackgroundGradient:

    def test_flat_image_near_zero(self):
        img = np.full((128, 128), 500.0, dtype=np.float32)
        grad = estimate_background_gradient(img, noise=5.0)
        assert grad < 0.05

    def test_scales_with_injected_slope(self):
        H, W = 128, 128
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        noise = 5.0
        weak = 500.0 + 0.05 * xx
        strong = 500.0 + 0.5 * xx
        grad_weak = estimate_background_gradient(weak, noise=noise)
        grad_strong = estimate_background_gradient(strong, noise=noise)
        assert grad_strong > grad_weak * 5

    def test_too_small_image_returns_zero(self):
        img = np.full((8, 8), 100.0)
        assert estimate_background_gradient(img, noise=5.0) == 0.0

    def test_noise_normalization(self):
        """Same absolute gradient, different noise floor -> different
        normalised magnitude (a gradient is more 'anomalous' relative to a
        quiet session than a noisy one)."""
        H, W = 128, 128
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        img = 500.0 + 0.3 * xx
        grad_low_noise = estimate_background_gradient(img, noise=1.0)
        grad_high_noise = estimate_background_gradient(img, noise=20.0)
        assert grad_low_noise > grad_high_noise


def _frame(path, accepted=True, snr=10.0, star_count=50, contrast=20.0,
          gradient_magnitude=0.0, ellipticity=0.1, score=50.0):
    return SimpleNamespace(
        path=path, accepted=accepted,
        metrics={'snr': snr, 'star_count': star_count, 'contrast': contrast,
                'gradient_magnitude': gradient_magnitude, 'ellipticity': ellipticity,
                'score': score},
    )


def _args(**overrides):
    base = dict(verbose=False, quality_filter=True, quality_threshold=50.0,
               max_ellipticity=0.5)
    base.update(overrides)
    return argparse.Namespace(**base)


class TestQualityGateGradientAnomaly:

    def test_high_gradient_plus_one_other_signal_rejected(self):
        lights = [_frame(f'f{i}.fits') for i in range(10)]
        # One frame: normal SNR/star_count/contrast but a wild gradient AND
        # a depressed SNR (needs 2 of 4 signals to actually reject).
        lights[5] = _frame('bad.fits', snr=2.0, gradient_magnitude=50.0)
        rejected_reasons = {}
        stats = SimpleNamespace()
        final = quality_gate(lights, _args(), rejected_reasons, stats,
                             stages=('statistical',))
        assert 'bad.fits' not in [f.path for f in final]
        assert 'gradient' in rejected_reasons['bad.fits']

    def test_high_gradient_alone_not_rejected(self):
        """A single anomalous signal (gradient only, everything else
        normal) should NOT be enough to reject -- matches the existing
        2-of-N multi-signal requirement for SNR/star_count/contrast."""
        lights = [_frame(f'f{i}.fits') for i in range(10)]
        lights[5] = _frame('odd_gradient.fits', gradient_magnitude=50.0)
        rejected_reasons = {}
        stats = SimpleNamespace()
        final = quality_gate(lights, _args(), rejected_reasons, stats,
                             stages=('statistical',))
        assert 'odd_gradient.fits' in [f.path for f in final]

    def test_low_gradient_not_penalised(self):
        """A frame with a WEAKER-than-average gradient shouldn't be
        flagged -- gradient anomaly is one-sided (only excess is bad)."""
        lights = [_frame(f'f{i}.fits', gradient_magnitude=10.0) for i in range(10)]
        lights[5] = _frame('quiet.fits', gradient_magnitude=0.0, snr=2.0)
        rejected_reasons = {}
        stats = SimpleNamespace()
        final = quality_gate(lights, _args(), rejected_reasons, stats,
                             stages=('statistical',))
        # snr=2.0 alone is only 1 signal -- should survive same as the
        # gradient-alone case above (low gradient contributes 0 flags, not 1).
        assert 'quiet.fits' in [f.path for f in final]

    def test_missing_gradient_metric_defaults_safely(self):
        """Frames from a checkpoint/caller predating this metric (no
        gradient_magnitude key) shouldn't crash or get spuriously flagged."""
        lights = [SimpleNamespace(path=f'f{i}.fits', accepted=True,
                                  metrics={'snr': 10.0, 'star_count': 50,
                                          'contrast': 20.0, 'ellipticity': 0.1,
                                          'score': 50.0})
                 for i in range(10)]
        rejected_reasons = {}
        stats = SimpleNamespace()
        final = quality_gate(lights, _args(), rejected_reasons, stats,
                             stages=('statistical',))
        assert len(final) == 10


if __name__ == '__main__':
    import unittest
    unittest.main()
