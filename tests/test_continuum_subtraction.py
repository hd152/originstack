"""Tests for quantitative narrowband continuum subtraction via Skewness
Transition Analysis (src/channel_combine.py::optimal_continuum_scale,
subtract_continuum).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.channel_combine import optimal_continuum_scale, subtract_continuum


def _synthetic_pair(true_scale=1.4, seed=0, h=96, w=96):
    """Continuum reference with several star-like Gaussian blobs (shared
    with the narrowband image, scaled by true_scale), plus the narrowband
    image gets extra 'emission' blobs at different locations the continuum
    doesn't have -- so the correct subtraction scale removes exactly the
    shared star signal and leaves only the emission + noise.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    # No flat pedestal: a real continuum reference used for this technique
    # is already sky-subtracted, so "sky" pixels carry ~zero correlated
    # signal in either image beyond noise -- a uniform baseline here would
    # confound the fit with a whole-frame additive term unrelated to the
    # star-scaling this method is actually meant to solve for.
    continuum = np.zeros((h, w))
    star_centers = rng.uniform(10, h - 10, (6, 2))
    for cy, cx in star_centers:
        continuum += 300.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 6.0 ** 2))

    narrowband = true_scale * continuum.copy()
    emission_centers = rng.uniform(10, h - 10, (3, 2))
    for cy, cx in emission_centers:
        narrowband += 150.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 15.0 ** 2))

    narrowband += rng.normal(0, 2.0, (h, w))
    continuum += rng.normal(0, 2.0, (h, w))
    return narrowband.astype(np.float32), continuum.astype(np.float32), true_scale


class TestOptimalContinuumScale:

    def test_recovers_approximate_true_scale(self):
        narrowband, continuum, true_scale = _synthetic_pair(true_scale=1.4)
        best_scale, diag = optimal_continuum_scale(
            narrowband, continuum, scale_range=(0.0, 3.0), n_steps=80)
        assert abs(best_scale - true_scale) < 0.5
        assert 'scales' in diag and 'skewness' in diag

    def test_fitted_scale_is_the_skewness_peak(self):
        # The fitted scale must actually be at (or immediately next to) the
        # maximum of the swept skewness curve -- skewness one grid step
        # away in either direction should be no higher than at the fit.
        narrowband, continuum, _ = _synthetic_pair(true_scale=1.2, seed=1)
        best_scale, diag = optimal_continuum_scale(narrowband, continuum, n_steps=80)
        scales, skews = diag['scales'], diag['skewness']
        valid = np.isfinite(skews)
        scales_v, skews_v = scales[valid], skews[valid]
        peak_idx = int(np.argmax(skews_v))
        assert abs(best_scale - scales_v[peak_idx]) < 2 * (scales_v[1] - scales_v[0])

    def test_shape_mismatch_raises(self):
        narrowband = np.zeros((10, 10))
        continuum = np.zeros((5, 5))
        with pytest.raises(ValueError):
            optimal_continuum_scale(narrowband, continuum)

    def test_degenerate_flat_input_does_not_crash(self):
        flat_nb = np.full((20, 20), 100.0)
        flat_cont = np.full((20, 20), 50.0)
        scale, diag = optimal_continuum_scale(flat_nb, flat_cont, n_steps=10)
        assert np.isfinite(scale)


class TestSubtractContinuum:

    def test_auto_fit_matches_optimal_continuum_scale(self):
        narrowband, continuum, _ = _synthetic_pair(seed=2)
        result, scale_used, diag = subtract_continuum(narrowband, continuum)
        expected_scale, _ = optimal_continuum_scale(narrowband, continuum)
        assert abs(scale_used - expected_scale) < 1e-9
        assert result.shape == narrowband.shape
        assert np.all(result >= 0.0)

    def test_manual_scale_skips_fit(self):
        narrowband, continuum, _ = _synthetic_pair(seed=3)
        result, scale_used, diag = subtract_continuum(narrowband, continuum, scale=2.0)
        assert scale_used == 2.0
        assert diag == {}
        expected = np.clip(narrowband.astype(np.float64) - 2.0 * continuum.astype(np.float64), 0, None)
        np.testing.assert_allclose(result, expected.astype(np.float32))

    def test_result_clipped_at_zero(self):
        narrowband = np.full((10, 10), 5.0, dtype=np.float32)
        continuum = np.full((10, 10), 100.0, dtype=np.float32)
        result, _, _ = subtract_continuum(narrowband, continuum, scale=1.0)
        assert np.all(result == 0.0)
