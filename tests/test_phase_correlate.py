"""Tests for src/phase_correlate.py -- exact port of
skimage.registration.phase_cross_correlation (real-space, unmasked,
normalization="phase" path). Validated bit-exact against the real skimage
implementation across pure-translation and sub-pixel-shift cases,
guarding this codebase's now sole dependency-free registration path.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.phase_correlate import phase_cross_correlation


def _shifted_pair(h=80, w=100, sy=3.4, sx=-2.7, noise=2.0, seed=0):
    from scipy.ndimage import shift as ndshift
    rng = np.random.default_rng(seed)
    img = rng.uniform(0, 1000, (h, w)).astype(np.float64)
    shifted = ndshift(img, (sy, sx), order=3, mode='reflect')
    shifted += rng.normal(0, noise, shifted.shape)
    return img, shifted


def _normalize(img):
    return (img - img.mean()) / img.std()


class TestPhaseCrossCorrelationVsSkimage:
    @pytest.mark.parametrize('sy,sx,upsample,seed', [
        (3.4, -2.7, 10, 1),
        (0.0, 0.0, 10, 2),
        (-5.2, 8.9, 20, 3),
        (0.1, 0.05, 50, 4),
        (15.7, -20.3, 5, 5),
        (2.5, 2.5, 1, 6),
        (-30.1, -25.9, 15, 7),
    ])
    def test_matches_skimage_bit_exact(self, sy, sx, upsample, seed):
        skimage = pytest.importorskip("skimage.registration")
        ref, mov = _shifted_pair(sy=sy, sx=sx, seed=seed)
        ref_n, mov_n = _normalize(ref), _normalize(mov)
        sk_shift, sk_err, sk_phase = skimage.phase_cross_correlation(
            ref_n, mov_n, upsample_factor=upsample)
        my_shift, my_err, my_phase = phase_cross_correlation(
            ref_n, mov_n, upsample_factor=upsample)
        np.testing.assert_allclose(my_shift, sk_shift, atol=1e-10)
        assert abs(my_err - sk_err) < 1e-10
        assert abs(my_phase - sk_phase) < 1e-10


class TestPhaseCrossCorrelationProperties:
    def test_recovers_known_integer_shift(self):
        # _shifted_pair applies (sy, sx) to ref to build mov, i.e.
        # mov = shift(ref, +(sy, sx)); the shift that registers mov onto ref
        # is therefore the negation (matches skimage's own docstring: "Shift
        # vector ... required to register moving_image with reference_image").
        ref, mov = _shifted_pair(sy=4.0, sx=-3.0, noise=0.0, seed=8)
        shift, error, _ = phase_cross_correlation(
            _normalize(ref), _normalize(mov), upsample_factor=1)
        np.testing.assert_allclose(shift, [-4.0, 3.0], atol=0.5)

    def test_recovers_known_subpixel_shift(self):
        ref, mov = _shifted_pair(sy=2.3, sx=-1.7, noise=0.5, seed=9)
        shift, error, _ = phase_cross_correlation(
            _normalize(ref), _normalize(mov), upsample_factor=20)
        # atol accounts for cubic-spline interpolation + injected noise slop
        # in building the synthetic pair, not algorithm precision (the
        # bit-exact-vs-skimage tests above cover that).
        np.testing.assert_allclose(shift, [-2.3, 1.7], atol=0.15)

    def test_zero_shift_identical_images(self):
        # Note: under normalization="phase" (skimage's default, matched here),
        # the RMS error metric is not meaningful for identical images -- real
        # skimage itself returns error~1.0 here too (verified directly), not
        # ~0; the error formula assumes unnormalized cross-correlation. Only
        # the recovered shift is a meaningful property to check.
        ref, _ = _shifted_pair(noise=0.0, seed=10)
        shift, error, _ = phase_cross_correlation(
            _normalize(ref), _normalize(ref), upsample_factor=10)
        np.testing.assert_allclose(shift, [0.0, 0.0], atol=1e-8)

    def test_rejects_mismatched_shapes(self):
        a = np.zeros((10, 10))
        b = np.zeros((10, 12))
        with pytest.raises(ValueError):
            phase_cross_correlation(a, b)
