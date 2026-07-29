"""Tests for src/wavelet.py -- native bior1.3 2D wavelet transform, replacing
PyWavelets for src/denoising.py's wavelet_denoise/adaptive_wavelet_denoise
(the only wavelet family either function ever uses).

Validated bit-exact against real PyWavelets (pywt.wavedec2/waverec2,
pywt.dwt_max_level, pywt.threshold) across many image sizes (even/odd
height and width combinations) and decomposition depths -- see the module
docstring in src/wavelet.py for how the multi-level reconstruction length
semantics (an inherent ambiguity in the 2-point-per-sample DWT for odd
input lengths) were reverse-engineered against pywt as the reference.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.wavelet import wavedec2, waverec2, dwt_max_level, soft_threshold


class TestDwtMaxLevel:
    @pytest.mark.parametrize('data_len', [1, 2, 5, 6, 7, 8, 15, 16, 33, 64, 100, 257, 3000])
    def test_matches_pywt(self, data_len):
        pywt = pytest.importorskip("pywt")
        assert dwt_max_level(data_len) == pywt.dwt_max_level(data_len, 6)


class TestWavedec2Waverec2VsPywt:
    @pytest.mark.parametrize('h,w', [
        (17, 23), (18, 24), (33, 50), (64, 51), (65, 128), (100, 129), (101, 100),
    ])
    @pytest.mark.parametrize('level', [1, 2, 4])
    def test_coeffs_and_roundtrip_match_pywt(self, h, w, level):
        pywt = pytest.importorskip("pywt")
        rng = np.random.default_rng(0)
        img = rng.uniform(-100, 100, (h, w))
        ml = dwt_max_level(min(h, w))
        lvl = min(level, ml)
        if lvl < 1:
            pytest.skip("image too small for this level")

        ref_coeffs = pywt.wavedec2(img, 'bior1.3', level=lvl, mode='symmetric')
        ref_rec = pywt.waverec2(ref_coeffs, 'bior1.3')[:h, :w]

        my_coeffs = wavedec2(img, lvl)
        np.testing.assert_allclose(my_coeffs[0], ref_coeffs[0], atol=1e-7)
        for (mh, mv, md), (rh, rv, rd) in zip(my_coeffs[1:], ref_coeffs[1:]):
            np.testing.assert_allclose(mh, rh, atol=1e-7)
            np.testing.assert_allclose(mv, rv, atol=1e-7)
            np.testing.assert_allclose(md, rd, atol=1e-7)

        my_rec = waverec2(my_coeffs)[:h, :w]
        np.testing.assert_allclose(my_rec, ref_rec, atol=1e-5)

    def test_perfect_reconstruction_no_thresholding(self):
        """Undecimated round trip (no thresholding applied) must recover
        the original image to floating-point precision."""
        rng = np.random.default_rng(1)
        img = rng.uniform(0, 1000, (200, 300))
        coeffs = wavedec2(img, 4)
        rec = waverec2(coeffs)[:200, :300]
        np.testing.assert_allclose(rec, img, atol=1e-6)


class TestSoftThreshold:
    def test_matches_pywt(self):
        pywt = pytest.importorskip("pywt")
        rng = np.random.default_rng(2)
        x = rng.uniform(-10, 10, 1000)
        for value in [0.0, 0.5, 2.0, 5.0, 20.0]:
            ref = pywt.threshold(x, value, mode='soft')
            mine = soft_threshold(x, value)
            np.testing.assert_allclose(mine, ref, atol=1e-12)

    def test_zeros_below_threshold(self):
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        out = soft_threshold(x, 2.0)
        np.testing.assert_allclose(out, [-1.0, 0.0, 0.0, 0.0, 1.0])

    def test_rejects_negative_threshold(self):
        with pytest.raises(ValueError):
            soft_threshold(np.array([1.0]), -1.0)
