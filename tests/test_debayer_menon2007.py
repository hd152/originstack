"""Tests for src/debayer.py's Menon (2007) DDFAPD demosaicing.

The numpy kernel (_debayer_menon2007_numpy) was validated bit-exact (float32
rounding only) against the `colour-demosaicing` package's reference
Menon2007 implementation across all 4 Bayer patterns, with both refining_step
on and off. That comparison is reproduced here and skips gracefully if the
reference package isn't installed (validation-only dependency, not a runtime
requirement -- see tests/test_debayer_malvar.py for the same pattern applied
to the Malvar kernel).
"""
from __future__ import annotations

import numpy as np
import pytest

import src.debayer as db


def _synthetic_raw(h=48, w=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1000, (h, w)).astype(np.float32)


class TestMenon2007AgainstReference:
    @pytest.mark.parametrize('pattern', ['RGGB', 'BGGR', 'GRBG', 'GBRG'])
    @pytest.mark.parametrize('refining_step', [True, False])
    def test_matches_colour_demosaicing_menon2007(self, pattern, refining_step):
        try:
            from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
        except ImportError:
            pytest.skip("colour-demosaicing not installed (validation-only dependency)")
        raw = _synthetic_raw(seed=hash((pattern, refining_step)) % 1000)
        ours = db._debayer_menon2007_numpy(raw, pattern, refining_step=refining_step)
        ref = demosaicing_CFA_Bayer_Menon2007(
            raw.astype(np.float64), pattern, refining_step=refining_step)
        m = 8  # interior only -- boundary handling isn't the point of this comparison
        np.testing.assert_allclose(
            ours[m:-m, m:-m, :], ref[m:-m, m:-m, :], atol=1e-2)


class TestMenon2007Properties:
    def test_output_shape_and_dtype(self):
        raw = _synthetic_raw()
        out = db._debayer_menon2007_numpy(raw, 'RGGB')
        assert out.shape == (48, 64, 3)
        assert out.dtype == np.float32

    def test_rejects_unknown_pattern(self):
        with pytest.raises(ValueError):
            db._debayer_menon2007_numpy(_synthetic_raw(), 'XYZW')

    def test_constant_image_reconstructed_exactly(self):
        raw = np.full((40, 50), 500.0, dtype=np.float32)
        out = db._debayer_menon2007_numpy(raw, 'RGGB')
        np.testing.assert_allclose(out, 500.0, atol=1e-2)

    def test_dispatch_routes_to_menon2007(self):
        raw = _synthetic_raw()
        out = db.debayer(raw, pattern='RGGB', method='menon2007')
        assert out.shape == (48, 64, 3)

    def test_higher_fidelity_than_bilinear_on_sharp_edge(self):
        h, w = 60, 60
        truth = np.zeros((h, w), dtype=np.float64)
        truth[:, w // 2:] = 1000.0
        mosaic = truth.copy().astype(np.float32)
        bilinear = db.debayer_bilinear(mosaic, 'RGGB')[:, :, 1]
        menon = db._debayer_menon2007_numpy(mosaic, 'RGGB')[:, :, 1]
        m = 10
        err_bilinear = np.abs(bilinear[m:-m, m:-m] - truth[m:-m, m:-m]).mean()
        err_menon = np.abs(menon[m:-m, m:-m] - truth[m:-m, m:-m]).mean()
        assert err_menon <= err_bilinear + 1e-6


class TestMenon2007NativeNumpyParity:
    def test_native_matches_numpy_when_available(self):
        if not db._HAS_NATIVE or not hasattr(db._native, 'debayer_menon2007'):
            pytest.skip("astro_native debayer_menon2007 kernel not built")
        raw = _synthetic_raw(seed=3)
        native_out = db._native.debayer_menon2007(np.ascontiguousarray(raw), 'RGGB')
        numpy_out = db._debayer_menon2007_numpy(raw, 'RGGB')
        np.testing.assert_allclose(native_out, numpy_out, atol=1e-2)
