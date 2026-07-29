"""Tests for src/debayer.py's Malvar-He-Cutler demosaicing.

The numpy kernel/mask logic (_debayer_malvar_numpy) was validated bit-exact
(float32 rounding only) against the `colour-demosaicing` package's reference
Malvar2004 implementation across all 4 Bayer patterns, with
_equalize_bayer_grid disabled for that comparison (it's this codebase's own
post-hoc checkerboard-bias correction, not part of the published algorithm).
That comparison is reproduced here and skips gracefully if the reference
package isn't installed (it's a validation-only dependency, not a runtime
requirement -- see src/debayer.py's HAS_CV2-free Malvar path).
"""
from __future__ import annotations

import numpy as np
import pytest

import src.debayer as db


def _synthetic_raw(h=64, w=80, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1000, (h, w)).astype(np.float32)


class TestMalvarAgainstReference:
    @pytest.mark.parametrize('pattern', ['RGGB', 'BGGR', 'GRBG', 'GBRG'])
    def test_matches_colour_demosaicing_malvar2004(self, pattern, monkeypatch):
        try:
            from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004
        except ImportError:
            pytest.skip("colour-demosaicing not installed (validation-only dependency)")
        monkeypatch.setattr(db, '_equalize_bayer_grid', lambda x: x)
        raw = _synthetic_raw(seed=hash(pattern) % 1000)
        ours = db._debayer_malvar_numpy(raw, pattern)
        ref = demosaicing_CFA_Bayer_Malvar2004(raw.astype(np.float64), pattern)
        m = 6  # interior only -- boundary mode differs (mirror vs reflect)
        np.testing.assert_allclose(
            ours[m:-m, m:-m, :], ref[m:-m, m:-m, :], atol=1e-3)


class TestMalvarProperties:
    def test_constant_image_reconstructed_exactly(self):
        # All 4 kernels sum to 8/8=1, so a flat field must debayer flat --
        # a strong sanity check independent of the reference package.
        raw = np.full((40, 50), 500.0, dtype=np.float32)
        out = db._debayer_malvar_numpy(raw, 'RGGB')
        np.testing.assert_allclose(out, 500.0, atol=1e-3)

    def test_output_shape_and_dtype(self):
        raw = _synthetic_raw()
        out = db._debayer_malvar_numpy(raw, 'RGGB')
        assert out.shape == (64, 80, 3)
        assert out.dtype == np.float32

    def test_rejects_unknown_pattern(self):
        with pytest.raises(ValueError):
            db._debayer_malvar_numpy(_synthetic_raw(), 'XYZW')

    def test_higher_fidelity_than_bilinear_on_sharp_edge(self):
        # A hard step edge is the classic case where Malvar's gradient-aware
        # kernels should out-perform plain bilinear interpolation.
        h, w = 60, 60
        truth = np.zeros((h, w), dtype=np.float64)
        truth[:, w // 2:] = 1000.0
        yy, xx = np.mgrid[0:h, 0:w]
        raw = np.where((yy % 2 == 0) & (xx % 2 == 0), truth, truth)  # mosaic sampling below
        # Build an RGGB mosaic sampling of a per-channel-identical scene.
        mosaic = truth.copy().astype(np.float32)
        bilinear = db.debayer_bilinear(mosaic, 'RGGB')[:, :, 1]
        malvar = db._debayer_malvar_numpy(mosaic, 'RGGB')[:, :, 1]
        m = 10
        err_bilinear = np.abs(bilinear[m:-m, m:-m] - truth[m:-m, m:-m]).mean()
        err_malvar = np.abs(malvar[m:-m, m:-m] - truth[m:-m, m:-m]).mean()
        assert err_malvar <= err_bilinear + 1e-6

    def test_dispatch_routes_to_malvar(self):
        raw = _synthetic_raw()
        out = db.debayer(raw, pattern='RGGB', method='malvar')
        assert out.shape == (64, 80, 3)


class TestMalvarNativeNumpyParity:
    def test_native_matches_numpy_when_available(self):
        if not db._HAS_NATIVE or not hasattr(db._native, 'debayer_malvar'):
            pytest.skip("astro_native debayer_malvar kernel not built")
        raw = _synthetic_raw(seed=3)
        native_out = db._native.debayer_malvar(np.ascontiguousarray(raw), 'RGGB')
        numpy_out = db._debayer_malvar_numpy(raw, 'RGGB')
        np.testing.assert_allclose(native_out, numpy_out, atol=1e-3)
