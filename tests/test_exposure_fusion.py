"""Tests for Mertens multiresolution exposure fusion (src/exposure_fusion.py),
used by --hdr-combine --hdr-blend-mode fusion as an alternative to the
original sigmoid-threshold HDR blend.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.exposure_fusion import _laplacian_pyramid, _quality_weights, _reconstruct_from_laplacian, fuse_exposures


def _synthetic_pair(h=64, w=64):
    """A long exposure with a saturated core + a short exposure that isn't."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    core = np.exp(-((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * 6.0 ** 2))
    long_exp = np.stack([100.0 + 900.0 * core] * 3, axis=-1).astype(np.float32)
    long_exp = np.clip(long_exp, 0, 1000.0)  # saturates the core at 1000
    short_exp = np.stack([20.0 + 180.0 * core] * 3, axis=-1).astype(np.float32)  # unsaturated
    return long_exp, short_exp


class TestFuseExposures:

    def test_output_shape_and_dtype(self):
        long_exp, short_exp = _synthetic_pair()
        out = fuse_exposures([long_exp, short_exp])
        assert out.shape == long_exp.shape
        assert out.dtype == np.float32

    def test_finite_and_nonnegative(self):
        long_exp, short_exp = _synthetic_pair()
        out = fuse_exposures([long_exp, short_exp])
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0.0)

    def test_requires_at_least_two_images(self):
        long_exp, _ = _synthetic_pair()
        with pytest.raises(ValueError):
            fuse_exposures([long_exp])

    def test_small_image_does_not_crash(self):
        # Fewer pixels than the default 6 pyramid levels can support --
        # levels must clamp down instead of raising.
        long_exp, short_exp = _synthetic_pair(h=6, w=6)
        out = fuse_exposures([long_exp, short_exp], levels=6)
        assert out.shape == long_exp.shape
        assert np.all(np.isfinite(out))

    def test_odd_sized_image_roundtrips(self):
        long_exp, short_exp = _synthetic_pair(h=37, w=41)
        out = fuse_exposures([long_exp, short_exp], levels=4)
        assert out.shape == long_exp.shape


class TestPyramidRoundTrip:

    def test_laplacian_reconstruct_is_identity(self):
        rng = np.random.default_rng(0)
        img = rng.uniform(0, 1, (40, 48, 3))
        lpyr = _laplacian_pyramid(img, levels=4)
        recon = _reconstruct_from_laplacian(lpyr)
        np.testing.assert_allclose(recon, img, atol=1e-6)


class TestQualityWeights:

    def test_flat_image_has_low_contrast_weight_everywhere(self):
        flat = np.full((32, 32, 3), 0.5)
        w_flat = _quality_weights(flat, 1.0, 1.0, 1.0)
        edgy = flat.copy()
        edgy[:, 16:, :] = 0.9
        w_edgy = _quality_weights(edgy, 1.0, 1.0, 1.0)
        # The edge region in `edgy` should carry more weight than the
        # equivalent flat region did (contrast term picks up the step).
        assert w_edgy[:, 16].mean() > w_flat[:, 16].mean()

    def test_midgray_more_exposed_than_near_clipped(self):
        midgray = np.full((16, 16, 3), 0.5)
        clipped = np.full((16, 16, 3), 0.98)
        w_mid = _quality_weights(midgray, 0.0, 0.0, 1.0)  # exposedness only
        w_clip = _quality_weights(clipped, 0.0, 0.0, 1.0)
        assert w_mid.mean() > w_clip.mean()
