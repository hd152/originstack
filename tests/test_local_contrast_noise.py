"""multiscale_local_contrast must enhance a small object, not the sky noise.

The earlier version (a) built its mask from the per-pixel luminance against an
adjacent-pixel noise estimate that reads far too low on denoised (spatially
correlated) data, so the mask switched on across the sky, and (b) capped
highlights at the whole-frame 97th percentile, which on a galaxy covering ~2.5%
of the frame sits below the galaxy -- zero enhancement on the object itself.
Measured on real stacks: display sky noise 1.8x, no structure gained.
"""
import numpy as np
from scipy import ndimage

from src.denoising import (
    _local_detail_sigma,
    _robust_sigma,
    _scale_local_contrast_scales,
    multiscale_local_contrast,
)


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    H, W = 400, 400
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = (yy - 200) ** 2 + (xx - 200) ** 2
    # small galaxy (~2.5% of the frame) with spiral-ish texture
    gal = 600.0 * np.exp(-r2 / (2 * 22.0 ** 2)) * (1 + 0.3 * np.sin(xx / 4.0) * np.sin(yy / 5.0))
    lum = 1000.0 + gal
    # correlated noise, as left by a denoiser
    noise = ndimage.gaussian_filter(rng.normal(0, 30.0, (H, W)), 1.5)
    img = (lum + noise)[..., None] * np.array([1.0, 1.0, 1.0])
    return img.astype(np.float32), r2


def test_sky_noise_not_amplified_and_object_enhanced():
    img, r2 = _scene()
    out = multiscale_local_contrast(img, strength=0.8)
    sky = r2 > 120 ** 2
    obj = r2 < 30 ** 2

    def hp(a):
        g = a[..., 1].astype(np.float64)
        return g - ndimage.gaussian_filter(g, 2.0)

    def band(a):
        g = a[..., 1].astype(np.float64)
        return ndimage.gaussian_filter(g, 1.5) - ndimage.gaussian_filter(g, 8.0)

    sky_ratio = hp(out)[sky].std() / hp(img)[sky].std()
    obj_ratio = band(out)[obj].std() / band(img)[obj].std()
    assert sky_ratio < 1.05, sky_ratio
    assert obj_ratio > 1.15, obj_ratio


def test_zero_strength_is_identity():
    img, _ = _scene(1)
    np.testing.assert_array_equal(multiscale_local_contrast(img, strength=0.0), img)


def test_local_noise_threshold_tracks_coarse_sky_variations():
    rng = np.random.default_rng(17)
    detail = np.empty((256, 512), dtype=np.float64)
    detail[:, :256] = rng.normal(0.0, 2.0, (256, 256))
    detail[:, 256:] = rng.normal(0.0, 10.0, (256, 256))
    sky = np.ones(detail.shape, dtype=bool)
    global_sigma = _robust_sigma(detail[::4, ::4])

    local_sigma = _local_detail_sigma(detail, sky, global_sigma)

    quiet = float(np.median(local_sigma[32:224, 32:224]))
    noisy = float(np.median(local_sigma[32:224, 288:480]))
    assert noisy > quiet * 1.5, (quiet, noisy)
    assert local_sigma.min() >= global_sigma * 0.5
    assert local_sigma.max() <= global_sigma * 2.0


def test_fwhm_scale_adaptation_is_bounded_and_keeps_reference_scales():
    scales = (2, 12, 40)
    assert _scale_local_contrast_scales(scales, None) == (2.0, 12.0, 40.0)
    assert _scale_local_contrast_scales(scales, 4.0) == (2.0, 12.0, 40.0)
    assert _scale_local_contrast_scales(scales, 1.0) == (1.0, 6.0, 20.0)
    assert _scale_local_contrast_scales(scales, 40.0) == (4.0, 24.0, 80.0)
