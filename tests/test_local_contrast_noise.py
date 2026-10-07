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

from src.denoising import multiscale_local_contrast


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
