"""Per-band noise sigma in directional_wavelet_denoise.

The denoiser used the finest diagonal band's MAD as the noise sigma of every subband. That is
right only for white noise; Origin stacks carry correlated noise (lag-1 ~0.4), whose power sits
in the coarser bands, so those were under-thresholded and the denoiser left 90-98% of the sky
noise. ``band_sigma=True`` (default) estimates each band's own sigma.
"""
import numpy as np
from scipy import ndimage as ndi

from src import wavelet
from src.denoising import _band_noise_sigma, directional_wavelet_denoise

N = 384


def _noise(rng, shape, smooth):
    n = rng.standard_normal(shape)
    if smooth > 0:
        n = ndi.gaussian_filter(n, smooth, mode='wrap')
    return n / n.std()


def _scene(rng, smooth):
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    truth = 0.9 * np.exp(-((yy - 120) ** 2 + (xx - 140) ** 2) / (2 * 25.0 ** 2))
    truth += 0.6 * np.exp(-((yy - 270) ** 2 + (xx - 250) ** 2) / (2 * 30.0 ** 2))
    img = np.stack([truth + _noise(rng, (N, N), smooth) for _ in range(3)], axis=2)
    sky = ((yy - 120) ** 2 + (xx - 140) ** 2 > 80 ** 2) & ((yy - 270) ** 2 + (xx - 250) ** 2 > 90 ** 2)
    return img.astype(np.float32), truth, sky


def _sky_rms(out, truth, sky):
    return float(np.sqrt(np.mean((out.astype(np.float64).mean(axis=2) - truth)[sky] ** 2)))


def test_correlated_noise_band_sigma_removes_more_noise():
    rng = np.random.default_rng(1)
    img, truth, sky = _scene(rng, 0.53)                  # lag-1 correlation ~0.3-0.4
    stock = directional_wavelet_denoise(img, band_sigma=False)
    banded = directional_wavelet_denoise(img)
    rms_in = _sky_rms(img, truth, sky)
    rms_stock, rms_band = _sky_rms(stock, truth, sky), _sky_rms(banded, truth, sky)
    assert rms_stock > 0.7 * rms_in                       # the defect: most noise left
    assert rms_band < 0.5 * rms_in
    assert rms_band < 0.5 * rms_stock


def test_white_noise_not_worse_with_band_sigma():
    rng = np.random.default_rng(2)
    img, truth, sky = _scene(rng, 0.0)
    a = _sky_rms(directional_wavelet_denoise(img, band_sigma=False), truth, sky)
    b = _sky_rms(directional_wavelet_denoise(img), truth, sky)
    assert b <= 1.03 * a                                  # never leaves more noise than before


def test_correlated_noise_keeps_faint_structure_flux():
    rng = np.random.default_rng(3)
    img, truth, sky = _scene(rng, 0.53)
    out = directional_wavelet_denoise(img).astype(np.float64).mean(axis=2)
    blob = truth > 0.3
    slope = np.polyfit(truth[blob], out[blob], 1)[0]
    assert 0.9 < slope < 1.1


def test_band_sigma_reads_each_band_and_floors_at_white_noise_value():
    rng = np.random.default_rng(4)
    n = _noise(rng, (N, N), 0.53)
    co = wavelet.wavedec2(n, 4)
    white = float(np.median(np.abs(co[-1][-1])) / 0.6745)
    # correlated noise: finest H/V and the coarser bands read well above the diagonal value
    assert _band_noise_sigma(co[-1][0], 1.0, white) > 1.3 * white
    assert _band_noise_sigma(co[-3][2], 1.0, white) > 1.5 * white
    # a band reading below the floor is clamped to it; the chroma multiplier scales both
    quiet = np.zeros((64, 64))
    assert _band_noise_sigma(quiet, 1.0, 0.7) == 0.7
    assert _band_noise_sigma(co[-3][2], 2.0, 2 * white) >= 2 * white
