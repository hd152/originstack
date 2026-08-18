"""Tests for FISTA sparse wavelet-domain deconvolution (--deconvolve sparse,
src/psf_deconvolution.py::sparse_wavelet_deconvolve) -- an alternative to
tv_regularized_deconvolve's spatial-gradient prior, regularising in this
project's own wavelet basis instead.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import laplace

from src.psf_deconvolution import sparse_wavelet_deconvolve, make_synthetic_psf


def _make_star_field_rgb(shape=(128, 128), n_stars=15, sigma=2.0, amp=500.0, bg=100.0):
    rng = np.random.RandomState(42)
    margin = 20
    lum = np.full(shape, bg, dtype=np.float64)
    positions = []
    for _ in range(n_stars):
        y = rng.randint(margin, shape[0] - margin)
        x = rng.randint(margin, shape[1] - margin)
        positions.append((y, x))
        yy, xx = np.indices(shape)
        r2 = (yy - y) ** 2.0 + (xx - x) ** 2.0
        lum += amp * np.exp(-r2 / (2.0 * sigma ** 2))
    rgb = np.stack([lum, lum, lum], axis=2).astype(np.float32)
    return rgb, positions


def test_increases_sharpness():
    rgb, _ = _make_star_field_rgb(shape=(64, 64), n_stars=5, sigma=1.5, amp=500.0)
    psf = make_synthetic_psf(fwhm=4.0, psf_size=15, model='gaussian')
    blurred = np.empty_like(rgb)
    for c in range(3):
        blurred[:, :, c] = fftconvolve(rgb[:, :, c], psf, mode='same')

    recovered = sparse_wavelet_deconvolve(blurred, psf, iterations=30, lam=0.01)

    def sharpness(img):
        lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        return float(np.var(laplace(lum)))

    assert sharpness(recovered) > sharpness(blurred)


def test_preserves_flux_approximately():
    rgb, _ = _make_star_field_rgb(shape=(64, 64), n_stars=5, sigma=2.0)
    psf = make_synthetic_psf(fwhm=4.0, psf_size=15)
    result = sparse_wavelet_deconvolve(rgb, psf, iterations=20, lam=0.01)
    flux_in = float(rgb.sum())
    flux_out = float(result.sum())
    assert abs(flux_out - flux_in) / flux_in < 0.1


def test_star_mask_blends_back_original():
    rgb, _ = _make_star_field_rgb(shape=(64, 64), n_stars=3, sigma=2.0)
    psf = make_synthetic_psf(fwhm=4.0, psf_size=15)
    mask = np.ones(rgb.shape[:2], dtype=np.float64)
    result = sparse_wavelet_deconvolve(rgb, psf, iterations=10, star_mask=mask)
    np.testing.assert_allclose(result, rgb, atol=0.05)


def test_output_shape_dtype_finite():
    rgb, _ = _make_star_field_rgb(shape=(48, 48), n_stars=4)
    psf = make_synthetic_psf(fwhm=3.0, psf_size=11)
    out = sparse_wavelet_deconvolve(rgb, psf, iterations=8)
    assert out.shape == rgb.shape
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))


def test_larger_lambda_smooths_more_than_smaller_lambda():
    # Total-variation (mean |gradient|) is a monotonic, artifact-robust
    # smoothness proxy here -- Laplacian variance turned out to pick up
    # thresholding artifacts at very large lambda rather than tracking
    # smoothness cleanly, so TV is used instead of assuming that measure
    # would behave as expected without checking.
    rgb, _ = _make_star_field_rgb(shape=(64, 64), n_stars=6, sigma=1.5, amp=400.0)
    psf = make_synthetic_psf(fwhm=4.0, psf_size=15)
    blurred = np.empty_like(rgb)
    for c in range(3):
        blurred[:, :, c] = fftconvolve(rgb[:, :, c], psf, mode='same')

    sharp_result = sparse_wavelet_deconvolve(blurred, psf, iterations=25, lam=0.001)
    smooth_result = sparse_wavelet_deconvolve(blurred, psf, iterations=25, lam=0.3)

    def total_variation(img):
        lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        return float(np.mean(np.abs(np.diff(lum, axis=0)))
                     + np.mean(np.abs(np.diff(lum, axis=1))))

    assert total_variation(sharp_result) > total_variation(smooth_result)
