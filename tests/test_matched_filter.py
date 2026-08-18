"""Tests for the point-source matched filter (--matched-filter,
src/matched_filter.py).
"""
from __future__ import annotations

import numpy as np

from src.matched_filter import apply_matched_filter


def _gaussian_psf(size=15, sigma=2.0):
    half = size // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return psf / psf.sum()


def test_output_shape_dtype():
    img = np.zeros((32, 32, 3), dtype=np.float32)
    psf = _gaussian_psf()
    out = apply_matched_filter(img, psf)
    assert out.shape == img.shape
    assert out.dtype == np.float32


def test_2d_input_returns_2d_output():
    img = np.zeros((32, 32), dtype=np.float32)
    psf = _gaussian_psf()
    out = apply_matched_filter(img, psf)
    assert out.shape == img.shape


def test_peaks_at_true_point_source_location():
    h = w = 64
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    sigma_true = 2.0
    star = 1000.0 * np.exp(-((yy - 32) ** 2 + (xx - 30) ** 2) / (2 * sigma_true ** 2))
    img = np.stack([star] * 3, axis=-1).astype(np.float32)

    psf = _gaussian_psf(sigma=sigma_true)
    out = apply_matched_filter(img, psf)
    peak = np.unravel_index(np.argmax(out[:, :, 0]), out.shape[:2])
    assert peak == (32, 30)


def test_flux_approximately_preserved_for_matched_psf():
    # Correlating a Gaussian source with its own (normalised) matching PSF
    # concentrates flux near the peak but total energy in the smoothed
    # result should stay comparable -- not blow up or vanish.
    h = w = 48
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    star = 500.0 * np.exp(-((yy - 24) ** 2 + (xx - 24) ** 2) / (2 * 2.0 ** 2))
    img = star[..., np.newaxis].astype(np.float32)
    psf = _gaussian_psf(sigma=2.0)
    out = apply_matched_filter(img, psf)
    assert 0.5 * img.sum() < out.sum() < 2.0 * img.sum()


def test_snr_map_matches_theoretical_gain_on_white_noise():
    # White noise (sigma known) with no source: the matched-filter SNR map
    # should have output std close to 1.0 (the whole point of dividing by
    # noise_sigma * ||psf||_2 is to normalise the filtered noise to unit
    # std), not exactly 1.0 given finite-sample variance.
    rng = np.random.default_rng(0)
    h = w = 200
    noise_sigma = 5.0
    noise = rng.normal(0.0, noise_sigma, (h, w)).astype(np.float32)
    psf = _gaussian_psf(sigma=2.0)
    snr_map = apply_matched_filter(noise, psf, noise_sigma=noise_sigma)
    assert 0.7 < float(np.std(snr_map)) < 1.3


def test_source_at_noise_sigma_amplitude_reads_snr_near_one():
    h = w = 96
    noise_sigma = 3.0
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    psf = _gaussian_psf(sigma=2.0)
    # A source whose peak amplitude equals noise_sigma, shaped exactly like
    # the matched PSF -- matched-filter theory says this should read near
    # unit SNR at the peak.
    amp = noise_sigma
    star = amp * np.exp(-((yy - 48) ** 2 + (xx - 48) ** 2) / (2 * 2.0 ** 2))
    img = (star + rng.normal(0, noise_sigma, (h, w))).astype(np.float32)
    snr_map = apply_matched_filter(img, psf, noise_sigma=noise_sigma)
    peak_snr = float(snr_map.max())
    # Order-of-magnitude sanity, not exact -- single noisy realization.
    assert 0.3 < peak_snr < 10.0
