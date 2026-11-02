"""Relative-kernel (target-PSF) deconvolution: psf_deconvolution.relative_kernel and friends."""
import numpy as np
import pytest
from scipy import signal

from src.psf_deconvolution import (
    clipped_core_mask,
    deconvolution_residual,
    make_synthetic_psf,
    relative_kernel,
    richardson_lucy_deconvolve,
)

FWHM = 4.4
TARGET = 3.5


def _fwhm_of(kernel):
    half = kernel.shape[0] // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    return float(np.sqrt(np.sum(kernel * (yy * yy + xx * xx)) / np.sum(kernel) / 2.0) * 2.355)


def test_kernel_reproduces_the_psf_and_has_the_expected_width():
    psf = make_synthetic_psf(FWHM, model='gaussian')
    s = relative_kernel(psf, TARGET)
    assert s.shape == psf.shape and abs(s.sum() - 1.0) < 1e-9 and s.min() >= 0
    target = make_synthetic_psf(TARGET, model='gaussian')
    rebuilt = signal.fftconvolve(target, s, mode='same')
    assert np.linalg.norm(rebuilt - psf) / np.linalg.norm(psf) < 0.06
    assert abs(_fwhm_of(s) - np.sqrt(FWHM ** 2 - TARGET ** 2)) < 0.3


def test_target_must_be_narrower():
    psf = make_synthetic_psf(FWHM, model='gaussian')
    with pytest.raises(ValueError):
        relative_kernel(psf, FWHM + 0.5)


def _star_field(seed=0, n=200):
    rng = np.random.default_rng(seed)
    psf = make_synthetic_psf(FWHM, model='gaussian')
    truth = np.zeros((n, n))
    pts = [(50, 50), (50, 150), (150, 60), (150, 150), (100, 100)]
    for y, x in pts:
        truth[y, x] = 5000.0
    img = signal.fftconvolve(truth, psf, mode='same') + 100.0
    noisy = img + rng.normal(0, 1.0, img.shape)
    return np.repeat(noisy[..., None], 3, axis=2).astype(np.float32), psf, pts


def _moat(img, pts, r_in, r_out):
    lum = img[..., 0].astype(np.float64)
    yy, xx = np.mgrid[:lum.shape[0], :lum.shape[1]]
    vals = []
    for y, x in pts:
        r = np.hypot(yy - y, xx - x)
        vals.append(lum[(r > r_in) & (r < r_out)].mean() - 100.0)
    return float(np.mean(vals))


def _star_fwhm(img, y, x, half=12):
    cut = img[y - half:y + half + 1, x - half:x + half + 1, 0].astype(np.float64) - 100.0
    cut = np.clip(cut, 0.0, None)
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    return float(np.sqrt(np.sum(cut * (yy * yy + xx * xx)) / np.sum(cut) / 2.0) * 2.355)


def test_deconvolving_to_the_target_psf_narrows_stars_without_a_deeper_moat():
    img, psf, pts = _star_field()
    s = relative_kernel(psf, TARGET)
    rel = richardson_lucy_deconvolve(img, s, iterations=15)
    full = richardson_lucy_deconvolve(img, psf, iterations=15)
    f_in, f_rel = _star_fwhm(img, *pts[0]), _star_fwhm(rel, *pts[0])
    assert f_rel < 0.9 * f_in                                        # narrower than the input...
    assert f_rel > 0.8 * TARGET                                      # ...but not a spike
    # ringing: the annulus just outside the star is no lower than with the full-PSF kernel
    assert _moat(rel, pts, 3 * FWHM, 5 * FWHM) >= _moat(full, pts, 3 * FWHM, 5 * FWHM) - 0.05


def test_clipped_core_mask_covers_plateaus_only():
    lum = np.full((100, 100), 50.0, np.float32)
    lum[40:44, 40:44] = 1000.0
    m = clipped_core_mask(lum, 4.0)
    assert m is not None and m[41, 41] > 0.9 and m[5, 5] < 0.01
    assert clipped_core_mask(np.zeros((10, 10), np.float32), 3.0) is None


def test_residual_consistency():
    rng = np.random.default_rng(3)
    y = np.repeat((100.0 + rng.normal(0, 2.0, (120, 120)))[..., None], 3, axis=2).astype(np.float32)
    delta = np.zeros((9, 9))
    delta[4, 4] = 1.0
    ok = deconvolution_residual(y, y, delta, 2.0)
    assert ok['chi2_sky'] < 1e-9
    bad = deconvolution_residual(y, y + rng.normal(0, 8.0, y.shape).astype(np.float32), delta, 2.0)
    assert bad['chi2_sky'] > 3
