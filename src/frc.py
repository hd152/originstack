"""Split-half resolution and denoiser-risk measures: Fourier ring correlation and friends.

Two independent half-stacks A and B of the same sky share the signal and differ in noise.
That alone gives, with no ground truth and no model of the noise:

* ``fourier_ring_correlation`` -- the correlation of A and B in rings of spatial frequency.
  It is 1 where signal dominates both halves and falls to 0 where only noise is left, so
  the frequency where it crosses a threshold (``resolution_period``) is the finest detail
  the stack really carries. Correlated noise (resampling) shows up as a slower fall than
  white noise would give.
* ``ssnr_from_frc`` -- the per-ring signal-to-noise ratio of a half and of the full stack
  (for halves of equal noise, ``SNR_half = FRC / (1 - FRC)``; the full stack has twice it).
* ``wiener_from_halves`` -- the per-ring Wiener filter of the full stack that SNR implies:
  a parameter-free denoising baseline, and a frequency cut-off for any other denoiser.
* ``scale_risk`` -- Noise2Inverse-style risk of a denoiser, per spatial scale: how far
  ``D(A)`` is from the independent ``B`` beyond B's own noise, against the risk of doing
  nothing. A ratio above 1 at a scale means the denoiser destroys structure there.
* ``method_noise`` -- whether what a denoiser removed looks like noise (white, and
  uncorrelated with what it kept).

Limits: a learned prior reproduces its hallucinations identically in both halves, so FRC
overstates the resolution of a network's output; pair it with a truth-based check. The
criterion ``FRC = 1/7`` is the usual full-image threshold, not a statement about any one
feature.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

try:
    from scipy import fft as _fft
    from scipy import ndimage as _ndimage
except Exception:  # pragma: no cover
    _fft = None
    _ndimage = None

FRC_THRESHOLD = 1.0 / 7.0


def _window(shape: Tuple[int, int]) -> np.ndarray:
    """Separable Hann window: stops the frame edge from showing up as signal at every frequency."""
    return np.outer(np.hanning(shape[0]), np.hanning(shape[1])).astype(np.float64)


def _ring_index(shape: Tuple[int, int], n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frequency ring number (-1 beyond Nyquist) and the ring centres in cycles/pixel."""
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    f = np.hypot(fy, fx)
    idx = np.floor(f / 0.5 * n_bins).astype(np.int64)
    idx[f >= 0.5] = -1
    centres = (np.arange(n_bins) + 0.5) * 0.5 / n_bins
    return idx, centres


def fourier_ring_correlation(a: np.ndarray, b: np.ndarray, n_bins: Optional[int] = None
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """``(freq, frc)``: frequency in cycles/pixel (ring centres up to 0.5) and the FRC.

    ``a`` and ``b`` are two 2-D images of the same field with independent noise (means are
    removed and a Hann window applied first)."""
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("fourier_ring_correlation needs two 2-D arrays of the same shape")
    if n_bins is None:
        n_bins = min(a.shape) // 2
    w = _window(a.shape)
    fa = _fft.fft2((a - np.mean(a)) * w, workers=-1)
    fb = _fft.fft2((b - np.mean(b)) * w, workers=-1)
    idx, centres = _ring_index(a.shape, n_bins)
    sel = idx >= 0
    i = idx[sel]
    cross = np.bincount(i, weights=(fa * np.conj(fb)).real[sel], minlength=n_bins)
    pa = np.bincount(i, weights=(np.abs(fa) ** 2)[sel], minlength=n_bins)
    pb = np.bincount(i, weights=(np.abs(fb) ** 2)[sel], minlength=n_bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        frc = np.where((pa > 0) & (pb > 0), cross / np.sqrt(pa * pb), 0.0)
    return centres, frc


def resolution_period(freq: np.ndarray, frc: np.ndarray, threshold: float = FRC_THRESHOLD,
                      smooth: int = 5) -> Optional[float]:
    """Spatial period (pixels) of the first frequency where the FRC stays under ``threshold``.

    The FRC is median-smoothed over ``smooth`` rings so a single noisy ring cannot end it
    early; the crossing is linearly interpolated. ``None`` when it never falls that far
    (the halves agree out to Nyquist) or starts below the threshold."""
    f = np.asarray(frc, dtype=np.float64)
    if smooth > 1 and _ndimage is not None:
        f = _ndimage.median_filter(f, size=smooth, mode='nearest')
    if f.size < 3 or f[0] < threshold:
        return None
    below = np.flatnonzero(f < threshold)
    if below.size == 0:
        return None
    j = int(below[0])
    f0, f1, y0, y1 = freq[j - 1], freq[j], f[j - 1], f[j]
    fc = f0 + (y0 - threshold) / max(y0 - y1, 1e-12) * (f1 - f0)
    return float(1.0 / fc) if fc > 0 else None


def ssnr_from_frc(frc: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-ring SNR from the FRC of two equal-noise halves: ``half = FRC / (1 - FRC)`` and
    ``full = 2 * half`` (the full stack averages twice the data). FRC is clipped to [0, 0.999]."""
    c = np.clip(np.asarray(frc, dtype=np.float64), 0.0, 0.999)
    half = c / (1.0 - c)
    return {'half': half, 'full': 2.0 * half}


def wiener_from_halves(a: np.ndarray, b: np.ndarray, n_bins: Optional[int] = None) -> np.ndarray:
    """Wiener-filtered mean of two halves, the gain of each frequency ring set by the halves'
    own agreement: ``gain = SNR_full / (1 + SNR_full)``.

    No tuning parameter. It filters isotropically, so it trades the noise in the rings the
    data cannot support for nothing else (no edge or star awareness): a baseline to beat
    and a place to read a sensible frequency cut-off, not a replacement for the denoisers."""
    if n_bins is None:
        n_bins = min(a.shape) // 2
    freq, frc = fourier_ring_correlation(a, b, n_bins)
    snr = ssnr_from_frc(frc)['full']
    gain_ring = snr / (1.0 + snr)
    mean = 0.5 * (np.asarray(a, np.float64) + np.asarray(b, np.float64))
    m0 = float(np.mean(mean))
    idx, _centres = _ring_index(a.shape, n_bins)
    g = np.where(idx >= 0, gain_ring[np.clip(idx, 0, n_bins - 1)], 0.0)
    g.flat[0] = 1.0                                   # keep the mean level
    out = _fft.ifft2(_fft.fft2(mean - m0, workers=-1) * g, workers=-1).real + m0
    return out.astype(np.float32)


def _band(x: np.ndarray, s_lo: float, s_hi: float) -> np.ndarray:
    """Difference-of-Gaussians band: structure between ``s_lo`` and ``s_hi`` pixels."""
    lo = _ndimage.gaussian_filter(x, s_lo) if s_lo > 0 else x
    return lo - _ndimage.gaussian_filter(x, s_hi)


def scale_risk(a: np.ndarray, b: np.ndarray, denoised_a: np.ndarray,
               mask: Optional[np.ndarray] = None,
               sigmas: Tuple[float, ...] = (1, 2, 4, 8, 16, 32)) -> Dict[str, np.ndarray]:
    """Per-scale risk of denoising ``a`` into ``denoised_a``, judged against the independent ``b``.

    For band ``k`` (a difference of Gaussians of width ``sigmas[k]`` to ``2 * sigmas[k]``):
    ``E_k = mean[(band(D(A)) - band(B))^2] - var_B,k`` is the squared error of D(A) against the
    truth, B's own noise taken out; ``var_B,k = var(band(A - B)) / 2`` for halves of equal noise,
    which is also the risk of doing nothing (``band(A)`` against the truth). ``ratio = E_k /
    var_B,k`` below 1 means the denoiser helps at that scale, above 1 that it harms it (the
    over-smoothing of faint extended light shows as a bump at 16-64 px). ``mask`` selects the
    pixels (sky, away from stars) the means are taken over."""
    a64, b64, d64 = (np.asarray(x, np.float64) for x in (a, b, denoised_a))
    m = np.ones(a.shape, bool) if mask is None else np.asarray(mask, bool)
    risk, noop = [], []
    for s in sigmas:
        s_lo, s_hi = float(s), 2.0 * float(s)
        bd, bb = _band(d64, s_lo, s_hi), _band(b64, s_lo, s_hi)
        var_b = float(np.var(_band(a64 - b64, s_lo, s_hi)[m])) / 2.0
        risk.append(float(np.mean((bd - bb)[m] ** 2)) - var_b)
        noop.append(var_b)
    risk, noop = np.asarray(risk), np.asarray(noop)
    return {'sigma': np.asarray(sigmas, dtype=np.float64), 'risk': risk, 'noop': noop,
            'ratio': risk / np.maximum(noop, 1e-30)}


def method_noise(noisy: np.ndarray, denoised: np.ndarray, mask: Optional[np.ndarray] = None,
                 lags: Tuple[int, ...] = (1, 2, 4)) -> Dict[str, object]:
    """Does what the denoiser removed look like noise? Autocorrelation of the removed
    component at the given lags (compare with the same of ``a - b`` for the true noise), and
    its correlation with the denoised image (structure leaking into the residual)."""
    r = np.asarray(noisy, np.float64) - np.asarray(denoised, np.float64)
    d = np.asarray(denoised, np.float64)
    m = np.ones(r.shape, bool) if mask is None else np.asarray(mask, bool)
    r0 = r - np.mean(r[m])
    ac = {}
    for lag in lags:
        p = (r0[:, :-lag] * r0[:, lag:])[m[:, :-lag] & m[:, lag:]]
        ac[lag] = float(np.mean(p) / max(np.var(r0[m]), 1e-30))
    dd = d - np.mean(d[m])
    corr = float(np.mean((r0 * dd)[m]) / max(np.std(r0[m]) * np.std(dd[m]), 1e-30))
    return {'autocorr': ac, 'corr_with_output': corr, 'removed_std': float(np.std(r0[m]))}
