"""Point-source matched filter (--matched-filter).

The matched filter theorem: for a known-shape signal (a point source, whose
shape is the PSF) buried in stationary white noise, the linear filter that
maximises output SNR is correlating the data with the signal's own shape.
This is the same underlying principle IVW combine already uses (weight by
1/noise^2 is the Gauss-Markov-optimal *per-frame* combiner), applied here to
the *spatial shape* of a point source within a single already-stacked image
instead of to noise level across frames -- a post-stack detection filter,
complementary to (not a replacement for) --stack-method ivw.

This codebase's PSF kernels (Gaussian/Moffat fits from
src.psf_deconvolution.estimate_psf/make_synthetic_psf) are radially
symmetric, so correlation and convolution are identical here -- no separate
flipped kernel is needed the way a genuinely asymmetric PSF would require.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def apply_matched_filter(img: np.ndarray, psf: np.ndarray,
                         noise_sigma: Optional[float] = None) -> np.ndarray:
    """Correlate each channel of ``img`` with ``psf`` (normalised to sum 1).

    When ``noise_sigma`` is supplied, the return value is instead a
    per-pixel SNR map: the matched-filter output divided by the filtered
    noise level (``noise_sigma * ||psf_normalised||_2``, the standard
    matched-filter SNR-gain formula for white noise passed through a linear
    filter) -- a point source at the PSF's own shape and amplitude
    ``noise_sigma`` above background reads as SNR ~= 1 in the output.

    Returns a float32 array the same shape as ``img``.
    """
    from scipy.signal import fftconvolve

    psf_n = psf.astype(np.float64)
    psf_sum = psf_n.sum()
    psf_n = psf_n / psf_sum if abs(psf_sum) > 1e-12 else psf_n

    if img.ndim == 2:
        channels = [img.astype(np.float64)]
    else:
        channels = [img[:, :, c].astype(np.float64) for c in range(img.shape[-1])]

    filtered = [fftconvolve(ch, psf_n, mode='same') for ch in channels]
    out = np.stack(filtered, axis=-1) if img.ndim == 3 else filtered[0]

    if noise_sigma is not None and noise_sigma > 0:
        filtered_noise = float(noise_sigma) * float(np.sqrt(np.sum(psf_n ** 2)))
        out = out / max(filtered_noise, 1e-12)

    return out.astype(np.float32)
