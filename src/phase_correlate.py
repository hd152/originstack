"""Subpixel image translation registration by phase cross-correlation.

Exact port of skimage.registration.phase_cross_correlation (real-space,
unmasked, normalization="phase" path only -- the only mode this codebase's
registration.py ever calls) -- itself a port of Manuel Guizar-Sicairos'
MATLAB implementation of the matrix-multiply-DFT upsampling algorithm:

    Guizar-Sicairos, Thurman & Fienup, "Efficient subpixel image
    registration algorithms," Optics Letters 33, 156-158 (2008).

Validated bit-exact (see tests/test_phase_correlate.py) against the real
skimage implementation across pure-translation and sub-pixel-shift synthetic
cases. No external dependency beyond scipy.fft, already a core dependency
of this codebase.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.fft as sfft


def _upsampled_dft(data: np.ndarray, upsampled_region_size: int,
                   upsample_factor: float, axis_offsets: Tuple[float, float]) -> np.ndarray:
    """Upsampled DFT by matrix multiplication -- computes only a small
    ``upsampled_region_size``-sized neighbourhood of the upsampled DFT
    around ``axis_offsets`` rather than zero-padding and FFT-ing the whole
    (upsample_factor-times-larger) array."""
    im2pi = 1j * 2 * np.pi
    for n_items, ax_offset in zip(data.shape[::-1], axis_offsets[::-1]):
        kernel = ((np.arange(upsampled_region_size) - ax_offset)[:, None]
                  * sfft.fftfreq(n_items, upsample_factor))
        kernel = np.exp(-im2pi * kernel).astype(data.dtype, copy=False)
        data = np.tensordot(kernel, data, axes=(1, -1))
    return data


def _compute_error(cc_max: complex, src_amp: float, target_amp: float) -> float:
    amp = src_amp * target_amp
    with np.errstate(invalid='ignore'):
        error = 1.0 - cc_max * np.conj(cc_max) / amp
    return float(np.sqrt(np.abs(error)))


def phase_cross_correlation(reference_image: np.ndarray, moving_image: np.ndarray,
                            upsample_factor: int = 1) -> Tuple[np.ndarray, float, float]:
    """Subpixel translation shift (in pixels) that best aligns ``moving_image``
    onto ``reference_image``, plus a normalized RMS error and phase-difference
    diagnostic.

    Returns (shift, error, phasediff) -- shift is a 2-element array
    (shift_row, shift_col); registering ``moving_image`` requires shifting it
    by ``shift``. Images will be registered to within ``1/upsample_factor``
    of a pixel.
    """
    if reference_image.shape != moving_image.shape:
        raise ValueError("images must be same shape")

    src_freq = sfft.fftn(reference_image)
    target_freq = sfft.fftn(moving_image)

    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    eps = np.finfo(image_product.real.dtype).eps
    image_product /= np.maximum(np.abs(image_product), 100 * eps)
    cross_correlation = sfft.ifftn(image_product)

    maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)), cross_correlation.shape)
    midpoint = np.array([np.trunc(axis_size / 2) for axis_size in shape])

    float_dtype = image_product.real.dtype
    shift = np.stack(maxima).astype(float_dtype, copy=False)
    shift[shift > midpoint] -= np.array(shape)[shift > midpoint]

    if upsample_factor == 1:
        src_amp = np.sum(np.real(src_freq * src_freq.conj())) / src_freq.size
        target_amp = np.sum(np.real(target_freq * target_freq.conj())) / target_freq.size
        cc_max = cross_correlation[maxima]
    else:
        upsample_factor = float(upsample_factor)
        shift = np.round(shift * upsample_factor) / upsample_factor
        upsampled_region_size = int(np.ceil(upsample_factor * 1.5))
        dftshift = np.trunc(upsampled_region_size / 2.0)
        sample_region_offset = dftshift - shift * upsample_factor
        cross_correlation = _upsampled_dft(
            image_product.conj(), upsampled_region_size, upsample_factor,
            tuple(sample_region_offset)).conj()
        maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)), cross_correlation.shape)
        cc_max = cross_correlation[maxima]
        maxima_arr = np.stack(maxima).astype(float_dtype, copy=False) - dftshift
        shift = shift + maxima_arr / upsample_factor

        src_amp = np.sum(np.real(src_freq * src_freq.conj()))
        target_amp = np.sum(np.real(target_freq * target_freq.conj()))

    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shift[dim] = 0.0

    error = _compute_error(cc_max, src_amp, target_amp)
    phasediff = float(np.arctan2(cc_max.imag, cc_max.real))
    return shift, error, phasediff
