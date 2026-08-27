"""Photon-transfer gain / read-noise estimate from raw calibration frames.

Classic two-frame difference method (Janesick, *Photon Transfer*): from a
matched pair of bias frames and a matched pair of flat frames,

    read_noise_adu = std(bias1 - bias2) / sqrt(2)
    gain [e-/ADU]  = ( mean(flat1) + mean(flat2) - mean(bias1) - mean(bias2) )
                     / ( var(flat1 - flat2) - var(bias1 - bias2) )
    read_noise_e   = read_noise_adu * gain

Differencing a matched pair removes the fixed flat-field / bias structure,
so the residual variance is pure temporal noise (shot + read). A central
sigma-clipped window dodges vignetting and amp-glow at the frame edges.

This is one flat level -> a single (signal, variance) point, not a full
photon-transfer curve: enough to put a real Poisson term into aperture
photometry (`--photometry`), not a linearity characterisation. Colour
sensors are reduced to luma (mean over channels / a 2x2 Bayer block is
not separated) -- the per-channel gains of an OSC differ slightly, but the
scalar is well within what the photometric error budget needs.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

_log = logging.getLogger("originstack")

_GAIN_PLAUSIBLE = (0.02, 100.0)   # e-/ADU
_MIN_SNR_SPREAD = 1.5             # flat must sit well above the bias noise


def _to_2d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 3:
        a = a.mean(axis=2)
    return a


def _central(arr: np.ndarray, frac: float = 0.6) -> np.ndarray:
    h, w = arr.shape
    dh = int(h * (1 - frac) / 2)
    dw = int(w * (1 - frac) / 2)
    return arr[dh:h - dh, dw:w - dw]


def _clipped_mean_var(arr: np.ndarray, sigma: float = 5.0,
                      iters: int = 3) -> Tuple[float, float]:
    """Sigma-clipped mean and variance (rejects hot pixels / cosmic rays)."""
    x = arr.ravel().astype(np.float64)
    for _ in range(iters):
        m = np.median(x)
        s = 1.4826 * np.median(np.abs(x - m))
        if s <= 0:
            break
        keep = np.abs(x - m) <= sigma * s
        if keep.sum() < 0.5 * x.size:
            break
        x = x[keep]
    return float(np.mean(x)), float(np.var(x))


def estimate_gain_ptc(bias_paths: List[str], flat_paths: List[str],
                      load_fn=None) -> Tuple[Optional[float], Optional[float]]:
    """Estimate (gain_e_per_adu, read_noise_e) from >=2 bias + >=2 flat
    frame paths. Returns (None, None) when the inputs are insufficient or
    the fit lands outside a plausible range.
    """
    if load_fn is None:
        from src.io_fits import load_frame as load_fn
    if len(bias_paths) < 2 or len(flat_paths) < 2:
        return None, None

    try:
        b1 = _central(_to_2d(load_fn(bias_paths[0])[0]))
        b2 = _central(_to_2d(load_fn(bias_paths[1])[0]))
        f1 = _central(_to_2d(load_fn(flat_paths[0])[0]))
        f2 = _central(_to_2d(load_fn(flat_paths[1])[0]))
    except Exception as exc:
        _log.debug("PTC: could not load calibration frames: %s", exc)
        return None, None

    if b1.shape != b2.shape or f1.shape != f2.shape or b1.shape != f1.shape:
        _log.debug("PTC: calibration frame shapes disagree")
        return None, None

    bmean1, _ = _clipped_mean_var(b1)
    bmean2, _ = _clipped_mean_var(b2)
    fmean1, _ = _clipped_mean_var(f1)
    fmean2, _ = _clipped_mean_var(f2)

    _, var_bias_diff = _clipped_mean_var(b1 - b2)
    _, var_flat_diff = _clipped_mean_var(f1 - f2)

    signal = (fmean1 + fmean2) - (bmean1 + bmean2)          # 2x mean flat signal
    noise_var = var_flat_diff - var_bias_diff               # 2x shot variance
    if signal <= 0 or noise_var <= 0:
        _log.debug("PTC: non-positive signal (%.3g) or noise variance (%.3g)",
                   signal, noise_var)
        return None, None
    if var_flat_diff < _MIN_SNR_SPREAD * var_bias_diff:
        _log.debug("PTC: flat pair not above the bias noise floor")
        return None, None

    gain = signal / noise_var
    if not (_GAIN_PLAUSIBLE[0] <= gain <= _GAIN_PLAUSIBLE[1]):
        _log.debug("PTC: gain %.3g e-/ADU outside plausible range", gain)
        return None, None

    read_noise_adu = np.sqrt(var_bias_diff / 2.0)
    read_noise_e = float(read_noise_adu * gain)
    return float(gain), read_noise_e
