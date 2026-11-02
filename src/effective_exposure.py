"""Effective integration time: how much good sky each frame adds.

A frame's worth to a sky-noise-limited stack is its signal-to-noise squared per second:
sharper stars concentrate the light in fewer noisy pixels (``1 / FWHM^2``), a brighter sky adds
noise (``1 / background``), and cloud dims the source against that noise (``transparency^2``).
The Dark Energy Survey combines the three into one number per exposure,
``t_eff = (FWHM_fid / FWHM)^2 (B_fid / B) F_trans`` (Morganson et al. 2018, arXiv:1801.03177),
and uses it to accept or reject exposures. Here the fiducials are the session medians, so a
typical frame has ``t_eff`` of about 1 and ``sum(t_eff * EXPTIME)`` is the session's integration
time *in median-quality seconds* -- the honest figure to compare sessions by, where
``INTGTIME`` counts a frame shot through cloud the same as a perfect one.

It is a diagnostic: nothing here changes a gate or a weight (proper coadd already weights by
the same factors). ``frame_t_eff`` also feeds the frame report's low-value flag.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

# DES accepts exposures above 0.2-0.3 by band; below this a frame adds almost nothing.
LOW_T_EFF = 0.25


def _pos(v) -> Optional[float]:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) and v > 0 else None


def session_fiducials(frames: Iterable) -> Dict[str, Optional[float]]:
    """Median FWHM and background of the frames (the fiducial a typical frame is measured against)."""
    out = {}
    for key in ('fwhm', 'background'):
        vals = [_pos((f.metrics or {}).get(key)) for f in frames]
        vals = [v for v in vals if v is not None]
        out[key] = float(np.median(vals)) if vals else None
    return out


def frame_t_eff(metrics: dict, fwhm_ref: Optional[float], bg_ref: Optional[float]) -> Optional[float]:
    """``(fwhm_ref / fwhm)^2 * (bg_ref / bg) * transparency^2``; None without a usable FWHM or background.
    A missing or non-positive transparency counts as 1 (no cloud information)."""
    fwhm, bg = _pos(metrics.get('fwhm')), _pos(metrics.get('background'))
    if fwhm is None or bg is None or not fwhm_ref or not bg_ref:
        return None
    tr = _pos(metrics.get('transparency')) or 1.0
    return float((fwhm_ref / fwhm) ** 2 * (bg_ref / bg) * tr ** 2)


def effective_integration(frames: Iterable) -> Optional[Dict[str, float]]:
    """``{'teff_time': s, 'ratio': teff_time / intg_time, 'n': frames used}`` or None.

    ``frames`` are FrameInfo (``metrics`` and ``header['EXPTIME']``); a frame without the
    metrics or an exposure time is left out of both sums."""
    frames = list(frames)
    fid = session_fiducials(frames)
    teff_time = intg = 0.0
    n = 0
    for f in frames:
        t = frame_t_eff(f.metrics or {}, fid['fwhm'], fid['background'])
        et = _pos((f.header or {}).get('EXPTIME'))
        if t is None or et is None:
            continue
        teff_time += t * et
        intg += et
        n += 1
    if n == 0 or intg <= 0:
        return None
    return {'teff_time': teff_time, 'ratio': teff_time / intg, 'n': n}
