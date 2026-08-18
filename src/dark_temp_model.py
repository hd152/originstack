"""Temperature-interpolated dark current model (--dark-temp-model).

``select_matching_darks`` (src/frame_discovery.py) selects darks by scoring
against the lights' majority ISO/exptime/temperature and narrowing to the
closest match -- it needs a dark within its scoring tolerance of the actual
session temperature. This module instead fits a per-pixel low-order
polynomial of dark signal vs. sensor temperature across the *whole* dark
library (the opposite goal: keep the temperature spread, don't narrow it),
so a session at a temperature the library has no close match for can still
get an accurate dark by interpolation/mild extrapolation instead of falling
back to whatever nearest-temperature dark happens to exist.

Simplification: assumes the supplied dark library is already homogeneous in
ISO/gain and exposure time (dark current scales with those too, and this
module doesn't attempt to also model them) -- a reasonable assumption for a
library built specifically to span multiple temperature sessions at one
fixed ISO/exptime, but not a substitute for select_matching_darks' ISO/
exptime filtering if the library mixes those too.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.models import FrameInfo
from src.utils import get_logger, safe_print

_log = get_logger()


def _frame_temp(f: FrameInfo) -> Optional[float]:
    for key in ('CCDTEMP', 'CCD-TEMP', 'TEMPERAT', 'CCD_TEMP', 'SET-TEMP'):
        val = f.header.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return None


def build_dark_temperature_model(frames: List[FrameInfo], degree: int = 2) -> Optional[dict]:
    """Fit a per-pixel polynomial of dark signal vs. sensor temperature.

    A single ``np.linalg.lstsq`` call solves the least-squares fit for every
    pixel at once (each pixel is one column of the right-hand side against
    the shared temperature-Vandermonde design matrix) -- not a per-pixel
    Python loop.

    Returns a model dict, or ``None`` if there are too few frames or too few
    *distinct* temperatures for a meaningful fit (``degree + 1`` distinct
    values is the bare minimum for a well-posed fit; ``degree + 2`` frames
    guards against an exactly-determined, zero-residual fit that gives no
    real indication of fit quality).
    """
    from src.io_fits import load_frame

    temps: List[float] = []
    imgs: List[np.ndarray] = []
    for f in frames:
        t = _frame_temp(f)
        if t is None:
            continue
        try:
            data, _ = load_frame(f.path)
        except Exception:
            continue
        temps.append(t)
        imgs.append(data.astype(np.float64))

    n_distinct = len(set(round(t, 1) for t in temps))
    if len(imgs) < degree + 2 or n_distinct < degree + 1:
        _log.warning(
            "build_dark_temperature_model: %d usable frames spanning %d distinct "
            "temperature(s) (need >= %d frames, >= %d distinct temps for degree=%d)",
            len(imgs), n_distinct, degree + 2, degree + 1, degree)
        return None

    shape = imgs[0].shape
    if any(im.shape != shape for im in imgs):
        keep = [i for i, im in enumerate(imgs) if im.shape == shape]
        imgs = [imgs[i] for i in keep]
        temps = [temps[i] for i in keep]
        n_distinct = len(set(round(t, 1) for t in temps))
        if len(imgs) < degree + 2 or n_distinct < degree + 1:
            return None

    temps_arr = np.array(temps, dtype=np.float64)
    stack = np.stack(imgs, axis=0)
    n = stack.shape[0]
    data = stack.reshape(n, -1)

    vander = np.vander(temps_arr, degree + 1, increasing=True)
    coeffs, _residuals, _rank, _sv = np.linalg.lstsq(vander, data, rcond=None)
    coeffs = coeffs.reshape((degree + 1,) + shape)

    return {
        'coeffs': coeffs,
        'degree': degree,
        'shape': shape,
        'temp_range': (float(temps_arr.min()), float(temps_arr.max())),
        'n_frames': n,
        'n_temps': n_distinct,
    }


def sample_dark_at_temperature(model: dict, temp_c: float) -> np.ndarray:
    """Evaluate a temperature dark model at a given sensor temperature.

    Mild extrapolation beyond the fitted range is allowed (a session a
    couple of degrees outside the library's span is common and still far
    better served by extrapolation than an unrelated-temperature dark), but
    warned about since polynomial extrapolation error grows quickly past
    the fitted domain.
    """
    lo, hi = model['temp_range']
    if temp_c < lo - 5.0 or temp_c > hi + 5.0:
        safe_print(f"  WARNING: dark temperature model: {temp_c:.1f}°C is well outside "
                   f"the fitted range [{lo:.1f}, {hi:.1f}]°C -- extrapolation may be unreliable")
    degree = model['degree']
    row = np.array([temp_c ** p for p in range(degree + 1)], dtype=np.float64)
    result = np.tensordot(row, model['coeffs'], axes=(0, 0))
    return np.maximum(result, 0.0).astype(np.float32)
