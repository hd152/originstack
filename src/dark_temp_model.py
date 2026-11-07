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
from src.utils import get_logger, header_get_first, safe_print

_log = get_logger()


def _frame_temp(f: FrameInfo) -> Optional[float]:
    return header_get_first(f.header, ('CCDTEMP', 'CCD-TEMP', 'TEMPERAT', 'CCD_TEMP', 'SET-TEMP'),
                            cast=float)


# How far outside the fitted temperatures the model may be evaluated. A degree-2 polynomial
# through darks spanning 3 degrees was evaluated 4-6 degrees below them (a 24 C session with
# 27-30 C darks) and returned either 0 or 18,000 ADU: a wrong dark is worse than none.
MAX_EXTRAPOLATION_C = 2.0
# Darks within this relative exposure difference are one exposure (15.0 s and 15.1 s).
_EXPOSURE_TOL = 0.05


def _frame_exposure(f: FrameInfo) -> Optional[float]:
    return header_get_first(f.header, ('EXPTIME', 'EXPOSURE'), cast=float)


def _exposure_groups(exps: List[Optional[float]]) -> List[List[int]]:
    """Indices of ``exps`` grouped by exposure (within ``_EXPOSURE_TOL``); frames with no
    exposure header form one group, so a library that never records it still works."""
    known = sorted((e, i) for i, e in enumerate(exps) if e is not None)
    groups: List[List[int]] = []
    start = None
    for e, i in known:
        if start is None or e > start * (1.0 + _EXPOSURE_TOL):
            groups.append([])
            start = e
        groups[-1].append(i)
    unknown = [i for i, e in enumerate(exps) if e is None]
    if unknown:
        groups.append(unknown)
    return groups


def build_dark_temperature_model(frames: List[FrameInfo], degree: int = 2,
                                 light_exposure: Optional[float] = None) -> Optional[dict]:
    """Fit a per-pixel polynomial of dark signal vs. sensor temperature.

    Dark current depends on exposure as well as temperature, so only darks of **one**
    exposure go into a fit: the group closest to ``light_exposure`` (else the largest) that has
    enough frames and distinct temperatures. Darks of other exposures are left out and named in
    the log. The model records ``exposure`` (its group's, None when no header) and
    ``median_range`` (the levels of the darks it was fitted to).

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
    exps: List[Optional[float]] = []
    for f in frames:
        t = _frame_temp(f)
        if t is None:
            continue
        try:
            data, _ = load_frame(f.path, replicate_mono=False)
        except Exception:
            continue
        temps.append(t)
        imgs.append(data.astype(np.float64))
        exps.append(_frame_exposure(f))

    # one exposure only
    groups = _exposure_groups(exps)
    if not groups:
        return None
    exposures_present = sorted({round(e, 1) for e in exps if e is not None})

    def _usable(g: List[int]) -> bool:
        return len(g) >= degree + 2 and len({round(temps[i], 1) for i in g}) >= degree + 1

    candidates = [g for g in groups if _usable(g)]
    if candidates:
        def _distance(g: List[int]) -> float:
            known = [exps[i] for i in g if exps[i] is not None]
            if not known or not light_exposure:
                return 0.0
            return abs(float(np.log(float(np.mean(known)) / light_exposure)))
        best = min(candidates, key=lambda g: (_distance(g), -len(g)))
        if len(groups) > 1:
            kept = sorted({round(exps[i], 1) for i in best if exps[i] is not None})
            skipped = [e for e in exposures_present if e not in kept]
            if skipped:
                _log.info("dark temperature model: darks of mixed exposure; fitted to %s s, "
                          "left out %s s", kept, skipped)
                safe_print("  Dark temperature model: the darks have different exposures; using "
                           f"only the {kept[0]:g} s ones (left out: "
                           f"{', '.join(f'{e:g} s' for e in skipped)})")
        imgs = [imgs[i] for i in best]
        temps = [temps[i] for i in best]
        exps = [exps[i] for i in best]
    else:
        _log.warning("build_dark_temperature_model: no single exposure has >= %d frames at >= %d "
                     "distinct temperatures (exposures present: %s s)", degree + 2, degree + 1,
                     exposures_present or 'unknown')
        return None

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

    known_exps = [e for e in exps if e is not None]
    levels = [float(np.median(im)) for im in imgs]
    return {
        'coeffs': coeffs,
        'degree': degree,
        'shape': shape,
        'temp_range': (float(temps_arr.min()), float(temps_arr.max())),
        'n_frames': n,
        'n_temps': n_distinct,
        'exposure': float(np.mean(known_exps)) if known_exps else None,
        'median_range': (min(levels), max(levels)),
    }


def model_unusable_reason(model: dict, temp_c: float) -> Optional[str]:
    """Why ``model`` should not be evaluated at ``temp_c`` (more than ``MAX_EXTRAPOLATION_C``
    outside the temperatures it was fitted to), or None when it can be."""
    lo, hi = model['temp_range']
    if temp_c < lo - MAX_EXTRAPOLATION_C or temp_c > hi + MAX_EXTRAPOLATION_C:
        return (f"{temp_c:.1f} C is more than {MAX_EXTRAPOLATION_C:g} C outside the temperatures "
                f"of the darks it was fitted to ({lo:.1f}-{hi:.1f} C)")
    return None


def sample_unusable_reason(model: dict, sample: np.ndarray) -> Optional[str]:
    """Why the dark ``sample`` (the model evaluated at the lights' temperature) cannot be right:
    its level is far outside the levels of the darks the model was fitted to. None when it is
    plausible or the model does not record those levels."""
    rng = model.get('median_range')
    if not rng:
        return None
    level = float(np.median(sample))
    lo, hi = rng
    slack = max(0.5 * (hi - lo), 0.25 * max(abs(lo), abs(hi)), 50.0)
    if level < lo - slack or level > hi + slack:
        return (f"it comes out at {level:.0f} ADU, but the darks it was fitted to are "
                f"{lo:.0f}-{hi:.0f} ADU")
    return None


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
