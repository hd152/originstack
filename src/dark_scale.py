"""Per-session correction of the dark's glow scale.

Phase 1 scales the master dark by the exposure ratio (light / dark). Amplifier glow is not
linear in exposure and depends steeply on the sensor temperature, so a dark taken at one
temperature over- or under-subtracts the smooth glow in a session shot at another. On the
Origin (one library dark, taken at 31.3 C) a session at 19-21 C needed the dark scaled by a
further 1 - 0.67 = 0.33: the exposure-ratio scaling left 140-160 ADU peak-to-peak of
glow-shaped residual that no sky polynomial describes.

A few lights spread through the session are calibrated as Phase 1 would, 64 px block medians
are taken, and each is fitted with a quadratic sky surface plus a multiple ``delta`` of the
dark's own glow shape. The session's ``delta`` is the median over the probes; it is accepted
only when it is small enough to be physical and tightly determined, then shrunk toward zero.
``masters['dark_scale_adjust'] = 1 + delta`` multiplies the dark scale of every frame.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.models import Config
from src.utils import safe_print

_log = logging.getLogger("originstack")

_NSIGMA = 3.0
_ITERS = 6


def _block_medians(a: np.ndarray, block: int) -> np.ndarray:
    hb, wb = a.shape[0] // block, a.shape[1] // block
    return np.median(a[:hb * block, :wb * block].reshape(hb, block, wb, block), axis=(1, 3))


def _poly2_design(hb: int, wb: int) -> np.ndarray:
    yy, xx = np.mgrid[0:hb, 0:wb]
    yn = (yy - (hb - 1) / 2) / max(hb / 2, 1)
    xn = (xx - (wb - 1) / 2) / max(wb / 2, 1)
    return np.stack([np.ones_like(xn), xn, yn, xn * xn, xn * yn, yn * yn],
                    axis=-1).reshape(-1, 6).astype(np.float64)


def fit_glow_delta(cal_blocks: np.ndarray, glow_blocks: np.ndarray
                   ) -> Optional[Tuple[float, float]]:
    """``cal ~ poly2(x, y) + delta * glow`` on block medians; returns ``(delta, se)``.

    Iteratively 3-sigma-clipped least squares (stars, nebulosity and bad blocks fall out);
    ``se`` is the standard error of ``delta`` from the residual scatter. ``delta`` is the
    fractional change of the dark scale that removes the glow-shaped residual. ``None`` when
    the grid is too small or the glow shape is (nearly) a quadratic and so unidentifiable."""
    if cal_blocks.shape != glow_blocks.shape or cal_blocks.size < 40:
        return None
    X = np.concatenate([_poly2_design(*cal_blocks.shape), glow_blocks.reshape(-1, 1)], axis=1)
    y = cal_blocks.reshape(-1).astype(np.float64)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    sg = 0.0
    coef = None
    for _ in range(_ITERS):
        if ok.sum() < 20:
            return None
        coef, *_r = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        r = y - X @ coef
        sg = 1.4826 * float(np.median(np.abs(r[ok] - np.median(r[ok]))))
        if sg <= 0:
            return None
        ok = ok & (np.abs(r) < _NSIGMA * sg)
    try:
        cov = np.linalg.inv(X[ok].T @ X[ok]) * sg ** 2
    except np.linalg.LinAlgError:
        return None
    var = float(cov[-1, -1])
    if not np.isfinite(var) or var <= 0:
        return None
    return float(coef[-1]), float(np.sqrt(var))


def decide_dark_scale(deltas: List[float], ses: List[float]) -> Optional[Dict[str, float]]:
    """Session verdict from the per-probe fits, or None to leave the scale alone.

    Accepted only when the median ``delta`` is physical (|delta| < ``DARK_SCALE_MAX_DELTA``),
    each fit is tight (median se < ``DARK_SCALE_MAX_SE``) and the probes agree (std below
    ``DARK_SCALE_MAX_SPREAD``). The accepted value is shrunk toward zero by
    ``tau^2 / (tau^2 + se^2)`` and dropped when under ``DARK_SCALE_MIN_APPLY`` (the usual
    warm session: nothing changes, bit-identical to before)."""
    if len(deltas) < Config.DARK_SCALE_MIN_PROBES:
        return None
    d = np.asarray(deltas, dtype=np.float64)
    se = float(np.median(ses))
    delta = float(np.median(d))
    spread = float(np.std(d))
    if (abs(delta) >= Config.DARK_SCALE_MAX_DELTA or se >= Config.DARK_SCALE_MAX_SE
            or spread >= Config.DARK_SCALE_MAX_SPREAD):
        return None
    tau2 = Config.DARK_SCALE_PRIOR_SD ** 2
    used = delta * tau2 / (tau2 + se ** 2)
    if abs(used) < Config.DARK_SCALE_MIN_APPLY:
        return None
    return {'delta': delta, 'se': se, 'spread': spread, 'used': float(used), 'n': len(deltas)}


def measure_session_dark_scale(lights: list, masters: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Fit the dark's glow scale on probe lights; see the module docstring.

    Needs a master dark with a known exposure time (and optionally bias and flat); mono and
    Bayer mosaics alike (block medians are taken on the 2-D mosaic, the fit's polynomial
    absorbs the colour pattern). Any failure returns None: the exposure-ratio scale stays."""
    dark = masters.get('dark')
    dark_et = masters.get('dark_exptime')
    if dark is None or not dark_et or dark_et <= 0:
        return None
    if len(lights) < Config.DARK_SCALE_MIN_FRAMES:
        return None
    from src.io_fits import load_frame
    bias = masters.get('bias')
    flat = masters.get('flat')
    block = Config.DARK_SCALE_BLOCK
    try:
        flat_norm = None
        if flat is not None and flat.shape == dark.shape:
            med = float(np.median(flat))
            if med > 1e-6:
                flat_norm = np.clip(flat / med, 0.4, 2.5).astype(np.float32)
        dark_term = dark.astype(np.float32)           # what calibrate_frame subtracts, per unit scale
        if bias is not None:
            dark_term = dark_term - bias.astype(np.float32)
        glow = dark_term / flat_norm if flat_norm is not None else dark_term
        glow_b = _block_medians(glow, block)
    except Exception as e:
        _log.debug("dark scale: glow shape failed: %s", e)
        return None

    n = len(lights)
    k = min(Config.DARK_SCALE_PROBE_FRAMES, n)
    idxs = sorted({int((j + 0.5) * n / k) for j in range(k)})
    deltas: List[float] = []
    ses: List[float] = []
    for i in idxs:
        try:
            data, hdr = load_frame(lights[i].path, replicate_mono=False)
            if data is None or data.ndim != 2 or data.shape != dark.shape:
                continue
            et = float(hdr.get('EXPTIME', 0) or 0)
            s = et / dark_et if et > 0 else 1.0
            cal = data.astype(np.float32)
            if bias is not None:
                cal = cal - bias
            cal = cal - s * dark_term
            if flat_norm is not None:
                cal = cal / flat_norm
            fit = fit_glow_delta(_block_medians(cal, block), glow_b)
            if fit is not None:
                # the fitted multiple of the unscaled glow is s * delta
                fit = (fit[0] / s, fit[1] / s)
        except Exception as e:
            _log.debug("dark scale: probe %s failed: %s", i, e)
            continue
        if fit is not None:
            deltas.append(fit[0])
            ses.append(fit[1])
    return decide_dark_scale(deltas, ses)


def apply_session_dark_scale(masters: Dict[str, Any], lights: list, args=None) -> None:
    """Measure the session's glow scale and store ``masters['dark_scale_adjust']``.

    Skipped (scale stays the exposure ratio) when ``args.session_dark_scale`` is false or the
    dark came from the temperature model, which is already evaluated at the lights'
    temperature."""
    if args is not None and not getattr(args, 'session_dark_scale', True):
        return
    if masters.get('dark_from_temp_model'):
        return
    res = measure_session_dark_scale(lights, masters)
    if res is None:
        return
    masters['dark_scale_adjust'] = 1.0 + res['used']
    safe_print(f"  Dark glow scale: x{1.0 + res['used']:.2f} on top of the exposure ratio "
               f"(fit {res['delta']:+.2f} +/- {res['se']:.2f} from {res['n']} frames, "
               f"sensor temperature differs from the dark's)")
