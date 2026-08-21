"""Saturated star core repair.

Bright stars clip to a flat white disk once their peak hits the sensor / stack
ceiling: the true PSF peak is lost and the core carries no colour. This module
finds saturated cores (connected clusters of near-max pixels), fits a Moffat
profile to each star's *unsaturated* wing **per channel**, and refills the
clipped core with the model. Because each RGB channel is fit independently, the
rebuilt core keeps the star's real colour (the wing colour) instead of pure
white, and the peak is restored to a natural rounded profile.

Moffat  I(r) = bg + A / (1 + (r/alpha)^2)^beta  is the standard stellar PSF
model (Gaussian is its beta->inf limit); its heavier wings fit real optics far
better, and it is what makes the rebuilt core blend seamlessly into the
surviving wing.

Only pixels inside the saturated core are altered; everything else is returned
untouched. Requires scipy (a hard dependency of the pipeline).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from src.utils import safe_print

try:
    from scipy import ndimage as _ndi
    from scipy.optimize import curve_fit
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - scipy is a hard dep in practice
    _ndi = None
    curve_fit = None
    _HAS_SCIPY = False

try:
    import astro_native as _native
    _HAS_NATIVE = hasattr(_native, 'fit_moffat_native')
except Exception:  # pragma: no cover
    _native = None
    _HAS_NATIVE = False


def _moffat(r: np.ndarray, amp: float, alpha: float, beta: float) -> np.ndarray:
    return amp / np.power(1.0 + (r / alpha) ** 2, beta)


def _fit_moffat_wing_numpy(r: np.ndarray, v: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """scipy.optimize.curve_fit fallback -- see `_fit_moffat_wing`."""
    if curve_fit is None or len(r) < 6:
        return None
    amp0 = float(np.max(v))
    if amp0 <= 0:
        return None
    # alpha ~ radius where intensity falls to half of the extrapolated peak.
    alpha0 = max(float(np.median(r)), 1.0)
    try:
        popt, _ = curve_fit(
            _moffat, r, v, p0=[amp0 * 2.0, alpha0, 2.5],
            bounds=([amp0 * 0.5, 0.5, 1.0], [amp0 * 50.0, 50.0, 8.0]),
            maxfev=2000)
        amp, alpha, beta = float(popt[0]), float(popt[1]), float(popt[2])
        if not (np.isfinite(amp) and np.isfinite(alpha) and np.isfinite(beta)):
            return None
        return amp, alpha, beta
    except Exception:
        return None


def _fit_moffat_wing(r: np.ndarray, v: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Fit background-subtracted intensities ``v`` at radii ``r`` to a Moffat.
    Returns (amp, alpha, beta) or None on failure.

    Native path (`fit_moffat_native`): scipy's bounded curve_fit calls back
    into this module's Python `_moffat` every optimizer iteration -- with up
    to 800 stars x 3 channels per repair run, that Python-callback overhead
    dominates, not the fit's own arithmetic. The native kernel is a
    from-scratch bounded Levenberg-Marquardt with the Moffat model and its
    Jacobian both inlined, so no callback ever crosses back into Python.
    """
    if _HAS_NATIVE:
        res = _native.fit_moffat_native(
            np.ascontiguousarray(r, dtype=np.float64),
            np.ascontiguousarray(v, dtype=np.float64))
        if res is None:
            return None
        amp, alpha, beta = res
        if not (np.isfinite(amp) and np.isfinite(alpha) and np.isfinite(beta)):
            return None
        return float(amp), float(alpha), float(beta)
    return _fit_moffat_wing_numpy(r, v)


def repair_saturated_stars(
    rgb: np.ndarray,
    sat_frac: float = 0.92,
    min_core: int = 3,
    max_core: int = 4000,
    wing_radius: int = 18,
    max_stars: int = 800,
    verbose: bool = False,
) -> np.ndarray:
    """Rebuild saturated (flat-top) star cores from their unsaturated wings.

    Args:
        rgb: (H, W, 3) float32 linear image.
        sat_frac: a pixel counts as saturated at >= sat_frac * channel-max
            (luminance-based core detection uses the luma max).
        min_core/max_core: accept saturated blobs with this many pixels
            (skips single hot pixels and huge bloomed regions).
        wing_radius: half-size of the fit window around each core (px).

    Returns a new (H, W, 3) float32 array; unsaturated pixels are unchanged.
    """
    if not _HAS_SCIPY:
        safe_print("  Saturated star repair requires scipy — skipping")
        return rgb
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return rgb

    H, W, _ = rgb.shape
    out = rgb.astype(np.float32, copy=True)
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    lum_max = float(lum.max())
    if lum_max <= 0:
        return out
    sat_thresh = sat_frac * lum_max
    core_bin = lum >= sat_thresh
    if not core_bin.any():
        if verbose:
            safe_print("  Saturated star repair: no saturated cores found")
        return out

    labels, n = _ndi.label(core_bin)
    if n == 0:
        return out
    slices = _ndi.find_objects(labels)
    # Per-channel saturation thresholds for masking each channel's own core.
    ch_sat = [sat_frac * float(rgb[:, :, c].max()) for c in range(3)]

    r = int(wing_radius)
    n_repaired = 0
    # Process brightest-core-first, cap total to bound cost.
    order = sorted(range(n), key=lambda i: -(labels[slices[i]] == (i + 1)).sum())
    for li in order[:max_stars]:
        sl = slices[li]
        blob = labels[sl] == (li + 1)
        core_size = int(blob.sum())
        if core_size < min_core or core_size > max_core:
            continue
        ys, xs = np.nonzero(blob)
        cy = float(ys.mean() + sl[0].start)
        cx = float(xs.mean() + sl[1].start)
        iy, ix = int(round(cy)), int(round(cx))
        if iy < r or iy >= H - r or ix < r or ix >= W - r:
            continue

        y0, y1 = iy - r, iy + r + 1
        x0, x1 = ix - r, ix + r + 1
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rr = np.hypot(yy - cy, xx - cx)

        repaired_any = False
        for c in range(3):
            win = out[y0:y1, x0:x1, c].astype(np.float64)
            core = win >= ch_sat[c]
            if int(core.sum()) < min_core:
                continue
            # Background from the outer annulus; wing = unsaturated ring
            # between the core and the window edge.
            outer = rr >= (r - 3)
            bg = float(np.median(win[outer])) if outer.any() else float(np.median(win))
            wing = (~core) & (rr <= r) & (win > bg)
            if int(wing.sum()) < 8:
                continue
            fit = _fit_moffat_wing(rr[wing], win[wing] - bg)
            if fit is None:
                continue
            amp, alpha, beta = fit
            model = bg + _moffat(rr, amp, alpha, beta)
            # Only lift the clipped core; never pull a pixel down.
            repl = core & (model > win)
            if repl.any():
                win[repl] = model[repl]
                out[y0:y1, x0:x1, c] = win.astype(np.float32)
                repaired_any = True
        if repaired_any:
            n_repaired += 1

    if verbose or n_repaired:
        safe_print(f"  Saturated star repair: rebuilt {n_repaired} star core(s)")
    return out
