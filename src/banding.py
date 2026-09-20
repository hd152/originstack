"""Row/column banding (fixed-pattern) removal (``--banding-removal``).

CMOS sensors add a per-row (and sometimes per-column) offset that changes from
frame to frame. Stacking averages it down but does not remove it, and once field
rotation and dither slide the frames over each other the leftover streaks turn
diagonal and are much harder to see and to fix. The correction belongs on each
calibrated frame, before debayering.

Estimate: for every row, the mean of the pixels that are not bright (a
``sigma``-MAD highlight cut keeps stars and nebulosity out of the average); the
band is that series minus its own smooth trend (a running median over
``smooth`` rows), so a real gradient across the frame is not mistaken for
banding. Offsets smaller than their own standard error are ignored (see ``_shrink``),
so a clean frame is left essentially as it was.

A Bayer mosaic is treated per colour plane: a sensor row's offset is common to
all its pixels whatever their filter colour, but each plane has its own level,
so the band is measured on each of the four sub-planes against that plane's own
trend and the two planes that share a row (or column) are averaged before the
correction is applied to both.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from scipy import ndimage
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    ndimage = None
    _HAS_SCIPY = False

_MIN_SMOOTH = 9


def _band_series(plane: np.ndarray, axis: int, sigma: float, smooth: int
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """(band, standard_error) along ``axis`` for one 2-D plane.

    ``axis=1`` averages across columns, giving one value per row (row banding);
    ``axis=0`` gives one value per column.
    """
    med = float(np.median(plane))
    # Per-pixel noise from neighbour differences along the averaging axis, not
    # from the plane's MAD: vignetting and sky gradients inflate a plane MAD
    # (786 ADU on a real Origin frame against a true pixel noise of ~455), which
    # made the standard errors -- and so the significance test -- too generous.
    d = np.diff(plane, axis=axis)
    pix = 1.4826 * float(np.median(np.abs(d - np.median(d)))) / np.sqrt(2.0)
    if not np.isfinite(pix) or pix <= 0:
        return np.zeros(plane.shape[1 - axis]), np.full(plane.shape[1 - axis], np.inf)
    # Highlight cut against the LOCAL level, so a vignetted corner is not
    # discarded wholesale as "bright" while a star on the flat sky is kept. A box
    # filter, not a gaussian: same purpose, ~7x cheaper, and this runs 8 times per
    # frame in every Phase 1 worker.
    local = ndimage.uniform_filter(plane, 17, mode='nearest')
    keep = plane <= local + sigma * pix
    vals, kept = plane, keep
    cnt = kept.sum(axis=axis).astype(np.float64)
    # Median, not mean, of what survives the highlight cut: a truncated *mean*
    # dips on every row that crosses a star (a variable number of pixels is cut)
    # and that dip read as banding -- measured on a synthetic star field, 60
    # stars left the row error at 31% of the injected offsets. Sort-based: cut
    # pixels become +inf and sort last, so the median is read at index cnt//2.
    srt = np.sort(np.where(kept, vals, np.inf), axis=axis)
    ci = cnt.astype(np.int64)
    lo_i = np.maximum((ci - 1) // 2, 0)
    hi_i = np.maximum(ci // 2, 0)
    idx = np.arange(len(ci))
    if axis == 1:
        series = 0.5 * (srt[idx, lo_i] + srt[idx, hi_i])
    else:
        series = 0.5 * (srt[lo_i, idx] + srt[hi_i, idx])
    series = np.where((ci > 0) & np.isfinite(series), series, med)
    w = int(max(_MIN_SMOOTH, smooth)) | 1
    # Trend: a small median first (takes out rows a star or a satellite trail
    # crosses), then a local quadratic fit over ``smooth`` rows. The fit, not a
    # running median, because at the ends a windowed median pads with something
    # (nearest / reflected samples) and either choice biased the trend by several
    # sigma over the first and last ~40 rows of a frame with a sky gradient;
    # least-squares polynomial fits (mode='interp') are unbiased there.
    if len(series) > w:
        pre = ndimage.median_filter(series, size=7, mode='nearest')
        from scipy.signal import savgol_filter
        trend = savgol_filter(pre, w, 2, mode='interp')
    else:
        trend = np.full_like(series, float(np.median(series)))
    se = 1.2533 * pix / np.sqrt(np.maximum(cnt, 1.0))
    return series - trend, se


def _shrink(band: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Gate offsets by significance: none below 1.5 standard errors, full above
    3.5, a linear ramp between. A soft threshold (subtract the SE) left a bias of
    about one SE on every real offset; this removes significant offsets fully
    while a pure-noise frame is changed by well under its own noise."""
    z = np.abs(band) / np.maximum(se, 1e-12)
    return band * np.clip((z - 1.5) / 2.0, 0.0, 1.0)


def banding_strength(mosaic: np.ndarray, sigma: float = 3.0, smooth: int = 65
                     ) -> Tuple[float, float, float]:
    """(row_rms, col_rms, pixel_noise) in ADU for a 2-D mosaic or one plane.

    For deciding whether the correction is worth turning on: banding matters
    when its rms is a sizeable fraction of the per-pixel noise divided by
    sqrt(row length) -- i.e. when it is visible against what averaging a row
    would give. Uses the raw (unshrunk) band so a clean frame reads ~0."""
    plane = np.asarray(mosaic, dtype=np.float32)
    if plane.ndim == 3:
        plane = plane.mean(axis=2)
    rb, _ = _band_series(plane, 1, sigma, smooth)
    cb, _ = _band_series(plane, 0, sigma, smooth)
    dd = np.diff(plane, axis=1)
    noise = 1.4826 * float(np.median(np.abs(dd - np.median(dd)))) / np.sqrt(2.0)
    return float(np.sqrt(np.mean(rb ** 2))), float(np.sqrt(np.mean(cb ** 2))), noise


def remove_banding_2d(plane: np.ndarray, amount: float = 1.0, sigma: float = 3.0,
                      smooth: int = 65, rows: bool = True, cols: bool = True
                      ) -> np.ndarray:
    """Banding removal on a single 2-D plane (mono or one colour)."""
    if not _HAS_SCIPY or amount <= 0 or plane.ndim != 2:
        return plane
    out = plane.astype(np.float32, copy=True)
    if rows:
        band, se = _band_series(out, 1, sigma, smooth)
        out -= (amount * _shrink(band, se)).astype(np.float32)[:, None]
    if cols:
        band, se = _band_series(out, 0, sigma, smooth)
        out -= (amount * _shrink(band, se)).astype(np.float32)[None, :]
    return out


def remove_banding_bayer(mosaic: np.ndarray, amount: float = 1.0, sigma: float = 3.0,
                         smooth: int = 65, rows: bool = True, cols: bool = True
                         ) -> np.ndarray:
    """Banding removal on a raw Bayer mosaic, per colour plane (see module doc).

    The four sub-planes are measured separately; the offsets of the two planes
    sharing a sensor row (or column) are averaged, shrunk against the combined
    standard error, and the same correction is subtracted from both."""
    if not _HAS_SCIPY or amount <= 0 or mosaic.ndim != 2:
        return mosaic
    out = mosaic.astype(np.float32, copy=True)
    H, W = out.shape
    H2, W2 = H - H % 2, W - W % 2
    sub = {(py, px): out[py:H2:2, px:W2:2] for py in (0, 1) for px in (0, 1)}

    if rows:
        for py in (0, 1):
            bands, ses = [], []
            for px in (0, 1):
                b, se = _band_series(sub[(py, px)], 1, sigma, smooth)
                bands.append(b)
                ses.append(se)
            band = 0.5 * (bands[0] + bands[1])
            se = 0.5 * np.sqrt(ses[0] ** 2 + ses[1] ** 2)
            corr = (amount * _shrink(band, se)).astype(np.float32)
            for px in (0, 1):
                sub[(py, px)] -= corr[:, None]
    if cols:
        for px in (0, 1):
            bands, ses = [], []
            for py in (0, 1):
                b, se = _band_series(sub[(py, px)], 0, sigma, smooth)
                bands.append(b)
                ses.append(se)
            band = 0.5 * (bands[0] + bands[1])
            se = 0.5 * np.sqrt(ses[0] ** 2 + ses[1] ** 2)
            corr = (amount * _shrink(band, se)).astype(np.float32)
            for py in (0, 1):
                sub[(py, px)] -= corr[None, :]
    return out


def remove_banding_rgb(rgb: np.ndarray, amount: float = 1.0, sigma: float = 3.0,
                       smooth: int = 65, rows: bool = True, cols: bool = True
                       ) -> np.ndarray:
    """Banding removal on an already-debayered (H, W, C) image, per channel.

    Debayering spreads a row offset across neighbouring rows, so this is less
    exact than working on the mosaic; used for RGB/TIFF inputs that never
    were a mosaic here."""
    if rgb.ndim != 3:
        return remove_banding_2d(rgb, amount, sigma, smooth, rows, cols)
    out = rgb.astype(np.float32, copy=True)
    for c in range(out.shape[2]):
        out[:, :, c] = remove_banding_2d(out[:, :, c], amount, sigma, smooth, rows, cols)
    return out
