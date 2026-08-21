"""Satellite / aircraft trail rejection (per frame, pre-stack).

Sigma-clip and friends only reject a trail where enough *other* frames cover
the same pixel cleanly — which fails at low frame counts (a bright Starlink
streak in 1 of 5 subs survives into the stack). This module finds long straight
trails in each frame *before* it enters the stack and erases them by filling the
streak pixels with the local background (normalised-convolution inpaint), so the
result is robust for any stacking method and any frame count.

Detection: bright-pixel threshold -> straight-line segment search (own
Hough-transform implementation, no external dependency) to find segments
longer than a fraction of the frame. Compact sources (stars) never form long
collinear runs, so they are left alone. A global area-fraction guard aborts
the fill if detection went haywire (e.g. a diffraction spike lattice), so a
bad detection can never gut the frame.

``_detect_line_segments`` is a standard (non-probabilistic) Hough transform:
accumulate votes for every (theta, rho) line through each bright pixel, take
accumulator peaks with non-max suppression, then walk each peak's line
gathering nearby points into gap-tolerant runs. This differs from skimage's
``probabilistic_hough_line`` (which randomly samples points and is itself
non-deterministic run-to-run) -- it's deterministic and was validated to find
the same synthetic trails skimage finds (see tests/test_trail_reject.py), not
to reproduce skimage's exact segment coordinates, which aren't reproducible
even between two skimage runs on the same input.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.utils import safe_print

try:
    from scipy.ndimage import binary_dilation, gaussian_filter
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    binary_dilation = gaussian_filter = None
    _HAS_SCIPY = False

try:
    import astro_native as _native
    _HAS_NATIVE = hasattr(_native, 'bresenham_line_native')
except Exception:  # pragma: no cover
    _native = None
    _HAS_NATIVE = False

# Safety cap on bright-pixel count fed into the Hough accumulator -- a frame
# this saturated is never a clean trail-on-empty-sky case, and letting it
# through would blow up the O(n_points * n_theta) vote pass for no benefit.
_MAX_HOUGH_POINTS = 200_000

_N_THETA = 360  # 0.5 deg resolution over [-pi/2, pi/2)


def _bresenham_line(r0: int, c0: int, r1: int, c1: int) -> Tuple[np.ndarray, np.ndarray]:
    """Integer pixel coordinates from (r0, c0) to (r1, c1) inclusive, via
    Bresenham's algorithm -- matches ``skimage.draw.line`` pixel-for-pixel
    (validated in tests/test_trail_reject.py; same symmetric ``2*dr - dc``
    error-term formulation skimage's own implementation uses, which avoids
    the tie-break ambiguity a naive ``dc // 2`` initial-error variant has).

    Zero-vectorization case (a sequential line-walk, no numpy call it could
    hide behind), so the native kernel is a direct port of this same
    algorithm rather than a redesign -- see `bresenham_line_native`."""
    if _HAS_NATIVE:
        rr, cc = _native.bresenham_line_native(int(r0), int(c0), int(r1), int(c1))
        return np.asarray(rr), np.asarray(cc)
    r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
    r, c = r0, c0
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if (r1 - r0) > 0 else -1
    sc = 1 if (c1 - c0) > 0 else -1
    steep = dr > dc
    if steep:
        r, c = c, r
        dr, dc = dc, dr
        sr, sc = sc, sr

    d = (2 * dr) - dc
    n = dc + 1
    rr = np.empty(n, dtype=np.intp)
    cc = np.empty(n, dtype=np.intp)
    for i in range(dc):
        if steep:
            rr[i], cc[i] = c, r
        else:
            rr[i], cc[i] = r, c
        while d >= 0:
            r += sr
            d -= 2 * dc
        c += sc
        d += 2 * dr
    rr[dc], cc[dc] = r1, c1
    return rr, cc


def _detect_line_segments(binary: np.ndarray, threshold: int = 10,
                          line_length: int = 50, line_gap: int = 8,
                          max_peaks: int = 20,
                          ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Find straight line segments in a binary image. Returns a list of
    ``((x0, y0), (x1, y1))`` pairs, matching
    ``skimage.transform.probabilistic_hough_line``'s return format."""
    ys, xs = np.nonzero(binary)
    n_points = len(xs)
    if n_points == 0 or n_points > _MAX_HOUGH_POINTS:
        return []

    H, W = binary.shape
    diag = int(np.ceil(np.hypot(H, W)))
    thetas = np.linspace(-np.pi / 2, np.pi / 2, _N_THETA, endpoint=False)
    cos_t = np.cos(thetas).astype(np.float32)
    sin_t = np.sin(thetas).astype(np.float32)

    xs_f = xs.astype(np.float32)
    ys_f = ys.astype(np.float32)
    # rho[i, j] = x_i*cos(theta_j) + y_i*sin(theta_j)
    rho_vals = np.outer(xs_f, cos_t) + np.outer(ys_f, sin_t)
    rho_idx = np.round(rho_vals).astype(np.int64) + diag
    n_rho = 2 * diag + 1

    accumulator = np.zeros((n_rho, _N_THETA), dtype=np.int32)
    theta_idx = np.broadcast_to(np.arange(_N_THETA), rho_idx.shape)
    np.add.at(accumulator, (rho_idx.ravel(), theta_idx.ravel()), 1)

    # Perpendicular distance of every point from a candidate (theta, rho) line.
    def _line_points(theta_i: int, rho: int, tol: float = 1.5):
        dist = np.abs(xs_f * cos_t[theta_i] + ys_f * sin_t[theta_i] - rho)
        return np.nonzero(dist <= tol)[0]

    segments: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    rho_suppress = max(8, line_gap * 2)
    theta_suppress = max(6, _N_THETA // 45)  # ~4 deg

    for _ in range(max_peaks):
        peak_idx = int(np.argmax(accumulator))
        rho_i, theta_i = np.unravel_index(peak_idx, accumulator.shape)
        votes = accumulator[rho_i, theta_i]
        if votes < threshold:
            break

        rho = rho_i - diag
        cand = _line_points(theta_i, rho)
        if len(cand) >= 2:
            # Project candidate points onto the line direction to order them
            # and split into gap-tolerant runs.
            t = -xs_f[cand] * sin_t[theta_i] + ys_f[cand] * cos_t[theta_i]
            order = np.argsort(t)
            cand = cand[order]
            t = t[order]
            gaps = np.diff(t)
            breaks = np.nonzero(gaps > line_gap)[0]
            run_starts = np.concatenate(([0], breaks + 1))
            run_ends = np.concatenate((breaks, [len(cand) - 1]))
            for rs, re in zip(run_starts, run_ends):
                run = cand[rs:re + 1]
                if len(run) < 2:
                    continue
                i0, i1 = run[0], run[-1]
                seg_len = np.hypot(xs_f[i1] - xs_f[i0], ys_f[i1] - ys_f[i0])
                if seg_len >= line_length:
                    segments.append(((int(xs[i0]), int(ys[i0])),
                                     (int(xs[i1]), int(ys[i1]))))

        # Non-max suppression around this peak so the next iteration finds a
        # genuinely different line instead of the same trail again.
        r_lo, r_hi = max(0, rho_i - rho_suppress), min(n_rho, rho_i + rho_suppress + 1)
        th_lo, th_hi = max(0, theta_i - theta_suppress), min(_N_THETA, theta_i + theta_suppress + 1)
        accumulator[r_lo:r_hi, th_lo:th_hi] = 0

    return segments


def detect_trail_mask(lum: np.ndarray, min_len_frac: float = 0.18,
                      thresh_sigma: float = 3.0, width: int = 4,
                      max_area_frac: float = 0.08) -> Tuple[Optional[np.ndarray], int]:
    """Return (boolean trail mask, n_segments) for a luminance frame, or
    (None, 0) if no trail was found / detection is unavailable."""
    if not _HAS_SCIPY:
        return None, 0
    H, W = lum.shape[:2]
    lum = np.asarray(lum, dtype=np.float32)

    med = float(np.median(lum))
    sig = 1.4826 * float(np.median(np.abs(lum - med)))
    if sig <= 0:
        sig = float(np.std(lum)) or 1.0
    binary = lum > (med + thresh_sigma * sig)
    if binary.sum() < H:  # essentially nothing bright
        return None, 0

    min_len = max(30, int(min_len_frac * min(H, W)))
    line_gap = max(3, int(0.02 * min(H, W)))
    try:
        segments = _detect_line_segments(binary, threshold=10, line_length=min_len,
                                         line_gap=line_gap)
    except Exception:
        return None, 0
    if not segments:
        return None, 0

    mask = np.zeros((H, W), dtype=bool)
    for (x0, y0), (x1, y1) in segments:
        rr, cc = _bresenham_line(int(y0), int(x0), int(y1), int(x1))
        rr = np.clip(rr, 0, H - 1)
        cc = np.clip(cc, 0, W - 1)
        mask[rr, cc] = True
    if width > 0:
        mask = binary_dilation(mask, iterations=int(width))

    # Guard: if the "trail" covers too much of the frame the detection is
    # almost certainly wrong (dense field, big galaxy edge) — do nothing.
    if mask.mean() > max_area_frac:
        return None, 0
    return mask, len(segments)


def reject_trails(rgb: np.ndarray, lum: Optional[np.ndarray] = None,
                  min_len_frac: float = 0.18, width: int = 4,
                  fill_sigma: float = 12.0,
                  verbose: bool = False) -> Tuple[np.ndarray, int]:
    """Detect straight trails and inpaint them with local background.

    Args:
        rgb: (H, W, 3) float32 frame.
        lum: optional precomputed luminance (else derived).
        fill_sigma: Gaussian bandwidth (px) of the normalised-convolution
            background used to refill trail pixels.

    Returns (cleaned_rgb, n_segments). Unchanged (and n=0) when no trail found.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return rgb, 0
    if lum is None:
        lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    mask, n = detect_trail_mask(lum, min_len_frac=min_len_frac, width=width)
    if mask is None or n == 0:
        return rgb, 0

    out = rgb.astype(np.float32, copy=True)
    keep = (~mask).astype(np.float32)
    # Normalised convolution: smooth background from the non-trail pixels only,
    # so the fill never smears trail flux back into the gap.
    denom = gaussian_filter(keep, sigma=fill_sigma)
    denom = np.maximum(denom, 1e-6)
    for c in range(3):
        ch = out[:, :, c]
        bg = gaussian_filter(ch * keep, sigma=fill_sigma) / denom
        ch[mask] = bg[mask]
        out[:, :, c] = ch
    if verbose:
        safe_print(f"      trail rejection: masked {n} segment(s), "
                   f"{int(mask.sum())} px inpainted")
    return out, n
