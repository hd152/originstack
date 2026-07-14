"""Per-frame Local Normalization (additive background matching).

Frames taken across a session drift in background: the Moon rises, light
pollution shifts, thin cloud rolls through, gradients rotate on an alt-az
mount. A single post-stack background extraction (DBE) cannot undo *per-frame*
differences — by then the frames are already averaged together, so a bright
gradient in a few subs has permanently tilted the stack and skewed pixel
rejection.

Local Normalization fixes this **before** the combine. For each aligned frame it
estimates a smooth per-channel background (star-rejected coarse mesh), builds a
robust **median background across all frames** as the reference, and subtracts
each frame's *deviation* from that reference:

    frame_i  ->  frame_i - (bg_i - bg_median)

The component common to every frame (real extended nebulosity, the sky pedestal)
lives in ``bg_median`` and cancels out, so genuine signal is preserved; only the
per-frame gradient/level *variation* is removed. Backgrounds end up mutually
consistent, which both flattens the stack and sharpens sigma-clip rejection
(outliers are now measured against a matched background).

This is the additive counterpart of PixInsight's LocalNormalization. It replaces
the old ``--local-normalize`` (which divided by local sigma and amplified
background noise — removed for that reason); this version only shifts levels,
never scales noise.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.utils import safe_print

try:
    from scipy.ndimage import zoom as _zoom, gaussian_filter as _gauss
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _zoom = _gauss = None
    _HAS_SCIPY = False


def _coarse_background(frame: np.ndarray, grid: int, pct: float) -> np.ndarray:
    """Star-rejected coarse background grid (grid, grid, C).

    Per mesh cell the background is a low percentile of the cell pixels — stars
    and bright objects sit in the high tail and are excluded, so the estimate
    tracks the sky level without a separate star mask."""
    h, w, c = frame.shape
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    bg = np.empty((grid, grid, c), dtype=np.float32)
    for iy in range(grid):
        y0, y1 = ys[iy], max(ys[iy] + 1, ys[iy + 1])
        for ix in range(grid):
            x0, x1 = xs[ix], max(xs[ix] + 1, xs[ix + 1])
            cell = frame[y0:y1, x0:x1, :].reshape(-1, c)
            bg[iy, ix] = np.percentile(cell, pct, axis=0)
    return bg


def _upsample(grid_bg: np.ndarray, h: int, w: int) -> np.ndarray:
    """Smoothly upsample a (g, g, C) grid to (h, w, C)."""
    g = grid_bg.shape[0]
    zy, zx = h / g, w / g
    out = _zoom(grid_bg, (zy, zx, 1), order=1)[:h, :w, :]
    return out.astype(np.float32)


def local_normalize_stack(mem_aligned: np.ndarray, grid: int = 24,
                          bg_percentile: float = 30.0,
                          verbose: bool = False) -> int:
    """Additively background-match every frame of an aligned stack in place.

    Args:
        mem_aligned: (N, H, W, C) float32 array/memmap of aligned, cropped
            frames. Modified in place.
        grid: coarse background mesh resolution per axis.
        bg_percentile: per-cell percentile taken as the sky level.

    Returns the number of frames normalized (0 if unavailable / degenerate).
    """
    if not _HAS_SCIPY:
        safe_print("  Local normalization requires scipy — skipping")
        return 0
    N = mem_aligned.shape[0]
    if N < 2:
        return 0
    H, W, C = mem_aligned.shape[1:]

    # Pass 1: coarse background grid per frame (small — (N, g, g, C)).
    grids = np.empty((N, grid, grid, C), dtype=np.float32)
    for i in range(N):
        grids[i] = _coarse_background(np.asarray(mem_aligned[i]), grid, bg_percentile)

    # Robust reference = per-cell median background across all frames.
    ref = np.median(grids, axis=0)

    # Pass 2: subtract each frame's smooth deviation from the reference.
    max_shift = 0.0
    for i in range(N):
        dev = grids[i] - ref                       # (g, g, C) coarse deviation
        max_shift = max(max_shift, float(np.abs(dev).max()))
        corr = _upsample(_gauss(dev, sigma=(0.6, 0.6, 0)), H, W)
        frame = np.asarray(mem_aligned[i], dtype=np.float32)
        frame -= corr
        mem_aligned[i] = frame
    if hasattr(mem_aligned, 'flush'):
        try:
            mem_aligned.flush()
        except Exception:
            pass

    if verbose:
        safe_print(f"  Local normalization: matched {N} frame backgrounds "
                   f"(max correction {max_shift:.1f} ADU)")
    return N
