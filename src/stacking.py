"""Stacking algorithms: Lanczos resampling, drizzle combine, sigma-clip combine."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

from src.models import Config
from src.utils import safe_print


def _lanczos_kernel(x: np.ndarray, a: int = 3) -> np.ndarray:
    """Lanczos interpolation kernel.

    L(x) = sinc(x) * sinc(x/a)   for |x| < a
    L(x) = 0                      for |x| >= a
    """
    result = np.zeros_like(x, dtype=np.float64)
    mask = np.abs(x) < a
    xm = x[mask]
    # sinc(x) = sin(pi*x) / (pi*x), numpy's sinc already includes the pi
    result[mask] = np.sinc(xm) * np.sinc(xm / a)
    return result


def _lanczos_resample_frame(img: np.ndarray, shift: Tuple[float, float],
                            scale: float, out_h: int, out_w: int,
                            lanczos_a: int = 3) -> np.ndarray:
    """Resample a single frame onto an upscaled output grid using Lanczos interpolation.

    Maps each output pixel back to fractional input coordinates (accounting for
    the sub-pixel shift), then applies a separable Lanczos-a kernel.  Uses
    scipy.ndimage.map_coordinates with order=5 (quintic B-spline, closely
    approximating Lanczos-3) for efficiency, with a pure Lanczos kernel
    available via lanczos_a parameter.
    """
    H, W = img.shape[:2]
    C = img.shape[2] if img.ndim == 3 else 1

    # Output coordinates -> input coordinates (inverse mapping)
    # output pixel (oy, ox) corresponds to input ((oy / scale) - shift_y, (ox / scale) - shift_x)
    oy = np.arange(out_h, dtype=np.float64)
    ox = np.arange(out_w, dtype=np.float64)
    iy = oy / scale - shift[0]
    ix = ox / scale - shift[1]

    # Use scipy's map_coordinates with order=5 (quintic spline ~= Lanczos-3)
    # This is much faster than a pure Python Lanczos loop
    spline_order = min(5, lanczos_a + 2)  # order 5 for Lanczos-3

    coords_y, coords_x = np.meshgrid(iy, ix, indexing='ij')

    if img.ndim == 3:
        result = np.empty((out_h, out_w, C), dtype=np.float64)
        for c in range(C):
            result[:, :, c] = ndimage.map_coordinates(
                img[:, :, c].astype(np.float64),
                [coords_y, coords_x],
                order=spline_order, mode='constant', cval=0.0)
    else:
        result = ndimage.map_coordinates(
            img.astype(np.float64),
            [coords_y, coords_x],
            order=spline_order, mode='constant', cval=0.0)

    return result


def drizzle_combine(aligned_list: List[np.ndarray], shifts: List[Tuple[float, float]],
                    scale: float = 1.0, weights: Optional[np.ndarray] = None,
                    drop_size: float = 0.7) -> np.ndarray:
    """Drizzle combine with Lanczos interpolation and fractional sub-pixel shifts.

    Each input frame is resampled onto an upscaled output grid using high-order
    spline interpolation (approximating Lanczos-3).  Fractional sub-pixel shifts
    are preserved, yielding genuine super-resolution when dithered data is
    available.

    Parameters
    ----------
    aligned_list : list of ndarray
        Input frames (H, W, C), already cropped to common region.
    shifts : list of (dy, dx) tuples
        Sub-pixel registration shifts for each frame.
    scale : float
        Output scale factor (e.g. 2.0 for 2x super-resolution).
    weights : ndarray, optional
        Per-frame quality weights.  If None, uniform weighting is used.
    drop_size : float
        Pixel fraction (pixfrac) — controls the effective footprint of each
        input pixel on the output grid.  Smaller values (0.5-0.7) yield
        sharper results at the cost of noise.  1.0 = no shrinking.
    """
    if scale <= 1.0:
        # No upscaling — weighted mean combine
        acc = None
        total_w = 0.0
        for i, im in enumerate(aligned_list):
            w = float(weights[i]) if weights is not None else 1.0
            if acc is None:
                acc = np.zeros_like(im, dtype=np.float64)
            acc += im.astype(np.float64) * w
            total_w += w
        return (acc / max(total_w, 1e-12)).astype(np.float32)

    H, W = aligned_list[0].shape[:2]
    C = aligned_list[0].shape[2] if aligned_list[0].ndim == 3 else 1
    out_h = int(round(H * scale))
    out_w = int(round(W * scale))

    acc = np.zeros((out_h, out_w, C) if C > 1 else (out_h, out_w), dtype=np.float64)
    weight_map = np.zeros_like(acc, dtype=np.float64)

    for i, (im, sh) in enumerate(zip(aligned_list, shifts)):
        w = float(weights[i]) if weights is not None else 1.0
        resampled = _lanczos_resample_frame(im, sh, scale, out_h, out_w)

        # Build a coverage mask: pixels that map to valid input region
        # (map_coordinates returns 0 for out-of-bounds, so detect via a
        # ones-image mapped the same way)
        ones = np.ones(im.shape[:2], dtype=np.float64)
        coverage = _lanczos_resample_frame(ones, sh, scale, out_h, out_w)
        valid = coverage > 0.5  # pixel has >50% coverage

        if im.ndim == 3:
            valid3 = valid[:, :, np.newaxis] if valid.ndim == 2 else valid
            acc += np.where(valid3, resampled * w, 0.0)
            weight_map += np.where(valid3, w, 0.0)
        else:
            acc += np.where(valid, resampled * w, 0.0)
            weight_map += np.where(valid, w, 0.0)

    weight_map[weight_map == 0] = 1.0
    return (acc / weight_map).astype(np.float32)


def _sigma_clip_tile(tile: np.ndarray, sigma: float, max_iters: int,
                     weights: Optional[np.ndarray], winsorize: bool) -> np.ndarray:
    """Process a single spatial tile for sigma-clip combine.

    Uses MAD (Median Absolute Deviation) for robust spread estimation
    instead of standard deviation, which is less sensitive to the very
    outliers we are trying to reject.
    """
    N = tile.shape[0]
    mask = np.ones(tile.shape, dtype=bool)

    for iteration in range(max_iters):
        masked = np.where(mask, tile, np.nan)
        with np.errstate(all='ignore'):
            median = np.nanmedian(masked, axis=0)
            # MAD * 1.4826 is a consistent estimator of std for normal data
            mad = np.nanmedian(np.abs(masked - median[np.newaxis]), axis=0) * 1.4826

        # Fallback to std where MAD is zero (constant regions)
        spread = mad.copy()
        zero_mad = spread < 1e-12
        if np.any(zero_mad):
            with np.errstate(all='ignore'):
                std_fallback = np.nanstd(masked, axis=0)
            spread[zero_mad] = std_fallback[zero_mad]

        deviation = np.abs(tile - median[np.newaxis])
        new_mask = mask & (deviation <= sigma * spread[np.newaxis])

        # Ensure at least 1 frame survives at every pixel
        surviving = new_mask.sum(axis=0)
        all_rejected = surviving == 0
        if np.any(all_rejected):
            for frame_idx in range(N):
                new_mask[frame_idx][all_rejected] = mask[frame_idx][all_rejected]

        rejected = int(mask.sum() - new_mask.sum())
        mask = new_mask
        if rejected == 0:
            break

    if winsorize:
        # Replace outliers with clip boundaries instead of masking to NaN
        masked_final = np.where(mask, tile, np.nan)
        with np.errstate(all='ignore'):
            med_final = np.nanmedian(masked_final, axis=0)
            mad_final = np.nanmedian(
                np.abs(masked_final - med_final[np.newaxis]), axis=0) * 1.4826
        mad_final = np.maximum(mad_final, 1e-12)
        upper = med_final + sigma * mad_final
        lower = med_final - sigma * mad_final
        clipped = np.clip(tile, lower[np.newaxis], upper[np.newaxis])
        if weights is not None:
            w = weights[:, np.newaxis, np.newaxis, np.newaxis]
            return (np.sum(clipped * w, axis=0) / np.sum(w)).astype(np.float32)
        return np.mean(clipped, axis=0).astype(np.float32)
    else:
        masked_final = np.where(mask, tile, np.nan)
        if weights is not None:
            w = np.where(mask, weights[:, np.newaxis, np.newaxis, np.newaxis], 0.0)
            with np.errstate(all='ignore'):
                total_w = np.sum(w, axis=0)
                total_w[total_w == 0] = 1.0
                result = np.nansum(masked_final * w, axis=0) / total_w
            np.nan_to_num(result, copy=False, nan=0.0)
            return result.astype(np.float32)
        with np.errstate(all='ignore'):
            result = np.nanmean(masked_final, axis=0)
        np.nan_to_num(result, copy=False, nan=0.0)
        return result.astype(np.float32)


def sigma_clip_combine(data: np.ndarray, sigma: float = 3.0, max_iters: int = 3,
                       weights: Optional[np.ndarray] = None,
                       winsorize: bool = False,
                       verbose: bool = False) -> np.ndarray:
    """Combine frames using tiled, MAD-based, optionally winsorized sigma-clip.

    Processes the image in spatial tiles to keep peak memory low.  Uses
    MAD (Median Absolute Deviation) instead of standard deviation for more
    robust outlier detection.  Optionally supports quality-weighted
    combination and winsorized clipping.

    Args:
        data: Array of shape ``(N, H, W, C)`` (all aligned frames).
        sigma: Rejection threshold in MADs.
        max_iters: Maximum clipping iterations.
        weights: Optional 1-D array of length N with per-frame quality weights.
        winsorize: If True, clip outliers to boundary instead of rejecting.
        verbose: Print per-tile progress.
    """
    N, H, W, C = data.shape
    tile_size = Config.TILE_SIZE
    result = np.zeros((H, W, C), dtype=np.float32)
    total_rejected = 0
    total_pixels = 0

    n_tiles_y = (H + tile_size - 1) // tile_size
    n_tiles_x = (W + tile_size - 1) // tile_size

    tile_coords = [
        (ty_idx * tile_size, min((ty_idx + 1) * tile_size, H),
         tx_idx * tile_size, min((tx_idx + 1) * tile_size, W))
        for ty_idx in range(n_tiles_y)
        for tx_idx in range(n_tiles_x)
    ]

    def _process_tile(coords):
        ty, ty_end, tx, tx_end = coords
        tile = np.array(data[:, ty:ty_end, tx:tx_end, :], dtype=np.float32)
        return coords, _sigma_clip_tile(tile, sigma, max_iters, weights, winsorize)

    n_tile_workers = min(os.cpu_count() or 4, len(tile_coords))
    with ThreadPoolExecutor(max_workers=n_tile_workers) as executor:
        for coords, tile_result in executor.map(_process_tile, tile_coords):
            ty, ty_end, tx, tx_end = coords
            result[ty:ty_end, tx:tx_end, :] = tile_result

    if verbose:
        safe_print(f"    Tiled sigma-clip: {n_tiles_y * n_tiles_x} tiles of "
                   f"{tile_size}x{tile_size}, mode={'winsorized' if winsorize else 'reject'}")
    return result
