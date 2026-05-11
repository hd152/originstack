"""Stacking algorithms: Lanczos resampling, drizzle combine, sigma-clip combine."""
from __future__ import annotations

import argparse
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from src.gpu_context import get_gpu
from src.models import Config, FrameInfo, ProcessingStats
from src.utils import safe_print, get_logger

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

_log = get_logger()


def lacosmic_reject(img: np.ndarray, sigclip: float = 4.5, objlim: float = 5.0,
                    gain: float = 1.0, readnoise: float = 6.5) -> np.ndarray:
    """L.A.Cosmic-style cosmic ray rejection for a single RGB frame.

    Implements a simplified version of the Laplacian edge detection algorithm
    (van Dokkum 2001) adapted for debayered RGB data.  Works per-channel so
    single-channel cosmic ray hits (the common case for OSC sensors) are
    detected and cleaned independently.

    Algorithm:
      1. Convolve channel with a 3x3 Laplacian kernel — cosmic rays produce
         a sharp positive spike in the fine-structure image.
      2. Build a Poisson + read-noise model: noise = sqrt(signal/gain + (RN/gain)²).
      3. Normalise the Laplacian by the noise model to get a detection statistic S.
      4. Reject pixels where S > ``sigclip`` **and** S / median(S, 3x3) > ``objlim``.
         The second condition prevents real compact objects (star cores) from
         being flagged — their Laplacian spike is surrounded by a similarly
         elevated neighbourhood, so the ratio stays below ``objlim``.
      5. Replace flagged pixels with the 5x5 local median.

    Args:
        img: Calibrated, debayered float32 RGB frame (H, W, 3).
        sigclip: Detection threshold in units of the per-pixel noise model
                 (default 4.5 — conservative to avoid touching star cores).
        objlim: Minimum ratio of S to local-median(S) required for CR flagging
                (default 5.0).  Increase to flag only sharper spikes.
        gain: Effective detector gain in e⁻/ADU (default 1.0).  Used in the
              Poisson noise model; inaccuracy has little effect on detection.
        readnoise: Read noise in ADU (default 6.5).  Added in quadrature to
                   the Poisson term.

    Returns:
        Cleaned float32 RGB image with cosmic rays replaced by local median.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        return img

    result = img.astype(np.float32, copy=True)
    lap_kernel = np.array([[0.0, -1.0, 0.0],
                           [-1.0,  4.0, -1.0],
                           [0.0, -1.0, 0.0]], dtype=np.float64)
    rn_term = (readnoise / gain) ** 2
    total_fixed = 0

    for c in range(3):
        ch = img[:, :, c].astype(np.float64)

        # Fine-structure image via Laplacian — keep only positive spikes
        fine = ndimage.convolve(ch, lap_kernel, mode='reflect')
        fine = np.clip(fine, 0.0, None)

        # Local median for signal estimate (5x5) and noise model
        med5 = ndimage.median_filter(ch, size=5)
        noise = np.sqrt(np.maximum(med5, 0.0) / gain + rn_term)
        noise = np.maximum(noise, 1e-6)

        # Normalised detection statistic
        S = fine / (2.0 * noise)  # factor of 2: Laplacian amplifies noise by ~2

        # Object rejection: local median of S in 3x3 neighbourhood
        S_med = ndimage.median_filter(S, size=3)
        ratio = S / np.maximum(S_med, 1e-6)

        mask = (S > sigclip) & (ratio > objlim)
        n_cr = int(np.sum(mask))
        if n_cr > 0:
            result[:, :, c][mask] = med5[mask].astype(np.float32)
            total_fixed += n_cr
            _log.debug("lacosmic ch%d: %d cosmic ray pixels replaced", c, n_cr)

    if total_fixed:
        _log.info("lacosmic_reject: replaced %d cosmic ray pixels total", total_fixed)
    return result


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
        img64 = img.astype(np.float64)
        result = np.empty((out_h, out_w, C), dtype=np.float64)
        for c in range(C):
            result[:, :, c] = ndimage.map_coordinates(
                img64[:, :, c],
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
    if not aligned_list:
        raise ValueError("drizzle_combine: aligned_list is empty")

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

    is_3d = aligned_list[0].ndim == 3
    acc = np.zeros((out_h, out_w, C) if is_3d else (out_h, out_w), dtype=np.float64)
    weight_map = np.zeros_like(acc, dtype=np.float64)

    # Output coordinate arrays (reused for each frame's coverage check)
    _oy = np.arange(out_h, dtype=np.float64)
    _ox = np.arange(out_w, dtype=np.float64)

    for i, (im, sh) in enumerate(zip(aligned_list, shifts)):
        w = float(weights[i]) if weights is not None else 1.0
        resampled = _lanczos_resample_frame(im, sh, scale, out_h, out_w)

        # Compute coverage mask analytically: output pixel (oy, ox) maps to
        # input (iy, ix) = (oy/scale - sh[0], ox/scale - sh[1]).  A pixel is
        # valid when both coordinates fall inside the input frame boundaries.
        # This replaces the previous approach of resampling a ones-image, which
        # ran a full Lanczos pass per frame for an invariant result.
        iy = _oy / scale - sh[0]
        ix = _ox / scale - sh[1]
        valid = (
            (iy[:, np.newaxis] >= 0) & (iy[:, np.newaxis] < H) &
            (ix[np.newaxis, :] >= 0) & (ix[np.newaxis, :] < W)
        )

        if is_3d:
            valid3 = valid[:, :, np.newaxis]
            acc += np.where(valid3, resampled * w, 0.0)
            weight_map += np.where(valid3, w, 0.0)
        else:
            acc += np.where(valid, resampled * w, 0.0)
            weight_map += np.where(valid, w, 0.0)

    weight_map[weight_map == 0] = 1.0
    return (acc / weight_map).astype(np.float32)


def _sigma_clip_tile(tile: np.ndarray, sigma: float, max_iters: int,
                     weights: Optional[np.ndarray], winsorize: bool,
                     use_mad: bool = True) -> np.ndarray:
    """Process a single spatial tile for sigma-clip combine.

    Uses MAD (Median Absolute Deviation) by default for robust spread
    estimation.  When ``use_mad=False``, uses standard deviation instead
    (equivalent to PixInsight's "Linear Clipping").
    """
    N = tile.shape[0]
    mask = np.ones(tile.shape, dtype=bool)
    # Pre-allocate once; avoids allocating a new float64 array on every iteration.
    # Using float32 throughout also prevents the silent float64 promotion that
    # np.where(mask, tile, np.nan) causes (np.nan is float64).
    masked = tile.astype(np.float32, copy=True)

    for iteration in range(max_iters):
        with np.errstate(all='ignore'):
            if use_mad:
                median = np.nanmedian(masked, axis=0)
                # MAD * 1.4826 is a consistent estimator of std for normal data
                spread = np.nanmedian(np.abs(masked - median[np.newaxis]), axis=0) * 1.4826
                center = median
            else:
                center = np.nanmean(masked, axis=0)
                spread = np.nanstd(masked, axis=0)

        # Fallback to std/mad where spread is zero (constant regions)
        zero_spread = spread < 1e-12
        if np.any(zero_spread):
            with np.errstate(all='ignore'):
                fallback = np.nanstd(masked, axis=0) if use_mad else \
                           np.nanmedian(np.abs(masked - np.nanmedian(masked, axis=0)[np.newaxis]), axis=0) * 1.4826
            spread[zero_spread] = fallback[zero_spread]

        deviation = np.abs(masked - center[np.newaxis])
        new_mask = mask & (deviation <= sigma * spread[np.newaxis])

        # Ensure at least 1 frame survives at every pixel
        surviving = new_mask.sum(axis=0)
        all_rejected = surviving == 0
        if np.any(all_rejected):
            for frame_idx in range(N):
                new_mask[frame_idx][all_rejected] = mask[frame_idx][all_rejected]

        newly_rejected = mask & ~new_mask
        rejected = int(mask.sum() - new_mask.sum())
        total_valid = int(mask.sum())
        mask = new_mask
        masked[newly_rejected] = np.nan
        if rejected == 0:
            break
        # Early stopping: <0.1% change means convergence
        if total_valid > 0 and rejected / total_valid < 0.001:
            break

    if winsorize:
        # Replace outliers with clip boundaries instead of masking to NaN
        # masked already equals np.where(mask, tile, np.nan) at this point
        with np.errstate(all='ignore'):
            if use_mad:
                med_final = np.nanmedian(masked, axis=0)
                spread_final = np.nanmedian(
                    np.abs(masked - med_final[np.newaxis]), axis=0) * 1.4826
                center_final = med_final
            else:
                center_final = np.nanmean(masked, axis=0)
                spread_final = np.nanstd(masked, axis=0)
        spread_final = np.maximum(spread_final, 1e-12)
        upper = center_final + sigma * spread_final
        lower = center_final - sigma * spread_final
        clipped = np.clip(tile, lower[np.newaxis], upper[np.newaxis])
        if weights is not None:
            w = weights[:, np.newaxis, np.newaxis, np.newaxis]
            return (np.sum(clipped * w, axis=0) / np.sum(w)).astype(np.float32)
        return np.mean(clipped, axis=0).astype(np.float32)
    else:
        if weights is not None:
            w = np.where(mask, weights[:, np.newaxis, np.newaxis, np.newaxis], 0.0)
            with np.errstate(all='ignore'):
                total_w = np.sum(w, axis=0)
                total_w[total_w == 0] = 1.0
                result = np.nansum(masked * w, axis=0) / total_w
            np.nan_to_num(result, copy=False, nan=0.0)
            return result.astype(np.float32)
        with np.errstate(all='ignore'):
            result = np.nanmean(masked, axis=0)
        np.nan_to_num(result, copy=False, nan=0.0)
        return result.astype(np.float32)


def sigma_clip_combine(data: np.ndarray, sigma: float = 3.0, max_iters: int = 3,
                       weights: Optional[np.ndarray] = None,
                       winsorize: bool = False,
                       use_mad: bool = True,
                       verbose: bool = False) -> np.ndarray:
    """Combine frames using tiled, optionally winsorized sigma-clip.

    Processes the image in spatial tiles to keep peak memory low.  Uses
    MAD (Median Absolute Deviation) by default; pass ``use_mad=False`` for
    standard-deviation-based clipping (PixInsight "Linear Clipping").

    Args:
        data: Array of shape ``(N, H, W, C)`` (all aligned frames).
        sigma: Rejection threshold in MADs (or std when use_mad=False).
        max_iters: Maximum clipping iterations.
        weights: Optional 1-D array of length N with per-frame quality weights.
        winsorize: If True, clip outliers to boundary instead of rejecting.
        use_mad: If True (default), use MAD for spread; False uses std.
        verbose: Print per-tile progress.
    """
    N, H, W, C = data.shape
    tile_size = Config.TILE_SIZE
    result = np.zeros((H, W, C), dtype=np.float32)

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
        return coords, _sigma_clip_tile(tile, sigma, max_iters, weights, winsorize, use_mad)

    n_tile_workers = min(os.cpu_count() or 4, len(tile_coords))
    with ThreadPoolExecutor(max_workers=n_tile_workers) as executor:
        for coords, tile_result in executor.map(_process_tile, tile_coords):
            ty, ty_end, tx, tx_end = coords
            result[ty:ty_end, tx:tx_end, :] = tile_result

    if verbose:
        estimator = 'MAD' if use_mad else 'std'
        mode = 'winsorized' if winsorize else 'reject'
        safe_print(f"    Tiled sigma-clip: {n_tiles_y * n_tiles_x} tiles of "
                   f"{tile_size}x{tile_size}, estimator={estimator}, mode={mode}")
    return result


def _percentile_clip_tile(tile: np.ndarray, low: float, high: float,
                          weights: Optional[np.ndarray]) -> np.ndarray:
    """Percentile-clip a single spatial tile."""
    N = tile.shape[0]
    lo = np.percentile(tile, low, axis=0)
    hi = np.percentile(tile, high, axis=0)
    mask = (tile >= lo[np.newaxis]) & (tile <= hi[np.newaxis])

    # Ensure at least 1 frame survives at every pixel
    surviving = mask.sum(axis=0)
    all_rejected = surviving == 0
    if np.any(all_rejected):
        for frame_idx in range(N):
            mask[frame_idx][all_rejected] = True

    masked = np.where(mask, tile, np.nan)
    if weights is not None:
        w = np.where(mask, weights[:, np.newaxis, np.newaxis, np.newaxis], 0.0)
        with np.errstate(all='ignore'):
            total_w = np.sum(w, axis=0)
            total_w[total_w == 0] = 1.0
            result = np.nansum(masked * w, axis=0) / total_w
    else:
        with np.errstate(all='ignore'):
            result = np.nanmean(masked, axis=0)
    np.nan_to_num(result, copy=False, nan=0.0)
    return result.astype(np.float32)


def percentile_clip_combine(data: np.ndarray, low: float = 20.0, high: float = 80.0,
                            weights: Optional[np.ndarray] = None,
                            verbose: bool = False) -> np.ndarray:
    """Combine frames using percentile clipping rejection.

    Rejects pixels outside the [low, high] percentile range across the
    frame stack at each pixel position, then averages survivors.  This is
    PixInsight's "Percentile Clipping" method.  Works well for small frame
    counts (< 8) where sigma-based estimators are unreliable.

    Args:
        data: Array of shape ``(N, H, W, C)``.
        low: Lower rejection percentile (default 20 — symmetric with high=80).
        high: Upper rejection percentile (default 80).
        weights: Optional 1-D per-frame quality weights.
        verbose: Print summary.
    """
    N, H, W, C = data.shape
    tile_size = Config.TILE_SIZE
    result = np.zeros((H, W, C), dtype=np.float32)

    n_tiles_y = (H + tile_size - 1) // tile_size
    n_tiles_x = (W + tile_size - 1) // tile_size
    tile_coords = [
        (ty * tile_size, min((ty + 1) * tile_size, H),
         tx * tile_size, min((tx + 1) * tile_size, W))
        for ty in range(n_tiles_y)
        for tx in range(n_tiles_x)
    ]

    def _process_tile(coords):
        ty, ty_end, tx, tx_end = coords
        tile = np.array(data[:, ty:ty_end, tx:tx_end, :], dtype=np.float32)
        return coords, _percentile_clip_tile(tile, low, high, weights)

    n_workers = min(os.cpu_count() or 4, len(tile_coords))
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for coords, tile_result in executor.map(_process_tile, tile_coords):
            ty, ty_end, tx, tx_end = coords
            result[ty:ty_end, tx:tx_end, :] = tile_result

    if verbose:
        safe_print(f"    Percentile clip: low={low}%, high={high}%, "
                   f"{n_tiles_y * n_tiles_x} tiles of {tile_size}x{tile_size}")
    return result


def _esd_clip_tile(tile: np.ndarray, max_outliers: int, significance: float,
                   weights: Optional[np.ndarray]) -> np.ndarray:
    """Generalized ESD (Extreme Studentized Deviate) rejection for a tile.

    Iteratively removes the most extreme outlier at each pixel position
    using the Grubbs test statistic, corrected for multiple comparisons via
    the t-distribution.  Better than sigma-clip for small N (< ~10 frames)
    because it accounts for the sample size in the rejection threshold.
    """
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        # Fall back to sigma-clip if scipy unavailable
        return _sigma_clip_tile(tile, 3.0, 3, weights, False, True)

    N = tile.shape[0]
    mask = np.ones(tile.shape, dtype=bool)

    # Precompute Grubbs critical values: lambda(n_eff, i) for each iteration i
    # lambda = (n-1)/sqrt(n) * t / sqrt(n-2+t^2)
    # where t = t_{alpha/(2*n), n-2} (two-tailed, Bonferroni-corrected)
    lambda_lut: dict = {}
    for n_eff in range(3, N + 1):
        for i in range(min(max_outliers, n_eff - 2)):
            n_cur = n_eff - i
            if n_cur <= 2:
                lambda_lut[(n_eff, i)] = np.inf
                continue
            p = significance / (2.0 * n_cur)
            p = min(max(p, 1e-10), 0.4999)
            df = max(n_cur - 2, 1)
            t_crit = scipy_stats.t.ppf(1.0 - p, df=df)
            denom = np.sqrt((n_cur - 2.0 + t_crit ** 2) * n_cur)
            lambda_lut[(n_eff, i)] = (n_cur - 1.0) * t_crit / denom if denom > 0 else np.inf

    for i in range(max_outliers):
        masked = np.where(mask, tile, np.nan)
        with np.errstate(all='ignore'):
            mean = np.nanmean(masked, axis=0)
            std = np.nanstd(masked, axis=0, ddof=1)
        std = np.maximum(std, 1e-12)

        deviation = np.where(mask, np.abs(tile - mean[np.newaxis]) / std[np.newaxis], -1.0)
        max_dev = deviation.max(axis=0)
        most_extreme_idx = deviation.argmax(axis=0)

        n_active = mask.sum(axis=0)  # (th, tw, C)

        # Build per-pixel critical value map from LUT using vectorized indexing
        lam_arr = np.full(N + 1, np.inf)
        for n_c in range(3, N + 1):
            lam_arr[n_c] = lambda_lut.get((n_c, i), np.inf)
        lambda_map = lam_arr[n_active]

        reject = max_dev > lambda_map
        if not np.any(reject):
            break

        # Remove the most extreme frame at each rejected pixel position
        for frame_idx in range(N):
            is_extreme_rejected = (most_extreme_idx == frame_idx) & reject
            mask[frame_idx][is_extreme_rejected] = False

        # Ensure at least 1 frame survives: restore only the least-bad frame
        surviving = mask.sum(axis=0)
        all_gone = surviving == 0
        if np.any(all_gone):
            raw_dev = np.abs(tile - mean[np.newaxis]) / std[np.newaxis]
            best_frame = raw_dev.argmin(axis=0)
            for frame_idx in range(N):
                is_best = (best_frame == frame_idx) & all_gone
                mask[frame_idx][is_best] = True

    masked_final = np.where(mask, tile, np.nan)
    if weights is not None:
        w = np.where(mask, weights[:, np.newaxis, np.newaxis, np.newaxis], 0.0)
        with np.errstate(all='ignore'):
            total_w = np.sum(w, axis=0)
            total_w[total_w == 0] = 1.0
            result = np.nansum(masked_final * w, axis=0) / total_w
    else:
        with np.errstate(all='ignore'):
            result = np.nanmean(masked_final, axis=0)
    np.nan_to_num(result, copy=False, nan=0.0)
    return result.astype(np.float32)


def esd_combine(data: np.ndarray, max_outliers: int = 0, significance: float = 0.05,
                weights: Optional[np.ndarray] = None,
                verbose: bool = False) -> np.ndarray:
    """Combine frames using Generalized ESD (Extreme Studentized Deviate) rejection.

    Recommended for small frame counts (< ~15) where sigma-clip's MAD
    estimator is unreliable.  Uses per-pixel Grubbs test with t-distribution
    critical values corrected for multiple comparisons.

    Args:
        data: Array of shape ``(N, H, W, C)``.
        max_outliers: Maximum outliers to remove per pixel (default: N//4).
        significance: Type-I error rate for the ESD test (default: 0.05).
        weights: Optional 1-D per-frame quality weights.
        verbose: Print summary.
    """
    N, H, W, C = data.shape
    if max_outliers <= 0:
        max_outliers = max(1, N // 4)

    tile_size = Config.TILE_SIZE
    result = np.zeros((H, W, C), dtype=np.float32)

    n_tiles_y = (H + tile_size - 1) // tile_size
    n_tiles_x = (W + tile_size - 1) // tile_size
    tile_coords = [
        (ty * tile_size, min((ty + 1) * tile_size, H),
         tx * tile_size, min((tx + 1) * tile_size, W))
        for ty in range(n_tiles_y)
        for tx in range(n_tiles_x)
    ]

    def _process_tile(coords):
        ty, ty_end, tx, tx_end = coords
        tile = np.array(data[:, ty:ty_end, tx:tx_end, :], dtype=np.float32)
        return coords, _esd_clip_tile(tile, max_outliers, significance, weights)

    n_workers = min(os.cpu_count() or 4, len(tile_coords))
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for coords, tile_result in executor.map(_process_tile, tile_coords):
            ty, ty_end, tx, tx_end = coords
            result[ty:ty_end, tx:tx_end, :] = tile_result

    if verbose:
        safe_print(f"    ESD: max_outliers={max_outliers}, significance={significance}, "
                   f"{n_tiles_y * n_tiles_x} tiles of {tile_size}x{tile_size}")
    return result


def run_stacking_phase(
    final: List[FrameInfo],
    final_indices: List[int],
    mem_rgb: np.ndarray,
    shifts: List[Tuple[float, float]],
    transforms: List[Optional[Any]],
    H: int,
    W: int,
    C: int,
    args: argparse.Namespace,
    stats: ProcessingStats,
) -> Tuple[np.ndarray, np.ndarray, int, int, int, int]:
    """Align and combine frames into a stacked image.

    Returns (stacked, fits_stacked, top, bottom, left, right).
    fits_stacked is the pre-post-processing copy saved in the FITS file.
    """
    from src.registration import apply_transform, calc_common_crop

    n_final = len(final)
    print(f"  Method: {args.stack_method}")
    print(f"  Combining {n_final} frames...")

    # Quality weights
    w_snr   = getattr(args, 'weight_snr',   1.0)
    w_fwhm  = getattr(args, 'weight_fwhm',  1.0)
    w_stars = getattr(args, 'weight_stars', 1.0)
    use_noise_weight = getattr(args, 'weight_noise', False)

    if w_snr != 1.0 or w_fwhm != 1.0 or w_stars != 1.0 or use_noise_weight:
        snr_vals  = np.array([f.metrics.get('snr', 1.0)       for f in final], dtype=np.float64)
        fwhm_vals = np.array([f.metrics.get('fwhm', 0.0)      for f in final], dtype=np.float64)
        star_vals = np.array([f.metrics.get('star_count', 1)  for f in final], dtype=np.float64)

        snr_factor = np.clip(snr_vals / max(snr_vals.max(), 1e-9), 0.01, 1.0)

        fwhm_pos = fwhm_vals[fwhm_vals > 0]
        if len(fwhm_pos):
            fwhm_inv = np.where(fwhm_vals > 0, 1.0 / np.maximum(fwhm_vals, 0.5), 1.0)
            fwhm_factor = np.clip(fwhm_inv / fwhm_inv.max(), 0.01, 1.0)
        else:
            fwhm_factor = np.ones(n_final)

        star_factor = np.clip(star_vals / max(star_vals.max(), 1e-9), 0.01, 1.0)
        weights = (snr_factor ** w_snr) * (fwhm_factor ** w_fwhm) * (star_factor ** w_stars)

        if use_noise_weight:
            brightness = np.array([f.metrics.get('brightness', 1.0) for f in final],
                                   dtype=np.float64)
            noise_est = brightness / np.maximum(snr_vals, 0.001)
            noise_factor = np.clip(noise_est.max() / np.maximum(noise_est, 1e-6), 0.01, 1.0)
            weights *= noise_factor

        weights = np.sqrt(weights / max(weights.max(), 1e-9))
        print(f"  Quality weights (per-component SNR^{w_snr} FWHM^{w_fwhm} stars^{w_stars}"
              f"{' noise' if use_noise_weight else ''}): "
              f"min={weights.min():.3f}, max={weights.max():.3f}, mean={weights.mean():.3f}")
    else:
        scores = np.array([f.metrics.get('score', 1.0) for f in final])
        weights = np.sqrt(scores / max(scores.max(), 1e-9))
        print(f"  Quality weights: min={weights.min():.3f}, max={weights.max():.3f}, "
              f"mean={weights.mean():.3f} (sqrt-compressed)")

    top, bottom, left, right = calc_common_crop(shifts, (H, W), transforms=transforms)
    stats.output_shape = (bottom - top, right - left)
    stats.cropped_pixels = (H - (bottom - top), W - (right - left))

    drizzle_scale = getattr(args, 'drizzle_scale', 1.0)
    use_aligned_memmap = (drizzle_scale <= 1.0 and
                          args.stack_method in ('median', 'sigma_clip', 'winsorized',
                                                'percentile', 'esd'))
    if use_aligned_memmap:
        mm_aligned_path = os.path.join(tempfile.gettempdir(), f'stack_aligned_{os.getpid()}.dat')
        crop_h, crop_w = bottom - top, right - left
        mem_aligned = np.memmap(mm_aligned_path, dtype='float32', mode='w+',
                                shape=(n_final, crop_h, crop_w, C))

        gpu = get_gpu()

        def _align_one(j):
            with gpu.stream_context():
                rgb = np.array(mem_rgb[final_indices[j]])
                aligned = apply_transform(rgb, shift=shifts[j], transform=transforms[j])
                mem_aligned[j] = aligned[top:bottom, left:right, :]

        n_align = (min(gpu.max_gpu_workers(Config.GPU_ALIGN_WORKER_MB,
                                           Config.GPU_VRAM_RESERVE_MB), n_final)
                   if gpu.active else min(os.cpu_count() or 4, n_final))
        with ThreadPoolExecutor(max_workers=n_align) as executor:
            futures = {executor.submit(_align_one, j): j for j in range(n_final)}
            for future in tqdm(as_completed(futures), total=n_final,
                               desc="  Aligning", unit="frame", disable=not args.verbose):
                future.result()
        mem_aligned.flush()

        if args.stack_method in ('sigma_clip', 'winsorized'):
            use_winsorize = (args.stack_method == 'winsorized')
            use_mad = (getattr(args, 'rejection_estimator', 'mad') == 'mad')
            print(f"  Sigma-clip: sigma={args.rejection_sigma}, iters={args.rejection_iters}, "
                  f"estimator={'MAD' if use_mad else 'std'}, "
                  f"mode={'winsorized' if use_winsorize else 'reject'}")
            stacked = sigma_clip_combine(mem_aligned, sigma=args.rejection_sigma,
                                         max_iters=args.rejection_iters, weights=weights,
                                         winsorize=use_winsorize, use_mad=use_mad,
                                         verbose=args.verbose)
        elif args.stack_method == 'percentile':
            low  = getattr(args, 'percentile_low',  20.0)
            high = getattr(args, 'percentile_high', 80.0)
            print(f"  Percentile clip: low={low}%, high={high}%")
            stacked = percentile_clip_combine(mem_aligned, low=low, high=high,
                                              weights=weights, verbose=args.verbose)
        elif args.stack_method == 'esd':
            max_out = getattr(args, 'esd_max_outliers', 0)
            sig     = getattr(args, 'esd_significance', 0.05)
            print(f"  ESD: max_outliers={'N//4' if max_out == 0 else max_out}, significance={sig}")
            stacked = esd_combine(mem_aligned, max_outliers=max_out, significance=sig,
                                  weights=weights, verbose=args.verbose)
        else:
            stacked = np.median(mem_aligned, axis=0).astype(np.float32)

        del mem_aligned
        try:
            os.remove(mm_aligned_path)
        except Exception:
            pass

    else:
        if drizzle_scale > 1.0:
            crop_h, crop_w = bottom - top, right - left
            out_h = int(round(crop_h * drizzle_scale))
            out_w = int(round(crop_w * drizzle_scale))
            print(f"  Drizzle: {drizzle_scale:.1f}x ({crop_h}x{crop_w} -> {out_h}x{out_w})")
            acc = np.zeros((out_h, out_w, C), dtype=np.float64)
            gpu = get_gpu()
            spline_order = 5
            inv_scale = 1.0 / drizzle_scale

            # Compose alignment + crop + upscale into a single affine_transform
            # per frame, avoiding large per-worker coordinate arrays.
            #
            # apply_transform uses ndimage.affine_transform(raw, R, offset) or
            # ndimage.shift(raw, shift), both computing:
            #   aligned[o] = raw[R @ o + offset]   (affine)
            #   aligned[o] = raw[o - shift]         (shift)
            #
            # We want: for drizzle output pixel (oy, ox), read from raw at:
            #   raw_coord = R @ (oy/scale + top, ox/scale + left) + alignment_offset
            # Rearranging as affine_transform form (raw[M @ out + off]):
            #   M = R / scale,  off = R @ [top, left] + alignment_offset
            #
            # calc_common_crop guarantees the crop is valid for all frames,
            # so all output pixels map to valid raw pixels — no per-frame
            # validity mask needed.

            acc_lock = threading.Lock()

            def _drizzle_one(j):
                """Single-pass drizzle: raw frame → upscaled output via one affine."""
                rgb = np.array(mem_rgb[final_indices[j]])
                w = float(weights[j])
                shift_j = shifts[j]
                transform_j = transforms[j]

                crop_offset = np.array([float(top), float(left)])

                if transform_j is not None:
                    R = transform_j.params[:2, :2]
                    t_xy = transform_j.params[:2, 2]
                    t_rowcol = np.array([t_xy[1], t_xy[0]])
                    align_offset = -R @ t_rowcol
                    # Compose: M = R * inv_scale, off = R @ crop_offset + align_offset
                    M = R * inv_scale
                    off = R @ crop_offset + align_offset
                else:
                    sy, sx = shift_j if shift_j is not None else (0.0, 0.0)
                    # shift case: raw = aligned - shift
                    # M = I * inv_scale, off = crop_offset - shift
                    M = np.array([[inv_scale, 0.0], [0.0, inv_scale]])
                    off = crop_offset - np.array([sy, sx])

                resampled = np.empty((out_h, out_w, C), dtype=np.float32)
                for c in range(C):
                    resampled[:, :, c] = ndimage.affine_transform(
                        rgb[:, :, c], M, offset=off,
                        output_shape=(out_h, out_w),
                        order=spline_order, mode='constant', cval=0.0)

                resampled *= w  # in-place float32, avoids float64 temporary
                with acc_lock:
                    np.add(acc, resampled, out=acc)  # float64 += float32 (promoted)
                    total_weight_ref[0] += w

            total_weight_ref = [0.0]

            # Cap workers based on available RAM.
            # Per worker: raw frame (H*W*C*4) + resampled output (out_h*out_w*C*4)
            # + affine_transform internal buffer (~out_h*out_w*8 per channel call).
            n_drizzle = min(os.cpu_count() or 4, n_final)
            try:
                import psutil
                avail_mb = psutil.virtual_memory().available / 1e6
                raw_mb = H * W * C * 4 / 1e6
                out_mb = out_h * out_w * C * 4 / 1e6
                affine_buf_mb = out_h * out_w * 8 / 1e6  # scipy internal float64
                per_worker_mb = raw_mb + out_mb + affine_buf_mb + 100
                safe_workers = max(1, int(avail_mb / per_worker_mb))
                if safe_workers < n_drizzle:
                    print(f"  NOTE: limiting drizzle threads {n_drizzle}\u2192{safe_workers} "
                          f"(avail RAM {avail_mb:.0f} MB, ~{per_worker_mb:.0f} MB/worker)")
                    n_drizzle = safe_workers
            except Exception:
                pass

            with ThreadPoolExecutor(max_workers=n_drizzle) as executor:
                futures = {executor.submit(_drizzle_one, j): j for j in range(n_final)}
                for future in tqdm(as_completed(futures), total=n_final,
                                   desc="  Drizzling", unit="frame",
                                   disable=not args.verbose):
                    future.result()

            stacked = (acc / max(total_weight_ref[0], 1e-12)).astype(np.float32)

        else:
            acc = np.zeros((bottom - top, right - left, C), dtype=np.float64)
            total_weight = 0.0
            gpu = get_gpu()

            def _align_crop(j):
                with gpu.stream_context():
                    rgb = np.array(mem_rgb[final_indices[j]])
                    return j, apply_transform(rgb, shift=shifts[j],
                                             transform=transforms[j])[top:bottom, left:right, :]

            n_workers = (min(gpu.max_gpu_workers(Config.GPU_ALIGN_WORKER_MB,
                                                  Config.GPU_VRAM_RESERVE_MB), n_final)
                         if gpu.active else min(os.cpu_count() or 4, n_final))
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_align_crop, j): j for j in range(n_final)}
                for future in tqdm(as_completed(futures), total=n_final,
                                   desc="  Stacking", unit="frame", disable=not args.verbose):
                    j, cropped = future.result()
                    w = float(weights[j])
                    acc += cropped.astype(np.float64) * w
                    total_weight += w
            stacked = (acc / max(total_weight, 1e-12)).astype(np.float32)

    # Save a pre-post-processing copy for FITS output (preserves high sky SNR)
    fits_stacked = stacked.copy()
    return stacked, fits_stacked, top, bottom, left, right


# ---------------------------------------------------------------------------
# HDR stack blending
# ---------------------------------------------------------------------------

def hdr_blend_stacks(short_stack: np.ndarray, long_stack: np.ndarray,
                     short_exptime: float = 1.0, long_exptime: float = 1.0,
                     transition_width: float = None) -> np.ndarray:
    """SNR-weighted HDR blend of two stacks with different exposure times.

    Merges a short-exposure stack (unsaturated bright cores) with a
    long-exposure stack (high-SNR faint regions) using a sigmoid transition
    centred at the saturation knee of the long-exposure stack.

    The blend weight for the long stack is:

        w_long(x) = sigmoid( (x_norm - knee) / transition_width )

    where ``x_norm`` is the per-pixel luminance normalised to [0, 1] and
    ``knee`` is estimated as the 98th percentile of the long-stack luminance
    (the level above which the long exposure is likely saturated or nonlinear).

    The short stack is rescaled to match the long stack's flux calibration
    via the ``short_exptime / long_exptime`` ratio before blending.

    Args:
        short_stack:      Float32 (H, W, 3) shorter-exposure stack.
        long_stack:       Float32 (H, W, 3) longer-exposure stack.
        short_exptime:    Short-stack total effective exposure (s or arbitrary).
        long_exptime:     Long-stack total effective exposure.
        transition_width: Fractional luminance range for the sigmoid
                          (default Config.HDR_TRANSITION_WIDTH = 0.1).

    Returns:
        Blended float32 HDR image (H, W, 3).
    """
    if transition_width is None:
        transition_width = Config.HDR_TRANSITION_WIDTH

    if short_stack.shape != long_stack.shape:
        raise ValueError(f"HDR blend: shape mismatch "
                         f"({short_stack.shape} vs {long_stack.shape})")

    # Scale short stack to long stack's calibration
    scale = float(long_exptime) / max(float(short_exptime), 1e-9)
    short_scaled = short_stack.astype(np.float64) * scale
    long_d = long_stack.astype(np.float64)

    lum_long = (0.299 * long_d[:, :, 0] + 0.587 * long_d[:, :, 1]
                + 0.114 * long_d[:, :, 2])
    white = float(np.percentile(lum_long, 98))
    if white < 1e-9:
        return long_stack.copy()

    lum_norm = np.clip(lum_long / white, 0.0, 1.0)
    knee = float(np.percentile(lum_norm, 95))

    # Sigmoid: 0 (use short) → 1 (use long) as luminance increases through knee
    sig_arg = np.clip((lum_norm - knee) / max(transition_width, 1e-4), -10.0, 10.0)
    w_long = 1.0 / (1.0 + np.exp(-sig_arg))   # high lum → short stack wins
    # Invert: for saturated bright areas we want the short (unsaturated) stack
    w_short = w_long          # bright pixels → more short weight
    w_long_use = 1.0 - w_short

    w_long_3 = w_long_use[:, :, np.newaxis]
    w_short_3 = w_short[:, :, np.newaxis]

    blended = w_long_3 * long_d + w_short_3 * short_scaled
    return np.clip(blended, 0.0, None).astype(np.float32)


# ---------------------------------------------------------------------------
# Comet dual-track stacking
# ---------------------------------------------------------------------------

def blend_comet_star_stacks(star_stack: np.ndarray,
                             comet_stack: np.ndarray,
                             comet_lum: np.ndarray,
                             blend_sigma: float = 30.0) -> np.ndarray:
    """Blend a star-aligned and comet-aligned stack for dual-track imaging.

    In standard comet stacking the astronomer must choose between sharpening
    the comet nucleus (comet alignment) and sharpening the star field (star
    alignment).  This function blends both stacks spatially:

    • Near the comet nucleus: ``comet_stack`` dominates (sharp nucleus,
      smeared stars).
    • Away from the nucleus: ``star_stack`` dominates (sharp stars,
      smeared nucleus).

    The blend mask is derived from the comet luminance: a Gaussian envelope
    centred on the brightest region (the nucleus) whose width is controlled
    by ``blend_sigma`` pixels.

    Args:
        star_stack:   Float32 (H, W, 3) star-aligned stack.
        comet_stack:  Float32 (H, W, 3) comet-aligned stack.
        comet_lum:    Float32 (H, W) luminance map for locating the nucleus
                      (typically the comet-aligned stack luminance).
        blend_sigma:  Gaussian half-width (px) of the comet blend zone.

    Returns:
        Blended float32 image (H, W, 3).
    """
    from scipy.ndimage import gaussian_filter as _gf

    H, W = star_stack.shape[:2]

    # Locate comet nucleus (brightest smooth blob)
    smoothed = _gf(comet_lum.astype(np.float64), sigma=5.0)
    peak_flat = int(np.argmax(smoothed))
    py, px = peak_flat // W, peak_flat % W

    # Build distance-based Gaussian mask centred on nucleus
    yy, xx = np.mgrid[:H, :W]
    dist2 = (yy - py) ** 2.0 + (xx - px) ** 2.0
    mask = np.exp(-dist2 / (2.0 * blend_sigma ** 2))   # 1 at nucleus, 0 far away

    mask3 = mask[:, :, np.newaxis]
    blended = (mask3 * comet_stack.astype(np.float64)
               + (1.0 - mask3) * star_stack.astype(np.float64))
    return np.clip(blended, 0.0, None).astype(np.float32)
