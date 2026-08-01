"""Stacking algorithms: Lanczos resampling, drizzle combine, sigma-clip combine."""
from __future__ import annotations

import argparse
import os
import tempfile
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from src.gpu_context import get_gpu
from src.models import Config, FrameInfo, ProcessingStats
from src.utils import safe_print, get_logger, format_time
from src.cleanup import register as _cleanup_register, deregister as _cleanup_deregister

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

# Optional native (Rust) kernels — graceful degradation to numpy if absent.
try:
    import astro_native as _native
    HAS_NATIVE = True
except Exception:
    _native = None
    HAS_NATIVE = False

_log = get_logger()


def _native_usable(data: np.ndarray) -> bool:
    """The native combine kernels require a C-contiguous float32 (N,H,W,C) view.
    A non-matching array (wrong dtype / non-contiguous) falls back to numpy."""
    return (HAS_NATIVE and data.ndim == 4 and data.dtype == np.float32
            and data.flags['C_CONTIGUOUS'])


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

    if HAS_NATIVE and img.dtype == np.float32 and img.flags['C_CONTIGUOUS']:
        try:
            return _native.lacosmic_reject_native(
                img, float(sigclip), float(objlim), float(gain), float(readnoise))
        except Exception as exc:
            _log.debug("native lacosmic_reject failed (%s); using numpy", exc)

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


# ---------------------------------------------------------------------------
# Memory-bounded helpers for large stacks
# ---------------------------------------------------------------------------

_REJ_MASK_THRESHOLD = 200 * 1024 * 1024  # 200 MB — above this use a disk memmap


def _make_rej_mask(N: int, H: int, W: int, C: int) -> np.ndarray:
    """Allocate a rejection mask, falling back to a disk-backed memmap for large N."""
    total = N * H * W * C  # bool: 1 byte per element
    if total > _REJ_MASK_THRESHOLD:
        path = os.path.join(tempfile.gettempdir(),
                            f'stack_rejmap_{os.getpid()}_{N}x{H}x{W}.dat')
        arr = np.memmap(path, dtype=bool, mode='w+', shape=(N, H, W, C))
        arr._cleanup_path = path  # type: ignore[attr-defined]
        _cleanup_register(path)
        return arr
    return np.zeros((N, H, W, C), dtype=bool)


def _cap_tile_workers(n_workers: int, N: int, tile_size: int, C: int) -> int:
    """Limit tile workers so concurrent tile buffers fit within available RAM.

    Each tile worker holds two (N, tile, tile, C) float32 buffers — the
    input tile and the masked copy inside _sigma_clip_tile.
    """
    per_worker_bytes = N * tile_size * tile_size * C * 4 * 2  # float32, 2 copies
    try:
        import psutil
        avail = psutil.virtual_memory().available
        safe = max(1, int(avail * 0.5 / max(per_worker_bytes, 1)))
    except ImportError:
        # Without psutil: conservatively allow 1 GB budget for tile workers
        safe = max(1, (1024 * 1024 * 1024) // max(per_worker_bytes, 1))
    return min(n_workers, safe)


def _free_rej_mask(mask: np.ndarray) -> None:
    """Delete a rejection mask and clean up its backing file if disk-based."""
    path = getattr(mask, '_cleanup_path', None)
    if path is not None:
        try:
            if hasattr(mask, '_mmap') and mask._mmap is not None:
                mask._mmap.close()
        except Exception:
            pass
    del mask
    if path is not None:
        try:
            os.remove(path)
        except Exception:
            pass
        _cleanup_deregister(path)


def _adaptive_tile_size(N: int, C: int) -> int:
    """Return the largest tile size that keeps one tile's working set in available RAM.

    For large N (e.g. 972 frames) the default 256-pixel tile requires
    N×T²×C×4×2 bytes per worker (float32 tile + masked/sorted copy).
    When that exceeds available RAM every tile triggers OS paging and
    throughput collapses by 10–100×.  This function scales T down until
    one tile fits comfortably, without dropping below 16 px.
    """
    default = Config.TILE_SIZE
    try:
        import psutil
        avail = psutil.virtual_memory().available
        # Budget: 40 % of available RAM for one tile worker's working set
        budget = avail * 0.4
        max_t_sq = budget / max(N * C * 4 * 2, 1)
        max_t = int(max_t_sq ** 0.5)
        # Round down to nearest multiple of 16, min 16, cap at default
        return max(16, (min(default, max_t) // 16) * 16)
    except ImportError:
        return default


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

    # Native path: the mapping is the diagonal affine
    #   input = diag(1/scale) @ output - shift
    # which the Rust Lanczos-3 warp handles on its separable fast path, all
    # channels in one pass. True Lanczos-3 rather than the quintic-spline
    # approximation below.
    if HAS_NATIVE and lanczos_a == 3 and img.ndim == 3:
        try:
            inv = 1.0 / scale
            res = _native.warp_affine_lanczos3(
                np.ascontiguousarray(img, dtype=np.float32),
                [inv, 0.0, 0.0, inv],
                [-float(shift[0]), -float(shift[1])],
                int(out_h), int(out_w), 0.0)
            return res.astype(np.float64)
        except Exception:
            pass

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
                    drop_size: float = 0.7, pixfrac: float = 1.0) -> np.ndarray:
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
        iy = _oy / scale - sh[0]
        ix = _ox / scale - sh[1]
        valid = (
            (iy[:, np.newaxis] >= 0) & (iy[:, np.newaxis] < H) &
            (ix[np.newaxis, :] >= 0) & (ix[np.newaxis, :] < W)
        )

        if pixfrac < 1.0:
            # Tent-kernel pixfrac weight: each output pixel's contribution falls
            # off toward the edge of the input pixel's footprint.
            half_drop = pixfrac * scale / 2.0
            iy_frac = np.abs((iy - np.round(iy)) * scale)  # (out_h,)
            ix_frac = np.abs((ix - np.round(ix)) * scale)  # (out_w,)
            w_y = np.maximum(0.0, 1.0 - iy_frac / max(half_drop, 1e-12))
            w_x = np.maximum(0.0, 1.0 - ix_frac / max(half_drop, 1e-12))
            pixfrac_weight = w_y[:, np.newaxis] * w_x[np.newaxis, :]  # (out_h, out_w)
            if is_3d:
                valid3 = valid[:, :, np.newaxis]
                pfw3 = pixfrac_weight[:, :, np.newaxis]
                acc += np.where(valid3, resampled * w * pfw3, 0.0)
                weight_map += np.where(valid3, w * pfw3, 0.0)
            else:
                acc += np.where(valid, resampled * w * pixfrac_weight, 0.0)
                weight_map += np.where(valid, w * pixfrac_weight, 0.0)
        else:
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


def _sigma_clip_tile_with_mask(tile: np.ndarray, sigma: float, max_iters: int,
                               weights: Optional[np.ndarray], winsorize: bool,
                               use_mad: bool = True):
    """Like _sigma_clip_tile but also returns the (N, th, tw, C) rejection mask."""
    N = tile.shape[0]
    mask = np.ones(tile.shape, dtype=bool)
    masked = tile.astype(np.float32, copy=True)

    for iteration in range(max_iters):
        with np.errstate(all='ignore'):
            if use_mad:
                median = np.nanmedian(masked, axis=0)
                spread = np.nanmedian(np.abs(masked - median[np.newaxis]), axis=0) * 1.4826
                center = median
            else:
                center = np.nanmean(masked, axis=0)
                spread = np.nanstd(masked, axis=0)
        zero_spread = spread < 1e-12
        if np.any(zero_spread):
            with np.errstate(all='ignore'):
                fallback = np.nanstd(masked, axis=0) if use_mad else \
                           np.nanmedian(np.abs(masked - np.nanmedian(masked, axis=0)[np.newaxis]), axis=0) * 1.4826
            spread[zero_spread] = fallback[zero_spread]
        deviation = np.abs(masked - center[np.newaxis])
        new_mask = mask & (deviation <= sigma * spread[np.newaxis])
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
        if total_valid > 0 and rejected / total_valid < 0.001:
            break

    rejection_mask = ~mask  # True where rejected

    if winsorize:
        with np.errstate(all='ignore'):
            if use_mad:
                med_final = np.nanmedian(masked, axis=0)
                spread_final = np.nanmedian(np.abs(masked - med_final[np.newaxis]), axis=0) * 1.4826
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
            return (np.sum(clipped * w, axis=0) / np.sum(w)).astype(np.float32), rejection_mask
        return np.mean(clipped, axis=0).astype(np.float32), rejection_mask
    else:
        if weights is not None:
            w = np.where(mask, weights[:, np.newaxis, np.newaxis, np.newaxis], 0.0)
            with np.errstate(all='ignore'):
                total_w = np.sum(w, axis=0)
                total_w[total_w == 0] = 1.0
                result = np.nansum(masked * w, axis=0) / total_w
            np.nan_to_num(result, copy=False, nan=0.0)
            return result.astype(np.float32), rejection_mask
        with np.errstate(all='ignore'):
            result = np.nanmean(masked, axis=0)
        np.nan_to_num(result, copy=False, nan=0.0)
        return result.astype(np.float32), rejection_mask


def sigma_clip_combine(data: np.ndarray, sigma: float = 3.0, max_iters: int = 3,
                       weights: Optional[np.ndarray] = None,
                       winsorize: bool = False,
                       use_mad: bool = True,
                       verbose: bool = False,
                       return_mask: bool = False):
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

    # Native fast path (Rust): per-pixel sigma-clip, row-parallel, streaming the
    # float32 memmap in place (no per-tile copies). Only when a rejection mask is
    # not requested — the mask variant stays on the numpy path.
    if not return_mask and _native_usable(data):
        w32 = weights.astype(np.float32, copy=False) if weights is not None else None
        try:
            result = _native.sigma_clip_combine(
                data, float(sigma), int(max_iters), w32, bool(winsorize), bool(use_mad))
            estimator = 'MAD' if use_mad else 'std'
            mode = 'winsorized' if winsorize else 'reject'
            safe_print(f"    [rust] sigma-clip combine (estimator={estimator}, mode={mode})")
            return result
        except Exception as exc:
            _log.debug("native sigma_clip_combine failed (%s); using numpy", exc)

    tile_size = _adaptive_tile_size(N, C)
    result = np.zeros((H, W, C), dtype=np.float32)
    rej_mask_full = _make_rej_mask(N, H, W, C) if return_mask else None

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
        if return_mask:
            tile_result, tile_mask = _sigma_clip_tile_with_mask(tile, sigma, max_iters, weights, winsorize, use_mad)
            return coords, tile_result, tile_mask
        return coords, _sigma_clip_tile(tile, sigma, max_iters, weights, winsorize, use_mad), None

    n_tile_workers = _cap_tile_workers(min(os.cpu_count() or 4, len(tile_coords)), N, tile_size, C)
    with ThreadPoolExecutor(max_workers=n_tile_workers) as executor:
        for coords, tile_result, tile_mask in executor.map(_process_tile, tile_coords):
            ty, ty_end, tx, tx_end = coords
            result[ty:ty_end, tx:tx_end, :] = tile_result
            if return_mask and tile_mask is not None:
                rej_mask_full[:, ty:ty_end, tx:tx_end, :] = tile_mask

    if verbose:
        estimator = 'MAD' if use_mad else 'std'
        mode = 'winsorized' if winsorize else 'reject'
        safe_print(f"    Tiled sigma-clip: {n_tiles_y * n_tiles_x} tiles of "
                   f"{tile_size}x{tile_size}, estimator={estimator}, mode={mode}")
    if return_mask:
        return result, rej_mask_full
    return result


def online_sigma_clip_seed_burnin(burn_stack: np.ndarray, coverage: np.ndarray,
                                   sigma: float = 3.0
                                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Seed a running Welford (mean, M2, n_acc) state from a small ``(K,H,W,C)``
    burn-in stack via one MAD-reject pass over the whole window.

    ``K`` is expected small and bounded (e.g. 10 — the streaming stacker's
    burn-in size), not the full session frame count. Part of the online
    (single-pass, frame-at-a-time) sigma-clip combine used by ``--stream``;
    see ``online_sigma_clip_fold_frame`` for the per-frame update this seeds.

    ``coverage`` is ``(K,H,W)`` (>=0.5 = that frame's warp covers this pixel;
    one mask per burn-in frame). A burn-in window mixes several frames' own
    out-of-frame zero-fill regions (large dithers easily reach 100+ px), so
    uncovered samples are excluded from the median/MAD/mean/M2 computation
    entirely — otherwise zero-fill pixels masquerade as real (very dark)
    samples at every frame's border.

    Returns ``(mean, m2, n_acc)`` each ``(H, W, C)`` float64, plus the
    rejected-sample count (summed over all pixels in the burn-in window;
    uncovered samples count as rejected too, since they never became a
    sample).
    """
    if _native_usable(burn_stack):
        try:
            mean, m2, n_acc, n_rej = _native.online_sigma_clip_seed_burnin(
                burn_stack, np.ascontiguousarray(coverage, dtype=np.float32), float(sigma))
            return np.asarray(mean), np.asarray(m2), np.asarray(n_acc), int(n_rej)
        except Exception as exc:
            _log.debug("native online_sigma_clip_seed_burnin failed (%s); using numpy", exc)

    burn_arr = burn_stack.astype(np.float64)
    valid = (coverage >= 0.5)[:, :, :, None]  # (K,H,W,1) -> broadcasts over C
    masked = np.where(valid, burn_arr, np.nan)
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        med = np.nan_to_num(np.nanmedian(masked, axis=0), nan=0.0)
        mad = np.nan_to_num(np.nanmedian(np.abs(masked - med), axis=0), nan=0.0)
    robust_sigma = np.maximum(1.4826 * mad, 1e-6)
    thresh0 = sigma * robust_sigma
    accept = valid & (np.abs(burn_arr - med) <= thresh0)

    n_acc = np.maximum(accept.sum(axis=0).astype(np.float64), 1.0)
    running_mean = np.where(accept, burn_arr, 0.0).sum(axis=0) / n_acc
    m2 = np.where(accept, (burn_arr - running_mean) ** 2, 0.0).sum(axis=0)
    n_rejected = int(np.size(accept) - accept.sum())
    return running_mean, m2, n_acc, n_rejected


def online_sigma_clip_fold_frame(mean: np.ndarray, m2: np.ndarray, n_acc: np.ndarray,
                                  frame: np.ndarray, coverage: np.ndarray,
                                  sigma: float = 3.0
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Fold ONE new frame into a running Welford (mean, M2, n_acc) state.

    Elementwise accept-test + update — no N-samples-per-pixel gather, unlike
    the batch combine kernels; this is the per-frame step a true streaming
    (one-frame-in-memory-at-a-time) stack calls once per accepted frame.
    ``coverage`` (H,W), >=0.5 = covered, matches the shift-coverage mask
    ``LiveStacker`` already computes (live_stack.py): pixels a shift didn't
    fill from within the frame pass through unchanged — not a sample, not a
    rejection either.

    Returns fresh ``(mean, m2, n_acc)`` arrays plus the rejected-pixel count
    for this frame (among covered pixels only).
    """
    if (HAS_NATIVE and mean.ndim == 3 and frame.ndim == 3 and coverage.ndim == 2
            and mean.dtype == np.float64 and frame.dtype == np.float32):
        try:
            new_mean, new_m2, new_n_acc, n_rej = _native.online_sigma_clip_fold_frame(
                np.ascontiguousarray(mean), np.ascontiguousarray(m2),
                np.ascontiguousarray(n_acc), np.ascontiguousarray(frame, dtype=np.float32),
                np.ascontiguousarray(coverage, dtype=np.float32), float(sigma))
            return np.asarray(new_mean), np.asarray(new_m2), np.asarray(new_n_acc), int(n_rej)
        except Exception as exc:
            _log.debug("native online_sigma_clip_fold_frame failed (%s); using numpy", exc)

    x = frame.astype(np.float64)
    var_est = m2 / np.maximum(n_acc, 1.0)
    std_est = np.sqrt(np.maximum(var_est, 1e-12))
    covered = (coverage >= 0.5)[:, :, None]
    accept = (np.abs(x - mean) <= sigma * std_est) & covered

    n_acc_new = n_acc + accept
    delta = x - mean
    inv_n = 1.0 / np.maximum(n_acc_new, 1.0)
    new_mean = np.where(accept, mean + delta * inv_n, mean)
    delta2 = x - new_mean
    new_m2 = np.where(accept, m2 + delta * delta2, m2)

    covered_count = int(np.broadcast_to(covered, mean.shape).sum())
    n_rejected = covered_count - int(accept.sum())
    return new_mean, new_m2, n_acc_new, n_rejected


def median_combine(data: np.ndarray, verbose: bool = False) -> np.ndarray:
    """Median-combine frames in spatial tiles to bound peak memory.

    ``np.median(data, axis=0)`` over an ``(N, H, W, C)`` stack materialises a
    full sort buffer the size of the whole stack at once.  Processing the image
    in tiles keeps peak memory to roughly one tile-stack, matching the other
    rejection combiners, while producing an identical result.

    Args:
        data: Array of shape ``(N, H, W, C)`` (all aligned frames; usually a memmap).
        verbose: Print tiling summary.
    """
    N, H, W, C = data.shape

    if _native_usable(data):
        try:
            result = _native.median_combine(data)
            safe_print("    [rust] median combine")
            return result
        except Exception as exc:
            _log.debug("native median_combine failed (%s); using numpy", exc)

    tile_size = _adaptive_tile_size(N, C)
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
        tile = np.asarray(data[:, ty:ty_end, tx:tx_end, :], dtype=np.float32)
        return coords, np.median(tile, axis=0).astype(np.float32)

    n_workers = _cap_tile_workers(min(os.cpu_count() or 4, len(tile_coords)), N, tile_size, C)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for coords, tile_result in executor.map(_process_tile, tile_coords):
            ty, ty_end, tx, tx_end = coords
            result[ty:ty_end, tx:tx_end, :] = tile_result

    if verbose:
        safe_print(f"    Tiled median: {n_tiles_y * n_tiles_x} tiles of "
                   f"{tile_size}x{tile_size}")
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
                            verbose: bool = False,
                            return_mask: bool = False):
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

    if not return_mask and _native_usable(data):
        w32 = weights.astype(np.float32, copy=False) if weights is not None else None
        try:
            result = _native.percentile_clip_combine(data, float(low), float(high), w32)
            safe_print(f"    [rust] percentile-clip combine ([{low}, {high}])")
            return result
        except Exception as exc:
            _log.debug("native percentile_clip_combine failed (%s); using numpy", exc)

    tile_size = _adaptive_tile_size(N, C)
    result = np.zeros((H, W, C), dtype=np.float32)
    rej_mask_full = _make_rej_mask(N, H, W, C) if return_mask else None

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
        tile_result = _percentile_clip_tile(tile, low, high, weights)
        if return_mask:
            lo = np.percentile(tile, low, axis=0)
            hi = np.percentile(tile, high, axis=0)
            tile_mask = ~((tile >= lo[np.newaxis]) & (tile <= hi[np.newaxis]))
            return coords, tile_result, tile_mask
        return coords, tile_result, None

    n_workers = _cap_tile_workers(min(os.cpu_count() or 4, len(tile_coords)), N, tile_size, C)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for coords, tile_result, tile_mask in executor.map(_process_tile, tile_coords):
            ty, ty_end, tx, tx_end = coords
            result[ty:ty_end, tx:tx_end, :] = tile_result
            if return_mask and tile_mask is not None:
                rej_mask_full[:, ty:ty_end, tx:tx_end, :] = tile_mask

    if verbose:
        safe_print(f"    Percentile clip: low={low}%, high={high}%, "
                   f"{n_tiles_y * n_tiles_x} tiles of {tile_size}x{tile_size}")
    if return_mask:
        return result, rej_mask_full
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


def _esd_lambda_table(N: int, max_outliers: int, significance: float) -> np.ndarray:
    """Grubbs critical-value table λ[(n_active, iteration)] for the native ESD
    kernel. Identical formula to `_esd_clip_tile`; +inf where undefined."""
    from scipy import stats as scipy_stats
    lut = np.full((N + 1, max_outliers), np.inf, dtype=np.float64)
    for n_eff in range(3, N + 1):
        for i in range(min(max_outliers, n_eff - 2)):
            n_cur = n_eff - i
            if n_cur <= 2:
                continue
            p = significance / (2.0 * n_cur)
            p = min(max(p, 1e-10), 0.4999)
            df = max(n_cur - 2, 1)
            t_crit = scipy_stats.t.ppf(1.0 - p, df=df)
            denom = np.sqrt((n_cur - 2.0 + t_crit ** 2) * n_cur)
            lut[n_eff, i] = (n_cur - 1.0) * t_crit / denom if denom > 0 else np.inf
    return lut


def esd_combine(data: np.ndarray, max_outliers: int = 0, significance: float = 0.05,
                weights: Optional[np.ndarray] = None,
                verbose: bool = False,
                return_mask: bool = False):
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

    if not return_mask and _native_usable(data):
        try:
            lut = _esd_lambda_table(N, max_outliers, significance)  # needs scipy
            w32 = weights.astype(np.float32, copy=False) if weights is not None else None
            result = _native.esd_combine(data, int(max_outliers), lut, w32)
            safe_print(f"    [rust] ESD combine (max_outliers={max_outliers}, "
                       f"significance={significance})")
            return result
        except Exception as exc:
            _log.debug("native esd_combine failed (%s); using numpy", exc)

    tile_size = _adaptive_tile_size(N, C)
    result = np.zeros((H, W, C), dtype=np.float32)
    rej_mask_full = _make_rej_mask(N, H, W, C) if return_mask else None

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
        tile_result = _esd_clip_tile(tile, max_outliers, significance, weights)
        if return_mask:
            # Re-compute mask for return: use nanmean+nanstd approach (approx)
            # The ESD mask is expensive to recompute; use a simpler proxy mask
            # based on whether each pixel is outside sigma*std of the result
            with np.errstate(all='ignore'):
                mean_r = np.mean(tile, axis=0)
                std_r = np.std(tile, axis=0)
                std_r = np.maximum(std_r, 1e-12)
                tile_mask = np.abs(tile - mean_r[np.newaxis]) > 3.0 * std_r[np.newaxis]
            return coords, tile_result, tile_mask
        return coords, tile_result, None

    n_workers = _cap_tile_workers(min(os.cpu_count() or 4, len(tile_coords)), N, tile_size, C)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for coords, tile_result, tile_mask in executor.map(_process_tile, tile_coords):
            ty, ty_end, tx, tx_end = coords
            result[ty:ty_end, tx:tx_end, :] = tile_result
            if return_mask and tile_mask is not None:
                rej_mask_full[:, ty:ty_end, tx:tx_end, :] = tile_mask

    if verbose:
        safe_print(f"    ESD: max_outliers={max_outliers}, significance={significance}, "
                   f"{n_tiles_y * n_tiles_x} tiles of {tile_size}x{tile_size}")
    if return_mask:
        return result, rej_mask_full
    return result


def trimmed_mean_combine(data: np.ndarray, trim_low: float = 0.2, trim_high: float = 0.2,
                         weights: Optional[np.ndarray] = None,
                         verbose: bool = False) -> np.ndarray:
    """Combine frames using trimmed mean (sorted, discard low/high fractions, mean)."""
    N, H, W, C = data.shape

    if _native_usable(data):
        try:
            result = _native.trimmed_mean_combine(data, float(trim_low), float(trim_high))
            safe_print(f"    [rust] trimmed-mean combine (trim=[{trim_low}, {trim_high}])")
            return result
        except Exception as exc:
            _log.debug("native trimmed_mean_combine failed (%s); using numpy", exc)

    tile_size = _adaptive_tile_size(N, C)
    result = np.zeros((H, W, C), dtype=np.float32)
    n_tiles_y = (H + tile_size - 1) // tile_size
    n_tiles_x = (W + tile_size - 1) // tile_size
    tile_coords = [(ty * tile_size, min((ty + 1) * tile_size, H),
                    tx * tile_size, min((tx + 1) * tile_size, W))
                   for ty in range(n_tiles_y) for tx in range(n_tiles_x)]

    def _process_tile(coords):
        ty, ty_end, tx, tx_end = coords
        tile = np.array(data[:, ty:ty_end, tx:tx_end, :], dtype=np.float32)
        N_t = tile.shape[0]
        n_low = max(0, int(np.floor(N_t * trim_low)))
        n_high = max(0, int(np.floor(N_t * trim_high)))
        n_keep = N_t - n_low - n_high
        if n_keep < 1:
            n_keep = 1
            n_low = 0
            n_high = 0
        sorted_tile = np.sort(tile, axis=0)
        trimmed = sorted_tile[n_low:n_low + n_keep]
        return coords, np.mean(trimmed, axis=0).astype(np.float32)

    n_workers = _cap_tile_workers(min(os.cpu_count() or 4, len(tile_coords)), N, tile_size, C)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for coords, tile_result in executor.map(_process_tile, tile_coords):
            ty, ty_end, tx, tx_end = coords
            result[ty:ty_end, tx:tx_end, :] = tile_result
    if verbose:
        safe_print(f"    Trimmed mean: trim_low={trim_low}, trim_high={trim_high}")
    return result


_PATCH_FRAME_CHUNK = 64  # frames processed per vectorised batch inside each tile worker


def _grid_sample_axis(n_out: int, offset: float, n_grid: int,
                      full: float) -> Tuple[np.ndarray, np.ndarray]:
    """Corner-aligned bilinear sample coordinates for one axis: output pixel i
    maps to grid coordinate (i+offset)*(n_grid-1)/(full-1) — the same mapping
    scipy zoom(order=1) uses when upsampling a grid to the full frame."""
    scale = (n_grid - 1) / (full - 1.0) if (full > 1.0 and n_grid > 1) else 0.0
    g = np.clip((np.arange(n_out, dtype=np.float64) + offset) * scale,
                0.0, n_grid - 1)
    g0 = np.minimum(g.astype(np.int64), max(n_grid - 2, 0))
    return g0, (g - g0).astype(np.float32)


def patch_weighted_mean_combine(
    mem_aligned: np.ndarray,
    quality_maps: List[np.ndarray],
    global_weights: Optional[np.ndarray] = None,
    rejection_mask: Optional[np.ndarray] = None,
    grid_geom: Optional[Tuple[float, float, float, float]] = None,
) -> np.ndarray:
    """Quality-map-weighted mean combine for per-pixel lucky-imaging stacking.

    Each frame contributes to each output pixel with a weight equal to its
    normalised per-patch quality score at that position, optionally further
    multiplied by the frame's global quality weight.  Pixels that fall in a
    sharp isoplanatic patch of a frame receive more weight than pixels in a
    blurry region of the same frame — exactly the lucky imaging principle
    applied to deep-sky (non-planetary) data.

    Implemented as a tiled, parallel combine matching the pattern used by
    sigma_clip_combine and median_combine.  Each tile reads frames in batches
    of _PATCH_FRAME_CHUNK and accumulates with a vectorised einsum, eliminating
    the per-frame Python loop overhead that dominated runtime for large stacks.

    Args:
        mem_aligned:    (N, H, W, C) float32 memmap of aligned, cropped frames.
        quality_maps:   List of N (H, W) float32 per-pixel quality weight arrays.
                        Must be pre-cropped to match the aligned frame dimensions.
        global_weights: Optional (N,) global per-frame weights (e.g. SNR-based).

    Returns:
        (H, W, C) float32 stacked image.
    """
    N, H, W, C = mem_aligned.shape
    tile_size = Config.TILE_SIZE

    gw = (np.asarray(global_weights, dtype=np.float32)
          if global_weights is not None else None)

    # Coarse-grid mode: quality_maps are (gh, gw) patch grids; sample them
    # bilinearly at full-frame coordinates instead of requiring full-res maps.
    _gtabs = None
    if grid_geom is not None and len(quality_maps) > 0:
        h_full, w_full, top_off, left_off = grid_geom
        gh, gw_ = quality_maps[0].shape
        gy0, gyf = _grid_sample_axis(H, top_off, gh, h_full)
        gx0, gxf = _grid_sample_axis(W, left_off, gw_, w_full)
        _gtabs = (gy0, gyf, gx0, gxf, gh, gw_)

    result = np.zeros((H, W, C), dtype=np.float32)

    n_tiles_y = (H + tile_size - 1) // tile_size
    n_tiles_x = (W + tile_size - 1) // tile_size
    tile_coords = [
        (ty * tile_size, min((ty + 1) * tile_size, H),
         tx * tile_size, min((tx + 1) * tile_size, W))
        for ty in range(n_tiles_y) for tx in range(n_tiles_x)
    ]

    def _process_tile(coords):
        ty, ty_end, tx, tx_end = coords
        th, tw = ty_end - ty, tx_end - tx

        acc = np.zeros((th, tw, C), dtype=np.float64)
        wsum = np.zeros((th, tw), dtype=np.float64)

        for start in range(0, N, _PATCH_FRAME_CHUNK):
            end = min(start + _PATCH_FRAME_CHUNK, N)
            F = end - start

            # (F, th, tw, C) float32 — one strided memmap read per chunk
            chunk = np.asarray(mem_aligned[start:end, ty:ty_end, tx:tx_end, :],
                               dtype=np.float32)

            # Per-frame patch weights for this tile: (F, th, tw)
            qstack = np.empty((F, th, tw), dtype=np.float32)
            if _gtabs is not None:
                gy0, gyf, gx0, gxf, gh, gw_ = _gtabs
                y0i = gy0[ty:ty_end]
                y1i = np.minimum(y0i + 1, gh - 1)
                fy = gyf[ty:ty_end][:, None]
                x0i = gx0[tx:tx_end]
                x1i = np.minimum(x0i + 1, gw_ - 1)
                fx = gxf[tx:tx_end][None, :]
                for k, j in enumerate(range(start, end)):
                    g = quality_maps[j]
                    q00 = g[np.ix_(y0i, x0i)]
                    q01 = g[np.ix_(y0i, x1i)]
                    q10 = g[np.ix_(y1i, x0i)]
                    q11 = g[np.ix_(y1i, x1i)]
                    top_v = q00 + (q01 - q00) * fx
                    bot_v = q10 + (q11 - q10) * fx
                    qstack[k] = top_v + (bot_v - top_v) * fy
            else:
                for k, j in enumerate(range(start, end)):
                    qmap = quality_maps[j]
                    if qmap.shape[0] >= ty_end and qmap.shape[1] >= tx_end:
                        qstack[k] = qmap[ty:ty_end, tx:tx_end]
                    else:
                        from scipy.ndimage import zoom as _zoom
                        zoomed = _zoom(qmap, (H / qmap.shape[0], W / qmap.shape[1]), order=1)
                        qstack[k] = np.clip(zoomed[ty:ty_end, tx:tx_end], 0.0, 1.0)

            w = qstack
            if gw is not None:
                w = w * gw[start:end, np.newaxis, np.newaxis]

            if rejection_mask is not None:
                rej = np.asarray(rejection_mask[start:end, ty:ty_end, tx:tx_end, :])
                w = w * (1.0 - rej.mean(axis=3).astype(np.float32))

            # Vectorised weighted sum across the frame dimension.
            # einsum avoids materialising an (F, th, tw, C) float64 intermediate.
            acc  += np.einsum('fhwc,fhw->hwc', chunk.astype(np.float64),
                              w.astype(np.float64))
            wsum += w.astype(np.float64).sum(axis=0)

        safe_denom = np.maximum(wsum[:, :, np.newaxis], 1e-12)
        return coords, (acc / safe_denom).astype(np.float32)

    # Peak per worker per chunk iteration: float64 chunk + float64 w (both copied)
    per_worker_mb = (_PATCH_FRAME_CHUNK * tile_size * tile_size
                     * (C * 8 + 8) * 2) / 1e6
    n_workers = min(os.cpu_count() or 4, len(tile_coords))
    try:
        import psutil
        avail_mb = psutil.virtual_memory().available / 1e6
        n_workers = min(n_workers, max(1, int(avail_mb * 0.5 / per_worker_mb)))
    except ImportError:
        n_workers = min(n_workers, max(1, int(512 / per_worker_mb)))

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for coords, tile_result in executor.map(_process_tile, tile_coords):
            ty, ty_end, tx, tx_end = coords
            result[ty:ty_end, tx:tx_end, :] = tile_result

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
    quality_maps: Optional[List[np.ndarray]] = None,
    displacement_fields: Optional[List[Optional[np.ndarray]]] = None,
) -> Tuple[np.ndarray, np.ndarray, int, int, int, int]:
    """Align and combine frames into a stacked image.

    Returns (stacked, fits_stacked, top, bottom, left, right).
    fits_stacked is the pre-post-processing copy saved in the FITS file.
    """
    from src.registration import apply_transform, calc_common_crop, sample_displacement_field

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

    extra_margin_px = 0.0
    if displacement_fields is not None:
        mags = [float(np.abs(f).max()) for f in displacement_fields if f is not None]
        extra_margin_px = max(mags) if mags else 0.0
    top, bottom, left, right = calc_common_crop(shifts, (H, W), transforms=transforms,
                                                extra_margin_px=extra_margin_px)
    stats.output_shape = (bottom - top, right - left)
    stats.cropped_pixels = (H - (bottom - top), W - (right - left))

    drizzle_scale = getattr(args, 'drizzle_scale', 1.0)
    use_aligned_memmap = (drizzle_scale <= 1.0 and
                          args.stack_method in ('median', 'sigma_clip', 'winsorized',
                                                'percentile', 'esd', 'trimmed_mean'))
    if use_aligned_memmap:
        mm_aligned_path = os.path.join(tempfile.gettempdir(), f'stack_aligned_{os.getpid()}.dat')
        crop_h, crop_w = bottom - top, right - left
        mem_aligned = np.memmap(mm_aligned_path, dtype='float32', mode='w+',
                                shape=(n_final, crop_h, crop_w, C))

        gpu = get_gpu()

        def _align_one(j):
            with gpu.stream_context():
                rgb = np.array(mem_rgb[final_indices[j]])
                lf = (displacement_fields[j]
                      if displacement_fields is not None and j < len(displacement_fields)
                      else None)
                aligned = apply_transform(rgb, shift=shifts[j], transform=transforms[j],
                                         local_field=lf)
                mem_aligned[j] = aligned[top:bottom, left:right, :]

        n_align = (min(gpu.max_gpu_workers(Config.GPU_ALIGN_WORKER_MB,
                                           Config.GPU_VRAM_RESERVE_MB), n_final)
                   if gpu.active else min(os.cpu_count() or 4, n_final))
        # Cap alignment workers so concurrent frame reads/writes don't thrash the
        # disk when available RAM is below the total working-set size.
        # Each worker needs: one source frame + one transformed frame in memory.
        try:
            import psutil as _ps
            _avail_mb = _ps.virtual_memory().available / 1e6
            _src_mb  = H * W * C * 4 / 1e6
            _crop_mb = crop_h * crop_w * C * 4 / 1e6
            _per_align_mb = _src_mb + _crop_mb + 64          # +64 MB headroom
            _safe = max(1, int(_avail_mb * 0.6 / _per_align_mb))
            if _safe < n_align:
                n_align = _safe
        except Exception:
            pass
        from src.registration import HAS_NATIVE as _reg_native
        if _reg_native:
            safe_print("    [rust] Lanczos-3 warp (per-frame alignment)")
        _t_align = time.time()
        from src.webview import get_webview as _get_wv
        _wv = _get_wv()
        _wv_done = 0
        with ThreadPoolExecutor(max_workers=n_align) as executor:
            futures = {executor.submit(_align_one, j): j for j in range(n_final)}
            for future in tqdm(as_completed(futures), total=n_final,
                               desc="  Aligning", unit="frame", disable=not args.verbose):
                future.result()
                _wv_done += 1
                _wv.progress('Aligning frames', _wv_done, n_final)
        mem_aligned.flush()
        safe_print(f"    Alignment: {n_final} frames in {format_time(time.time() - _t_align)} "
                   f"({n_align} workers, {n_final / max(time.time() - _t_align, 1e-9):.1f} frame/s)")

        # Local Normalization: additively match every frame's background to the
        # per-frame median before the rejection combine, so per-frame gradients
        # (moonlight, LP drift, thin cloud) don't tilt the stack or skew
        # sigma-clip. Operates in place on the aligned stack.
        if getattr(args, 'local_normalize', False):
            _t_ln = time.time()
            try:
                from src.local_normalize import local_normalize_stack
                _nln = local_normalize_stack(mem_aligned, verbose=args.verbose)
                if _nln:
                    mem_aligned.flush()
                    safe_print(f"    Local normalization: {_nln} frames "
                               f"({format_time(time.time() - _t_ln)})")
            except Exception as _lnexc:
                safe_print(f"    WARNING: local normalization failed: {_lnexc}")

        # Patch-weighted lucky-imaging combine (if quality maps were computed in
        # Phase 2). quality_maps holds small per-frame patch-score GRIDS in
        # aligned space (not full-res maps): both combine paths sample them
        # bilinearly at full-frame coordinates, so N full-resolution weight
        # maps (~5 GB at 200+ frames) are never materialised.
        if quality_maps is not None and len(quality_maps) == n_final:
            print(f"  Patch-weighted mean combine ({n_final} frames × {top},{bottom},{left},{right} crop)...")
            qgrids = np.ascontiguousarray(np.stack(quality_maps), dtype=np.float32)
            qgrid_geom = (float(H), float(W), float(top), float(left))
            # Fused native fast path: sigma-clip rejection + patch weighting in a
            # single Rust pass, no (N,H,W,C) rejection-mask array. Covers the
            # sigma_clip/winsorized methods (the mask is clip-based, identical
            # for both). Falls back to the two-pass numpy path below on any error.
            _fused_done = False
            if (args.stack_method in ('sigma_clip', 'winsorized')
                    and _native_usable(mem_aligned)):
                try:
                    _t_fused = time.time()
                    _w32 = (weights.astype(np.float32, copy=False)
                            if weights is not None else None)
                    _use_mad = (getattr(args, 'rejection_estimator', 'mad') == 'mad')
                    stacked = _native.patch_weighted_sigma_combine(
                        mem_aligned, qgrids, _w32, float(args.rejection_sigma),
                        int(args.rejection_iters), _use_mad, qgrid_geom)
                    safe_print(f"    [rust] fused patch-weighted + sigma-clip combine "
                               f"({format_time(time.time() - _t_fused)})")
                    _fused_done = True
                except Exception as _fexc:
                    _log.debug("native fused patch combine failed (%s); using numpy", _fexc)

            # Pre-rejection + patch-weighted hybrid stacking (numpy two-pass)
            rej_mask = None
            if _fused_done:
                pass
            elif args.stack_method in ('sigma_clip', 'winsorized'):
                use_winsorize = (args.stack_method == 'winsorized')
                use_mad = (getattr(args, 'rejection_estimator', 'mad') == 'mad')
                _, rej_mask = sigma_clip_combine(mem_aligned, sigma=args.rejection_sigma,
                                                 max_iters=args.rejection_iters, weights=weights,
                                                 winsorize=use_winsorize, use_mad=use_mad,
                                                 return_mask=True)
            elif args.stack_method == 'percentile':
                low_p = getattr(args, 'percentile_low', 20.0)
                high_p = getattr(args, 'percentile_high', 80.0)
                _, rej_mask = percentile_clip_combine(mem_aligned, low=low_p, high=high_p,
                                                      weights=weights, return_mask=True)
            elif args.stack_method == 'esd':
                max_out = getattr(args, 'esd_max_outliers', 0)
                sig_esd = getattr(args, 'esd_significance', 0.05)
                _, rej_mask = esd_combine(mem_aligned, max_outliers=max_out, significance=sig_esd,
                                          weights=weights, return_mask=True)
            if not _fused_done:
                _t_pw = time.time()
                stacked = patch_weighted_mean_combine(mem_aligned, list(qgrids),
                                                      global_weights=weights,
                                                      rejection_mask=rej_mask,
                                                      grid_geom=qgrid_geom)
                safe_print(f"    Patch-weighted combine (numpy 2-pass): "
                           f"{format_time(time.time() - _t_pw)}")
            if rej_mask is not None:
                _free_rej_mask(rej_mask)
                rej_mask = None
        elif args.stack_method in ('sigma_clip', 'winsorized'):
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
        elif args.stack_method == 'trimmed_mean':
            tl = getattr(args, 'trim_low', 0.2)
            th = getattr(args, 'trim_high', 0.2)
            print(f"  Trimmed mean: trim_low={tl}, trim_high={th}")
            stacked = trimmed_mean_combine(mem_aligned, trim_low=tl, trim_high=th,
                                           weights=weights, verbose=args.verbose)
        else:
            stacked = median_combine(mem_aligned, verbose=args.verbose)

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
            pixfrac = float(getattr(args, 'drizzle_pixfrac', 1.0))
            use_pixfrac = pixfrac < 1.0 - 1e-9
            msg = f"  Drizzle: {drizzle_scale:.1f}x ({crop_h}x{crop_w} -> {out_h}x{out_w})"
            if use_pixfrac:
                msg += f", pixfrac={pixfrac:.2f}"
            print(msg)
            if HAS_NATIVE:
                print("    [rust] Lanczos-3 warp (drizzle resample)")
            acc = np.zeros((out_h, out_w, C), dtype=np.float64)
            gpu = get_gpu()

            use_elastic = displacement_fields is not None
            if use_pixfrac or use_elastic:
                # Tent-kernel footprint shrink (classic drizzle pixfrac): each
                # input pixel's contribution falls off toward the edge of its
                # shrunken footprint instead of covering the whole output cell
                # it lands on. half_drop is in raw (input-pixel) units, so it
                # stays correct under rotation -- the footprint is defined in
                # input space, independent of the output grid's orientation.
                half_drop = max(pixfrac / 2.0, 1e-6)
                _grid_oy, _grid_ox = np.meshgrid(
                    np.arange(out_h, dtype=np.float64),
                    np.arange(out_w, dtype=np.float64), indexing='ij')
                if use_pixfrac:
                    weight_map = np.zeros((out_h, out_w, 1), dtype=np.float64)
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
                lf = (displacement_fields[j]
                      if use_elastic and j < len(displacement_fields) else None)

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

                raw_y = raw_x = None
                if lf is not None:
                    # Elastic correction: bypass the native/scipy affine fast
                    # path (matrix+offset only) for a per-pixel coordinate
                    # grid instead -- sample the field at reference-space
                    # (pre-crop, pre-upscale) coordinates, then subtract the
                    # rotated displacement from the affine source coordinate
                    # (same composition as apply_transform's local_field).
                    raw_y = M[0, 0] * _grid_oy + M[0, 1] * _grid_ox + off[0]
                    raw_x = M[1, 0] * _grid_oy + M[1, 1] * _grid_ox + off[1]
                    aligned_y = _grid_oy * inv_scale + top
                    aligned_x = _grid_ox * inv_scale + left
                    dy, dx = sample_displacement_field(lf, H, W, aligned_y, aligned_x)
                    R_ = transform_j.params[:2, :2] if transform_j is not None else np.eye(2)
                    raw_y = raw_y - (R_[0, 0] * dy + R_[0, 1] * dx)
                    raw_x = raw_x - (R_[1, 0] * dy + R_[1, 1] * dx)
                    resampled = np.empty((out_h, out_w, C), dtype=np.float32)
                    for c in range(C):
                        resampled[:, :, c] = ndimage.map_coordinates(
                            rgb[:, :, c], [raw_y, raw_x], order=spline_order,
                            mode='constant', cval=0.0)
                else:
                    resampled = None
                    if HAS_NATIVE:
                        # Rust Lanczos-3 warp: all channels in one pass; the
                        # no-rotation case (diagonal M) takes its separable fast
                        # path. True Lanczos-3 vs the quintic-spline
                        # approximation of the scipy fallback.
                        try:
                            resampled = _native.warp_affine_lanczos3(
                                np.ascontiguousarray(rgb, dtype=np.float32),
                                [float(M[0, 0]), float(M[0, 1]),
                                 float(M[1, 0]), float(M[1, 1])],
                                [float(off[0]), float(off[1])],
                                int(out_h), int(out_w), 0.0)
                        except Exception:
                            resampled = None
                    if resampled is None:
                        resampled = np.empty((out_h, out_w, C), dtype=np.float32)
                        for c in range(C):
                            resampled[:, :, c] = ndimage.affine_transform(
                                rgb[:, :, c], M, offset=off,
                                output_shape=(out_h, out_w),
                                order=spline_order, mode='constant', cval=0.0)

                if use_pixfrac:
                    # Map each output pixel back to the raw frame with the
                    # same M/off used for the resample (or the elastic-
                    # corrected raw_y/raw_x above, if set), then weight it by
                    # how close it falls to the shrunken input-pixel footprint.
                    if raw_y is None:
                        raw_y = M[0, 0] * _grid_oy + M[0, 1] * _grid_ox + off[0]
                        raw_x = M[1, 0] * _grid_oy + M[1, 1] * _grid_ox + off[1]
                    frac_y = np.abs(raw_y - np.round(raw_y))
                    frac_x = np.abs(raw_x - np.round(raw_x))
                    w_y = np.maximum(0.0, 1.0 - frac_y / half_drop)
                    w_x = np.maximum(0.0, 1.0 - frac_x / half_drop)
                    pfw = (w_y * w_x * w)[:, :, np.newaxis]  # (out_h, out_w, 1)
                    resampled = resampled.astype(np.float64, copy=False) * pfw
                    with acc_lock:
                        np.add(acc, resampled, out=acc)
                        np.add(weight_map, pfw, out=weight_map)
                else:
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
                # pixfrac path builds several out_h x out_w float64 temporaries
                # (raw_y/raw_x/frac_y/frac_x/w_y/w_x/pfw) per call.
                pixfrac_buf_mb = (out_h * out_w * 8 * 7 / 1e6) if use_pixfrac else 0.0
                per_worker_mb = raw_mb + out_mb + affine_buf_mb + pixfrac_buf_mb + 100
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

            if use_pixfrac:
                # Pixels no dithered frame's shrunken footprint ever landed on
                # (holes) are left at zero rather than divided by an empty sum.
                safe_weight = np.where(weight_map <= 0.0, 1.0, weight_map)
                stacked = (acc / safe_weight).astype(np.float32)
            else:
                stacked = (acc / max(total_weight_ref[0], 1e-12)).astype(np.float32)

        else:
            acc = np.zeros((bottom - top, right - left, C), dtype=np.float64)
            total_weight = 0.0
            gpu = get_gpu()

            def _align_crop(j):
                with gpu.stream_context():
                    rgb = np.array(mem_rgb[final_indices[j]])
                    lf = (displacement_fields[j]
                          if displacement_fields is not None and j < len(displacement_fields)
                          else None)
                    return j, apply_transform(rgb, shift=shifts[j], transform=transforms[j],
                                             local_field=lf)[top:bottom, left:right, :]

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
    try:
        from src.webview import get_webview
        get_webview().preview(stacked, 'Linear stack (pre-post-processing)',
                              args=args, min_interval=0.0)
    except Exception:
        pass
    return stacked, fits_stacked, top, bottom, left, right


# ---------------------------------------------------------------------------
# HDR stack blending
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Comet dual-track stacking
# ---------------------------------------------------------------------------

def blend_comet_star_stacks(star_stack: np.ndarray,
                             comet_stack: np.ndarray,
                             comet_lum: np.ndarray,
                             blend_sigma: float = 30.0,
                             tail_pa_deg: Optional[float] = None) -> np.ndarray:
    """Blend a star-aligned and comet-aligned stack for dual-track imaging.

    In standard comet stacking the astronomer must choose between sharpening
    the comet nucleus (comet alignment) and sharpening the star field (star
    alignment).  This function blends both stacks spatially:

    • Near the comet nucleus: ``comet_stack`` dominates (sharp nucleus,
      smeared stars).
    • Away from the nucleus: ``star_stack`` dominates (sharp stars,
      smeared nucleus).

    When ``tail_pa_deg`` is provided, the blend mask is elongated along the
    tail direction with axis ratio 3:1, extending 3× further in the tail
    direction than radially.

    The blend mask is derived from the comet luminance: a Gaussian envelope
    centred on the brightest region (the nucleus) whose width is controlled
    by ``blend_sigma`` pixels.

    Args:
        star_stack:   Float32 (H, W, 3) star-aligned stack.
        comet_stack:  Float32 (H, W, 3) comet-aligned stack.
        comet_lum:    Float32 (H, W) luminance map for locating the nucleus
                      (typically the comet-aligned stack luminance).
        blend_sigma:  Gaussian half-width (px) of the comet blend zone.
        tail_pa_deg:  Optional position angle of the tail in degrees from
                      North (clockwise).  When provided, an elliptical mask
                      elongated 3:1 along the tail is used.

    Returns:
        Blended float32 image (H, W, 3).
    """
    from scipy.ndimage import gaussian_filter as _gf

    H, W = star_stack.shape[:2]

    # Locate comet nucleus (brightest smooth blob)
    smoothed = _gf(comet_lum.astype(np.float64), sigma=5.0)
    peak_flat = int(np.argmax(smoothed))
    py, px = peak_flat // W, peak_flat % W

    # Build blend mask centred on nucleus
    yy, xx = np.mgrid[:H, :W]
    dy = (yy - py).astype(np.float64)
    dx = (xx - px).astype(np.float64)

    if tail_pa_deg is not None:
        # Elliptical mask elongated along the tail direction (axis ratio 3:1)
        # PA is measured clockwise from North (up = -row direction)
        # tail_pa_deg: N=0, E=90, S=180, W=270
        pa_rad = np.radians(float(tail_pa_deg))
        # Unit vector along the tail (in row, col convention):
        # North is -row, East is +col -> tail_row = -cos(pa), tail_col = sin(pa)
        tail_row = -np.cos(pa_rad)
        tail_col = np.sin(pa_rad)
        # Project offsets onto tail direction and perpendicular
        proj_tail = dy * tail_row + dx * tail_col      # along tail axis
        proj_perp = dy * (-tail_col) + dx * tail_row   # perpendicular to tail
        # Elliptical distance: tail sigma = 3 * blend_sigma, perp sigma = blend_sigma
        sigma_tail = blend_sigma * 3.0
        sigma_perp = blend_sigma
        dist2 = (proj_tail / sigma_tail) ** 2 + (proj_perp / sigma_perp) ** 2
    else:
        # Circular Gaussian mask
        dist2 = (dy ** 2 + dx ** 2) / (blend_sigma ** 2)

    mask = np.exp(-0.5 * dist2)   # 1 at nucleus, 0 far away

    mask3 = mask[:, :, np.newaxis]
    blended = (mask3 * comet_stack.astype(np.float64)
               + (1.0 - mask3) * star_stack.astype(np.float64))
    return np.clip(blended, 0.0, None).astype(np.float32)
