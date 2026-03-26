"""Image registration: shift calculation, affine transform, cropping, dither detection."""
from __future__ import annotations

import argparse
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

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

try:
    from skimage.registration import phase_cross_correlation
except Exception:
    phase_cross_correlation = None

try:
    from skimage.transform import EuclideanTransform
    from skimage.measure import ransac
    HAS_SKIMAGE_TRANSFORM = True
except Exception:
    HAS_SKIMAGE_TRANSFORM = False

try:
    from PIL import Image
except Exception:
    Image = None


def match_stars_affine(ref_positions, img_positions,
                       initial_shift: Tuple[float, float] = (0.0, 0.0)):
    """Match star catalogs and compute a Euclidean (rotation+translation) transform.

    Uses nearest-neighbor matching after applying an initial translation estimate,
    then RANSAC-robust fitting. Returns None if matching fails.
    """
    if not HAS_SKIMAGE_TRANSFORM:
        return None
    from scipy.spatial import cKDTree

    max_stars = Config.AFFINE_MAX_STARS
    if ref_positions is None or img_positions is None:
        return None
    if len(ref_positions) < 3 or len(img_positions) < 3:
        return None

    ref_pts = np.array([(float(s['xcentroid']), float(s['ycentroid']))
                        for s in ref_positions[:max_stars]])
    img_pts = np.array([(float(s['xcentroid']), float(s['ycentroid']))
                        for s in img_positions[:max_stars]])

    # Shift img points by initial estimate for better matching.
    # Initial shift is the amount needed to move 'img' to align with 'ref'.
    # So img_points = ref_points + shift. To find correspondence, we map 
    # img_points back to ref space: img_points - shift.
    shift_vec = np.array([initial_shift[1], initial_shift[0]]) # [sx, sy]
    img_pts_shifted = img_pts - shift_vec

    tree = cKDTree(ref_pts)
    distances, indices = tree.query(img_pts_shifted, k=1)

    good = distances < Config.AFFINE_MATCH_RADIUS
    if good.sum() < 3:
        return None

    src = img_pts[good]
    dst = ref_pts[indices[good]]

    try:
        model, inliers = ransac(
            (src, dst), EuclideanTransform,
            min_samples=3, residual_threshold=2.0, max_trials=1000
        )
        if inliers is not None and inliers.sum() >= 3:
            return model
    except Exception:
        pass
    return None


def apply_transform(img: np.ndarray, shift: Tuple[float, float] = None,
                    transform=None) -> np.ndarray:
    """Apply translation or affine transform to a multi-channel image.

    Uses cubic spline interpolation (order=3) for subpixel accuracy.
    Dispatches to GPU (cupyx.scipy.ndimage) when available.
    """
    # Ensure 3D shape for consistent channel processing (H, W, 1) for grayscale
    original_ndim = img.ndim
    if original_ndim == 2:
        img = img[:, :, np.newaxis]
        squeeze_back = True
    else:
        squeeze_back = False

    gpu = get_gpu()
    xp = gpu.xp
    _ndimage = gpu.xndimage
    img_d = gpu.to_device(img)
    
    if transform is not None:
        matrix = transform.params
        R = matrix[:2, :2]        # EuclideanTransform stores t in (x=col, y=row) space.
        # scipy.ndimage.affine_transform operates in (row, col) space and applies the
        # inverse mapping: input_coord = M @ output_coord + offset.
        t_xy = matrix[:2, 2]      # [tx, ty] in (col, row)
        t_rowcol = np.array([t_xy[1], t_xy[0]])  # swap to (row, col)
        offset = -R @ t_rowcol
        
        if gpu.active:
            mat_d = xp.asarray(R)
            off_d = xp.asarray(offset)
        else:
            mat_d = R
            off_d = offset

        result = xp.zeros_like(img_d)
        for c in range(img_d.shape[2]):
            result[:, :, c] = _ndimage.affine_transform(
                img_d[:, :, c], mat_d, offset=off_d,
                order=3, mode='constant', cval=0.0
            )
        return gpu.to_host(result)
        
    elif shift is not None:
        result = xp.zeros_like(img_d)
        for c in range(img_d.shape[2]):
            result[:, :, c] = _ndimage.shift(
                img_d[:, :, c], shift=shift, order=3,
                mode='constant', cval=0.0, prefilter=True
            )
        return gpu.to_host(result)
        
    # If neither transform nor shift is provided, return original img
    # (ensure it is on CPU as per function convention)
    if squeeze_back:
        img = img[0]
    return img


def calculate_shift(ref: np.ndarray, img: np.ndarray, upsample: int = 10, verbose: bool = False, debug: bool = False, frame_name: str = "", skip_phase_cc: bool = False, use_pyramid: bool = True) -> Tuple[float, float]:
    """Calculate the (shift_y, shift_x) needed to align ``img`` to ``ref``.

    Registration cascade:
      1. Multi-scale pyramid (coarse-to-fine) — handles large shifts reliably
      2. Phase cross-correlation (skimage) — sub-pixel accurate when error < 0.1.
      3. FFT cross-correlation at full resolution — fallback for phase-cc failure.
      4. Centroid difference — last resort for featureless or very noisy frames.
    """
    debug_info = []

    # Debug mode: save diagnostic images
    if debug and frame_name:
        try:
            os.makedirs('_registration_debug', exist_ok=True)
            # Save normalized versions
            ref_norm = (ref - np.min(ref)) / (np.max(ref) - np.min(ref) + 1e-12) * 255
            img_norm = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-12) * 255

            if Image is not None:
                Image.fromarray(ref_norm.astype(np.uint8)).save(f'_registration_debug/{frame_name}_ref.png')
                Image.fromarray(img_norm.astype(np.uint8)).save(f'_registration_debug/{frame_name}_img.png')

            # Save statistics
            with open(f'_registration_debug/{frame_name}_stats.txt', 'w') as f:
                f.write(f"Reference:\n  min={np.min(ref):.2f}, max={np.max(ref):.2f}, mean={np.mean(ref):.2f}, std={np.std(ref):.2f}\n")
                f.write(f"Image:\n  min={np.min(img):.2f}, max={np.max(img):.2f}, mean={np.mean(img):.2f}, std={np.std(img):.2f}\n")
        except Exception as e:
            pass

    # Step 1: Multi-scale pyramid registration (coarse-to-fine).
    pyramid_shift = (0.0, 0.0)
    if use_pyramid:
        try:
            psy, psx = calculate_shift_pyramid(ref, img)
            if np.isfinite(psy) and np.isfinite(psx):
                pyramid_shift = (psy, psx)
                debug_info.append(f"pyramid: shift=({psx:.1f}, {psy:.1f})")
        except Exception as exc:
            debug_info.append(f"pyramid error: {type(exc).__name__}")

    # Use phase cross correlation for subpixel shifts when available
    if not skip_phase_cc and phase_cross_correlation is not None:
        try:
            psy, psx = pyramid_shift
            if psy != 0.0 or psx != 0.0:
                img_pre = ndimage.shift(img, shift=(psy, psx), order=1,
                                        mode='constant', cval=0.0)
            else:
                img_pre = img

            ref_norm = (ref - np.mean(ref)) / (np.std(ref) + 1e-12)
            img_norm = (img_pre - np.mean(img_pre)) / (np.std(img_pre) + 1e-12)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                shift, error, diffphase = phase_cross_correlation(ref_norm, img_norm, upsample_factor=upsample)
            debug_info.append(f"phase_cc: shift={shift}, error={error:.4f}")

            if np.isfinite(shift).all() and np.abs(shift).max() < max(ref.shape) * 0.5:
                if error < 0.1:
                    total_sy = float(shift[0]) + psy
                    total_sx = float(shift[1]) + psx
                    if verbose:
                        if total_sy == 0.0 and total_sx == 0.0:
                            print(f"      [phase_correlation: zero shift (error={error:.4f}, images well-aligned)]")
                        else:
                            print(f"      [phase_correlation succeeded: shift=({total_sx:.3f}, {total_sy:.3f}), error={error:.4f}]")
                    return total_sy, total_sx
                else:
                    debug_info.append(f"phase_cc rejected: high error ({error:.4f} >= 0.1)")
            else:
                debug_info.append(f"phase_cc rejected: nan/inf or too large ({np.abs(shift).max():.1f} > {max(ref.shape) * 0.5:.1f})")
        except Exception as e:
            debug_info.append(f"phase_cc error: {type(e).__name__}")

    # FFT cross-correlation at full resolution with zero-padding.
    try:
        gpu = get_gpu()
        xp = gpu.xp

        psy, psx = pyramid_shift
        if psy != 0.0 or psx != 0.0:
            img_fft = ndimage.shift(img, shift=(psy, psx), order=1,
                                    mode='constant', cval=0.0)
        else:
            img_fft = img

        ref_norm = xp.asarray((ref - np.mean(ref)), dtype=xp.float64)
        img_norm = xp.asarray((img_fft - np.mean(img_fft)), dtype=xp.float64)

        h, w = ref_norm.shape
        pad_h, pad_w = 2 * h, 2 * w
        F_ref = xp.fft.rfft2(ref_norm, s=(pad_h, pad_w))
        F_img = xp.fft.rfft2(img_norm, s=(pad_h, pad_w))
        corr = xp.fft.irfft2(F_ref * xp.conj(F_img), s=(pad_h, pad_w))
        del F_ref, F_img, ref_norm, img_norm  # free VRAM immediately

        peak_flat = int(xp.argmax(corr))
        peak = (peak_flat // corr.shape[1], peak_flat % corr.shape[1])
        dy = peak[0] if peak[0] < h else peak[0] - pad_h
        dx = peak[1] if peak[1] < w else peak[1] - pad_w

        # Parabolic subpixel refinement (wrap-safe)
        py, px = peak
        sub_y = sub_x = 0.0
        vc = float(corr[py, px])
        vm = float(corr[(py - 1) % pad_h, px])
        vp = float(corr[(py + 1) % pad_h, px])
        denom = 2.0 * (2.0 * vc - vm - vp)
        if abs(denom) > 1e-12:
            sub_y = max(-0.5, min(0.5, (vm - vp) / denom))
        vm = float(corr[py, (px - 1) % pad_w])
        vp = float(corr[py, (px + 1) % pad_w])
        denom = 2.0 * (2.0 * vc - vm - vp)
        if abs(denom) > 1e-12:
            sub_x = max(-0.5, min(0.5, (vm - vp) / denom))
        del corr  # free VRAM

        shift_y = float(dy + sub_y) + psy
        shift_x = float(dx + sub_x) + psx

        if np.isfinite(shift_y) and np.isfinite(shift_x) and max(abs(shift_y), abs(shift_x)) < max(h, w) * 0.5:
            if verbose:
                print(f"      [fft_xcorr: shift=({shift_x:.1f}, {shift_y:.1f})]")
            return shift_y, shift_x
        else:
            debug_info.append(f"fft_xcorr rejected: shift ({shift_y:.1f}, {shift_x:.1f}) too large")
    except Exception as e:
        debug_info.append(f"fft_xcorr error: {type(e).__name__}")

    # Fallback to centroid difference
    best_shift = (0.0, 0.0)
    best_score = float('inf')

    for percentile in Config.CENTROID_PERCENTILES:
        try:
            thresh_ref = np.percentile(ref, percentile)
            thresh_img = np.percentile(img, percentile)
            rmask = ref > thresh_ref
            imask = img > thresh_img
            n_ref = rmask.sum()
            n_img = imask.sum()

            if n_ref > 10 and n_img > 10:
                cim = ndimage.center_of_mass(ref * rmask)
                cim2 = ndimage.center_of_mass(img * imask)
                if cim and cim2:
                    shift_y = float(cim[0] - cim2[0])
                    shift_x = float(cim[1] - cim2[1])
                    shift_mag = np.sqrt(shift_x**2 + shift_y**2)

                    if shift_mag > 0.1 or n_ref > 50: 
                        if verbose:
                            print(f"      [centroid fallback (p{percentile}): ({shift_x:.1f}, {shift_y:.1f})]")
                        return shift_y, shift_x

            debug_info.append(f"centroid(p{percentile}): n_ref={n_ref}, n_img={n_img}")
        except Exception as e:
            pass

    if verbose and debug_info:
        print(f"      [CRITICAL: no registration method succeeded] " + " | ".join(debug_info))
    return 0.0, 0.0


def apply_shift(img: np.ndarray, shift: Tuple[float, float]) -> np.ndarray:
    return ndimage.shift(img, shift=shift, order=3, mode='constant', cval=0.0, prefilter=True)


def _fft_shift_single(ref: np.ndarray, img: np.ndarray) -> Tuple[float, float]:
    """Fast integer-accurate FFT cross-correlation without GPU, for pyramid levels."""
    ref_n = ref - ref.mean()
    img_n = img - img.mean()
    h, w = ref_n.shape
    pad_h, pad_w = 2 * h, 2 * w
    F_ref = np.fft.rfft2(ref_n, s=(pad_h, pad_w))
    F_img = np.fft.rfft2(img_n, s=(pad_h, pad_w))
    corr = np.fft.irfft2(F_ref * np.conj(F_img), s=(pad_h, pad_w))
    peak_flat = int(np.argmax(corr))
    py, px = peak_flat // corr.shape[1], peak_flat % corr.shape[1]
    dy = py if py < h else py - pad_h
    dx = px if px < w else px - pad_w
    return float(dy), float(dx)


def calculate_shift_pyramid(ref: np.ndarray, img: np.ndarray,
                             levels: int = 4,
                             min_size: int = 32) -> Tuple[float, float]:
    """Coarse-to-fine multi-scale pyramid registration."""
    def _downsample(arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape
        h2, w2 = (h // 2) * 2, (w // 2) * 2  # trim to even
        patch = arr[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2)
        return patch.mean(axis=(1, 3))

    # Build pyramids
    ref_pyr = [ref.astype(np.float64)]
    img_pyr = [img.astype(np.float64)]
    for _ in range(levels - 1):
        if min(ref_pyr[-1].shape) // 2 < min_size:
            break
        ref_pyr.append(_downsample(ref_pyr[-1]))
        img_pyr.append(_downsample(img_pyr[-1]))

    actual_levels = len(ref_pyr)
    total_sy, total_sx = 0.0, 0.0

    # Loop from coarsest (N-1) to finest (0)
    for lvl in range(actual_levels - 1, -1, -1):
        r = ref_pyr[lvl]
        i = img_pyr[lvl]

        # Apply accumulated shift (scaled to this level's resolution) then compute residual
        if total_sy != 0.0 or total_sx != 0.0:
            i = ndimage.shift(i, shift=(total_sy, total_sx), order=1,
                              mode='constant', cval=0.0)

        sy_res, sx_res = _fft_shift_single(r, i)
        total_sy += sy_res
        total_sx += sx_res

        if lvl > 0:
            # Scale accumulated shift to next finer level (pixel units * 2)
            # We multiply total here so that the accumulated shift from coarser levels 
            # is properly scaled when used as a starting point for the finer level.
            total_sy *= 2.0
            total_sx *= 2.0

    _log.debug("pyramid_registration: shift=(%.2f, %.2f) over %d levels",
               total_sy, total_sx, actual_levels)
    return total_sy, total_sx


def calc_common_crop(shifts: List[Tuple[float, float]], shape: Tuple[int, int],
                     transforms=None) -> Tuple[int, int, int, int]:
    """Compute the largest axis-aligned crop valid in all aligned frames."""
    H, W = shape
    transforms = transforms or [None] * len(shifts)
    top_vals, bottom_vals, left_vals, right_vals = [], [], [], []

    for shift, transform in zip(shifts, transforms):
        sy, sx = shift
        if transform is not None:
            matrix = transform.params
            R = matrix[:2, :2]      # rotation in (col=x, row=y) space
            t_xy = matrix[:2, 2]    # [tx, ty] in (col=x, row=y)
            corners_xy = np.array([[0.0, 0.0], [W, 0.0], [0.0, H], [W, H]])
            out_xy = (R @ corners_xy.T).T + t_xy  # forward map to output
            cols_out = out_xy[:, 0]  # x = col
            rows_out = out_xy[:, 1]  # y = row
            
            top_vals.append(max(rows_out[0], rows_out[1]))
            bottom_vals.append(min(rows_out[2], rows_out[3]))
            left_vals.append(max(cols_out[0], cols_out[2]))
            right_vals.append(min(cols_out[1], cols_out[3]))
        else:
            top_vals.append(max(0.0, sy))
            bottom_vals.append(min(float(H), H + sy))
            left_vals.append(max(0.0, sx))
            right_vals.append(min(float(W), W + sx))

    top = int(np.ceil(max(top_vals))) + Config.CROP_MARGIN
    bottom = int(np.floor(min(bottom_vals))) - Config.CROP_MARGIN
    left = int(np.ceil(max(left_vals))) + Config.CROP_MARGIN
    right = int(np.floor(min(right_vals))) - Config.CROP_MARGIN
    
    if top >= bottom or left >= right:
        return 0, H, 0, W
    return top, bottom, left, right


def detect_dither(shifts: List[Tuple[float, float]], verbose: bool = False) -> dict:
    """Analyse registration shifts to detect dithering patterns."""
    if len(shifts) < 3:
        return {'is_dithered': False, 'pattern': 'aligned',
                'mean_magnitude': 0.0, 'unique_positions': len(shifts),
                'direction_spread_deg': 0.0, 'autocorrelation': 0.0}

    sy = np.array([s[0] for s in shifts])
    sx = np.array([s[1] for s in shifts])
    mags = np.sqrt(sy ** 2 + sx ** 2)

    mean_mag = float(np.mean(mags))
    int_positions = set((int(round(y)), int(round(x))) for y, x in shifts)
    unique_positions = len(int_positions)

    non_zero = mags > 0.5
    if non_zero.sum() >= 3:
        angles = np.degrees(np.arctan2(sy[non_zero], sx[non_zero]))
        sin_mean = np.mean(np.sin(np.radians(angles)))
        cos_mean = np.mean(np.cos(np.radians(angles)))
        R = np.sqrt(sin_mean ** 2 + cos_mean ** 2)
        direction_spread = float(np.degrees(np.sqrt(-2.0 * np.log(max(R, 1e-12)))))
    else:
        direction_spread = 0.0

    if len(shifts) >= 4:
        dx = np.diff(sx)
        dy = np.diff(sy)
        if len(dx) >= 2:
            autocorr_x = float(np.corrcoef(dx[:-1], dx[1:])[0, 1]) if np.std(dx) > 1e-6 else 0.0
            autocorr_y = float(np.corrcoef(dy[:-1], dy[1:])[0, 1]) if np.std(dy) > 1e-6 else 0.0
            autocorrelation = (autocorr_x + autocorr_y) / 2.0
            if not np.isfinite(autocorrelation):
                autocorrelation = 0.0
        else:
            autocorrelation = 0.0
    else:
        autocorrelation = 0.0

    all_near_zero = mean_mag < 1.0
    is_random_direction = direction_spread > 40.0
    is_low_autocorr = abs(autocorrelation) < 0.5
    has_many_positions = unique_positions >= len(shifts) * 0.5

    if all_near_zero:
        pattern = 'aligned'
        is_dithered = False
    elif is_random_direction and is_low_autocorr and has_many_positions:
        pattern = 'dithered'
        is_dithered = True
    elif not is_random_direction or not is_low_autocorr:
        pattern = 'tracking_drift'
        is_dithered = False
    else:
        pattern = 'dithered'
        is_dithered = True

    result = {
        'is_dithered': is_dithered,
        'pattern': pattern,
        'mean_magnitude': mean_mag,
        'unique_positions': unique_positions,
        'direction_spread_deg': direction_spread,
        'autocorrelation': autocorrelation,
    }

    if verbose:
        labels = {'dithered': 'Dithered (random offsets detected)',
                  'tracking_drift': 'Tracking drift (systematic trend)',
                  'aligned': 'Well-aligned (minimal offsets)'}
        safe_print(f"\n  Dither analysis:")
        safe_print(f"    Pattern: {labels.get(pattern, pattern)}")
        safe_print(f"    Mean offset: {mean_mag:.1f} px")
        safe_print(f"    Unique positions: {unique_positions}/{len(shifts)} frames")
        if direction_spread > 0:
            safe_print(f"    Direction spread: {direction_spread:.1f} deg")
        safe_print(f"    Autocorrelation: {autocorrelation:.2f}")

    return result


def run_registration_phase(
    final: List[FrameInfo],
    final_indices: List[int],
    best: FrameInfo,
    best_idx: int,
    ref_lum: np.ndarray,
    mem_lum: np.ndarray,
    cached_lums: List,
    H: int,
    W: int,
    args: argparse.Namespace,
    stats: ProcessingStats,
) -> Tuple[List, List, Dict]:
    """Compute per-frame registration shifts/transforms for all accepted frames."""
    ref_stars = best.metrics.get('_star_sources')
    if ref_stars is None and HAS_SKIMAGE_TRANSFORM and not getattr(args, 'no_affine', False):
        safe_print("  ⚠ No stars detected in reference frame — affine (rotation) "
                   "registration disabled, falling back to translation only")

    shifts = [None] * len(final)
    transforms = [None] * len(final)
    print(f"  Calculating shifts for {len(final)} frames...")

    gpu = get_gpu()

    def _register_one(j, f, orig_idx):
        if orig_idx == best_idx or args.no_registration:
            return j, (0.0, 0.0), None
        with gpu.stream_context():
            lum = (cached_lums[orig_idx] if cached_lums[orig_idx] is not None
                   else np.array(mem_lum[orig_idx]))
            use_affine = HAS_SKIMAGE_TRANSFORM and not getattr(args, 'no_affine', False)
            if use_affine:
                sy, sx = calculate_shift(ref_lum, lum, verbose=False,
                                         skip_phase_cc=args.skip_phase_correlation)
                affine_tf = match_stars_affine(ref_stars, f.metrics.get('_star_sources'),
                                               initial_shift=(sy, sx))
                if affine_tf is not None:
                    return j, (affine_tf.params[1, 2], affine_tf.params[0, 2]), affine_tf
            sy, sx = calculate_shift(
                ref_lum, lum, verbose=args.verbose,
                debug=args.debug_registration,
                frame_name=os.path.splitext(os.path.basename(f.path))[0],
                skip_phase_cc=args.skip_phase_correlation)
            if abs(sx) > 0.1 * W or abs(sy) > 0.1 * H:
                safe_print(f'Unrealistic shift {sx},{sy} for {f.path}, ignoring')
                sx, sy = 0.0, 0.0
            return j, (sy, sx), None

    if gpu.active:
        n_workers = min(gpu.max_gpu_workers(Config.GPU_FFT_WORKER_MB,
                                            Config.GPU_VRAM_RESERVE_MB), len(final))
    else:
        n_workers = min(os.cpu_count() or 4, len(final))
        
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_register_one, j, f, orig_idx): j
                   for j, (f, orig_idx) in enumerate(zip(final, final_indices))}
        for future in tqdm(as_completed(futures), total=len(final),
                           desc="  Registering", unit="frame", disable=args.verbose):
            j, shift_val, transform_val = future.result()
            shifts[j] = shift_val
            transforms[j] = transform_val
            final[j].shift = shift_val
            if args.verbose:
                f = final[j]
                if transform_val is not None:
                    tx, ty = shift_val[1], shift_val[0]
                    rot_deg = np.degrees(np.arctan2(transform_val.params[1, 0],
                                                    transform_val.params[0, 0]))
                    safe_print(f'    {os.path.basename(f.path)}: affine shift=({tx:+.1f}, '
                               f'{ty:+.1f}) px, rotation={rot_deg:+.3f} deg')
                elif shift_val != (0.0, 0.0):
                    sy, sx = shift_val
                    safe_print(f'    {os.path.basename(f.path)}: shift=({sx:+.1f}, {sy:+.1f}) px, '
                               f'magnitude={np.sqrt(sy**2 + sx**2):.2f} px')

    # Shift statistics
    shift_x = [s[1] for s in shifts]
    shift_y = [s[0] for s in shifts]
    shift_mags = [np.sqrt(sx**2 + sy**2) for sx, sy in shifts]
    if not args.no_registration:
        print(f"  Shift statistics:")
        print(f"    X: mean={np.mean(shift_x):+.1f}px, std={np.std(shift_x):.1f}px, "
              f"range=[{np.min(shift_x):+.1f}, {np.max(shift_x):+.1f}]")
        print(f"    Y: mean={np.mean(shift_y):+.1f}px, std={np.std(shift_y):.1f}px, "
              f"range=[{np.min(shift_y):+.1f}, {np.max(shift_y):+.1f}]")
        print(f"    Magnitude: mean={np.mean(shift_mags):.1f}px, max={np.max(shift_mags):.1f}px")
        if np.max(shift_mags) > Config.LARGE_SHIFT_WARNING_PX:
            warning = f"Large shifts detected (max={np.max(shift_mags):.1f}px)"
            stats.add_warning(warning)
            safe_print(f"  ⚠ {warning}")

    shift_set = set(f.shift for f in final)
    zero_shifts = sum(1 for f in final if f.shift == (0.0, 0.0))
    if len(shift_set) == 1 and len(final) > 2:
        unique_shift = list(shift_set)[0]
        if unique_shift != (0.0, 0.0):
            warning = f'All {len(final)} frames have IDENTICAL shift — registration failure!'
            stats.add_warning(warning)
            safe_print(f'\n  ⚠ WARNING: {warning}')
        elif len(final) <= 3:
            safe_print(f'\n  INFO: All frames registered with zero shift — well-aligned.')
    elif zero_shifts > len(final) * 0.8 and len(final) > 2:
        warning = f'{zero_shifts}/{len(final)} frames have zero shift'
        stats.add_warning(warning)
        safe_print(f'\n  ⚠ WARNING: {warning}')

    dither_info = detect_dither(shifts, verbose=False)
    if not args.no_registration and len(shifts) > 2:
        labels = {'dithered': 'Dithered (random offsets)',
                  'tracking_drift': 'Tracking drift (systematic trend)',
                  'aligned': 'Well-aligned (minimal offsets)'}
        print(f"\n  Dither analysis:")
        print(f"    Pattern: {labels.get(dither_info['pattern'], dither_info['pattern'])}")
        print(f"    Mean shift: {dither_info['mean_magnitude']:.1f} px")
        print(f"    Unique positions: {dither_info['unique_positions']}/{len(shifts)} frames")
        if dither_info.get('direction_spread_deg', 0) > 0:
            print(f"    Direction spread: {dither_info['direction_spread_deg']:.1f} deg")
        if dither_info['is_dithered'] and args.stack_method == 'mean':
            safe_print(f"    Warning: mean stacking does not reject cosmic rays; "
                       f"consider --stack-method sigma_clip")
        if dither_info['is_dithered'] and getattr(args, 'drizzle_scale', 1.0) <= 1.0:
            safe_print(f"    Tip: dithered data detected — add --drizzle-scale 2.0 "
                       f"to enable sub-pixel super-resolution stacking")

    return shifts, transforms, dither_info
