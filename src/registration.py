"""Image registration: shift calculation, affine transform, cropping, dither detection."""
from __future__ import annotations

import os
import warnings
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

from src.gpu_context import get_gpu
from src.models import Config
from src.utils import safe_print

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

    # Shift img points by initial estimate for better matching
    img_pts_shifted = img_pts + np.array([initial_shift[1], initial_shift[0]])

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
    With accurate registration (full-resolution FFT cross-correlation),
    subpixel shifts preserve detail better than integer rounding.
    Dispatches to GPU (cupyx.scipy.ndimage) when available.
    """
    gpu = get_gpu()
    xp = gpu.xp
    _ndimage = gpu.xndimage
    img_d = gpu.to_device(img)
    if transform is not None:
        matrix = transform.params
        R = matrix[:2, :2]        # EuclideanTransform stores t in (x=col, y=row) space.
        # scipy.ndimage.affine_transform operates in (row, col) space and applies the
        # inverse mapping: input_coord = M @ output_coord + offset.
        # Correct offset: -R @ [ty, tx]  (swap x/y to row/col, then negate for inverse).
        t_xy = matrix[:2, 2]      # [tx, ty] in (col, row)=(x, y)
        t_rowcol = np.array([t_xy[1], t_xy[0]])  # swap to (row, col)
        offset = -R @ t_rowcol
        mat_d = xp.asarray(R) if gpu.active else R
        off_d = xp.asarray(offset) if gpu.active else offset
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
    return img


def calculate_shift(ref: np.ndarray, img: np.ndarray, upsample: int = 10, verbose: bool = False, debug: bool = False, frame_name: str = "", skip_phase_cc: bool = False) -> Tuple[float, float]:
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

    # Use phase cross correlation for subpixel shifts when available
    if not skip_phase_cc and phase_cross_correlation is not None:
        try:
            # Normalize images for phase correlation (zero mean, unit variance)
            # This is critical for good phase correlation performance
            ref_norm = (ref - np.mean(ref)) / (np.std(ref) + 1e-12)
            img_norm = (img - np.mean(img)) / (np.std(img) + 1e-12)

            # Suppress overflow warnings in phase correlation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                shift, error, diffphase = phase_cross_correlation(ref_norm, img_norm, upsample_factor=upsample)
            debug_info.append(f"phase_cc: shift={shift}, error={error:.4f}")

            # Validate shift magnitude (sanity check)
            if np.isfinite(shift).all() and np.abs(shift).max() < max(ref.shape) * 0.5:
                # Check error threshold - phase correlation must have low error to be trusted
                # error < 0.01 is excellent, error < 0.1 is acceptable, error >= 0.5 is failure
                if error < 0.1:
                    if verbose:
                        if np.allclose(shift, 0.0):
                            print(f"      [phase_correlation: zero shift (error={error:.4f}, images well-aligned)]")
                        else:
                            print(f"      [phase_correlation succeeded: shift={shift}, error={error:.4f}]")
                    return float(shift[0]), float(shift[1])
                else:
                    debug_info.append(f"phase_cc rejected: high error ({error:.4f} >= 0.1)")
            else:
                debug_info.append(f"phase_cc rejected: nan/inf or too large ({np.abs(shift).max():.1f} > {max(ref.shape) * 0.5:.1f})")
        except Exception as e:
            debug_info.append(f"phase_cc error: {type(e).__name__}")

    # FFT cross-correlation at full resolution with zero-padding.
    # No downscaling or windowing — these cause multi-pixel registration
    # errors that broaden stars in the stacked result.
    # Uses GPU (CuPy) when available for ~10x speedup on large images.
    try:
        gpu = get_gpu()
        xp = gpu.xp

        ref_norm = xp.asarray((ref - np.mean(ref)), dtype=xp.float64)
        img_norm = xp.asarray((img - np.mean(img)), dtype=xp.float64)

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

        shift_y = float(dy + sub_y)
        shift_x = float(dx + sub_x)

        if np.isfinite(shift_y) and np.isfinite(shift_x) and max(abs(shift_y), abs(shift_x)) < max(h, w) * 0.5:
            if verbose:
                print(f"      [fft_xcorr: shift=({shift_x:.1f}, {shift_y:.1f})]")
            return shift_y, shift_x
        else:
            debug_info.append(f"fft_xcorr rejected: shift ({shift_y:.1f}, {shift_x:.1f}) too large")
    except Exception as e:
        debug_info.append(f"fft_xcorr error: {type(e).__name__}")

    # Fallback to centroid difference - try multiple percentiles for robustness
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
                    # Shift needed to align img with ref: ref_center - img_center
                    # If img is down by 5px, img_center.y = ref_center.y + 5
                    # We need to shift img UP by -5, which is ref_center.y - img_center.y
                    shift_y = float(cim[0] - cim2[0])
                    shift_x = float(cim[1] - cim2[1])
                    shift_mag = np.sqrt(shift_x**2 + shift_y**2)

                    # Prefer solution from lower percentile (more pixels = more stable)
                    # But if shift is too large, use lower percentile threshold
                    if shift_mag > 0.1 or n_ref > 50:  # Have a real shift or many pixels
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


def calc_common_crop(shifts: List[Tuple[float, float]], shape: Tuple[int, int],
                     transforms=None) -> Tuple[int, int, int, int]:
    """Compute the largest axis-aligned crop valid in all aligned frames.

    For translation-only frames uses shift magnitudes.  For frames with a
    rotation transform the four corners of the input are forward-mapped to
    output space; the inner axis-aligned rectangle of the resulting rotated
    quad is used as the per-frame valid region.
    """
    H, W = shape
    transforms = transforms or [None] * len(shifts)
    top_vals, bottom_vals, left_vals, right_vals = [], [], [], []

    for shift, transform in zip(shifts, transforms):
        sy, sx = shift
        if transform is not None:
            matrix = transform.params
            R = matrix[:2, :2]      # rotation in (col=x, row=y) space
            t_xy = matrix[:2, 2]    # [tx, ty] in (col=x, row=y)
            # 4 corners of input in (col=x, row=y): TL, TR, BL, BR
            corners_xy = np.array([[0.0, 0.0], [W, 0.0], [0.0, H], [W, H]])
            out_xy = (R @ corners_xy.T).T + t_xy  # forward map to output
            cols_out = out_xy[:, 0]  # x = col
            rows_out = out_xy[:, 1]  # y = row
            # Inner axis-aligned rect that fits inside the rotated quad
            # (valid for |rotation| < 45 deg, which always holds for field rotation)
            top_vals.append(max(rows_out[0], rows_out[1]))   # max row of top edge
            bottom_vals.append(min(rows_out[2], rows_out[3]))  # min row of bottom edge
            left_vals.append(max(cols_out[0], cols_out[2]))  # max col of left edge
            right_vals.append(min(cols_out[1], cols_out[3]))  # min col of right edge
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
    """Analyse registration shifts to detect dithering patterns.

    Returns a dict with:
      - ``is_dithered`` (bool)
      - ``pattern`` (str): 'dithered', 'tracking_drift', or 'aligned'
      - ``mean_magnitude`` (float): mean shift magnitude in pixels
      - ``unique_positions`` (int): approx distinct integer pixel positions
      - ``direction_spread_deg`` (float): std of shift angles in degrees
      - ``autocorrelation`` (float): correlation between consecutive shift vectors
    """
    if len(shifts) < 3:
        return {'is_dithered': False, 'pattern': 'aligned',
                'mean_magnitude': 0.0, 'unique_positions': len(shifts),
                'direction_spread_deg': 0.0, 'autocorrelation': 0.0}

    sy = np.array([s[0] for s in shifts])
    sx = np.array([s[1] for s in shifts])
    mags = np.sqrt(sy ** 2 + sx ** 2)

    mean_mag = float(np.mean(mags))

    # Count approximately unique integer positions
    int_positions = set((int(round(y)), int(round(x))) for y, x in shifts)
    unique_positions = len(int_positions)

    # Direction spread — std of angles (in degrees)
    non_zero = mags > 0.5  # ignore near-zero shifts
    if non_zero.sum() >= 3:
        angles = np.degrees(np.arctan2(sy[non_zero], sx[non_zero]))
        # Circular std: use the Mardia definition
        sin_mean = np.mean(np.sin(np.radians(angles)))
        cos_mean = np.mean(np.cos(np.radians(angles)))
        R = np.sqrt(sin_mean ** 2 + cos_mean ** 2)
        direction_spread = float(np.degrees(np.sqrt(-2.0 * np.log(max(R, 1e-12)))))
    else:
        direction_spread = 0.0

    # Autocorrelation of consecutive shift vectors (low = random / dithered)
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

    # Classification heuristics
    all_near_zero = mean_mag < 1.0
    is_random_direction = direction_spread > 40.0  # degrees
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
