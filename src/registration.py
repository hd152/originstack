"""Image registration: shift calculation, affine transform, cropping, dither detection."""
from __future__ import annotations

import argparse
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.fft as sfft
from scipy import ndimage

from src.gpu_context import GpuContext, get_gpu
from src.models import Config, FrameInfo, ProcessingStats
from src.utils import safe_print, get_logger

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

_log = get_logger()

# Optional native (Rust) Lanczos-3 warp — CPU only; GPU path is unaffected.
try:
    import astro_native as _native
    HAS_NATIVE = True
except Exception:
    _native = None
    HAS_NATIVE = False

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

try:
    import astroalign as _astroalign
    HAS_ASTROALIGN = True
except Exception:
    _astroalign = None  # type: ignore[assignment]
    HAS_ASTROALIGN = False


def match_stars_affine(ref_positions: Optional[Any], img_positions: Optional[Any],
                       initial_shift: Tuple[float, float] = (0.0, 0.0)) -> Optional[Any]:
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
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='No inliers found',
                                    category=UserWarning)
            model, inliers = ransac(
                (src, dst), EuclideanTransform,
                min_samples=3, residual_threshold=2.0, max_trials=1000
            )
        if inliers is not None and inliers.sum() >= 3:
            return model
    except Exception:
        pass
    return None


def _astroalign_transform(ref_lum: np.ndarray,
                          img_lum: np.ndarray) -> Optional[Any]:
    """Use astroalign triangle-pattern matching to find a Euclidean transform.

    Called as a fallback when star-catalog RANSAC matching fails (e.g. too few
    detected stars, large rotation, or significant scale mismatch between panels).
    Returns a skimage EuclideanTransform compatible with apply_transform, or None.
    """
    if not HAS_ASTROALIGN or not HAS_SKIMAGE_TRANSFORM:
        return None
    try:
        transform, _ = _astroalign.find_transform(
            img_lum.astype(np.float32),
            ref_lum.astype(np.float32),
        )
        return EuclideanTransform(
            rotation=transform.rotation,
            translation=(transform.translation[0], transform.translation[1]),
        )
    except Exception:
        return None


def apply_transform(img: np.ndarray, shift: Optional[Tuple[float, float]] = None,
                    transform: Optional[Any] = None,
                    local_field: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply translation or affine transform to a multi-channel image.

    Uses cubic spline interpolation (order=3) for subpixel accuracy.
    Dispatches to GPU (cupyx.scipy.ndimage) when available.

    ``local_field`` (optional): a coarse (Gc, Gc, 2) elastic displacement
    field from ``fit_displacement_field``, composed into the SAME resample
    pass as the affine/shift warp (source coordinate = affine_src(o) -
    R @ D(o), sampled via ``map_coordinates``) rather than a second warp —
    two independent interpolation passes would compound blur/ringing.
    Bypasses the native Rust Lanczos-3 fast path (matrix+offset only, no
    per-pixel coordinate grid); falls through to scipy/cupyx cubic
    ``map_coordinates`` instead, only for frames actually using it.
    """
    # Ensure 3D shape for consistent channel processing (H, W, 1) for grayscale
    original_ndim = img.ndim
    if original_ndim == 2:
        img = img[:, :, np.newaxis]
        squeeze_back = True
    else:
        squeeze_back = False

    gpu = get_gpu()

    def _run(xp, _ndimage, img_arr):
        if local_field is not None:
            H_i, W_i = img_arr.shape[:2]
            oy, ox = np.mgrid[0:H_i, 0:W_i].astype(np.float64)
            if transform is not None:
                matrix = transform.params
                R = matrix[:2, :2]
                t_xy = matrix[:2, 2]
                t_rowcol = np.array([t_xy[1], t_xy[0]])
                offset = -R @ t_rowcol
                base_y = R[0, 0] * oy + R[0, 1] * ox + offset[0]
                base_x = R[1, 0] * oy + R[1, 1] * ox + offset[1]
            elif shift is not None:
                R = np.eye(2)
                base_y = oy - shift[0]
                base_x = ox - shift[1]
            else:
                R = np.eye(2)
                base_y, base_x = oy, ox
            dy, dx = sample_displacement_field(local_field, H_i, W_i, oy, ox)
            src_y = base_y - (R[0, 0] * dy + R[0, 1] * dx)
            src_x = base_x - (R[1, 0] * dy + R[1, 1] * dx)
            coords = xp.asarray(np.stack([src_y, src_x], axis=0))
            result = xp.empty_like(img_arr)
            for c in range(img_arr.shape[2]):
                result[:, :, c] = _ndimage.map_coordinates(
                    img_arr[:, :, c], coords, order=3, mode='constant', cval=0.0
                )
            return result
        if transform is not None:
            matrix = transform.params
            R = matrix[:2, :2]
            t_xy = matrix[:2, 2]
            t_rowcol = np.array([t_xy[1], t_xy[0]])
            offset = -R @ t_rowcol
            mat_d = xp.asarray(R)
            off_d = xp.asarray(offset)
            result = xp.empty_like(img_arr)
            for c in range(img_arr.shape[2]):
                result[:, :, c] = _ndimage.affine_transform(
                    img_arr[:, :, c], mat_d, offset=off_d,
                    order=3, mode='constant', cval=0.0
                )
            return result
        elif shift is not None:
            result = xp.empty_like(img_arr)
            for c in range(img_arr.shape[2]):
                result[:, :, c] = _ndimage.shift(
                    img_arr[:, :, c], shift=shift, order=3,
                    mode='constant', cval=0.0, prefilter=True
                )
            return result
        return img_arr

    if transform is None and shift is None and local_field is None:
        if squeeze_back:
            img = img[:, :, 0]
        return img

    if gpu.active:
        try:
            result = _run(gpu.xp, gpu.xndimage, gpu.to_device(img))
            out = gpu.to_host(result)
            return out[:, :, 0] if squeeze_back else out
        except Exception as exc:
            if gpu.is_oom(exc):
                gpu.free_pool()
            else:
                raise

    # CPU path (also OOM fallback). Prefer the native Lanczos-3 warp: it holds
    # star FWHM and flux identical to scipy order-3, its mild per-frame ringing
    # averages out across dithered frames (validated), and it is multithreaded.
    img_cpu = np.asarray(img)
    if (local_field is None and HAS_NATIVE
            and img_cpu.dtype == np.float32 and img_cpu.flags['C_CONTIGUOUS']):
        try:
            H_i, W_i = img_cpu.shape[:2]
            if transform is not None:
                matrix = transform.params
                R = matrix[:2, :2]
                t_xy = matrix[:2, 2]
                t_rowcol = np.array([t_xy[1], t_xy[0]])
                mat = R
                off = -R @ t_rowcol
            else:  # pure translation: out[o] = in[o - shift]
                mat = np.eye(2)
                off = np.array([-shift[0], -shift[1]], dtype=np.float64)
            out = _native.warp_affine_lanczos3(
                img_cpu, mat.astype(np.float64).ravel().tolist(),
                off.astype(np.float64).tolist(), int(H_i), int(W_i), 0.0)
            return out[:, :, 0] if squeeze_back else out
        except Exception as exc:
            _log.debug("native warp failed (%s); using scipy", exc)

    from scipy import ndimage as _scipy_ndimage
    result = _run(np, _scipy_ndimage, img_cpu)
    out = np.asarray(result)
    return out[:, :, 0] if squeeze_back else out


def calculate_shift(ref: np.ndarray, img: np.ndarray, upsample: int = 10, verbose: bool = False, debug: bool = False, frame_name: str = "", skip_phase_cc: bool = False, use_pyramid: bool = True, seed_shift: Optional[Tuple[float, float]] = None, masked_correlation: bool = False, corr_downsample: int = 1) -> Tuple[float, float]:
    """Calculate the (shift_y, shift_x) needed to align ``img`` to ``ref``.

    Registration cascade:
      1. Multi-scale pyramid (coarse-to-fine) — handles large shifts reliably
      2. Phase cross-correlation (skimage) — sub-pixel accurate when error < 0.1.
      3. FFT cross-correlation at full resolution — fallback for phase-cc failure.
      4. Centroid difference — last resort for featureless or very noisy frames.

    ``seed_shift`` supplies a pre-computed coarse shift (e.g. the pyramid shift
    already calculated during reference-frame selection).  When given it is used
    in place of running the pyramid pass again, so the expensive coarse-to-fine
    registration runs only once per frame across the whole pipeline.
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
    # A caller-supplied seed (from the reference-selection pyramid pass) lets us
    # skip recomputing the pyramid — it is the same coarse-to-fine result.
    pyramid_shift = (0.0, 0.0)
    if seed_shift is not None:
        psy, psx = seed_shift
        if np.isfinite(psy) and np.isfinite(psx):
            pyramid_shift = (float(psy), float(psx))
            debug_info.append(f"pyramid (seeded): shift=({psx:.1f}, {psy:.1f})")
    elif use_pyramid:
        try:
            psy, psx = calculate_shift_pyramid(ref, img)
            if np.isfinite(psy) and np.isfinite(psx):
                pyramid_shift = (psy, psx)
                debug_info.append(f"pyramid: shift=({psx:.1f}, {psy:.1f})")
        except Exception as exc:
            debug_info.append(f"pyramid error: {type(exc).__name__}")

    # Pre-apply pyramid shift once; reused by both phase_cc and FFT paths
    psy, psx = pyramid_shift
    if psy != 0.0 or psx != 0.0:
        img_shifted = ndimage.shift(img, shift=(psy, psx), order=1,
                                    mode='constant', cval=0.0)
    else:
        img_shifted = img

    # Masked cross-correlation: suppress bright nebula emission in background
    if masked_correlation:
        try:
            ref_lum_m = ref.copy().astype(np.float32)
            img_lum_m = img_shifted.copy().astype(np.float32)
            bg_ref = float(np.median(ref_lum_m))
            fill_ref = float(np.mean(ref_lum_m > 2 * bg_ref))
            if fill_ref > 0.3:
                from scipy.ndimage import binary_dilation
                thresh_ref = float(np.percentile(ref_lum_m, 85))
                thresh_img = float(np.percentile(img_lum_m, 85))
                struct = np.ones((21, 21), dtype=bool)
                mask_ref = binary_dilation(ref_lum_m > thresh_ref, structure=struct)
                mask_img = binary_dilation(img_lum_m > thresh_img, structure=struct)
                ref_lum_m[mask_ref] = 0.0
                img_lum_m[mask_img] = 0.0
                ref = ref_lum_m
                img_shifted = img_lum_m
        except Exception:
            pass

    # Use phase cross correlation for subpixel shifts when available
    if not skip_phase_cc and phase_cross_correlation is not None:
        try:
            img_pre = img_shifted

            ref_std = float(np.std(ref))
            img_std = float(np.std(img_pre))
            # Skip phase_cc when either image has near-zero variance — normalization
            # produces huge values that confuse the correlation.
            if ref_std < 1.0 or img_std < 1.0:
                debug_info.append("phase_cc skipped: near-zero image variance")
                raise StopIteration  # caught by outer except, falls through to FFT

            ref_norm = (ref - np.mean(ref)) / ref_std
            img_norm = (img_pre - np.mean(img_pre)) / img_std

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("ignore", UserWarning)
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

        img_fft = img_shifted  # reuse pyramid-shifted image computed above

        h, w = ref.shape
        if psy != 0.0 or psx != 0.0:
            # The pre-shift fills vacated edges with zeros. A whole-image mean
            # includes those zeros, making the padded border strongly negative
            # after subtraction. The FFT treats this as a large feature and
            # produces a spurious correlation peak that fails the validity check,
            # causing unnecessary fallback to centroid.
            # Fix: compute means only over the valid overlap region; zero-fill
            # outside so the FFT only operates on real pixel data.
            y0 = max(0, int(round(psy)))
            y1 = h + min(0, int(round(psy)))
            x0 = max(0, int(round(psx)))
            x1 = w + min(0, int(round(psx)))
            # float32 (complex64 FFT) halves the cost/memory of the zero-padded
            # 2H×2W transform; only the correlation-peak location is used.
            if y1 > y0 and x1 > x0:
                ref_region = ref[y0:y1, x0:x1].astype(np.float32)
                img_region = img_fft[y0:y1, x0:x1].astype(np.float32)
                ref_np = np.zeros((h, w), dtype=np.float32)
                img_np = np.zeros((h, w), dtype=np.float32)
                ref_np[y0:y1, x0:x1] = ref_region - ref_region.mean()
                img_np[y0:y1, x0:x1] = img_region - img_region.mean()
            else:
                ref_np = (ref - np.mean(ref)).astype(np.float32)
                img_np = (img_fft - np.mean(img_fft)).astype(np.float32)
        else:
            ref_np = (ref - np.mean(ref)).astype(np.float32)
            img_np = (img_fft - np.mean(img_fft)).astype(np.float32)

        # Optional block-average downsample before correlating: the residual
        # being solved for here is small (the pyramid/seed already removed
        # the bulk of the shift), so a coarser grid still resolves it — and
        # halving the side length quarters the FFT's O(N^2 log N) cost per
        # factor of 2. The recovered (integer + parabolic) offset is in
        # downsampled-pixel units and is scaled back up by the factor.
        scale = 1
        if corr_downsample > 1:
            r_ds, i_ds = ref_np, img_np
            n = corr_downsample
            while n > 1 and min(r_ds.shape) >= 64:
                r_ds = _downsample_half(r_ds)
                i_ds = _downsample_half(i_ds)
                scale *= 2
                n //= 2
            ref_np, img_np = r_ds, i_ds

        h_c, w_c = ref_np.shape
        if gpu.active:
            pad_h, pad_w = 2 * h_c, 2 * w_c
            ref_norm = xp.asarray(ref_np)
            img_norm = xp.asarray(img_np)
            F_ref = xp.fft.rfft2(ref_norm, s=(pad_h, pad_w))
            F_img = xp.fft.rfft2(img_norm, s=(pad_h, pad_w))
            corr = xp.fft.irfft2(F_ref * xp.conj(F_img), s=(pad_h, pad_w))
            del F_ref, F_img, ref_norm, img_norm  # free VRAM immediately
        else:
            # scipy's vendored pocketfft is ~2.7x faster than numpy's at these
            # sizes (measured on this codebase's actual padded shapes) despite
            # being the same algorithm family -- separately-built/optimized
            # copy. workers=1: this call already runs inside a per-frame
            # ThreadPoolExecutor, so internal multi-threading would just
            # oversubscribe the same cores the outer pool is using.
            # Padding to next_fast_len (a highly-composite size >= 2x, never
            # smaller) instead of exactly 2x avoids landing on a dimension
            # with a large prime factor (e.g. 2*3056=6112=2^5*191 forces
            # Bluestein's algorithm); ~2x faster on top of the numpy->scipy
            # switch, same correlation-peak result since it only adds extra
            # zero margin.
            pad_h, pad_w = sfft.next_fast_len(2 * h_c), sfft.next_fast_len(2 * w_c)
            F_ref = sfft.rfft2(ref_np, s=(pad_h, pad_w), workers=1)
            F_img = sfft.rfft2(img_np, s=(pad_h, pad_w), workers=1)
            corr = sfft.irfft2(F_ref * np.conj(F_img), s=(pad_h, pad_w), workers=1)
            del F_ref, F_img

        peak_flat = int(xp.argmax(corr))
        peak = (peak_flat // corr.shape[1], peak_flat % corr.shape[1])
        dy = peak[0] if peak[0] < h_c else peak[0] - pad_h
        dx = peak[1] if peak[1] < w_c else peak[1] - pad_w

        # Parabolic subpixel refinement (wrap-safe)
        py, px = peak
        sub_y = sub_x = 0.0
        vc = float(corr[py, px])
        vm = float(corr[(py - 1) % pad_h, px])
        vp = float(corr[(py + 1) % pad_h, px])
        denom = 2.0 * (2.0 * vc - vm - vp)
        if abs(denom) > 1e-12:
            sub_y = max(-0.5, min(0.5, (vp - vm) / denom))
        vm = float(corr[py, (px - 1) % pad_w])
        vp = float(corr[py, (px + 1) % pad_w])
        denom = 2.0 * (2.0 * vc - vm - vp)
        if abs(denom) > 1e-12:
            sub_x = max(-0.5, min(0.5, (vp - vm) / denom))
        del corr  # free VRAM

        shift_y = float((dy + sub_y) * scale) + psy
        shift_x = float((dx + sub_x) * scale) + psx

        if np.isfinite(shift_y) and np.isfinite(shift_x) and max(abs(shift_y), abs(shift_x)) < max(h, w) * 0.5:
            if verbose:
                print(f"      [fft_xcorr: shift=({shift_x:.1f}, {shift_y:.1f})]")
            return shift_y, shift_x
        else:
            debug_info.append(f"fft_xcorr rejected: shift ({shift_y:.1f}, {shift_x:.1f}) too large")
    except Exception as e:
        debug_info.append(f"fft_xcorr error: {type(e).__name__}")

    # Fallback to centroid difference — compute all percentile thresholds in one pass
    best_shift = (0.0, 0.0)
    best_score = float('inf')

    _pcts = list(Config.CENTROID_PERCENTILES)
    _ref_thresholds = np.percentile(ref, _pcts)
    _img_thresholds = np.percentile(img, _pcts)

    for percentile, thresh_ref, thresh_img in zip(_pcts, _ref_thresholds, _img_thresholds):
        try:
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
    # float32 transforms (complex64) halve the FFT cost/memory vs float64; the
    # result is only used for integer-peak detection, which is unaffected.
    ref_n = (ref - ref.mean()).astype(np.float32, copy=False)
    img_n = (img - img.mean()).astype(np.float32, copy=False)
    h, w = ref_n.shape
    pad_h, pad_w = sfft.next_fast_len(2 * h), sfft.next_fast_len(2 * w)
    F_ref = sfft.rfft2(ref_n, s=(pad_h, pad_w), workers=1)
    F_img = sfft.rfft2(img_n, s=(pad_h, pad_w), workers=1)
    corr = sfft.irfft2(F_ref * np.conj(F_img), s=(pad_h, pad_w), workers=1)
    peak_flat = int(np.argmax(corr))
    py, px = peak_flat // corr.shape[1], peak_flat % corr.shape[1]
    dy = py if py < h else py - pad_h
    dx = px if px < w else px - pad_w
    return float(dy), float(dx)


def _downsample_half(arr: np.ndarray) -> np.ndarray:
    """2x box-downsample (even-cropped 2x2 average)."""
    h2 = (arr.shape[0] // 2) * 2
    w2 = (arr.shape[1] // 2) * 2
    a = arr[:h2, :w2]
    return (a[::2, ::2] + a[1::2, ::2] + a[::2, 1::2] + a[1::2, 1::2]) * 0.25


def _int_shift(arr: np.ndarray, sy: int, sx: int) -> np.ndarray:
    """Integer translation with zero fill — pyramid totals are always whole
    pixels, so bilinear ndimage.shift is pure overhead here."""
    out = np.zeros_like(arr)
    h, w = arr.shape
    ys0, ys1 = max(0, sy), min(h, h + sy)
    xs0, xs1 = max(0, sx), min(w, w + sx)
    if ys1 > ys0 and xs1 > xs0:
        out[ys0:ys1, xs0:xs1] = arr[ys0 - sy:ys1 - sy, xs0 - sx:xs1 - sx]
    return out


# The pyramid's job is a coarse integer seed: calculate_shift's residual FFT
# correlation re-solves the shift on top of it and tolerates seed errors of
# several pixels (measured: 0.9px seed offset -> 0.013px final). Correlating
# the finest (full-res) level therefore buys nothing the residual pass does
# not redo -- and it is 4x the work of the half-res level. Stop there.
_PYRAMID_STOP_LEVEL = 1


def prepare_ref_pyramid(ref: np.ndarray, levels: int = 4, min_size: int = 32) -> list:
    """Precompute a fixed reference's pyramid FFTs once, for reuse across many
    frames. Returns a coarsest-to-finest... no — finest-to-coarsest list matching
    ``calculate_shift_pyramid`` indexing: entry lvl = (F_ref, h, w, pad_h, pad_w).

    When one reference is registered against N frames (reference selection, and
    the whole registration phase), this removes N-1 redundant reference-pyramid
    builds and N-1 redundant reference FFTs per level. Levels below the
    pyramid stop level hold None (never correlated — see _PYRAMID_STOP_LEVEL);
    f32 throughout, the correlation peak is integer-precision anyway.
    """
    ref_pyr = [ref.astype(np.float32)]
    for _ in range(levels - 1):
        if min(ref_pyr[-1].shape) // 2 < min_size:
            break
        ref_pyr.append(_downsample_half(ref_pyr[-1]))
    stop = min(_PYRAMID_STOP_LEVEL, len(ref_pyr) - 1)
    prepared = []
    for lvl, r in enumerate(ref_pyr):
        if lvl < stop:
            prepared.append(None)
            continue
        ref_n = (r - r.mean()).astype(np.float32, copy=False)
        h, w = ref_n.shape
        pad_h, pad_w = sfft.next_fast_len(2 * h), sfft.next_fast_len(2 * w)
        F_ref = sfft.rfft2(ref_n, s=(pad_h, pad_w), workers=1)
        prepared.append((F_ref, h, w, pad_h, pad_w))
    return prepared


def calculate_shift_pyramid_pref(prepared: list, img: np.ndarray) -> Tuple[float, float]:
    """Pyramid registration using a precomputed reference pyramid (see
    ``prepare_ref_pyramid``). Matches ``calculate_shift_pyramid`` — only
    the reference-side work is hoisted out of the per-frame loop."""
    levels = len(prepared)
    stop = next((i for i, p in enumerate(prepared) if p is not None), 0)
    img_pyr: List[Optional[np.ndarray]] = [np.asarray(img, dtype=np.float32)]
    for lvl in range(1, levels):
        img_pyr.append(_downsample_half(img_pyr[-1]))
    total_sy, total_sx = 0.0, 0.0
    for lvl in range(levels - 1, stop - 1, -1):
        F_ref, h, w, pad_h, pad_w = prepared[lvl]
        i = img_pyr[lvl]
        if total_sy != 0.0 or total_sx != 0.0:
            i = _int_shift(i, int(total_sy), int(total_sx))
        img_n = (i - i.mean()).astype(np.float32, copy=False)
        F_img = sfft.rfft2(img_n, s=(pad_h, pad_w), workers=1)
        corr = sfft.irfft2(F_ref * np.conj(F_img), s=(pad_h, pad_w), workers=1)
        peak_flat = int(np.argmax(corr))
        py, px = peak_flat // corr.shape[1], peak_flat % corr.shape[1]
        dy = py if py < h else py - pad_h
        dx = px if px < w else px - pad_w
        total_sy += dy
        total_sx += dx
        if lvl > 0:
            total_sy *= 2.0
            total_sx *= 2.0
    return total_sy, total_sx


def calculate_shift_pyramid(ref: np.ndarray, img: np.ndarray,
                             levels: int = 4,
                             min_size: int = 32) -> Tuple[float, float]:
    """Coarse-to-fine multi-scale pyramid registration."""
    def _downsample(arr: np.ndarray) -> np.ndarray:
        h2 = (arr.shape[0] // 2) * 2
        w2 = (arr.shape[1] // 2) * 2
        a = arr[:h2, :w2]
        return (a[::2, ::2] + a[1::2, ::2] + a[::2, 1::2] + a[1::2, 1::2]) * 0.25

    # Build pyramids (f32: the correlation peak is integer-precision anyway)
    ref_pyr = [np.asarray(ref, dtype=np.float32)]
    img_pyr = [np.asarray(img, dtype=np.float32)]
    for _ in range(levels - 1):
        if min(ref_pyr[-1].shape) // 2 < min_size:
            break
        ref_pyr.append(_downsample(ref_pyr[-1]))
        img_pyr.append(_downsample(img_pyr[-1]))

    actual_levels = len(ref_pyr)
    total_sy, total_sx = 0.0, 0.0

    # Loop from coarsest (N-1) down to the stop level (see _PYRAMID_STOP_LEVEL:
    # the finest level only re-derives what the caller's residual correlation
    # solves again anyway, at 4x the cost of the half-res level).
    stop = min(_PYRAMID_STOP_LEVEL, actual_levels - 1)
    for lvl in range(actual_levels - 1, stop - 1, -1):
        r = ref_pyr[lvl]
        i = img_pyr[lvl]

        # Apply accumulated (always-integer) shift, then compute the residual
        if total_sy != 0.0 or total_sx != 0.0:
            i = _int_shift(i, int(total_sy), int(total_sx))

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
                     transforms: Optional[List[Optional[Any]]] = None,
                     extra_margin_px: float = 0.0) -> Tuple[int, int, int, int]:
    """Compute the largest axis-aligned crop valid in all aligned frames.

    ``extra_margin_px``: additional trim on all four sides, on top of
    ``Config.CROP_MARGIN``. Used by elastic registration to keep the crop
    safely inside the region every frame's local displacement field could
    have pulled pixels from — the corner-based rectangle otherwise has no way
    to know about a spatially-varying (non-rigid) warp.
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

    margin = Config.CROP_MARGIN + int(np.ceil(extra_margin_px))
    top = int(np.ceil(max(top_vals))) + margin
    bottom = int(np.floor(min(bottom_vals))) - margin
    left = int(np.ceil(max(left_vals))) + margin
    right = int(np.floor(min(right_vals))) - margin
    
    if top >= bottom or left >= right:
        return 0, H, 0, W
    return top, bottom, left, right


def detect_dither(shifts: List[Tuple[float, float]], verbose: bool = False) -> Dict[str, Any]:
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


def _geometric_median_2d(points: np.ndarray, max_iters: int = 50,
                          tol: float = 1e-5) -> Tuple[float, float]:
    """Weiszfeld algorithm for the L1 geometric median of 2D shift vectors.

    More outlier-robust than the arithmetic mean; used to locate the "central"
    shift position when selecting an alignment-optimal reference frame.
    """
    if len(points) == 0:
        return 0.0, 0.0
    p = points.astype(np.float64)
    med = p.mean(axis=0)
    for _ in range(max_iters):
        dists = np.linalg.norm(p - med, axis=1)
        dists = np.maximum(dists, 1e-10)
        weights = 1.0 / dists
        new_med = (weights[:, None] * p).sum(axis=0) / weights.sum()
        if np.linalg.norm(new_med - med) < tol:
            break
        med = new_med
    return float(med[0]), float(med[1])


def select_reference_frame(
    final: List[FrameInfo],
    final_indices: List[int],
    mem_lum: np.ndarray,
    H: int,
    W: int,
    centrality_weight: float = None,
    n_workers: int = None,
    cached_lums: Optional[List[Optional[np.ndarray]]] = None,
) -> Tuple[FrameInfo, int, List[Tuple[float, float]]]:
    """Choose reference frame by blending quality score with alignment centrality.

    A purely quality-ranked reference may sit at the edge of the shift
    distribution, forcing all other frames to be shifted further (accumulating
    interpolation error).  This function does a cheap pyramid-only registration
    pass, finds the geometric median of all shifts, and re-scores each frame as:

        combined = quality_norm * (1 - w) + centrality_norm * w

    where w = centrality_weight (default Config.ALIGNMENT_CENTRALITY_WEIGHT).

    Returns:
        best_frame       – FrameInfo of the chosen reference.
        best_lights_idx  – index into the *lights* list (not final).
        pyramid_shifts   – cheap shifts computed during this pass (recycled by
                           the outlier filter so the pyramid pass runs only once).
    """
    from src.models import Config
    if centrality_weight is None:
        centrality_weight = Config.ALIGNMENT_CENTRALITY_WEIGHT
    if n_workers is None:
        # Uncapped (was hard-limited to Config.REF_PYRAMID_WORKERS=4) — same
        # class of bug as the old Phase-1 8-worker cap: this pass is threaded
        # (ThreadPoolExecutor, GIL-releasing FFT/scipy ops), so it scales with
        # real cores. On a real 233-frame/16-core run this pass was consuming
        # an unaccounted ~3min inside the Registration phase at only 4 workers.
        n_workers = min(os.cpu_count() or 4, len(final))

    def _get_lum(orig_idx: int) -> np.ndarray:
        if cached_lums is not None and orig_idx < len(cached_lums) and cached_lums[orig_idx] is not None:
            return np.asarray(cached_lums[orig_idx], dtype=np.float32)
        return np.array(mem_lum[orig_idx], dtype=np.float32)

    # Tentative reference: highest quality frame (for consistent pyramid base)
    tentative_best = max(final, key=lambda f: f.metrics.get('score', 0.0))
    tentative_idx_in_lights = None
    # We need the lights-list index; final_indices maps final[j] → lights[orig_idx]
    tentative_j = final.index(tentative_best)
    tentative_idx_in_lights = final_indices[tentative_j]
    ref_lum = _get_lum(tentative_idx_in_lights)

    # Precompute the (fixed) reference pyramid + per-level FFTs ONCE — every
    # frame registers against this same reference, so this removes N-1 redundant
    # reference-pyramid builds and reference FFTs.
    ref_prepared = prepare_ref_pyramid(ref_lum)

    # Run cheap pyramid registration in parallel
    pyramid_shifts: List[Tuple[float, float]] = [(0.0, 0.0)] * len(final)

    def _pyramid_one(j: int, orig_idx: int) -> Tuple[int, float, float]:
        if orig_idx == tentative_idx_in_lights:
            return j, 0.0, 0.0
        lum = _get_lum(orig_idx)
        try:
            sy, sx = calculate_shift_pyramid_pref(ref_prepared, lum)
            if np.isfinite(sy) and np.isfinite(sx):
                return j, float(sy), float(sx)
        except Exception:
            pass
        return j, 0.0, 0.0

    safe_print(f"  Pyramid pass for reference selection ({len(final)} frames, {n_workers} workers)...")
    _t_pyramid = time.time()
    from src.webview import get_webview as _get_wv
    _wv = _get_wv()
    _wv_done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futs = {executor.submit(_pyramid_one, j, orig_idx): j
                for j, orig_idx in enumerate(final_indices)}
        for fut in tqdm(as_completed(futs), total=len(final),
                        desc="  Ref-select", unit="frame"):
            j, sy, sx = fut.result()
            pyramid_shifts[j] = (sy, sx)
            _wv_done += 1
            _wv.progress('Reference selection (pyramid pass)', _wv_done, len(final))
    from src.utils import format_time as _format_time
    safe_print(f"    Pyramid pass: {_format_time(time.time() - _t_pyramid)} "
               f"({len(final) / max(time.time() - _t_pyramid, 1e-9):.1f} frame/s)")

    # Geometric median of shift cloud
    shift_arr = np.array(pyramid_shifts, dtype=np.float64)  # (N, 2)
    med_sy, med_sx = _geometric_median_2d(shift_arr)

    # Centrality score: inversely proportional to distance from geometric median
    dists = np.sqrt((shift_arr[:, 0] - med_sy) ** 2 +
                    (shift_arr[:, 1] - med_sx) ** 2)
    max_dist = dists.max()
    if max_dist < 1e-6:
        centrality = np.ones(len(final))
    else:
        centrality = 1.0 - dists / max_dist  # 1 = at median, 0 = furthest away

    # Quality scores (normalised)
    q_scores = np.array([f.metrics.get('score', 1.0) for f in final], dtype=np.float64)
    q_max = q_scores.max()
    q_norm = q_scores / max(q_max, 1e-9)

    combined = q_norm * (1.0 - centrality_weight) + centrality * centrality_weight
    best_j = int(np.argmax(combined))
    best_frame = final[best_j]
    best_lights_idx = final_indices[best_j]

    safe_print(
        f"  Reference selection: {os.path.basename(best_frame.path)} "
        f"(quality={q_norm[best_j]:.3f}, centrality={centrality[best_j]:.3f}, "
        f"combined={combined[best_j]:.3f})"
    )
    if best_frame is not tentative_best:
        safe_print(
            f"    (Alignment-centrality promoted over pure-quality choice "
            f"{os.path.basename(tentative_best.path)})"
        )

    return best_frame, best_lights_idx, pyramid_shifts


def _filter_shift_outliers(
    final: List[FrameInfo],
    final_indices: List[int],
    pyramid_shifts: List[Tuple[float, float]],
    sigma: float = None,
) -> np.ndarray:
    """Flag frames whose pyramid shift lies >sigma*MAD from the session median.

    Returns a boolean array (len = len(final)) where True = frame is an outlier
    that should be skipped in the expensive full registration pass.  Outliers
    are caused by large mount slippage, guiding failures, or meridian flips
    that score fine on per-frame quality metrics.
    """
    from src.models import Config
    if sigma is None:
        sigma = Config.SHIFT_OUTLIER_SIGMA

    shifts_arr = np.array(pyramid_shifts, dtype=np.float64)
    magnitudes = np.sqrt(shifts_arr[:, 0] ** 2 + shifts_arr[:, 1] ** 2)

    med = float(np.median(magnitudes))
    mad = float(np.median(np.abs(magnitudes - med)))
    threshold = med + sigma * max(1.4826 * mad, 1.0)  # floor of 1px avoids zero-threshold

    outlier_mask = magnitudes > threshold
    n_out = int(outlier_mask.sum())
    if n_out > 0:
        safe_print(
            f"  Shift outlier filter: {n_out}/{len(final)} frames exceed "
            f"{sigma:.1f}σ threshold ({threshold:.1f} px); "
            f"skipping expensive registration for these frames"
        )
        for j, is_out in enumerate(outlier_mask):
            if is_out:
                _log.debug(
                    "  Outlier frame: %s (shift=%.1f px)",
                    os.path.basename(final[j].path), magnitudes[j]
                )
    return outlier_mask


def _match_frame_stars(
    lum: np.ndarray,
    shift: Tuple[float, float],
    transform,
    ref_tree,
    ref_xy: np.ndarray,
    noise_val: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Warp ``lum`` into aligned space with the frame's shift/transform,
    re-detect stars, and KDTree-match them to the reference catalog.

    Shared by ``score_registration_residuals`` (RMS check) and
    ``fit_displacement_field`` (elastic registration) so both spend exactly
    one detect+match pass per frame instead of two.

    Returns (ref_xy_matched, frame_xy_matched) — both (N, 2) arrays of matched
    star pairs in (x, y) order — or None if fewer than 3 stars matched.
    """
    from src.quality import _sep_detect_stars, _detect_stars_multi_fwhm, _ensure_photutils

    aligned_lum = ndimage.shift(
        lum, shift=shift, order=1, mode='constant', cval=0.0
    ) if transform is None else ndimage.affine_transform(
        lum.astype(np.float64),
        transform.params[:2, :2],
        offset=-(transform.params[:2, :2] @ np.array([shift[0], shift[1]])),
        order=1, mode='constant', cval=0.0
    )
    frame_stars = _sep_detect_stars(aligned_lum.astype(np.float32), noise_val)
    if frame_stars is None or len(frame_stars) < 3:
        _ensure_photutils()
        from src.quality import DAOStarFinder
        if DAOStarFinder is not None:
            bg = float(np.median(aligned_lum))
            std = max(noise_val, 1e-6)
            frame_stars = _detect_stars_multi_fwhm(
                aligned_lum - bg, threshold=5.0 * std
            )
    if frame_stars is None or len(frame_stars) < 3:
        return None
    frame_xy = np.array(
        [(float(s['xcentroid']), float(s['ycentroid'])) for s in frame_stars],
        dtype=np.float64
    )
    dists, idx = ref_tree.query(frame_xy, k=1, distance_upper_bound=10.0)
    valid = np.isfinite(dists) & (dists < 10.0)
    if valid.sum() < 3:
        return None
    return ref_xy[idx[valid]], frame_xy[valid]


def score_registration_residuals(
    final: List[FrameInfo],
    shifts: List[Tuple[float, float]],
    transforms,
    ref_lum: np.ndarray,
    ref_stars,
    mem_lum: np.ndarray,
    final_indices: List[int],
    best_lights_idx: int,
    max_residual_px: float = None,
    cached_lums: Optional[List[Optional[np.ndarray]]] = None,
    return_correspondences: bool = False,
    force_full_check: bool = False,
) -> Tuple[List[float], List[bool], Optional[List[Optional[Tuple[np.ndarray, np.ndarray]]]]]:
    """Post-registration centroid residual check.

    After registration, re-detects stars in each registered luminance frame
    and matches them to the reference star catalog.  Frames whose RMS centroid
    displacement exceeds ``max_residual_px`` are flagged as misaligned.

    Args:
        return_correspondences: also return the per-frame matched (ref_xy,
            frame_xy) star pairs (needed by elastic registration's field fit).
        force_full_check: check every frame instead of the risky/random sample
            used for >25 frames (elastic registration needs a field for every
            frame, not just the sampled RMS-check subset).

    Returns:
        residuals       – per-frame RMS centroid displacement in pixels.
        passed          – per-frame bool (True = within tolerance).
        correspondences – per-frame (ref_xy, frame_xy) matched pairs, or None
                          for frames with no match / when not requested.
    """
    from src.models import Config
    from src.quality import _sep_detect_stars, _detect_stars_multi_fwhm, _ensure_photutils

    if max_residual_px is None:
        max_residual_px = Config.REG_RESIDUAL_MAX_PX

    residuals: List[float] = [0.0] * len(final)
    passed: List[bool] = [True] * len(final)
    correspondences: Optional[List[Optional[Tuple[np.ndarray, np.ndarray]]]] = (
        [None] * len(final) if return_correspondences else None
    )

    if ref_stars is None or len(ref_stars) == 0:
        return residuals, passed, correspondences

    try:
        ref_xy = np.array(
            [(float(s['xcentroid']), float(s['ycentroid'])) for s in ref_stars],
            dtype=np.float64
        )
    except Exception:
        return residuals, passed, correspondences

    from scipy.spatial import cKDTree
    ref_tree = cKDTree(ref_xy)

    def _check_one(j: int, f: FrameInfo, orig_idx: int) -> Tuple[int, float, bool, Optional[Tuple[np.ndarray, np.ndarray]]]:
        if orig_idx == best_lights_idx:
            return j, 0.0, True, None
        try:
            if cached_lums is not None and orig_idx < len(cached_lums) and cached_lums[orig_idx] is not None:
                lum = np.asarray(cached_lums[orig_idx], dtype=np.float32)
            else:
                lum = np.array(mem_lum[orig_idx], dtype=np.float32)
            noise_val = float(f.metrics.get('noise', 1.0)) if f.metrics else 1.0
            match = _match_frame_stars(lum, shifts[j], transforms[j], ref_tree,
                                       ref_xy, noise_val)
            if match is None:
                return j, 0.0, True, None
            matched_ref_xy, matched_frame_xy = match
            dists = np.hypot(matched_ref_xy[:, 0] - matched_frame_xy[:, 0],
                             matched_ref_xy[:, 1] - matched_frame_xy[:, 1])
            rms = float(np.sqrt(np.mean(dists ** 2)))
            return j, rms, rms <= max_residual_px, match
        except Exception as exc:
            _log.debug("Residual check failed for frame %s: %s",
                       os.path.basename(f.path), exc)
            return j, 0.0, True, None

    # Sampled check: each frame costs a full-res warp + star detection, and
    # on a healthy run every frame passes. Check the riskiest frames (largest
    # shifts) plus a deterministic ~20% spread of the rest; escalate to the
    # full set only if anything in the sample fails, so the safety net is
    # only paid for when registration actually went wrong.
    n_frames = len(final)
    if n_frames > 25 and not force_full_check:
        mags = [float(np.hypot(s[0], s[1])) if s is not None else 0.0
                for s in shifts]
        order = np.argsort(mags)[::-1]
        n_risky = max(5, n_frames // 10)
        check = set(int(i) for i in order[:n_risky])
        rng = np.random.default_rng(0)
        n_extra = max(10, n_frames // 5) - len(check)
        rest = [j for j in range(n_frames) if j not in check]
        if n_extra > 0 and rest:
            check.update(int(i) for i in
                         rng.choice(rest, size=min(n_extra, len(rest)),
                                    replace=False))
    else:
        check = set(range(n_frames))

    def _run_checks(indices: List[int], label: str) -> None:
        from src.webview import get_webview as _get_wv
        _wv = _get_wv()
        _done = 0
        n_workers = min(os.cpu_count() or 4, max(len(indices), 1))
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futs = {executor.submit(_check_one, j, final[j], final_indices[j]): j
                    for j in indices}
            for fut in tqdm(as_completed(futs), total=len(indices),
                            desc=label, unit="frame"):
                j, rms, ok, match = fut.result()
                residuals[j] = rms
                passed[j] = ok
                if correspondences is not None:
                    correspondences[j] = match
                _done += 1
                _wv.progress('Residual check', _done, len(indices))

    _run_checks(sorted(check), "  Residual check")
    if len(check) < n_frames and any(not passed[j] for j in check):
        remaining = [j for j in range(n_frames) if j not in check]
        safe_print(f"  Residual check: failures in the {len(check)}-frame sample "
                   f"— checking all {len(remaining)} remaining frames")
        _run_checks(remaining, "  Residual check (full)")

    n_failed = sum(1 for p in passed if not p)
    if n_failed > 0:
        safe_print(
            f"  Post-registration residual check: {n_failed}/{len(final)} frames "
            f"exceeded {max_residual_px:.1f} px RMS threshold"
        )

    return residuals, passed, correspondences


def _patch_grid_geometry(H: int, W: int, grid_size: int = None) -> Tuple[int, int, int, int]:
    """Patch height/width and grid dims for a (H, W) frame — shared by the
    Phase-1 scorer and the Phase-2 map builder so their grids always agree."""
    from src.models import Config
    if grid_size is None:
        grid_size = Config.PATCH_GRID_SIZE
    min_patch = Config.PATCH_MIN_SIZE
    # Enforce minimum patch size; fall back to 1x1 grid if image is tiny
    ph = max(H // grid_size, min_patch)
    pw = max(W // grid_size, min_patch)
    ny = max(H // ph, 1)
    nx = max(W // pw, 1)
    return ph, pw, ny, nx


def compute_patch_scores(lum: np.ndarray, grid_size: int = None) -> np.ndarray:
    """Per-patch Brenner sharpness on a coarse (ny, nx) grid, unnormalised.

    Cheap enough to run in Phase 1 while the frame is already in worker
    memory (one diff pass over the luminance); the tiny grid is stored in
    the frame metrics ('_patch_scores') so Phase 2 does not have to re-read
    and re-warp the full-resolution frame just to score it.
    """
    H, W = lum.shape[:2]
    ph, pw, ny, nx = _patch_grid_geometry(H, W, grid_size)
    patch_scores = np.zeros((ny, nx), dtype=np.float32)
    lum_f = lum.astype(np.float64)
    for iy in range(ny):
        for ix in range(nx):
            y0, y1 = iy * ph, min((iy + 1) * ph, H)
            x0, x1 = ix * pw, min((ix + 1) * pw, W)
            patch = lum_f[y0:y1, x0:x1]
            if patch.size < 4:
                continue
            diff = patch[:, 2:] - patch[:, :-2]
            patch_scores[iy, ix] = float(np.mean(diff * diff))
    return patch_scores


def patch_scores_to_map(patch_scores: np.ndarray, H: int, W: int) -> np.ndarray:
    """Normalise a coarse patch-score grid and upsample it to (H, W)."""
    patch_scores = patch_scores.astype(np.float32, copy=True)
    ny, nx = patch_scores.shape
    pmax = patch_scores.max()
    if pmax > 1e-12:
        patch_scores /= pmax
    if ny == H and nx == W:
        return patch_scores
    zoom_y = H / ny
    zoom_x = W / nx
    quality_map = ndimage.zoom(patch_scores, (zoom_y, zoom_x), order=1)
    quality_map = np.clip(quality_map, 0.0, 1.0).astype(np.float32)
    # Ensure exact output shape (zoom can differ by 1 pixel due to rounding)
    if quality_map.shape != (H, W):
        from scipy.ndimage import map_coordinates
        gy = np.linspace(0, ny - 1, H)
        gx = np.linspace(0, nx - 1, W)
        coords_y, coords_x = np.meshgrid(gy, gx, indexing='ij')
        quality_map = map_coordinates(
            patch_scores.astype(np.float64),
            [coords_y, coords_x], order=1, mode='nearest'
        ).astype(np.float32)
        np.clip(quality_map, 0.0, 1.0, out=quality_map)
    return quality_map


def compute_patch_quality_map(lum: np.ndarray, grid_size: int = None) -> np.ndarray:
    """Divide luminance frame into a grid and compute per-patch Brenner sharpness.

    Returns a float32 (H, W) array where each pixel holds the normalised quality
    weight of its patch.  The map is bilinearly interpolated from the patch grid
    to full resolution so it can be used as a per-pixel stacking weight.

    Higher values indicate sharper, better-seeing patches (suitable for
    weighted stacking in the lucky imaging paradigm).
    """
    from src.models import Config
    if grid_size is None:
        grid_size = Config.PATCH_GRID_SIZE
    min_patch = Config.PATCH_MIN_SIZE

    H, W = lum.shape[:2]
    # Enforce minimum patch size; fall back to 1x1 grid if image is tiny
    ph = max(H // grid_size, min_patch)
    pw = max(W // grid_size, min_patch)
    ny = max(H // ph, 1)
    nx = max(W // pw, 1)

    patch_scores = np.zeros((ny, nx), dtype=np.float32)
    lum_f = lum.astype(np.float64)

    for iy in range(ny):
        for ix in range(nx):
            y0, y1 = iy * ph, min((iy + 1) * ph, H)
            x0, x1 = ix * pw, min((ix + 1) * pw, W)
            patch = lum_f[y0:y1, x0:x1]
            if patch.size < 4:
                continue
            diff = patch[:, 2:] - patch[:, :-2]
            patch_scores[iy, ix] = float(np.mean(diff * diff))

    # Normalise to [0, 1]
    pmax = patch_scores.max()
    if pmax > 1e-12:
        patch_scores /= pmax

    # Upsample patch grid to full frame resolution via bilinear interpolation
    if ny == H and nx == W:
        return patch_scores

    zoom_y = H / ny
    zoom_x = W / nx
    quality_map = ndimage.zoom(patch_scores, (zoom_y, zoom_x), order=1)
    quality_map = np.clip(quality_map, 0.0, 1.0).astype(np.float32)
    # Ensure exact output shape (zoom can differ by 1 pixel due to rounding)
    if quality_map.shape != (H, W):
        from scipy.ndimage import map_coordinates
        gy = np.linspace(0, ny - 1, H)
        gx = np.linspace(0, nx - 1, W)
        coords_y, coords_x = np.meshgrid(gy, gx, indexing='ij')
        quality_map = map_coordinates(
            patch_scores.astype(np.float64),
            [coords_y, coords_x], order=1, mode='nearest'
        ).astype(np.float32)
        np.clip(quality_map, 0.0, 1.0, out=quality_map)

    return quality_map


def fit_displacement_field(ref_xy: np.ndarray, frame_xy: np.ndarray,
                           H: int, W: int) -> Optional[np.ndarray]:
    """Fit a smooth (Gc, Gc, 2) local displacement field (dy, dx) from sparse
    matched-star correspondences.

    Reuses DBE's Gaussian-weighted local-linear regression + Tukey-biweight
    IRLS kernel (``src.background``'s ``_dbe_fit_surface_numpy`` / the native
    Rust ``dbe_fit_surface``), repurposed for two scalar channels (row/col
    residual) instead of brightness. Calls the low-level kernel directly
    rather than the ``_fit_background_surface`` wrapper: that wrapper forces a
    full-resolution ``zoom`` output (this field stays coarse, sampled on
    demand — same rationale as the ``quality_maps`` grids) and falls back to
    an unbounded polynomial extrapolation below 6 samples, which is exactly
    wrong for a displacement correction (an unbounded low-sample extrapolation
    on a background *brightness* surface is a subtle blemish; on a pixel
    *displacement* it can misregister the whole frame). ``ref_xy``/``frame_xy``
    are (x, y) matched star pairs in the same aligned (reference) coordinate
    space as returned by ``_match_frame_stars``.

    Returns None below ``Config.LOCAL_WARP_MIN_STARS`` matches — caller must
    fall back to the frame's existing unmodified affine/translation warp.
    """
    n = len(ref_xy)
    if n < Config.LOCAL_WARP_MIN_STARS:
        return None

    from src.background import _dbe_fit_surface_numpy
    try:
        import astro_native as _dbe_native
        has_dbe_native = hasattr(_dbe_native, 'dbe_fit_surface')
    except Exception:
        _dbe_native = None
        has_dbe_native = False

    # coords[:,0] = row fraction (y/H), coords[:,1] = col fraction (x/W) --
    # matches _dbe_fit_surface_numpy's own convention (background.py:958-959).
    coords = np.column_stack([ref_xy[:, 1] / H, ref_xy[:, 0] / W])
    dy = ref_xy[:, 1] - frame_xy[:, 1]
    dx = ref_xy[:, 0] - frame_xy[:, 0]

    Gc = Config.LOCAL_WARP_GRID_SIZE
    sigma_px = Config.LOCAL_WARP_BANDWIDTH_FRAC * min(H, W)
    tukey_c = 4.685 * (Config.LOCAL_WARP_OUTLIER_SIGMA / 2.5)
    iters = Config.LOCAL_WARP_OUTLIER_ITERS

    def _fit(values: np.ndarray) -> np.ndarray:
        coarse = None
        if has_dbe_native:
            try:
                coarse, _ = _dbe_native.dbe_fit_surface(
                    np.ascontiguousarray(coords, dtype=np.float64),
                    np.ascontiguousarray(values, dtype=np.float64),
                    float(H), float(W), Gc, Gc, float(sigma_px),
                    float(tukey_c), int(iters))
            except Exception:
                coarse = None
        if coarse is None:
            coarse, _ = _dbe_fit_surface_numpy(
                coords, values, float(H), float(W), Gc, Gc, float(sigma_px),
                tukey_c=tukey_c, irls_iters=iters)
        return coarse

    field_y = _fit(dy)
    field_x = _fit(dx)
    field = np.stack([field_y, field_x], axis=-1).astype(np.float32)

    mag = np.hypot(field[..., 0], field[..., 1])
    clamp = Config.LOCAL_WARP_MAX_DISPLACEMENT_PX
    scale = np.minimum(1.0, clamp / np.maximum(mag, 1e-9))
    field *= scale[..., None]
    return field


def sample_displacement_field(field: np.ndarray, H: int, W: int,
                              y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Bilinearly sample a coarse (Gc, Gc, 2) displacement field at full-res
    reference-space coordinates ``(y, x)`` (may be a dense meshgrid or
    fractional drizzle-space coordinates). Returns (dy, dx) matching the
    input arrays' shape."""
    Gc = field.shape[0]
    gi = np.clip(np.asarray(y, dtype=np.float64) / max(H - 1, 1) * (Gc - 1), 0, Gc - 1)
    gj = np.clip(np.asarray(x, dtype=np.float64) / max(W - 1, 1) * (Gc - 1), 0, Gc - 1)
    dy = ndimage.map_coordinates(field[..., 0], [gi, gj], order=1, mode='nearest')
    dx = ndimage.map_coordinates(field[..., 1], [gi, gj], order=1, mode='nearest')
    return dy, dx


_REG_STEP_LABELS = {
    'ref_star_detect': 'Reference star (re-)detection',
    'shift_calculation': 'Shift calculation (phase-corr/RANSAC, all frames)',
    'residual_check': 'Post-registration residual check',
    'patch_quality_maps': 'Patch quality maps (--patch-registration)',
    'displacement_fields': 'Local displacement fields (--elastic-registration)',
}


def _print_registration_breakdown(timings: Dict[str, float]) -> None:
    """Print which Registration sub-step dominates. Unlike Phase 1, the main
    'shift_calculation' step is itself thread-parallel across frames, so these
    numbers are wall-clock for each sub-step directly — no worker-count math
    needed to interpret them."""
    total = sum(timings.values())
    if total <= 0:
        return
    from src.utils import format_time
    safe_print(f"\n  Registration breakdown (wall-clock per sub-step):")
    for key, label in _REG_STEP_LABELS.items():
        t = timings.get(key, 0.0)
        if t <= 0.01 and key not in ('ref_star_detect', 'shift_calculation'):
            continue  # skip unused optional steps (patch/optical-flow off)
        safe_print(f"    {label:<50} {format_time(t):>8}  ({t / total * 100:4.1f}%)")


def run_registration_phase(
    final: List[FrameInfo],
    final_indices: List[int],
    best: FrameInfo,
    best_idx: int,
    ref_lum: np.ndarray,
    mem_lum: np.ndarray,
    cached_lums: List[Optional[np.ndarray]],
    H: int,
    W: int,
    args: argparse.Namespace,
    stats: ProcessingStats,
    pyramid_shifts: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[List[Tuple[float, float]], List[Optional[Any]], Dict[str, Any]]:
    """Compute per-frame registration shifts/transforms for all accepted frames.

    When ``pyramid_shifts`` is supplied (from select_reference_frame), the
    shift-outlier filter is applied before the expensive full-registration pass,
    saving time on frames with catastrophic guiding failures.
    """
    _reg_timings: Dict[str, float] = {}
    _t = time.time()

    ref_stars = best.metrics.get('_star_sources')
    if ref_stars is not None and len(ref_stars) == 0:
        ref_stars = None  # treat empty array same as absent

    # If star sources are missing (lost from checkpoint JSON serialisation, or
    # detection unavailable in Phase 1), attempt re-detection now using the
    # reference luminance already in memory.  Try in order: SEP (fastest, no
    # extra deps), DAOStarFinder (photutils), local-maxima fallback (scipy only).
    if ref_stars is None and HAS_SKIMAGE_TRANSFORM and not getattr(args, 'no_affine', False):
        noise_val = float(best.metrics.get('noise', 1.0)) if best.metrics else 1.0
        _redet_tried: list = []
        try:
            # 1. SEP — handles high-pedestal images well via local background mesh
            from src.quality import _sep_detect_stars
            _redet_tried.append('SEP')
            ref_stars = _sep_detect_stars(ref_lum.astype(np.float32), noise_val)
            if ref_stars is not None and len(ref_stars) > 0:
                safe_print(f"  Re-detected {len(ref_stars)} stars via SEP")
            else:
                ref_stars = None
        except Exception as e:
            _log.debug("SEP re-detection failed: %s", e)

        if ref_stars is None:
            try:
                # 2. DAOStarFinder with sigma-clipped background estimate
                from src.quality import _detect_stars_multi_fwhm, _ensure_photutils, DAOStarFinder
                from astropy.stats import sigma_clipped_stats
                _ensure_photutils()
                if DAOStarFinder is not None and sigma_clipped_stats is not None:
                    _redet_tried.append('DAOStarFinder')
                    _, bg_med, bg_std = sigma_clipped_stats(ref_lum, sigma=3.0, maxiters=5)
                    bg_std_f = float(bg_std) if bg_std else 0.0
                    if bg_std_f > 0:
                        bg_sub = ref_lum - float(bg_med)
                        ref_stars = _detect_stars_multi_fwhm(bg_sub, threshold=5.0 * bg_std_f)
                        if ref_stars is not None and len(ref_stars) > 0:
                            safe_print(f"  Re-detected {len(ref_stars)} stars via DAOStarFinder")
                        else:
                            ref_stars = None
            except Exception as e:
                _log.debug("DAOStarFinder re-detection failed: %s", e)

        if ref_stars is None:
            try:
                # 3. Local-maxima fallback — pure scipy, always available
                from scipy.ndimage import maximum_filter, gaussian_filter
                _redet_tried.append('local-maxima')
                smoothed = gaussian_filter(ref_lum.astype(np.float64), sigma=2.0)
                bg = float(np.median(smoothed))
                thresh = bg + 5.0 * max(noise_val, float(np.std(smoothed)) * 0.5)
                local_max = maximum_filter(smoothed, size=11)
                mask = (smoothed == local_max) & (smoothed > thresh)
                peak_ys, peak_xs = np.where(mask)
                peak_vals = smoothed[peak_ys, peak_xs]
                if len(peak_ys) > 0:
                    order = np.argsort(peak_vals)[::-1]
                    n_stars = min(len(order), 200)
                    dt = np.dtype([('xcentroid', np.float64), ('ycentroid', np.float64),
                                   ('flux', np.float64), ('peak', np.float64),
                                   ('roundness1', np.float64), ('roundness2', np.float64),
                                   ('sharpness', np.float64),
                                   ('a', np.float64), ('b', np.float64), ('theta', np.float64)])
                    ref_stars = np.zeros(n_stars, dtype=dt)
                    ref_stars['xcentroid'] = peak_xs[order[:n_stars]]
                    ref_stars['ycentroid'] = peak_ys[order[:n_stars]]
                    ref_stars['peak'] = peak_vals[order[:n_stars]]
                    ref_stars['flux'] = ref_stars['peak']
                    ref_stars['a'] = 2.0
                    ref_stars['b'] = 2.0
                    ref_stars['roundness1'] = 0.1
                    ref_stars['roundness2'] = 0.1
                    ref_stars['sharpness'] = 0.5
                    safe_print(f"  Re-detected {n_stars} stars via local-maxima fallback")
            except Exception as e:
                _log.debug("Local-maxima re-detection failed: %s", e)

        if ref_stars is not None and len(ref_stars) > 0:
            best.metrics['_star_sources'] = ref_stars
        else:
            tried_str = ', '.join(_redet_tried) if _redet_tried else 'none available'
            safe_print(f"  ⚠ No stars detected in reference frame (tried: {tried_str}) — "
                       f"affine (rotation) registration disabled, falling back to translation only")
            ref_stars = None

    elif ref_stars is None and HAS_SKIMAGE_TRANSFORM and not getattr(args, 'no_affine', False):
        safe_print("  ⚠ No stars detected in reference frame — affine (rotation) "
                   "registration disabled, falling back to translation only")

    _reg_timings['ref_star_detect'], _t = time.time() - _t, time.time()

    # Consensus reference frame: override the selected reference with the frame
    # whose pyramid shift is closest to the session median (most central frame).
    if (getattr(args, 'consensus_ref', False)
            and pyramid_shifts is not None
            and len(final) >= 10):
        try:
            ps_arr = np.array(pyramid_shifts, dtype=np.float64)
            median_sy = float(np.median(ps_arr[:, 0]))
            median_sx = float(np.median(ps_arr[:, 1]))
            dists = np.sqrt((ps_arr[:, 0] - median_sy) ** 2 +
                            (ps_arr[:, 1] - median_sx) ** 2)
            best_j = int(np.argmin(dists))
            new_best_idx = final_indices[best_j]
            if new_best_idx != best_idx:
                best = final[best_j]
                best_idx = new_best_idx
                if cached_lums is not None and best_idx < len(cached_lums) and cached_lums[best_idx] is not None:
                    ref_lum = np.asarray(cached_lums[best_idx], dtype=np.float32)
                else:
                    ref_lum = np.array(mem_lum[best_idx], dtype=np.float32)
                safe_print(f"  Consensus reference: {os.path.basename(best.path)} "
                           f"(shift closest to median)")
        except Exception as _ce:
            _log.debug("Consensus ref failed: %s", _ce)

    # Shift-space outlier pre-filter: skip expensive full registration for
    # frames whose pyramid shift is catastrophically far from the session median.
    outlier_mask = np.zeros(len(final), dtype=bool)
    if (pyramid_shifts is not None
            and not getattr(args, 'no_shift_outlier_filter', False)
            and len(pyramid_shifts) == len(final)):
        outlier_mask = _filter_shift_outliers(final, final_indices, pyramid_shifts)

    shifts = [None] * len(final)
    transforms = [None] * len(final)
    print(f"  Calculating shifts for {len(final)} frames...")

    # Pre-seed outlier frames with their pyramid shift so they still appear in
    # the output with a reasonable (if coarse) alignment rather than zero.
    for j in range(len(final)):
        if outlier_mask[j] and pyramid_shifts is not None:
            shifts[j] = pyramid_shifts[j]
            transforms[j] = None

    # Seed each frame's full registration with the pyramid shift already computed
    # during reference selection, so the coarse-to-fine pyramid runs once per
    # frame across the pipeline instead of twice.  The select_reference_frame
    # pyramid pass measured shifts relative to a *tentative* reference; re-baseline
    # them to the actually-chosen reference by subtracting the reference frame's
    # own pyramid shift (exact for the translation-only pyramid).
    seed_shifts: Optional[List[Tuple[float, float]]] = None
    if pyramid_shifts is not None and len(pyramid_shifts) == len(final):
        try:
            ref_j = final_indices.index(best_idx)
            ref_sy, ref_sx = pyramid_shifts[ref_j]
        except (ValueError, IndexError):
            ref_sy, ref_sx = 0.0, 0.0
        seed_shifts = [(sy - ref_sy, sx - ref_sx) for (sy, sx) in pyramid_shifts]

    gpu = get_gpu()

    def _register_one(j, f, orig_idx):
        if outlier_mask[j]:
            osy, osx = shifts[j] or (0.0, 0.0)
            if abs(osx) > 0.1 * W or abs(osy) > 0.1 * H:
                safe_print(f'Unrealistic pyramid shift {osx},{osy} for {f.path}, ignoring')
                osy, osx = 0.0, 0.0
            return j, (osy, osx), None
        if orig_idx == best_idx or args.no_registration:
            return j, (0.0, 0.0), None
        seed = seed_shifts[j] if seed_shifts is not None else None
        with gpu.stream_context():
            lum = (cached_lums[orig_idx] if cached_lums[orig_idx] is not None
                   else np.array(mem_lum[orig_idx]))
            _masked_corr = getattr(args, 'masked_correlation', False)
            use_affine = HAS_SKIMAGE_TRANSFORM and not getattr(args, 'no_affine', False)
            if use_affine:
                sy, sx = calculate_shift(ref_lum, lum, verbose=False,
                                         skip_phase_cc=args.skip_phase_correlation,
                                         seed_shift=seed,
                                         masked_correlation=_masked_corr,
                                         corr_downsample=2)
                affine_tf = match_stars_affine(ref_stars, f.metrics.get('_star_sources'),
                                               initial_shift=(sy, sx))
                if affine_tf is None:
                    affine_tf = _astroalign_transform(ref_lum, lum)
                if affine_tf is not None:
                    tf_tx, tf_ty = affine_tf.params[0, 2], affine_tf.params[1, 2]
                    tf_rot_deg = abs(np.degrees(np.arctan2(
                        affine_tf.params[1, 0], affine_tf.params[0, 0])))
                    if (abs(tf_tx) > 0.1 * W or abs(tf_ty) > 0.1 * H
                            or tf_rot_deg > Config.AFFINE_MAX_ROTATION_DEG):
                        safe_print(
                            f'Unrealistic affine fit shift=({tf_tx:.1f},{tf_ty:.1f})px '
                            f'rotation={tf_rot_deg:.1f}deg for {f.path}, '
                            f'falling back to translation-only')
                    else:
                        return j, (tf_ty, tf_tx), affine_tf
            sy, sx = calculate_shift(
                ref_lum, lum, verbose=args.verbose,
                debug=args.debug_registration,
                frame_name=os.path.splitext(os.path.basename(f.path))[0],
                skip_phase_cc=args.skip_phase_correlation,
                seed_shift=seed,
                masked_correlation=_masked_corr,
                corr_downsample=2)
            if abs(sx) > 0.1 * W or abs(sy) > 0.1 * H:
                safe_print(f'Unrealistic shift {sx},{sy} for {f.path}, ignoring')
                sx, sy = 0.0, 0.0
            return j, (sy, sx), None

    if gpu.active:
        n_workers = min(gpu.max_gpu_workers(Config.GPU_FFT_WORKER_MB,
                                            Config.GPU_VRAM_RESERVE_MB), len(final))
    else:
        n_workers = min(os.cpu_count() or 4, len(final))
        
    from src.webview import get_webview as _get_wv
    _wv = _get_wv()
    _wv_done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_register_one, j, f, orig_idx): j
                   for j, (f, orig_idx) in enumerate(zip(final, final_indices))}
        for future in tqdm(as_completed(futures), total=len(final),
                           desc="  Registering", unit="frame", disable=args.verbose):
            j, shift_val, transform_val = future.result()
            shifts[j] = shift_val
            transforms[j] = transform_val
            final[j].shift = shift_val
            _wv_done += 1
            _wv.progress('Registering frames', _wv_done, len(final))
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

    _reg_timings['shift_calculation'], _t = time.time() - _t, time.time()

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

    # Post-registration centroid residual check: verify alignment quality by
    # re-detecting stars in each registered frame and measuring RMS star position error.
    # Elastic registration reuses this same detect+match pass for its field fit,
    # forcing it to run (full coverage, not the sampled subset) even if the
    # RMS-reject gate itself was disabled via --no-reg-residual-check.
    elastic_on = getattr(args, 'elastic_registration', False) and not args.no_registration
    correspondences: Optional[List[Optional[Tuple[np.ndarray, np.ndarray]]]] = None
    if ((not getattr(args, 'no_reg_residual_check', False) or elastic_on)
            and ref_stars is not None
            and len(ref_stars) >= 5
            and not args.no_registration):
        if elastic_on and getattr(args, 'no_reg_residual_check', False):
            safe_print("  --elastic-registration needs the residual-check star match "
                       "pass; running it despite --no-reg-residual-check "
                       "(RMS-reject gate stays disabled)")
        safe_print(f"  Post-registration residual check ({len(final)} frames)...")
        residuals, res_passed, correspondences = score_registration_residuals(
            final, shifts, transforms, ref_lum, ref_stars,
            mem_lum, final_indices, best_idx,
            cached_lums=cached_lums,
            return_correspondences=elastic_on,
            force_full_check=elastic_on,
        )
        n_res_failed = sum(1 for p in res_passed if not p)
        for j, (f, passed, rms) in enumerate(zip(final, res_passed, residuals)):
            if f.metrics is not None:
                f.metrics['reg_residual_px'] = round(rms, 3)
        if (n_res_failed > 0 and not getattr(args, 'no_reg_residual_check', False)
                and getattr(args, 'reg_residual_reject', False)):
            rejected_by_residual = [j for j, p in enumerate(res_passed) if not p]
            safe_print(
                f"  Residual rejection (--reg-residual-reject): "
                f"removing {n_res_failed} frame(s)"
            )
            keep_mask = [p for p in res_passed]
            final = [f for f, k in zip(final, keep_mask) if k]
            final_indices = [i for i, k in zip(final_indices, keep_mask) if k]
            shifts = [s for s, k in zip(shifts, keep_mask) if k]
            transforms = [t for t, k in zip(transforms, keep_mask) if k]
            if correspondences is not None:
                correspondences = [c for c, k in zip(correspondences, keep_mask) if k]

    _reg_timings['residual_check'], _t = time.time() - _t, time.time()

    # Elastic (non-rigid) local displacement fields, fit from the matched-star
    # residuals collected above. Frames with too few matched stars fall back
    # to unmodified affine-only warping (None field).
    displacement_fields: Optional[List[Optional[np.ndarray]]] = None
    if elastic_on and correspondences is not None:
        safe_print(f"  Fitting local displacement fields ({len(final)} frames)...")
        displacement_fields = []
        n_fit = 0
        for j in range(len(final)):
            corr = correspondences[j]
            if corr is None:
                displacement_fields.append(None)
                continue
            ref_m, frame_m = corr
            field = fit_displacement_field(ref_m, frame_m, H, W)
            displacement_fields.append(field)
            if field is not None:
                n_fit += 1
        safe_print(f"  Local displacement fields: {n_fit}/{len(final)} frames "
                   f"(others fall back to affine-only: too few matched stars)")

    _reg_timings['displacement_fields'], _t = time.time() - _t, time.time()

    # Patch quality maps for per-pixel lucky-imaging weighted stacking.
    quality_maps: Optional[List[np.ndarray]] = None
    if getattr(args, 'patch_registration', False) and not args.no_registration:
        safe_print(f"\n  Computing patch quality maps ({len(final)} frames)...")
        quality_maps = []
        H_map, W_map = int(mem_lum.shape[1]), int(mem_lum.shape[2])
        ph_g, pw_g, _, _ = _patch_grid_geometry(H_map, W_map)
        n_from_scores = 0
        for j, orig_idx in enumerate(final_indices):
            grid = final[j].metrics.get('_patch_scores') if final[j].metrics else None
            if grid is not None and np.asarray(grid).ndim == 2:
                # Fast path: Brenner patch scores were already computed in
                # Phase 1 while the frame was in worker memory. The map is a
                # smooth per-patch weight field (patches are hundreds of px),
                # so translating the coarse grid by the frame's shift in
                # patch units matches the old full-res warp+score to well
                # under a patch width; the <=0.3deg field rotations move
                # corner patches by <0.05 patch and are ignored.
                if transforms[j] is not None:
                    t_xy = transforms[j].params[:2, 2]
                    sy, sx = float(t_xy[1]), float(t_xy[0])
                else:
                    sy, sx = shifts[j]
                g = np.asarray(grid, dtype=np.float32)
                if sy != 0.0 or sx != 0.0:
                    g = ndimage.shift(g, shift=(sy / ph_g, sx / pw_g),
                                      order=1, mode='nearest')
                gmax = float(g.max())
                if gmax > 1e-12:
                    g = g / gmax
                quality_maps.append(np.clip(g, 0.0, 1.0))
                n_from_scores += 1
                continue
            # Fallback (frames without Phase-1 scores, e.g. resumed
            # checkpoints): re-read, warp, and score at coarse resolution.
            lum = np.array(mem_lum[orig_idx]).astype(np.float32)
            if transforms[j] is not None:
                from scipy.ndimage import affine_transform as _aff
                tfm = transforms[j]
                R = tfm.params[:2, :2]
                t_xy = tfm.params[:2, 2]
                t_rc = np.array([t_xy[1], t_xy[0]])
                offset = -R @ t_rc
                lum = _aff(lum.astype(np.float64), R, offset=offset,
                           order=1, mode='constant', cval=0.0).astype(np.float32)
            elif shifts[j] != (0.0, 0.0):
                lum = ndimage.shift(lum, shift=shifts[j], order=1,
                                    mode='constant', cval=0.0)
            g = compute_patch_scores(lum).astype(np.float32)
            gmax = float(g.max())
            if gmax > 1e-12:
                g = g / gmax
            quality_maps.append(np.clip(g, 0.0, 1.0))
        if n_from_scores:
            safe_print(f"  Patch quality maps computed "
                       f"({n_from_scores}/{len(final)} from Phase-1 scores).")
        else:
            safe_print(f"  Patch quality maps computed.")

    _reg_timings['patch_quality_maps'] = time.time() - _t
    _print_registration_breakdown(_reg_timings)

    dither_info = detect_dither(shifts, verbose=False)
    dither_info['quality_maps'] = quality_maps  # carry through to stacking
    if displacement_fields is not None:
        dither_info['displacement_fields'] = displacement_fields
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

    return shifts, transforms, dither_info, final, final_indices


# ---------------------------------------------------------------------------
# Comet stacking support
# ---------------------------------------------------------------------------

def find_comet_centroid(lum: np.ndarray,
                        smooth_sigma: float = 5.0,
                        percentile: float = 99.5,
                        seed: Optional[Tuple[float, float]] = None,
                        search_radius: float = 50.0) -> Tuple[float, float]:
    """Locate the brightest compact extended object in a luminance frame.

    Uses a Difference-of-Gaussians (DoG) approach to suppress point sources
    (stars) while enhancing the diffuse coma/nucleus blob.  Falls back to the
    original Gaussian-blur method if DoG produces no candidates.

    Args:
        lum:           2-D luminance frame (H, W).
        smooth_sigma:  Gaussian blur radius for the legacy fallback path.
        percentile:    Threshold percentile for nucleus detection.
        seed:          Optional (row, col) approximate nucleus position.  When
                       provided the peak search is restricted to within
                       ``search_radius`` pixels of the seed.
        search_radius: Pixel radius around the seed used to restrict the search.

    Returns:
        (cy, cx) sub-pixel centroid of the brightest region.
    """
    lum64 = lum.astype(np.float64)
    H, W = lum64.shape

    def _find_in_map(filtered: np.ndarray) -> Optional[Tuple[float, float]]:
        """Find the best centroid in a filtered luminance map, optionally restricted by seed."""
        work = filtered.copy()
        if seed is not None:
            # Build a mask that keeps only pixels within search_radius of the seed
            sy, sx = float(seed[0]), float(seed[1])
            yy, xx = np.mgrid[:H, :W]
            outside = (yy - sy) ** 2 + (xx - sx) ** 2 > search_radius ** 2
            work[outside] = work.min() - 1.0  # push outside pixels below any threshold

        threshold = np.percentile(work, percentile)
        mask = work > threshold
        if not mask.any():
            return None

        labeled, n_labels = ndimage.label(mask)
        if n_labels == 0:
            cy, cx = np.unravel_index(int(np.argmax(work)), work.shape)
            return float(cy), float(cx)

        fluxes = [float(np.sum(work[labeled == i])) for i in range(1, n_labels + 1)]
        best_label = int(np.argmax(fluxes)) + 1
        cy, cx = ndimage.center_of_mass(work, labeled, best_label)
        return float(cy), float(cx)

    # --- DoG detection: suppress point sources, enhance diffuse blobs ---
    try:
        sigma_small = Config.COMET_DOG_SIGMA_SMALL
        sigma_large = Config.COMET_DOG_SIGMA_LARGE
        dog = (ndimage.gaussian_filter(lum64, sigma=sigma_large) -
               ndimage.gaussian_filter(lum64, sigma=sigma_small))
        dog = np.clip(dog, 0.0, None)  # keep only positive (bright blob) response
        if dog.max() > 0:
            result = _find_in_map(dog)
            if result is not None:
                return result
    except Exception:
        pass

    # --- Fallback: original Gaussian-blur + threshold method ---
    smoothed = ndimage.gaussian_filter(lum64, sigma=smooth_sigma)
    result = _find_in_map(smoothed)
    if result is not None:
        return result

    return lum.shape[0] / 2.0, lum.shape[1] / 2.0


def find_comet_tail_pa(lum: np.ndarray, nucleus_y: float, nucleus_x: float,
                       sample_radius: float = 50.0) -> float:
    """Estimate the position angle (degrees, N through E) of the comet tail.

    The tail extends anti-solar (away from the intensity gradient), so we
    compute the gradient of the smoothed image at the nucleus position and
    return the angle pointing *away* from the brighter direction.

    Args:
        lum:           2-D luminance (H, W).
        nucleus_y:     Row coordinate of the nucleus.
        nucleus_x:     Column coordinate of the nucleus.
        sample_radius: Radius around nucleus used to compute the gradient.

    Returns:
        Tail position angle in degrees (measured clockwise from North = up).
        Returns 0.0 if gradient is below noise.
    """
    try:
        H, W = lum.shape
        # Smooth strongly to get a clean gradient at the coma scale
        smoothed = ndimage.gaussian_filter(lum.astype(np.float64), sigma=max(sample_radius * 0.3, 5.0))
        # Compute gradient at nucleus location (via sobel or finite differences on the smoothed image)
        gy = ndimage.sobel(smoothed, axis=0)
        gx = ndimage.sobel(smoothed, axis=1)
        # Sample gradient in a small window around nucleus
        r = max(1, int(sample_radius * 0.2))
        y0 = max(0, int(nucleus_y) - r)
        y1 = min(H, int(nucleus_y) + r + 1)
        x0 = max(0, int(nucleus_x) - r)
        x1 = min(W, int(nucleus_x) + r + 1)
        gy_n = float(np.mean(gy[y0:y1, x0:x1]))
        gx_n = float(np.mean(gx[y0:y1, x0:x1]))
        if abs(gy_n) < 1e-12 and abs(gx_n) < 1e-12:
            return 0.0
        # The tail is anti-solar (opposite to the gradient direction)
        # angle_from_north_cw: north is -row direction, east is +col direction
        # PA (N through E CW) = atan2(gx_tail, -gy_tail)
        # tail direction is anti-gradient: (-gx_n, -gy_n) in (col, row) convention
        pa_rad = np.arctan2(-gx_n, gy_n)  # tail col component, then row component
        return float(np.degrees(pa_rad) % 360.0)
    except Exception:
        return 0.0


def fetch_comet_ephemeris(designation: str, obs_times: List[str],
                          observer_location: Optional[str] = None):
    """Fetch predicted RA/Dec of a comet at given UTC times using JPL Horizons.

    Args:
        designation:       JPL Horizons designation, e.g. "C/2023 A3".
        obs_times:         List of ISO UTC timestamp strings ('YYYY-MM-DDTHH:MM:SS').
        observer_location: MPC code or 'lon,lat,elev' string (default: geocentric).

    Returns:
        List of (ra_deg, dec_deg) tuples per obs_time, or None on failure.
    """
    try:
        from astroquery.jplhorizons import Horizons
    except ImportError:
        safe_print("  [Comet] WARNING: astroquery not installed — ephemeris tracking disabled")
        return None

    try:
        # Parse observer location
        location = None
        if observer_location:
            parts = observer_location.split(',')
            if len(parts) == 3:
                try:
                    location = {'lon': float(parts[0]), 'lat': float(parts[1]),
                                'elevation': float(parts[2])}
                except ValueError:
                    location = observer_location  # try as MPC code
            else:
                location = observer_location  # MPC code

        results = []
        for t in obs_times:
            try:
                obj = Horizons(id=designation, location=location,
                               epochs={'start': t, 'stop': t, 'step': '1m'})
                eph = obj.ephemerides()
                if eph is not None and len(eph) > 0:
                    results.append((float(eph['RA'][0]), float(eph['DEC'][0])))
                else:
                    results.append(None)
            except Exception as _e:
                _log.debug("Horizons query failed for %s at %s: %s", designation, t, _e)
                results.append(None)
        return results
    except Exception as e:
        safe_print(f"  [Comet] WARNING: ephemeris fetch failed: {e}")
        return None


def run_comet_registration_phase(
    final: List[FrameInfo],
    final_indices: List[int],
    best_idx: int,
    ref_lum: np.ndarray,
    mem_lum: np.ndarray,
    H: int,
    W: int,
    args: argparse.Namespace,
    stats: ProcessingStats,
) -> Tuple[List[Tuple[float, float]], List[Optional[Any]], Dict[str, Any]]:
    """Compute per-frame registration shifts aligned on a comet nucleus.

    Instead of aligning on the star field, this tracks the brightest extended
    blob (comet nucleus) in each frame relative to the reference frame.

    Supports:
        - DoG-based nucleus detection (more robust than simple Gaussian blur).
        - Manual seed via args.comet_xy (X,Y string parsed to floats).
        - Frame-to-frame predicted position tracking within search_radius.
        - Optional affine (rotation+scale) correction via args.comet_affine.
        - Tail position angle estimation stored in returned dither_info.
        - Ephemeris-aided tracking via args.comet_designation.
        - Trailing warning when angular velocity x exptime > 1 pixel.

    Returns:
        (shifts, transforms, comet_dither_info)
    """
    search_radius = float(getattr(args, 'comet_search_radius', 50.0))

    # --- Parse manual seed (--comet-xy X,Y) ---
    ref_seed = None
    comet_xy_str = getattr(args, 'comet_xy', None)
    if comet_xy_str:
        try:
            parts = str(comet_xy_str).split(',')
            cx_user = float(parts[0].strip())
            cy_user = float(parts[1].strip())
            # CLI gives (X=col, Y=row) -> seed is (row, col)
            ref_seed = (cy_user, cx_user)
            safe_print(f"  [Comet] Using manual seed: X={cx_user:.1f}, Y={cy_user:.1f}")
        except Exception as _e:
            safe_print(f"  [Comet] WARNING: could not parse --comet-xy '{comet_xy_str}': {_e}")

    # --- Ephemeris-aided tracking ---
    eph_positions = None
    comet_designation = getattr(args, 'comet_designation', None)
    if comet_designation:
        # Extract DATE-OBS from each frame header
        obs_times = []
        for f in final:
            t = f.header.get('DATE-OBS') or f.header.get('DATE_OBS') or ''
            obs_times.append(str(t))
        observer_site = getattr(args, 'observer_site', None)
        safe_print(f"  [Comet] Fetching ephemeris for '{comet_designation}'...")
        eph_positions = fetch_comet_ephemeris(comet_designation, obs_times, observer_site)
        if eph_positions:
            n_ok = sum(1 for p in eph_positions if p is not None)
            safe_print(f"  [Comet] Ephemeris: {n_ok}/{len(final)} positions retrieved")

    print(f"  [Comet] Locating nucleus in reference frame...")
    # Find the reference frame index in final
    try:
        ref_j = next(j for j, orig_idx in enumerate(final_indices) if orig_idx == best_idx)
    except StopIteration:
        ref_j = 0

    ref_cy, ref_cx = find_comet_centroid(ref_lum, seed=ref_seed, search_radius=search_radius)
    print(f"  [Comet] Reference nucleus centroid: ({ref_cx:.1f}, {ref_cy:.1f})")

    # Estimate tail PA from the reference frame
    tail_pa_deg = find_comet_tail_pa(ref_lum, ref_cy, ref_cx)
    if getattr(args, 'verbose', False):
        safe_print(f"  [Comet] Estimated tail PA: {tail_pa_deg:.1f} deg")

    # --- Trailing warning from ephemeris ---
    if eph_positions is not None and len(eph_positions) >= 2:
        try:
            valid_ephs = [(i, p) for i, p in enumerate(eph_positions) if p is not None]
            if len(valid_ephs) >= 2:
                i0, p0 = valid_ephs[0]
                i1, p1 = valid_ephs[-1]
                # Angular separation in arcsec
                dra = (p1[0] - p0[0]) * np.cos(np.radians((p0[1] + p1[1]) / 2.0))
                ddec = p1[1] - p0[1]
                ang_sep_deg = np.sqrt(dra ** 2 + ddec ** 2)
                # Time difference: use index difference as proxy if no timestamps available
                n_frames = max(i1 - i0, 1)
                # Check plate scale from first frame header
                hdr0 = final[0].header if final else {}
                pixscale = hdr0.get('PIXSCALE') or hdr0.get('CDELT1')
                if pixscale is not None:
                    try:
                        pixscale_arcsec = abs(float(pixscale))
                        if pixscale_arcsec < 0.001:  # CDELT1 is in degrees
                            pixscale_arcsec *= 3600.0
                        exptime = float(hdr0.get('EXPTIME', 60.0) or 60.0)
                        # Angular velocity per frame interval (very rough)
                        ang_per_frame_arcsec = ang_sep_deg * 3600.0 / n_frames
                        trailing_px = ang_per_frame_arcsec / max(pixscale_arcsec, 1e-6)
                        if trailing_px > 1.0:
                            safe_print(
                                f"  WARNING: comet trailing ~{trailing_px:.1f} px per frame "
                                f"at current exposure time"
                            )
                    except Exception:
                        pass
        except Exception:
            pass

    shifts: List[Tuple[float, float]] = [(0.0, 0.0)] * len(final)
    transforms: List[Optional[Any]] = [None] * len(final)
    # Store per-frame centroids for affine post-correction
    centroids: List[Optional[Tuple[float, float]]] = [None] * len(final)
    centroids[ref_j] = (ref_cy, ref_cx)

    use_affine = getattr(args, 'comet_affine', False) and HAS_SKIMAGE_TRANSFORM

    def _register_comet(j: int, orig_idx: int) -> Tuple[int, Tuple[float, float], Optional[Tuple[float, float]]]:
        if orig_idx == best_idx or getattr(args, 'no_registration', False):
            return j, (0.0, 0.0), (ref_cy, ref_cx)
        lum = np.array(mem_lum[orig_idx])
        # Predicted seed: previous shift extrapolated linearly (use ref centroid displaced)
        pred_seed = None
        if j > 0 and centroids[j - 1] is not None:
            prev_cy, prev_cx = centroids[j - 1]
            # Extrapolate linearly: keep same position as last known
            pred_seed = (prev_cy, prev_cx)
        elif ref_seed is not None:
            pred_seed = ref_seed
        cy, cx = find_comet_centroid(lum, seed=pred_seed, search_radius=search_radius)
        sy = ref_cy - cy
        sx = ref_cx - cx
        return j, (sy, sx), (cy, cx)

    gpu = get_gpu()
    n_workers = min(
        gpu.max_gpu_workers(Config.GPU_FFT_WORKER_MB, Config.GPU_VRAM_RESERVE_MB)
        if gpu.active else (os.cpu_count() or 4),
        len(final),
    )

    print(f"  [Comet] Tracking nucleus in {len(final)} frames...")
    # Process sequentially to allow linear seed extrapolation
    for j, orig_idx in enumerate(tqdm(
            list(enumerate(final_indices)),
            total=len(final), desc="  Comet tracking", unit="frame",
            disable=getattr(args, 'verbose', False))):
        j_idx, orig_idx = j, orig_idx
        break  # tqdm wrapping doesn't work cleanly with enumerate; just iterate below

    for j, orig_idx in enumerate(final_indices):
        j_result, shift_val, centroid_val = _register_comet(j, orig_idx)
        shifts[j] = shift_val
        centroids[j] = centroid_val
        if getattr(args, 'verbose', False):
            sy, sx = shift_val
            safe_print(
                f"    {os.path.basename(final[j].path)}: "
                f"comet shift=({sx:+.1f}, {sy:+.1f}) px"
            )

    # --- Optional affine correction around nucleus ---
    if use_affine:
        safe_print(f"  [Comet] Computing affine (rotation+scale) corrections per frame...")
        ref_stars = None
        try:
            ref_lum_np = ref_lum.astype(np.float32)
            from src.quality import _sep_detect_stars
            ref_stars = _sep_detect_stars(ref_lum_np, float(np.std(ref_lum_np)))
        except Exception:
            pass

        if ref_stars is not None and len(ref_stars) >= 3:
            for j, orig_idx in enumerate(final_indices):
                if orig_idx == best_idx:
                    continue
                try:
                    lum = np.array(mem_lum[orig_idx]).astype(np.float32)
                    from src.quality import _sep_detect_stars
                    frame_stars = _sep_detect_stars(lum, float(np.std(lum)))
                    if frame_stars is not None and len(frame_stars) >= 3:
                        sy, sx = shifts[j]
                        affine_tf = match_stars_affine(ref_stars, frame_stars,
                                                       initial_shift=(sy, sx))
                        if affine_tf is not None:
                            transforms[j] = affine_tf
                except Exception as _ae:
                    _log.debug("Comet affine failed for frame %d: %s", j, _ae)
        else:
            safe_print("  [Comet] Not enough stars for affine correction — using translation only")

    shift_mags = [np.sqrt(s[0] ** 2 + s[1] ** 2) for s in shifts]
    print(f"  [Comet] Mean shift: {np.mean(shift_mags):.1f} px, "
          f"max: {np.max(shift_mags):.1f} px")

    comet_dither_info: Dict[str, Any] = {
        'nucleus_y': ref_cy,
        'nucleus_x': ref_cx,
        'tail_pa_deg': tail_pa_deg,
        'centroids': centroids,
    }

    return shifts, transforms, comet_dither_info
