"""Frame-level processing: calibration, debayering, parallel dispatch, quality gating."""
from __future__ import annotations

import argparse
import os
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.gpu_context import get_gpu
from src.models import Config, FrameInfo, ProcessingStats
from src.utils import safe_print, print_quality_table, format_time
from src.io_fits import load_fits, load_frame
from src.debayer import (debayer, green_equalize, remove_hot_pixels_bayer,
                         apply_hot_pixel_map_bayer,
                         remove_hot_pixels_rgb, remove_hot_pixels_rgb_with_lum,
                         white_balance_grayworld, white_balance_whitepatch,
                         correct_chromatic_aberration,
                         measure_chromatic_aberration, apply_chromatic_aberration)
from src.quality import validate_image_data, compute_quality_metrics
from src.stacking import lacosmic_reject

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable


def _get_webview():
    from src.webview import get_webview
    return get_webview()


def _publish_frame_thumb(wv, args, name: str, rgb, counter: list) -> None:
    """Publish a per-frame thumbnail every Nth accepted frame (Phase 1).

    counter is a one-element list [int] of accepted frames seen so far, mutated
    in place. `every` comes from --web-view-frame-every (0 disables); the first
    frame is always shown so the viewer lights up immediately.
    """
    every = int(getattr(args, 'web_view_frame_every', 5) or 0)
    if every <= 0 or not wv.active or rgb is None:
        return
    counter[0] += 1
    c = counter[0]
    if c == 1 or c % every == 0:
        try:
            wv.frame_preview(name, np.asarray(rgb), args=args)
        except Exception:
            pass


def _get_frame_rotation(hdr: dict) -> Optional[float]:
    """Extract camera/rotator angle from FITS header. Returns degrees or None."""
    for key in ('ROTATANG', 'ROTANGLE', 'POSANGLE', 'PA', 'ANGLE', 'ROTATOR'):
        val = hdr.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return None


def _build_flat_norm(
    masters: Dict[str, Optional[np.ndarray]],
    lights: Optional[List['FrameInfo']] = None,
) -> None:
    """Pre-compute masters['_flat_norm'], rotating the flat to match lights if needed.

    Stores the result in masters['_flat_norm'].  Safe to call multiple times —
    returns immediately if '_flat_norm' is already set.
    """
    if masters.get('_flat_norm') is not None:
        return

    flat_arr = masters.get('flat')
    if flat_arr is None:
        return

    med = float(np.median(flat_arr))
    if med <= 1e-6:
        return

    zero_frac = float(np.mean(flat_arr < med * 0.05))
    if zero_frac > 0.01:
        safe_print(f"  WARNING: flat field has {zero_frac * 100:.1f}% near-zero pixels "
                   f"(< 5% of median) — possible sensor defects or bad flat")

    flat_norm = np.clip(flat_arr / med, 0.4, 2.5).astype(np.float32)

    # Rotate flat to match light frame orientation when a mismatch is detected.
    # A flat taken with the camera at a different rotation angle creates a
    # vignetting correction that is offset from the actual vignetting pattern
    # of the light frames, producing swirly/crescent dark artifacts.
    flat_rotation = masters.get('flat_rotation')
    if flat_rotation is not None and lights:
        light_rots = [_get_frame_rotation(f.header) for f in lights]
        light_rots = [r for r in light_rots if r is not None]
        if light_rots:
            light_rotation = float(np.median(light_rots))
            delta = (light_rotation - flat_rotation + 180.0) % 360.0 - 180.0
            if abs(delta) > 0.5:
                try:
                    from scipy.ndimage import rotate as _ndimage_rotate
                    safe_print(f"  Rotating master flat by {delta:+.1f}° to match lights "
                               f"(flat={flat_rotation:.1f}°, lights={light_rotation:.1f}°)")
                    flat_norm = _ndimage_rotate(flat_norm, -delta, reshape=False,
                                               order=1, mode='nearest')
                    flat_norm = np.clip(flat_norm, 0.4, 2.5).astype(np.float32)
                except Exception as e:
                    safe_print(f"  WARNING: flat rotation correction failed ({e}) — "
                               "proceeding with unrotated flat")

    masters['_flat_norm'] = flat_norm


def _process_single_frame(path: str, header: dict, masters: Dict[str, Optional[np.ndarray]],
                          debayer_method: str, white_balance: str,
                          ca_correction: bool = False,
                          cosmic_ray_rejection: bool = False,
                          quick_quality: bool = False,
                          skip_quality: bool = False,
                          advanced_metrics: bool = True,
                          preloaded_data: Optional[tuple] = None,
                          session_bayer: Optional[str] = None,
                          pre_gradient_removal: bool = False,
                          ca_shifts: Optional[dict] = None,
                          trail_reject: bool = False) -> Dict[str, Any]:
    """Process one frame: load, calibrate, debayer, hot-pixel, quality.

    Returns dict with keys: 'rgb', 'lum', 'metrics', 'error'.
    Used by both sequential and parallel paths.
    """
    timings: Dict[str, float] = {}
    _t = time.perf_counter()

    try:
        if preloaded_data is not None:
            data, hdr = preloaded_data
        else:
            data, hdr = load_frame(path)
    except Exception as e:
        return {'error': f'load error: {e}'}
    if data is None or data.size == 0:
        return {'error': 'empty data array'}
    timings['load'], _t = time.perf_counter() - _t, time.perf_counter()

    # ── GPU calibration probe: decide once per frame size whether GPU is faster ──
    _gpu_ctx = get_gpu()
    _use_gpu_calib = False
    if _gpu_ctx.active and data.ndim == 2:
        _shape = data.shape
        if _shape not in _gpu_calib_cache:
            with _gpu_calib_lock:
                if _shape not in _gpu_calib_cache:
                    _ensure_gpu_masters(masters, _gpu_ctx)
                    _gpu_calib_cache[_shape] = _probe_gpu_calibration(
                        _shape[0], _shape[1], _gpu_ctx, masters)
        _use_gpu_calib = _gpu_calib_cache.get(_shape, False)

    # Calibration — preserve negative noise through bias/dark subtraction,
    # clip only once after all steps to avoid cumulative truncation of shadow detail
    try:
        if _use_gpu_calib:
            # ── GPU path: H→D transfer once, then dark/flat on device ──
            # data stays as CuPy through green_equalize + bilinear debayer.
            try:
                xp  = _gpu_ctx.xp
                gm  = _gpu_masters
                data_g = xp.asarray(data.astype(np.float32))

                bias_g = gm.get('bias')
                if bias_g is not None and bias_g.shape == data_g.shape:
                    data_g = data_g - bias_g

                dark_g = gm.get('dark')
                if dark_g is not None and dark_g.shape == data_g.shape:
                    dark_exptime = masters.get('dark_exptime') or None
                    light_exptime = float(hdr.get('EXPTIME', 0) or 0) or None
                    if dark_exptime and light_exptime and dark_exptime > 0:
                        dark_scale = light_exptime / dark_exptime
                        if abs(dark_scale - 1.0) > 0.05:
                            _wk = (round(dark_exptime, 1), round(light_exptime, 1))
                            if _wk not in _warned_dark_scales:
                                _warned_dark_scales.add(_wk)
                                safe_print(f"  NOTE: dark scaling {dark_scale:.2f}x "
                                           f"(light {light_exptime:.1f}s / dark {dark_exptime:.1f}s) "
                                           f"-- ensure dark exposure matches lights for best results")
                    else:
                        dark_scale = 1.0
                    data_g = data_g - dark_g * dark_scale
                    if bias_g is not None and bias_g.shape == data_g.shape:
                        data_g = data_g + bias_g * dark_scale

                flat_norm_g = gm.get('_flat_norm')
                if flat_norm_g is not None and flat_norm_g.shape == data_g.shape:
                    data_g = data_g / flat_norm_g

                if not bool(xp.all(xp.isfinite(data_g))):
                    return {'error': 'calibration produced non-finite values'}
                data_g = xp.clip(data_g, 0, None)

                # Hot-pixel map needs CPU; do a brief round-trip only when map exists.
                # Statistical Bayer removal is skipped — _fix_hot_rgb catches hot pixels
                # post-debayer without needing a separate Bayer-space pass.
                hot_map = masters.get('hot_pixel_map')
                if hot_map is not None and hot_map.shape == data_g.shape:
                    _data_np = data_g.get()
                    _data_np = apply_hot_pixel_map_bayer(_data_np, hot_map)
                    data_g = xp.asarray(_data_np)

                data = data_g  # CuPy; green_equalize + bilinear debayer consume it directly
            except Exception as _gpu_exc:
                # GPU calibration failed — disable for this shape and fall back to CPU
                _gpu_calib_cache[data.shape] = False
                _use_gpu_calib = False
                if _gpu_ctx.is_oom(_gpu_exc):
                    _gpu_ctx.free_pool()

        if not _use_gpu_calib:
            bias_arr = masters.get('bias')
            if bias_arr is not None and bias_arr.shape != data.shape:
                bias_arr = None
            if bias_arr is not None:
                data = data.astype(np.float32, copy=False)
                data -= bias_arr

            dark_arr = masters.get('dark')
            if dark_arr is not None and dark_arr.shape == data.shape:
                dark_exptime = masters.get('dark_exptime') or None
                light_exptime = float(hdr.get('EXPTIME', 0) or 0) or None
                if dark_exptime and light_exptime and dark_exptime > 0:
                    dark_scale = light_exptime / dark_exptime
                    if abs(dark_scale - 1.0) > 0.05:
                        key = (round(dark_exptime, 1), round(light_exptime, 1))
                        if key not in _warned_dark_scales:
                            _warned_dark_scales.add(key)
                            safe_print(f"  NOTE: dark scaling {dark_scale:.2f}x "
                                       f"(light {light_exptime:.1f}s / dark {dark_exptime:.1f}s) "
                                       f"-- ensure dark exposure matches lights for best results")
                else:
                    dark_scale = 1.0
                # Subtract scaled dark in-place: data -= (dark - bias) * scale
                # = data -= dark*scale, then += bias*scale (avoiding a full dark_current copy)
                data = data.astype(np.float32, copy=False)
                np.subtract(data, dark_arr * dark_scale, out=data)
                if bias_arr is not None:
                    data += bias_arr * dark_scale

            flat_norm = masters.get('_flat_norm')  # pre-computed by caller when possible
            if flat_norm is None:
                flat_arr = masters.get('flat')
                if flat_arr is not None and flat_arr.shape == data.shape:
                    med = np.median(flat_arr)
                    if med > 1e-6:
                        zero_frac = float(np.mean(flat_arr < med * 0.05))
                        if zero_frac > 0.01:
                            safe_print(f"  WARNING: flat field has {zero_frac * 100:.1f}% near-zero "
                                       f"pixels -- possible sensor defects or bad flat")
                        flat_norm = np.clip(flat_arr / med, 0.4, 2.5)
            if flat_norm is not None and flat_norm.shape == data.shape:
                data = data.astype(np.float32, copy=False)
                data /= flat_norm

            if not np.isfinite(data).all():
                return {'error': 'calibration produced non-finite values'}
            data = np.clip(data, 0, None)
            if data.ndim == 2 and masters.get('hot_pixel_map') is not None:
                hot_map = masters['hot_pixel_map']
                if hot_map.shape == data.shape:
                    data = apply_hot_pixel_map_bayer(data, hot_map)
            if data.ndim == 2:
                data = remove_hot_pixels_bayer(data)
    except Exception as e:
        return {'error': f'calibration error: {e}'}
    timings['calibrate'], _t = time.perf_counter() - _t, time.perf_counter()

    # Debayer
    try:
        if data.ndim == 2:
            bayer = hdr.get('BAYERPAT', hdr.get('COLORTYP', session_bayer or 'RGGB'))
            data = green_equalize(data, pattern=bayer)
            # Malvar/VNG use cv2 which requires numpy — transfer D→H if data is on GPU
            if debayer_method != 'bilinear' and hasattr(data, 'get'):
                data = data.get()
            rgb = debayer(data, pattern=bayer, method=debayer_method)
        else:
            rgb = data
    except Exception as e:
        return {'error': f'debayering error: {e}'}
    timings['debayer'], _t = time.perf_counter() - _t, time.perf_counter()

    # Hot pixel removal — returns (rgb_fixed, lum) to avoid recomputing luminance.
    try:
        if rgb.ndim != 3 or rgb.shape[2] < 1:
            return {'error': f'Invalid RGB shape: {rgb.shape}'}
        rgb, lum = remove_hot_pixels_rgb_with_lum(rgb)
    except Exception as e:
        return {'error': f'hot pixel removal error: {e}'}
    timings['hotpix'], _t = time.perf_counter() - _t, time.perf_counter()

    # Vignetting/background calibration map (--vignette-map): must run here,
    # in native per-frame sensor space before registration warps this frame
    # to some session-specific orientation, and before white balance (which
    # would rescale channels by a different per-frame factor than whatever
    # the map was built under) -- see src/vignette_calib.py. rgb may still
    # be a CuPy array at this point (GPU calibration path keeps it on-device
    # through debayer); the map itself is always host numpy, so force host
    # here rather than requiring apply_vignette_correction to be GPU-aware.
    vignette_map = masters.get('vignette')
    if vignette_map is not None:
        try:
            from src.vignette_calib import apply_vignette_correction
            rgb = get_gpu().to_host(rgb)
            rgb = apply_vignette_correction(rgb, vignette_map)
        except Exception as e:
            return {'error': f'vignette correction error: {e}'}
    timings['vignette'], _t = time.perf_counter() - _t, time.perf_counter()

    # White balance
    try:
        if white_balance == 'grayworld':
            rgb = white_balance_grayworld(rgb)
        elif white_balance == 'whitepatch':
            rgb = white_balance_whitepatch(rgb)
    except Exception as e:
        return {'error': f'white balance error: {e}'}

    gpu = get_gpu()
    rgb = gpu.to_host(rgb)
    timings['white_balance'], _t = time.perf_counter() - _t, time.perf_counter()

    if ca_correction:
        try:
            if ca_shifts is not None:
                # Session-constant shifts measured once up front — CA is a
                # property of the optics, fixed in the sensor frame; only the
                # (cheap) warp runs per frame.
                rgb = apply_chromatic_aberration(rgb, ca_shifts)
            else:
                rgb = correct_chromatic_aberration(rgb)
        except Exception:
            pass
    timings['ca_correction'], _t = time.perf_counter() - _t, time.perf_counter()

    if cosmic_ray_rejection:
        try:
            rgb = lacosmic_reject(rgb)
        except Exception:
            pass
    timings['cosmic_ray_rejection'], _t = time.perf_counter() - _t, time.perf_counter()

    # Satellite / aircraft trail rejection — erase long straight streaks before
    # the frame enters the stack (robust even at low frame counts, where
    # sigma-clip can't reject them).
    if trail_reject:
        try:
            from src.trail_reject import reject_trails
            rgb, _ = reject_trails(rgb, verbose=False)
        except Exception:
            pass
    timings['trail_reject'], _t = time.perf_counter() - _t, time.perf_counter()

    # Recompute lum only when white balance or post-processing changed the image.
    try:
        if white_balance != 'none' or ca_correction or cosmic_ray_rejection:
            lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        else:
            lum = np.asarray(lum)  # ensure host numpy array
    except Exception as e:
        return {'error': f'luminance computation error: {e}'}
    timings['lum_recompute'], _t = time.perf_counter() - _t, time.perf_counter()

    # Per-frame polynomial gradient removal (degree-2 background subtraction)
    if pre_gradient_removal:
        try:
            lum_pg = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
            H_pg, W_pg = lum_pg.shape
            step = 32
            ys = np.arange(step // 2, H_pg, step)
            xs = np.arange(step // 2, W_pg, step)
            Ys, Xs = np.meshgrid(ys, xs, indexing='ij')
            vals = lum_pg[Ys, Xs]
            Yn = Ys.ravel() / H_pg
            Xn = Xs.ravel() / W_pg
            A = np.column_stack([np.ones(len(Yn)), Yn, Xn, Yn * Yn, Yn * Xn, Xn * Xn])
            b_vec = vals.ravel().astype(np.float64)
            coeffs, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
            rows_n = np.arange(H_pg, dtype=np.float64) / H_pg
            cols_n = np.arange(W_pg, dtype=np.float64) / W_pg
            Rf, Cf = np.meshgrid(rows_n, cols_n, indexing='ij')
            bg_model = (coeffs[0] + coeffs[1] * Rf + coeffs[2] * Cf
                        + coeffs[3] * Rf * Rf + coeffs[4] * Rf * Cf
                        + coeffs[5] * Cf * Cf).astype(np.float32)
            for c in range(rgb.shape[2]):
                rgb[:, :, c] = np.clip(rgb[:, :, c] - bg_model, 0.0, None)
            lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        except Exception:
            pass
    timings['pre_gradient_removal'], _t = time.perf_counter() - _t, time.perf_counter()

    is_valid, validation_error = validate_image_data(lum, os.path.basename(path))
    if not is_valid:
        return {'error': f'validation failed: {validation_error}'}
    timings['validate'], _t = time.perf_counter() - _t, time.perf_counter()

    metrics = {} if skip_quality else compute_quality_metrics(
        lum, quick=quick_quality, advanced_metrics=advanced_metrics)
    timings['quality'], _t = time.perf_counter() - _t, time.perf_counter()

    # Patch quality scores for --patch-registration, computed here while the
    # luminance is already in worker memory. Phase 2 shifts this tiny coarse
    # grid into aligned space instead of re-reading and re-warping the full
    # frame per map. Computed unconditionally: whether patch registration is
    # enabled is decided by the auto-advisor AFTER Phase 1, and one Brenner
    # pass over an in-cache frame is trivial next to the steps above.
    try:
        from src.registration import compute_patch_scores
        metrics['_patch_scores'] = compute_patch_scores(np.asarray(lum))
    except Exception:
        pass
    timings['patch_scores'] = time.perf_counter() - _t
    return {'rgb': rgb, 'lum': lum, 'metrics': metrics, 'error': None, 'timings': timings}


# Module-level state for parallel workers (must be module-level for pickling)
_worker_masters: Dict[str, Any] = {}
_worker_trail_reject: bool = False  # per-session trail-rejection flag for pool workers
_warned_dark_scales: set = set()  # dedup dark-scale mismatch warnings across frames

# GPU calibration probe cache — probed once per unique frame size per session
_gpu_calib_cache: Dict[Tuple[int, int], bool] = {}
_gpu_calib_lock = threading.Lock()
_gpu_masters: Dict[str, Any] = {}        # GPU-resident copies of master arrays
_gpu_masters_sig: Optional[Tuple] = None  # signature to detect master changes


def _masters_sig(masters: Dict) -> Tuple:
    """Lightweight signature so we can detect when masters change between sessions."""
    def _s(key):
        arr = masters.get(key)
        return (arr.shape, arr.dtype.str) if isinstance(arr, np.ndarray) else None
    return (_s('dark'), _s('flat'), _s('bias'))


def _ensure_gpu_masters(masters: Dict, gpu) -> None:
    """Upload master calibration arrays to GPU once per session; re-uploads on change."""
    global _gpu_masters, _gpu_masters_sig
    sig = _masters_sig(masters)
    if sig == _gpu_masters_sig and _gpu_masters:
        return
    xp = gpu.xp
    result: Dict[str, Any] = {}
    for key in ('bias', 'dark', 'flat', '_flat_norm', 'hot_pixel_map'):
        arr = masters.get(key)
        if isinstance(arr, np.ndarray):
            result[key] = xp.asarray(arr)
        elif arr is not None:
            result[key] = arr
    # Derive GPU _flat_norm if not pre-computed
    if result.get('_flat_norm') is None and result.get('flat') is not None:
        flat_g = result['flat']
        med = float(xp.median(flat_g))
        if med > 1e-6:
            result['_flat_norm'] = xp.clip(flat_g / med, 0.4, 2.5)
    for key, val in masters.items():
        if key not in result:
            result[key] = val
    _gpu_masters = result
    _gpu_masters_sig = sig


def _probe_gpu_calibration(H: int, W: int, gpu, masters: Dict) -> bool:
    """Benchmark CPU vs GPU for one frame's calibration.

    Measures CPU numpy (dark/flat ops) vs GPU path (PCIe H→D transfer + CuPy ops).
    Masters are assumed pre-uploaded to GPU.  Returns True if GPU wins.
    """
    import time
    xp = gpu.xp
    has_dark = masters.get('dark') is not None
    has_flat = masters.get('_flat_norm') is not None or masters.get('flat') is not None

    rng = np.random.default_rng(0)
    frame_np = rng.random((H, W)).astype(np.float32)
    dark_np  = np.full((H, W), 0.05, dtype=np.float32) if has_dark else None
    flat_np  = np.ones((H, W), dtype=np.float32)        if has_flat else None
    dark_g   = xp.asarray(dark_np) if dark_np is not None else None
    flat_g   = xp.asarray(flat_np) if flat_np is not None else None

    N = 8
    # CPU warmup
    for _ in range(2):
        d = frame_np.copy()
        if dark_np is not None: d -= dark_np
        if flat_np is not None: d /= flat_np
    t0 = time.perf_counter()
    for _ in range(N):
        d = frame_np.copy()
        if dark_np is not None: np.subtract(d, dark_np, out=d)
        if flat_np is not None: d /= flat_np
    cpu_s = (time.perf_counter() - t0) / N

    try:
        # GPU warmup (H→D transfer included; masters already on device)
        for _ in range(3):
            d_g = xp.asarray(frame_np)
            if dark_g is not None: d_g -= dark_g
            if flat_g is not None: d_g /= flat_g
            xp.cuda.Device(0).synchronize()
        t0 = time.perf_counter()
        for _ in range(N):
            d_g = xp.asarray(frame_np)
            if dark_g is not None: d_g -= dark_g
            if flat_g is not None: d_g /= flat_g
        xp.cuda.Device(0).synchronize()
        gpu_s = (time.perf_counter() - t0) / N
    except Exception:
        return False

    winner = 'GPU' if gpu_s < cpu_s else 'CPU'
    safe_print(f"  Calibration probe {W}x{H}: "
               f"CPU {cpu_s*1000:.1f}ms  GPU {gpu_s*1000:.1f}ms  -> {winner} path selected")
    return gpu_s < cpu_s


# Per-worker rayon thread cap — see _pin_worker_to_single_thread. Isolated
# lacosmic_reject_native scaling (1/2/4/8/16 threads -> 3865/2150/1118/680/603
# ms) argued for capping at 4 rather than 1, to keep some of that speedup. But
# a real 233-frame production run with n_workers == os.cpu_count() (Phase 1
# already saturates every core one-frame-per-process) showed the cap=4 choice
# was net negative: lacosmic's real per-call cost was 7955ms/frame under
# n_workers x 4 = 64-thread oversubscription -- *worse* than the isolated
# fully-serial 1-thread number (3865ms), let alone the 4-thread one (1118ms).
# When workers already equal core count, any per-worker internal parallelism
# is oversubscription by construction; there's no free core left for rayon to
# use. 1 thread avoids that regardless of what isolated (uncontended)
# benchmarks show.
_RAYON_WORKER_CAP = 1


def _pin_worker_to_single_thread() -> None:
    """Bound (not necessarily eliminate) each worker process's internal
    threading, so Phase 1's ProcessPoolExecutor parallelism (one process per
    worker, already using all cores) isn't multiplied by each worker ALSO
    spinning up a full-core-count thread pool internally.

    Two different libraries need two different treatments here, found by
    measuring rather than assuming:

    - cv2 (debayer) and BLAS/OpenMP (numpy/scipy) are pinned to exactly 1
      thread. These have no per-call algorithmic dependency on parallelism in
      our usage (small per-frame arrays), and confirmed by measurement: with
      W worker processes each defaulting to `cores` internal threads (cv2
      alone defaults to 16 threads/call on a 16-core box here), that's up to
      W x cores OS threads fighting over `cores` physical cores — pinning
      these to 1 measurably cut Calibrate/Debayer time (~25-28%).

    - rayon (our own native kernels — lacosmic, median filter, etc.) is
      capped at a small number instead of 1. Unlike cv2/BLAS, these calls ARE
      where the real per-frame work happens, and forcing them fully serial
      costs a lot: measured 603ms (full/no-contention) vs 3865ms (1 thread)
      for lacosmic_reject_native on a full-res frame — 6.4x slower. Capping
      at _RAYON_WORKER_CAP keeps most of that speedup (~3.5x back at 4
      threads) while bounding worst-case oversubscription to
      workers x _RAYON_WORKER_CAP instead of workers x cores.

    Env vars are read lazily by OpenBLAS/MKL/rayon at first use, so setting
    them here (in the pool initializer, before any task runs) takes effect.
    """
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[var] = '1'
    os.environ['RAYON_NUM_THREADS'] = str(_RAYON_WORKER_CAP)
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass


def _init_worker_shm(shm_specs: Dict[str, tuple], trail_reject: bool = False) -> None:
    """Initializer for pool workers — attach to shared-memory calibration arrays.

    *shm_specs* maps master name → (shm_name, dtype_str, shape).  Workers
    attach (read-only view) without copying data or touching the filesystem.
    *trail_reject* is a per-session flag stashed in a module global so it need
    not be threaded through the per-frame task tuple.
    """
    _pin_worker_to_single_thread()
    global _worker_masters, _worker_trail_reject
    _worker_trail_reject = bool(trail_reject)
    _worker_masters = {}
    for name, (shm_name, dtype_str, shape) in shm_specs.items():
        shm = SharedMemory(name=shm_name, create=False)
        arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        # Keep shm alive for the process lifetime; store both so we can close later.
        _worker_masters[name] = arr
        _worker_masters[f'_shm_{name}'] = shm


def _measure_session_ca(frames: List[FrameInfo], args) -> Optional[dict]:
    """Median CA channel shifts from a few sample frames, or None to fall
    back to per-frame measurement.

    Lateral chromatic aberration is a property of the optics, fixed in the
    sensor frame for a whole session (field rotation moves the sky, not the
    lens), so measuring it on every frame re-derives the same constant ~200
    times. Three frames spread through the session are measured (in
    parallel) on uncalibrated debayered data — dark/flat correction shifts
    no channel structure and the correlation is global — and the medians are
    applied to every frame. Requires >=2 agreeing samples per channel;
    returns None (per-frame fallback) otherwise. Skipped for short sessions
    where the measurement overhead wouldn't pay back.
    """
    if not getattr(args, 'ca_correction', False) or len(frames) < 12:
        return None
    n = len(frames)
    idxs = sorted({n // 6, n // 2, (5 * n) // 6})
    _sb = getattr(args, '_session_bayer', None)

    def _one(i: int):
        data, hdr = load_frame(frames[i].path)
        if data is None or data.size == 0:
            return None
        if data.ndim == 2:
            bayer = hdr.get('BAYERPAT', hdr.get('COLORTYP', _sb or 'RGGB'))
            data = green_equalize(np.asarray(data), pattern=bayer)
            rgb = debayer(np.asarray(data), pattern=bayer, method='bilinear')
        else:
            rgb = data
        return measure_chromatic_aberration(
            np.ascontiguousarray(rgb, dtype=np.float32))

    samples = []
    try:
        with ThreadPoolExecutor(max_workers=len(idxs)) as ex:
            for fut in [ex.submit(_one, i) for i in idxs]:
                try:
                    s = fut.result()
                    if s is not None:
                        samples.append(s)
                except Exception:
                    pass
    except Exception:
        return None

    out: dict = {}
    measured_any = False
    for c in (0, 2):
        vals = [s[c] for s in samples if s.get(c) is not None]
        if len(vals) >= 2:
            measured_any = True
            sy = float(np.median([v[0] for v in vals]))
            sx = float(np.median([v[1] for v in vals]))
            # Sub-threshold skip: correcting less than a quarter pixel does
            # more harm than good — the Lanczos resample costs time and adds
            # mild ringing, while a <0.25px channel offset is invisible under
            # a ~5px FWHM PSF. None here means "measured, negligible": the
            # apply step becomes a no-op rather than falling back to
            # per-frame measurement.
            if max(abs(sy), abs(sx)) < Config.CA_MIN_SHIFT_PX:
                out[c] = None
            else:
                out[c] = (sy, sx)
        else:
            out[c] = None
    if not measured_any:
        return None  # measurement failed -> per-frame fallback
    if out[0] is None and out[2] is None:
        safe_print("  CA correction: measured shifts below "
                   f"{Config.CA_MIN_SHIFT_PX}px — no correction needed")
    return out


def _fmt_ca(s: Optional[Tuple[float, float]]) -> str:
    return f"({s[1]:+.2f}, {s[0]:+.2f})px" if s is not None else "none"


def _parallel_frame_worker(
        args_tuple: tuple) -> Tuple[int, Optional[dict], Optional[str], Optional[dict]]:
    """Worker function for ProcessPoolExecutor. Must be module-level for pickling."""
    (path, frame_idx, debayer_method, white_balance,
     mm_rgb_path, mm_lum_path, rgb_shape, lum_shape,
     ca_correction, cosmic_ray_rejection, advanced_metrics, session_bayer,
     pre_gradient_removal, skip_quality, ca_shifts) = args_tuple
    global _worker_masters
    result = _process_single_frame(path, {}, _worker_masters, debayer_method, white_balance,
                                   ca_correction=ca_correction,
                                   cosmic_ray_rejection=cosmic_ray_rejection,
                                   advanced_metrics=advanced_metrics,
                                   skip_quality=skip_quality,
                                   session_bayer=session_bayer,
                                   pre_gradient_removal=pre_gradient_removal,
                                   ca_shifts=ca_shifts,
                                   trail_reject=_worker_trail_reject)
    if result.get('error'):
        return (frame_idx, None, result['error'], None)

    metrics_clean = dict(result['metrics'])
    timings = result.get('timings', {})

    _t = time.perf_counter()
    try:
        mem_rgb = np.memmap(mm_rgb_path, dtype='float32', mode='r+', shape=rgb_shape)
        mem_lum = np.memmap(mm_lum_path, dtype='float32', mode='r+', shape=lum_shape)
        mem_rgb[frame_idx] = result['rgb']
        mem_lum[frame_idx] = result['lum']
        # Flush deferred to main process after all workers complete — flushing
        # the entire memmap on every frame causes excessive concurrent I/O.
        del mem_rgb, mem_lum
    except Exception as e:
        return (frame_idx, None, f'memmap write error: {e}', None)
    timings['memmap_write'] = time.perf_counter() - _t

    return (frame_idx, metrics_clean, None, timings)


def execute_frame_processing(
    lights: List[FrameInfo],
    masters: Dict[str, Optional[np.ndarray]],
    args: argparse.Namespace,
    mem_rgb: np.ndarray,
    mem_lum: np.ndarray,
    mm_rgb_path: str,
    mm_lum_path: str,
    cached_lums: list,
    rgb_shape: tuple,
    lum_shape: tuple,
    rejected_reasons: dict,
    stats: ProcessingStats,
) -> None:
    """Process all light frames and write results to memmaps.

    Selects between ProcessPool, ThreadPool, or sequential execution.
    Mutates lights[*].metrics / .accepted and rejected_reasons in-place.
    """
    # Aggregate per-step timing (sum of per-frame seconds) across whichever
    # dispatch path runs below, so Phase 1 can report which sub-step actually
    # dominates "Quality+Load" instead of leaving it as one opaque number.
    _step_totals: Dict[str, float] = {}
    _step_frames = 0
    _worker_count_used = 1  # updated below once the real dispatch path is known
    _wv_thumb_count = [0]   # accepted frames seen, for per-frame thumbnails

    def _accum(timings: Optional[dict]) -> None:
        nonlocal _step_frames
        if not timings:
            return
        _step_frames += 1
        for k, v in timings.items():
            _step_totals[k] = _step_totals.get(k, 0.0) + v

    n = len(lights)
    use_process_pool = (getattr(args, 'parallel', 1) != 1
                        and not get_gpu().active
                        and n >= 4)

    # Pre-compute flat_norm once (with rotation correction) so workers don't redo it.
    _build_flat_norm(masters, lights)

    if use_process_pool:
        # Auto: use all cores (RAM cap below governs the real limit). The old
        # hard cap of 8 throttled Phase 1 on high-core machines; -j N overrides.
        workers = args.parallel if args.parallel > 0 else min(os.cpu_count() or 4, n)

        # Cap workers so total frame-data memory stays within available RAM.
        # Each worker peak: raw Bayer + calibration intermediates (~3×raw) + RGB + Python overhead.
        try:
            import psutil
            avail_mb = psutil.virtual_memory().available / 1e6
            H_f, W_f = rgb_shape[1], rgb_shape[2]
            bayer_mb = H_f * W_f * 4 / 1e6
            rgb_mb   = H_f * W_f * 3 * 4 / 1e6
            per_worker_mb = bayer_mb * 4 + rgb_mb + 200  # 4× bayer for cal intermediates + RGB + Python
            safe_workers = max(1, int(avail_mb / per_worker_mb))
            if safe_workers < workers:
                safe_print(f"  NOTE: limiting workers {workers}→{safe_workers} "
                           f"(avail RAM {avail_mb:.0f} MB, ~{per_worker_mb:.0f} MB/worker)")
                workers = safe_workers
        except Exception:
            pass

        _worker_count_used = max(workers, 1)
        print(f"  Processing {n} frames in parallel ({workers} workers)...")

        # Share calibration arrays via shared memory — zero disk I/O, one copy
        # in RAM shared across all workers (read-only view per worker process).
        shm_blocks: list = []
        shm_specs: Dict[str, tuple] = {}
        for name, arr in masters.items():
            if arr is None or name.startswith('_shm_') or not isinstance(arr, np.ndarray):
                continue
            arr_c = np.ascontiguousarray(arr)
            shm = SharedMemory(create=True, size=arr_c.nbytes)
            shm_arr = np.ndarray(arr_c.shape, dtype=arr_c.dtype, buffer=shm.buf)
            shm_arr[:] = arr_c
            shm_blocks.append(shm)
            shm_specs[name] = (shm.name, arr_c.dtype.str, arr_c.shape)

        _ca = getattr(args, 'ca_correction', False)
        _cr = getattr(args, 'cosmic_ray_rejection', False)
        _adv = getattr(args, 'advanced_metrics', True)
        _sb = getattr(args, '_session_bayer', None)
        _pgr = getattr(args, 'pre_gradient_removal', False)
        _tr = getattr(args, 'trail_reject', False)
        _ca_shifts = _measure_session_ca(lights, args) if _ca else None
        if _ca_shifts is not None:
            safe_print(f"  CA correction: session-constant shifts "
                       f"R={_fmt_ca(_ca_shifts.get(0))} B={_fmt_ca(_ca_shifts.get(2))} "
                       f"(measured once, applied per frame)")
        tasks = [(lights[i].path, i, args.debayer_method, args.white_balance,
                  mm_rgb_path, mm_lum_path, rgb_shape, lum_shape, _ca, _cr, _adv, _sb, _pgr, False,
                  _ca_shifts)
                 for i in range(n)]

        try:
            with ProcessPoolExecutor(max_workers=workers,
                                     initializer=_init_worker_shm,
                                     initargs=(shm_specs, _tr)) as pool:
                futures = {pool.submit(_parallel_frame_worker, t): t[1] for t in tasks}
                _wv = _get_webview()
                _wv_done = 0
                for future in tqdm(as_completed(futures), total=n,
                                   desc="  Processing", unit="frame",
                                   disable=args.verbose):
                    idx = futures[future]
                    frame_idx, metrics, error, timings = future.result()
                    _accum(timings)
                    f = lights[frame_idx]
                    _wv_done += 1
                    _wv.progress('Processing frames', _wv_done, n)
                    _wv.frame_metrics(os.path.basename(f.path), metrics,
                                      accepted=error is None)
                    if error:
                        f.accepted = False
                        f.metrics = {'error': error}
                        rejected_reasons[f.path] = error
                        stats.add_error(f.path, error)
                        if args.verbose:
                            print(f'  REJECT {os.path.basename(f.path)}: {error}')
                    else:
                        f.metrics = metrics
                        _publish_frame_thumb(_wv, args,
                                             os.path.basename(f.path),
                                             mem_rgb[frame_idx], _wv_thumb_count)
                        if args.verbose:
                            m = f.metrics
                            safe_print(f'    {os.path.basename(f.path)}: '
                                       f'score={m["score"]:.0f}  SNR={m["snr"]:.1f}  '
                                       f'stars={m["star_count"]}  FWHM={m.get("fwhm",0):.1f}  '
                                       f'sharpness={m.get("sharpness",0):.0f}')
                            safe_print(f'      bg={m.get("background",0):.1f}  '
                                       f'noise={m.get("noise",0):.2f}  '
                                       f'brightness={m.get("brightness",0):.1f}  '
                                       f'contrast={m.get("contrast",0):.1f}  '
                                       f'dynamic_range={m.get("dynamic_range",0):.0f}')
        finally:
            # Release shared memory after all workers are done.
            for shm in shm_blocks:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass

        # Flush memmaps once after all workers complete — per-frame flushing
        # inside workers causes excessive concurrent full-file I/O.
        _t_flush = time.perf_counter()
        mem_rgb.flush()
        mem_lum.flush()
        _step_totals['final_flush'] = time.perf_counter() - _t_flush

    elif n >= 2:
        gpu = get_gpu()
        if gpu.active:
            n_workers = min(gpu.max_gpu_workers(Config.GPU_PHASE1_WORKER_MB,
                                                Config.GPU_VRAM_RESERVE_MB), n)
        else:
            n_workers = min(os.cpu_count() or 4, n)

        _adv = getattr(args, 'advanced_metrics', True)
        _sb  = getattr(args, '_session_bayer', None)
        _pgr = getattr(args, 'pre_gradient_removal', False)
        _ca  = getattr(args, 'ca_correction', False)
        _cr  = getattr(args, 'cosmic_ray_rejection', False)
        _ca_shifts = _measure_session_ca(lights, args) if _ca else None
        if _ca_shifts is not None:
            safe_print(f"  CA correction: session-constant shifts "
                       f"R={_fmt_ca(_ca_shifts.get(0))} B={_fmt_ca(_ca_shifts.get(2))} "
                       f"(measured once, applied per frame)")

        # I/O prefetch pool — sized to keep GPU workers fed without thrashing the disk
        _io_workers = min(max(n_workers, 8), 32, n)
        _io_pool = ThreadPoolExecutor(max_workers=_io_workers)
        _load_futures = {i: _io_pool.submit(load_frame, f.path)
                         for i, f in enumerate(lights)}

        # When GPU is active, quality metrics (CPU-bound: star detection, FWHM) run
        # in a dedicated CPU pool so GPU workers are never idle waiting for photutils.
        # GPU workers skip quality (skip_quality=True) and return immediately after
        # writing to memmap, keeping VRAM freed as quickly as possible.
        _use_qpool = gpu.active
        _n_cpu = min(os.cpu_count() or 4, n)
        _qpool = ThreadPoolExecutor(max_workers=_n_cpu) if _use_qpool else None
        _qfuts: Dict[int, Any] = {}   # frame index → quality Future

        _worker_count_used = max(n_workers, 1)
        safe_print(f"  Processing {n} frames: {n_workers} GPU + {_n_cpu} quality threads"
                   if _use_qpool else
                   f"  Processing {n} frames with {n_workers} threads...")

        def _thread_process_frame(i, f):
            try:
                preloaded = _load_futures[i].result()
            except Exception as e:
                return i, None, f'load error: {e}', None, None
            with gpu.stream_context():
                result = _process_single_frame(
                    f.path, f.header, masters, args.debayer_method, args.white_balance,
                    ca_correction=_ca,
                    cosmic_ray_rejection=_cr,
                    advanced_metrics=_adv,
                    preloaded_data=preloaded,
                    session_bayer=_sb,
                    pre_gradient_removal=_pgr,
                    skip_quality=_use_qpool,  # GPU workers skip quality
                    ca_shifts=_ca_shifts,
                    trail_reject=getattr(args, 'trail_reject', False))
            if result.get('error'):
                return i, None, result['error'], None, None
            mem_rgb[i] = result['rgb']
            mem_lum[i] = result['lum']
            return i, result['metrics'], None, result['lum'], result.get('timings')

        _completed = 0
        _free_interval = Config.GPU_POOL_FREE_INTERVAL

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_thread_process_frame, i, f): i
                       for i, f in enumerate(lights)}
            # Always show the progress bar here: unlike the CPU path, per-frame
            # quality metrics are computed asynchronously and only printed AFTER
            # this loop, so disabling the bar under -v would leave the whole GPU
            # processing loop with no output at all (looks hung).
            for future in tqdm(as_completed(futures), total=n,
                               desc="  Processing", unit="frame",
                               disable=False):
                i, metrics, error, lum_arr, timings = future.result()
                _accum(timings)
                f = lights[i]
                if error:
                    f.accepted = False
                    f.metrics = {'error': error}
                    rejected_reasons[f.path] = error
                    stats.add_error(f.path, error)
                    if args.verbose:
                        safe_print(f'  REJECT {os.path.basename(f.path)}: {error}')
                else:
                    cached_lums[i] = lum_arr
                    if _use_qpool and lum_arr is not None:
                        # Submit quality to CPU pool; GPU thread is already freed.
                        # Its compute time runs concurrently with other frames'
                        # GPU work and isn't attributable to a single frame here,
                        # so it is not folded into the per-step totals below —
                        # the GPU path's timing breakdown is best-effort.
                        _qfuts[i] = _qpool.submit(
                            compute_quality_metrics, lum_arr, advanced_metrics=_adv)
                    else:
                        f.metrics = metrics
                        if args.verbose:
                            m = f.metrics
                            safe_print(f'    {os.path.basename(f.path)}: '
                                       f'score={m["score"]:.0f}  SNR={m["snr"]:.1f}  '
                                       f'stars={m["star_count"]}  FWHM={m.get("fwhm",0):.1f}  '
                                       f'sharpness={m.get("sharpness",0):.0f}')
                            safe_print(f'      bg={m.get("background",0):.1f}  '
                                       f'noise={m.get("noise",0):.2f}  '
                                       f'brightness={m.get("brightness",0):.1f}  '
                                       f'contrast={m.get("contrast",0):.1f}  '
                                       f'dynamic_range={m.get("dynamic_range",0):.0f}')
                _completed += 1
                _wv = _get_webview()
                _wv.progress('Processing frames', _completed, n)
                if error is None and f.metrics:
                    _wv.frame_metrics(os.path.basename(f.path), f.metrics)
                if error is None:
                    _publish_frame_thumb(_wv, args, os.path.basename(f.path),
                                         mem_rgb[i], _wv_thumb_count)
                # Periodically free CuPy's cached memory pool to prevent VRAM exhaustion
                # from accumulating unused cached blocks across many completed frames.
                if gpu.active and (_completed % _free_interval == 0):
                    gpu.free_pool()

        # Collect deferred quality results (CPU pool runs while GPU was active)
        if _qfuts:
            for i, q_fut in _qfuts.items():
                f = lights[i]
                try:
                    f.metrics = q_fut.result()
                except Exception:
                    f.metrics = {}
                if args.verbose and f.metrics:
                    m = f.metrics
                    safe_print(f'    {os.path.basename(lights[i].path)}: '
                               f'score={m.get("score",0):.0f}  SNR={m.get("snr",0):.1f}  '
                               f'stars={m.get("star_count",0)}  '
                               f'FWHM={m.get("fwhm",0):.1f}  '
                               f'sharpness={m.get("sharpness",0):.0f}')
            _qpool.shutdown(wait=False)

        if gpu.active:
            gpu.free_pool()  # final pool flush before moving to Phase 2

        _io_pool.shutdown(wait=False)
        mem_rgb.flush()
        mem_lum.flush()

    else:
        print(f"  Processing {n} frames sequentially...")
        _sb = getattr(args, '_session_bayer', None)
        _pgr = getattr(args, 'pre_gradient_removal', False)
        _ca_shifts = (_measure_session_ca(lights, args)
                      if getattr(args, 'ca_correction', False) else None)
        if _ca_shifts is not None:
            safe_print(f"  CA correction: session-constant shifts "
                       f"R={_fmt_ca(_ca_shifts.get(0))} B={_fmt_ca(_ca_shifts.get(2))} "
                       f"(measured once, applied per frame)")
        for i, f in tqdm(enumerate(lights), total=n,
                         desc="  Processing", unit="frame",
                         disable=args.verbose):
            result = _process_single_frame(
                f.path, f.header, masters, args.debayer_method, args.white_balance,
                ca_correction=getattr(args, 'ca_correction', False),
                cosmic_ray_rejection=getattr(args, 'cosmic_ray_rejection', False),
                advanced_metrics=getattr(args, 'advanced_metrics', True),
                session_bayer=_sb,
                pre_gradient_removal=_pgr,
                ca_shifts=_ca_shifts,
                trail_reject=getattr(args, 'trail_reject', False))
            _wv = _get_webview()
            _wv.progress('Processing frames', i + 1, n)
            if result.get('error'):
                f.accepted = False
                f.metrics = {'error': result['error']}
                rejected_reasons[f.path] = result['error']
                stats.add_error(f.path, result['error'])
                if args.verbose:
                    print(f'  REJECT {os.path.basename(f.path)}: {result["error"]}')
            else:
                mem_rgb[i] = result['rgb']
                mem_lum[i] = result['lum']
                cached_lums[i] = result['lum']
                f.metrics = result['metrics']
                _wv.frame_metrics(os.path.basename(f.path), f.metrics)
                _publish_frame_thumb(_wv, args, os.path.basename(f.path),
                                     result['rgb'], _wv_thumb_count)
                _accum(result.get('timings'))
                if args.verbose:
                    m = f.metrics
                    print(f'    {os.path.basename(f.path)}: '
                          f'score={m["score"]:.0f}  SNR={m["snr"]:.1f}  '
                          f'stars={m["star_count"]}  FWHM={m.get("fwhm",0):.1f}  '
                          f'sharpness={m.get("sharpness",0):.0f}')
                    print(f'      bg={m.get("background",0):.1f}  '
                          f'noise={m.get("noise",0):.2f}  '
                          f'brightness={m.get("brightness",0):.1f}  '
                          f'contrast={m.get("contrast",0):.1f}  '
                          f'dynamic_range={m.get("dynamic_range",0):.0f}')
        mem_rgb.flush()
        mem_lum.flush()

    _print_step_breakdown(_step_totals, _step_frames, _worker_count_used)


_STEP_ORDER = ('load', 'calibrate', 'debayer', 'hotpix', 'vignette', 'white_balance',
              'ca_correction', 'cosmic_ray_rejection', 'lum_recompute',
              'pre_gradient_removal', 'validate', 'quality',
              'patch_scores', 'memmap_write', 'final_flush')
_STEP_LABELS = {
    'load': 'Load (disk read)',
    'calibrate': 'Calibrate (bias/dark/flat)',
    'debayer': 'Debayer',
    'hotpix': 'Hot-pixel removal',
    'vignette': 'Vignette map (--vignette-map)',
    'white_balance': 'White balance',
    'ca_correction': 'CA correction (--ca-correction)',
    'cosmic_ray_rejection': 'Cosmic-ray rejection (lacosmic)',
    'lum_recompute': 'Luminance recompute',
    'pre_gradient_removal': 'Pre-gradient removal',
    'validate': 'Validate',
    'quality': 'Quality metrics (star detect, FWHM)',
    'patch_scores': 'Patch quality scores (for Phase 2)',
    'memmap_write': 'Memmap write',
    'final_flush': 'Final memmap flush',
}


def _print_step_breakdown(step_totals: Dict[str, float], n_frames: int,
                          worker_count: int = 1) -> None:
    """Print which Phase-1 sub-step actually dominates "Quality+Load".

    Values are summed per-frame seconds across however many workers ran in
    parallel, so the raw total looks alarming (can exceed the actual wall
    clock many times over) — an "est. wall-clock" column divides by the actual
    worker count used for this run to give the real contribution. On the GPU
    dispatch path this is best-effort: deferred quality metrics run on a
    separate CPU pool and are not attributable to a single frame, so they are
    omitted here.
    """
    if not step_totals or n_frames == 0:
        return
    total = sum(step_totals.values())
    if total <= 0:
        return
    w = max(worker_count, 1)
    # final_flush is one call in the main process after all workers finish —
    # already wall-clock, not summed-across-workers like everything else.
    _not_parallel = {'final_flush'}
    safe_print(f"\n  Quality+Load breakdown ({n_frames} frames, {w} workers):")
    safe_print(f"    {'':<38} {'sum (all workers)':>18}  {'est. wall-clock':>16}  {'':>7}")
    keys = list(_STEP_ORDER) + [k for k in step_totals if k not in _STEP_ORDER]
    for key in keys:
        t = step_totals.get(key)
        if t is None:
            continue
        label = _STEP_LABELS.get(key, key)
        wall = t if key in _not_parallel else t / w
        safe_print(f"    {label:<38} {format_time(t):>18}  {format_time(wall):>16}  "
                   f"({t / total * 100:4.1f}%)  [{t / n_frames * 1000:5.1f} ms/frame]")


def reload_accepted_frames(
    final: List[FrameInfo],
    final_indices: List[int],
    masters: Dict[str, Optional[np.ndarray]],
    args: argparse.Namespace,
    mem_rgb: np.ndarray,
    mem_lum: np.ndarray,
    cached_lums: list,
    mm_rgb_path: str = '',
    mm_lum_path: str = '',
    rgb_shape: tuple = (),
    lum_shape: tuple = (),
) -> None:
    """Re-load and calibrate accepted frames into memmaps after a checkpoint restore.

    Skips quality analysis — metrics were already restored from the checkpoint JSON.
    Mirrors execute_frame_processing: uses ProcessPoolExecutor when args.parallel != 1
    (bypasses the GIL for debayering), and adds an I/O prefetch pool on the thread
    path so compute threads are never idle waiting for FITS reads.
    """
    n = len(final)
    gpu = get_gpu()

    _build_flat_norm(masters, final)

    _ca  = getattr(args, 'ca_correction', False)
    _cr  = getattr(args, 'cosmic_ray_rejection', False)
    _sb  = getattr(args, '_session_bayer', None)
    _pgr = getattr(args, 'pre_gradient_removal', False)
    _tr  = getattr(args, 'trail_reject', False)
    _ca_shifts = _measure_session_ca(final, args) if _ca else None
    if _ca_shifts is not None:
        safe_print(f"  CA correction: session-constant shifts "
                   f"R={_fmt_ca(_ca_shifts.get(0))} B={_fmt_ca(_ca_shifts.get(2))}")

    def _worker_count_cap(n_workers: int) -> int:
        try:
            import psutil
            avail_mb = psutil.virtual_memory().available / 1e6
            H_f, W_f = mem_rgb.shape[1], mem_rgb.shape[2]
            bayer_mb = H_f * W_f * 4 / 1e6
            rgb_mb   = H_f * W_f * 3 * 4 / 1e6
            per_worker_mb = bayer_mb * 4 + rgb_mb + 200
            safe = max(1, int(avail_mb / per_worker_mb))
            if safe < n_workers:
                safe_print(f"  NOTE: limiting reload workers {n_workers}→{safe} "
                           f"(avail RAM {avail_mb:.0f} MB, ~{per_worker_mb:.0f} MB/worker)")
                return safe
        except Exception:
            pass
        return n_workers

    use_process_pool = (getattr(args, 'parallel', 1) != 1
                        and not gpu.active
                        and n >= 4
                        and bool(mm_rgb_path and mm_lum_path and rgb_shape and lum_shape))

    n_failed = 0

    if use_process_pool:
        workers = args.parallel if args.parallel > 0 else min(os.cpu_count() or 4, n)
        workers = _worker_count_cap(workers)

        safe_print(f"  Reloading {n} accepted frames ({workers} workers, "
                   f"quality analysis skipped)...")

        shm_blocks: list = []
        shm_specs: Dict[str, tuple] = {}
        for name, arr in masters.items():
            if arr is None or name.startswith('_shm_') or not isinstance(arr, np.ndarray):
                continue
            arr_c = np.ascontiguousarray(arr)
            shm = SharedMemory(create=True, size=arr_c.nbytes)
            shm_arr = np.ndarray(arr_c.shape, dtype=arr_c.dtype, buffer=shm.buf)
            shm_arr[:] = arr_c
            shm_blocks.append(shm)
            shm_specs[name] = (shm.name, arr_c.dtype.str, arr_c.shape)

        tasks = [(final[i].path, final_indices[i], args.debayer_method, args.white_balance,
                  mm_rgb_path, mm_lum_path, rgb_shape, lum_shape,
                  _ca, _cr, False, _sb, _pgr, True, _ca_shifts)
                 for i in range(n)]

        _orig_to_j = {orig: j for j, orig in enumerate(final_indices)}

        try:
            with ProcessPoolExecutor(max_workers=workers,
                                     initializer=_init_worker_shm,
                                     initargs=(shm_specs, _tr)) as pool:
                futures = {pool.submit(_parallel_frame_worker, t): t[1] for t in tasks}
                for future in tqdm(as_completed(futures), total=n,
                                   desc="  Reloading", unit="frame",
                                   disable=args.verbose):
                    orig_idx, _, error, _ = future.result()
                    if error:
                        j = _orig_to_j.get(orig_idx, -1)
                        fname = (os.path.basename(final[j].path)
                                 if j >= 0 else str(orig_idx))
                        safe_print(f"  WARNING: Could not reload {fname}: {error}")
                        n_failed += 1
                    elif args.verbose:
                        j = _orig_to_j.get(orig_idx, -1)
                        if j >= 0:
                            safe_print(f"    Reloaded: {os.path.basename(final[j].path)}")
        finally:
            for shm in shm_blocks:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass

        mem_rgb.flush()
        mem_lum.flush()

    else:
        # Thread pool with I/O prefetch — mirrors execute_frame_processing thread path.
        # Reload is I/O-dominated (FITS reads >> debayer time), so allow 2× cpu_count
        # threads: threads blocked on disk let others continue, saturating both disk
        # and CPU rather than one or the other.
        if gpu.active:
            n_workers = min(gpu.max_gpu_workers(Config.GPU_PHASE1_WORKER_MB,
                                                Config.GPU_VRAM_RESERVE_MB), n)
        else:
            n_workers = min((os.cpu_count() or 4) * 2, 32, n)
        n_workers = _worker_count_cap(n_workers)

        safe_print(f"  Reloading {n} accepted frames ({n_workers} threads, "
                   f"quality analysis skipped)...")

        # Match I/O prefetch threads to compute threads so compute is never
        # idle waiting on file reads.
        _io_pool = ThreadPoolExecutor(max_workers=min(n_workers, n))
        _load_futures = {final_indices[j]: _io_pool.submit(load_frame, final[j].path)
                         for j in range(n)}

        def _reload_one(j: int, f: FrameInfo, orig_idx: int):
            try:
                preloaded = _load_futures[orig_idx].result()
            except Exception as e:
                return j, orig_idx, None, None, f'load error: {e}'
            with gpu.stream_context():
                result = _process_single_frame(
                    f.path, f.header, masters, args.debayer_method, args.white_balance,
                    ca_correction=_ca,
                    cosmic_ray_rejection=_cr,
                    skip_quality=True,
                    session_bayer=_sb,
                    pre_gradient_removal=_pgr,
                    preloaded_data=preloaded,
                    ca_shifts=_ca_shifts,
                    trail_reject=_tr)
            if result.get('error'):
                return j, orig_idx, None, None, result['error']
            return j, orig_idx, result['rgb'], result['lum'], None

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_reload_one, j, f, orig_idx): j
                       for j, (f, orig_idx) in enumerate(zip(final, final_indices))}
            for future in tqdm(as_completed(futures), total=n,
                               desc="  Reloading", unit="frame",
                               disable=args.verbose):
                j, orig_idx, rgb, lum, error = future.result()
                if error:
                    safe_print(f"  WARNING: Could not reload "
                               f"{os.path.basename(final[j].path)}: {error}")
                    n_failed += 1
                else:
                    mem_rgb[orig_idx] = rgb
                    mem_lum[orig_idx] = lum
                    cached_lums[orig_idx] = lum
                    if args.verbose:
                        safe_print(f"    Reloaded: {os.path.basename(final[j].path)}")

        _io_pool.shutdown(wait=False)
        mem_rgb.flush()
        mem_lum.flush()

    if n_failed > 0:
        safe_print(f"  WARNING: {n_failed}/{n} frames failed to reload")
        if n_failed == n:
            raise RuntimeError("All frames failed to reload — cannot proceed "
                               "(likely out of memory)")

def quality_gate(
    lights: List[FrameInfo],
    args: argparse.Namespace,
    rejected_reasons: dict,
    stats: ProcessingStats,
) -> List[FrameInfo]:
    """Apply hard-limit, statistical-outlier, and percentile quality filters.

    Mutates lights[*].accepted and rejected_reasons in-place.
    Updates stats.accepted_frames / rejected_frames.
    Returns the list of accepted FrameInfo objects.
    """
    n = len(lights)

    # Hard-limit rejection
    accepted = []
    for f in lights:
        if not f.accepted or not f.metrics or 'score' not in f.metrics:
            continue
        m = f.metrics
        reject_reason = None
        if m['star_count'] == 0:
            reject_reason = "no stars detected"
        elif m['snr'] < 0.5:
            reject_reason = f"extremely low SNR ({m['snr']:.2f} < 0.5)"
        elif m['contrast'] < 2.0:
            reject_reason = f"extremely low contrast ({m['contrast']:.1f} < 2.0)"
        elif m['dynamic_range'] < 20:
            reject_reason = f"extremely low dynamic range ({m['dynamic_range']:.1f} < 20)"
        # NOTE: the old "excessive noise" hard limit (noise > brightness*0.8)
        # compared a robust noise *sigma* to the background *median* — a
        # dimensionless-mismatch that fires on well-calibrated frames: dark
        # subtraction lowers the pedestal (brightness) while a light-pollution
        # gradient inflates the whole-frame MAD (noise), so good frames with
        # healthy SNR get rejected wholesale. Genuinely noise-dominated frames
        # are already caught by the snr < 0.5 check above and by the relative
        # SNR outlier test in the statistical stage below (SNR = signal / noise,
        # so an abnormally noisy frame shows up as a low-SNR outlier).
        if reject_reason:
            f.accepted = False
            rejected_reasons[f.path] = reject_reason
            if args.verbose:
                print(f'  REJECT {os.path.basename(f.path)}: {reject_reason}')
        else:
            accepted.append(f)

    # Statistical outlier detection
    if len(accepted) > 3:
        snrs      = np.array([f.metrics['snr']         for f in accepted])
        star_cnts = np.array([f.metrics['star_count']  for f in accepted])
        contrasts = np.array([f.metrics['contrast']    for f in accepted])

        def _is_inlier(values, threshold=2.5):
            if len(values) < 3:
                return np.ones(len(values), dtype=bool)
            m, s = np.mean(values), np.std(values)
            if s < 1e-6:
                # All values identical — everyone is an inlier.
                # Always return an array; returning a Python bool causes
                # (~bool).astype(int) to raise AttributeError.
                return np.ones(len(values), dtype=bool)
            return np.abs((values - m) / s) < threshold

        snr_ok   = _is_inlier(snrs)
        star_ok  = _is_inlier(star_cnts)
        cont_ok  = _is_inlier(contrasts)
        n_outlier = (~snr_ok).astype(int) + (~star_ok).astype(int) + (~cont_ok).astype(int)
        for i, f in enumerate(accepted):
            if n_outlier[i] >= 2:
                parts = []
                if not snr_ok[i]:
                    parts.append(f"SNR={f.metrics['snr']:.1f}")
                if not star_ok[i]:
                    parts.append(f"stars={f.metrics['star_count']}")
                if not cont_ok[i]:
                    parts.append(f"contrast={f.metrics['contrast']:.1f}")
                rejected_reasons[f.path] = "statistical outlier: " + ", ".join(parts)
                f.accepted = False
            else:
                f.accepted = True

    # Relative quality threshold: reject frames whose score falls more than
    # quality_threshold% below a robust reference score.
    #
    # Reference is the 90th-percentile score rather than the single maximum.
    # The quality score formula is multiplicative with a star_factor that spans
    # 0.01–1.0 (100× range), so one frame with unusually good star detection
    # would set an unreachable threshold if we used the raw maximum.  The 90th
    # percentile is much more robust while still reflecting the top tier of the
    # session.
    if args.quality_filter and accepted:
        valid = [f for f in accepted if f.accepted]
        if valid:
            scores = np.array([f.metrics['score'] for f in valid])
            if len(scores) >= 10:
                ref_score = float(np.percentile(scores, 90))
            else:
                ref_score = float(scores.max())
            if ref_score > 1e-6:
                min_score = ref_score * (1.0 - args.quality_threshold / 100.0)
                for f in valid:
                    if f.metrics['score'] < min_score:
                        f.accepted = False
                        rejected_reasons[f.path] = (
                            f'score {f.metrics["score"]:.1f} < {min_score:.1f} '
                            f'({args.quality_threshold:.0f}% below ref {ref_score:.1f})'
                        )

    # Soft ellipticity check — warns but does NOT reject frames
    max_ellipticity = getattr(args, 'max_ellipticity', 0.5)
    if max_ellipticity > 0:
        for f in [fr for fr in lights if fr.accepted and fr.metrics]:
            ellip = f.metrics.get('ellipticity', 0.0)
            if ellip > max_ellipticity:
                print(f"  WARNING: {os.path.basename(f.path)}: high star ellipticity "
                      f"({ellip:.3f} > {max_ellipticity:.2f}) — tracking error suspected")

    final = [f for f in lights if f.accepted]
    stats.accepted_frames = len(final)
    stats.rejected_frames = n - len(final)

    if args.verbose:
        print_quality_table(lights, show_all=len(lights) <= 50)
    safe_print(f"  ✓ Accepted: {len(final)}/{n} ({len(final)/n*100:.1f}%)")
    if stats.rejected_frames > 0:
        reason_counts: dict = {}
        for reason in rejected_reasons.values():
            if 'score' in reason:
                cat = 'Below quality threshold'
            elif 'outlier' in reason:
                cat = 'Statistical outlier'
            elif any(k in reason for k in ('brightness', 'contrast', 'dynamic', 'noise')):
                cat = 'Poor quality'
            elif 'star' in reason:
                cat = 'No stars detected'
            elif 'load' in reason or 'empty' in reason:
                cat = 'Load/data errors'
            else:
                cat = 'Other'
            reason_counts[cat] = reason_counts.get(cat, 0) + 1
        safe_print(f"  ✗ Rejected: {stats.rejected_frames} "
                   f"({', '.join(f'{c}: {v}' for c, v in reason_counts.items())})")

    return final
