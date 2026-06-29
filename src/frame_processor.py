"""Frame-level processing: calibration, debayering, parallel dispatch, quality gating."""
from __future__ import annotations

import argparse
import os
import queue
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.gpu_context import get_gpu
from src.models import Config, FrameInfo, ProcessingStats
from src.utils import safe_print, print_quality_table
from src.io_fits import load_fits, load_frame
from src.debayer import (debayer, green_equalize, remove_hot_pixels_bayer,
                         apply_hot_pixel_map_bayer,
                         remove_hot_pixels_rgb, remove_hot_pixels_rgb_with_lum,
                         white_balance_grayworld, white_balance_whitepatch,
                         correct_chromatic_aberration)
from src.quality import validate_image_data, compute_quality_metrics
from src.stacking import lacosmic_reject

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable


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
                          pre_gradient_removal: bool = False) -> Dict[str, Any]:
    """Process one frame: load, calibrate, debayer, hot-pixel, quality.

    Returns dict with keys: 'rgb', 'lum', 'metrics', 'error'.
    Used by both sequential and parallel paths.
    """
    try:
        if preloaded_data is not None:
            data, hdr = preloaded_data
        else:
            data, hdr = load_frame(path)
    except Exception as e:
        return {'error': f'load error: {e}'}
    if data is None or data.size == 0:
        return {'error': 'empty data array'}

    # Calibration — preserve negative noise through bias/dark subtraction,
    # clip only once after all steps to avoid cumulative truncation of shadow detail
    try:
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
                                   f"— ensure dark exposure matches lights for best results")
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
                                   f"pixels — possible sensor defects or bad flat")
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

    # Debayer
    try:
        if data.ndim == 2:
            bayer = hdr.get('BAYERPAT', hdr.get('COLORTYP', session_bayer or 'RGGB'))
            data = green_equalize(data, pattern=bayer)
            rgb = debayer(data, pattern=bayer, method=debayer_method)
        else:
            rgb = data
    except Exception as e:
        return {'error': f'debayering error: {e}'}

    # Hot pixel removal — returns (rgb_fixed, lum) to avoid recomputing luminance.
    try:
        if rgb.ndim != 3 or rgb.shape[2] < 1:
            return {'error': f'Invalid RGB shape: {rgb.shape}'}
        rgb, lum = remove_hot_pixels_rgb_with_lum(rgb)
    except Exception as e:
        return {'error': f'hot pixel removal error: {e}'}

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

    if ca_correction:
        try:
            rgb = correct_chromatic_aberration(rgb)
        except Exception:
            pass

    if cosmic_ray_rejection:
        try:
            rgb = lacosmic_reject(rgb)
        except Exception:
            pass

    # Recompute lum only when white balance or post-processing changed the image.
    try:
        if white_balance != 'none' or ca_correction or cosmic_ray_rejection:
            lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        else:
            lum = np.asarray(lum)  # ensure host numpy array
    except Exception as e:
        return {'error': f'luminance computation error: {e}'}

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

    is_valid, validation_error = validate_image_data(lum, os.path.basename(path))
    if not is_valid:
        return {'error': f'validation failed: {validation_error}'}

    metrics = {} if skip_quality else compute_quality_metrics(
        lum, quick=quick_quality, advanced_metrics=advanced_metrics)
    return {'rgb': rgb, 'lum': lum, 'metrics': metrics, 'error': None}


# Module-level state for parallel workers (must be module-level for pickling)
_worker_masters: Dict[str, Any] = {}
_warned_dark_scales: set = set()  # dedup dark-scale mismatch warnings across frames


def _init_worker_shm(shm_specs: Dict[str, tuple]) -> None:
    """Initializer for pool workers — attach to shared-memory calibration arrays.

    *shm_specs* maps master name → (shm_name, dtype_str, shape).  Workers
    attach (read-only view) without copying data or touching the filesystem.
    """
    global _worker_masters
    _worker_masters = {}
    for name, (shm_name, dtype_str, shape) in shm_specs.items():
        shm = SharedMemory(name=shm_name, create=False)
        arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        # Keep shm alive for the process lifetime; store both so we can close later.
        _worker_masters[name] = arr
        _worker_masters[f'_shm_{name}'] = shm


def _parallel_frame_worker(args_tuple: tuple) -> Tuple[int, Optional[dict], Optional[str]]:
    """Worker function for ProcessPoolExecutor. Must be module-level for pickling."""
    (path, frame_idx, debayer_method, white_balance,
     mm_rgb_path, mm_lum_path, rgb_shape, lum_shape,
     ca_correction, cosmic_ray_rejection, advanced_metrics, session_bayer,
     pre_gradient_removal, skip_quality) = args_tuple
    global _worker_masters
    result = _process_single_frame(path, {}, _worker_masters, debayer_method, white_balance,
                                   ca_correction=ca_correction,
                                   cosmic_ray_rejection=cosmic_ray_rejection,
                                   advanced_metrics=advanced_metrics,
                                   skip_quality=skip_quality,
                                   session_bayer=session_bayer,
                                   pre_gradient_removal=pre_gradient_removal)
    if result.get('error'):
        return (frame_idx, None, result['error'])

    metrics_clean = dict(result['metrics'])

    try:
        mem_rgb = np.memmap(mm_rgb_path, dtype='float32', mode='r+', shape=rgb_shape)
        mem_lum = np.memmap(mm_lum_path, dtype='float32', mode='r+', shape=lum_shape)
        mem_rgb[frame_idx] = result['rgb']
        mem_lum[frame_idx] = result['lum']
        # Flush deferred to main process after all workers complete — flushing
        # the entire memmap on every frame causes excessive concurrent I/O.
        del mem_rgb, mem_lum
    except Exception as e:
        return (frame_idx, None, f'memmap write error: {e}')

    return (frame_idx, metrics_clean, None)


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
    n = len(lights)
    use_process_pool = (getattr(args, 'parallel', 1) != 1
                        and not get_gpu().active
                        and n >= 4)

    # Pre-compute flat_norm once (with rotation correction) so workers don't redo it.
    _build_flat_norm(masters, lights)

    if use_process_pool:
        workers = args.parallel if args.parallel > 0 else min(os.cpu_count() or 4, n, 8)

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
        tasks = [(lights[i].path, i, args.debayer_method, args.white_balance,
                  mm_rgb_path, mm_lum_path, rgb_shape, lum_shape, _ca, _cr, _adv, _sb, _pgr, False)
                 for i in range(n)]

        try:
            with ProcessPoolExecutor(max_workers=workers,
                                     initializer=_init_worker_shm,
                                     initargs=(shm_specs,)) as pool:
                futures = {pool.submit(_parallel_frame_worker, t): t[1] for t in tasks}
                for future in tqdm(as_completed(futures), total=n,
                                   desc="  Processing", unit="frame",
                                   disable=args.verbose):
                    idx = futures[future]
                    frame_idx, metrics, error = future.result()
                    f = lights[frame_idx]
                    if error:
                        f.accepted = False
                        f.metrics = {'error': error}
                        rejected_reasons[f.path] = error
                        stats.add_error(f.path, error)
                        if args.verbose:
                            print(f'  REJECT {os.path.basename(f.path)}: {error}')
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
        mem_rgb.flush()
        mem_lum.flush()

    elif n >= 2:
        gpu = get_gpu()
        if gpu.active:
            n_workers = min(gpu.max_gpu_workers(Config.GPU_PHASE1_WORKER_MB,
                                                Config.GPU_VRAM_RESERVE_MB), n)
        else:
            n_workers = min(os.cpu_count() or 4, n)
        safe_print(f"  Processing {n} frames with {n_workers} threads"
                   f"{' (GPU)' if gpu.active else ''}...")

        # Prefetch FITS data in a dedicated I/O thread pool.  FITS reads release
        # the GIL (system calls), so these overlap with compute threads for free.
        # The I/O pool runs up to 4 concurrent reads regardless of n_workers so
        # we don't swamp the disk with too many concurrent seeks.
        _adv = getattr(args, 'advanced_metrics', True)
        _sb = getattr(args, '_session_bayer', None)
        _io_workers = min(4, n)
        _io_pool = ThreadPoolExecutor(max_workers=_io_workers)
        _load_futures = {i: _io_pool.submit(load_frame, f.path)
                         for i, f in enumerate(lights)}

        _pgr = getattr(args, 'pre_gradient_removal', False)

        def _thread_process_frame(i, f):
            try:
                preloaded = _load_futures[i].result()
            except Exception as e:
                return i, None, f'load error: {e}', None
            with gpu.stream_context():
                result = _process_single_frame(
                    f.path, f.header, masters, args.debayer_method, args.white_balance,
                    ca_correction=getattr(args, 'ca_correction', False),
                    cosmic_ray_rejection=getattr(args, 'cosmic_ray_rejection', False),
                    advanced_metrics=_adv,
                    preloaded_data=preloaded,
                    session_bayer=_sb,
                    pre_gradient_removal=_pgr)
            if result.get('error'):
                return i, None, result['error'], None
            mem_rgb[i] = result['rgb']
            mem_lum[i] = result['lum']
            return i, result['metrics'], None, result['lum']

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_thread_process_frame, i, f): i
                       for i, f in enumerate(lights)}
            for future in tqdm(as_completed(futures), total=n,
                               desc="  Processing", unit="frame",
                               disable=args.verbose):
                i, metrics, error, lum_arr = future.result()
                f = lights[i]
                if error:
                    f.accepted = False
                    f.metrics = {'error': error}
                    rejected_reasons[f.path] = error
                    stats.add_error(f.path, error)
                    if args.verbose:
                        safe_print(f'  REJECT {os.path.basename(f.path)}: {error}')
                else:
                    f.metrics = metrics
                    cached_lums[i] = lum_arr
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
        _io_pool.shutdown(wait=False)
        mem_rgb.flush()
        mem_lum.flush()

    else:
        print(f"  Processing {n} frames sequentially...")
        _sb = getattr(args, '_session_bayer', None)
        _pgr = getattr(args, 'pre_gradient_removal', False)
        for i, f in tqdm(enumerate(lights), total=n,
                         desc="  Processing", unit="frame",
                         disable=args.verbose):
            result = _process_single_frame(
                f.path, f.header, masters, args.debayer_method, args.white_balance,
                ca_correction=getattr(args, 'ca_correction', False),
                cosmic_ray_rejection=getattr(args, 'cosmic_ray_rejection', False),
                advanced_metrics=getattr(args, 'advanced_metrics', True),
                session_bayer=_sb,
                pre_gradient_removal=_pgr)
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
                  _ca, _cr, False, _sb, _pgr, True)
                 for i in range(n)]

        _orig_to_j = {orig: j for j, orig in enumerate(final_indices)}

        try:
            with ProcessPoolExecutor(max_workers=workers,
                                     initializer=_init_worker_shm,
                                     initargs=(shm_specs,)) as pool:
                futures = {pool.submit(_parallel_frame_worker, t): t[1] for t in tasks}
                for future in tqdm(as_completed(futures), total=n,
                                   desc="  Reloading", unit="frame",
                                   disable=args.verbose):
                    orig_idx, _, error = future.result()
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
                    preloaded_data=preloaded)
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
        elif m['noise'] > m['brightness'] * 0.8:
            reject_reason = f"excessive noise ({m['noise']:.1f} > {m['brightness']*0.8:.1f})"
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
