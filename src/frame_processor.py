"""Frame-level processing: calibration, debayering, parallel dispatch, quality gating."""
from __future__ import annotations

import argparse
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

try:
    from multiprocessing.shared_memory import SharedMemory as _SharedMemory
    _HAS_SHM = True
except ImportError:
    _HAS_SHM = False

import numpy as np

from src.gpu_context import get_gpu
from src.models import Config, FrameInfo, ProcessingStats
from src.utils import safe_print, print_quality_table
from src.io_fits import load_fits
from src.debayer import (debayer, remove_hot_pixels_bayer, apply_hot_pixel_map_bayer,
                         remove_hot_pixels_rgb, white_balance_grayworld, white_balance_whitepatch,
                         correct_chromatic_aberration)
from src.quality import validate_image_data, compute_quality_metrics
from src.stacking import lacosmic_reject

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable


def _process_single_frame(path: str, header: dict, masters: Dict[str, Optional[np.ndarray]],
                          debayer_method: str, white_balance: str,
                          ca_correction: bool = False,
                          cosmic_ray_rejection: bool = False) -> Dict:
    """Process one frame: load, calibrate, debayer, hot-pixel, quality.

    Returns dict with keys: 'rgb', 'lum', 'metrics', 'error'.
    Used by both sequential and parallel paths.
    """
    try:
        data, hdr = load_fits(path)
    except Exception as e:
        return {'error': f'load error: {e}'}
    if data is None or data.size == 0:
        return {'error': 'empty data array'}

    # Calibration — preserve negative noise through bias/dark subtraction,
    # clip only once after all steps to avoid cumulative truncation of shadow detail
    try:
        bias_arr = masters.get('bias') if (masters.get('bias') is not None
                                           and masters['bias'].shape == data.shape) else None
        if bias_arr is not None:
            data = data - bias_arr
        if masters.get('dark') is not None and masters['dark'].shape == data.shape:
            dark_arr = masters['dark']
            dark_current = dark_arr - (bias_arr if bias_arr is not None else 0.0)
            dark_exptime = masters.get('dark_exptime') or None
            light_exptime = float(hdr.get('EXPTIME', 0) or 0) or None
            if dark_exptime and light_exptime and dark_exptime > 0:
                dark_scale = light_exptime / dark_exptime
            else:
                dark_scale = 1.0
            data = data - dark_current * dark_scale
        if masters.get('flat_norm') is not None and masters['flat_norm'].shape == data.shape:
            data = data / masters['flat_norm']
        elif masters.get('flat') is not None and masters['flat'].shape == data.shape:
            flat = masters['flat']
            med = np.median(flat)
            if med > 1e-6:
                flat_norm = np.clip(flat / med, 0.4, 2.5)
                data = data / flat_norm
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
            bayer = hdr.get('BAYERPAT', hdr.get('COLORTYP', 'RGGB'))
            rgb = debayer(data, pattern=bayer, method=debayer_method)
        else:
            rgb = data
    except Exception as e:
        return {'error': f'debayering error: {e}'}

    # Hot pixel removal (single-channel detection, 3x faster)
    try:
        if rgb.ndim != 3 or rgb.shape[2] < 1:
            return {'error': f'Invalid RGB shape: {rgb.shape}'}
        rgb = remove_hot_pixels_rgb(rgb)
    except Exception as e:
        return {'error': f'hot pixel removal error: {e}'}

    # White balance
    if white_balance == 'grayworld':
        rgb = white_balance_grayworld(rgb)
    elif white_balance == 'whitepatch':
        rgb = white_balance_whitepatch(rgb)

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

    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    is_valid, validation_error = validate_image_data(lum, os.path.basename(path))
    if not is_valid:
        return {'error': f'validation failed: {validation_error}'}

    metrics = compute_quality_metrics(lum)
    return {'rgb': rgb, 'lum': lum, 'metrics': metrics, 'error': None}


# Module-level state for parallel workers (must be module-level for pickling)
_worker_masters: Dict[str, Optional[np.ndarray]] = {}
_worker_shm_handles: Dict[str, object] = {}  # keep SharedMemory handles alive in workers


def _init_worker(master_paths: Dict[str, str]):
    """Initializer for pool workers — load master calibration arrays from disk."""
    global _worker_masters
    _worker_masters = {}
    for name, p in master_paths.items():
        _worker_masters[name] = np.load(p)


def _init_worker_shm(shm_info: Dict[str, tuple], scalar_masters: Dict[str, object]):
    """Initializer for pool workers — attach to shared-memory blocks for arrays.

    shm_info maps name → (shm_name, shape, dtype_str).
    scalar_masters carries 0-d values (e.g. dark_exptime) that cannot go through shm.
    """
    global _worker_masters, _worker_shm_handles
    _worker_masters = dict(scalar_masters)
    _worker_shm_handles = {}
    for name, (shm_name, shape, dtype_str) in shm_info.items():
        shm = _SharedMemory(name=shm_name)
        _worker_shm_handles[name] = shm          # prevent GC / handle close
        _worker_masters[name] = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)


def _parallel_frame_worker(args_tuple):
    """Worker function for ProcessPoolExecutor. Must be module-level for pickling."""
    path, frame_idx, debayer_method, white_balance, mm_rgb_path, mm_lum_path, rgb_shape, lum_shape, ca_correction, cosmic_ray_rejection = args_tuple
    global _worker_masters
    result = _process_single_frame(path, {}, _worker_masters, debayer_method, white_balance,
                                   ca_correction=ca_correction,
                                   cosmic_ray_rejection=cosmic_ray_rejection)
    if result.get('error'):
        return (frame_idx, None, result['error'])

    metrics_clean = dict(result['metrics'])

    try:
        mem_rgb = np.memmap(mm_rgb_path, dtype='float32', mode='r+', shape=rgb_shape)
        mem_lum = np.memmap(mm_lum_path, dtype='float32', mode='r+', shape=lum_shape)
        mem_rgb[frame_idx] = result['rgb']
        mem_lum[frame_idx] = result['lum']
        mem_rgb.flush()
        mem_lum.flush()
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
    # Precompute flat_norm once — saves an O(H×W) division + clip on every frame.
    if masters.get('flat') is not None and masters.get('flat_norm') is None:
        _flat = masters['flat']
        _med = float(np.median(_flat))
        if _med > 1e-6:
            masters['flat_norm'] = np.clip(_flat / _med, 0.4, 2.5).astype(np.float32)

    n = len(lights)
    use_process_pool = (getattr(args, 'parallel', 1) != 1
                        and not get_gpu().active
                        and n >= 4)

    if use_process_pool:
        workers = args.parallel if args.parallel > 0 else min(os.cpu_count() or 4, n, 8)
        print(f"  Processing {n} frames in parallel ({workers} workers)...")

        # Share master calibration arrays via shared memory (zero disk I/O).
        # If SharedMemory is unavailable or any allocation fails, fall back to
        # the original temporary-.npy approach for the entire batch.
        shm_list: list = []
        shm_info: Dict[str, tuple] = {}
        scalar_masters: Dict[str, object] = {}
        use_shm = _HAS_SHM

        if use_shm:
            for name, val in masters.items():
                if val is None:
                    continue
                arr = np.asarray(val)
                if arr.ndim == 0:
                    scalar_masters[name] = val
                    continue
                try:
                    shm = _SharedMemory(create=True, size=max(arr.nbytes, 1))
                    shm_list.append(shm)
                    np.copyto(np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf), arr)
                    shm_info[name] = (shm.name, arr.shape, arr.dtype.str)
                except Exception:
                    use_shm = False
                    for s in shm_list:
                        try:
                            s.close()
                            s.unlink()
                        except Exception:
                            pass
                    shm_list.clear()
                    shm_info.clear()
                    break

        if not use_shm:
            master_paths: Dict[str, str] = {}
            for name, val in masters.items():
                if val is not None:
                    p = os.path.join(tempfile.gettempdir(),
                                     f'master_{name}_{os.getpid()}.npy')
                    np.save(p, np.asarray(val))
                    master_paths[name] = p

        _ca = getattr(args, 'ca_correction', False)
        _cr = getattr(args, 'cosmic_ray_rejection', False)
        tasks = [(lights[i].path, i, args.debayer_method, args.white_balance,
                  mm_rgb_path, mm_lum_path, rgb_shape, lum_shape, _ca, _cr)
                 for i in range(n)]

        if use_shm:
            _initializer, _initargs = _init_worker_shm, (shm_info, scalar_masters)
        else:
            _initializer, _initargs = _init_worker, (master_paths,)

        try:
            with ProcessPoolExecutor(max_workers=workers,
                                     initializer=_initializer,
                                     initargs=_initargs) as pool:
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
        finally:
            for shm in shm_list:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass
            if not use_shm:
                for p in master_paths.values():
                    try:
                        os.remove(p)
                    except Exception:
                        pass

    elif n >= 2:
        gpu = get_gpu()
        if gpu.active:
            n_workers = min(gpu.max_gpu_workers(Config.GPU_PHASE1_WORKER_MB,
                                                Config.GPU_VRAM_RESERVE_MB), n)
        else:
            n_workers = min(os.cpu_count() or 4, n)
        safe_print(f"  Processing {n} frames with {n_workers} threads"
                   f"{' (GPU)' if gpu.active else ''}...")

        def _thread_process_frame(i, f):
            with gpu.stream_context():
                result = _process_single_frame(
                    f.path, f.header, masters, args.debayer_method, args.white_balance,
                    ca_correction=getattr(args, 'ca_correction', False),
                    cosmic_ray_rejection=getattr(args, 'cosmic_ray_rejection', False))
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
        mem_rgb.flush()
        mem_lum.flush()

    else:
        print(f"  Processing {n} frames sequentially...")
        for i, f in tqdm(enumerate(lights), total=n,
                         desc="  Processing", unit="frame",
                         disable=args.verbose):
            result = _process_single_frame(
                f.path, f.header, masters, args.debayer_method, args.white_balance,
                ca_correction=getattr(args, 'ca_correction', False),
                cosmic_ray_rejection=getattr(args, 'cosmic_ray_rejection', False))
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
