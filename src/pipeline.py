"""Core stacking pipeline: single frame processing, parallel workers, stack_target."""
from __future__ import annotations

import argparse
import os
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from src.gpu_context import get_gpu
from src.models import Config, FrameInfo, ProcessingStats
from src.utils import safe_print, print_phase, print_quality_table, format_time, get_memory_usage_mb
from src.io_fits import load_fits, save_preview_rgb, populate_fits_header
from src.debayer import (debayer, remove_hot_pixels_bayer, apply_hot_pixel_map_bayer,
                         remove_hot_pixels_rgb, white_balance_grayworld, white_balance_whitepatch,
                         correct_chromatic_aberration)
from src.quality import validate_image_data, compute_quality_metrics, generate_star_mask, _detect_stars_multi_fwhm
from src.registration import (calculate_shift, apply_transform, calc_common_crop,
                               detect_dither, match_stars_affine, HAS_SKIMAGE_TRANSFORM)
from src.stacking import (sigma_clip_combine, percentile_clip_combine, esd_combine,
                          _lanczos_resample_frame, drizzle_combine, lacosmic_reject)
from src.background import (apply_background_extraction, remove_sky_residual,
                            sky_floor_normalize, dynamic_background_extraction)
from src.denoising import (wavelet_denoise, adaptive_wavelet_denoise, nlm_denoise,
                           bilateral_denoise, local_normalize, reduce_chroma_noise,
                           estimate_denoise_strength, reduce_stars,
                           multiscale_local_contrast)
from src.psf_deconvolution import estimate_psf, make_synthetic_psf, richardson_lucy_deconvolve
from src.plate_solve import solve_plate

try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from photutils.detection import DAOStarFinder
except Exception:
    DAOStarFinder = None

try:
    from astropy.stats import sigma_clipped_stats
except Exception:
    sigma_clipped_stats = None

try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False


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
            # Correct dark subtraction: the master dark still contains its own bias
            # pedestal, so subtract bias-corrected dark current only, scaled to the
            # light frame's exposure time to avoid over/under-subtraction.
            dark_arr = masters['dark']
            dark_current = dark_arr - (bias_arr if bias_arr is not None else 0.0)
            dark_exptime = masters.get('dark_exptime') or None
            light_exptime = float(hdr.get('EXPTIME', 0) or 0) or None
            if dark_exptime and light_exptime and dark_exptime > 0:
                dark_scale = light_exptime / dark_exptime
            else:
                dark_scale = 1.0
            data = data - dark_current * dark_scale
        if masters.get('flat') is not None and masters['flat'].shape == data.shape:
            flat = masters['flat']
            med = np.median(flat)
            if med > 1e-6:
                flat_norm = np.clip(flat / med, 0.4, 2.5)
                data = data / flat_norm
        if not np.isfinite(data).all():
            return {'error': 'calibration produced non-finite values'}
        data = np.clip(data, 0, None)
        # Apply dark-derived hot pixel map on Bayer data (before debayering)
        if data.ndim == 2 and masters.get('hot_pixel_map') is not None:
            hot_map = masters['hot_pixel_map']
            if hot_map.shape == data.shape:
                data = apply_hot_pixel_map_bayer(data, hot_map)
        # Per-sub-channel Bayer hot pixel detection (catches hot pixels
        # not in the dark frame: intermittent defects, cosmic rays, etc.)
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

    # Ensure arrays are on CPU (GPU/CuPy paths return device arrays)
    gpu = get_gpu()
    rgb = gpu.to_host(rgb)

    # Chromatic aberration correction (per-channel sub-pixel alignment to green)
    if ca_correction:
        try:
            rgb = correct_chromatic_aberration(rgb)
        except Exception:
            pass  # non-critical; fall through with uncorrected rgb

    # Cosmic ray rejection (L.A.Cosmic per-channel Laplacian detection)
    if cosmic_ray_rejection:
        try:
            rgb = lacosmic_reject(rgb)
        except Exception:
            pass  # non-critical; fall through with un-cleaned rgb

    # Compute luminance & quality
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    is_valid, validation_error = validate_image_data(lum, os.path.basename(path))
    if not is_valid:
        return {'error': f'validation failed: {validation_error}'}

    metrics = compute_quality_metrics(lum)
    return {'rgb': rgb, 'lum': lum, 'metrics': metrics, 'error': None}


# Module-level state for parallel workers
_worker_masters: Dict[str, Optional[np.ndarray]] = {}


def _init_worker(master_paths: Dict[str, str]):
    """Initializer for pool workers — load master calibration arrays from disk."""
    global _worker_masters
    _worker_masters = {}
    for name, p in master_paths.items():
        _worker_masters[name] = np.load(p)


def _parallel_frame_worker(args_tuple):
    """Worker function for ProcessPoolExecutor. Must be module-level for pickling."""
    path, frame_idx, debayer_method, white_balance, mm_rgb_path, mm_lum_path, rgb_shape, lum_shape, ca_correction, cosmic_ray_rejection = args_tuple
    global _worker_masters
    result = _process_single_frame(path, {}, _worker_masters, debayer_method, white_balance,
                                   ca_correction=ca_correction,
                                   cosmic_ray_rejection=cosmic_ray_rejection)
    if result.get('error'):
        return (frame_idx, None, result['error'])

    rgb = result['rgb']
    lum = result['lum']
    metrics = result['metrics']
    # _star_sources is an astropy QTable which IS picklable; keep it so that
    # Phase 2 can use star positions for affine registration even on the
    # process-pool path (the prior stripping was incorrect).
    metrics_clean = dict(metrics)

    # Write processed data to shared memmap
    try:
        mem_rgb = np.memmap(mm_rgb_path, dtype='float32', mode='r+', shape=rgb_shape)
        mem_lum = np.memmap(mm_lum_path, dtype='float32', mode='r+', shape=lum_shape)
        mem_rgb[frame_idx] = rgb
        mem_lum[frame_idx] = lum
        mem_rgb.flush()
        mem_lum.flush()
        del mem_rgb, mem_lum
    except Exception as e:
        return (frame_idx, None, f'memmap write error: {e}')

    return (frame_idx, metrics_clean, None)


def stack_target(frames: List[FrameInfo], output_path: str, args: argparse.Namespace,
                 masters: Dict[str, Optional[np.ndarray]], stats: ProcessingStats):
    lights = [f for f in frames if f.type == 'light']
    if not lights:
        print('  No light frames found for target')
        return None

    stats.total_frames = len(lights)
    n = len(lights)

    # ======================================================================
    # PHASE 1: Process & Analyse (fused — each frame loaded only ONCE)
    # ======================================================================
    print_phase(1, "Processing & Quality Analysis")
    phase_start = time.time()

    # Probe first frame for dimensions
    first_data, first_hdr = load_fits(lights[0].path)
    if first_data.ndim == 2:
        first_rgb = debayer(first_data,
                            pattern=first_hdr.get('BAYERPAT', first_hdr.get('COLORTYP', 'RGGB')),
                            method=args.debayer_method)
        H_rgb, W_rgb, C = first_rgb.shape
    else:
        H_rgb, W_rgb = first_data.shape[:2]
        C = first_data.shape[2] if first_data.ndim == 3 else 1
    del first_data

    # Create memmaps for ALL frames (unaligned, white-balanced, calibrated)
    mm_rgb_path = os.path.join(tempfile.gettempdir(), f'stack_rgb_{os.getpid()}.dat')
    mm_lum_path = os.path.join(tempfile.gettempdir(), f'stack_lum_{os.getpid()}.dat')
    rgb_shape = (n, H_rgb, W_rgb, C)
    lum_shape = (n, H_rgb, W_rgb)
    mem_rgb = np.memmap(mm_rgb_path, dtype='float32', mode='w+', shape=rgb_shape)
    mem_lum = np.memmap(mm_lum_path, dtype='float32', mode='w+', shape=lum_shape)
    # Cache lum arrays from Phase 1 so Phase 2 can register without re-reading memmap
    cached_lums: list = [None] * n

    rejected_reasons = {}
    use_process_pool = (getattr(args, 'parallel', 1) != 1
                        and not get_gpu().active
                        and n >= 4)

    if use_process_pool:
        # --- ProcessPool path (-j N): separate processes, true parallelism,
        #     master arrays serialised to disk for IPC. ---
        workers = args.parallel if args.parallel > 0 else min(os.cpu_count() or 4, n, 8)
        print(f"  Processing {n} frames in parallel ({workers} workers)...")

        master_paths = {}
        for name, arr in masters.items():
            if arr is not None:
                p = os.path.join(tempfile.gettempdir(), f'master_{name}_{os.getpid()}.npy')
                np.save(p, arr)
                master_paths[name] = p

        _ca = getattr(args, 'ca_correction', False)
        _cr = getattr(args, 'cosmic_ray_rejection', False)
        tasks = [(lights[i].path, i, args.debayer_method, args.white_balance,
                  mm_rgb_path, mm_lum_path, rgb_shape, lum_shape, _ca, _cr)
                 for i in range(n)]

        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_init_worker,
                                 initargs=(master_paths,)) as pool:
            futures = {pool.submit(_parallel_frame_worker, t): t[1] for t in tasks}
            done_iter = tqdm(as_completed(futures), total=n,
                             desc="  Processing", unit="frame",
                             disable=args.verbose)
            for future in done_iter:
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

        # Cleanup master temp files
        for p in master_paths.values():
            try:
                os.remove(p)
            except Exception:
                pass

    elif n >= 2:
        # --- Thread-parallel path (default for both CPU and GPU) ---
        # ThreadPoolExecutor shares memory: no master serialization, no IPC
        # overhead, and _star_sources are preserved for affine registration.
        # GPU: VRAM-limited workers with per-thread CUDA streams.
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
        # --- Sequential path (single frame) ---
        print(f"  Processing {n} frames sequentially...")
        frame_iter = tqdm(enumerate(lights), total=n,
                          desc="  Processing", unit="frame",
                          disable=args.verbose)
        for i, f in frame_iter:
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

    # --- Quality gating ---
    accepted = []
    for f in lights:
        if not f.accepted or not f.metrics or 'score' not in f.metrics:
            continue
        metrics = f.metrics
        reject_reason = None
        if metrics['star_count'] < 3:
            reject_reason = f"insufficient stars ({metrics['star_count']} < 3)"
        elif metrics['snr'] < 0.5:
            reject_reason = f"extremely low SNR ({metrics['snr']:.2f} < 0.5)"
        elif metrics['contrast'] < 2.0:
            reject_reason = f"extremely low contrast ({metrics['contrast']:.1f} < 2.0)"
        elif metrics['dynamic_range'] < 20:
            reject_reason = f"extremely low dynamic range ({metrics['dynamic_range']:.1f} < 20)"
        elif metrics['noise'] > metrics['brightness'] * 0.8:
            reject_reason = f"excessive noise ({metrics['noise']:.1f} > {metrics['brightness']*0.8:.1f})"
        if reject_reason:
            f.accepted = False
            rejected_reasons[f.path] = reject_reason
            if args.verbose:
                print(f'  REJECT {os.path.basename(f.path)}: {reject_reason}')
        else:
            accepted.append(f)

    # Statistical outlier detection
    if len(accepted) > 3:
        snrs = np.array([f.metrics['snr'] for f in accepted])
        star_counts = np.array([f.metrics['star_count'] for f in accepted])
        contrasts = np.array([f.metrics['contrast'] for f in accepted])

        def reject_outliers(values, threshold=2.5):
            if len(values) < 3:
                return np.ones(len(values), dtype=bool)
            m, s = np.mean(values), np.std(values)
            if s < 1e-6:
                return np.ones(len(values), dtype=bool)
            return np.abs((values - m) / s) < threshold

        snr_ok = reject_outliers(snrs)
        star_ok = reject_outliers(star_counts)
        contrast_ok = reject_outliers(contrasts)
        outlier_count = (~snr_ok).astype(int) + (~star_ok).astype(int) + (~contrast_ok).astype(int)
        for i, f in enumerate(accepted):
            if outlier_count[i] >= 2:
                parts = []
                if not snr_ok[i]:
                    parts.append(f"SNR={f.metrics['snr']:.1f}")
                if not star_ok[i]:
                    parts.append(f"stars={f.metrics['star_count']}")
                if not contrast_ok[i]:
                    parts.append(f"contrast={f.metrics['contrast']:.1f}")
                reason = "statistical outlier: " + ", ".join(parts)
                rejected_reasons[f.path] = reason
                f.accepted = False
            else:
                f.accepted = True

    if args.quality_filter and accepted:
        valid = [f for f in accepted if f.accepted]
        if valid:
            scores = np.array([f.metrics['score'] for f in valid])
            pct = np.percentile(scores, args.quality_threshold)
            for f in valid:
                if f.metrics['score'] < pct:
                    f.accepted = False
                    rejected_reasons[f.path] = f'score {f.metrics["score"]:.1f} < {pct:.1f}'

    final = [f for f in lights if f.accepted]
    # Build index map: for each final frame, its original index in `lights`
    _lights_index = {id(f): i for i, f in enumerate(lights)}
    final_indices = [_lights_index[id(f)] for f in final]
    stats.accepted_frames = len(final)
    stats.rejected_frames = n - len(final)
    stats.quality_time = time.time() - phase_start

    if args.verbose:
        print_quality_table(lights, show_all=len(lights) <= 50)
    safe_print(f"  ✓ Accepted: {len(final)}/{n} ({len(final)/n*100:.1f}%)")
    if stats.rejected_frames > 0:
        reason_counts = {}
        for reason in rejected_reasons.values():
            if 'score' in reason:
                cat = 'Below quality threshold'
            elif 'outlier' in reason:
                cat = 'Statistical outlier'
            elif 'brightness' in reason or 'contrast' in reason or 'dynamic' in reason or 'noise' in reason:
                cat = 'Poor quality'
            elif 'star' in reason:
                cat = 'No stars detected'
            elif 'load' in reason or 'empty' in reason:
                cat = 'Load/data errors'
            else:
                cat = 'Other'
            reason_counts[cat] = reason_counts.get(cat, 0) + 1
        safe_print(f"  ✗ Rejected: {stats.rejected_frames} "
                   f"({', '.join(f'{c}: {n}' for c, n in reason_counts.items())})")

    if not final:
        print(f'\n  ERROR: All {n} frames rejected!')
        if rejected_reasons:
            print('  Rejection reasons:')
            for path, reason in list(rejected_reasons.items())[:10]:
                print(f'    - {os.path.basename(path)}: {reason}')
        # Cleanup memmaps
        del mem_rgb, mem_lum
        for p in (mm_rgb_path, mm_lum_path):
            try:
                os.remove(p)
            except Exception:
                pass
        return None

    # AI parameter advisor: runs after Phase 1 quality analysis so Claude has
    # the full quality statistics, and before Phase 2 so its recommendations
    # (stack method, rejection sigma, drizzle scale, denoise strength) take
    # effect in the subsequent phases.
    if getattr(args, 'ai_advisor', False):
        from src.ai_advisor import get_parameter_recommendations, apply_recommendations
        rec, explanation = get_parameter_recommendations(final, rejected_reasons, args)
        if rec is not None:
            print(f"\n  AI Advisor:\n  {explanation}")
            if rec.warnings:
                for w in rec.warnings:
                    safe_print(f"  ⚠  {w}")
            changes = apply_recommendations(rec, args)
            if changes:
                safe_print("  Applied recommendations:")
                for c in changes:
                    safe_print(f"    • {c}")
            else:
                safe_print("  Current settings look good — no changes applied.")

    # Heuristic auto-advisor: pure Python, no API key required.
    # Classifies the target from Phase 1 metrics and applies optimal settings
    # before Phase 2 so they take effect for registration, stacking, and post-
    # processing.
    if getattr(args, 'auto', False):
        from src.auto_settings import apply_auto_settings
        target_type, label, signals, changes = apply_auto_settings(final, args)
        print(f"\n  Auto Advisor: detected '{label}'")
        if signals:
            print(f"    median_filling={signals.get('median_filling', 0):.2f}  "
                  f"diffuse_excess={signals.get('diffuse_excess', 0):.2f}  "
                  f"peak_excess={signals.get('peak_excess', 0):.1f}  "
                  f"stars={signals.get('star_count', 0):.0f}  "
                  f"FWHM={signals.get('fwhm', 0):.1f}px  "
                  f"frames={signals.get('n_frames', 0)}")
        if changes:
            safe_print("  Applied auto settings:")
            for c in changes:
                safe_print(f"    * {c}")
        else:
            safe_print("  Current settings already optimal — no changes applied.")

    # ======================================================================
    # PHASE 2: Registration (reads from stored memmaps — no re-load)
    # ======================================================================
    print_phase(2, "Registration")
    phase_start = time.time()

    best = max(final, key=lambda x: x.metrics.get('score', 0))
    best_idx = lights.index(best)
    print(f"  Reference frame: {os.path.basename(best.path)} "
          f"(score={best.metrics.get('score', 0):.1f})")

    ref_lum = np.array(mem_lum[best_idx])  # copy from memmap
    H, W = ref_lum.shape
    ref_lum_std = np.std(ref_lum)

    if args.verbose:
        print(f'  Reference luminance: min={np.min(ref_lum):.1f}, max={np.max(ref_lum):.1f}, '
              f'mean={np.mean(ref_lum):.1f}, std={ref_lum_std:.1f}')

    # Get reference star positions for affine registration
    ref_stars = best.metrics.get('_star_sources')
    if ref_stars is None and HAS_SKIMAGE_TRANSFORM and not getattr(args, 'no_affine', False):
        safe_print("  ⚠ No stars detected in reference frame — affine (rotation) "
                   "registration disabled, falling back to translation only")

    shifts = [None] * len(final)
    transforms = [None] * len(final)  # For affine mode
    print(f"  Calculating shifts for {len(final)} frames...")

    def _register_one_frame(j, f, orig_idx):
        """Compute registration for a single frame (thread-safe, reads only)."""
        if orig_idx == best_idx or args.no_registration:
            return j, (0.0, 0.0), None

        with gpu.stream_context():
            lum = cached_lums[orig_idx] if cached_lums[orig_idx] is not None else np.array(mem_lum[orig_idx])

            # Try affine (rotation+translation) registration when stars are available.
            # Enabled by default when skimage is present; use --no-affine to disable.
            affine_tf = None
            use_affine = HAS_SKIMAGE_TRANSFORM and not getattr(args, 'no_affine', False)
            if use_affine:
                img_stars = f.metrics.get('_star_sources')
                sy, sx = calculate_shift(ref_lum, lum, verbose=False,
                                         skip_phase_cc=args.skip_phase_correlation)
                affine_tf = match_stars_affine(ref_stars, img_stars,
                                               initial_shift=(sy, sx))
                if affine_tf is not None:
                    tx = affine_tf.params[0, 2]
                    ty = affine_tf.params[1, 2]
                    return j, (ty, tx), affine_tf

            # Translation-only registration
            sy, sx = calculate_shift(
                ref_lum, lum, verbose=args.verbose,
                debug=args.debug_registration,
                frame_name=os.path.splitext(os.path.basename(f.path))[0],
                skip_phase_cc=args.skip_phase_correlation)
            if abs(sx) > 0.1 * W or abs(sy) > 0.1 * H:
                safe_print(f'Unrealistic shift {sx},{sy} for {f.path}, ignoring')
                sx, sy = 0.0, 0.0
            return j, (sy, sx), None

    gpu = get_gpu()
    if gpu.active:
        n_reg_workers = min(gpu.max_gpu_workers(Config.GPU_FFT_WORKER_MB,
                                                Config.GPU_VRAM_RESERVE_MB), len(final))
    else:
        n_reg_workers = min(os.cpu_count() or 4, len(final))
    with ThreadPoolExecutor(max_workers=n_reg_workers) as executor:
        futures = {
            executor.submit(_register_one_frame, j, f, orig_idx): j
            for j, (f, orig_idx) in enumerate(zip(final, final_indices))
        }
        for future in tqdm(as_completed(futures), total=len(final),
                           desc="  Registering", unit="frame",
                           disable=args.verbose):
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
                    mag = np.sqrt(sy**2 + sx**2)
                    safe_print(f'    {os.path.basename(f.path)}: shift=({sx:+.1f}, {sy:+.1f}) px, '
                               f'magnitude={mag:.2f} px')

    stats.registration_time = time.time() - phase_start
    del cached_lums  # free in-memory lum cache now that registration is complete

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

    # Shift pattern warnings
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

    # Resolve 'auto' and legacy --winsorize shorthand
    n_frames_for_auto = len(final)
    if args.stack_method == 'auto':
        if n_frames_for_auto < 8:
            args.stack_method = 'percentile'
            safe_print(f"    Auto-selected percentile clipping (<8 frames)")
        else:
            args.stack_method = 'sigma_clip'
            safe_print(f"    Auto-selected sigma_clip ({n_frames_for_auto} frames)")
    elif getattr(args, 'winsorize', False) and args.stack_method not in ('winsorized',):
        args.stack_method = 'winsorized'

    # ======================================================================
    # PHASE 3: Stacking (quality-weighted combine)
    # ======================================================================
    print_phase(3, "Stacking")
    phase_start = time.time()
    print(f"  Method: {args.stack_method}")
    print(f"  Combining {len(final)} frames...")

    # Compute quality weights for weighted stacking
    w_snr   = getattr(args, 'weight_snr',   1.0)
    w_fwhm  = getattr(args, 'weight_fwhm',  1.0)
    w_stars = getattr(args, 'weight_stars', 1.0)
    use_noise_weight = getattr(args, 'weight_noise', False)

    if w_snr != 1.0 or w_fwhm != 1.0 or w_stars != 1.0 or use_noise_weight:
        # Per-component weighting: each factor is independently normalized to [0,1]
        # and raised to the user-specified exponent (0 = ignore, >1 = emphasize).
        snr_vals   = np.array([f.metrics.get('snr', 1.0)        for f in final], dtype=np.float64)
        fwhm_vals  = np.array([f.metrics.get('fwhm', 0.0)       for f in final], dtype=np.float64)
        star_vals  = np.array([f.metrics.get('star_count', 1)   for f in final], dtype=np.float64)

        snr_max = snr_vals.max() if snr_vals.max() > 0 else 1.0
        snr_factor = np.clip(snr_vals / snr_max, 0.01, 1.0)

        # FWHM: lower = sharper = better; frames without FWHM get neutral weight
        fwhm_pos = fwhm_vals[fwhm_vals > 0]
        if len(fwhm_pos):
            fwhm_inv = np.where(fwhm_vals > 0, 1.0 / np.maximum(fwhm_vals, 0.5), 1.0)
            fwhm_factor = np.clip(fwhm_inv / fwhm_inv.max(), 0.01, 1.0)
        else:
            fwhm_factor = np.ones(len(final))

        star_max = star_vals.max() if star_vals.max() > 0 else 1.0
        star_factor = np.clip(star_vals / star_max, 0.01, 1.0)

        weights = (snr_factor ** w_snr) * (fwhm_factor ** w_fwhm) * (star_factor ** w_stars)

        if use_noise_weight:
            # Noise ≈ signal / SNR; weight by 1/noise (lower noise = better)
            brightness = np.array([f.metrics.get('brightness', 1.0) for f in final], dtype=np.float64)
            noise_est = brightness / np.maximum(snr_vals, 0.001)
            noise_max = noise_est.max() if noise_est.max() > 0 else 1.0
            noise_factor = np.clip(noise_max / np.maximum(noise_est, 1e-6), 0.01, 1.0)
            weights *= noise_factor

        weights = np.sqrt(weights / (weights.max() if weights.max() > 0 else 1.0))
        print(f"  Quality weights (per-component SNR^{w_snr} FWHM^{w_fwhm} stars^{w_stars}"
              f"{' noise' if use_noise_weight else ''}): "
              f"min={weights.min():.3f}, max={weights.max():.3f}, mean={weights.mean():.3f}")
    else:
        scores = np.array([f.metrics.get('score', 1.0) for f in final])
        max_score = scores.max() if scores.max() > 0 else 1.0
        weights = np.sqrt(scores / max_score)
        print(f"  Quality weights: min={weights.min():.3f}, max={weights.max():.3f}, "
              f"mean={weights.mean():.3f} (sqrt-compressed)")

    # Crop to common valid region
    top, bottom, left, right = calc_common_crop(shifts, (H, W), transforms=transforms)
    stats.output_shape = (bottom - top, right - left)
    stats.cropped_pixels = (H - (bottom - top), W - (right - left))

    n_final = len(final)
    use_aligned_memmap = args.stack_method in ('median', 'sigma_clip', 'winsorized',
                                               'percentile', 'esd')

    if use_aligned_memmap:
        # Create aligned memmap for median/sigma_clip
        mm_aligned_path = os.path.join(tempfile.gettempdir(), f'stack_aligned_{os.getpid()}.dat')
        crop_h, crop_w = bottom - top, right - left
        mem_aligned = np.memmap(mm_aligned_path, dtype='float32', mode='w+',
                                shape=(n_final, crop_h, crop_w, C))

        gpu = get_gpu()

        def _align_one_frame(j):
            with gpu.stream_context():
                orig_idx = final_indices[j]
                rgb = np.array(mem_rgb[orig_idx])
                aligned = apply_transform(rgb, shift=shifts[j], transform=transforms[j])
                mem_aligned[j] = aligned[top:bottom, left:right, :]

        if gpu.active:
            n_align_workers = min(gpu.max_gpu_workers(Config.GPU_ALIGN_WORKER_MB,
                                                      Config.GPU_VRAM_RESERVE_MB), n_final)
        else:
            n_align_workers = min(os.cpu_count() or 4, n_final)
        with ThreadPoolExecutor(max_workers=n_align_workers) as executor:
            futures = {executor.submit(_align_one_frame, j): j for j in range(n_final)}
            for future in tqdm(as_completed(futures), total=n_final,
                               desc="  Aligning", unit="frame",
                               disable=args.verbose):
                future.result()  # propagate exceptions
        mem_aligned.flush()

        if args.stack_method in ('sigma_clip', 'winsorized'):
            use_winsorize = (args.stack_method == 'winsorized')
            use_mad = (getattr(args, 'rejection_estimator', 'mad') == 'mad')
            estimator_label = 'MAD' if use_mad else 'std'
            mode_label = 'winsorized' if use_winsorize else 'reject'
            print(f"  Sigma-clip: sigma={args.rejection_sigma}, iters={args.rejection_iters}, "
                  f"estimator={estimator_label}, mode={mode_label}")
            stacked = sigma_clip_combine(
                mem_aligned, sigma=args.rejection_sigma,
                max_iters=args.rejection_iters,
                weights=weights,
                winsorize=use_winsorize,
                use_mad=use_mad,
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
            print(f"  ESD: max_outliers={'N//4' if max_out == 0 else max_out}, "
                  f"significance={sig}")
            stacked = esd_combine(mem_aligned, max_outliers=max_out,
                                  significance=sig, weights=weights,
                                  verbose=args.verbose)
        else:
            stacked = np.median(mem_aligned, axis=0).astype(np.float32)

        del mem_aligned
        try:
            os.remove(mm_aligned_path)
        except Exception:
            pass
    else:
        drizzle_scale = getattr(args, 'drizzle_scale', 1.0)
        drizzle_active = drizzle_scale > 1.0

        if drizzle_active:
            # Drizzle combine — resample each frame onto upscaled grid with
            # Lanczos interpolation, preserving sub-pixel shifts for genuine
            # super-resolution.
            crop_h, crop_w = bottom - top, right - left
            out_h = int(round(crop_h * drizzle_scale))
            out_w = int(round(crop_w * drizzle_scale))
            print(f"  Drizzle: {drizzle_scale:.1f}x ({crop_h}x{crop_w} -> {out_h}x{out_w})")

            acc = np.zeros((out_h, out_w, C), dtype=np.float64)
            weight_map = np.zeros((out_h, out_w, C), dtype=np.float64)
            gpu = get_gpu()

            # Coverage template for detecting valid pixels
            ones_template = None

            for j in tqdm(range(n_final), desc="  Drizzling", unit="frame",
                          disable=args.verbose):
                with gpu.stream_context():
                    orig_idx = final_indices[j]
                    rgb = np.array(mem_rgb[orig_idx])
                    aligned = apply_transform(rgb, shift=shifts[j], transform=transforms[j])
                    cropped = aligned[top:bottom, left:right, :]

                w = float(weights[j])
                # Resample with sub-pixel shift onto upscaled grid
                # The shift relative to reference is already in shifts[j],
                # but apply_transform already applied it. The residual
                # fractional component for drizzle comes from the shift itself.
                # Since apply_transform uses spline interpolation at native
                # resolution, we re-resample the already-aligned frame onto
                # the upscaled grid with identity shift (0,0).
                resampled = _lanczos_resample_frame(cropped, (0.0, 0.0),
                                                     drizzle_scale, out_h, out_w)

                # Coverage mask
                if ones_template is None or ones_template.shape != cropped.shape[:2]:
                    ones_template = np.ones(cropped.shape[:2], dtype=np.float64)
                coverage = _lanczos_resample_frame(ones_template, (0.0, 0.0),
                                                    drizzle_scale, out_h, out_w)
                valid = coverage > 0.5
                valid3 = valid[:, :, np.newaxis]
                acc += np.where(valid3, resampled * w, 0.0)
                weight_map += np.where(valid3, w, 0.0)

            weight_map[weight_map == 0] = 1.0
            stacked = (acc / weight_map).astype(np.float32)
        else:
            # Weighted mean combine — align in parallel, accumulate as results arrive
            acc = np.zeros((bottom - top, right - left, C), dtype=np.float64)
            total_weight = 0.0

            gpu = get_gpu()

            def _align_and_crop(j):
                with gpu.stream_context():
                    orig_idx = final_indices[j]
                    rgb = np.array(mem_rgb[orig_idx])
                    aligned = apply_transform(rgb, shift=shifts[j], transform=transforms[j])
                    return j, aligned[top:bottom, left:right, :]

            if gpu.active:
                n_mean_workers = min(gpu.max_gpu_workers(Config.GPU_ALIGN_WORKER_MB,
                                                         Config.GPU_VRAM_RESERVE_MB), n_final)
            else:
                n_mean_workers = min(os.cpu_count() or 4, n_final)
            with ThreadPoolExecutor(max_workers=n_mean_workers) as executor:
                futures = {executor.submit(_align_and_crop, j): j for j in range(n_final)}
                for future in tqdm(as_completed(futures), total=n_final,
                                   desc="  Stacking", unit="frame",
                                   disable=args.verbose):
                    j, cropped = future.result()
                    w = float(weights[j])
                    acc += cropped.astype(np.float64) * w
                    total_weight += w

            stacked = (acc / max(total_weight, 1e-12)).astype(np.float32)

    stats.stacking_time = time.time() - phase_start

    # Cleanup input memmaps
    del mem_rgb, mem_lum
    for p in (mm_rgb_path, mm_lum_path):
        try:
            os.remove(p)
        except Exception:
            pass

    # Save a copy of the stacked data BEFORE post-processing for the FITS
    # output.  Post-processing (background extraction, local contrast, etc.)
    # drops the sky level from ~5000+ ADU to ~5 ADU, reducing the sky SNR
    # from 60-100 to ~4.  FITS viewers auto-stretch this low-SNR data and
    # show prominent chromatic noise ("multicolored pixels").  By saving the
    # pre-processed stack, the FITS retains the original sky level and SNR,
    # matching what tools like DeepSkyStacker and Siril's own stacker produce.
    # The JPG preview still uses the fully post-processed data.
    fits_stacked = stacked.copy()

    # ======================================================================
    # PHASE 4: Post-processing
    # ======================================================================
    print_phase(4, "Post-processing")
    phase_start = time.time()

    # Per-channel hot pixel removal.  The per-frame removal (remove_hot_pixels_rgb)
    # detects on luminance, so single-channel hot pixels (especially red on Bayer
    # sensors) are diluted by the other channels and survive.  Persistent hot pixels
    # also survive sigma-clip stacking because they appear in every frame.
    #
    # Detection: a pixel is hot if ONE channel spikes well above its local
    # median AND the other channels do NOT show a proportional spike.  This
    # distinguishes sensor hot pixels (single-channel) from bright nebula
    # features (all channels elevated together).
    print("\n  Removing residual hot pixels (per-channel)...")
    _hp_start = time.time()
    _hp_fixed = 0
    # Single 3D median filter (size 5×5 per channel, no cross-channel mixing)
    _hp_meds = ndimage.median_filter(stacked, size=(5, 5, 1))
    _hp_diffs = stacked - _hp_meds                              # (H, W, 3)
    _hp_mads = np.median(np.abs(_hp_diffs), axis=(0, 1))        # (3,)
    _hp_sigmas = np.maximum(_hp_mads * 1.4826, 1e-6)            # (3,)
    ch_spikes = _hp_diffs / _hp_sigmas[np.newaxis, np.newaxis, :]  # (H, W, 3)
    for c in range(3):
        others = [i for i in range(3) if i != c]
        other_normal = np.all(ch_spikes[:, :, others] < 4.0, axis=2)
        hot = (ch_spikes[:, :, c] > 12.0) & other_normal
        n_hot = int(np.sum(hot))
        if n_hot > 0:
            stacked[:, :, c][hot] = _hp_meds[:, :, c][hot]
            _hp_fixed += n_hot
    safe_print(f"  ✓ Per-channel hot pixel removal: {_hp_fixed} pixels fixed "
               f"({format_time(time.time() - _hp_start)})")

    # Detect stars once — reused by background extraction, wavelet, NLM, and deconvolution.
    # Use the same multi-FWHM + quality-filter logic as compute_quality_metrics so that
    # varying seeing, focal ratios, and stacked noise floors don't produce zero detections.
    pp_star_mask = None
    _pp_sources = None
    if DAOStarFinder is not None and sigma_clipped_stats is not None:
        try:
            _pp_lum = (0.299 * stacked[:, :, 0] + 0.587 * stacked[:, :, 1]
                       + 0.114 * stacked[:, :, 2])
            _, _bg_med, _bg_std = sigma_clipped_stats(_pp_lum, sigma=3.0, maxiters=5)
            _threshold = 5.0 * float(_bg_std)
            _bg_sub = _pp_lum - float(_bg_med)

            _pp_sources = _detect_stars_multi_fwhm(_bg_sub, _threshold)

            if _pp_sources is not None and len(_pp_sources) > 0:
                pp_star_mask = generate_star_mask(_pp_lum.shape, _pp_sources, fwhm=4.0)
                if args.verbose:
                    safe_print(f"    Post-processing star mask: {len(_pp_sources)} stars "
                               f"(best FWHM trial)")
        except Exception:
            pass

    # 1. Background extraction (DBE or legacy mesh)
    if args.background_extraction:
        use_dbe = getattr(args, 'dbe', True)
        bg_start = time.time()
        if use_dbe:
            dbe_patch = getattr(args, 'dbe_patch_size', Config.DBE_PATCH_SIZE)
            print(f"\n  Applying Dynamic Background Extraction "
                  f"(patch={dbe_patch}px, RBF thin-plate-spline, "
                  f"sigma={args.bg_clip_sigma})...")
            stacked = dynamic_background_extraction(
                stacked,
                patch_size=dbe_patch,
                clip_sigma=args.bg_clip_sigma,
                verbose=args.verbose,
                star_mask=pp_star_mask)
            safe_print(f"  ✓ Dynamic Background Extraction "
                       f"({format_time(time.time() - bg_start)})")
        else:
            print(f"\n  Applying background extraction (mesh={args.bg_mesh_size}, "
                  f"sigma={args.bg_clip_sigma})...")
            stacked = apply_background_extraction(
                stacked, mesh_size=args.bg_mesh_size,
                filter_size=args.bg_filter_size,
                clip_sigma=args.bg_clip_sigma,
                verbose=args.verbose,
                star_mask=pp_star_mask)
            safe_print(f"  ✓ Background extraction "
                       f"({format_time(time.time() - bg_start)})")

    # 2. Chroma noise reduction
    if getattr(args, 'chroma_nr', True):
        cnr_sigma = getattr(args, 'chroma_nr_sigma', 2.0)
        print(f"\n  Applying chroma noise reduction (sigma={cnr_sigma})...")
        cnr_start = time.time()
        stacked = reduce_chroma_noise(stacked, sigma=cnr_sigma)
        safe_print(f"  ✓ Chroma noise reduction ({format_time(time.time() - cnr_start)})")

    # Sky floor correction: subtract residual pedestal from background
    # extraction.  Uses luminance to compute a single uniform offset for
    # all channels — per-channel subtraction would introduce color casts
    # (e.g. subtracting 0.3 from R but 0 from G creates magenta when
    # stretched aggressively in FITS viewers).
    if args.background_extraction:
        try:
            H_s, W_s = stacked.shape[:2]
            lum_s = (0.299 * stacked[:, :, 0] + 0.587 * stacked[:, :, 1]
                     + 0.114 * stacked[:, :, 2])

            # Build sky mask: start with all pixels, then exclude galaxy/nebula
            sky_mask = np.ones((H_s, W_s), dtype=bool)
            try:
                smooth_sigma = max(20.0, min(H_s, W_s) / 50.0)
                lum_smooth = ndimage.gaussian_filter(lum_s, sigma=smooth_sigma)
                by = max(10, int(H_s * Config.BORDER_FRAC))
                bx = max(10, int(W_s * Config.BORDER_FRAC))
                border_pix = np.concatenate([
                    lum_smooth[:by, :].ravel(), lum_smooth[-by:, :].ravel(),
                    lum_smooth[by:-by, :bx].ravel(), lum_smooth[by:-by, -bx:].ravel(),
                ])
                sky_med_lum = float(np.median(border_pix))
                sky_std_lum = float(np.std(border_pix))
                peak_y, peak_x = np.unravel_index(int(np.argmax(lum_smooth)), (H_s, W_s))
                peak_val = float(lum_smooth[peak_y, peak_x])
                detect_thresh = sky_med_lum + max(
                    2.0 * max(sky_std_lum, 1.0), 0.05 * (peak_val - sky_med_lum))
                frac_bright = float(np.mean(lum_smooth > detect_thresh))
                if peak_val > detect_thresh and frac_bright > 0.001:
                    excl_radius = int(min(H_s, W_s) * 0.30)
                    yy, xx = np.mgrid[:H_s, :W_s]
                    remaining_lum = lum_smooth.copy()
                    primary_peak = peak_val
                    for _src_i in range(3):
                        py, px = np.unravel_index(
                            int(np.argmax(remaining_lum)), (H_s, W_s))
                        pv = float(remaining_lum[py, px])
                        if pv <= detect_thresh:
                            break
                        if _src_i > 0:
                            primary_excess = primary_peak - sky_med_lum
                            current_excess = pv - sky_med_lum
                            if primary_excess > 0 and current_excess < 0.5 * primary_excess:
                                break
                        dist = np.sqrt((yy - py) ** 2 + (xx - px) ** 2)
                        sky_mask &= (dist >= excl_radius)
                        remaining_lum[dist < excl_radius] = float(
                            np.min(remaining_lum))
            except Exception:
                pass

            # Exclude bright point sources (stars)
            try:
                if sigma_clipped_stats is not None:
                    sample = lum_s[sky_mask].ravel() if sky_mask.any() else lum_s.ravel()
                    _, lum_med, lum_std = sigma_clipped_stats(sample, sigma=3.0, maxiters=5)
                    star_thresh = float(lum_med) + 3.0 * float(lum_std)
                    sky_mask &= (lum_s < star_thresh)
            except Exception:
                pass

            # Subtract per-channel sky floor so each channel is independently
            # zeroed.  A single luminance offset can leave residual color casts
            # when Bayer channels have different sky backgrounds.
            if sky_mask.sum() > 1000:
                for c in range(3):
                    col = stacked[:, :, c][sky_mask].ravel()
                    if sigma_clipped_stats is not None:
                        try:
                            _, sky_floor, _ = sigma_clipped_stats(col, sigma=3.0, maxiters=5)
                            sky_floor = float(sky_floor)
                        except Exception:
                            sky_floor = float(np.median(col))
                    else:
                        sky_floor = float(np.median(col))
                    if sky_floor > 0:
                        stacked[:, :, c] -= sky_floor
                        if args.verbose:
                            safe_print(f"    Sky floor correction ch{c}: -{sky_floor:.2f}")
        except Exception:
            pass

    # 3. Local normalization
    if getattr(args, 'local_normalize', False):
        ln_sigma = getattr(args, 'local_normalize_sigma', 50.0)
        print(f"\n  Applying local normalization (sigma={ln_sigma})...")
        ln_start = time.time()
        stacked = local_normalize(stacked, sigma=ln_sigma)
        safe_print(f"  ✓ Local normalization ({format_time(time.time() - ln_start)})")

    # 4. Wavelet denoising (luma/chroma split, star-protected)
    if getattr(args, 'denoise', False):
        chroma_boost = getattr(args, 'denoise_chroma_boost', 2.0)
        use_adaptive = getattr(args, 'denoise_adaptive', False)
        dn_start = time.time()
        if use_adaptive:
            print(f"\n  Applying adaptive wavelet denoising (BayesShrink, "
                  f"chroma_factor={chroma_boost:.1f})...")
            stacked = adaptive_wavelet_denoise(stacked,
                                               chroma_factor=chroma_boost,
                                               star_mask=pp_star_mask)
        else:
            strength = getattr(args, 'denoise_strength', 3.0)
            if getattr(args, 'auto_denoise_strength', True):
                fwhm_vals = [f.metrics.get('fwhm', 0.0) for f in final
                             if f.metrics and f.metrics.get('fwhm', 0.0) > 0]
                fwhm_mean = float(np.mean(fwhm_vals)) if fwhm_vals else 0.0
                strength = estimate_denoise_strength(stacked, fwhm_mean=fwhm_mean)
                fwhm_note = f', FWHM={fwhm_mean:.1f}px' if fwhm_mean > 0 else ''
                safe_print(f"\n  Auto-denoise strength: {strength:.2f}"
                           f" (from stacked SNR{fwhm_note})")
            print(f"\n  Applying wavelet denoising "
                  f"(luma={strength:.1f}, chroma={strength * chroma_boost:.1f})...")
            stacked = wavelet_denoise(stacked, threshold_factor=strength,
                                      chroma_factor=chroma_boost,
                                      star_mask=pp_star_mask)
        safe_print(f"  ✓ Wavelet denoise ({format_time(time.time() - dn_start)})")

    # 4.5. Sky residual correction
    # Background extraction leaves mesh-scale residuals that become visible
    # when stretched in FITS viewers or after denoising amplifies them.
    # Always run after background extraction to ensure a clean sky.
    if args.background_extraction:
        _sr_mesh = max(32, args.bg_mesh_size // 2)
        _H_pp, _W_pp = stacked.shape[:2]
        # Broad pass first: removes large-scale gradient residuals (LP
        # gradient, IFN halo) that survive the main extraction because the
        # galaxy exclusion zone covers a large image fraction.  Using a mesh
        # cell ~1/6 of the shorter image dimension captures structure at the
        # same spatial scale as the exclusion zone.
        _sr_broad_mesh = max(args.bg_mesh_size, min(_H_pp, _W_pp) // 6)
        print(f"\n  Correcting sky residuals "
              f"(broad={_sr_broad_mesh}px, fine={_sr_mesh}px)...")
        _sr_start = time.time()
        stacked = remove_sky_residual(
            stacked, mesh_size=_sr_broad_mesh, filter_size=1,
            clip_sigma=args.bg_clip_sigma,
            star_mask=pp_star_mask, verbose=args.verbose)
        for _sr_pass in range(2):
            stacked = remove_sky_residual(
                stacked, mesh_size=_sr_mesh, filter_size=1,
                clip_sigma=args.bg_clip_sigma,
                star_mask=pp_star_mask, verbose=(args.verbose and _sr_pass == 0))
        safe_print(f"  ✓ Sky residual correction "
                   f"({format_time(time.time() - _sr_start)})")

        # Sky floor normalisation: subtract the per-channel constant pedestal
        # so the true sky reads zero.  Background extraction removes the
        # *gradient* across the image; this step removes the remaining flat
        # *offset* (residual bias, dark current, or extraction reference level).
        stacked = sky_floor_normalize(
            stacked, star_mask=pp_star_mask, verbose=args.verbose)

    # 5. Non-local means denoising (optional second pass for faint extended emission)
    if getattr(args, 'denoise_nlm', False):
        nlm_h = getattr(args, 'denoise_nlm_strength', 1.0)
        nlm_blend = getattr(args, 'denoise_nlm_blend', 0.5)
        print(f"\n  Applying NLM denoising (h={nlm_h:.1f}, blend={nlm_blend:.2f})...")
        nlm_start = time.time()
        stacked = nlm_denoise(stacked, h=nlm_h, blend=nlm_blend)
        safe_print(f"  ✓ NLM denoise ({format_time(time.time() - nlm_start)})")

    # 6. Bilateral filter denoising (spatially uniform alternative to NLM)
    if getattr(args, 'denoise_bilateral', False):
        bil_sigma_color = getattr(args, 'denoise_bilateral_sigma_color', None)
        bil_sigma_space = getattr(args, 'denoise_bilateral_sigma_space', 3.0)
        sc_str = f"{bil_sigma_color:.2f}" if bil_sigma_color is not None else "auto"
        print(f"\n  Applying bilateral denoising "
              f"(sigma_color={sc_str}, sigma_space={bil_sigma_space:.1f})...")
        bil_start = time.time()
        stacked = bilateral_denoise(stacked,
                                    sigma_color=bil_sigma_color,
                                    sigma_space=bil_sigma_space)
        safe_print(f"  ✓ Bilateral denoise ({format_time(time.time() - bil_start)})")

    # 7. Richardson-Lucy deconvolution
    if getattr(args, 'deconvolve', False):
        rl_iters = getattr(args, 'deconvolve_iterations', Config.RL_DEFAULT_ITERATIONS)
        rl_fwhm_override = getattr(args, 'deconvolve_fwhm', None)
        rl_model = getattr(args, 'deconvolve_psf_model', 'moffat')

        psf = None
        psf_fwhm = 0.0
        if rl_fwhm_override is not None:
            # User-supplied FWHM — synthesize PSF
            psf = make_synthetic_psf(rl_fwhm_override, model=rl_model)
            psf_fwhm = rl_fwhm_override
            safe_print(f"\n  Richardson-Lucy deconvolution (FWHM={psf_fwhm:.2f}px manual, "
                       f"iters={rl_iters})...")
        elif _pp_sources is not None and len(_pp_sources) > 0:
            safe_print(f"\n  Estimating PSF from star profiles ({rl_model} model)...")
            psf, psf_fwhm = estimate_psf(stacked, _pp_sources, model=rl_model)
            if psf is not None:
                safe_print(f"    PSF FWHM: {psf_fwhm:.2f} px")
                safe_print(f"  Richardson-Lucy deconvolution (iters={rl_iters})...")
            else:
                safe_print("    PSF estimation failed — skipping deconvolution")
        else:
            safe_print("\n  No star detections for PSF estimation — use --deconvolve-fwhm "
                       "to specify manually")

        if psf is not None:
            rl_start = time.time()
            stacked = richardson_lucy_deconvolve(stacked, psf,
                                                  iterations=rl_iters,
                                                  star_mask=pp_star_mask)
            safe_print(f"  ✓ Richardson-Lucy deconvolution ({format_time(time.time() - rl_start)})")

    # 8. Star reduction — shrink star halos to improve galaxy-to-star visual ratio.
    # Stars in galaxy fields compete visually with the delicate spiral arms and dust
    # lanes.  This step softens star cores via blending with a Gaussian-blurred
    # version, making stars appear slightly smaller without removing them.
    if getattr(args, 'star_reduce', False):
        sr_factor = float(getattr(args, 'star_reduce_factor', 0.4))
        sr_sigma = float(getattr(args, 'star_reduce_sigma', 1.5))
        print(f"\n  Applying star reduction (factor={sr_factor:.2f}, blur_sigma={sr_sigma:.1f})...")
        sr_start = time.time()
        stacked = reduce_stars(stacked, pp_star_mask,
                               reduction_factor=sr_factor,
                               blur_sigma=sr_sigma)
        safe_print(f"  ✓ Star reduction ({format_time(time.time() - sr_start)})")

    # 9. Multiscale local contrast enhancement — reveals galaxy structure at
    # fine (dust-lane edges), medium (spiral arm boundaries), and coarse
    # (overall brightness profile) spatial scales via luminance unsharp masking.
    # A mid-tone protection mask prevents sky-noise amplification and nucleus
    # blowout; the medium scale (σ ≈ 12 px) is the most effective for typical
    # galaxy targets like M64 where the dust lane is the key feature.
    if getattr(args, 'local_contrast', False):
        lc_strength = float(getattr(args, 'local_contrast_strength', 0.7))
        print(f"\n  Applying multiscale local contrast enhancement "
              f"(strength={lc_strength:.2f}, scales=2/12/40 px)...")
        lc_start = time.time()
        stacked = multiscale_local_contrast(stacked, strength=lc_strength,
                                            star_mask=pp_star_mask)
        safe_print(f"  ✓ Local contrast enhancement ({format_time(time.time() - lc_start)})")

    stats.post_processing_time = time.time() - phase_start

    # Update memory usage
    if HAS_PSUTIL:
        stats.peak_memory_mb = get_memory_usage_mb()

    # Save FITS from the pre-processed stacked data (high sky SNR).
    # Post-processing is only used for the JPG preview.
    from astropy.io import fits
    out_h, out_w, _ = stacked.shape
    hdu = fits.PrimaryHDU()
    data_out = np.transpose(fits_stacked, (2, 0, 1)).astype(np.float32)
    hdu.data = data_out
    del fits_stacked  # free memory

    populate_fits_header(
        header=hdu.header, frames=final, stats=stats, args=args,
        stacked_shape=stacked.shape, shifts=shifts,
        masters=masters, dither_info=dither_info)
    hdu.writeto(output_path, overwrite=True)

    # Plate solving
    plate_solved = False
    if getattr(args, 'plate_solve', False):
        if args.verbose:
            print("\n  Attempting plate solving...")
        plate_solved = solve_plate(data_out, hdu.header, output_path, verbose=args.verbose)
        if plate_solved:
            hdu.writeto(output_path, overwrite=True)
    elif args.verbose:
        print("\n  Plate solving skipped (use --plate-solve to enable)")

    # Preview with configurable stretch
    preview_path = os.path.splitext(output_path)[0] + '.jpg'
    stretch_method = getattr(args, 'stretch', 'linear')
    save_preview_rgb(stacked, preview_path, stretch=stretch_method,
                     ghs_b=float(getattr(args, 'ghs_b', 8.0)),
                     ghs_sp=float(getattr(args, 'ghs_sp', 0.15)),
                     ghs_hp=float(getattr(args, 'ghs_hp', 0.95)))

    print(f"  Output size: {out_h}x{out_w} "
          f"(cropped {stats.cropped_pixels[0]}x{stats.cropped_pixels[1]} pixels)")

    # Summary
    from src.utils import print_header
    print_header("SUMMARY", "=")
    print(f"  Frames analyzed:  {stats.total_frames}")
    print(f"  Frames stacked:   {stats.accepted_frames} "
          f"({stats.accepted_frames/stats.total_frames*100:.1f}%)")
    if stats.rejected_frames > 0:
        print(f"  Frames rejected:  {stats.rejected_frames}")
    print(f"  Output:           {os.path.basename(output_path)} ({out_h}x{out_w}x3)")
    print(f"  Preview:          {os.path.basename(preview_path)} ({stretch_method} stretch)")
    print(f"  Processing time:  {format_time(stats.total_time())}")
    print(f"    Quality+Load:   {format_time(stats.quality_time)}")
    print(f"    Registration:   {format_time(stats.registration_time)}")
    print(f"    Stacking:       {format_time(stats.stacking_time)}")
    print(f"    Post-process:   {format_time(stats.post_processing_time)}")
    if HAS_PSUTIL:
        print(f"  Peak memory:      {stats.peak_memory_mb:.1f} MB")

    if stats.warnings:
        safe_print(f"\n  Warnings:")
        for w in stats.warnings[:5]:
            safe_print(f"    - {w}")
    if stats.errors:
        safe_print(f"\n  Errors: {len(stats.errors)}")

    safe_print(f"\n  Stack complete!")

    # AI session report: streams a narrative summary to stdout and saves a
    # Markdown file alongside the output FITS.
    if getattr(args, 'ai_report', False):
        from src.ai_advisor import build_report_context, generate_session_report
        report_ctx = build_report_context(
            final=final,
            rejected_reasons=rejected_reasons,
            args=args,
            stats=stats,
            shifts=shifts,
            dither_info=dither_info,
            output_path=output_path,
            stacked_shape=stacked.shape,
        )
        generate_session_report(report_ctx, output_path)

    return output_path
