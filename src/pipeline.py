"""Pipeline orchestrator: stack_target ties together the four processing phases."""
from __future__ import annotations

import argparse
import os
import tempfile
import time
from typing import Dict, List, Optional

import numpy as np

from src.models import FrameInfo, ProcessingStats
from src.utils import safe_print, print_phase, format_time, get_memory_usage_mb
from src.io_fits import load_fits, save_preview_rgb, populate_fits_header
from src.debayer import debayer
from src.plate_solve import solve_plate
from src.frame_processor import execute_frame_processing, quality_gate
from src.registration import run_registration_phase
from src.stacking import run_stacking_phase
from src.postprocess import postprocess_stack


try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False


def stack_target(frames: List[FrameInfo], output_path: str, args: argparse.Namespace,
                 masters: Dict[str, Optional[np.ndarray]], stats: ProcessingStats):
    lights = [f for f in frames if f.type == 'light']
    if not lights:
        print('  No light frames found for target')
        return None

    stats.total_frames = len(lights)
    n = len(lights)

    # ======================================================================
    # PHASE 1: Process & Analyse
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

    mm_rgb_path = os.path.join(tempfile.gettempdir(), f'stack_rgb_{os.getpid()}.dat')
    mm_lum_path = os.path.join(tempfile.gettempdir(), f'stack_lum_{os.getpid()}.dat')
    rgb_shape = (n, H_rgb, W_rgb, C)
    lum_shape = (n, H_rgb, W_rgb)
    mem_rgb = np.memmap(mm_rgb_path, dtype='float32', mode='w+', shape=rgb_shape)
    mem_lum = np.memmap(mm_lum_path, dtype='float32', mode='w+', shape=lum_shape)
    cached_lums: list = [None] * n
    rejected_reasons: dict = {}

    execute_frame_processing(
        lights, masters, args,
        mem_rgb, mem_lum, mm_rgb_path, mm_lum_path,
        cached_lums, rgb_shape, lum_shape,
        rejected_reasons, stats)

    final = quality_gate(lights, args, rejected_reasons, stats)
    stats.quality_time = time.time() - phase_start

    if not final:
        print(f'\n  ERROR: All {n} frames rejected!')
        if rejected_reasons:
            print('  Rejection reasons:')
            for path, reason in list(rejected_reasons.items())[:10]:
                print(f'    - {os.path.basename(path)}: {reason}')
        del mem_rgb, mem_lum
        for p in (mm_rgb_path, mm_lum_path):
            try:
                os.remove(p)
            except Exception:
                pass
        return None

    _lights_index = {id(f): i for i, f in enumerate(lights)}
    final_indices = [_lights_index[id(f)] for f in final]

    # AI parameter advisor (runs between phase 1 and 2)
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

    # Heuristic auto-advisor
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
    # PHASE 2: Registration
    # ======================================================================
    print_phase(2, "Registration")
    phase_start = time.time()

    best = max(final, key=lambda x: x.metrics.get('score', 0))
    best_idx = lights.index(best)
    print(f"  Reference frame: {os.path.basename(best.path)} "
          f"(score={best.metrics.get('score', 0):.1f})")

    ref_lum = np.array(mem_lum[best_idx])
    H, W = ref_lum.shape
    if args.verbose:
        print(f'  Reference luminance: min={np.min(ref_lum):.1f}, max={np.max(ref_lum):.1f}, '
              f'mean={np.mean(ref_lum):.1f}, std={np.std(ref_lum):.1f}')

    shifts, transforms, dither_info = run_registration_phase(
        final, final_indices, best, best_idx,
        ref_lum, mem_lum, cached_lums, H, W, args, stats)

    stats.registration_time = time.time() - phase_start
    del cached_lums

    # Resolve 'auto' and legacy --winsorize shorthand
    if args.stack_method == 'auto':
        if len(final) < 8:
            args.stack_method = 'percentile'
            safe_print(f"    Auto-selected percentile clipping (<8 frames)")
        else:
            args.stack_method = 'sigma_clip'
            safe_print(f"    Auto-selected sigma_clip ({len(final)} frames)")
    elif getattr(args, 'winsorize', False) and args.stack_method not in ('winsorized',):
        args.stack_method = 'winsorized'

    # ======================================================================
    # PHASE 3: Stacking
    # ======================================================================
    print_phase(3, "Stacking")
    phase_start = time.time()

    stacked, fits_stacked, top, bottom, left, right = run_stacking_phase(
        final, final_indices, mem_rgb,
        shifts, transforms, H, W, C, args, stats)

    stats.stacking_time = time.time() - phase_start

    del mem_rgb, mem_lum
    for p in (mm_rgb_path, mm_lum_path):
        try:
            os.remove(p)
        except Exception:
            pass

    # ======================================================================
    # PHASE 4: Post-processing
    # ======================================================================
    print_phase(4, "Post-processing")
    phase_start = time.time()

    stacked = postprocess_stack(stacked, args, final, stats)

    stats.post_processing_time = time.time() - phase_start

    # ======================================================================
    # Output
    # ======================================================================
    if HAS_PSUTIL:
        stats.peak_memory_mb = get_memory_usage_mb()

    from astropy.io import fits
    out_h, out_w, _ = stacked.shape
    hdu = fits.PrimaryHDU()
    data_out = np.transpose(fits_stacked, (2, 0, 1)).astype(np.float32)
    hdu.data = data_out
    del fits_stacked

    populate_fits_header(
        header=hdu.header, frames=final, stats=stats, args=args,
        stacked_shape=stacked.shape, shifts=shifts,
        masters=masters, dither_info=dither_info)
    hdu.writeto(output_path, overwrite=True)

    if getattr(args, 'plate_solve', False):
        if args.verbose:
            print("\n  Attempting plate solving...")
        if solve_plate(data_out, hdu.header, output_path, verbose=args.verbose):
            hdu.writeto(output_path, overwrite=True)
    elif args.verbose:
        print("\n  Plate solving skipped (use --plate-solve to enable)")

    preview_path = os.path.splitext(output_path)[0] + '.jpg'
    stretch_method = getattr(args, 'stretch', 'linear')
    save_preview_rgb(stacked, preview_path, stretch=stretch_method,
                     ghs_b=float(getattr(args, 'ghs_b', 8.0)),
                     ghs_sp=float(getattr(args, 'ghs_sp', 0.15)),
                     ghs_hp=float(getattr(args, 'ghs_hp', 0.95)))

    print(f"  Output size: {out_h}x{out_w} "
          f"(cropped {stats.cropped_pixels[0]}x{stats.cropped_pixels[1]} pixels)")

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

    if getattr(args, 'ai_report', False):
        from src.ai_advisor import build_report_context, generate_session_report
        report_ctx = build_report_context(
            final=final, rejected_reasons=rejected_reasons, args=args, stats=stats,
            shifts=shifts, dither_info=dither_info, output_path=output_path,
            stacked_shape=stacked.shape)
        generate_session_report(report_ctx, output_path)

    return output_path
