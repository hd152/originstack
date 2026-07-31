"""Pipeline orchestrator: stack_target ties together the four processing phases."""
from __future__ import annotations

import argparse
import csv
import gc
import os
import shutil
import tempfile
import time
from typing import Dict, List, Optional

import numpy as np

from src.models import FrameInfo, ProcessingStats
from src.utils import safe_print, print_phase, format_time, get_memory_usage_mb
from src.io_fits import load_fits, load_frame, save_preview_rgb, populate_fits_header
from src.debayer import debayer
from src.plate_solve import solve_plate
from src.frame_processor import execute_frame_processing, quality_gate, reload_accepted_frames
from src.registration import run_registration_phase, select_reference_frame
from src.stacking import run_stacking_phase
from src.postprocess import postprocess_stack
from src.checkpoint import (save_checkpoint, save_raw_stack, load_raw_stack,
                            can_resume, restore_frame_state, cleanup_checkpoint,
                            load_transforms, load_field_list)
from src.cleanup import register as _cleanup_register, deregister as _cleanup_deregister


try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False


class _MemmapManager:
    """Context manager for safe memmap creation and cleanup."""

    def __init__(self):
        self._files: List[str] = []
        self._memmaps: List = []

    def create(self, prefix: str, dtype: str, shape: tuple) -> np.ndarray:
        fd, path = tempfile.mkstemp(suffix='.dat', prefix=prefix)
        os.close(fd)
        mm = np.memmap(path, dtype=dtype, mode='w+', shape=shape)
        self._files.append(path)
        self._memmaps.append(mm)
        _cleanup_register(path)
        return mm

    def cleanup(self):
        # Explicitly close the underlying mmap handles.  On Windows,
        # os.remove() on an open memory-mapped file raises PermissionError,
        # so we must release the OS file lock before attempting deletion.
        for mm in self._memmaps:
            try:
                mm.flush()
            except Exception:
                pass
            try:
                if hasattr(mm, '_mmap') and mm._mmap is not None:
                    mm._mmap.close()
            except Exception:
                pass
        self._memmaps.clear()
        gc.collect()  # flush any remaining mmap references held by GC cycles
        for p in self._files:
            try:
                os.remove(p)
            except Exception:
                pass
            _cleanup_deregister(p)
        self._files.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()
        return False


def _save_blink_frames(final: List[FrameInfo], final_indices: List[int],
                       mem_rgb: np.ndarray, shifts: List, transforms: List,
                       top: int, bottom: int, left: int, right: int,
                       output_path: str) -> None:
    """Save aligned, cropped frames as a multi-extension FITS for blink comparison."""
    from src.registration import apply_transform
    from astropy.io import fits as afits

    blink_path = os.path.splitext(output_path)[0] + '_blink.fits'
    try:
        hdu_list = [afits.PrimaryHDU()]
        for j, f in enumerate(final):
            rgb = mem_rgb[final_indices[j]]
            aligned = apply_transform(rgb, shift=shifts[j], transform=transforms[j])
            cropped = aligned[top:bottom, left:right, :]
            data_out = np.transpose(cropped, (2, 0, 1)).astype(np.float32)
            ext = afits.ImageHDU(data=data_out, name=os.path.basename(f.path)[:68])
            ext.header['FRAMEIDX'] = (j, 'Frame index in stack')
            ext.header['QSCORE'] = (round(f.metrics.get('score', 0), 1), 'Quality score')
            ext.header['SHIFT_Y'] = (round(f.shift[0], 2), 'Applied Y shift (pixels)')
            ext.header['SHIFT_X'] = (round(f.shift[1], 2), 'Applied X shift (pixels)')
            hdu_list.append(ext)
        afits.HDUList(hdu_list).writeto(blink_path, overwrite=True)
        safe_print(f"  Saved blink frames: {os.path.basename(blink_path)} "
                   f"({len(final)} extensions)")
    except Exception as e:
        safe_print(f"  Blink frame save failed: {e}")


# ---------------------------------------------------------------------------
# New output helpers
# ---------------------------------------------------------------------------

def _write_quality_report(lights: List[FrameInfo], rejected_reasons: dict,
                           report_path: str) -> None:
    """Write per-frame quality metrics to a CSV file."""
    rows = []
    for f in lights:
        m = f.metrics or {}
        rows.append({
            'filename':        os.path.basename(f.path),
            'snr':             round(float(m.get('snr', 0)), 3),
            'fwhm':            round(float(m.get('fwhm', 0)), 3),
            'star_count':      int(m.get('star_count', 0)),
            'quality_score':   round(float(m.get('score', 0)), 2),
            'accepted':        getattr(f, 'accepted', True),
            'rejection_reason': rejected_reasons.get(f.path, ''),
        })
    if not rows:
        return
    try:
        with open(report_path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        safe_print(f"  Quality report: {os.path.basename(report_path)} ({len(rows)} frames)")
    except Exception as e:
        safe_print(f"  WARNING: could not write quality report: {e}")


def _export_frame_jpegs(final: List[FrameInfo], final_indices: List[int],
                         mem_rgb: np.ndarray, export_dir: str,
                         args: argparse.Namespace) -> None:
    """Write a stretched JPEG preview for each accepted frame."""
    os.makedirs(export_dir, exist_ok=True)
    stretch = getattr(args, 'stretch', 'ghs')
    ghs_b  = float(getattr(args, 'ghs_b', 8.0))
    ghs_sp = float(getattr(args, 'ghs_sp', 0.15))
    ghs_hp = float(getattr(args, 'ghs_hp', 0.95))
    for j, f in zip(final_indices, final):
        rgb = np.array(mem_rgb[j])
        stem = os.path.splitext(os.path.basename(f.path))[0]
        out_path = os.path.join(export_dir, stem + '.jpg')
        try:
            save_preview_rgb(rgb, out_path, stretch=stretch,
                             ghs_b=ghs_b, ghs_sp=ghs_sp, ghs_hp=ghs_hp)
        except Exception as e:
            safe_print(f"  WARNING: frame JPEG export failed for {stem}: {e}")
    safe_print(f"  Exported {len(final)} frame JPEGs → {export_dir}")


def _run_auto_advisor(final: List[FrameInfo], args,
                      prior_type: Optional[str] = None,
                      prior_confidence: float = 0.0) -> None:
    """Classify the target and apply heuristic post-processing settings in-place.

    Runs on both the normal path and the checkpoint-resume path: the galaxy /
    nebula presets (SCNR, photometric calibration, deconvolution, etc.) are set
    here, so skipping it on resume would silently post-process with bare CLI
    defaults and reintroduce colour casts the presets exist to remove.
    """
    if not getattr(args, 'auto', False):
        return
    from src.auto_settings import apply_auto_settings
    _target_type, label, signals, changes = apply_auto_settings(
        final, args, prior_type=prior_type, prior_confidence=prior_confidence)
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


def _save_tiff(stacked: np.ndarray, output_path: str) -> None:
    """Save (H, W, 3) float32 image as TIFF alongside the FITS output."""
    tiff_path = os.path.splitext(output_path)[0] + '.tiff'
    try:
        import tifffile
        # Planar (3, H, W) is the most widely compatible layout for 32-bit float
        data = np.ascontiguousarray(np.transpose(stacked.astype(np.float32), (2, 0, 1)))
        tifffile.imwrite(tiff_path, data, photometric='rgb', planarconfig='contig',
                         imagej=False)
    except ImportError:
        try:
            from PIL import Image
            arr16 = (np.clip(stacked, 0, 1) * 65535).astype(np.uint16)
            Image.fromarray(arr16).save(tiff_path)
            safe_print("  NOTE: tifffile not installed; saved 16-bit TIFF via Pillow")
        except Exception as e:
            safe_print(f"  WARNING: TIFF export failed: {e}")
            return
    safe_print(f"  TIFF output: {os.path.basename(tiff_path)}")


def _save_xisf(stacked: np.ndarray, output_path: str, header) -> None:
    """Save stacked image as PixInsight XISF alongside the FITS output."""
    xisf_path = os.path.splitext(output_path)[0] + '.xisf'
    try:
        from src.xisf_writer import write_xisf
        meta = {k: header.get(k) for k in
                ['OBJECT', 'DATE-OBS', 'EXPTIME', 'TELESCOP', 'INSTRUME',
                 'CRVAL1', 'CRVAL2', 'EQUINOX']
                if header.get(k) is not None}
        write_xisf(stacked, xisf_path, header_meta=meta)
        safe_print(f"  XISF output: {os.path.basename(xisf_path)}")
    except Exception as e:
        safe_print(f"  WARNING: XISF export failed: {e}")


def _hdr_blend(long_stack: np.ndarray, short_fits_path: str,
               verbose: bool = False) -> np.ndarray:
    """Blend a short-exposure stack into the saturated regions of the long stack.

    Rescales the short stack to match the long stack's background level, then
    blends smoothly in the regions where the long stack is near or above its
    98th-percentile value.
    """
    try:
        short_data, _ = load_fits(short_fits_path)
    except Exception as e:
        safe_print(f"  WARNING: HDR combine: could not load '{short_fits_path}': {e}")
        return long_stack

    # Normalise short to (H, W, 3)
    if short_data.ndim == 3 and short_data.shape[0] == 3:
        short_rgb = np.transpose(short_data, (1, 2, 0)).astype(np.float32)
    elif short_data.ndim == 3 and short_data.shape[2] == 3:
        short_rgb = short_data.astype(np.float32)
    else:
        safe_print("  WARNING: HDR combine: unexpected short-stack shape — skipping")
        return long_stack

    H, W, C = long_stack.shape
    if short_rgb.shape[:2] != (H, W):
        from scipy.ndimage import zoom
        zy, zx = H / short_rgb.shape[0], W / short_rgb.shape[1]
        short_rgb = zoom(short_rgb, (zy, zx, 1), order=1).astype(np.float32)

    # Per-channel sky-level scale: match short sky to long sky
    sat_threshold = float(np.percentile(long_stack, 98))
    # sky_mask is 2D: pixels where luminance is below median
    lum_long = long_stack.mean(axis=2)
    sky_mask = lum_long < float(np.percentile(lum_long, 50))
    for c in range(C):
        sky_long  = long_stack[:, :, c][sky_mask]
        sky_short = short_rgb[:, :, c][sky_mask]
        if len(sky_short) > 0 and sky_short.mean() > 0:
            short_rgb[:, :, c] *= sky_long.mean() / sky_short.mean()

    # Smooth blend mask centred around the saturation threshold
    blend_width = sat_threshold * 0.1
    sat_mask = np.clip((long_stack - (sat_threshold - blend_width)) / (2 * blend_width),
                       0.0, 1.0)
    blended = long_stack * (1.0 - sat_mask) + short_rgb * sat_mask
    if verbose:
        n_blended = int(np.sum(sat_mask > 0.01))
        safe_print(f"  HDR blend: {n_blended:,} pixels blended from short stack")
    return blended.astype(np.float32)


def stack_target(frames: List[FrameInfo], output_path: str, args: argparse.Namespace,
                 masters: Dict[str, Optional[np.ndarray]], stats: ProcessingStats) -> Optional[str]:
    lights = [f for f in frames if f.type == 'light']
    if not lights:
        print('  No light frames found for target')
        return None

    stats.total_frames = len(lights)
    n = len(lights)

    try:
        from src.webview import get_webview
        get_webview().set_run_info(
            target=getattr(args, '_inferred_target', None)
                   or os.path.basename(os.path.dirname(lights[0].path)),
            output=os.path.basename(output_path), n_frames=n)
    except Exception:
        pass

    # Check for a checkpoint from a previous interrupted run
    resume_phase = 0
    ckpt_state: Optional[Dict] = None
    if not getattr(args, 'no_resume', False):
        _ok, resume_phase, ckpt_state = can_resume(output_path, lights)
        # Drizzle must re-run phase 3: the saved raw_stack.npy is at input
        # resolution, not the upscaled drizzle output.  Downgrade so stacking
        # is repeated with the correct drizzle pass.
        if resume_phase >= 3 and getattr(args, 'drizzle_scale', 1.0) > 1.0:
            resume_phase = 2
            safe_print("  Drizzle requested — phase 3 checkpoint skipped; stacking will re-run")

    # Load session info (info.json written by capture app)
    from src.session_info import load_session_info
    _directory = getattr(args, '_input_directory', None)
    if not isinstance(_directory, (str, os.PathLike)):
        _directory = getattr(args, 'directory', None)
    if not isinstance(_directory, (str, os.PathLike)):
        _directory = os.path.dirname(output_path)
    _session_info = load_session_info(_directory)
    args._session_info = _session_info
    args._session_bayer = _session_info.bayer if _session_info else None
    if _session_info:
        _si_parts = [f"  Session info (info.json):"]
        if _session_info.object_name:
            _si_parts.append(f"object={_session_info.object_name!r}")
        if _session_info.bayer:
            _si_parts.append(f"bayer={_session_info.bayer}")
        if _session_info.has_wcs:
            _si_parts.append("WCS=yes")
        if _session_info.has_gps:
            _si_parts.append(f"GPS=yes")
        safe_print("  ".join(_si_parts))
    else:
        safe_print(f"  Session info: no info.json found in {os.path.basename(_directory)!r}")

    # Probe first frame for dimensions
    first_data, first_hdr = load_frame(lights[0].path)
    if first_data.ndim == 2:
        _probe_bayer = first_hdr.get('BAYERPAT',
                                     first_hdr.get('COLORTYP', args._session_bayer or 'RGGB'))
        first_rgb = debayer(first_data, pattern=_probe_bayer, method=args.debayer_method)
        H_rgb, W_rgb, C = first_rgb.shape
    else:
        H_rgb, W_rgb = first_data.shape[:2]
        C = first_data.shape[2] if first_data.ndim == 3 else 1
    del first_data

    rejected_reasons: dict = {}

    if resume_phase >= 3:
        # ======================================================================
        # PHASE 3 RESUME: load saved raw stack, skip phases 1-3 entirely
        # ======================================================================
        print_phase(3, "Stacking (resumed from checkpoint — loading saved raw stack)")
        stacked = load_raw_stack(output_path)
        if stacked is None:
            safe_print("  WARNING: raw_stack.npy missing — falling back to phase 2 resume")
            resume_phase = 2

        if resume_phase >= 3:
            final = restore_frame_state(lights, ckpt_state)
            _lights_index = {f.path: i for i, f in enumerate(lights)}
            final_indices = [_lights_index[f.path] for f in final if f.path in _lights_index]
            shifts = [tuple(s) for s in ckpt_state.get('shifts', [[0.0, 0.0]] * len(final))]
            transforms = [None] * len(final)
            dither_info = ckpt_state.get('dither_info', {})
            crop = ckpt_state.get('crop', [0, stacked.shape[0], 0, stacked.shape[1]])
            top, bottom, left, right = [int(v) for v in crop]
            fits_stacked = stacked.copy()
            H, W, C = stacked.shape
            stats.accepted_frames = len(final)
            stats.rejected_frames = n - len(final)

            # Auto-advisor must also run on resume: phases 1-3 are skipped here,
            # so without this the galaxy/nebula presets (SCNR, photometric
            # calibration, deconvolution) never get applied and post-processing
            # runs with bare CLI defaults — reintroducing colour casts.
            # Recover the metadata prior (folder/header/session) locally so the
            # classifier matches a fresh run; skip Simbad to avoid a network hit.
            if getattr(args, 'auto', False):
                from src.target_inference import infer_target_from_metadata
                _r_name, _r_type, _r_conf, _r_src = infer_target_from_metadata(
                    _directory, final, use_simbad=False,
                    session_name=_session_info.object_name if _session_info else None)
                if _r_name and _r_type and _r_type != 'unknown':
                    safe_print(f"\n  Target: {_r_name} "
                               f"[{_r_type.replace('_', ' ').title()}]  "
                               f"conf={_r_conf:.0%}  source={_r_src}")
                _run_auto_advisor(final, args,
                                  prior_type=_r_type, prior_confidence=_r_conf)

    if resume_phase < 3:
        # ======================================================================
        # PHASES 1-3: Normal path (requires memmap)
        # ======================================================================
        mm_mgr = _MemmapManager()
        rgb_shape = (n, H_rgb, W_rgb, C)
        lum_shape = (n, H_rgb, W_rgb)
        bytes_needed = (n * H_rgb * W_rgb * C * 4) + (n * H_rgb * W_rgb * 4)
        try:
            tmpdir = tempfile.gettempdir()
            _, _, free_bytes = shutil.disk_usage(tmpdir)
            if free_bytes < bytes_needed:
                safe_print(f"  WARNING: temp dir may lack space for memmaps: "
                           f"need ~{bytes_needed / 1e9:.2f} GB, "
                           f"{free_bytes / 1e9:.2f} GB free in {tmpdir}")
        except Exception:
            pass
        mem_rgb = mm_mgr.create('stack_rgb_', 'float32', rgb_shape)
        mm_rgb_path = mm_mgr._files[-1]
        mem_lum = mm_mgr.create('stack_lum_', 'float32', lum_shape)
        mm_lum_path = mm_mgr._files[-1]
        cached_lums: list = [None] * n

        try:
            # Resolve the cosmic-ray-rejection tri-state (None = auto).
            # Per-frame L.A.Cosmic earns its cost only when the stack cannot
            # reject outliers itself: with >=20 frames and any rejection-based
            # combine, per-pixel stack rejection removes cosmic rays
            # statistically better than per-frame detection, and lacosmic is
            # the single most expensive Phase-1 step. Drizzle has no per-pixel
            # rejection, so it keeps lacosmic. Resolved before the resume
            # branch so checkpoint reloads see the same setting.
            if getattr(args, 'cosmic_ray_rejection', None) is None:
                _method = getattr(args, 'stack_method', 'auto')
                _drizzling = float(getattr(args, 'drizzle_scale', 1.0) or 1.0) > 1.0
                if n >= 20 and _method != 'mean' and not _drizzling:
                    args.cosmic_ray_rejection = False
                    safe_print(f"  NOTE: cosmic-ray rejection skipped: {n} frames with "
                               f"rejection stacking removes cosmic rays per-pixel "
                               f"(force with --cosmic-ray-rejection)")
                else:
                    args.cosmic_ray_rejection = True

            # ======================================================================
            # PHASE 1: Process & Analyse
            # ======================================================================
            if resume_phase >= 1:
                print_phase(1, "Processing & Quality Analysis (resumed from checkpoint)")
                final = restore_frame_state(lights, ckpt_state)
                _lights_index = {id(f): i for i, f in enumerate(lights)}
                final_indices = [_lights_index[id(f)] for f in final]
                stats.accepted_frames = len(final)
                stats.rejected_frames = n - len(final)
                safe_print(f"  Restored {len(final)}/{n} accepted frames — "
                           f"reloading pixel data (skipping quality analysis)...")
                reload_accepted_frames(final, final_indices, masters, args,
                                       mem_rgb, mem_lum, cached_lums,
                                       mm_rgb_path=mm_rgb_path,
                                       mm_lum_path=mm_lum_path,
                                       rgb_shape=rgb_shape,
                                       lum_shape=lum_shape)
                stats.quality_time = 0.0
            else:
                print_phase(1, "Processing & Quality Analysis")
                phase_start = time.time()

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
                    mm_mgr.cleanup()
                    return None

                _lights_index = {id(f): i for i, f in enumerate(lights)}
                final_indices = [_lights_index[id(f)] for f in final]

                # Free rejected frames' luminance — they are never needed again.
                accepted_set = set(final_indices)
                for _i in range(len(cached_lums)):
                    if _i not in accepted_set:
                        cached_lums[_i] = None

                # If the remaining cache would consume more than half of available
                # RAM, release it entirely and let registration read from the
                # mem_lum memmap instead (slightly slower, no RAM cost).
                try:
                    import psutil
                    _cache_bytes = sum(
                        a.nbytes for a in cached_lums if a is not None)
                    _avail_bytes = psutil.virtual_memory().available
                    if _cache_bytes > _avail_bytes * 0.5:
                        safe_print(
                            f"  NOTE: luminance cache ({_cache_bytes / 1e9:.1f} GB) "
                            f"exceeds 50 % of available RAM "
                            f"({_avail_bytes / 1e9:.1f} GB) — "
                            f"releasing cache; registration will read from disk")
                        for _i in range(len(cached_lums)):
                            cached_lums[_i] = None
                except Exception:
                    pass

                # Quality metrics CSV export
                if getattr(args, 'quality_report', None):
                    _write_quality_report(lights, rejected_reasons, args.quality_report)

                # Per-frame JPEG export
                if getattr(args, 'export_frames_dir', None):
                    _export_frame_jpegs(final, final_indices, mem_rgb,
                                        args.export_frames_dir, args)

                # Save checkpoint after phase 1
                save_checkpoint(output_path, phase=1, lights=lights, final=final, stats=stats)

                # Target inference from metadata (always runs; result written to header)
                from src.target_inference import infer_target_from_metadata
                _si = getattr(args, '_session_info', None)
                _inferred_name, _inferred_type, _inferred_conf, _inferred_src = \
                    infer_target_from_metadata(
                        _directory,
                        final,
                        use_simbad=True,
                        session_name=_si.object_name if _si else None,
                    )
                args._inferred_target     = _inferred_name
                args._inferred_type       = _inferred_type
                args._inferred_confidence = _inferred_conf
                args._inferred_source     = _inferred_src
                if _inferred_name and _inferred_type and _inferred_type != 'unknown':
                    safe_print(
                        f"\n  Target: {_inferred_name} "
                        f"[{_inferred_type.replace('_', ' ').title()}]  "
                        f"conf={_inferred_conf:.0%}  source={_inferred_src}"
                    )

                # Heuristic auto-advisor
                _run_auto_advisor(final, args,
                                  prior_type=_inferred_type,
                                  prior_confidence=_inferred_conf)

            if not final:
                print(f'\n  ERROR: No accepted frames after checkpoint restore!')
                mm_mgr.cleanup()
                return None

            # ======================================================================
            # PHASE 2: Registration
            # ======================================================================
            H, W = H_rgb, W_rgb

            if resume_phase >= 2:
                print_phase(2, "Registration (resumed from checkpoint)")
                shifts = [tuple(s) for s in ckpt_state.get('shifts', [[0.0, 0.0]] * len(final))]
                transforms = load_transforms(output_path, len(final))
                n_affine = sum(1 for t in transforms if t is not None)
                dither_info = ckpt_state.get('dither_info', {})
                # Displacement fields (--elastic-registration) and patch quality
                # maps (--patch-registration) are ndarray lists saved separately
                # as .npy (JSON can't hold them); restore so phase-3 stacking
                # reproduces the uninterrupted result rather than silently
                # dropping the elastic/patch-weighted correction.
                _df = load_field_list(output_path, 'displacement_fields', len(final))
                if _df is not None:
                    dither_info['displacement_fields'] = _df
                _qm = load_field_list(output_path, 'quality_maps', len(final))
                if _qm is not None:
                    dither_info['quality_maps'] = _qm
                n_disp = sum(1 for d in (_df or []) if d is not None)
                safe_print(f"  Restored {len(shifts)} frame shifts"
                           + (f", {n_affine} affine transforms" if n_affine else "")
                           + (f", {n_disp} displacement fields" if n_disp else "")
                           + " from checkpoint")
                stats.registration_time = 0.0
                del cached_lums
            else:
                print_phase(2, "Registration")
                phase_start = time.time()

                # Reference frame selection: alignment-centrality scoring
                # (does a cheap pyramid pass, then blends quality + centrality).
                # Falls back to pure quality if --no-alignment-centrality is set.
                pyramid_shifts = None
                if getattr(args, 'no_alignment_centrality', False):
                    best = max(final, key=lambda x: x.metrics.get('score', 0))
                    best_idx = lights.index(best)
                else:
                    best, best_idx, pyramid_shifts = select_reference_frame(
                        final, final_indices, mem_lum, H, W,
                        cached_lums=cached_lums,
                    )

                print(f"  Reference frame: {os.path.basename(best.path)} "
                      f"(score={best.metrics.get('score', 0):.1f})")

                ref_lum = np.array(mem_lum[best_idx])
                if args.verbose:
                    print(f'  Reference luminance: min={np.min(ref_lum):.1f}, '
                          f'max={np.max(ref_lum):.1f}, mean={np.mean(ref_lum):.1f}, '
                          f'std={np.std(ref_lum):.1f}')

                shifts, transforms, dither_info, final, final_indices = run_registration_phase(
                    final, final_indices, best, best_idx,
                    ref_lum, mem_lum, cached_lums, H, W, args, stats,
                    pyramid_shifts=pyramid_shifts)

                # The post-registration residual check (--reg-residual-reject)
                # can drop frames here, on top of Phase 1's quality gate --
                # keep the frame counts the summary reports in sync.
                stats.accepted_frames = len(final)
                stats.rejected_frames = stats.total_frames - len(final)

                stats.registration_time = time.time() - phase_start
                del cached_lums

                save_checkpoint(output_path, phase=2, lights=lights, final=final,
                                shifts=shifts, transforms=transforms,
                                dither_info=dither_info, stats=stats)

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

            quality_maps = dither_info.pop('quality_maps', None)
            displacement_fields = dither_info.pop('displacement_fields', None)
            stacked, fits_stacked, top, bottom, left, right = run_stacking_phase(
                final, final_indices, mem_rgb,
                shifts, transforms, H, W, C, args, stats,
                quality_maps=quality_maps,
                displacement_fields=displacement_fields)

            stats.stacking_time = time.time() - phase_start

            # Save blink comparator frames (before memmap cleanup)
            if getattr(args, 'keep_intermediates', False):
                _save_blink_frames(final, final_indices, mem_rgb, shifts, transforms,
                                   top, bottom, left, right, output_path)

            # Comet stacking: second pass aligned on comet nucleus
            if getattr(args, 'comet_mode', False):
                print_phase(3, "Comet Stacking (nucleus-aligned pass)")
                from src.registration import run_comet_registration_phase
                comet_shifts, comet_transforms, comet_dither_info = run_comet_registration_phase(
                    final, final_indices, best_idx,
                    ref_lum, mem_lum, H, W, args, stats)
                comet_stacked, _, ct, cb, cl, cr = run_stacking_phase(
                    final, final_indices, mem_rgb,
                    comet_shifts, comet_transforms, H, W, C, args, stats)
                comet_out = os.path.splitext(output_path)[0] + '_comet.fits'
                from astropy.io import fits as _cfits
                comet_hdu = _cfits.PrimaryHDU(
                    data=np.transpose(comet_stacked.astype(np.float32), (2, 0, 1))
                )
                comet_hdu.header['COMET'] = (True, 'Nucleus-aligned comet stack')
                comet_hdu.header['CREATOR'] = 'originstack.py comet_mode'
                comet_hdu.writeto(comet_out, overwrite=True)
                comet_prev = os.path.splitext(comet_out)[0] + '.jpg'
                save_preview_rgb(comet_stacked, comet_prev,
                                 stretch=getattr(args, 'stretch', 'ghs'),
                                 ghs_b=float(getattr(args, 'ghs_b', 8.0)),
                                 ghs_sp=float(getattr(args, 'ghs_sp', 0.15)),
                                 ghs_hp=float(getattr(args, 'ghs_hp', 0.95)))
                safe_print(f"  Comet stack: {os.path.basename(comet_out)}")

                # Blend star-aligned and comet-aligned stacks into a composite
                from src.stacking import blend_comet_star_stacks
                comet_lum = np.mean(comet_stacked, axis=2)
                tail_pa = comet_dither_info.get('tail_pa_deg', None)
                blended = blend_comet_star_stacks(
                    stacked, comet_stacked, comet_lum,
                    blend_sigma=float(getattr(args, 'comet_blend_sigma', 30.0)),
                    tail_pa_deg=tail_pa,
                )
                blended_out = os.path.splitext(output_path)[0] + '_comet_blended.fits'
                blended_hdu = _cfits.PrimaryHDU(
                    data=np.transpose(blended.astype(np.float32), (2, 0, 1))
                )
                blended_hdu.header['COMET'] = (True, 'Blended comet+star stack')
                blended_hdu.header['CREATOR'] = 'originstack.py comet_mode'
                if tail_pa is not None:
                    blended_hdu.header['TAIL_PA'] = (round(tail_pa, 1), 'Estimated tail PA (deg)')
                blended_hdu.writeto(blended_out, overwrite=True)
                blended_prev = os.path.splitext(blended_out)[0] + '.jpg'
                save_preview_rgb(blended, blended_prev,
                                 stretch=getattr(args, 'stretch', 'ghs'),
                                 ghs_b=float(getattr(args, 'ghs_b', 8.0)),
                                 ghs_sp=float(getattr(args, 'ghs_sp', 0.15)),
                                 ghs_hp=float(getattr(args, 'ghs_hp', 0.95)))
                safe_print(f"  Comet blended stack: {os.path.basename(blended_out)}")

            # Save raw stack so post-processing can be re-run without phases 1-3
            if getattr(args, 'keep_checkpoint', False):
                save_raw_stack(output_path, stacked)
                save_checkpoint(output_path, phase=3, lights=lights, final=final,
                                shifts=shifts, transforms=transforms,
                                dither_info=dither_info, stats=stats,
                                crop=[top, bottom, left, right])

        finally:
            mm_mgr.cleanup()
        # end: if resume_phase < 3

    # ======================================================================
    # Incremental merge: fold previously saved linear stacks into this run
    # before Phase 4, so post-processing runs once on the combined result.
    # ======================================================================
    merge_info = None
    if getattr(args, 'merge', None):
        if float(getattr(args, 'drizzle_scale', 1.0) or 1.0) > 1.0:
            raise ValueError("--merge is not supported with --drizzle-scale > 1 "
                             "(pixel grids differ between runs)")
        from src.merge import merge_previous_stacks
        safe_print(f"\n  Merging {len(args.merge)} previous stack(s)...")
        _t_merge = time.time()
        stacked, merge_info = merge_previous_stacks(
            stacked, len(final), list(args.merge), verbose=args.verbose)
        fits_stacked = stacked.copy()
        safe_print(f"  ✓ Merge: {merge_info['total_frames']} total frames "
                   f"({format_time(time.time() - _t_merge)})")
        # The preview depth rule capped preview_black_sigma using tonight's
        # frame count; the merged stack is deeper. Re-apply the rule with the
        # combined count, restoring toward the preset value it capped.
        _precap = getattr(args, '_preview_black_sigma_precap', None)
        if _precap is not None:
            _n_total = merge_info['total_frames']
            if _n_total >= 60:
                args.preview_black_sigma = _precap
            elif _n_total >= 20 and _precap > 2.0:
                args.preview_black_sigma = 2.0

    # ======================================================================
    # PHASE 4: Post-processing
    # ======================================================================
    print_phase(4, "Post-processing")
    phase_start = time.time()

    # Resolve diagnostic snapshot directory
    if getattr(args, 'diagnostic', False):
        _diag_dir = getattr(args, 'diagnostic_dir', None) or \
                    os.path.splitext(output_path)[0] + '_diagnostic'
        os.makedirs(_diag_dir, exist_ok=True)
        _h, _w, _c = stacked.shape
        _est_mb = (_h * _w * _c * 4 * 14) / (1024 ** 2)
        safe_print(f"\n  [diagnostic] Output folder: {_diag_dir}")
        safe_print(f"  [diagnostic] Estimated disk usage: ~{_est_mb:.0f} MB "
                   f"(up to 14 snapshots x {_h * _w * _c * 4 / (1024**2):.0f} MB each)")
        args._diagnostic_dir = _diag_dir
    else:
        args._diagnostic_dir = None

    stacked = postprocess_stack(stacked, args, final, stats)

    stats.post_processing_time = time.time() - phase_start

    # HDR multi-exposure blend (applied to post-processed stack)
    if getattr(args, 'hdr_combine', None):
        safe_print(f"\n  Applying HDR blend from {os.path.basename(args.hdr_combine)}...")
        hdr_start = time.time()
        stacked = _hdr_blend(stacked, args.hdr_combine, verbose=args.verbose)
        safe_print(f"  ✓ HDR blend ({format_time(time.time() - hdr_start)})")

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
        masters=masters, dither_info=dither_info,
        post_processed=False)
    if merge_info is not None:
        from src.merge import apply_merge_header
        apply_merge_header(hdu.header, merge_info)
    hdu.writeto(output_path, overwrite=True)

    # The output header already carries a WCS from the session info.json when the
    # capture app (e.g. Celestron Origin) plate-solved the session — written by
    # populate_fits_header via build_wcs_keywords. --plate-solve re-solves and
    # overwrites it with an astrometry.net/ASTAP solution.
    _si = getattr(args, '_session_info', None)
    _session_has_wcs = _si is not None and _si.has_wcs

    if getattr(args, 'plate_solve', False):
        if args.verbose:
            print("\n  Attempting plate solving...")
        solver   = getattr(args, 'plate_solver', 'astrometry')
        astap_bin = getattr(args, 'astap_path', None)
        if solve_plate(data_out, hdu.header, output_path,
                       verbose=args.verbose, solver=solver, astap_path=astap_bin):
            hdu.writeto(output_path, overwrite=True)
            _wcs_available = True
            safe_print("  Plate solved: WCS from astrometry solver")
        elif _session_has_wcs:
            _wcs_available = True
            safe_print("  Plate solve failed — output keeps WCS from session info.json")
        else:
            _wcs_available = False
            safe_print("  Plate solve failed — no WCS written")
    elif _session_has_wcs:
        # No solver run, but the session already provides a plate solution.
        _wcs_available = True
        safe_print("  Plate solving not run — output tagged with WCS from "
                   "session info.json (Origin session solve)")
    else:
        _wcs_available = False
        if args.verbose:
            print("\n  Plate solving skipped (use --plate-solve to enable)")

    if getattr(args, 'color_calibrate', False) and _wcs_available:
        safe_print("\n  Applying photometric colour calibration...")
        cc_start = time.time()
        try:
            from src.color_calibrate import run_photometric_calibration
            stacked_cc, scales = run_photometric_calibration(
                stacked, hdu.header, verbose=args.verbose)
            if scales != (1.0, 1.0, 1.0):
                stacked = stacked_cc
                hdu.data = np.transpose(stacked.astype(np.float32), (2, 0, 1))
                hdu.header['COLCAL'] = (True, 'Photometric colour calibration applied')
                hdu.header['COLCAL_R'] = (round(scales[0], 4), 'R scale factor')
                hdu.header['COLCAL_G'] = (round(scales[1], 4), 'G scale factor')
                hdu.header['COLCAL_B'] = (round(scales[2], 4), 'B scale factor')
                hdu.writeto(output_path, overwrite=True)
                safe_print(
                    f"  Colour calibration: "
                    f"R={scales[0]:.4f} G={scales[1]:.4f} B={scales[2]:.4f} "
                    f"({format_time(time.time() - cc_start)})"
                )
            else:
                safe_print("  Colour calibration: no correction applied (scale≈1.0)")
        except Exception as e:
            safe_print(f"  WARNING: colour calibration failed: {e}")

    # TIFF export (always written alongside FITS+JPG; --export tiff is now a no-op kept
    # for backward compatibility with existing scripts/configs)
    _save_tiff(stacked, output_path)

    # XISF export
    if getattr(args, 'output_xisf', False):
        _save_xisf(stacked, output_path, hdu.header)

    preview_path = os.path.splitext(output_path)[0] + '.jpg'
    stretch_method = getattr(args, 'stretch', 'linear')
    save_preview_rgb(stacked, preview_path, stretch=stretch_method,
                     ghs_b=float(getattr(args, 'ghs_b', 8.0)),
                     ghs_sp=float(getattr(args, 'ghs_sp', 0.15)),
                     ghs_hp=float(getattr(args, 'ghs_hp', 0.95)),
                     black_sigma=float(getattr(args, 'preview_black_sigma', 0.0)))

    crop_str = (f"(cropped {stats.cropped_pixels[0]}x{stats.cropped_pixels[1]} pixels)"
                if stats.cropped_pixels else "(crop info not available)")
    print(f"  Output size: {out_h}x{out_w} {crop_str}")

    try:
        from src.webview import get_webview
        _wv = get_webview()
        if _wv.active:
            _wv.preview(stacked, 'Final (post-processed)', args=args,
                        slot='final', min_interval=0.0)
            _fw = [f.metrics.get('fwhm', 0) for f in final
                   if f.metrics and f.metrics.get('fwhm', 0) > 0]
            _wv.summary(
                frames_stacked=f"{stats.accepted_frames} / {stats.total_frames}",
                avg_fwhm_px=(f"{np.mean(_fw):.2f}" if _fw else "n/a"),
                output=os.path.basename(output_path),
                total_time=format_time(stats.total_time()))
    except Exception:
        pass

    from src.utils import print_header
    print_header("SUMMARY", "=")
    print(f"  Frames analyzed:  {stats.total_frames}")
    print(f"  Frames stacked:   {stats.accepted_frames} "
          f"({stats.accepted_frames/stats.total_frames*100:.1f}%)")
    if stats.rejected_frames > 0:
        print(f"  Frames rejected:  {stats.rejected_frames}")
    # Integration time
    total_integration = 0.0
    for f in final:
        try:
            total_integration += float(f.header.get('EXPTIME', 0) or 0)
        except (ValueError, TypeError):
            pass
    if total_integration > 0:
        if total_integration >= 3600:
            int_str = f"{total_integration/3600:.1f} hours"
        elif total_integration >= 60:
            int_str = f"{total_integration/60:.1f} minutes"
        else:
            int_str = f"{total_integration:.0f} seconds"
        print(f"  Integration time: {int_str}")
    print(f"  Output:           {os.path.basename(output_path)} ({out_h}x{out_w}x3)")
    print(f"  Preview:          {os.path.basename(preview_path)} ({stretch_method} stretch)")
    # Stack quality summary
    fwhms = [f.metrics.get('fwhm', 0) for f in final if f.metrics and f.metrics.get('fwhm', 0) > 0]
    snrs = [f.metrics.get('snr', 0) for f in final if f.metrics]
    if fwhms:
        print(f"  Avg FWHM:         {np.mean(fwhms):.2f} px (best: {np.min(fwhms):.2f})")
    if snrs:
        print(f"  Avg SNR:          {np.mean(snrs):.1f} (best: {np.max(snrs):.1f})")
    # Per-phase timing with % of wall-clock so the bottleneck is obvious. The
    # four phases rarely sum to the total — "Other" captures frame discovery,
    # master-calibration building, plate-solve/WCS, colour calibration, and the
    # file writes between phases. A large "Other" means the limiter is outside
    # the four phases (usually I/O: master building or the output/memmap writes).
    _tt = max(stats.total_time(), 1e-9)
    _phases = [
        ("Quality+Load", stats.quality_time),
        ("Registration", stats.registration_time),
        ("Stacking", stats.stacking_time),
        ("Post-process", stats.post_processing_time),
    ]
    _other = max(_tt - sum(t for _, t in _phases), 0.0)
    print(f"  Processing time:  {format_time(_tt)}")
    for _name, _t in _phases:
        print(f"    {_name+':':<15} {format_time(_t):>8}  ({_t / _tt * 100:4.1f}%)")
    print(f"    {'Other (I/O):':<15} {format_time(_other):>8}  ({_other / _tt * 100:4.1f}%)")
    if HAS_PSUTIL:
        print(f"  Peak memory:      {stats.peak_memory_mb:.1f} MB")

    if stats.warnings:
        safe_print(f"\n  Warnings:")
        for w in stats.warnings[:5]:
            safe_print(f"    - {w}")
    if stats.errors:
        safe_print(f"\n  Errors: {len(stats.errors)}")

    safe_print(f"\n  Stack complete!")
    if getattr(args, 'keep_checkpoint', False):
        safe_print("  Checkpoint preserved (--keep-checkpoint). Re-run with same "
                   "output path to test different post-processing settings.")
    else:
        cleanup_checkpoint(output_path)

    return output_path
