"""Two-pass genuine streaming stack (``--stream``): O(1) full-resolution memory.

Sibling to ``--live`` (src/live_stack.py) for an ALREADY-COMPLETE directory
instead of watching a growing one. Two passes:

  Pass A (survey): load/calibrate/debayer/quality-analyze each frame once,
    apply the hard-limit-only quality gate. The calibrated+debayered RGB
    (post hot-pixel-removal -- the expensive part of a load) is written to
    a disk-backed memmap (``_MemmapManager``, reused from src/pipeline.py)
    instead of being discarded; only a lightweight FrameRecord (path,
    header, metrics including the star catalog, and the frame's slot index
    in that memmap) stays resident in RAM.

  Pass B (fold): for each accepted frame, read its calibrated RGB back from
    the disk cache (no re-load/re-calibrate/re-debayer -- `lum` is cheaply
    re-derived from the cached RGB) or, for the rare frame whose cache
    write failed (shape mismatch, first-frame-before-cache-existed), fall
    back to a full reload. Register against the fixed reference, warp, and
    fold into a running Welford (mean, M2, n_acc) accumulator via the
    online sigma-clip kernels (``online_sigma_clip_seed_burnin`` /
    ``online_sigma_clip_fold_frame`` in src/stacking.py).

This keeps RAM at O(1) in frame count (same as before) while avoiding the
double load+calibrate+debayer cost a naive two-pass design would pay twice
per frame -- at the cost of real disk space: the cache memmap is the same
size class as the batch pipeline's own ``mem_rgb``/``mem_aligned``
intermediates (one (N,H,W,C) float32 file). Not a RAM-vs-disk-free lunch,
an explicit trade the batch pipeline already makes for its own memmaps.

v1 scope (documented, not silently dropped):
  - Reference selection is argmax(quality score) -- the same fallback
    ``--no-alignment-centrality`` already uses in the batch pipeline, not
    the full shift-centrality blend (which needs per-frame image pyramids
    kept resident across the whole session -- a bigger v2 project).
  - ``quality_gate`` runs hard-limit only (no statistical/percentile stages
    -- both need population-wide stats this pass doesn't retain).
  - No ``--elastic-registration``, no drizzle, no patch-weighted-quality
    combine.
  - Sequential, no prefetch-while-folding threading.
  - Burn-in window (default 10 frames) held fully in RAM -- bounded, not
    O(N) -- on top of the disk cache.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.models import Config, FrameInfo, ProcessingStats
from src.utils import format_time, get_memory_usage_mb, safe_print

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from scipy import ndimage as _ndi
    _HAS_SCIPY = True
except Exception:
    _ndi = None
    _HAS_SCIPY = False


@dataclass
class FrameRecord:
    """Lightweight per-frame record retained after Pass A discards the
    full-resolution pixel data from RAM (but see ``cache_index`` -- the
    calibrated RGB itself lives on disk, not gone)."""
    path: str
    header: dict
    metrics: dict
    accepted: bool = True
    cache_index: Optional[int] = None


def survey(args, directory: str, masters: Dict, mem_mgr
          ) -> Tuple[List[FrameRecord], Optional[np.memmap]]:
    """Pass A: load/calibrate/analyze every light frame once, apply the
    hard-limit-only quality gate, retain lightweight metadata plus a disk
    cache of each frame's calibrated RGB (see module docstring).

    ``mem_mgr``: a ``src.pipeline._MemmapManager`` the caller owns and
    cleans up after Pass B is done with the cache (this function only
    creates the memmap; it does not close/delete it).

    Returns ``(records, rgb_cache)`` -- ``rgb_cache`` is ``None`` if no
    frame loaded successfully (nothing to cache).
    """
    from src.frame_discovery import discover_frames
    from src.frame_processor import _process_single_frame, quality_gate

    frames = discover_frames(directory)
    lights = frames.get('light', [])
    sb = getattr(args, '_session_bayer', None)
    verbose = getattr(args, 'verbose', False)

    light_infos: List[FrameInfo] = []
    cache_indices: List[Optional[int]] = []  # parallel to light_infos, by position
    rgb_cache: Optional[np.memmap] = None
    cache_shape: Optional[Tuple[int, int, int]] = None

    t_start = time.time()
    for i, fi in enumerate(tqdm(lights, desc='  STREAM survey', unit='frame',
                                disable=verbose or not lights)):
        t0 = time.time()
        try:
            res = _process_single_frame(
                fi.path, fi.header, masters,
                args.debayer_method, args.white_balance,
                ca_correction=getattr(args, 'ca_correction', False),
                cosmic_ray_rejection=False,
                advanced_metrics=False,
                session_bayer=sb,
                pre_gradient_removal=getattr(args, 'pre_gradient_removal', False),
                trail_reject=getattr(args, 'trail_reject', False))
        except Exception as exc:
            safe_print(f"  STREAM survey skip {os.path.basename(fi.path)}: {exc}")
            continue
        if res.get('error'):
            safe_print(f"  STREAM survey skip {os.path.basename(fi.path)}: {res['error']}")
            continue

        rgb = np.asarray(res['rgb'], dtype=np.float32)
        cache_idx: Optional[int] = None
        if rgb_cache is None:
            cache_shape = rgb.shape
            rgb_cache = mem_mgr.create('stream_rgb_', 'float32', (len(lights),) + cache_shape)
        if rgb.shape == cache_shape:
            rgb_cache[i] = rgb
            cache_idx = i
        else:
            safe_print(f"  STREAM survey: {os.path.basename(fi.path)} shape {rgb.shape} "
                       f"!= session shape {cache_shape}; will reload in fold")

        light_infos.append(FrameInfo(path=fi.path, type='light', header=fi.header,
                                      accepted=True, metrics=res.get('metrics') or {}))
        cache_indices.append(cache_idx)
        if verbose:
            m = light_infos[-1].metrics
            safe_print(f"    [{len(light_infos)}/{len(lights)}] {os.path.basename(fi.path)}: "
                       f"score={m.get('score', 0.0):.1f} stars={m.get('star_count', 0)} "
                       f"snr={m.get('snr', 0.0):.2f} ({time.time() - t0:.1f}s, "
                       f"total {format_time(time.time() - t_start)})")

    rejected_reasons: dict = {}
    stats = ProcessingStats()
    stats.total_frames = len(lights)
    quality_gate(light_infos, args, rejected_reasons, stats, stages=('hard_limit',))

    records = [FrameRecord(path=info.path, header=info.header, metrics=info.metrics,
                           accepted=info.accepted, cache_index=cidx)
               for info, cidx in zip(light_infos, cache_indices)]
    return records, rgb_cache


def _run_auto_advisor_for_stream(records: List[FrameRecord], args, directory: str) -> None:
    """--auto support for --stream: classify the target from the accepted
    frames' metadata/metrics and apply the same heuristic presets
    (deconvolve, denoise, SCNR, photometric calibration, etc.) the batch
    pipeline's _run_auto_advisor applies -- reusing that exact function
    rather than reimplementing it, so a preset change (like disabling
    deconvolve by default for nebula targets) only needs to happen once.

    Mirrors pipeline.py's normal-path call: must run BEFORE fold() since
    the presets affect registration (masked_correlation,
    pre_gradient_removal) as well as Phase 4 post-processing. No-ops if
    --auto wasn't passed (checked inside _run_auto_advisor itself).
    """
    if not getattr(args, 'auto', False):
        return
    from src.pipeline import _run_auto_advisor
    from src.target_inference import infer_target_from_metadata

    accepted = [r for r in records if r.accepted]
    final = [FrameInfo(path=r.path, type='light', header=r.header,
                       accepted=True, metrics=r.metrics) for r in accepted]

    session_info = getattr(args, '_session_info', None)
    name, target_type, confidence, source = infer_target_from_metadata(
        directory, final, use_simbad=True,
        session_name=session_info.object_name if session_info else None)
    if name and target_type and target_type != 'unknown':
        safe_print(f"\n  Target: {name} [{target_type.replace('_', ' ').title()}]  "
                   f"conf={confidence:.0%}  source={source}")
    _run_auto_advisor(final, args, prior_type=target_type, prior_confidence=confidence)


def select_reference(records: List[FrameRecord]) -> FrameRecord:
    """v1: argmax(quality score) -- the same fallback
    ``--no-alignment-centrality`` uses in the batch pipeline. Full
    shift-centrality selection needs per-frame image pyramids kept resident
    across the whole survey; deferred as a v2 project (see module docstring).
    """
    candidates = [r for r in records if r.accepted]
    if not candidates:
        raise ValueError("no frames survived the streaming quality gate")
    return max(candidates, key=lambda r: r.metrics.get('score', 0.0))


def _register_frame_to_reference(ref_lum: np.ndarray, ref_stars, img_lum: np.ndarray,
                                  img_stars, H: int, W: int, args
                                  ) -> Tuple[float, float, Optional[object]]:
    """Per-frame registration cascade against a FIXED reference, mirroring
    the cheap-cascade order production registration.py uses (seeded
    star-catalog match -> blind match -> pyramid/FFT translation fallback)
    -- but seed-free, since v1 has no pyramid seed_shifts available (that
    pyramid pass is itself the population-wide step the batch pipeline pays
    for once during reference selection; --stream's reference is picked by
    score alone, so there's no free seed to reuse here). Without a seed,
    ``match_stars_affine`` is skipped (it requires one) and the cascade
    starts at the seed-free blind match -- see registration.py's own
    handling of a None seed.

    Returns (shift_y, shift_x, transform_or_None, method) where ``method`` is
    one of 'blind_match', 'blind_match+seed', 'pyramid_fft' -- which tier of
    the cascade actually produced the result (for -v per-frame logging).
    """
    from src.registration import (
        HAS_SKIMAGE_TRANSFORM,
        _blind_match_transform,
        calculate_shift,
        match_stars_affine,
    )

    use_affine = (HAS_SKIMAGE_TRANSFORM and not getattr(args, 'no_affine', False)
                  and ref_stars is not None and img_stars is not None)
    if use_affine:
        affine_tf = _blind_match_transform(ref_stars, img_stars)
        method = 'blind_match'
        if affine_tf is None:
            sy, sx = calculate_shift(
                ref_lum, img_lum, verbose=False,
                skip_phase_cc=getattr(args, 'skip_phase_correlation', False),
                masked_correlation=getattr(args, 'masked_correlation', False),
                corr_downsample=2)
            affine_tf = match_stars_affine(ref_stars, img_stars, initial_shift=(sy, sx))
            method = 'blind_match+seed'
        if affine_tf is not None:
            tf_tx, tf_ty = affine_tf.params[0, 2], affine_tf.params[1, 2]
            tf_rot_deg = abs(np.degrees(np.arctan2(
                affine_tf.params[1, 0], affine_tf.params[0, 0])))
            if not (abs(tf_tx) > 0.1 * W or abs(tf_ty) > 0.1 * H
                    or tf_rot_deg > Config.AFFINE_MAX_ROTATION_DEG):
                return tf_ty, tf_tx, affine_tf, method
            # Unrealistic affine fit -- fall through to translation-only.

    sy, sx = calculate_shift(
        ref_lum, img_lum, verbose=getattr(args, 'verbose', False),
        skip_phase_cc=getattr(args, 'skip_phase_correlation', False),
        masked_correlation=getattr(args, 'masked_correlation', False))
    return sy, sx, None, 'pyramid_fft'


def _coverage_mask(H: int, W: int, shift: Tuple[float, float],
                   transform: Optional[object]) -> np.ndarray:
    """(H,W) float32 mask of pixels this frame's warp actually filled
    (matches LiveStacker's shift-coverage semantics, live_stack.py)."""
    from src.registration import apply_transform
    ones = np.ones((H, W), dtype=np.float32)
    if transform is not None:
        cov = apply_transform(ones[:, :, None], transform=transform)[:, :, 0]
    else:
        cov = _ndi.shift(ones, shift=shift, order=0, mode='constant', cval=0.0)
    return (cov > 0.5).astype(np.float32)


def fold(args, reference: FrameRecord, records: List[FrameRecord], masters: Dict,
        burn_in: int = 10, sigma: float = 3.0,
        rgb_cache: Optional[np.memmap] = None
        ) -> Tuple[np.ndarray, List[FrameInfo], List[Tuple[float, float]], float, int]:
    """Pass B: for each accepted frame, read its calibrated RGB back from
    ``rgb_cache`` (Pass A's disk-backed memmap -- no re-load/re-calibrate/
    re-debayer) via ``rec.cache_index``, or fall back to a full reload for
    the rare frame that wasn't cached (shape mismatch during survey).
    Register against ``reference``, warp, and fold into a running Welford
    accumulator.

    Returns (linear_stack, frame_infos, shifts, total_exposure, n_rejected_pixels).
    """
    from src.frame_processor import _process_single_frame
    from src.stacking import online_sigma_clip_fold_frame, online_sigma_clip_seed_burnin

    accepted = [r for r in records if r.accepted]
    if not accepted:
        raise ValueError("no accepted frames to fold")
    sb = getattr(args, '_session_bayer', None)

    def _load(rec: FrameRecord):
        return _process_single_frame(
            rec.path, rec.header, masters,
            args.debayer_method, args.white_balance,
            ca_correction=getattr(args, 'ca_correction', False),
            cosmic_ray_rejection=False, advanced_metrics=False,
            session_bayer=sb,
            pre_gradient_removal=getattr(args, 'pre_gradient_removal', False),
            trail_reject=getattr(args, 'trail_reject', False))

    def _get_rgb_lum(rec: FrameRecord):
        """Cache hit: read RGB back from disk, re-derive lum cheaply (no
        re-calibration/re-debayer). Cache miss: full reload, same as before
        caching existed."""
        if rec.cache_index is not None and rgb_cache is not None:
            rgb = np.array(rgb_cache[rec.cache_index], dtype=np.float32)
            lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
            return {'rgb': rgb, 'lum': lum, 'error': None}
        return _load(rec)

    ref_res = _get_rgb_lum(reference)
    if ref_res.get('error'):
        raise ValueError(f"reference frame failed to reload: {ref_res['error']}")
    ref_lum = np.asarray(ref_res['lum'], dtype=np.float32)
    ref_stars = reference.metrics.get('_star_sources')
    H, W = ref_lum.shape

    frame_infos: List[FrameInfo] = []
    shifts: List[Tuple[float, float]] = []
    total_exposure = 0.0
    n_rejected_pixels = 0

    burn_frames: List[np.ndarray] = []
    burn_covs: List[np.ndarray] = []
    mean = m2 = n_acc = None

    verbose = getattr(args, 'verbose', False)
    t_start = time.time()
    for i, rec in enumerate(accepted):
        if rec.path == reference.path:
            res, sy, sx, tf, method = ref_res, 0.0, 0.0, None, 'reference'
        else:
            res = _get_rgb_lum(rec)
            if res.get('error'):
                safe_print(f"  STREAM reject {os.path.basename(rec.path)}: {res['error']}")
                continue
            lum = np.asarray(res['lum'], dtype=np.float32)
            if lum.shape != (H, W):
                safe_print(f"  STREAM reject {os.path.basename(rec.path)}: shape "
                           f"{lum.shape} != reference {(H, W)}")
                continue
            img_stars = rec.metrics.get('_star_sources')
            sy, sx, tf, method = _register_frame_to_reference(
                ref_lum, ref_stars, lum, img_stars, H, W, args)
            if not (np.isfinite(sy) and np.isfinite(sx)) or abs(sy) > 0.3 * H or abs(sx) > 0.3 * W:
                safe_print(f"  STREAM reject {os.path.basename(rec.path)}: bad shift "
                           f"({sx:.1f}, {sy:.1f})")
                continue

        rgb = np.asarray(res['rgb'], dtype=np.float32)
        aligned = apply_transform_dispatch(rgb, sy, sx, tf)
        cov = _coverage_mask(H, W, (sy, sx), tf)

        frame_infos.append(FrameInfo(path=rec.path, type='light', header=rec.header,
                                     accepted=True, metrics=rec.metrics, shift=(sy, sx)))
        shifts.append((sy, sx))
        try:
            total_exposure += float((rec.header or {}).get('EXPTIME', 0) or 0)
        except (TypeError, ValueError):
            pass

        if verbose:
            cov_pct = 100.0 * float(cov.mean())
            safe_print(f"    [{len(shifts)}/{len(accepted)}] {os.path.basename(rec.path)}: "
                       f"method={method} shift=({sx:+.2f}, {sy:+.2f})px "
                       f"transform={'affine' if tf is not None else 'translation'} "
                       f"coverage={cov_pct:.1f}%")

        if mean is None:
            burn_frames.append(aligned)
            burn_covs.append(cov)
            if len(burn_frames) < burn_in and i < len(accepted) - 1:
                continue
            burn_stack = np.stack(burn_frames, axis=0).astype(np.float32)
            cov_stack = np.stack(burn_covs, axis=0).astype(np.float32)
            mean, m2, n_acc, n_rej = online_sigma_clip_seed_burnin(
                burn_stack, cov_stack, sigma=sigma)
            n_rejected_pixels += n_rej
            burn_frames, burn_covs = [], []  # free -- state now lives in mean/m2/n_acc
            safe_print(f"  STREAM burn-in seeded from {len(shifts)} frames "
                       f"({n_rej} pixel-samples rejected)")
            continue

        n_rej = online_sigma_clip_fold_frame(mean, m2, n_acc, aligned, cov, sigma=sigma)
        n_rejected_pixels += n_rej
        if verbose:
            safe_print(f"    -> folded, {n_rej} pixel-samples rejected this frame")
        if (i + 1) % 5 == 0 or i == len(accepted) - 1:
            safe_print(f"  STREAM folded {len(shifts)}/{len(accepted)} "
                       f"({format_time(time.time() - t_start)}, "
                       f"{get_memory_usage_mb():.0f} MB)")

    if mean is None:
        raise ValueError("no frames survived registration in the fold pass")

    return mean.astype(np.float32), frame_infos, shifts, total_exposure, n_rejected_pixels


def apply_transform_dispatch(rgb: np.ndarray, sy: float, sx: float, tf) -> np.ndarray:
    """Thin wrapper so `transform` (when present) always wins over `shift` --
    same precedence apply_transform itself applies, made explicit at the
    call site for readability."""
    from src.registration import apply_transform
    if tf is not None:
        return apply_transform(rgb, transform=tf)
    return apply_transform(rgb, shift=(sy, sx))


def run_stream_stack(args) -> int:
    """Entry point for ``--stream``: survey, pick a reference, fold, then run
    the same Phase 4 post-processing + output the batch pipeline uses."""
    if not _HAS_SCIPY:
        safe_print("  ERROR: --stream requires scipy")
        return 1
    directory = args.directory
    if not os.path.isdir(directory):
        safe_print(f"  ERROR: not a directory: {directory}")
        return 1

    from src.live_stack import build_session_masters
    from src.pipeline import _MemmapManager
    masters, _frames = build_session_masters(args, directory)

    t0 = time.time()
    safe_print(f"\n  STREAM: surveying {directory} ...")
    with _MemmapManager() as mem_mgr:
        records, rgb_cache = survey(args, directory, masters, mem_mgr)
        n_total = len(records)
        n_accepted = sum(1 for r in records if r.accepted)
        safe_print(f"  STREAM survey: {n_accepted}/{n_total} frames accepted "
                   f"({format_time(time.time() - t0)})")
        if n_accepted == 0:
            safe_print("  ERROR: no frames survived the streaming quality gate")
            return 1

        _run_auto_advisor_for_stream(records, args, directory)

        reference = select_reference(records)
        safe_print(f"  STREAM reference: {os.path.basename(reference.path)} "
                   f"(score={reference.metrics.get('score', 0.0):.1f})")

        burn_in = int(getattr(args, 'stream_burnin', 10) or 10)
        sigma = float(getattr(args, 'stream_sigma', None) or getattr(args, 'rejection_sigma', None) or 3.0)
        t1 = time.time()
        stacked, frame_infos, shifts, total_exposure, n_rej = fold(
            args, reference, records, masters, burn_in=burn_in, sigma=sigma,
            rgb_cache=rgb_cache)
        safe_print(f"  STREAM fold: {len(frame_infos)} frames combined "
                   f"({format_time(time.time() - t1)}, peak {get_memory_usage_mb():.0f} MB)")
        # mem_mgr.cleanup() (via __exit__) removes the disk cache here --
        # everything past this point works from `stacked` (in RAM) only.

    stats = ProcessingStats()
    stats.total_frames = n_total
    stats.accepted_frames = len(frame_infos)
    stats.rejected_frames = n_total - len(frame_infos)
    stats.registration_time = time.time() - t1

    args.stack_method = 'online_sigma_clip'

    from astropy.io import fits

    from src.io_fits import populate_fits_header, save_preview_rgb
    from src.pipeline import _save_tiff
    from src.postprocess import postprocess_stack

    fits_stacked = stacked.copy()

    t2 = time.time()
    processed = postprocess_stack(stacked, args, frame_infos, stats)
    stats.post_processing_time = time.time() - t2

    output_path = args.output
    hdu = fits.PrimaryHDU()
    hdu.data = np.transpose(fits_stacked, (2, 0, 1)).astype(np.float32)
    populate_fits_header(
        header=hdu.header, frames=frame_infos, stats=stats, args=args,
        stacked_shape=fits_stacked.shape, shifts=shifts,
        masters=masters, dither_info=None, post_processed=False)
    if total_exposure > 0:
        hdu.header['TOTEXP'] = (round(total_exposure, 1), 'Total exposure (s)')
    hdu.header['STREAMED'] = (True, 'Produced by --stream (frame-at-a-time)')
    hdu.writeto(output_path, overwrite=True)

    if getattr(args, 'output_tiff', False):
        _save_tiff(processed, output_path)

    preview_path = os.path.splitext(output_path)[0] + '.jpg'
    save_preview_rgb(processed, preview_path, stretch=getattr(args, 'stretch', 'ghs'),
                     ghs_b=float(getattr(args, 'ghs_b', 8.0)),
                     ghs_sp=float(getattr(args, 'ghs_sp', 0.15)),
                     ghs_hp=float(getattr(args, 'ghs_hp', 0.95)),
                     black_sigma=float(getattr(args, 'preview_black_sigma', 0.0) or 0.0))

    safe_print(f"\n  STREAM complete: {len(frame_infos)}/{n_total} frames, "
              f"{n_rej} pixel-samples rejected, total {format_time(time.time() - t0)}")
    safe_print(f"  Output: {output_path}")
    safe_print(f"  TIFF: {os.path.splitext(output_path)[0] + '.tiff'}")
    safe_print(f"  Preview: {preview_path}")
    return 0
