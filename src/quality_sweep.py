"""Collection quality sweep: walk a folder tree, score every light frame,
and flag poor ones by renaming (``Light0001.fits`` -> ``Light0001.fits.rejected``).

The extension suffix hides flagged files from all future frame discovery
(``discover_frames`` matches only ``.fit``/``.fits`` endings) and is trivially
reversible with ``--sweep-undo``.

Judgement reuses the pipeline's own machinery end to end: frames are scored
with ``compute_quality_metrics`` on uncalibrated debayered luminance (the
metrics are MAD-robust and need no masters), then each folder's lights are
passed through ``quality_gate`` — the exact hard-reject / statistical-outlier
/ relative-score decision the stacker applies, folder-relative and honoring
``--quality-threshold``. Dry-run by default; ``--apply`` performs renames.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.models import FrameInfo, ProcessingStats
from src.utils import format_time, safe_print

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

REJECT_SUFFIX = '.rejected'


def _prefetch_one(path: str) -> None:
    """Read a file fully to warm the OS page cache; result is discarded."""
    try:
        with open(path, 'rb') as fh:
            while fh.read(1 << 20):
                pass
    except OSError:
        pass


def _prefetch_files(lights: List[FrameInfo], pool: ThreadPoolExecutor) -> None:
    """Fire-and-forget: warm the OS page cache for a folder's files ahead of
    the CPU-bound ProcessPoolExecutor workers, so a slow or network drive's
    read latency overlaps with scoring instead of stalling it. SER virtual
    frames (``path::index``) share one real file -- dedupe so a folder of
    many sub-frames doesn't re-read the same video repeatedly."""
    from src.io_ser import _split_vpath, is_ser_virtual_path

    seen = set()
    for f in lights:
        real = _split_vpath(f.path)[0] if is_ser_virtual_path(f.path) else f.path
        if real in seen:
            continue
        seen.add(real)
        pool.submit(_prefetch_one, real)


def _score_one(path: str) -> Tuple[str, Optional[dict], Optional[str]]:
    """ProcessPool worker: load -> luminance proxy -> gate-only quality metrics.

    Uncalibrated on purpose — the sweep judges raw lights without needing the
    session's masters, and the metrics degrade gracefully without calibration.

    A 2D (Bayer) frame is reduced to a half-resolution luminance proxy by
    averaging each 2x2 mosaic cell instead of running a full Malvar debayer:
    every 2x2 cell holds exactly one R, two G and one B sample for any RGGB-
    family pattern, so the cell mean ~= 0.25R + 0.5G + 0.25B — close enough to
    Rec.601 luma for a keep/reject gate, and it skips the single most
    expensive step per frame. FWHM (measured on the half-res proxy) is scaled
    back to full-resolution pixels before returning.
    """
    try:
        from src.io_fits import load_frame
        from src.quality import compute_quality_metrics

        data, hdr = load_frame(path)
        if data is None or data.size == 0:
            return path, None, 'empty data array'
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 2:
            h, w = arr.shape[0] & ~1, arr.shape[1] & ~1
            lum = 0.25 * (arr[0:h:2, 0:w:2] + arr[0:h:2, 1:w:2]
                          + arr[1:h:2, 0:w:2] + arr[1:h:2, 1:w:2])
            fwhm_scale = 2.0
        else:
            lum = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1]
                   + 0.114 * arr[:, :, 2]).astype(np.float32)
            fwhm_scale = 1.0
        metrics = compute_quality_metrics(lum, advanced_metrics=False, gate_only=True)
        if fwhm_scale != 1.0 and metrics.get('fwhm'):
            metrics['fwhm'] *= fwhm_scale
        metrics.pop('_star_sources', None)  # not consumed by the sweep; keep IPC/cache small
        return path, metrics, None
    except Exception as exc:
        return path, None, f'{type(exc).__name__}: {exc}'


# Bump when _score_one's metric logic changes so stale caches are ignored.
_CACHE_VERSION = 2
_CACHE_NAME = '.sweepcache.json'


def _load_cache(root: str) -> dict:
    """Return {abspath: {"mt": mtime_ns, "sz": size, "m": metrics}} or {}."""
    try:
        with open(os.path.join(root, _CACHE_NAME), encoding='utf-8') as fh:
            blob = json.load(fh)
        if blob.get('v') != _CACHE_VERSION:
            return {}
        return blob.get('frames', {})
    except (OSError, ValueError):
        return {}


def _save_cache(root: str, frames: dict) -> None:
    try:
        tmp = os.path.join(root, _CACHE_NAME + '.tmp')
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump({'v': _CACHE_VERSION, 'frames': frames}, fh)
        os.replace(tmp, os.path.join(root, _CACHE_NAME))
    except OSError as exc:
        safe_print(f"  (could not write {_CACHE_NAME}: {exc})")


def _walk_light_folders(root: str) -> List[Tuple[str, List[FrameInfo]]]:
    """Recursively collect (directory, lights) for every folder containing
    light frames, using the pipeline's own classification (darks/flats/bias
    and pipeline outputs are excluded)."""
    from src.frame_discovery import discover_frames

    folders: List[Tuple[str, List[FrameInfo]]] = []
    for dirpath, dirnames, _ in os.walk(root):
        dirnames.sort()
        try:
            frames = discover_frames(dirpath)
        except Exception:
            continue
        lights = frames.get('light', [])
        if lights:
            folders.append((dirpath, lights))
    return folders


def run_quality_sweep(root: str, args) -> int:
    """Score every light under ``root`` and flag poor ones. Returns exit code."""
    from src.frame_processor import _pin_worker_to_single_thread, quality_gate

    apply_renames = bool(getattr(args, 'apply', False))
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        safe_print(f"ERROR: not a directory: {root}")
        return 1

    safe_print("=" * 70)
    safe_print("Quality sweep" + ("" if apply_renames else " (dry run — no files renamed)"))
    safe_print("=" * 70)
    safe_print(f"Root: {root}")

    folders = _walk_light_folders(root)
    n_total = sum(len(l) for _, l in folders)
    if not folders:
        safe_print("No light frames found.")
        return 0
    safe_print(f"Found {n_total} lights in {len(folders)} folder(s)\n")

    workers = getattr(args, 'parallel', 0) or (os.cpu_count() or 4)
    t0 = time.time()
    n_flagged_total = 0
    n_renamed = 0
    n_cache_hits = 0
    csv_rows: List[str] = []

    use_cache = not getattr(args, 'sweep_no_cache', False)
    cache = _load_cache(root) if use_cache else {}
    new_cache: dict = {}

    def _cache_probe(fpath: str):
        """Return (key, stat_tuple, cached_metrics|None) for a frame."""
        from src.io_ser import is_ser_virtual_path
        if not use_cache or is_ser_virtual_path(fpath):
            return None, None, None
        key = os.path.abspath(fpath)
        try:
            st = os.stat(fpath)
        except OSError:
            return key, None, None
        sig = [st.st_mtime_ns, st.st_size]
        hit = cache.get(key)
        if hit and hit.get('sig') == sig:
            return key, sig, hit.get('m')
        return key, sig, None

    prefetch_workers = max(4, min(16, workers * 2))
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_pin_worker_to_single_thread) as pool, \
         ThreadPoolExecutor(max_workers=prefetch_workers) as prefetch_pool:
        if folders:
            _prefetch_files(folders[0][1], prefetch_pool)

        for idx, (dirpath, lights) in enumerate(folders):
            rel = os.path.relpath(dirpath, root)

            # Serve unchanged frames straight from the on-disk cache; only
            # submit the rest to the worker pool.
            futs = {}
            sigs: Dict[str, list] = {}
            for f in lights:
                key, sig, cached = _cache_probe(f.path)
                if key is not None:
                    sigs[f.path] = [key, sig]
                if cached is not None:
                    f.metrics = dict(cached)
                    new_cache[key] = {'sig': sig, 'm': cached}
                    n_cache_hits += 1
                else:
                    futs[pool.submit(_score_one, f.path)] = f
            # Overlap the next folder's disk I/O with this folder's CPU-bound
            # scoring instead of waiting on it at the top of the next iteration.
            if idx + 1 < len(folders):
                _prefetch_files(folders[idx + 1][1], prefetch_pool)
            n_err = 0
            for fut in tqdm(as_completed(futs), total=len(futs),
                            desc=f"  {rel}", unit="frame",
                            disable=getattr(args, 'verbose', False)):
                path, metrics, err = fut.result()
                f = futs[fut]
                if err:
                    # Unreadable counts as a hard failure worth flagging.
                    f.metrics = {'score': 0.0, 'star_count': 0, 'snr': 0.0,
                                 'contrast': 0.0, 'dynamic_range': 0.0,
                                 '_sweep_error': err}
                    n_err += 1
                else:
                    f.metrics = metrics
                    ks = sigs.get(path)
                    if ks and ks[1] is not None:
                        new_cache[ks[0]] = {'sig': ks[1], 'm': metrics}

            # The pipeline's own gate: hard rejects + statistical outliers +
            # relative score threshold, folder-relative. The sweep runs the
            # statistical-outlier stage tighter (2.0 sigma vs the stacking
            # pipeline's default 2.5) since it's an advisory pass reviewed by
            # a human before --apply renames anything, not a silent stacking
            # decision — worth catching more marginal frames.
            rejected_reasons: Dict[str, str] = {}
            stats = ProcessingStats()
            quality_gate(lights, args, rejected_reasons, stats, outlier_sigma=2.0)

            flagged = [f for f in lights if not f.accepted]
            n_flagged_total += len(flagged)

            # Reason tally for the folder summary line
            tally: Dict[str, int] = {}
            for f in flagged:
                reason = rejected_reasons.get(f.path, 'low score')
                key = reason.split('(')[0].strip()
                tally[key] = tally.get(key, 0) + 1
            tally_s = ", ".join(f"{v} {k}" for k, v in sorted(tally.items()))
            err_s = f"; {n_err} unreadable" if n_err else ""
            safe_print(f"  {rel}: {len(lights)} lights, {len(flagged)} flagged"
                       + (f" ({tally_s})" if flagged else "") + err_s)

            for f in lights:
                m = f.metrics or {}
                csv_rows.append(
                    f"{f.path},{m.get('snr', 0):.2f},{m.get('fwhm', 0):.2f},"
                    f"{m.get('star_count', 0)},{m.get('score', 0):.1f},"
                    f"{f.accepted},{rejected_reasons.get(f.path, '')}")

            if apply_renames:
                for f in flagged:
                    from src.io_ser import is_ser_virtual_path
                    if is_ser_virtual_path(f.path):
                        safe_print(f"    SKIP (SER frame, cannot rename in-place): {f.path}")
                        continue
                    dst = f.path + REJECT_SUFFIX
                    if os.path.exists(dst):
                        safe_print(f"    SKIP (exists): {os.path.basename(dst)}")
                        continue
                    try:
                        os.rename(f.path, dst)
                        n_renamed += 1
                        if getattr(args, 'verbose', False):
                            safe_print(f"    renamed: {os.path.basename(f.path)}"
                                       f" -> {os.path.basename(dst)}")
                    except OSError as exc:
                        safe_print(f"    ERROR renaming "
                                   f"{os.path.basename(f.path)}: {exc}")

    if use_cache:
        _save_cache(root, new_cache)

    report_path = getattr(args, 'quality_report', None)
    if report_path:
        with open(report_path, 'w', encoding='utf-8') as fh:
            fh.write("filename,snr,fwhm,star_count,quality_score,"
                     "accepted,rejection_reason\n")
            fh.write("\n".join(csv_rows) + "\n")
        safe_print(f"\nPer-frame CSV: {report_path}")

    safe_print("")
    safe_print("=" * 70)
    n_scored = n_total - n_cache_hits
    safe_print(f"Swept {n_total} lights in {format_time(time.time() - t0)} "
               f"({n_scored} scored, {n_cache_hits} from cache): "
               f"{n_flagged_total} flagged"
               + (f", {n_renamed} renamed to *{REJECT_SUFFIX}" if apply_renames
                  else ""))
    if not apply_renames and n_flagged_total:
        safe_print("Dry run — re-run with --apply to rename flagged files "
                   "(reversible with --sweep-undo)")
    elif apply_renames and n_renamed:
        safe_print("Renamed files are invisible to stacking; restore any time "
                   "with --sweep-undo")
    safe_print("=" * 70)
    return 0


def undo_quality_sweep(root: str) -> int:
    """Strip the ``.rejected`` suffix from every flagged file under ``root``."""
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        safe_print(f"ERROR: not a directory: {root}")
        return 1
    n_restored = 0
    n_skipped = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in filenames:
            if not name.endswith(REJECT_SUFFIX):
                continue
            src = os.path.join(dirpath, name)
            dst = os.path.join(dirpath, name[:-len(REJECT_SUFFIX)])
            if os.path.exists(dst):
                safe_print(f"  SKIP (target exists): {dst}")
                n_skipped += 1
                continue
            try:
                os.rename(src, dst)
                n_restored += 1
            except OSError as exc:
                safe_print(f"  ERROR restoring {src}: {exc}")
                n_skipped += 1
    safe_print(f"Restored {n_restored} file(s)"
               + (f", {n_skipped} skipped" if n_skipped else ""))
    return 0
