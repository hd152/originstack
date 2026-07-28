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

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.models import FrameInfo, ProcessingStats
from src.utils import safe_print, format_time

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

REJECT_SUFFIX = '.rejected'


def _score_one(path: str) -> Tuple[str, Optional[dict], Optional[str]]:
    """ProcessPool worker: load -> debayer -> luminance -> quality metrics.

    Uncalibrated on purpose — the sweep judges raw lights without needing the
    session's masters, and the metrics degrade gracefully without calibration.
    """
    try:
        from src.io_fits import load_frame
        from src.debayer import debayer, green_equalize
        from src.quality import compute_quality_metrics

        data, hdr = load_frame(path)
        if data is None or data.size == 0:
            return path, None, 'empty data array'
        if data.ndim == 2:
            bayer = hdr.get('BAYERPAT', hdr.get('COLORTYP', 'RGGB'))
            data = green_equalize(np.asarray(data), pattern=bayer)
            rgb = debayer(np.asarray(data), pattern=bayer, method='bilinear')
        else:
            rgb = data
        lum = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1]
               + 0.114 * rgb[:, :, 2]).astype(np.float32)
        metrics = compute_quality_metrics(lum, advanced_metrics=False)
        return path, metrics, None
    except Exception as exc:
        return path, None, f'{type(exc).__name__}: {exc}'


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
    from src.frame_processor import quality_gate, _pin_worker_to_single_thread

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
    csv_rows: List[str] = []

    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_pin_worker_to_single_thread) as pool:
        for dirpath, lights in folders:
            rel = os.path.relpath(dirpath, root)
            futs = {pool.submit(_score_one, f.path): f for f in lights}
            n_err = 0
            for fut in tqdm(as_completed(futs), total=len(lights),
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

    report_path = getattr(args, 'quality_report', None)
    if report_path:
        with open(report_path, 'w', encoding='utf-8') as fh:
            fh.write("filename,snr,fwhm,star_count,quality_score,"
                     "accepted,rejection_reason\n")
            fh.write("\n".join(csv_rows) + "\n")
        safe_print(f"\nPer-frame CSV: {report_path}")

    safe_print("")
    safe_print("=" * 70)
    safe_print(f"Swept {n_total} lights in {format_time(time.time() - t0)}: "
               f"{n_flagged_total} flagged"
               + (f", {n_renamed} renamed to *{REJECT_SUFFIX}" if apply_renames
                  else ""))
    if not apply_renames and n_flagged_total:
        safe_print(f"Dry run — re-run with --apply to rename flagged files "
                   f"(reversible with --sweep-undo)")
    elif apply_renames and n_renamed:
        safe_print(f"Renamed files are invisible to stacking; restore any time "
                   f"with --sweep-undo")
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
