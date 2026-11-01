"""Per-frame health report (``--frame-report``): why each light was kept or rejected.

Built from data the pipeline already has -- ``rejected_reasons`` from the Phase 1
``quality_gate``, the Phase 2 drop notes registration leaves in ``FrameInfo.metrics``
(``phase2_drop``), and the frame metrics -- so it changes no gate and no number.
Written as ``<stem>_frames.csv`` / ``<stem>_frames.json`` and published through
``ui_events`` for the desktop app.
"""
import csv
import json
import os
from typing import Dict, List, Optional

import numpy as np

from src.utils import safe_print

# A kept frame is annotated (never rejected) when it is this far off the session median.
FWHM_FLAG_RATIO = 1.5
STARS_FLAG_RATIO = 0.5
BACKGROUND_FLAG_RATIO = 1.5

COLUMNS = ('filename', 'verdict', 'stage', 'reason', 'notes', 'score', 'snr', 'fwhm',
           'ellipticity', 'star_count', 'background', 'shift_x', 'shift_y',
           'reg_residual_px', 'transparency')


def _num(v) -> Optional[float]:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _stage(reason: str) -> str:
    """Which gate produced a Phase 1 rejection reason."""
    r = reason.lower()
    if r.startswith('statistical outlier'):
        return 'statistical'
    if r.startswith('score '):
        return 'percentile'
    if r.startswith(('no stars', 'extremely low')):
        return 'hard_limit'
    return 'load'


def _median(lights, key) -> Optional[float]:
    vals = [_num((f.metrics or {}).get(key)) for f in lights]
    vals = [v for v in vals if v is not None and v > 0]
    return float(np.median(vals)) if vals else None


def build_frame_report(lights, rejected_reasons: Dict[str, str], final,
                       max_ellipticity: float = 0.5) -> List[dict]:
    """One row per light: verdict (kept/rejected), the gate and plain-language reason
    it fell at, advisory notes for frames far off the session median, and the key
    metrics."""
    kept = {id(f) for f in final}
    med = {k: _median(lights, k) for k in ('fwhm', 'star_count', 'background')}
    rows = []
    for f in lights:
        m = f.metrics or {}
        stage, reason = '', ''
        if id(f) in kept:
            verdict = 'kept'
        else:
            verdict = 'rejected'
            if f.path in rejected_reasons:
                reason = rejected_reasons[f.path]
                stage = _stage(reason)
            elif m.get('phase2_drop'):
                reason = str(m['phase2_drop'])
                stage = 'registration' if 'residual' in reason else 'transparency'
            else:
                reason, stage = 'rejected (no reason recorded)', 'unknown'
        notes = []
        fwhm, stars, bg = _num(m.get('fwhm')), _num(m.get('star_count')), _num(m.get('background'))
        if fwhm and med['fwhm'] and fwhm > FWHM_FLAG_RATIO * med['fwhm']:
            notes.append(f"FWHM {fwhm:.1f} px > {FWHM_FLAG_RATIO:g}x median ({med['fwhm']:.1f})")
        if stars is not None and med['star_count'] and stars < STARS_FLAG_RATIO * med['star_count']:
            notes.append(f"{int(stars)} stars < {STARS_FLAG_RATIO:g}x median ({med['star_count']:.0f})")
        if bg and med['background'] and bg > BACKGROUND_FLAG_RATIO * med['background']:
            notes.append(f"background {bg:.0f} > {BACKGROUND_FLAG_RATIO:g}x median ({med['background']:.0f})")
        ell = _num(m.get('ellipticity'))
        if max_ellipticity and ell is not None and ell > max_ellipticity:
            notes.append(f"ellipticity {ell:.2f} > {max_ellipticity:g}")
        sh = f.shift or (0.0, 0.0)
        rows.append({
            'filename': os.path.basename(f.path), 'verdict': verdict, 'stage': stage,
            'reason': reason, 'notes': '; '.join(notes),
            'score': _num(m.get('score')), 'snr': _num(m.get('snr')), 'fwhm': fwhm,
            'ellipticity': ell, 'star_count': None if stars is None else int(stars),
            'background': bg,
            'shift_x': _num(sh[0]) if len(sh) > 1 else None,
            'shift_y': _num(sh[1]) if len(sh) > 1 else None,
            'reg_residual_px': _num(m.get('reg_residual_px')),
            'transparency': _num(m.get('transparency')),
        })
    return rows


def write_frame_report(rows: List[dict], output_path: str) -> List[str]:
    """Write ``<stem>_frames.csv`` and ``.json`` next to the output; return the paths."""
    stem = os.path.splitext(output_path)[0]
    counts: Dict[str, int] = {}
    for r in rows:
        key = 'kept' if r['verdict'] == 'kept' else r['stage']
        counts[key] = counts.get(key, 0) + 1
    paths = []
    try:
        with open(stem + '_frames.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(COLUMNS))
            w.writeheader()
            w.writerows({k: ('' if r[k] is None else r[k]) for k in COLUMNS} for r in rows)
        paths.append(stem + '_frames.csv')
        with open(stem + '_frames.json', 'w', encoding='utf-8') as fh:
            json.dump({'summary': counts, 'frames': rows}, fh, indent=1)
        paths.append(stem + '_frames.json')
    except OSError as e:
        safe_print(f"  WARNING: could not write frame report: {e}")
    return paths


def emit_frame_report(lights, rejected_reasons, final, args, output_path: str) -> List[dict]:
    """Build, write and publish the report; with ``-v`` print counts per reason."""
    rows = build_frame_report(lights, rejected_reasons, final,
                              float(getattr(args, 'max_ellipticity', 0.5) or 0.0))
    paths = write_frame_report(rows, output_path)
    try:
        from src.ui_events import get_ui_events
        get_ui_events().frame_report(rows)
    except Exception:
        pass
    if len(paths) == 2:
        safe_print(f"  Frame report: {os.path.basename(paths[0])}, "
                   f"{os.path.basename(paths[1])} ({len(rows)} frames)")
    if getattr(args, 'verbose', False):
        counts: Dict[str, int] = {}
        for r in rows:
            if r['verdict'] == 'rejected':
                key = f"{r['stage']}: {r['reason'].split(':')[0].split('(')[0].strip()}"
                counts[key] = counts.get(key, 0) + 1
        n_kept = sum(1 for r in rows if r['verdict'] == 'kept')
        n_flag = sum(1 for r in rows if r['verdict'] == 'kept' and r['notes'])
        safe_print(f"  Frame report: {n_kept} kept ({n_flag} with notes), "
                   f"{len(rows) - n_kept} rejected")
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            safe_print(f"    {v:4d}  {k}")
    return rows
