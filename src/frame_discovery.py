"""Frame discovery and classification."""
from __future__ import annotations

import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional

from src.models import FrameInfo
from src.io_fits import _read_fits_header
from src.utils import safe_print


def discover_frames(directory: str) -> Dict[str, List[FrameInfo]]:
    """Discover FITS and RAW files and classify them by heuristics and headers."""
    try:
        from src.io_raw import RAW_EXTENSIONS, read_raw_header, HAS_RAWPY
        _raw_exts: tuple = RAW_EXTENSIONS if HAS_RAWPY else ()
    except Exception:
        _raw_exts = ()

    _all_exts = ('.fit', '.fits') + _raw_exts

    files = sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(_all_exts)
    )
    frames = {'light': [], 'dark': [], 'flat': [], 'bias': []}

    def _read_header(path: str) -> dict:
        if _raw_exts and path.lower().endswith(_raw_exts):
            return read_raw_header(path)
        return _read_fits_header(path)

    with ThreadPoolExecutor() as ex:
        headers = list(ex.map(_read_header, files))
    for p, hdr in zip(files, headers):
        ftype = classify_frame(p, hdr)
        if ftype == 'skip':
            continue
        frames[ftype].append(FrameInfo(path=p, type=ftype, header=hdr))
    return frames


def classify_frame(path: str, header: dict) -> str:
    name = os.path.basename(path).lower()
    # Skip files produced by this pipeline (stacked outputs)
    if header.get('COMBINED') or header.get('CREATOR', '').startswith('astro_stack'):
        return 'skip'
    if 'dark' in name or header.get('IMAGETYP', '').lower() == 'dark':
        return 'dark'
    if 'flat' in name or header.get('IMAGETYP', '').lower() == 'flat':
        return 'flat'
    if 'bias' in name or header.get('IMAGETYP', '').lower() == 'bias' or header.get('EXPTIME', 1) == 0:
        return 'bias'
    return 'light'


def _frame_iso(f: FrameInfo):
    """Extract ISO/gain value from a frame's header."""
    return f.header.get('ISOSPEED') or f.header.get('ISO') or f.header.get('GAIN')


def _frame_exptime(f: FrameInfo) -> Optional[float]:
    """Extract exposure time from a frame's header."""
    val = f.header.get('EXPTIME')
    if val is not None:
        try:
            return float(val)
        except (ValueError, TypeError):
            pass
    return None


def select_matching_darks(lights: List[FrameInfo], darks: List[FrameInfo]) -> List[FrameInfo]:
    """Select dark frames that best match the light frames.

    Matching priority:
      1. ISO/gain (must match majority of lights)
      2. Exposure time (prefer matching, but accept with scaling)
      3. Dimensions (must match)

    If all darks already share the same properties, returns them unchanged.
    If no darks match, returns all darks with a warning (better than nothing).
    """
    if not lights or not darks:
        return darks

    # Determine majority light properties
    light_isos = [str(_frame_iso(f)) for f in lights if _frame_iso(f) is not None]
    light_ets = [round(_frame_exptime(f), 1) for f in lights if _frame_exptime(f) is not None]
    light_dims = [(f.header.get('NAXIS2'), f.header.get('NAXIS1')) for f in lights]

    majority_iso = Counter(light_isos).most_common(1)[0][0] if light_isos else None
    majority_et = Counter(light_ets).most_common(1)[0][0] if light_ets else None
    majority_dims = Counter(light_dims).most_common(1)[0][0] if light_dims else None

    # Check if all darks are identical — no filtering needed
    dark_isos = set(str(_frame_iso(f)) for f in darks if _frame_iso(f) is not None)
    dark_ets = set(round(_frame_exptime(f), 1) for f in darks if _frame_exptime(f) is not None)
    if len(dark_isos) <= 1 and len(dark_ets) <= 1:
        return darks

    # Score each dark frame: higher is better
    # ISO match is critical (wrong ISO = wrong gain = wrong noise profile)
    # Exposure match is helpful but darks can be exposure-scaled
    from typing import Tuple
    scored: List[Tuple[int, FrameInfo]] = []
    for d in darks:
        score = 0
        d_iso = str(_frame_iso(d)) if _frame_iso(d) is not None else None
        d_et = round(_frame_exptime(d), 1) if _frame_exptime(d) is not None else None
        d_dims = (d.header.get('NAXIS2'), d.header.get('NAXIS1'))

        # Dimension mismatch is disqualifying
        if majority_dims and d_dims != (None, None) and d_dims != majority_dims:
            score -= 1000

        # ISO match (most important)
        if majority_iso is not None and d_iso is not None:
            if d_iso == majority_iso:
                score += 100
            else:
                score -= 100  # Wrong ISO is harmful

        # Exposure time match (helpful)
        if majority_et is not None and d_et is not None:
            if abs(d_et - majority_et) < 0.5:
                score += 10  # Exact match
            elif abs(d_et - majority_et) / max(majority_et, 0.1) < 0.5:
                score += 5   # Within 50% — scalable
            # else: no bonus, but don't penalize (exposure scaling handles it)

        scored.append((score, d))

    if not scored:
        return darks

    best_score = max(s for s, _ in scored)
    selected = [d for s, d in scored if s == best_score]

    # Report selection if we actually filtered something out
    if len(selected) < len(darks):
        excluded = len(darks) - len(selected)
        sel_iso = str(_frame_iso(selected[0])) if _frame_iso(selected[0]) is not None else '?'
        sel_et = _frame_exptime(selected[0])
        et_str = f" exp={sel_et:.1f}s" if sel_et else ''
        safe_print(f"  ℹ Dark selection: kept {len(selected)}/{len(darks)} darks "
                   f"(ISO={sel_iso}{et_str}) matching lights, excluded {excluded}")

    return selected
