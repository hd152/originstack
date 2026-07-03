"""Frame discovery and classification."""
from __future__ import annotations

import json
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
        _merge_json_sidecar(p, hdr)
        ftype = classify_frame(p, hdr)
        if ftype == 'skip':
            continue
        frames[ftype].append(FrameInfo(path=p, type=ftype, header=hdr))
    return frames


_JSON_FITS_MAP = (
    # (json_key, fits_key, transform_fn)
    ('iso',                'ISOSPEED', None),
    ('gain',               'GAIN',     None),
    ('captureTemperatureC','CCD-TEMP', None),
    ('exposureTimeMS',     'EXPTIME',  lambda v: v / 1000.0),
    ('bayerPattern',       'BAYERPAT', lambda v: str(v).upper()),
)


def _merge_json_sidecar(fits_path: str, hdr: dict) -> None:
    """Backfill FITS header from a co-located JSON sidecar (if present).

    Celestron Origin and similar apps write per-frame JSON sidecars with ISO,
    gain, temperature, etc. that may not be embedded in the FITS header.
    Only fills in keys that are absent from the header — never overwrites.
    """
    json_path = os.path.splitext(fits_path)[0] + '.json'
    if not os.path.isfile(json_path):
        return
    try:
        with open(json_path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
    except Exception:
        return
    for jkey, fkey, transform in _JSON_FITS_MAP:
        if fkey not in hdr and jkey in data and data[jkey] is not None:
            try:
                val = data[jkey]
                hdr[fkey] = transform(val) if transform else val
            except Exception:
                pass


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


def _frame_temp(f: FrameInfo) -> Optional[float]:
    """Extract CCD temperature (°C) from a frame's header."""
    for key in ('CCDTEMP', 'CCD-TEMP', 'TEMPERAT', 'CCD_TEMP', 'SET-TEMP'):
        val = f.header.get(key)
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
      2. CCD temperature (dark current doubles every ~6 °C)
      3. Exposure time (prefer matching, but accept with scaling)
      4. Dimensions (must match)

    If all darks already share the same properties, returns them unchanged.
    If no darks match, returns all darks with a warning (better than nothing).
    """
    if not lights or not darks:
        return darks

    # Determine majority light properties
    light_isos = [str(_frame_iso(f)) for f in lights if _frame_iso(f) is not None]
    light_ets = [round(_frame_exptime(f), 1) for f in lights if _frame_exptime(f) is not None]
    light_temps = [_frame_temp(f) for f in lights if _frame_temp(f) is not None]
    light_dims = [(f.header.get('NAXIS2'), f.header.get('NAXIS1')) for f in lights]

    majority_iso = Counter(light_isos).most_common(1)[0][0] if light_isos else None
    majority_et = Counter(light_ets).most_common(1)[0][0] if light_ets else None
    majority_temp = sum(light_temps) / len(light_temps) if light_temps else None
    majority_dims = Counter(light_dims).most_common(1)[0][0] if light_dims else None

    # Check if all darks are already homogeneous — no filtering needed
    dark_isos = set(str(_frame_iso(f)) for f in darks if _frame_iso(f) is not None)
    dark_ets = set(round(_frame_exptime(f), 1) for f in darks if _frame_exptime(f) is not None)
    dark_temps_raw = [_frame_temp(f) for f in darks if _frame_temp(f) is not None]
    dark_temp_spread = (max(dark_temps_raw) - min(dark_temps_raw)) if len(dark_temps_raw) > 1 else 0
    if len(dark_isos) <= 1 and len(dark_ets) <= 1 and dark_temp_spread <= 2.0:
        return darks

    scored = []
    for d in darks:
        score = 0
        d_iso = str(_frame_iso(d)) if _frame_iso(d) is not None else None
        d_et = round(_frame_exptime(d), 1) if _frame_exptime(d) is not None else None
        d_temp = _frame_temp(d)
        d_dims = (d.header.get('NAXIS2'), d.header.get('NAXIS1'))

        # Dimension mismatch is disqualifying
        if majority_dims and d_dims != (None, None) and d_dims != majority_dims:
            score -= 1000

        # ISO match (most important: wrong ISO = wrong gain = wrong noise profile)
        if majority_iso is not None and d_iso is not None:
            if d_iso == majority_iso:
                score += 100
            else:
                score -= 100

        # Temperature match (dark current is exponential in temperature)
        if majority_temp is not None and d_temp is not None:
            delta = abs(d_temp - majority_temp)
            if delta <= 1.0:
                score += 30
            elif delta <= 3.0:
                score += 15
            elif delta <= 7.0:
                score += 5
            else:
                score -= 30   # Large mismatch: dark current will differ significantly

        # Exposure time match (helpful, but darks can be exposure-scaled)
        if majority_et is not None and d_et is not None:
            if abs(d_et - majority_et) < 0.5:
                score += 10  # Exact match
            elif abs(d_et - majority_et) / max(majority_et, 0.1) < 0.5:
                score += 5   # Within 50% — scalable

        scored.append((score, d))

    if not scored:
        return darks

    best_score = max(s for s, _ in scored)
    selected = [d for s, d in scored if s == best_score]

    if len(selected) < len(darks):
        excluded = len(darks) - len(selected)
        sel_iso = str(_frame_iso(selected[0])) if _frame_iso(selected[0]) is not None else '?'
        sel_et = _frame_exptime(selected[0])
        sel_temp = _frame_temp(selected[0])
        et_str = f" exp={sel_et:.1f}s" if sel_et is not None else ''
        temp_str = f" temp={sel_temp:.1f}°C" if sel_temp is not None else ''
        safe_print(f"  ℹ Dark selection: kept {len(selected)}/{len(darks)} darks "
                   f"(ISO={sel_iso}{et_str}{temp_str}) matching lights, excluded {excluded}")

    return selected


def select_matching_flats(lights: List[FrameInfo], flats: List[FrameInfo]) -> List[FrameInfo]:
    """Select flat frames that best match the light frames.

    Matches on optical filter (FILTER header) then sensor dimensions.
    If all flats are already homogeneous, returns them unchanged.
    """
    if not lights or not flats:
        return flats

    flat_filters = set(f.header.get('FILTER') or '' for f in flats)
    flat_dims = set((f.header.get('NAXIS2'), f.header.get('NAXIS1')) for f in flats)
    if len(flat_filters) <= 1 and len(flat_dims) <= 1:
        return flats

    light_filters = [f.header.get('FILTER') or '' for f in lights]
    light_dims = [(f.header.get('NAXIS2'), f.header.get('NAXIS1')) for f in lights]
    majority_filter = Counter(light_filters).most_common(1)[0][0] if light_filters else None
    majority_dims = Counter(light_dims).most_common(1)[0][0] if light_dims else None

    scored = []
    for fl in flats:
        score = 0
        fl_filter = fl.header.get('FILTER') or ''
        fl_dims = (fl.header.get('NAXIS2'), fl.header.get('NAXIS1'))

        if majority_dims and fl_dims != (None, None) and fl_dims != majority_dims:
            score -= 1000

        if majority_filter and fl_filter:
            if fl_filter.lower() == majority_filter.lower():
                score += 50
            else:
                score -= 50

        scored.append((score, fl))

    if not scored:
        return flats

    best_score = max(s for s, _ in scored)
    selected = [f for s, f in scored if s == best_score]

    if len(selected) < len(flats):
        excluded = len(flats) - len(selected)
        sel_filter = selected[0].header.get('FILTER') or '(none)'
        safe_print(f"  ℹ Flat selection: kept {len(selected)}/{len(flats)} flats "
                   f"(filter={sel_filter}) matching lights, excluded {excluded}")

    return selected
