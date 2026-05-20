"""rawpy-based loader for DSLR/mirrorless RAW files.

Reads the raw Bayer mosaic directly (no in-camera demosaicing) and returns a
(float32 array, header dict) pair that is compatible with the FITS load path.
The existing debayer step (bilinear / Malvar / VNG) then handles demosaicing.

Supported extensions: CR2, CR3, NEF, ARW, DNG, ORF, RW2, RAF, PEF, 3FR, …
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np

# Canonical set of extensions handled by rawpy
RAW_EXTENSIONS: Tuple[str, ...] = (
    '.cr2', '.cr3',               # Canon
    '.nef', '.nrw',               # Nikon
    '.arw', '.srf', '.sr2',       # Sony
    '.dng',                       # Adobe Digital Negative (multi-vendor)
    '.orf',                       # Olympus / OM System
    '.rw2', '.rwl',               # Panasonic / Leica
    '.raf',                       # Fujifilm
    '.pef', '.ptx',               # Pentax
    '.3fr',                       # Hasselblad
    '.mrw',                       # Minolta
    '.x3f',                       # Sigma
    '.iiq',                       # Phase One
)

try:
    import rawpy as _rawpy
    HAS_RAWPY = True
except Exception:
    _rawpy = None  # type: ignore[assignment]
    HAS_RAWPY = False


def is_raw_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in RAW_EXTENSIONS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bayer_pattern_str(raw) -> str:
    """Convert rawpy's raw_pattern matrix + color_desc to 'RGGB' / 'BGGR' etc."""
    try:
        desc = raw.color_desc.decode('ascii')   # e.g. b'RGBG' → 'RGBG'
        pattern = raw.raw_pattern               # 2×2 int array
        return ''.join(desc[int(pattern[r, c])] for r in range(2) for c in range(2))
    except Exception:
        return 'RGGB'


def _exif_from_pillow(path: str) -> dict:
    """Best-effort EXIF extraction via Pillow (already a core dependency)."""
    try:
        from PIL import Image as _Img
        from PIL.ExifTags import TAGS
        img = _Img.open(path)
        raw_exif = img._getexif()  # type: ignore[attr-defined]
        if not raw_exif:
            return {}
        exif = {TAGS.get(k, k): v for k, v in raw_exif.items()}
        out: dict = {}
        if 'ExposureTime' in exif:
            et = exif['ExposureTime']
            try:
                out['EXPTIME'] = float(et)
            except Exception:
                try:
                    out['EXPTIME'] = float(et[0]) / float(et[1])
                except Exception:
                    pass
        if 'ISOSpeedRatings' in exif:
            try:
                out['ISO'] = int(exif['ISOSpeedRatings'])
                out['ISOSPEED'] = out['ISO']
            except Exception:
                pass
        if 'Model' in exif:
            out['INSTRUME'] = str(exif['Model']).strip()
        if 'Make' in exif:
            out['TELESCOP'] = str(exif['Make']).strip()
        return out
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_raw_header(path: str) -> dict:
    """Read metadata from a RAW file without decoding pixel data.

    Returns a dict shaped like the FITS header dicts used elsewhere.
    Used by frame_discovery for fast header-only scanning.
    """
    hdr: dict = {'_RAW_FILE': True, 'IMAGETYP': 'Light Frame'}
    if not HAS_RAWPY:
        return hdr
    try:
        with _rawpy.imread(path) as raw:
            sz = raw.sizes
            hdr.update({
                'NAXIS':   2,
                'NAXIS1':  sz.width,
                'NAXIS2':  sz.height,
                'BAYERPAT': _bayer_pattern_str(raw),
                'COLORTYP': _bayer_pattern_str(raw),
            })
        hdr.update(_exif_from_pillow(path))
    except Exception:
        pass
    return hdr


def read_raw(path: str) -> tuple:
    """Load a RAW file and return ``(float32_bayer_mosaic, header_dict)``.

    The Bayer mosaic is black-level corrected and normalised to [0, 1].
    The header carries BAYERPAT / COLORTYP so the debayer step knows how to
    demosaic the returned 2-D array.
    """
    if not HAS_RAWPY:
        raise RuntimeError(
            f"rawpy is not installed — cannot load RAW file: {path}\n"
            "Install with: pip install rawpy"
        )
    with _rawpy.imread(path) as raw:
        sz = raw.sizes
        bayer_str = _bayer_pattern_str(raw)
        hdr: dict = {
            'NAXIS':    2,
            'NAXIS1':   sz.width,
            'NAXIS2':   sz.height,
            'BAYERPAT': bayer_str,
            'COLORTYP': bayer_str,
            'IMAGETYP': 'Light Frame',
            '_RAW_FILE': True,
        }
        hdr.update(_exif_from_pillow(path))

        # Raw Bayer mosaic as float32
        bayer = raw.raw_image_visible.copy().astype(np.float32)

        # Per-channel black-level subtraction
        bl_per_ch = raw.black_level_per_channel  # list of 4 ints [R, G1, G2, B]
        pattern = raw.raw_pattern
        if bl_per_ch is not None:
            for row in range(2):
                for col in range(2):
                    ch = int(pattern[row, col])
                    bl = float(bl_per_ch[ch])
                    bayer[row::2, col::2] = np.maximum(
                        bayer[row::2, col::2] - bl, 0.0)

        # Normalise to [0, 1] using camera white level
        white = float(raw.white_level) if raw.white_level else 65535.0
        max_bl = (max(float(b) for b in bl_per_ch)
                  if bl_per_ch is not None else 0.0)
        scale = white - max_bl
        if scale > 0.0:
            bayer /= scale

    return bayer, hdr
