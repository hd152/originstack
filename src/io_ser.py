"""Reader for SER (planetary/lucky-imaging video) files.

Used by FireCapture, SharpCap, Genika, and similar capture tools for
planetary/lunar/solar lucky imaging. A single .ser file holds many
sequential frames -- unlike every other format this pipeline reads, so a
.ser file is never a valid frame path on its own. ``expand_ser_files``
turns each .ser file into N virtual paths of the form
``"<real_path>::<frame_index>"`` (one per frame), which flow through the
rest of the pipeline as ordinary ``FrameInfo.path`` strings -- ``::`` is
not a filesystem path separator on Windows or POSIX, so filename-substring
classification, dict/set keys, logging, and checkpoint JSON serialization
all keep working unmodified.

No third-party dependency: SER is a simple, fully-specified binary format
(178-byte fixed header + sequential raw frames), parsed with the stdlib
``struct`` module and ``numpy.memmap`` for O(1) random-access frame reads.

SER 178-byte header layout (all integers little-endian per the format spec):
  FileID              14s   "LUCAM-RECORDER"
  LuID                i     (legacy, unused)
  ColorID             i     see _COLOR_ID_MAP below
  LittleEndian        i     nonzero -> samples stored little-endian
  ImageWidth          i
  ImageHeight         i
  PixelDepthPerPlane  i     8 or 16
  FrameCount          i
  Observer            40s
  Instrume            40s
  Telescope           40s
  DateTime            q     .NET ticks, local time
  DateTime_UTC        q     .NET ticks, UTC

KNOWN GOTCHA: several widely-used SER-writing tools have historically
written the ``LittleEndian`` flag inconsistently with how they actually
byte-order 16-bit samples (a long-standing, widely-documented ambiguity in
the SER ecosystem, not unique to this reader). This implementation follows
the field's literal documented meaning. If pixel values from a real capture
look implausible (byte-swapped/noisy), check this first against a sample
from the capturing tool -- this repo has not been validated against a real
FireCapture/SharpCap file.
"""
from __future__ import annotations

import os
import re
import struct
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np

SER_EXTENSIONS: Tuple[str, ...] = ('.ser',)
SER_HEADER_SIZE = 178

_VPATH_RE = re.compile(r'^(.*\.ser)::(\d+)$', re.IGNORECASE)

# SER ColorID -> BAYERPAT string. None = mono (no Bayer pattern). RGB/BGR are
# already 3-channel per frame, handled separately (not a Bayer pattern).
_COLOR_ID_BAYER = {
    0: None,       # MONO
    8: 'RGGB',
    9: 'GRBG',
    10: 'GBRG',
    11: 'BGGR',
    100: 'RGB',
    101: 'BGR',
}
# CMY-filter Bayer variants: rare, not supported by src/debayer.py's 4
# standard patterns. Explicitly rejected rather than silently mismapped.
_COLOR_ID_UNSUPPORTED = {16: 'CYYM', 17: 'YCMY', 18: 'MYCY', 19: 'YMCY'}


def is_ser_virtual_path(path: str) -> bool:
    return bool(_VPATH_RE.match(path))


def _split_vpath(virtual_path: str) -> Tuple[str, int]:
    m = _VPATH_RE.match(virtual_path)
    if not m:
        raise ValueError(f'Not a SER virtual path: {virtual_path!r}')
    return m.group(1), int(m.group(2))


@lru_cache(maxsize=64)
def _parse_ser_header(real_path: str) -> dict:
    """Parse the 178-byte SER header, memoized per real file path so every
    virtual frame of one file shares a single header parse."""
    with open(real_path, 'rb') as fh:
        raw = fh.read(SER_HEADER_SIZE)
    if len(raw) < SER_HEADER_SIZE:
        raise ValueError(f'Not a valid SER file (truncated header): {real_path}')
    (file_id, _lu_id, color_id, little_endian, width, height, depth,
     frame_count, _observer, _instrume, _telescope, _dt, _dt_utc) = struct.unpack(
        '<14s7i40s40s40sqq', raw)
    if not file_id.startswith(b'LUCAM-RECORDER'):
        raise ValueError(f'Not a valid SER file (bad FileID): {real_path}')

    bytes_per_sample = 1 if depth <= 8 else 2
    num_planes = 3 if color_id in (100, 101) else 1
    return {
        'ColorID': color_id,
        'Width': width,
        'Height': height,
        'PixelDepth': depth,
        'FrameCount': frame_count,
        'LittleEndian': bool(little_endian),
        'BytesPerSample': bytes_per_sample,
        'NumPlanes': num_planes,
        'FrameSize': width * height * bytes_per_sample * num_planes,
    }


def expand_ser_files(directory: str) -> List[str]:
    """Pre-pass for discover_frames: parse each .ser file's header once (O(1)
    I/O per file), synthesize one virtual path per frame it contains."""
    out: List[str] = []
    try:
        entries = sorted(os.listdir(directory))
    except OSError:
        return out
    for fn in entries:
        if not fn.lower().endswith('.ser'):
            continue
        real = os.path.join(directory, fn)
        try:
            info = _parse_ser_header(real)
        except Exception:
            continue
        out.extend(f'{real}::{i}' for i in range(info['FrameCount']))
    return out


def read_ser_frame_header(virtual_path: str) -> dict:
    real_path, _idx = _split_vpath(virtual_path)
    info = _parse_ser_header(real_path)
    color_id = info['ColorID']
    hdr: dict = {
        '_SER_FILE': True,
        'IMAGETYP': 'Light Frame',
        'NAXIS': 2,
        'NAXIS1': info['Width'],
        'NAXIS2': info['Height'],
    }
    bayer = _COLOR_ID_BAYER.get(color_id)
    if bayer and bayer not in ('RGB', 'BGR'):
        hdr['BAYERPAT'] = bayer
        hdr['COLORTYP'] = bayer
    return hdr


def read_ser_frame(virtual_path: str) -> Tuple[np.ndarray, dict]:
    """Load a single frame from a SER file by virtual path index.

    Mono (ColorID==0) frames are replicated to (H,W,3) so they route through
    the pipeline's existing "already 3-channel, skip debayer" pass-through
    instead of being (wrongly) demosaiced as an unknown Bayer pattern.
    """
    real_path, idx = _split_vpath(virtual_path)
    info = _parse_ser_header(real_path)
    color_id = info['ColorID']
    if color_id in _COLOR_ID_UNSUPPORTED:
        raise NotImplementedError(
            f'SER ColorID {color_id} ({_COLOR_ID_UNSUPPORTED[color_id]}, a CMY-filter '
            f'Bayer variant) is not supported by this pipeline\'s debayer step: {real_path}'
        )
    if idx < 0 or idx >= info['FrameCount']:
        raise IndexError(f'Frame index {idx} out of range for {real_path} '
                         f'(FrameCount={info["FrameCount"]})')

    if info['BytesPerSample'] == 1:
        dtype = np.dtype(np.uint8)
    else:
        dtype = np.dtype('<u2') if info['LittleEndian'] else np.dtype('>u2')

    shape = ((info['Height'], info['Width'], info['NumPlanes'])
             if info['NumPlanes'] == 3 else (info['Height'], info['Width']))
    offset = SER_HEADER_SIZE + idx * info['FrameSize']

    mm = np.memmap(real_path, dtype=dtype, mode='r', offset=offset, shape=shape)
    # Copy out of the memmap before returning -- downstream calibration code
    # mutates frame arrays in place, and a bare memmap view would corrupt the
    # source file on write-through.
    data = np.array(mm, dtype=np.float32)
    del mm

    # No rescaling: unlike RAW (which corrects a real per-camera black-level
    # offset via EXIF/rawpy metadata), SER carries no calibration metadata at
    # all -- sample values are kept as-is (same ADU-count convention as
    # load_fits/read_tiff) so a SER light stays on the same scale as sibling
    # FITS/TIFF calibration frames. Rescaling to [0,1] here previously made
    # dark/flat subtraction against ADU-scale masters clip everything to
    # ~zero ("flat image" rejection) -- caught by an end-to-end mixed-format
    # smoke test, not a unit test in isolation.

    if color_id == 0:
        data = np.repeat(data[:, :, None], 3, axis=2)
    elif color_id == 101:  # BGR -> RGB channel order
        data = data[:, :, ::-1].copy()

    hdr = read_ser_frame_header(virtual_path)
    return data, hdr
