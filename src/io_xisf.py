"""Hand-rolled reader for PixInsight XISF 1.0 files.

Counterpart to ``src/xisf_writer.py``. Covers the common/simple case this
codebase's own writer produces, and most straightforward external exports:
uncompressed, attached (not embedded/inline) pixel data, Float32/UInt16/
UInt32/UInt8 sample format, mono or RGB, planar or normal pixel storage.
Compressed blocks, embedded/inline data, and other sample formats or colour
spaces raise a clear error naming what is unsupported rather than silently
misreading.

XISF layout (see xisf_writer.py's module docstring for the exact byte
layout this writer produces; this reader follows the general XISF 1.0
spec so it can also read files from other tools):
  Bytes 0-7:    signature "XISF0100"
  Bytes 8-11:   uint32 LE  length of the XML header block (NOT including
                            this 16-byte preamble)
  Bytes 12-15:  uint32 LE  reserved
  Bytes 16-N:   XML header (UTF-8)
  Bytes N-pad:  zero padding
  attachment:   raw pixel data at the byte offset named in the Image
                element's ``location`` attribute
"""
from __future__ import annotations

import os
import struct
import xml.etree.ElementTree as ET
from typing import Optional, Tuple

import numpy as np

XISF_EXTENSIONS: Tuple[str, ...] = ('.xisf',)

_XISF_NS = '{http://www.pixinsight.com/xisf}'
_SIGNATURE = b'XISF0100'

_SAMPLE_DTYPES = {
    'UInt8': np.uint8,
    'UInt16': np.uint16,
    'UInt32': np.uint32,
    'Float32': np.float32,
    'Float64': np.float64,
}


def is_xisf_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in XISF_EXTENSIONS


def _find_image_element(root: ET.Element) -> ET.Element:
    img = root.find(f'{_XISF_NS}Image')
    if img is None:
        img = root.find('Image')  # tolerate files without the namespace declared
    if img is None:
        raise ValueError('XISF file has no <Image> element')
    return img


def _parse_xisf_preamble(fh) -> Tuple[ET.Element, int]:
    """Read and validate the 16-byte preamble + XML header block.

    Returns (image_element, xml_header_end_offset).
    """
    preamble = fh.read(16)
    if len(preamble) < 16 or preamble[:8] != _SIGNATURE:
        raise ValueError(
            f'Not a supported XISF file (expected signature {_SIGNATURE!r}, '
            f'got {preamble[:8]!r})'
        )
    header_length, _reserved = struct.unpack('<II', preamble[8:16])
    xml_bytes = fh.read(header_length)
    root = ET.fromstring(xml_bytes)
    image_el = _find_image_element(root)
    return image_el, 16 + header_length


def _parse_geometry(geometry: str) -> Tuple[int, int, int]:
    parts = [int(p) for p in geometry.split(':')]
    if len(parts) == 2:
        w, h = parts
        c = 1
    elif len(parts) == 3:
        w, h, c = parts
    else:
        raise ValueError(f'Unsupported XISF geometry: {geometry!r}')
    return w, h, c


def _image_element_to_header(image_el: ET.Element) -> dict:
    hdr: dict = {'_XISF_FILE': True, 'IMAGETYP': 'Light Frame'}
    w, h, c = _parse_geometry(image_el.get('geometry', ''))
    hdr['NAXIS'] = 3 if c > 1 else 2
    hdr['NAXIS1'] = w
    hdr['NAXIS2'] = h
    if c > 1:
        hdr['NAXIS3'] = c
    kw_el = image_el.find(f'{_XISF_NS}FITSKeywords')
    if kw_el is None:
        kw_el = image_el.find('FITSKeywords')
    if kw_el is not None:
        for kw in kw_el:
            name = kw.get('name')
            value = kw.get('value')
            if not name or value is None:
                continue
            try:
                hdr[name] = float(value)
            except (TypeError, ValueError):
                hdr[name] = value
    return hdr


def read_xisf_header(path: str) -> dict:
    """Read shape/metadata from an XISF file without decoding pixel data."""
    try:
        with open(path, 'rb') as fh:
            image_el, _ = _parse_xisf_preamble(fh)
        return _image_element_to_header(image_el)
    except Exception:
        return {'_XISF_FILE': True, 'IMAGETYP': 'Light Frame'}


def read_xisf(path: str) -> Tuple[np.ndarray, dict]:
    """Load an XISF file and return ``(float32_array, header_dict)``.

    Returns ``(H, W, 3)`` for RGB, or ``(H, W, 3)`` for mono too (the single
    channel replicated across R/G/B) -- XISF is a processed/calibrated-image
    format, never a raw Bayer mosaic, so a mono XISF must never be routed
    through the pipeline's debayer step the way a mono FITS/RAW frame is.
    """
    with open(path, 'rb') as fh:
        image_el, _ = _parse_xisf_preamble(fh)

        compression = image_el.get('compression')
        if compression:
            raise NotImplementedError(
                f'XISF compression ({compression!r}) is not supported by this '
                f'reader: {path}'
            )

        sample_format = image_el.get('sampleFormat', 'Float32')
        dtype = _SAMPLE_DTYPES.get(sample_format)
        if dtype is None:
            raise ValueError(
                f'Unsupported XISF sampleFormat {sample_format!r} in {path} '
                f'(supported: {sorted(_SAMPLE_DTYPES)})'
            )

        color_space = image_el.get('colorSpace', 'Gray')
        if color_space not in ('Gray', 'RGB'):
            raise ValueError(
                f'Unsupported XISF colorSpace {color_space!r} in {path} '
                f'(supported: Gray, RGB)'
            )

        location = image_el.get('location', '')
        if not location.startswith('attachment:'):
            kind = location.split(':', 1)[0] if location else '(missing)'
            raise NotImplementedError(
                f'XISF location type {kind!r} is not supported by this reader '
                f'(only "attachment" blocks are); install the xisf package for '
                f'full XISF compatibility: {path}'
            )
        _, offset_str, size_str = location.split(':')
        offset, size = int(offset_str), int(size_str)

        w, h, c = _parse_geometry(image_el.get('geometry', ''))
        storage = image_el.get('pixelStorage', 'Normal')

        fh.seek(offset)
        raw = fh.read(size)

    flat = np.frombuffer(raw, dtype=dtype)
    if storage == 'Planar':
        data = flat.reshape(c, h, w).transpose(1, 2, 0) if c > 1 else flat.reshape(h, w)
    else:
        data = flat.reshape(h, w, c) if c > 1 else flat.reshape(h, w)

    data = data.astype(np.float32)
    if color_space == 'Gray' or c == 1:
        if data.ndim == 3:
            data = data[:, :, 0]
        data = np.repeat(data[:, :, None], 3, axis=2)

    hdr = _image_element_to_header(image_el)
    return data, hdr
