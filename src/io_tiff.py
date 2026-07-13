"""tifffile-based loader for 16/32-bit linear TIFF astro images.

Common intermediate/export format from acquisition software (N.I.N.A.,
SharpCap) and other stacking tools (PixInsight, DeepSkyStacker). Returns a
(float32 array, header dict) pair compatible with the FITS/RAW load path.

TIFF carries no standard tag for Bayer pattern, exposure time, or ISO the way
FITS headers or RAW EXIF do. A 3-channel TIFF is treated as already-debayered
RGB (passed through untouched, matching how a pre-debayered FITS/RAW frame is
handled). A 2-D (mono) TIFF is genuinely ambiguous -- it could be a real
monochrome capture or an undemosaiced Bayer export with no way to tell them
apart from the file alone -- so it is left as a bare 2-D array and falls
through to the pipeline's existing session-Bayer/RGGB default, exactly like a
headerless FITS Bayer frame. Supply a `.json` sidecar with a `bayerPattern`
key (the same mechanism already used for FITS/RAW) to override this.
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np

TIFF_EXTENSIONS: Tuple[str, ...] = ('.tif', '.tiff')

try:
    import tifffile as _tifffile
    HAS_TIFFFILE = True
except Exception:
    _tifffile = None  # type: ignore[assignment]
    HAS_TIFFFILE = False


def is_tiff_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in TIFF_EXTENSIONS


def read_tiff_header(path: str) -> dict:
    """Read shape/dtype from TIFF tags only, without decoding pixel data."""
    hdr: dict = {'_TIFF_FILE': True, 'IMAGETYP': 'Light Frame'}
    if not HAS_TIFFFILE:
        return hdr
    try:
        with _tifffile.TiffFile(path) as tf:
            shape = tf.pages[0].shape
        hdr['NAXIS'] = len(shape)
        hdr['NAXIS2'] = shape[0]
        hdr['NAXIS1'] = shape[1]
        if len(shape) == 3:
            hdr['NAXIS3'] = shape[2]
    except Exception:
        pass
    return hdr


def read_tiff(path: str) -> Tuple[np.ndarray, dict]:
    """Load a TIFF file and return ``(float32_array, header_dict)``.

    Returns a 2-D array for mono/Bayer-mosaic TIFFs (routes through the
    pipeline's normal debayer step) or ``(H, W, 3)`` for already-RGB TIFFs
    (passed through untouched, same as a pre-debayered FITS/RAW frame).
    """
    if not HAS_TIFFFILE:
        raise RuntimeError(
            f"tifffile is not installed — cannot load TIFF file: {path}\n"
            "Install with: pip install tifffile"
        )
    data = _tifffile.imread(path)
    if data.ndim == 3 and data.shape[2] > 3:
        # RGBA or similar -- drop extra channels, keep RGB
        data = data[:, :, :3]
    # No rescaling: integer sample values are cast to float32 as-is (same
    # convention as load_fits, which never normalises pixel values either) so
    # a TIFF light stays on the same ADU-count scale as sibling FITS/RAW
    # calibration frames -- rescaling here would silently break bias/dark/flat
    # subtraction against calibration masters built from a different format.
    data = data.astype(np.float32)

    hdr: dict = {
        '_TIFF_FILE': True,
        'IMAGETYP': 'Light Frame',
        'NAXIS': data.ndim,
        'NAXIS2': data.shape[0],
        'NAXIS1': data.shape[1],
    }
    if data.ndim == 3:
        hdr['NAXIS3'] = data.shape[2]
    return data, hdr
