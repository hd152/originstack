"""Per-instrument vignetting/background calibration map: load + apply side.

The map itself is built offline from many past sessions' individual
calibrated (dark/flat/debayer-corrected) light frames -- see
``tools/build_vignette_map.py``. It captures whatever smooth, spatially
fixed pattern survives robust-averaging across sessions pointed in
different sky directions: real optical vignetting, a repeatable
flat-calibration residual, and/or fixed local light-pollution direction all
show up here and all genuinely benefit from being subtracted before DBE/the
wavelet background extractor have to guess at them from one session's sparse
sky patches alone.

Critical constraint: the map lives in *native sensor pixel space* (built
from individual frames before any registration warp rotates/crops them). It
must be applied per-frame, right after debayer and before registration --
never to a final stack, which has already been warped to some session's
arbitrary reference-frame orientation and would no longer line up with it.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

try:
    from src.utils import safe_print
except ImportError:
    def safe_print(*args, **kwargs):
        print(*args, **kwargs)


def load_vignette_map(path: str) -> Optional[np.ndarray]:
    """Load a vignetting calibration map saved by ``build_vignette_map.py``.

    Returns an (H, W, 3) float32 array, or None if the file is missing/unreadable.
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        from astropy.io import fits
        with fits.open(path) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float32)
        if data.ndim == 3 and data.shape[0] in (3, 4):
            data = np.moveaxis(data, 0, -1)[..., :3]
        if data.ndim != 3 or data.shape[2] != 3:
            safe_print(f"  WARNING: vignette map {path} has unexpected shape "
                       f"{data.shape} -- ignoring")
            return None
        return data
    except Exception as e:
        safe_print(f"  WARNING: could not load vignette map {path}: {e}")
        return None


def apply_vignette_correction(rgb: np.ndarray, vignette_map: np.ndarray) -> np.ndarray:
    """Subtract the calibration map from a calibrated+debayered RGB frame.

    Resizes the map to match ``rgb`` if it was built at a lower resolution
    (the map is smooth by construction, so a cubic resize introduces no
    meaningful error). Result is NOT clipped to non-negative here -- the
    caller's existing post-calibration clip handles that, same as flat
    division.
    """
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return rgb
    vmap = vignette_map
    if vmap.shape[:2] != rgb.shape[:2]:
        from scipy.ndimage import zoom
        zy = rgb.shape[0] / vmap.shape[0]
        zx = rgb.shape[1] / vmap.shape[1]
        vmap = zoom(vmap, (zy, zx, 1.0), order=3)
        # zoom can overshoot the exact target shape by a pixel from rounding
        vmap = vmap[:rgb.shape[0], :rgb.shape[1], :]
    out = rgb.astype(np.float32, copy=True)
    out[..., :3] -= vmap[..., :3]
    return out
