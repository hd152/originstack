"""ZOGY candidate real/bogus triage (``--transient-triage``).

``detect_transients`` (``src/difference_imaging.py``) returns every ``S_corr``
peak above threshold with no filtering: cosmic rays, sub-pixel
registration-slip dipoles and hot pixels all surface as candidates alongside
genuine transients. This scores each candidate with a small CNN -- the same
role ZTF's BTSbot / Rubin's DIA triage play downstream of classical image
differencing -- and attaches a ``real_probability`` to it. Advisory only: it
never drops a candidate, exactly like ``originvision.py`` never touches
``FrameInfo.accepted``/``metrics['score']``.

Native-only for now, deliberately -- unlike every other native kernel in this
project, there is no numpy/onnxruntime fallback yet (see the module docstring
in ``ext/astro_native/src/lib.rs``'s ``mod transient_triage``). Without
``astro_native`` built, ``--transient-triage`` self-disables with a warning.

The bundled model (``src/data/transient_triage.onnx``) is trained entirely on
synthetic data (``tools/gen_transient_triage_data.py`` +
``tools/train_transient_triage.py``) -- no labelled real transients exist yet
-- so it is a first cut, not a production classifier.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger('originstack')

try:
    import astro_native as _native
    _HAS_NATIVE_TRIAGE = hasattr(_native, 'transient_triage_score')
except Exception:  # pragma: no cover
    _native = None
    _HAS_NATIVE_TRIAGE = False

DEFAULT_STAMP_SIZE = 31


def scoring_backend_available() -> bool:
    """True when the native kernel is built. No fallback exists yet."""
    return _HAS_NATIVE_TRIAGE


def bundled_model_path() -> str:
    """Path to the model shipped inside the package."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                        'transient_triage.onnx')


def resolve_model_path(explicit: Optional[str] = None) -> Optional[str]:
    """Return the first usable model path: an explicit override, else the
    bundled copy. ``None`` if neither exists on disk."""
    for cand in (explicit, bundled_model_path()):
        if cand and os.path.isfile(cand):
            return cand
    return None


def _extract_stamp(arr: np.ndarray, y: float, x: float, size: int) -> np.ndarray:
    """Fixed-size square cutout centred on ``(y, x)``, reflect-padded at the
    frame border -- same boundary convention as this codebase's other
    fixed-window extractions (``_resize_center_crop``, the native
    gaussian/median kernels' mirror boundary)."""
    h, w = arr.shape
    half = size // 2
    cy, cx = int(round(y)), int(round(x))
    top, left = cy - half, cx - half
    pad_top = max(0, -top)
    pad_left = max(0, -left)
    pad_bottom = max(0, (top + size) - h)
    pad_right = max(0, (left + size) - w)
    if pad_top or pad_left or pad_bottom or pad_right:
        padded = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)),
                        mode='reflect')
        top += pad_top
        left += pad_left
        return padded[top:top + size, left:left + size]
    return arr[top:top + size, left:left + size]


def _normalize(stamp: np.ndarray, sigma: float) -> np.ndarray:
    s = sigma if (sigma is not None and np.isfinite(sigma) and sigma > 0) else 1.0
    # A candidate near the footprint edge can pull in NaN fill from outside
    # the warped reference's coverage (difference_imaging.py's `valid` mask);
    # zero is the right fill -- both epochs arrive background-subtracted, so
    # zero already means "sky" everywhere else in this pipeline's ZOGY code.
    clean = np.nan_to_num(stamp.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return clean / np.float32(s)


def build_stamps(new_lum: np.ndarray, ref_lum: np.ndarray,
                 difference: np.ndarray, positions: Sequence,
                 sigma_new: float, sigma_ref: float,
                 sigma_diff: float, size: int = DEFAULT_STAMP_SIZE) -> np.ndarray:
    """Build the ``(N, 3, size, size)`` NCHW input for ``transient_triage_score``.

    Channels are ``new``, ``ref``, ``difference`` (the standard real/bogus
    "triplet"), each normalized by its own frame-level robust sigma so a
    candidate's stamp is architecture-agnostic across sessions of different
    noise level -- not the ``originvision`` percentile stretch, which is for
    photographic display and would destroy the physical sigma units ZOGY's
    own significance already relies on.
    """
    n = len(positions)
    out = np.zeros((n, 3, size, size), dtype=np.float32)
    for i, (y, x) in enumerate(positions):
        out[i, 0] = _normalize(_extract_stamp(new_lum, y, x, size), sigma_new)
        out[i, 1] = _normalize(_extract_stamp(ref_lum, y, x, size), sigma_ref)
        out[i, 2] = _normalize(_extract_stamp(difference, y, x, size), sigma_diff)
    return out


def score_candidates(new_lum: np.ndarray, ref_lum: np.ndarray,
                     difference: np.ndarray, transients: Sequence,
                     sigma_new: float, sigma_ref: float, sigma_diff: float,
                     *, model_path: Optional[str] = None,
                     size: int = DEFAULT_STAMP_SIZE) -> List[Optional[float]]:
    """Return one ``real_probability`` (or ``None``) per entry in
    ``transients``, in the same order. ``None`` for every candidate -- logged
    once, not per candidate -- when the native backend or the model file
    isn't available."""
    if not transients:
        return []

    if not _HAS_NATIVE_TRIAGE:
        logger.warning("transient triage requested but astro_native's "
                       "transient_triage_score is unavailable -- skipping "
                       "(no candidates will be scored)")
        return [None] * len(transients)

    mp = resolve_model_path(model_path)
    if mp is None:
        logger.warning("transient triage requested but no model found "
                       "(bundled src/data/transient_triage.onnx missing) "
                       "-- skipping")
        return [None] * len(transients)

    positions = [(t.y, t.x) for t in transients]
    stamps = build_stamps(np.asarray(new_lum, dtype=np.float32),
                          np.asarray(ref_lum, dtype=np.float32),
                          np.asarray(difference, dtype=np.float32),
                          positions, sigma_new, sigma_ref, sigma_diff, size=size)
    try:
        probs = _native.transient_triage_score(stamps, mp, size)
    except Exception as exc:
        logger.warning(f"transient triage: native inference failed ({exc}) "
                       f"-- skipping")
        return [None] * len(transients)
    return [float(p) for p in probs]
