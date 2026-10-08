"""ZOGY candidate real/bogus triage (``--transient-triage``).

``detect_transients`` (``src/difference_imaging.py``) returns every ``S_corr``
peak above threshold with no filtering: cosmic rays, sub-pixel
registration-slip dipoles and hot pixels all surface as candidates alongside
genuine transients. This scores each candidate with a small CNN -- the same
role ZTF's BTSbot / Rubin's DIA triage play downstream of classical image
differencing -- and attaches a ``real_probability`` to it. Advisory only: it
never drops a candidate.

Two backends: the native ``astro_native.transient_triage_score`` (tract ONNX
runtime, only in builds with the crate's ``triage`` feature -- off by default
and in release builds) and a numpy forward pass (``_score_numpy``) that reads
the weights from the same ``.onnx`` file with a minimal protobuf reader
(``load_onnx_weights``), so the feature works everywhere with no ONNX
dependency. The numpy path supports exactly the ``TriageNet`` architecture
``tools/train_transient_triage.py`` exports (three 3x3 conv + ReLU, two 2x2
max-pools, global average pool, one linear unit) and refuses any other graph
rather than guessing; a ``--transient-triage-model`` with a different
architecture needs the native build.

The bundled model (``src/data/transient_triage.onnx``) is trained on synthetic
pairs (``tools/gen_transient_triage_data.py``) plus real two-night pairs of
three targets (``tools/gen_transient_triage_real.py``: point sources injected
with each night's own PSF as positives, every other real candidate as
negatives), via ``tools/train_transient_triage.py``. The synthetic-only model
it replaced scored ROC AUC 0.59-0.77 on real pairs; held out one target at a
time, the retrained one scores 0.95-0.999. Its positives are injected, not
observed transients: no labelled real transient exists yet.
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
    """Always True: the numpy forward pass needs nothing beyond numpy. The
    native kernel (``triage`` Cargo feature) is used when present."""
    return True


# --- numpy backend -----------------------------------------------------------
#
# The ONNX file is a protobuf. Only a few fields matter (numbers from
# onnx.proto): ModelProto.graph = 7; GraphProto.node = 1, .initializer = 5;
# NodeProto.op_type = 4; TensorProto.dims = 1, .data_type = 2,
# .float_data = 4, .name = 8, .raw_data = 9.

_TRIAGE_OPS = ('Conv', 'Relu', 'MaxPool', 'Conv', 'Relu', 'MaxPool', 'Conv', 'Relu',
               'GlobalAveragePool', 'Flatten', 'Gemm', 'Constant', 'Squeeze')
_TRIAGE_WEIGHTS = ('features.0.weight', 'features.0.bias', 'features.3.weight',
                   'features.3.bias', 'features.6.weight', 'features.6.bias',
                   'fc.weight', 'fc.bias')
_ONNX_FLOAT = 1
_numpy_models: dict = {}


def _pb_varint(buf: bytes, i: int):
    shift = result = 0
    while True:
        if i >= len(buf):
            raise ValueError('truncated protobuf varint')
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def _pb_fields(buf: bytes):
    """Yield ``(field_number, wire_type, value)`` for one protobuf message;
    ``value`` is an int for varint/fixed fields, bytes for length-delimited."""
    i, n = 0, len(buf)
    while i < n:
        key, i = _pb_varint(buf, i)
        field, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _pb_varint(buf, i)
        elif wt == 1:
            v, i = int.from_bytes(buf[i:i + 8], 'little'), i + 8
        elif wt == 2:
            ln, i = _pb_varint(buf, i)
            if i + ln > n:
                raise ValueError('truncated protobuf field')
            v, i = buf[i:i + ln], i + ln
        elif wt == 5:
            v, i = int.from_bytes(buf[i:i + 4], 'little'), i + 4
        else:
            raise ValueError(f'unsupported protobuf wire type {wt}')
        yield field, wt, v


def _pb_tensor(buf: bytes):
    name, dims, dtype, raw, floats = '', [], None, None, []
    for f, wt, v in _pb_fields(buf):
        if f == 1:
            if wt == 2:  # packed repeated int64
                j = 0
                while j < len(v):
                    d, j = _pb_varint(v, j)
                    dims.append(d)
            else:
                dims.append(v)
        elif f == 2:
            dtype = v
        elif f == 4:
            floats.append(np.frombuffer(v, dtype='<f4') if wt == 2
                          else np.array([v], dtype='<u4').view('<f4'))
        elif f == 8:
            name = bytes(v).decode('utf-8')
        elif f == 9:
            raw = v
    if dtype != _ONNX_FLOAT:
        raise ValueError(f'initializer {name!r}: data type {dtype}, expected float32')
    if raw is not None:
        arr = np.frombuffer(raw, dtype='<f4')
    else:
        arr = np.concatenate(floats) if floats else np.zeros(0, np.float32)
    count = int(np.prod(dims)) if dims else 1
    if arr.size != count:
        raise ValueError(f'initializer {name!r}: {arr.size} values for dims {dims}')
    return name, arr.astype(np.float32).reshape(dims)


def load_onnx_weights(path: str) -> dict:
    """Read the ``TriageNet`` weights (as float64) from an ONNX file.

    Raises ``ValueError`` unless the graph's op sequence and weight shapes are
    exactly the architecture ``_score_numpy`` implements."""
    with open(path, 'rb') as fh:
        model = fh.read()
    graph = next((v for f, wt, v in _pb_fields(model) if f == 7 and wt == 2), None)
    if graph is None:
        raise ValueError('no graph in ONNX file')
    ops, weights = [], {}
    for f, wt, v in _pb_fields(graph):
        if f == 1 and wt == 2:
            ops.append(next((bytes(x).decode('utf-8') for g, w, x in _pb_fields(v)
                             if g == 4 and w == 2), ''))
        elif f == 5 and wt == 2:
            name, arr = _pb_tensor(v)
            weights[name] = arr
    if tuple(ops) != _TRIAGE_OPS:
        raise ValueError(f'unsupported triage graph {ops} (the numpy backend implements '
                         f'only the TriageNet of tools/train_transient_triage.py)')
    missing = [k for k in _TRIAGE_WEIGHTS if k not in weights]
    if missing:
        raise ValueError(f'triage model is missing weights {missing}')
    w = {k: weights[k].astype(np.float64) for k in _TRIAGE_WEIGHTS}
    c_in = 3
    for k in ('features.0', 'features.3', 'features.6'):
        kw, kb = w[k + '.weight'], w[k + '.bias']
        if kw.ndim != 4 or kw.shape[1:] != (c_in, 3, 3) or kb.shape != (kw.shape[0],):
            raise ValueError(f'unexpected {k} shapes {kw.shape}/{kb.shape}')
        c_in = kw.shape[0]
    if w['fc.weight'].shape != (1, c_in) or w['fc.bias'].shape != (1,):
        raise ValueError(f"unexpected fc shapes {w['fc.weight'].shape}/{w['fc.bias'].shape}")
    return w


def _conv3x3_relu(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """3x3 convolution, zero padding 1, stride 1 (ONNX ``Conv`` is a
    cross-correlation, like torch's), then ReLU."""
    n, _, h, w = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    out = np.empty((n, weight.shape[0], h, w))
    out[:] = bias[None, :, None, None]
    for dy in range(3):
        for dx in range(3):
            out += np.einsum('nchw,oc->nohw', xp[:, :, dy:dy + h, dx:dx + w],
                             weight[:, :, dy, dx], optimize=True)
    return np.maximum(out, 0.0, out=out)


def _maxpool2(x: np.ndarray) -> np.ndarray:
    """2x2 max-pool, stride 2, floor (ONNX ``ceil_mode=0``)."""
    n, c, h, w = x.shape
    h2, w2 = h // 2, w // 2
    return x[:, :, :2 * h2, :2 * w2].reshape(n, c, h2, 2, w2, 2).max(axis=(3, 5))


def _score_numpy(stamps: np.ndarray, model_path: str) -> np.ndarray:
    """Forward pass of ``TriageNet`` in float64; returns sigmoid
    probabilities, as the native kernel does."""
    w = _numpy_models.get(model_path)
    if w is None:
        w = load_onnx_weights(model_path)
        _numpy_models[model_path] = w
    out = np.empty(len(stamps))
    for s in range(0, len(stamps), 64):
        x = stamps[s:s + 64].astype(np.float64)
        x = _maxpool2(_conv3x3_relu(x, w['features.0.weight'], w['features.0.bias']))
        x = _maxpool2(_conv3x3_relu(x, w['features.3.weight'], w['features.3.bias']))
        x = _conv3x3_relu(x, w['features.6.weight'], w['features.6.bias'])
        logit = x.mean(axis=(2, 3)) @ w['fc.weight'][0] + w['fc.bias'][0]
        out[s:s + 64] = 1.0 / (1.0 + np.exp(-logit))
    return out


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
    noise level -- not a percentile display stretch, which would destroy the
    physical sigma units ZOGY's own significance already relies on.
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
    once, not per candidate -- when the model file is missing or neither
    backend can run it."""
    if not transients:
        return []

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
    if _HAS_NATIVE_TRIAGE:
        try:
            return [float(p) for p in _native.transient_triage_score(stamps, mp, size)]
        except Exception as exc:
            logger.warning(f"transient triage: native inference failed ({exc}) "
                           f"-- trying the numpy backend")
    try:
        probs = _score_numpy(stamps, mp)
    except Exception as exc:
        logger.warning(f"transient triage: cannot score with {mp} ({exc}) -- skipping")
        return [None] * len(transients)
    return [float(p) for p in probs]
