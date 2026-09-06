"""In-process astrollm ONNX inference -- the wired-in replacement for the
old ``vendor/astrollm/infer_onnx.py`` subprocess.

This is a numpy/scipy port of that entry point plus the two ``data/``
helpers it imported (``imageops.py``, ``shape_features.py``); it drops the
OpenCV dependency (this codebase has none) and runs the exported model
directly through ``onnxruntime`` in the calling process. ``onnxruntime`` is
an *optional* dependency, guarded here the same way ``rawpy``/``cupy`` are
elsewhere -- ``--astrollm`` self-disables with a warning when it or the
bundled model file is absent.

The bundled model lives at ``src/data/astrollm.onnx`` (see
``vendor/astrollm/VENDORED_FROM.txt`` for provenance / re-sync). An explicit
model path (``--astrollm-model`` / ``--astrollm-dir``) still overrides it.

Advisory only -- see ``src/astrollm.py`` for how the result dict is used
(never sets ``FrameInfo.accepted`` or ``metrics['score']``).
"""
from __future__ import annotations

import logging
import math
import os
import threading
from typing import Optional

import numpy as np
from scipy import ndimage

logger = logging.getLogger('originstack')

try:  # optional dependency -- guarded (lint OS002)
    import onnxruntime as _ort
except Exception:  # pragma: no cover - onnxruntime absent
    _ort = None

_FITS_EXTS = ('.fits', '.fit', '.fts')

# the comet shape-gate's thresholds were tuned on features computed at 512px
# (upstream data/shape_features.py), independent of the model's own input
# size -- keep that.
_SHAPE_GATE_SIZE = 512

# grid-searched on astrollm's val set -- see upstream data/shape_features.py
_COMET_GATE_CENTER_DIST = 0.20
_COMET_GATE_N_COMPONENTS = 80

_SESSION_CACHE: dict = {}
_SESSION_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# availability / model resolution
# ---------------------------------------------------------------------------

def onnxruntime_available() -> bool:
    return _ort is not None


def bundled_model_path() -> str:
    """Path to the model shipped inside the package (``src/data/astrollm.onnx``)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'astrollm.onnx')


def resolve_model_path(explicit: Optional[str] = None) -> Optional[str]:
    """Return the first usable model path: an explicit override, else the
    bundled copy. ``None`` if neither exists on disk."""
    for cand in (explicit, bundled_model_path()):
        if cand and os.path.isfile(cand):
            return cand
    return None


def _get_session(model_path: str):
    sess = _SESSION_CACHE.get(model_path)
    if sess is None:
        with _SESSION_LOCK:
            sess = _SESSION_CACHE.get(model_path)
            if sess is None:
                sess = _ort.InferenceSession(
                    model_path, providers=['CPUExecutionProvider'])
                _SESSION_CACHE[model_path] = sess
    return sess


# ---------------------------------------------------------------------------
# preprocessing (numpy/scipy port of upstream data/imageops.py)
# ---------------------------------------------------------------------------

def _stretch_to_uint8(img: np.ndarray, low_pct: float = 0.5,
                      high_pct: float = 99.5) -> np.ndarray:
    """Per-channel percentile clip + linear stretch to uint8. Byte-for-byte
    the upstream ``imageops.stretch_to_uint8`` (already numpy-only there)."""
    if img.ndim == 2:
        lo, hi = np.percentile(img, [low_pct, high_pct])
        if hi <= lo:
            hi = lo + 1.0
        out = np.clip((img.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
        return (out * 255).astype(np.uint8)
    out = np.empty(img.shape, dtype=np.uint8)
    for c in range(img.shape[2]):
        lo, hi = np.percentile(img[..., c], [low_pct, high_pct])
        if hi <= lo:
            hi = lo + 1.0
        chan = np.clip((img[..., c].astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
        out[..., c] = (chan * 255).astype(np.uint8)
    return out


def _resize_center_crop(img: np.ndarray, size: int) -> np.ndarray:
    """Resize shorter side to ``size``, centre-crop to ``size x size``.

    Upstream used ``cv2.resize`` (INTER_AREA downscale / INTER_CUBIC
    upscale). This uses ``scipy.ndimage.zoom`` (bilinear) with a Gaussian
    pre-filter on downscale to approximate area averaging -- the closest
    pixel match available without re-adding an OpenCV dependency (~0.5/255
    mean diff on a real frame). The class/flag heads are unaffected to
    ~0.002; the ``quality`` regression head is resampling-sensitive and can
    shift a few points on its 0-400 scale, which is immaterial here since
    it's only ever used session-relative (outlier detection across frames
    scored the same way), never as an absolute gate.
    """
    h, w = img.shape[:2]
    scale = size / min(h, w)
    f = img.astype(np.float32)
    if scale < 1.0:
        sigma = ((1.0 / scale) - 1.0) / 2.0
        if sigma > 0.01:
            f = ndimage.gaussian_filter(
                f, (sigma, sigma, 0) if f.ndim == 3 else (sigma, sigma))
    factor = (scale, scale, 1) if f.ndim == 3 else (scale, scale)
    z = ndimage.zoom(f, factor, order=1, mode='reflect')
    zh, zw = z.shape[:2]
    top = max(0, (zh - size) // 2)
    left = max(0, (zw - size) // 2)
    out = z[top:top + size, left:left + size]
    # zoom's rounding can land a pixel short of `size`; pad-edge if so.
    if out.shape[0] != size or out.shape[1] != size:
        pad = [(0, max(0, size - out.shape[0])), (0, max(0, size - out.shape[1]))]
        if out.ndim == 3:
            pad.append((0, 0))
        out = np.pad(out, pad, mode='edge')[:size, :size]
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# comet shape gate (numpy/scipy port of upstream data/shape_features.py)
# ---------------------------------------------------------------------------

def _blob_shape_features(gray: np.ndarray, thresh_percentile: float = 99.0,
                         min_area: int = 5):
    """``(n_components, center_dist_norm)`` for the largest bright connected
    component in a grayscale uint8 image. 8-connectivity. Matches upstream
    ``shape_features.blob_shape_features`` (which used
    ``cv2.connectedComponentsWithStats``)."""
    h, w = gray.shape
    thresh_val = max(float(np.percentile(gray, thresh_percentile)), 1.0)
    binary = gray > thresh_val
    labels, n_labels = ndimage.label(binary, structure=np.ones((3, 3), dtype=int))
    if n_labels == 0:
        return 0, 1.0
    areas = np.bincount(labels.ravel())[1:]  # drop background label 0
    keep = np.nonzero(areas >= min_area)[0]
    if keep.size == 0:
        return 0, 1.0
    largest = int(keep[np.argmax(areas[keep])]) + 1
    cy, cx = ndimage.center_of_mass(binary, labels, largest)
    center_dist = ((cx - w / 2) ** 2 + (cy - h / 2) ** 2) ** 0.5
    center_dist_norm = center_dist / (0.5 * (w ** 2 + h ** 2) ** 0.5)
    return int(keep.size), float(center_dist_norm)


def _gate_comet_prediction(probs: np.ndarray, gray: np.ndarray,
                           categories) -> int:
    """Raw argmax, unless it's ``comet`` and the shape features disagree, in
    which case fall back to the 2nd-best class. Only ever makes the comet
    head more conservative."""
    order = np.argsort(probs)[::-1]
    top = int(order[0])
    if 'comet' not in categories or categories[top] != 'comet':
        return top
    n_components, center_dist_norm = _blob_shape_features(gray)
    if (center_dist_norm > _COMET_GATE_CENTER_DIST
            or n_components > _COMET_GATE_N_COMPONENTS):
        return int(order[1])
    return top


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------

def _meta(session) -> dict:
    m = session.get_modelmeta().custom_metadata_map
    epoch = m.get('epoch')
    if epoch is not None and epoch.lstrip('-').isdigit():
        epoch = int(epoch)
    return {
        'tasks': m.get('tasks', '').split(',') if m.get('tasks') else [],
        'head_order': m.get('head_order', '').split(',') if m.get('head_order') else [],
        'categories': m.get('categories', '').split(',') if m.get('categories') else [],
        'exposures': [float(x) for x in m.get('exposures', '').split(',') if x],
        'quality_scale': float(m.get('quality_scale', '400.0')),
        'stray_light_threshold': float(m.get('stray_light_threshold', '27.0')),
        'epoch': epoch,
    }


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def score_rgb(rgb: np.ndarray, *, model_path: Optional[str] = None,
              size: int = 256, shape_gate: bool = True) -> Optional[dict]:
    """Score an ``(H, W, 3)`` RGB array (any range/dtype -- it's percentile-
    stretched here). Returns the same result dict the old
    ``infer_onnx.py --json`` produced, or ``None`` on any failure (logged).
    """
    if _ort is None:
        return None
    mp = resolve_model_path(model_path)
    if mp is None:
        return None
    try:
        arr = np.asarray(rgb)
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.moveaxis(arr, 0, -1)          # (C,H,W) -> (H,W,C)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return None
        arr = arr[:, :, :3]

        stretched = _stretch_to_uint8(arr)
        model_in = _resize_center_crop(stretched, size)
        x = (model_in.transpose(2, 0, 1).astype(np.float32) / 255.0)[np.newaxis]

        sess = _get_session(mp)
        meta = _meta(sess)
        tasks = meta['tasks'] or meta['head_order']
        raw = sess.run(None, {'image': x})
        # Upstream infer_onnx.py uses zip(..., strict=True) here: a model
        # whose graph outputs don't line up with the head_order metadata is
        # an export bug, not something to score around silently.
        if len(raw) != len(meta['head_order']):
            logger.warning(
                f"astrollm: model has {len(raw)} outputs but head_order "
                f"metadata lists {len(meta['head_order'])} -- refusing to guess")
            return None
        out = dict(zip(meta['head_order'], raw))

        # Gate EVERY head on `tasks`, never on presence in `out`/head_order.
        # The exported graph always builds all 8 heads (head_order lists them
        # all), but a head the checkpoint never trained -- e.g. `trailing` on
        # the v4 epoch-15 model -- is excluded from `tasks` and its output is
        # random noise. `trailing` and `background_grid` are deliberately not
        # surfaced below: OriginStack has no use for the spatial
        # background-grid map (it runs its own DBE), and trailing is untrained
        # on the current model. Wire either only when `'<head>' in tasks`.
        result: dict = {'checkpoint_epoch': meta['epoch'], 'tasks': tasks}
        if 'reject' in tasks:
            p = float(_sigmoid(float(out['reject'][0])))
            result['defect_probability'] = p
            result['is_defective'] = p > 0.5
        if 'quality' in tasks:
            result['quality_score'] = float(out['quality'][0]) * meta['quality_scale']
        if 'category' in tasks:
            probs = _softmax(out['category'][0])
            cats = meta['categories']
            order = np.argsort(probs)[::-1]
            top = int(order[0])
            if shape_gate and 'comet' in cats and cats[top] == 'comet':
                # Shape features are only consulted to (maybe) demote a
                # comet top pick -- compute the 512px crop lazily here.
                feat_rgb = _resize_center_crop(stretched, _SHAPE_GATE_SIZE)
                gray = (0.299 * feat_rgb[..., 0] + 0.587 * feat_rgb[..., 1]
                        + 0.114 * feat_rgb[..., 2]).astype(np.uint8)
                picked = _gate_comet_prediction(probs, gray, cats)
            else:
                picked = top
            result['category'] = cats[picked]
            result['category_confidence'] = float(probs[picked])
            result['top_categories'] = [[cats[i], float(probs[i])] for i in order[:3]]
            result['category_shape_gated'] = picked != top
        if 'exposure' in tasks:
            probs = _softmax(out['exposure'][0])
            i = int(np.argmax(probs))
            if meta['exposures']:
                result['predicted_exposure_s'] = meta['exposures'][i]
            result['exposure_confidence'] = float(probs[i])
        if 'sky_brightness' in tasks:
            result['sky_brightness'] = float(out['sky_brightness'][0]) * 255
        if 'stray_light_gradient' in tasks:
            g = float(out['stray_light_gradient'][0]) * 255
            result['stray_light_gradient'] = g
            result['stray_light_flag'] = g > meta['stray_light_threshold']
        return result
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"astrollm: in-process inference failed: {exc}")
        return None


def _load_image_any(path: str) -> Optional[np.ndarray]:
    """Load a FITS/TIFF/PNG/JPG path to an ``(H, W, 3)`` float32 RGB array.
    FITS is debayered via the pipeline's own loader; everything else goes
    through ``tifffile`` then ``Pillow``. ``None`` if unreadable."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in _FITS_EXTS:
            from src.io_fits import load_frame
            data, hdr = load_frame(path)
            arr = np.asarray(data)
            if arr.ndim == 2:
                from src.debayer import debayer
                pattern = str(hdr.get('BAYERPAT') or hdr.get('COLORTYP')
                              or 'GBRG').strip().upper()
                return debayer(arr.astype(np.float32), pattern=pattern)
            if arr.ndim == 3:
                rgb = arr.astype(np.float32)
                if rgb.shape[0] in (1, 3) and rgb.shape[-1] not in (1, 3):
                    rgb = np.moveaxis(rgb, 0, -1)
                return rgb
            return None
        try:
            import tifffile
            if ext in ('.tif', '.tiff'):
                return np.asarray(tifffile.imread(path)).astype(np.float32)
        except Exception:
            pass
        from PIL import Image
        return np.asarray(Image.open(path).convert('RGB')).astype(np.float32)
    except Exception as exc:
        logger.warning(f"astrollm: could not read {os.path.basename(path)}: {exc}")
        return None


def score_path(path: str, *, model_path: Optional[str] = None,
               size: int = 256, shape_gate: bool = True) -> Optional[dict]:
    """``score_rgb`` for a file path. Returns the result dict (with an
    ``image`` key added) or ``None`` on any failure."""
    rgb = _load_image_any(path)
    if rgb is None:
        return None
    result = score_rgb(rgb, model_path=model_path, size=size,
                       shape_gate=shape_gate)
    if result is not None:
        result['image'] = path
    return result
