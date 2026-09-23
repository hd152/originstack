"""In-process originvision inference -- the wired-in replacement for the
old ``vendor/originvision/infer_onnx.py`` subprocess.

Two backends, picked automatically:

* **native** -- ``astro_native.originvision_score`` runs the whole path
  (preprocessing + the ONNX forward pass via the pure-Rust ``tract``
  runtime) with no Python-level ONNX dependency. This is the primary path
  and the only one in the packaged app.
* **onnxruntime fallback** -- a numpy/scipy port of the upstream
  ``infer_onnx.py`` + ``data/imageops.py`` (no OpenCV) driving a Python
  ``onnxruntime`` ``InferenceSession``. Used only in a source checkout where
  ``astro_native`` isn't built. Kept converged with the native kernel:
  same preprocessing, same head gating, same `comet`-demotion.

``--originvision`` self-disables with a warning when neither backend nor the
bundled model file is available.

The bundled model lives at ``src/data/originvision.onnx`` (see
``vendor/originvision/VENDORED_FROM.txt`` for provenance / re-sync). An explicit
model path (``--originvision-model`` / ``--originvision-dir``) still overrides it.

Advisory only -- see ``src/originvision.py`` for how the result dict is used
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

try:  # native (Rust/tract) inference -- the primary backend when astro_native is built
    import astro_native as _native
    _HAS_NATIVE_OV = hasattr(_native, 'originvision_score')
except Exception:  # pragma: no cover
    _native = None
    _HAS_NATIVE_OV = False

try:  # onnxruntime -- fallback path for a source checkout without astro_native
    import onnxruntime as _ort
except Exception:  # pragma: no cover - onnxruntime absent
    _ort = None

_FITS_EXTS = ('.fits', '.fit', '.fts')

_SESSION_CACHE: dict = {}
_SESSION_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# availability / model resolution
# ---------------------------------------------------------------------------

def scoring_backend_available() -> bool:
    """True when a scoring backend exists -- the native tract kernel
    (``astro_native.originvision_score``) or the ``onnxruntime`` fallback.
    Gates whether ``--originvision`` runs."""
    return _HAS_NATIVE_OV or _ort is not None


def backend_name() -> str:
    return 'native' if _HAS_NATIVE_OV else ('onnxruntime' if _ort is not None else 'none')


def bundled_model_path() -> str:
    """Path to the model shipped inside the package (``src/data/originvision.onnx``)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'originvision.onnx')


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


def _prep_rgb(rgb: np.ndarray) -> Optional[np.ndarray]:
    """Coerce any of (H,W), (H,W,3+), (1|3,H,W) into a contiguous (H,W,3)
    float32 array, or None if it can't be interpreted as an image."""
    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.moveaxis(arr, 0, -1)             # (C,H,W) -> (H,W,C)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    return np.ascontiguousarray(arr[:, :, :3], dtype=np.float32)


def _downsample_if_large(arr: np.ndarray, max_long_side: int) -> np.ndarray:
    """Shrink ``arr`` (aspect preserved, no crop) if its longer side exceeds
    ``max_long_side``, same gaussian-prefiltered bilinear zoom as
    ``_resize_center_crop`` -- just earlier, before the percentile stretch,
    and without the crop.

    Measured on a full Origin sensor frame (1936x1096): the percentile
    stretch + final resize scale with *input* pixel count even though only a
    256x256 crop is ever used, so at full resolution 85% of a
    ``score_rgb`` call (1315 -> 196 ms) was spent processing pixels the model
    never sees. Both backends call this identically (before the
    native/onnxruntime dispatch below), so native/fallback parity holds and
    this is a pure precomputation, not a behaviour fork.

    Not free of numerical effect -- a second resampling stage changes the
    stretch's own percentile estimate and adds another antialiasing pass on
    top of ``_resize_center_crop``'s. Validated against the un-downsampled
    path the same way ``_resize_center_crop``'s own cv2->scipy swap was
    (docstring above): category/defect/stray-light flags unchanged, quality
    score within the same few-points-on-a-0-400-scale tolerance already
    accepted there (session-relative only, never an absolute gate).
    """
    h, w = arr.shape[:2]
    long_side = max(h, w)
    if long_side <= max_long_side:
        return arr
    scale = max_long_side / long_side
    sigma = ((1.0 / scale) - 1.0) / 2.0
    f = arr
    if sigma > 0.01:
        f = ndimage.gaussian_filter(f, (sigma, sigma, 0) if f.ndim == 3 else (sigma, sigma))
    factor = (scale, scale, 1) if f.ndim == 3 else (scale, scale)
    return np.ascontiguousarray(ndimage.zoom(f, factor, order=1, mode='reflect'),
                                dtype=np.float32)


# `_resize_center_crop` already needs 2x oversample margin on the shorter
# side to antialias well into `size`; capping the longer side at this factor
# leaves that margin on both axes while still discarding the bulk of a
# full-res frame's pixels before the expensive full-frame percentile scan.
#
# Opt-in (score_rgb's fast_preprocess=False by default), not a default-on
# speedup: measured on a real full-res frame (1370x2833, Fireworks Galaxy),
# the `reject`/`quality` heads are genuinely resolution-sensitive, not just
# resampling-noise-sensitive like _resize_center_crop's own cv2->scipy swap
# (which left them within ~0.002/a few points). Sweeping this factor on that
# same frame (ms/call, speedup, defect_probability delta, quality_score
# delta vs. no pre-downsample):
#   factor=2 (512px):   400ms  7.04x  defect +0.40  quality -160
#   factor=3 (768px):   413ms  6.82x  defect +0.26  quality  -92
#   factor=4 (1024px):  492ms  5.72x  defect +0.13  quality  -35
#   factor=6 (1536px):  822ms  3.42x  defect +0.06  quality  -22
#   factor=8 (2048px): 1494ms  1.88x  defect +0.04  quality   -3
# defect_probability swinging by tenths (not thousandths) on one real frame
# at every tested factor -- including the mild ones -- is enough to flip
# is_defective on a frame that sits near 0.5, and that flag feeds
# auto_settings.py's defensive nudges (trail-reject on, stronger chroma
# denoise), not just a log line. Left off by default pending a decision on
# whether/how to expose it (a CLI flag, a specific factor) rather than
# shipping a silent accuracy/speed tradeoff.
_PREDOWNSAMPLE_FACTOR = 4


def score_rgb(rgb: np.ndarray, *, model_path: Optional[str] = None,
              size: int = 256, shape_gate: bool = True,
              fast_preprocess: bool = False) -> Optional[dict]:
    """Score an ``(H, W, 3)`` RGB array (any range/dtype -- it's percentile-
    stretched here). Returns the result dict, or ``None`` on any failure
    (logged). Uses the native tract kernel when ``astro_native`` is built,
    otherwise the ``onnxruntime`` fallback.

    ``fast_preprocess`` (default off): pre-downsample large inputs before the
    percentile stretch (see ``_downsample_if_large``'s docstring for the
    measured speed-vs-accuracy tradeoff) -- real speedup, but the `reject`/
    `quality` heads shift more than this project's usual resampling
    tolerance, so it's opt-in, not the default.
    """
    mp = resolve_model_path(model_path)
    if mp is None:
        return None
    arr = _prep_rgb(rgb)
    if arr is None:
        return None
    if fast_preprocess:
        arr = _downsample_if_large(arr, size * _PREDOWNSAMPLE_FACTOR)

    if _HAS_NATIVE_OV:
        try:
            return _native.originvision_score(arr, mp, size, shape_gate)
        except ValueError:
            return None
        except Exception as exc:            # RuntimeError from a bad model, etc.
            logger.warning(f"originvision: native inference failed ({exc}); "
                           f"{'falling back to onnxruntime' if _ort is not None else 'no fallback'}")
            if _ort is None:
                return None

    if _ort is None:
        return None
    try:
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
                f"originvision: model has {len(raw)} outputs but head_order "
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
            # `comet` is suppressed: the current checkpoint's comet class isn't
            # trusted, so a top `comet` pick is demoted to the runner-up.
            # (`shape_gate=False` disables the suppression.) Mirrors the native
            # kernel's `compute`.
            if shape_gate and len(order) > 1 and cats[top] == 'comet':
                picked = int(order[1])
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
        logger.warning(f"originvision: in-process inference failed: {exc}")
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
        logger.warning(f"originvision: could not read {os.path.basename(path)}: {exc}")
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
