"""Learned denoiser (``--denoiser nn``): a small U-Net trained with Noise2Noise on real
Origin sessions.

Training needs no clean images. ``tools/gen_n2n_pairs.py`` stacks each session's frames
``i % 4 == q`` into four quarter-stacks: the same sky with independent noise. A network
trained to predict one from another with an L2 loss converges to the noise-free mean,
because the target's own noise is zero-mean and independent of the input
(Lehtinen et al. 2018). ``tools/train_nn_denoise.py`` trains it; the weights ship as
``src/data/nn_denoise.npz`` and run here in plain numpy (no torch at run time).

The network works in noise units: each channel has a smooth background removed and is
divided by its own measured noise sigma (``normalise``, shared with the training tool so
both see identical inputs). That makes it independent of exposure, gain and stack depth,
so one model trained on quarter-stacks applies to a full stack. Everything is undone
exactly after the forward pass; ``strength`` blends towards the input.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

try:
    import astro_native as _native
    _HAS_NATIVE = hasattr(_native, 'nn_conv3x3')
except Exception:
    _native = None
    _HAS_NATIVE = False

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'nn_denoise.npz')
# The same network trained on seeing pairs (--deconvolve nn, dev-notes/nn-deconv-runbook.md):
# sharpest-30% stack as target, so it sharpens as well as denoises. Run with nn_denoise().
DECONV_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'nn_deconv.npz')

BG_BLOCK = 64          # background mesh cell (px); the surface is smooth on this scale
ASINH_SOFT = 4.0       # network input = asinh(z / ASINH_SOFT): linear near the sky, compressed on stars
TILE = 384             # forward-pass tile (px), a multiple of 4 (two 2x poolings)
MARGIN = 32            # context on each side of a tile; > the network's receptive-field radius (~24 px)
LEAK = 0.1             # LeakyReLU slope
RESTORE_SIGMA = 4.0    # px; low-frequency restore of faint extended light (nn_denoise)

_weights_cache: Dict[str, Dict[str, np.ndarray]] = {}


# ── Normalisation (shared with tools/train_nn_denoise.py) ─────────────────────

def noise_sigma(img: np.ndarray) -> np.ndarray:
    """Per-channel noise sigma of an (H, W, C) image: MAD of lag-4 row differences on every
    2nd row. Lag 4 skips the short-range correlation the debayer and warp put into the noise,
    and differences cancel the signal on all but the sharpest structure."""
    out = np.empty(img.shape[2], np.float64)
    for c in range(img.shape[2]):
        x = img[::2, :, c]
        d = (x[:, 4:] - x[:, :-4]).ravel()
        d = d[np.isfinite(d)]
        out[c] = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0) if d.size else 1.0
    return np.maximum(out, 1e-12)


def background_grid(img: np.ndarray) -> np.ndarray:
    """Per-channel smooth background of an (H, W, C) image on a ``BG_BLOCK`` mesh:
    block medians, lightly smoothed. Evaluate it on any window with ``background_at``."""
    h, w, c = img.shape
    gh, gw = max(1, h // BG_BLOCK), max(1, w // BG_BLOCK)
    crop = img[:gh * BG_BLOCK, :gw * BG_BLOCK]
    blocks = crop.reshape(gh, BG_BLOCK, gw, BG_BLOCK, c).transpose(0, 2, 4, 1, 3).reshape(gh, gw, c, -1)
    grid = np.nanmedian(blocks, axis=-1)
    for k in range(c):
        grid[:, :, k] = ndi.median_filter(grid[:, :, k], size=3, mode='nearest')
        grid[:, :, k] = ndi.gaussian_filter(grid[:, :, k], 1.0, mode='nearest')
    return grid.astype(np.float32)


def background_at(grid: np.ndarray, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """The background surface over rows ``y0:y0+h``, cols ``x0:x0+w`` (bilinear on the mesh,
    block centres at ``(i + 0.5) * BG_BLOCK``)."""
    gy = (np.arange(y0, y0 + h, dtype=np.float64) + 0.5) / BG_BLOCK - 0.5
    gx = (np.arange(x0, x0 + w, dtype=np.float64) + 0.5) / BG_BLOCK - 0.5
    yy, xx = np.meshgrid(gy, gx, indexing='ij')
    out = np.empty((h, w, grid.shape[2]), np.float32)
    for k in range(grid.shape[2]):
        out[:, :, k] = ndi.map_coordinates(grid[:, :, k], [yy, xx], order=1, mode='nearest')
    return out


def normalise(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(background grid, sigma)`` for an (H, W, C) image; ``z = (img - bg) / sigma``."""
    return background_grid(img), noise_sigma(img)


def to_input(z: np.ndarray) -> np.ndarray:
    """Network input from noise-unit data."""
    return np.arcsinh(z / ASINH_SOFT)


def output_scale(z: np.ndarray) -> np.ndarray:
    """The network predicts its correction in units of the expected local noise,
    ~ sqrt(1 + signal) in sky-sigma units (sky-limited at 1, Poisson above)."""
    return np.sqrt(1.0 + np.maximum(z, 0.0))


# ── Network ──────────────────────────────────────────────────────────────────

def load_weights(path: Optional[str] = None) -> Dict[str, np.ndarray]:
    path = path or _MODEL_PATH
    w = _weights_cache.get(path)
    if w is None:
        with np.load(path) as f:
            w = {k: f[k].astype(np.float32) for k in f.files}
        _weights_cache[path] = w
    return w


def model_available(path: Optional[str] = None) -> bool:
    return os.path.exists(path or _MODEL_PATH)


def _conv3x3(x: np.ndarray, w: np.ndarray, b: np.ndarray, act: bool = True) -> np.ndarray:
    """'same' 3x3 convolution with zero padding (+ bias, LeakyReLU when ``act``)."""
    if _HAS_NATIVE:
        return _native.nn_conv3x3(np.ascontiguousarray(x, dtype=np.float32), np.ascontiguousarray(w),
                                  np.ascontiguousarray(b), float(LEAK), bool(act))
    return _conv3x3_numpy(x, w, b, act)


def _conv3x3_numpy(x: np.ndarray, w: np.ndarray, b: np.ndarray, act: bool = True) -> np.ndarray:
    """'same' 3x3 convolution with zero padding, as torch.nn.Conv2d(padding=1).
    x: (C, H, W); w: (O, C, 3, 3). One (O, 9C) @ (9C, H*W) product over an im2col buffer:
    2.7x faster than nine (O, C) products, whose inner dimension is too small for BLAS."""
    c, h, wd = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    cols = np.empty((9, c, h, wd), np.float32)
    for ky in range(3):
        for kx in range(3):
            cols[ky * 3 + kx] = xp[:, ky:ky + h, kx:kx + wd]
    out = w.transpose(0, 2, 3, 1).reshape(w.shape[0], -1) @ cols.reshape(9 * c, -1)
    out += b[:, None]
    out = out.reshape(-1, h, wd)
    if act:
        np.maximum(out, LEAK * out, out=out)
    return out


def _pool2(x: np.ndarray) -> np.ndarray:
    """2x2 average pool (torch AvgPool2d(2)); four strided adds beat a strided mean 3x."""
    s = x[:, ::2, ::2] + x[:, 1::2, ::2]
    s += x[:, ::2, 1::2]
    s += x[:, 1::2, 1::2]
    s *= np.float32(0.25)
    return s


def _up2_cat(x: np.ndarray, skip: np.ndarray) -> np.ndarray:
    """``concat([2x nearest-neighbour upsample of x, skip])`` (torch Upsample(scale_factor=2)
    then torch.cat), written straight into one buffer: four strided copies, no temporaries."""
    c = x.shape[0]
    out = np.empty((c + skip.shape[0],) + skip.shape[1:], np.float32)
    for dy in (0, 1):
        for dx in (0, 1):
            out[:c, dy::2, dx::2] = x
    out[c:] = skip
    return out


def forward(x: np.ndarray, w: Dict[str, np.ndarray]) -> np.ndarray:
    """U-Net forward pass on (C, H, W) float32, H and W multiples of 4. Mirrors
    ``NnDenoiseNet`` in tools/train_nn_denoise.py layer for layer."""
    def conv(name, t, act=True):
        return _conv3x3(t, w[name + '.weight'], w[name + '.bias'], act)

    s0 = conv('e0b', conv('e0a', x))
    s1 = conv('e1b', conv('e1a', _pool2(s0)))
    m = conv('mb', conv('ma', _pool2(s1)))
    d1 = conv('d1b', conv('d1a', _up2_cat(m, s1)))
    d0 = conv('d0b', conv('d0a', _up2_cat(d1, s0)))
    out = np.einsum('oc,chw->ohw', w['out.weight'][:, :, 0, 0], d0, optimize=True)
    return out + w['out.bias'][:, None, None]


def denoise_z(z: np.ndarray, w: Dict[str, np.ndarray]) -> np.ndarray:
    """Denoise noise-unit data (H, W, C) tile by tile; returns the same shape."""
    h, wd, _ = z.shape
    ph, pw = (-h) % TILE, (-wd) % TILE
    zp = np.pad(z, ((MARGIN, MARGIN + ph), (MARGIN, MARGIN + pw), (0, 0)), mode='reflect')
    out = np.empty_like(z)
    for y0 in range(0, h, TILE):
        for x0 in range(0, wd, TILE):
            t = zp[y0:y0 + TILE + 2 * MARGIN, x0:x0 + TILE + 2 * MARGIN]
            pred = forward(to_input(t).transpose(2, 0, 1).astype(np.float32), w).transpose(1, 2, 0)
            res = (t + pred * output_scale(t))[MARGIN:MARGIN + TILE, MARGIN:MARGIN + TILE]
            hh, ww = min(TILE, h - y0), min(TILE, wd - x0)
            out[y0:y0 + hh, x0:x0 + ww] = res[:hh, :ww]
    return out


def nn_denoise(img: np.ndarray, strength: float = 1.0, model_path: Optional[str] = None,
               restore_sigma: float = RESTORE_SIGMA,
               weights: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """Denoise a linear (H, W, 3) image with the bundled Noise2Noise U-Net.

    ``restore_sigma`` (px, 0 = off) adds back the Gaussian low-pass of what the network
    removed: pixel-scale noise stays removed, faint extended light it flattened returns.
    A minimum-MSE denoiser shrinks faint isolated sources toward the sky (they are rare
    in its training data); injected exponential discs of 0.5-3 sky sigma kept 48-79% of
    their flux without it, 86-97% with 4 px, at roughly twice the (still ~20x smaller
    than the wavelet denoiser's) residual error. ``strength`` 0-1 then blends between
    the input (0) and that result (1). ``weights`` (e.g. from ``refit_output_layer``)
    replace the file's."""
    w = weights if weights is not None else load_weights(model_path)
    img = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(img)
    src = np.where(finite, img, 0.0).astype(np.float32)
    grid, sigma = normalise(src)
    bg = background_at(grid, 0, 0, *src.shape[:2])
    sig = sigma.astype(np.float32)
    z = (src - bg) / sig
    den = denoise_z(z, w) * sig + bg
    if restore_sigma > 0:
        res = src - den
        for c in range(res.shape[2]):
            den[..., c] += ndi.gaussian_filter(res[..., c], restore_sigma, mode='reflect')
    out = src + np.float32(strength) * (den - src)
    return np.where(finite, out, img).astype(np.float32)


# ── Per-session refit of the output layer (--nn-refit) ───────────────────────

SKY_MATCH_SIGMA = 32.0     # px; far above a star's scale, so its sharpening survives
REFIT_LAMBDA = 0.01        # ridge toward the shipped weights, relative to the mean diagonal
REFIT_SHARP_FRAC = 0.3     # sharpest frames in the session's target stack, as in training
REFIT_MIN_FRAMES = 10      # fewer frames: no split, the shipped weights


def target_variance(z: np.ndarray, sky_level: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-pixel noise variance of a stack in noise units, ``a * (1 + max(signal, 0) / B)``:
    sky-limited Poisson noise grows with the light, at the rate the sky level B sets.

    ``a`` is measured directly (MAD of lag-4 row differences on the faintest 30% of the
    smoothed image -- lag 4 skips the debayer/warp correlation). ``sky_level`` is B per
    channel in the same noise units. Two fits were tried and dropped: the intercept of a
    variance-vs-signal line fell to its 1e-3 floor on several sessions (1000x sky weight,
    loss in the thousands), and its slope, from lag-4 differences on stars and nebulae,
    measured their structure rather than their noise (normalised noise 0.14-12 on the
    brightest pixels). Without ``sky_level`` (or B <= 0) the variance is ``a`` alone."""
    out = np.empty_like(z)
    for ch in range(z.shape[2]):
        x = z[..., ch]
        sm = ndi.gaussian_filter(x, 2.0)
        d = ((x[:, 4:] - x[:, :-4]) / np.sqrt(2.0)).ravel()
        sv = sm[:, 2:-2].ravel()
        sky = d[sv < np.quantile(sv, 0.3)]
        a = max((1.4826 * np.median(np.abs(sky - np.median(sky)))) ** 2, 1e-6)
        bl = float(sky_level[ch]) if sky_level is not None else 0.0
        out[..., ch] = a * (1.0 + np.maximum(sm, 0.0) / bl) if bl > 0 else a
    return out


def sky_level_z(bg: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """The sky level B per channel in noise units (median background / sigma)."""
    return np.median(bg.reshape(-1, bg.shape[-1]), axis=0) / sigma


def model_digest(path: Optional[str] = None) -> str:
    """Short hash of a weights file: a saved refit belongs to the model it was fitted on."""
    import hashlib
    with open(path or _MODEL_PATH, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def load_refit(path: str) -> Optional[Dict[str, np.ndarray]]:
    """The deconvolver weights with a saved ``--nn-refit`` output layer
    (``<stem>_nnrefit.npz``), or None when the file is missing or was fitted on another
    model file."""
    if not os.path.exists(path) or not model_available(DECONV_MODEL_PATH):
        return None
    with np.load(path) as f:
        if str(f['model_sha']) != model_digest(DECONV_MODEL_PATH):
            return None
        w = dict(load_weights(DECONV_MODEL_PATH))
        w['out.weight'] = f['out_weight'].astype(np.float32)
        w['out.bias'] = f['out_bias'].astype(np.float32)
    return w


def features(x: np.ndarray, w: Dict[str, np.ndarray]) -> np.ndarray:
    """d0, the last feature map of ``forward`` (which maps it through the 1x1 'out' layer)."""
    def conv(name, t, act=True):
        return _conv3x3(t, w[name + '.weight'], w[name + '.bias'], act)
    s0 = conv('e0b', conv('e0a', x))
    s1 = conv('e1b', conv('e1a', _pool2(s0)))
    m = conv('mb', conv('ma', _pool2(s1)))
    d1 = conv('d1b', conv('d1a', _up2_cat(m, s1)))
    return conv('d0b', conv('d0a', _up2_cat(d1, s0)))


def refit_output_layer(a: np.ndarray, b: np.ndarray, w: Dict[str, np.ndarray],
                       lam: float = REFIT_LAMBDA, cancel_event=None) -> Dict[str, np.ndarray]:
    """Refit the 1x1 output layer on a session's own seeing pair, in closed form.

    ``a``: stack of all but the sharpest frames, ``b``: of the sharpest (H, W, 3) linear.
    The prediction is linear in d0, so the training loss ``w (zA + pred s - zB)^2`` (with
    the training's sky match and target-variance weight) is a weighted least-squares problem
    in the layer's 3 x 25 weights; solved per output channel with a ridge toward the
    shipped weights. Held-out halves of 9 held-out pairs (tools/eval_session_refit.py,
    dev-notes/nn-deconv-session-refit.md): error -9.1% on average, lower on 9/9, mean
    sharpness unchanged, sky error +6%. Returns a copy of ``w`` with 'out.*' replaced."""
    b = b - np.stack([ndi.gaussian_filter(b[..., c] - a[..., c], SKY_MATCH_SIGMA) for c in range(3)], -1)
    grid, sigma = normalise(a)
    h, wd, _ = a.shape
    bg = background_at(grid, 0, 0, h, wd)
    s32 = sigma.astype(np.float32)
    za, zb = (a - bg) / s32, (b - bg) / s32
    wt = (1.0 / target_variance(zb, sky_level_z(bg, sigma))).astype(np.float32)
    ph, pw = (-h) % TILE, (-wd) % TILE
    zp = np.pad(za, ((MARGIN, MARGIN + ph), (MARGIN, MARGIN + pw), (0, 0)), mode='reflect')
    nf = w['out.weight'].shape[1] + 1
    A = np.zeros((3, nf, nf))
    B = np.zeros((3, nf))
    edge = 16                                   # skip the stack's ragged edge
    for y0 in range(0, h, TILE):
        for x0 in range(0, wd, TILE):
            if cancel_event is not None and cancel_event.is_set():
                from src.models import RunCancelled
                raise RunCancelled()
            t = zp[y0:y0 + TILE + 2 * MARGIN, x0:x0 + TILE + 2 * MARGIN]
            d0 = features(to_input(t).transpose(2, 0, 1).astype(np.float32), w)[:, MARGIN:MARGIN + TILE,
                                                                                 MARGIN:MARGIN + TILE]
            y1, x1 = min(y0 + TILE, h - edge), min(x0 + TILE, wd - edge)
            ys, xs = max(y0, edge), max(x0, edge)
            if y1 <= ys or x1 <= xs:
                continue
            d0 = d0[:, ys - y0:y1 - y0, xs - x0:x1 - x0]
            X = np.concatenate([d0.reshape(d0.shape[0], -1),
                                np.ones((1, d0[0].size), np.float32)]).astype(np.float64)
            at = za[ys:y1, xs:x1]
            s = output_scale(at)
            r = (zb[ys:y1, xs:x1] - at) / s
            q = wt[ys:y1, xs:x1] * s ** 2
            for c in range(3):
                qc = q[..., c].ravel().astype(np.float64)
                A[c] += (X * qc) @ X.T
                B[c] += X @ (qc * r[..., c].ravel())
    W0 = w['out.weight'][:, :, 0, 0].astype(np.float64)
    b0 = w['out.bias'].astype(np.float64)
    Wn, bn = np.empty_like(W0), np.empty_like(b0)
    for c in range(3):
        reg = lam * np.trace(A[c]) / nf
        th = np.linalg.solve(A[c] + reg * np.eye(nf), B[c] + reg * np.r_[W0[c], b0[c]])
        Wn[c], bn[c] = th[:-1], th[-1]
    out = dict(w)
    out['out.weight'] = Wn.astype(np.float32)[:, :, None, None]
    out['out.bias'] = bn.astype(np.float32)
    return out
