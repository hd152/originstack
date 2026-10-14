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

BG_BLOCK = 64          # background mesh cell (px); the surface is smooth on this scale
ASINH_SOFT = 4.0       # network input = asinh(z / ASINH_SOFT): linear near the sky, compressed on stars
TILE = 384             # forward-pass tile (px), a multiple of 4 (two 2x poolings)
MARGIN = 32            # context on each side of a tile; > the network's receptive-field radius (~24 px)
LEAK = 0.1             # LeakyReLU slope

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


def nn_denoise(img: np.ndarray, strength: float = 1.0, model_path: Optional[str] = None) -> np.ndarray:
    """Denoise a linear (H, W, 3) image with the bundled Noise2Noise U-Net.

    ``strength`` 0-1 blends between the input (0) and the network's output (1)."""
    w = load_weights(model_path)
    img = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(img)
    src = np.where(finite, img, 0.0).astype(np.float32)
    grid, sigma = normalise(src)
    bg = background_at(grid, 0, 0, *src.shape[:2])
    sig = sigma.astype(np.float32)
    z = (src - bg) / sig
    den = denoise_z(z, w) * sig + bg
    out = src + np.float32(strength) * (den - src)
    return np.where(finite, out, img).astype(np.float32)
