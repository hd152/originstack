"""Learned star removal (``--star-removal nn``): a U-Net that takes the stars out of a
linear stack, trained on real Origin stacks with injected stars.

Training needs starless targets, which no real image has. ``tools/train_nn_starless.py``
uses real stacks as the *background* and adds stars to them: each session's own
empirical PSF (median of its bright isolated stars, per channel, so the blue halo and
the session's seeing come with it), its own star colours and its own saturation level,
plus the stars' shot noise. The network learns to return the stack without the added
stars; the stack's own stars are masked out of the loss, so it is never asked to keep or
remove them. Everything the network sees at run time -- galaxies, nebulae, gradients,
noise -- was present in the backgrounds it learned to keep.

Same normalisation, layers and forward-pass code as the learned denoiser
(``src/nn_denoise.py``): noise units, ``asinh`` input, 3x3 convolutions (native kernel),
2x2 average pooling and nearest upsampling, here with four pooling levels so that the
receptive field (~120 px) spans a bright star's halo. The output is only ever allowed
to *remove* light: ``starless <= input`` pixel by pixel, so stars + starless == input
exactly and nothing brighter than the data is created. What lies under a star in the
starless layer is the network's estimate of the background there, not a measurement.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from src import nn_denoise as _nd

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'nn_starless.npz')

TILE = 512             # forward-pass tile (px), a multiple of 16 (four 2x poolings)
MARGIN = 64            # context on each side of a tile; > the receptive-field radius (~60 px)


def model_available(path: Optional[str] = None) -> bool:
    return os.path.exists(path or _MODEL_PATH)


def depth_of(w: Dict[str, np.ndarray]) -> int:
    """Number of pooling levels of a weight set (encoder blocks ``e0a`` .. ``e{D-1}a``)."""
    d = 0
    while f'e{d}a.weight' in w:
        d += 1
    return d


def forward(x: np.ndarray, w: Dict[str, np.ndarray]) -> np.ndarray:
    """U-Net forward pass on (C, H, W) float32, H and W multiples of 2**depth. Mirrors
    ``StarlessNet`` in tools/train_nn_starless.py layer for layer; with depth 2 it is the
    denoiser's network."""
    def conv(name, t, act=True):
        return _nd._conv3x3(t, w[name + '.weight'], w[name + '.bias'], act)

    skips: List[np.ndarray] = []
    t = x
    for k in range(depth_of(w)):
        t = conv(f'e{k}b', conv(f'e{k}a', t))
        skips.append(t)
        t = _nd._pool2(t)
    t = conv('mb', conv('ma', t))
    for k in reversed(range(len(skips))):
        t = conv(f'd{k}b', conv(f'd{k}a', _nd._up2_cat(t, skips[k])))
    out = np.einsum('oc,chw->ohw', w['out.weight'][:, :, 0, 0], t, optimize=True)
    return out + w['out.bias'][:, None, None]


def starless_z(z: np.ndarray, w: Dict[str, np.ndarray]) -> np.ndarray:
    """Remove the stars from noise-unit data (H, W, C) tile by tile; same shape. The
    network predicts the change in units of the expected local noise (as the denoiser)."""
    h, wd, _ = z.shape
    ph, pw = (-h) % TILE, (-wd) % TILE
    zp = np.pad(z, ((MARGIN, MARGIN + ph), (MARGIN, MARGIN + pw), (0, 0)), mode='reflect')
    out = np.empty_like(z)
    for y0 in range(0, h, TILE):
        for x0 in range(0, wd, TILE):
            t = zp[y0:y0 + TILE + 2 * MARGIN, x0:x0 + TILE + 2 * MARGIN]
            pred = forward(_nd.to_input(t).transpose(2, 0, 1).astype(np.float32), w).transpose(1, 2, 0)
            res = (t + pred * _nd.output_scale(t))[MARGIN:MARGIN + TILE, MARGIN:MARGIN + TILE]
            hh, ww = min(TILE, h - y0), min(TILE, wd - x0)
            out[y0:y0 + hh, x0:x0 + ww] = res[:hh, :ww]
    return out


def remove_stars_nn(img: np.ndarray, model_path: Optional[str] = None
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """``(starless, stars)`` for a linear (H, W, 3) image, with ``starless + stars == img``
    and ``stars >= 0`` (non-finite pixels pass through in ``starless``)."""
    w = _nd.load_weights(model_path or _MODEL_PATH)
    img = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(img)
    src = np.where(finite, img, 0.0).astype(np.float32)
    grid, sigma = _nd.normalise(src)
    bg = _nd.background_at(grid, 0, 0, *src.shape[:2])
    sig = sigma.astype(np.float32)
    starless = starless_z((src - bg) / sig, w) * sig + bg
    starless = np.minimum(starless, src).astype(np.float32)    # only ever remove light
    stars = np.where(finite, src - starless, 0.0).astype(np.float32)
    starless = np.where(finite, starless, img).astype(np.float32)
    return starless, stars
