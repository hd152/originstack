"""Multiresolution exposure fusion (Mertens, Kautz & Van Reeth 2007).

Blends two or more differently-exposed versions of the same scene by
weighting each pixel of each source by local contrast, saturation, and
"well-exposedness" (how close to mid-grey it is, penalising both near-black
and near-clipped pixels), then blending the weighted sources through a
Laplacian pyramid instead of compositing through a single spatial mask.

Used by ``src/pipeline.py``'s ``--hdr-combine`` (short-exposure sub-stack
recovering saturated star cores in the main long-exposure stack) as an
alternative to its original sigmoid-threshold blend (``--hdr-blend-mode
fusion`` vs. the default ``threshold``) -- a hard/sigmoid threshold blends
cleanly within each region but can leave a visible seam at the transition
band; Laplacian-pyramid blending distributes each source's contribution
across spatial frequency, not just a spatial mask, which is what avoids
that seam. No explicit HDR radiance map or tone-mapping step, matching the
original Mertens formulation.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


def _fit_shape(x: np.ndarray, target_shape) -> np.ndarray:
    """Pad (edge) or crop x's first two axes to exactly target_shape[:2] --
    pyramid up/downsampling by 2 doesn't round-trip exactly at odd sizes."""
    h, w = x.shape[0], x.shape[1]
    th, tw = target_shape[0], target_shape[1]
    if h == th and w == tw:
        return x
    pad_h, pad_w = max(0, th - h), max(0, tw - w)
    if pad_h or pad_w:
        pad_width = [(0, pad_h), (0, pad_w)] + [(0, 0)] * (x.ndim - 2)
        x = np.pad(x, pad_width, mode='edge')
    return x[:th, :tw, ...]


def _pyr_down(x: np.ndarray) -> np.ndarray:
    """Blur + downsample by 2 in the first two axes; a trailing channel
    axis, if present, is left untouched."""
    sigma = (1.0, 1.0, 0.0) if x.ndim == 3 else (1.0, 1.0)
    blurred = gaussian_filter(x, sigma=sigma, mode='reflect')
    return blurred[::2, ::2, ...]


def _pyr_up(x: np.ndarray, out_shape) -> np.ndarray:
    """Upsample to out_shape's spatial size (nearest ratio via zoom, then
    fit exactly), with a light blur to match the paper's interpolating
    upsample rather than a blocky one."""
    zy = out_shape[0] / x.shape[0]
    zx = out_shape[1] / x.shape[1]
    zoom_factors = (zy, zx, 1.0) if x.ndim == 3 else (zy, zx)
    up = zoom(x, zoom_factors, order=1)
    up = _fit_shape(up, out_shape)
    sigma = (1.0, 1.0, 0.0) if x.ndim == 3 else (1.0, 1.0)
    return gaussian_filter(up, sigma=sigma, mode='reflect')


def _gaussian_pyramid(x: np.ndarray, levels: int) -> List[np.ndarray]:
    pyr = [x]
    cur = x
    for _ in range(levels - 1):
        cur = _pyr_down(cur)
        pyr.append(cur)
    return pyr


def _laplacian_pyramid(x: np.ndarray, levels: int) -> List[np.ndarray]:
    gpyr = _gaussian_pyramid(x, levels)
    lpyr = []
    for i in range(levels - 1):
        upsampled = _pyr_up(gpyr[i + 1], gpyr[i].shape)
        lpyr.append(gpyr[i] - upsampled)
    lpyr.append(gpyr[-1])  # coarsest level: low-pass residual, stored as-is
    return lpyr


def _reconstruct_from_laplacian(lpyr: List[np.ndarray]) -> np.ndarray:
    cur = lpyr[-1]
    for i in range(len(lpyr) - 2, -1, -1):
        cur = _pyr_up(cur, lpyr[i].shape) + lpyr[i]
    return cur


def _quality_weights(norm_img: np.ndarray, contrast_w: float, saturation_w: float,
                     exposedness_w: float, sigma: float = 0.2) -> np.ndarray:
    """Per-pixel scalar quality weight (H, W) for one (H, W, C) source
    already normalised to [0, 1] -- Mertens et al.'s three measures, each
    raised to its own exponent and multiplied (their eq. 1)."""
    gray = norm_img.mean(axis=-1)
    # Contrast: a difference-of-Gaussians magnitude, a simple standard
    # stand-in for the paper's own Laplacian-magnitude measure.
    contrast = np.abs(gaussian_filter(gray, 1.0) - gaussian_filter(gray, 2.0))

    # Saturation: std across channels -- low for a washed-out/near-grey pixel.
    saturation = norm_img.std(axis=-1)

    # Well-exposedness: a Gaussian curve per channel centred at 0.5,
    # multiplied across channels -- penalises both near-black and
    # near-clipped pixels in any channel.
    exposedness = np.ones_like(gray)
    for c in range(norm_img.shape[-1]):
        exposedness *= np.exp(-0.5 * ((norm_img[..., c] - 0.5) ** 2) / (sigma ** 2))

    return (np.maximum(contrast, 1e-12) ** contrast_w
            * np.maximum(saturation, 1e-12) ** saturation_w
            * np.maximum(exposedness, 1e-12) ** exposedness_w)


def fuse_exposures(images: Sequence[np.ndarray], levels: int = 6,
                   contrast_w: float = 1.0, saturation_w: float = 1.0,
                   exposedness_w: float = 1.0) -> np.ndarray:
    """Mertens exposure fusion of >= 2 same-shape (H, W, C) float images on
    any common scale (values aren't assumed to already be [0, 1] -- only
    relatively comparable across inputs, which the caller's own sky-level
    matching should ensure for an astronomical long/short-exposure pair).

    Returns a float32 (H, W, C) array, clipped at zero (no upper clip --
    the whole point is recovering detail past the long stack's saturation).
    """
    if len(images) < 2:
        raise ValueError("fuse_exposures needs at least 2 images")
    imgs = [np.asarray(im, dtype=np.float64) for im in images]
    H, W = imgs[0].shape[:2]
    max_levels = max(1, int(np.floor(np.log2(max(2, min(H, W))))))
    levels = max(1, min(levels, max_levels))

    lo = min(float(im.min()) for im in imgs)
    hi = max(float(im.max()) for im in imgs)
    span = max(hi - lo, 1e-12)
    norm_imgs = [np.clip((im - lo) / span, 0.0, 1.0) for im in imgs]

    weights = [_quality_weights(nim, contrast_w, saturation_w, exposedness_w)
              for nim in norm_imgs]
    wsum = np.sum(weights, axis=0)
    wsum = np.where(wsum <= 1e-12, 1.0 / len(weights), wsum)
    weights = [w / wsum for w in weights]

    blended_lpyr = None
    for im, w in zip(imgs, weights):
        lpyr = _laplacian_pyramid(im, levels)
        wpyr = _gaussian_pyramid(w, levels)
        if blended_lpyr is None:
            blended_lpyr = [lv * wp[..., np.newaxis] for lv, wp in zip(lpyr, wpyr)]
        else:
            for i in range(levels):
                blended_lpyr[i] = blended_lpyr[i] + lpyr[i] * wpyr[i][..., np.newaxis]

    result = _reconstruct_from_laplacian(blended_lpyr)
    return np.clip(result, 0.0, None).astype(np.float32)
