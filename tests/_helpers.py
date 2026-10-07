"""Shared synthetic-data helpers for the test suite."""
from __future__ import annotations

import math

import numpy as np

# A Gaussian is below 1.6e-8 of its peak beyond 6 sigma.
STAMP_SIGMAS = 6.0


def add_gaussian_stars(img: np.ndarray, stars, sigma: float) -> np.ndarray:
    """Add circular Gaussian stars ``amp * exp(-r^2 / (2 sigma^2))`` to a 2-D
    ``img`` in place and return it. ``stars`` is an iterable of ``(y, x, amp)``.

    Each star is rendered only inside a +-6 sigma stamp instead of over the
    whole frame, so many stars on a large frame stay cheap; the values agree
    with a full-frame render to float precision.
    """
    H, W = img.shape
    half = int(math.ceil(STAMP_SIGMAS * sigma)) + 1
    inv = 1.0 / (2.0 * sigma * sigma)
    for y, x, amp in stars:
        y0, y1 = max(int(y) - half, 0), min(int(y) + half + 1, H)
        x0, x1 = max(int(x) - half, 0), min(int(x) + half + 1, W)
        if y0 >= y1 or x0 >= x1:
            continue
        yy = (np.arange(y0, y1) - y)[:, None]
        xx = (np.arange(x0, x1) - x)[None, :]
        img[y0:y1, x0:x1] += (amp * np.exp(-(yy * yy + xx * xx) * inv)).astype(img.dtype)
    return img


def star_field(shape, stars, sigma: float, bg: float = 0.0, noise: float = 0.0,
               rng: np.random.Generator | None = None, dtype=np.float32) -> np.ndarray:
    """A 2-D frame: constant ``bg`` + Gaussian ``stars`` [(y, x, amp), ...] +
    optional Gaussian ``noise`` drawn from ``rng`` (default seed 0).
    Accumulates in float64, returns ``dtype``."""
    img = np.full(shape, bg, dtype=np.float64)
    add_gaussian_stars(img, stars, sigma)
    if noise:
        rng = np.random.default_rng(0) if rng is None else rng
        img += rng.normal(0.0, noise, shape)
    return img.astype(dtype)


def write_fits(path, data, header=None) -> None:
    """Write ``data`` as float32 to a primary-HDU FITS file, overwriting.
    ``header`` is a dict (or astropy Header) of extra keywords."""
    from astropy.io import fits

    hdr = fits.Header()
    for k, v in (header or {}).items():
        hdr[k] = v
    fits.writeto(str(path), np.asarray(data, dtype=np.float32), header=hdr, overwrite=True)


def auto_args(**overrides):
    """argparse.Namespace with the attributes src.auto_settings' rule
    functions read, all at their non-auto defaults."""
    import argparse

    base = dict(
        _explicit_cli_dests=set(), stack_method='auto', deconvolve=True,
        debayer_method='malvar',
        denoise_acdnr=False,
        denoise_curvelet=False, deconvolve_tv=False,
        patch_registration=False, consensus_ref=False, preview_black_sigma=0.0,
        variance_stabilize=False, drizzle_scale=1.0, drizzle_kernel='lanczos3',
        hdr_combine=None, hdr_blend_mode='threshold',
        color_calibrate=False, color_calibrate_method='colorindex',
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def write_rgb_star_frame(path, H=160, W=160, shift=(0, 0), n_stars=45, seed=0,
                         bg=1000.0, header=None, hit=None) -> None:
    """Write an (H, W, 3) FITS star field on a broad nebula, for the live and
    streaming stackers. Star positions are the same in every call (one sky),
    moved by integer ``shift``; ``seed`` sets the noise. ``hit``: optional
    (y, x, amplitude) single bright outlier pixel (a cosmic ray / trail)."""
    from astropy.io import fits

    rng = np.random.default_rng(seed)
    img = np.full((H, W), bg, np.float32)
    # Broad nebula so the frame has real dynamic range / structure to register.
    yy, xx = np.mgrid[0:H, 0:W]
    img += (300.0 * np.exp(-(((xx - W / 2) / (W * 0.3)) ** 2
                            + ((yy - H / 2) / (H * 0.3)) ** 2))).astype(np.float32)
    gy, gx = np.mgrid[-4:5, -4:5]
    g = np.exp(-(gx * gx + gy * gy) / (2 * 1.5 ** 2))
    star_rng = np.random.default_rng(999)
    for _ in range(n_stars):
        y0 = star_rng.integers(20, H - 20) + shift[0]
        x0 = star_rng.integers(20, W - 20) + shift[1]
        if 4 <= y0 < H - 4 and 4 <= x0 < W - 4:
            img[y0 - 4:y0 + 5, x0 - 4:x0 + 5] += (5000 * g).astype(np.float32)
    img += rng.standard_normal((H, W)).astype(np.float32) * 5.0
    if hit is not None:
        hy, hx, amp = hit
        img[hy, hx] += amp
    hdu = fits.PrimaryHDU(data=np.stack([img, img, img], axis=2).astype(np.float32))
    for k, v in (header or {}).items():
        hdu.header[k] = v
    hdu.writeto(path, overwrite=True)
