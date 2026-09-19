"""Regressions from a real alt-az Whirlpool session (noisy subs, low altitude).

1. A thin star catalog (2-6 stars) made affine matching decline for every
   frame, so registration silently went translation-only and stars arced with
   field rotation; the residual check (needs >= 5 reference stars) never ran.
2. A corner-anchored glow beat the real galaxy in ``find_extended_source_ellipse``
   and was then protected from background extraction.
3. DBE's emission mask preserved that glow as "nebulosity" even when an
   explicit exclusion mask already protected the target.
"""
from __future__ import annotations

import numpy as np

from src.background import _dbe_prepare_emission_mask, _flatten_edge_glow
from src.models import Config
from src.registration import find_extended_source_ellipse, registration_stars


def _star_field(n, H=600, W=800, seed=1, amp=400.0, sigma=1.6):
    rng = np.random.default_rng(seed)
    img = rng.normal(1000.0, 30.0, (H, W))
    yy, xx = np.mgrid[0:H, 0:W]
    xs = rng.uniform(40, W - 40, n)
    ys = rng.uniform(40, H - 40, n)
    for x, y in zip(xs, ys):
        img += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return img.astype(np.float32), xs, ys


def _corner_glow(H, W, amp=3000.0, scale=0.12):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    return amp * np.exp(-np.hypot(yy, xx) / (scale * max(H, W)))


def test_registration_stars_rescues_thin_catalog():
    img, xs, ys = _star_field(40)
    thin = np.zeros(0, dtype=[('xcentroid', float), ('ycentroid', float), ('flux', float)])
    out = registration_stars(img, 30.0, existing=thin)
    assert out is not None and len(out) >= Config.REG_MIN_STARS
    # Rescued coordinates are in the original (unbinned) pixel grid.
    d = np.hypot(out['xcentroid'][:, None] - xs[None, :], out['ycentroid'][:, None] - ys[None, :])
    assert np.median(d.min(axis=1)) < 1.5


def test_registration_stars_keeps_adequate_catalog():
    img, _, _ = _star_field(40)
    existing = np.zeros(Config.REG_MIN_STARS, dtype=[('xcentroid', float), ('ycentroid', float), ('flux', float)])
    assert registration_stars(img, 30.0, existing=existing) is existing


def test_extended_source_ignores_corner_glow():
    H, W = 500, 700
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    rng = np.random.default_rng(0)
    galaxy = 800.0 * np.exp(-((yy - 250) ** 2 + (xx - 420) ** 2) / (2 * 30.0 ** 2))
    lum = (1000.0 + _corner_glow(H, W, amp=1500.0, scale=0.08) + galaxy
           + rng.normal(0, 20, (H, W))).astype(np.float32)
    fit = find_extended_source_ellipse(lum)
    assert fit is not None
    assert abs(fit[0] - 250) < 30 and abs(fit[1] - 420) < 30


def test_flatten_edge_glow_keeps_central_object():
    H, W = 400, 400
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    central = 500.0 * np.exp(-((yy - 300) ** 2 + (xx - 330) ** 2) / (2 * 30.0 ** 2))
    lum = 1000.0 + _corner_glow(H, W, amp=2000.0, scale=0.2) + central
    out = _flatten_edge_glow(lum, 1000.0, 100.0)
    assert out[0, 0] == 1000.0             # glow flattened
    assert out[300, 330] == lum[300, 330]  # separate central object untouched


def test_emission_mask_ignores_corner_glow_only_with_exclusion():
    H, W = 400, 500
    rng = np.random.default_rng(3)
    glow = _corner_glow(H, W, amp=1500.0, scale=0.2)
    rgb = np.stack([1000.0 + glow + rng.normal(0, 50, (H, W)) for _ in range(3)],
                   axis=-1).astype(np.float32)
    excl = np.zeros((H, W), np.float32)
    excl[190:210, 240:260] = 1.0
    with_excl, *_ = _dbe_prepare_emission_mask(rgb, None, excl, False, "t")
    without, *_ = _dbe_prepare_emission_mask(rgb, None, None, False, "t")
    assert float(np.mean(with_excl[:60, :60])) < 0.05   # glow left to DBE
    assert float(np.mean(without[:60, :60])) > 0.9      # nebula-protecting default unchanged
    assert with_excl[200, 250] == 1.0                    # target still excluded


def test_chroma_nr_keeps_zero_centred_sky_noise():
    """DBE centres the sky on zero; the sky pedestal that keeps noise positive
    runs later. Chroma NR must not clip negatives: doing so half-wave-rectified
    the sky (50% exact zeros + positive spikes) and the stretch rendered the
    spikes as white dots."""
    from src.denoising import reduce_chroma_noise
    rng = np.random.default_rng(7)
    img = rng.normal(0.0, 100.0, (128, 128, 3)).astype(np.float32)
    out = reduce_chroma_noise(img, sigma=2.0)
    assert float((out == 0).mean()) < 0.01
    assert float(out.min()) < -100.0
    assert abs(float(np.median(out))) < 10.0
