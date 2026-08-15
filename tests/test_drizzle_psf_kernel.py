"""Tests for the PSF-matched drizzle resample kernel (--drizzle-kernel psf).

build_drizzle_psf_table converts an estimated PSF into a subpixel tap-weight
table via a windowed Wiener inverse filter (_build_wiener_sharpen_kernel) --
NOT the raw PSF shape, which measurably broadens stars (two Gaussians of
width sigma convolve to sigma*sqrt(2); verified directly before this design
was corrected). warp_affine_kernel_table (native, with a numpy mirror
_warp_affine_kernel_table_numpy) resamples an affine warp using that table
instead of the fixed Lanczos-3 formula. These tests check: the native kernel
matches its numpy mirror (parity, the pattern every other native kernel in
this file uses), the table's phases each sum to 1 (flux-preserving), that a
Gaussian-PSF table gives genuine (not negative) sharpening on a matched star,
and that flat noise isn't amplified (the real risk with any inverse-filter
style kernel).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.stacking import (
    build_drizzle_psf_table,
    warp_affine_psf,
    _warp_affine_kernel_table_numpy,
)

native = pytest.importorskip("astro_native")


def _gaussian_psf(size: int = 31, sigma: float = 2.0) -> np.ndarray:
    half = size // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return psf / psf.sum()


def test_table_shape_and_normalization():
    psf = _gaussian_psf()
    table, halo = build_drizzle_psf_table(psf, halo=4, phases=8)
    taps = 2 * halo + 1
    assert table.shape == (8 * 8 * taps * taps,)
    table4 = table.reshape(8, 8, taps, taps)
    for py in range(8):
        for px in range(8):
            s = table4[py, px].sum()
            assert abs(s - 1.0) < 1e-6 or s == 0.0


def test_table_is_not_the_raw_psf():
    # The whole point of the fix: the table must NOT just be the raw
    # (cropped, renormalized) PSF, since that measurably broadens stars.
    psf = _gaussian_psf()
    halo = 4
    table, _ = build_drizzle_psf_table(psf, halo=halo, phases=4)
    taps = 2 * halo + 1
    table4 = table.reshape(4, 4, taps, taps)
    center = psf.shape[0] // 2
    cropped = psf[center - halo:center + halo + 1, center - halo:center + halo + 1]
    cropped = cropped / cropped.sum()
    assert not np.allclose(table4[0, 0], cropped, atol=1e-3)


def test_native_matches_numpy_mirror_identity():
    rng = np.random.default_rng(0)
    H, W, C = 40, 48, 3
    data = rng.uniform(0, 1000, (H, W, C)).astype(np.float32)
    psf = _gaussian_psf()
    table, halo = build_drizzle_psf_table(psf, halo=4, phases=8)

    mat = [1.0, 0.0, 0.0, 1.0]
    off = [0.3, -0.7]  # subpixel shift, exercises multiple phases
    out_h, out_w = H, W

    got_native = np.asarray(native.warp_affine_kernel_table(
        np.ascontiguousarray(data), mat, off, out_h, out_w,
        np.ascontiguousarray(table, dtype=np.float64), halo, 8, 0.0))
    got_numpy = _warp_affine_kernel_table_numpy(
        data, mat, off, out_h, out_w, table, halo, 8, 0.0)

    np.testing.assert_allclose(got_native, got_numpy, atol=1e-3, rtol=1e-3)


def test_native_matches_numpy_mirror_rotation_scale():
    rng = np.random.default_rng(1)
    H, W, C = 50, 50, 3
    data = rng.uniform(0, 500, (H, W, C)).astype(np.float32)
    psf = _gaussian_psf(sigma=1.5)
    table, halo = build_drizzle_psf_table(psf, halo=4, phases=8)

    theta = np.deg2rad(3.0)
    scale = 1.7
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) * scale
    off = [2.0, -1.5]
    mat = R.ravel().tolist()

    got_native = np.asarray(native.warp_affine_kernel_table(
        np.ascontiguousarray(data), mat, off, H, W,
        np.ascontiguousarray(table, dtype=np.float64), halo, 8, 0.0))
    got_numpy = _warp_affine_kernel_table_numpy(data, mat, off, H, W, table, halo, 8, 0.0)

    np.testing.assert_allclose(got_native, got_numpy, atol=1e-3, rtol=1e-3)


def test_warp_affine_psf_dispatcher_matches_native():
    rng = np.random.default_rng(2)
    H, W, C = 32, 32, 3
    data = rng.uniform(0, 200, (H, W, C)).astype(np.float32)
    psf = _gaussian_psf()
    table, halo = build_drizzle_psf_table(psf, halo=4, phases=8)
    mat = [1.0, 0.0, 0.0, 1.0]
    off = [0.0, 0.0]

    via_dispatch = warp_affine_psf(data, mat, off, H, W, table, halo, 8, 0.0)
    via_native = np.asarray(native.warp_affine_kernel_table(
        np.ascontiguousarray(data), mat, off, H, W,
        np.ascontiguousarray(table, dtype=np.float64), halo, 8, 0.0))
    np.testing.assert_allclose(via_dispatch, via_native, atol=1e-4)


def _measure_sigma(im, H, W):
    b = np.clip(im[:, :, 0].astype(np.float64), 0, None)
    tot = b.sum()
    gy, gx = np.mgrid[0:H, 0:W]
    cy_ = (gy * b).sum() / tot
    cx_ = (gx * b).sum() / tot
    var = ((gy - cy_) ** 2 * b).sum() / tot + ((gx - cx_) ** 2 * b).sum() / tot
    return np.sqrt(var / 2)


def test_gaussian_psf_kernel_sharpens_matched_star_not_broadens():
    """A star whose true profile matches the PSF used to build the table
    should come through resampling at least as sharp as it went in -- the
    whole point of using a Wiener inverse filter instead of the raw PSF.
    Checked at a subpixel (non-integer-phase) center, the realistic case."""
    H = W = 64
    sigma_true = 2.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cy, cx = 32.3, 31.7  # subpixel center
    star = 1000.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma_true ** 2))
    img = np.stack([star] * 3, axis=-1).astype(np.float32)

    psf = _gaussian_psf(size=31, sigma=sigma_true)
    table, halo = build_drizzle_psf_table(psf, halo=4, phases=16)

    out = warp_affine_psf(img, [1.0, 0.0, 0.0, 1.0], [0.0, 0.0], H, W,
                          table, halo, 16, 0.0)

    sigma_out = _measure_sigma(out, H, W)
    assert sigma_out <= sigma_true * 1.02  # never broadens (within measurement noise)
    assert sigma_out >= sigma_true * 0.7   # sharpening stays controlled, not degenerate


def test_native_matches_numpy_mirror_degenerate_zero_weight_phase():
    """A pathological (e.g. ill-conditioned Wiener) table can have a phase
    whose taps are all exactly zero. The native kernel's interior fast path
    must still fall back to cval there, matching the numpy mirror's uniform
    'valid & nonzero-weight' rule -- not silently return 0.0 for every
    interior pixel landing in that phase (a real native/numpy parity bug
    this test guards against)."""
    H = W = 12
    halo = 1
    phases = 2
    taps = 2 * halo + 1
    data = (np.arange(H * W, dtype=np.float32).reshape(H, W, 1) + 1.0)

    table4 = np.zeros((phases, phases, taps, taps), dtype=np.float64)
    table4[1, 1] = 1.0 / (taps * taps)  # only phase (1,1) has nonzero weight
    table = table4.ravel()

    mat = [1.0, 0.0, 0.0, 1.0]
    off = [0.2, 0.3]  # frac (0.2, 0.3) -> phase (0, 0) for every output pixel
    cval = 7.5

    got_native = np.asarray(native.warp_affine_kernel_table(
        np.ascontiguousarray(data), mat, off, H, W, table, halo, phases, cval))
    got_numpy = _warp_affine_kernel_table_numpy(data, mat, off, H, W, table, halo, phases, cval)

    np.testing.assert_allclose(got_native, got_numpy)
    np.testing.assert_allclose(got_native, cval)


def test_phases_zero_raises_value_error():
    H = W = 8
    data = np.zeros((H, W, 1), dtype=np.float32)
    table = np.zeros((0,), dtype=np.float64)
    with pytest.raises(ValueError):
        native.warp_affine_kernel_table(
            np.ascontiguousarray(data), [1.0, 0.0, 0.0, 1.0], [0.0, 0.0],
            H, W, table, 1, 0, 0.0)


def test_flat_noise_is_not_amplified():
    """The real risk of any inverse-filter-flavoured kernel: noise gain.
    Resampling white noise through the table should not increase its std."""
    H = W = 64
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 1.0, (H, W))
    img = np.stack([noise] * 3, axis=-1).astype(np.float32)

    psf = _gaussian_psf(size=31, sigma=2.0)
    table, halo = build_drizzle_psf_table(psf, halo=4, phases=8)
    out = warp_affine_psf(img, [1.0, 0.0, 0.0, 1.0], [0.0, 0.0], H, W,
                          table, halo, 8, 0.0)

    assert float(np.std(out[:, :, 0])) <= float(np.std(noise)) * 1.05
