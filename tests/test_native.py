"""Parity tests for the optional native (Rust) kernels.

Skipped entirely when `astro_native` is not built/installed, so the suite
still passes in a pure-Python environment.
"""
import numpy as np
import pytest

import astro_stack as astro

native = pytest.importorskip("astro_native")


def _stack(n=30, h=24, w=28, c=3, seed=0, outliers=True):
    rng = np.random.default_rng(seed)
    d = rng.normal(1000.0, 30.0, (n, h, w, c)).astype(np.float32)
    if outliers:
        for _ in range(n * h * w // 15):
            d[rng.integers(0, n), rng.integers(0, h), rng.integers(0, w),
              rng.integers(0, c)] += rng.choice([-1, 1]) * rng.uniform(200, 2000)
    return d


@pytest.mark.parametrize("use_mad", [True, False])
@pytest.mark.parametrize("winsorize", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_sigma_clip_matches_numpy(use_mad, winsorize, weighted):
    """Native combine must match the numpy reference within float tolerance."""
    d = _stack(seed=hash((use_mad, winsorize, weighted)) & 0xFFFF)
    w = None
    if weighted:
        w = np.random.default_rng(1).uniform(0.5, 1.5, d.shape[0]).astype(np.float32)

    # numpy reference: pass float64 so _native_usable() returns False (dtype
    # guard) and the pure-numpy tiled path runs. Its per-tile float32 cast makes
    # the inputs identical to what the native kernel sees.
    ref = astro.sigma_clip_combine(
        d.astype(np.float64), sigma=3.0, max_iters=3, weights=w,
        winsorize=winsorize, use_mad=use_mad)
    got = native.sigma_clip_combine(d, 3.0, 3, w, winsorize, use_mad)

    assert got.shape == ref.shape
    assert got.dtype == np.float32
    # background ~1000 ADU; sub-ADU agreement is well within stacking tolerance.
    assert float(np.max(np.abs(ref.astype(np.float64) - got))) < 2.0


def test_median_matches_numpy():
    d = _stack(seed=7)
    ref = astro.median_combine(d.astype(np.float64))
    got = native.median_combine(d)
    assert float(np.max(np.abs(ref.astype(np.float64) - got))) < 1e-3


@pytest.mark.parametrize("weighted", [False, True])
def test_percentile_clip_matches_numpy(weighted):
    d = _stack(seed=11)
    w = (np.random.default_rng(2).uniform(0.5, 1.5, d.shape[0]).astype(np.float32)
         if weighted else None)
    ref = astro.percentile_clip_combine(d.astype(np.float64), low=20.0, high=80.0, weights=w)
    got = native.percentile_clip_combine(d, 20.0, 80.0, w)
    assert float(np.max(np.abs(ref.astype(np.float64) - got))) < 2.0


def test_trimmed_mean_matches_numpy():
    d = _stack(seed=13)
    ref = astro.trimmed_mean_combine(d.astype(np.float64), trim_low=0.2, trim_high=0.2)
    got = native.trimmed_mean_combine(d, 0.2, 0.2)
    assert float(np.max(np.abs(ref.astype(np.float64) - got))) < 1e-3


@pytest.mark.parametrize("n,weighted", [(8, False), (30, False), (30, True)])
def test_esd_matches_numpy(n, weighted):
    d = _stack(n=n, seed=n)
    mo = max(1, n // 4)
    w = (np.random.default_rng(4).uniform(0.5, 1.5, n).astype(np.float32)
         if weighted else None)
    ref = astro.esd_combine(d.astype(np.float64), max_outliers=mo, significance=0.05, weights=w)
    lut = astro._esd_lambda_table(n, mo, 0.05)
    got = native.esd_combine(d, mo, lut, w)
    assert float(np.max(np.abs(ref.astype(np.float64) - got))) < 2.0


def test_warp_preserves_fwhm_and_matches_scipy():
    """Native Lanczos-3 warp must hold star FWHM and agree closely with scipy."""
    from scipy import ndimage
    H = W = 200
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W, 3), 100.0, np.float32)
    for c in range(3):
        img[:, :, c] += 5000.0 * np.exp(-((yy - 100) ** 2 + (xx - 100) ** 2) / (2 * 1.8 ** 2))
    theta = np.deg2rad(0.4)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    off = -R @ np.array([2.3, -1.6])

    sci = np.empty_like(img)
    for c in range(3):
        sci[:, :, c] = ndimage.affine_transform(img[:, :, c], R, offset=off, order=3,
                                                 mode='constant', cval=0.0)
    got = native.warp_affine_lanczos3(img, R.ravel().tolist(), off.tolist(), H, W, 0.0)

    assert got.shape == (H, W, 3) and got.dtype == np.float32
    assert np.isfinite(got).all()

    def fwhm(im):
        b = im[80:120, 80:120, 1].astype(np.float64) - 100.0
        b[b < 0] = 0
        tot = b.sum()
        gy, gx = np.mgrid[0:b.shape[0], 0:b.shape[1]]
        cy, cx = (gy * b).sum() / tot, (gx * b).sum() / tot
        var = ((gy - cy) ** 2 * b).sum() / tot + ((gx - cx) ** 2 * b).sum() / tot
        return 2.3548 * np.sqrt(var / 2)

    # FWHM within 2% of scipy, and the two warps highly correlated.
    assert abs(fwhm(got) - fwhm(sci)) / fwhm(sci) < 0.02
    m = 20
    a = sci[m:-m, m:-m, 1].ravel(); b = got[m:-m, m:-m, 1].ravel()
    assert np.corrcoef(a, b)[0, 1] > 0.999


@pytest.mark.parametrize("option", [1, 2])
def test_anisotropic_diffusion_matches_numpy(option):
    import src.denoising as dn
    rng = np.random.default_rng(option)
    img = np.clip(rng.normal(300, 40, (64, 80, 3)), 0, None).astype(np.float32)
    h = dn._HAS_NATIVE
    dn._HAS_NATIVE = False
    ref = dn.anisotropic_diffusion(img, iterations=12, kappa=30.0, gamma=0.1, option=option)
    dn._HAS_NATIVE = h
    got = dn.anisotropic_diffusion(img, iterations=12, kappa=30.0, gamma=0.1, option=option)
    assert float(np.max(np.abs(ref.astype(np.float64) - got))) < 1e-3


@pytest.mark.parametrize("use_mad", [True, False])
def test_fused_patch_combine_matches_numpy(use_mad):
    """Fused patch-weighted+sigma-clip must match the numpy two-pass path."""
    d = _stack(n=20, h=32, w=40, seed=99)
    rng = np.random.default_rng(3)
    qmaps = [rng.uniform(0.2, 1.0, (32, 40)).astype(np.float32) for _ in range(20)]
    gw = rng.uniform(0.5, 1.5, 20).astype(np.float32)
    _, rej = astro.sigma_clip_combine(d.astype(np.float64), sigma=3.0, max_iters=3,
                                      weights=gw, use_mad=use_mad, return_mask=True)
    ref = astro.patch_weighted_mean_combine(d, qmaps, global_weights=gw, rejection_mask=rej)
    qm = np.ascontiguousarray(np.stack(qmaps), dtype=np.float32)
    got = native.patch_weighted_sigma_combine(d, qm, gw, 3.0, 3, use_mad)
    assert got.shape == ref.shape and got.dtype == np.float32
    assert float(np.max(np.abs(ref.astype(np.float64) - got))) < 1.0


def test_all_nan_pixel_is_zero():
    d = _stack(n=8, h=4, w=4, c=1, outliers=False)
    d[:, 0, 0, 0] = np.nan
    got = native.sigma_clip_combine(d, 3.0, 3, None, False, True)
    assert np.isfinite(got).all()
    assert got[0, 0, 0] == 0.0
