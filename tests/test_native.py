"""Parity tests for the optional native (Rust) kernels.

Skipped entirely when `astro_native` is not built/installed, so the suite
still passes in a pure-Python environment.
"""
import numpy as np
import pytest

import astro_stack as astro
import src.stacking as _stacking_mod

native = pytest.importorskip("astro_native")


def _numpy_lacosmic(rgb, **kw):
    """Force the true numpy/scipy path. lacosmic_reject auto-dispatches to the
    native kernel when astro_native is installed (as it is here, since this
    whole module only runs when it's importable), so calling it directly
    would compare the native kernel against itself."""
    had = _stacking_mod.HAS_NATIVE
    _stacking_mod.HAS_NATIVE = False
    try:
        return astro.lacosmic_reject(rgb, **kw)
    finally:
        _stacking_mod.HAS_NATIVE = had


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


def test_warp_shift_fast_path():
    """Pure-translation warp (separable fast path): an integer shift must
    reproduce the input exactly (Lanczos at zero fractional offset is a delta),
    and a fractional shift must track scipy order-3 closely."""
    from scipy import ndimage
    rng = np.random.default_rng(5)
    img = np.ascontiguousarray(rng.normal(500, 50, (64, 80, 3)).astype(np.float32))
    ident = [1.0, 0.0, 0.0, 1.0]

    # Integer shift: out[o] = in[o + off] exactly, cval outside.
    got = native.warp_affine_lanczos3(img, ident, [3.0, -2.0], 64, 80, 0.0)
    assert np.array_equal(got[:-3, 2:], img[3:, :78])

    # Fractional shift: compare against scipy.ndimage.shift order-3.
    off = [-1.3, 2.7]  # in[o + off] convention -> scipy shift by -off
    got = native.warp_affine_lanczos3(img, ident, off, 64, 80, 0.0)
    ref = np.empty_like(img)
    for ch in range(3):
        ref[:, :, ch] = ndimage.shift(img[:, :, ch], shift=(1.3, -2.7), order=3,
                                      mode='constant', cval=0.0)
    m = 6
    a = ref[m:-m, m:-m, :].ravel()
    b = got[m:-m, m:-m, :].ravel()
    assert np.corrcoef(a, b)[0, 1] > 0.999


def test_lacosmic_reject_matches_numpy():
    """Native L.A.Cosmic (f32 internally, not f64 — see lib.rs for why) must
    closely match the numpy/scipy f64 reference, including protecting compact
    bright sources. Not exact: f32 occasionally flips a threshold-boundary
    pixel's reject/keep decision (S vs sigclip right at the edge), so the
    bound here is "rare and small", not zero — checked two ways: the total
    number of pixels whose reject/keep decision disagrees must be tiny, and
    every value (including any flipped pixel) must stay within a sane ADU
    bound of the reference (not an arbitrarily wrong replacement)."""
    rng = np.random.default_rng(2)
    H, W = 120, 140
    rgb = np.clip(rng.normal(500, 60, (H, W, 3)), 0, None).astype(np.float32)
    idx = rng.integers(0, [H, W, 3], size=(60, 3))
    rgb[idx[:, 0], idx[:, 1], idx[:, 2]] += rng.uniform(2000, 8000, 60)
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(5):
        cy, cx = rng.uniform(15, H - 15), rng.uniform(15, W - 15)
        g = rng.uniform(3000, 9000) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.2 ** 2))
        for c in range(3):
            rgb[:, :, c] += g
    rgb = np.ascontiguousarray(rgb)

    ref = _numpy_lacosmic(rgb.copy())
    got = native.lacosmic_reject_native(rgb.copy(), 4.5, 5.0, 1.0, 6.5)
    assert got.shape == ref.shape and got.dtype == np.float32

    diff = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    n_pixels = H * W * 3
    n_disagree = int(np.sum(diff > 1.0))
    assert n_disagree < max(5, n_pixels // 10000), (
        f"{n_disagree}/{n_pixels} pixels disagree on reject/keep — too many for "
        f"f32 threshold rounding, suggests a real bug")
    assert float(diff.max()) < 500.0, "a disagreeing pixel is wildly off, not a boundary flip"


def test_lacosmic_reject_non_rgb_passthrough():
    """Non-3-channel input must pass through unchanged (matches the Python
    early-return), not error."""
    img = np.zeros((10, 12, 1), dtype=np.float32)
    img[3, 4, 0] = 7.0
    out = native.lacosmic_reject_native(img, 4.5, 5.0, 1.0, 6.5)
    assert np.array_equal(out, img)


@pytest.mark.parametrize("size", [3, 5, 9, 17])
def test_median_filter_native_matches_scipy(size):
    from scipy import ndimage
    rng = np.random.default_rng(3)
    a = rng.normal(500, 50, (80, 96)).astype(np.float32)
    ref = ndimage.median_filter(a, size=size)
    got = native.median_filter_native(a, size)
    assert float(np.max(np.abs(ref.astype(np.float64) - got.astype(np.float64)))) < 1e-4


def test_all_nan_pixel_is_zero():
    d = _stack(n=8, h=4, w=4, c=1, outliers=False)
    d[:, 0, 0, 0] = np.nan
    got = native.sigma_clip_combine(d, 3.0, 3, None, False, True)
    assert np.isfinite(got).all()
    assert got[0, 0, 0] == 0.0


def test_dbe_fit_surface_matches_numpy():
    """Native DBE robust local-regression fit vs the numpy mirror in
    src/background.py — same accumulators, IRLS schedule, and truncation, so
    they should agree to float64 summation-order tolerance."""
    from src.background import _dbe_fit_surface_numpy
    rng = np.random.default_rng(7)
    pts, vals = [], []
    for gy in np.linspace(0.02, 0.98, 30):
        for gx in np.linspace(0.02, 0.98, 40):
            if 0.6 < gy < 0.8 and gx > 0.7:
                continue  # a sample gap
            pts.append((gy, gx))
            v = 5000.0 + 300.0 * gy + rng.normal(0, 6.0)
            if rng.random() < 0.05:
                v += rng.uniform(200, 900)  # contaminated patches
            vals.append(v)
    coords = np.ascontiguousarray(pts, dtype=np.float64)
    values = np.ascontiguousarray(vals, dtype=np.float64)

    img_h, img_w, gh, gw, sigma = 1000.0, 1400.0, 40, 56, 70.0
    got_s, got_w = native.dbe_fit_surface(coords, values, img_h, img_w,
                                          gh, gw, sigma, 4.685, 3)
    ref_s, ref_w = _dbe_fit_surface_numpy(coords, values, img_h, img_w,
                                          gh, gw, sigma,
                                          tukey_c=4.685, irls_iters=3)
    np.testing.assert_allclose(got_w, ref_w, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(got_s, ref_s, rtol=1e-8, atol=1e-6)


def test_fit_background_surface_numpy_fallback_bounded():
    """The numpy fallback path (HAS_NATIVE forced off) must satisfy the same
    bounded-in-gap behaviour as the native path."""
    import src.background as bg_mod
    rng = np.random.default_rng(0)
    pts = []
    for gy in np.linspace(0.02, 0.98, 25):
        for gx in np.linspace(0.02, 0.98, 25):
            if gx > 0.55 and gy > 0.55:
                continue
            pts.append((gy, gx))
    coords = np.array(pts)
    values = 5000.0 + rng.normal(0, 5.0, len(pts))
    had = bg_mod.HAS_NATIVE
    bg_mod.HAS_NATIVE = False
    try:
        surface = bg_mod._fit_background_surface(
            coords, values, H=256, W=256, outlier_sigma=2.5, max_iter=3,
            patch_size=32, verbose=False)
    finally:
        bg_mod.HAS_NATIVE = had
    gap = surface[220:256, 220:256]
    assert abs(float(np.median(gap)) - 5000.0) < 200.0
    assert float(np.max(np.abs(surface - 5000.0))) < 500.0


def test_dbe_sample_patches_matches_numpy():
    """Native patch sampler vs the pure-Python loop: identical patch
    selection and coordinates; medians within f32 tolerance."""
    import src.background as bg
    rng = np.random.default_rng(5)
    H, W = 480, 640
    yy, xx = np.mgrid[0:H, 0:W]
    channel = (5000.0 + 200.0 * (yy / H) + rng.normal(0, 60, (H, W)))
    for _ in range(15):
        cy, cx = rng.uniform(20, H - 20), rng.uniform(20, W - 20)
        channel += rng.uniform(1000, 8000) * np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.0 ** 2))
    emission = (rng.random((H, W)) < 0.02).astype(np.float32)

    c1, v1 = bg._sample_background_patches(channel, emission, 48, 0.30,
                                           5100.0, 60.0)
    had = bg.HAS_NATIVE
    bg.HAS_NATIVE = False
    try:
        c2, v2 = bg._sample_background_patches(channel, emission, 48, 0.30,
                                               5100.0, 60.0)
    finally:
        bg.HAS_NATIVE = had
    assert len(v1) == len(v2)
    np.testing.assert_allclose(c1, c2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(v1, v2, rtol=0, atol=1e-3)


def test_patch_combine_grid_mode_matches_fullres():
    """Coarse-grid qmap sampling (native kernel + numpy fallback) must match
    the old materialise-full-res-then-crop path."""
    from scipy.ndimage import zoom as _zoom
    d = _stack(n=12, h=40, w=48, seed=21)
    rng = np.random.default_rng(6)
    H_full, W_full, top, left = 56, 64, 9, 10   # crop region 40x48 inside 56x64
    grids = [rng.uniform(0.2, 1.0, (8, 8)).astype(np.float32) for _ in range(12)]
    gw = rng.uniform(0.5, 1.5, 12).astype(np.float32)

    # Old-style reference: upsample each grid to full res, crop, two-pass numpy.
    full = []
    for g in grids:
        m = _zoom(g, (H_full / 8, W_full / 8), order=1)
        # match patch_scores_to_map: exact-shape guard via same zoom mapping
        full.append(np.clip(m, 0.0, 1.0).astype(np.float32)[top:top + 40, left:left + 48])
    _, rej = astro.sigma_clip_combine(d.astype(np.float64), sigma=3.0, max_iters=3,
                                      weights=gw, use_mad=True, return_mask=True)
    ref = astro.patch_weighted_mean_combine(d, full, global_weights=gw,
                                            rejection_mask=rej)

    geom = (float(H_full), float(W_full), float(top), float(left))
    qm = np.ascontiguousarray(np.stack(grids), dtype=np.float32)

    got_native = native.patch_weighted_sigma_combine(d, qm, gw, 3.0, 3, True, geom)
    got_numpy = astro.patch_weighted_mean_combine(d, list(qm), global_weights=gw,
                                                  rejection_mask=rej,
                                                  grid_geom=geom)
    # Weights are smooth [0,1] fields; zoom vs direct bilinear differ at
    # float tolerance, and the combine averages ~1000 ADU pixels.
    assert float(np.max(np.abs(ref.astype(np.float64) - got_numpy))) < 1.0
    assert float(np.max(np.abs(ref.astype(np.float64) - got_native))) < 2.0
