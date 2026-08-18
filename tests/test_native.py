"""Parity tests for the optional native (Rust) kernels.

Skipped entirely when `astro_native` is not built/installed, so the suite
still passes in a pure-Python environment.
"""
import numpy as np
import pytest

import originstack as astro
import src.stacking as _stacking_mod
import src.wavelet as _wavelet_mod
import src.blind_match as _blind_match_mod
import src.denoising as _denoising_mod
import src.debayer as _debayer_mod
import src.robust_pca as _robust_pca_mod
import src.channel_combine as _channel_combine_mod

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


@pytest.mark.parametrize("weighted", [False, True])
def test_linear_fit_clip_matches_numpy(weighted):
    d = _stack(seed=17)
    n = d.shape[0]
    w = (np.random.default_rng(5).uniform(0.5, 1.5, n).astype(np.float32)
         if weighted else None)
    ref = astro.linear_fit_clip_combine(
        d.astype(np.float64), sigma_low=4.0, sigma_high=2.0, max_iters=5, weights=w)
    got = native.linear_fit_clip_combine(d, 4.0, 2.0, 5, w)
    assert got.shape == ref.shape
    assert got.dtype == np.float32
    assert float(np.max(np.abs(ref.astype(np.float64) - got))) < 2.0


def test_linear_fit_clip_rejects_injected_outliers():
    """A single wild sample per pixel (cosmic-ray-like) must not survive into
    the combined result -- the whole point of the algorithm. An unrejected
    +5000 spike in 1 of 20 frames would shift the mean by 5000/20=250 ADU;
    correct rejection keeps the combine within the stack's own sampling
    noise (sigma/sqrt(N) ~ 20/sqrt(19) ~ 4.6) of the true 1000 ADU signal."""
    rng = np.random.default_rng(21)
    n, h, w, c = 20, 16, 16, 3
    clean = rng.normal(1000.0, 20.0, (n, h, w, c)).astype(np.float32)
    spiked = clean.copy()
    spiked[5] += 5000.0  # every pixel in one frame is a huge spike
    got = native.linear_fit_clip_combine(spiked, 4.0, 2.0, 5, None)
    assert float(np.abs(got.mean() - 1000.0)) < 20.0
    assert float(np.max(np.abs(got.astype(np.float64) - 1000.0))) < 60.0


@pytest.mark.parametrize("with_gain,weighted", [(False, False), (True, False), (False, True)])
def test_ivw_matches_numpy(with_gain, weighted):
    d = _stack(seed=23, outliers=False)
    n = d.shape[0]
    rng = np.random.default_rng(6)
    noise = rng.uniform(10.0, 40.0, n).astype(np.float32)
    sky = np.full(n, 1000.0, dtype=np.float32) if with_gain else None
    gain = 2.0 if with_gain else None
    w = rng.uniform(0.5, 1.5, n).astype(np.float32) if weighted else None
    ref = astro.ivw_combine(d.astype(np.float64), noise, sky, gain, w)
    got = native.ivw_combine(d, noise, sky, gain, w)
    assert got.shape == ref.shape
    assert got.dtype == np.float32
    assert float(np.max(np.abs(ref.astype(np.float64) - got))) < 2.0


def test_ivw_downweights_noisier_frames():
    """A frame with much higher noise should contribute less to the combined
    result than an equally-sized low-noise frame -- the core inverse-variance
    property, not just a smoke test that it runs."""
    n, h, w, c = 2, 8, 8, 3
    rng = np.random.default_rng(9)
    quiet = np.full((h, w, c), 1000.0, dtype=np.float32)
    noisy = np.full((h, w, c), 1000.0, dtype=np.float32) + 500.0  # way off
    stack = np.stack([quiet, noisy]).astype(np.float32)
    noise = np.array([5.0, 500.0], dtype=np.float32)  # noisy frame is 100x noisier
    got = native.ivw_combine(stack, noise, None, None, None)
    # inverse-variance weight ratio is 100^2 = 10000:1 in favour of "quiet" --
    # combined result should sit almost exactly at the quiet frame's value.
    assert float(np.abs(got.mean() - 1000.0)) < 1.0


def _numpy_wavedec2(*a, **kw):
    """Force the pure-Python apply_along_axis path (native dispatch happens
    inside _dwt2/_idwt2, so calling wavedec2/waverec2 directly with native
    installed would compare the native kernel against itself)."""
    had = _wavelet_mod._HAS_NATIVE
    _wavelet_mod._HAS_NATIVE = False
    try:
        return _wavelet_mod.wavedec2(*a, **kw)
    finally:
        _wavelet_mod._HAS_NATIVE = had


def _numpy_waverec2(*a, **kw):
    had = _wavelet_mod._HAS_NATIVE
    _wavelet_mod._HAS_NATIVE = False
    try:
        return _wavelet_mod.waverec2(*a, **kw)
    finally:
        _wavelet_mod._HAS_NATIVE = had


@pytest.mark.parametrize("h,w", [(32, 32), (33, 45), (64, 65), (101, 100)])
@pytest.mark.parametrize("level", [1, 2, 3])
def test_wavedec2_bior13_matches_numpy(h, w, level):
    rng = np.random.default_rng(hash((h, w, level)) & 0xFFFF)
    img = rng.uniform(0, 1000, (h, w))
    ref_coeffs = _numpy_wavedec2(img, level, wavelet='bior1.3')
    got_coeffs = _wavelet_mod.wavedec2(img, level, wavelet='bior1.3')
    assert _wavelet_mod._HAS_NATIVE  # sanity: this run actually used native
    for ref, got in zip(ref_coeffs, got_coeffs):
        ref_arrs = ref if isinstance(ref, tuple) else (ref,)
        got_arrs = got if isinstance(got, tuple) else (got,)
        for r, g in zip(ref_arrs, got_arrs):
            np.testing.assert_allclose(r, g, atol=1e-9)


@pytest.mark.parametrize("h,w", [(32, 32), (33, 45), (100, 129)])
@pytest.mark.parametrize("level", [1, 2, 3])
def test_wavedec2_db4_matches_numpy(h, w, level):
    rng = np.random.default_rng(hash((h, w, level, 'db4')) & 0xFFFF)
    img = rng.uniform(0, 1000, (h, w))
    ref_coeffs = _numpy_wavedec2(img, level, wavelet='db4')
    got_coeffs = _wavelet_mod.wavedec2(img, level, wavelet='db4')
    for ref, got in zip(ref_coeffs, got_coeffs):
        ref_arrs = ref if isinstance(ref, tuple) else (ref,)
        got_arrs = got if isinstance(got, tuple) else (got,)
        for r, g in zip(ref_arrs, got_arrs):
            np.testing.assert_allclose(r, g, atol=1e-9)


@pytest.mark.parametrize("h,w", [(32, 32), (33, 45), (64, 65), (101, 100), (200, 150)])
@pytest.mark.parametrize("level", [1, 2, 4])
def test_waverec2_roundtrip_matches_numpy_and_reconstructs(h, w, level):
    rng = np.random.default_rng(hash((h, w, level, 'rt')) & 0xFFFF)
    img = rng.uniform(0, 1000, (h, w))
    coeffs = _wavelet_mod.wavedec2(img, level, wavelet='bior1.3')
    ref_rec = _numpy_waverec2(coeffs)
    got_rec = _wavelet_mod.waverec2(coeffs)
    np.testing.assert_allclose(ref_rec, got_rec, atol=1e-9)
    np.testing.assert_allclose(got_rec[:h, :w], img, atol=1e-6)


_STAR_DT = np.dtype([('xcentroid', np.float64), ('ycentroid', np.float64), ('flux', np.float64)])


def _blind_match_catalog(n, w=3000, h=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros(n, dtype=_STAR_DT)
    out['xcentroid'] = rng.uniform(50, w - 50, n)
    out['ycentroid'] = rng.uniform(50, h - 50, n)
    out['flux'] = rng.uniform(500, 50000, n)
    return out


def _blind_match_rotate(cat, theta_deg, tx, ty, w=3000, h=2000, seed=1):
    rng = np.random.default_rng(seed)
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    cx, cy = w / 2, h / 2
    x = cat['xcentroid'] - cx
    y = cat['ycentroid'] - cy
    out = np.zeros(len(cat), dtype=_STAR_DT)
    out['xcentroid'] = c * x - s * y + cx + tx + rng.normal(0, 0.2, len(cat))
    out['ycentroid'] = s * x + c * y + cy + ty + rng.normal(0, 0.2, len(cat))
    out['flux'] = cat['flux']
    return out


def _numpy_match_rigid(*a, **kw):
    """Force the pure-numpy hypothesis search (native dispatch happens
    inside match_rigid_unknown_rotation, so calling it directly with native
    installed would compare the native kernel against itself)."""
    had = _blind_match_mod._HAS_NATIVE
    _blind_match_mod._HAS_NATIVE = False
    try:
        return _blind_match_mod.match_rigid_unknown_rotation(*a, **kw)
    finally:
        _blind_match_mod._HAS_NATIVE = had


@pytest.mark.parametrize("theta,tx,ty", [
    (5.0, 12.3, -8.7), (37.0, 100.5, -50.2), (91.0, -30.0, 40.0), (-45.0, -20.1, 60.4),
])
def test_blind_match_hypotheses_matches_numpy(theta, tx, ty):
    src_cat = _blind_match_catalog(40, seed=hash((theta, tx, ty)) & 0xFFFF)
    dst_cat = _blind_match_rotate(src_cat, theta, tx, ty, seed=7)
    src = _blind_match_mod._extract_xy(src_cat, 40)
    dst = _blind_match_mod._extract_xy(dst_cat, 40)
    min_sep = max(3.0 * 4.0, 10.0)
    target_inliers = int(np.ceil(0.9 * min(len(src), len(dst))))

    ref_r, ref_t, ref_n = _blind_match_mod._match_hypotheses_numpy(
        src, dst, 3.0, 0.01, min_sep, target_inliers, 20000)
    got_r, got_t, got_n = native.blind_match_hypotheses(
        np.ascontiguousarray(src), np.ascontiguousarray(dst),
        3.0, 0.01, min_sep, target_inliers, 20000)

    assert got_n == ref_n
    np.testing.assert_allclose(got_r, ref_r, atol=1e-9)
    np.testing.assert_allclose(got_t, ref_t, atol=1e-9)


@pytest.mark.parametrize('theta,tx,ty,seed', [
    (5.0, 12.3, -8.7, 0), (37.0, 100.5, -50.2, 1), (91.0, -30.0, 40.0, 2), (178.0, 5.0, 5.0, 3),
])
def test_match_rigid_unknown_rotation_matches_numpy_end_to_end(theta, tx, ty, seed):
    src_cat = _blind_match_catalog(45, seed=seed)
    dst_cat = _blind_match_rotate(src_cat, theta, tx, ty, seed=seed + 100)
    ref = _numpy_match_rigid(src_cat, dst_cat, max_stars=45)
    got = _blind_match_mod.match_rigid_unknown_rotation(src_cat, dst_cat, max_stars=45)
    assert ref is not None and got is not None
    np.testing.assert_allclose(got.params, ref.params, atol=1e-6)


def test_dct2_ortho_matches_scipy():
    scipy_fft = pytest.importorskip("scipy.fft")
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 1000, (8, 8))
    ref = scipy_fft.dctn(x, axes=(0, 1), norm='ortho')
    got = native.dct2_ortho_native(np.ascontiguousarray(x, dtype=np.float64))
    np.testing.assert_allclose(got, ref, atol=1e-9)

    ref_i = scipy_fft.idctn(ref, axes=(0, 1), norm='ortho')
    got_i = native.idct2_ortho_native(np.ascontiguousarray(ref, dtype=np.float64))
    np.testing.assert_allclose(got_i, ref_i, atol=1e-9)
    np.testing.assert_allclose(got_i, x, atol=1e-9)  # roundtrip recovers input exactly


def test_bm3d_denoise_native_matches_numpy():
    scipy_fft = pytest.importorskip("scipy.fft")
    rng = np.random.default_rng(11)
    h, w = 40, 40
    clean = 500.0 + 200.0 * np.sin(np.mgrid[0:h, 0:w][0] / 5.0)
    y = clean + rng.normal(0, 15.0, (h, w))
    sigma_psd = 15.0
    bs, stride, sw, group_size = 8, 4, 16, 8

    ref = _denoising_mod._bm3d_step12_numpy(
        y, bs, stride, sw, group_size, sigma_psd, scipy_fft.dctn, scipy_fft.idctn)
    got = native.bm3d_denoise_native(
        np.ascontiguousarray(y, dtype=np.float64), bs, stride, sw, group_size, sigma_psd)

    np.testing.assert_allclose(got, ref, atol=1e-6)
    # Sanity: denoising actually reduces noise vs the raw noisy input.
    assert np.abs(got - clean).mean() < np.abs(y - clean).mean()


def _numpy_sigma_clipped_median(*a, **kw):
    """Force the pure-numpy iterative sigma-clip path."""
    had = _debayer_mod._HAS_NATIVE
    _debayer_mod._HAS_NATIVE = False
    try:
        return _debayer_mod._sigma_clipped_median(*a, **kw)
    finally:
        _debayer_mod._HAS_NATIVE = had


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_sigma_clipped_median_matches_numpy(seed):
    rng = np.random.default_rng(seed)
    arr = rng.uniform(400.0, 700.0, 2000).astype(np.float32)
    idx = rng.integers(0, 2000, 20)
    arr[idx] += rng.uniform(500.0, 2000.0, 20)
    ref = _numpy_sigma_clipped_median(arr)
    got = _debayer_mod._sigma_clipped_median(arr)
    assert abs(ref - got) < 1e-3


def test_sigma_clipped_median_rejects_outliers():
    rng = np.random.default_rng(5)
    clean = rng.normal(500.0, 5.0, 2000).astype(np.float32)
    spiked = clean.copy()
    spiked[rng.integers(0, 2000, 40)] += 5000.0
    got = _debayer_mod._sigma_clipped_median(spiked)
    assert abs(got - 500.0) < 5.0  # unrejected spikes would shift this far more


def test_fix_hot_bayer_matches_reference():
    """_fix_hot_bayer now routes its per-sub-channel median through the
    native 3x3 kernel (via a contiguous copy of the strided Bayer view) --
    confirm it still matches scipy's ndimage.median_filter on the same data."""
    from scipy import ndimage as _nd
    rng = np.random.default_rng(6)
    data = rng.uniform(400.0, 700.0, (64, 80)).astype(np.float32)
    idx = rng.integers(0, data.size, 30)
    data.ravel()[idx] += rng.uniform(1000.0, 3000.0, 30)

    got = _debayer_mod._fix_hot_bayer(data.copy(), threshold=5.0)

    # Reference: same algorithm, scipy median_filter directly on each
    # strided sub-channel (the pre-fix code path).
    ref = data.astype(np.float32, copy=True)
    for dy in range(2):
        for dx in range(2):
            sub = ref[dy::2, dx::2]
            med = _nd.median_filter(sub, size=3)
            diff = sub - med
            mad = np.median(np.abs(diff))
            sigma = mad * 1.4826
            if sigma < 1e-6:
                continue
            stat_mask = diff > 5.0 * sigma
            sub[stat_mask] = med[stat_mask]

    np.testing.assert_allclose(got, ref, atol=1e-3)


@pytest.mark.parametrize("with_mono", [False, True])
def test_hot_pixel_box_replace_matches_scipy(with_mono):
    from scipy import ndimage as _nd
    rng = np.random.default_rng(7)
    h, w, c = 48, 56, 3
    rgb = rng.uniform(400.0, 700.0, (h, w, c)).astype(np.float32)
    mask = rng.random((h, w)) < 0.08

    ref = rgb.copy()
    for ch in range(c):
        filt = _nd.uniform_filter(rgb[:, :, ch], size=3)
        ref[:, :, ch] = np.where(mask, filt, rgb[:, :, ch])

    got = native.hot_pixel_box_replace_native(
        np.ascontiguousarray(rgb), np.ascontiguousarray(mask, dtype=np.uint8))
    np.testing.assert_allclose(got, ref, atol=1e-3)
    np.testing.assert_allclose(got[~mask], rgb[~mask])  # unmasked pixels untouched


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


def _synthetic_starfield(h=300, w=400, n_stars=25, seed=0):
    """Deterministic synthetic star field: flat sky + Gaussian PSF stars +
    Poisson-like noise, for a controlled (not just real-data) parity check."""
    rng = np.random.default_rng(seed)
    img = rng.normal(1000.0, 15.0, (h, w)).astype(np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(n_stars):
        cy = rng.uniform(20, h - 20)
        cx = rng.uniform(20, w - 20)
        amp = rng.uniform(200, 5000)
        sigma = rng.uniform(1.5, 3.0)
        img += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    return img.astype(np.float32)


def test_detect_stars_matched_filter_matches_numpy_synthetic():
    """Native matched-filter star detector vs the numpy mirror in
    src/star_detect.py on a synthetic field -- same mesh construction,
    convolution, and two-pass centroid refinement, so results should match
    to float64 summation-order tolerance (not just "close")."""
    from src.star_detect import _detect_stars_matched_filter_numpy

    img = _synthetic_starfield()
    got = native.detect_stars_matched_filter(img, 5.5, 22.0, 64, 0.5, 2)
    ref = _detect_stars_matched_filter_numpy(img.astype(np.float64), 5.5, 22.0, 64, 0.5, 2)

    assert got.shape[0] == len(ref)
    assert got.shape[0] > 0  # sanity: the synthetic field should yield detections
    got_sorted = got[np.argsort(got[:, 0])]
    ref_sorted = np.sort(ref, order='xcentroid')
    np.testing.assert_allclose(got_sorted[:, 0], ref_sorted['xcentroid'], rtol=0, atol=1e-5)
    np.testing.assert_allclose(got_sorted[:, 1], ref_sorted['ycentroid'], rtol=0, atol=1e-5)
    np.testing.assert_allclose(got_sorted[:, 2], ref_sorted['flux'], rtol=1e-6, atol=1e-3)


def test_detect_stars_matched_filter_empty_field_no_detections():
    """Pure noise, no stars -- both paths should return zero detections,
    not spurious noise-driven candidates (this exact failure mode -- a
    mesh-interpolation edge artifact producing false positives -- was a
    real bug caught during development, see src/star_detect.py docstring)."""
    from src.star_detect import _detect_stars_matched_filter_numpy

    rng = np.random.default_rng(1)
    img = rng.normal(1000.0, 15.0, (200, 250)).astype(np.float32)
    got = native.detect_stars_matched_filter(img, 5.5, 22.0, 64, 0.5, 2)
    ref = _detect_stars_matched_filter_numpy(img.astype(np.float64), 5.5, 22.0, 64, 0.5, 2)
    assert got.shape[0] == 0
    assert len(ref) == 0


def test_detect_stars_matched_filter_speedup():
    """Native path should be meaningfully faster than the numpy mirror on a
    real-sized field -- not a strict regression gate (timing is
    environment-dependent), just a sanity check that the native path is
    actually doing the heavy lifting."""
    import time

    from src.star_detect import _detect_stars_matched_filter_numpy

    img = _synthetic_starfield(h=800, w=1000, n_stars=80)
    t0 = time.time()
    native.detect_stars_matched_filter(img, 5.5, 22.0, 64, 0.5, 2)
    t_native = time.time() - t0

    t0 = time.time()
    _detect_stars_matched_filter_numpy(img.astype(np.float64), 5.5, 22.0, 64, 0.5, 2)
    t_numpy = time.time() - t0

    assert t_native < t_numpy


def test_fit_rigid_ransac_matches_numpy_mirror_on_shared_seed():
    """Native RANSAC-rigid-transform fit vs the numpy mirror in
    src/affine_fit.py -- same Umeyama closed-form solve, same RANSAC loop
    semantics; for a shared seed both should converge to the same fit
    (verified, not assumed -- see src/affine_fit.py docstring for why
    parity with skimage itself is a different, statistical question)."""
    from src.affine_fit import _ransac_rigid_numpy

    rng = np.random.default_rng(11)
    n_inliers, n_outliers = 35, 12
    theta = np.radians(2.3)
    t = np.array([8.0, -4.5])
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    src_in = rng.uniform(0, 400, (n_inliers, 2))
    dst_in = (R @ src_in.T).T + t + rng.normal(0, 0.1, (n_inliers, 2))
    src_out = rng.uniform(0, 400, (n_outliers, 2))
    dst_out = rng.uniform(0, 400, (n_outliers, 2))
    src = np.vstack([src_in, src_out])
    dst = np.vstack([dst_in, dst_out])

    params_native, inliers_native = native.fit_rigid_ransac(
        np.ascontiguousarray(src), np.ascontiguousarray(dst), 3, 2.0, 1000, 13)
    model_numpy, inliers_numpy = _ransac_rigid_numpy(
        src, dst, min_samples=3, residual_threshold=2.0, max_trials=1000,
        rng=np.random.default_rng(13))

    assert params_native is not None
    assert int(np.sum(inliers_native)) == int(inliers_numpy.sum())
    np.testing.assert_allclose(np.asarray(params_native), model_numpy.params, atol=1e-8)


def test_fit_rigid_ransac_too_few_points_returns_none():
    src = np.array([[0.0, 0.0], [1.0, 1.0]])
    params, inliers = native.fit_rigid_ransac(src, src, 3, 2.0, 100, -1)
    assert params is None and inliers is None


# ---------------------------------------------------------------------------
# Online (streaming) sigma-clip: burn-in seed + per-frame fold kernels.
# These back --stream: a genuine frame-at-a-time stacker (as opposed to
# online_sigma_clip_combine above, which takes the whole (N,H,W,C) array at
# once purely to benchmark algorithm cost).
# ---------------------------------------------------------------------------

def _full_coverage(burn_stack):
    """(K,H,W) all-covered mask matching a (K,H,W,C) burn-in stack."""
    k, h, w = burn_stack.shape[:3]
    return np.ones((k, h, w), dtype=np.float32)


def _numpy_seed_burnin(burn_stack, coverage=None, sigma=3.0):
    """Force the numpy path (HAS_NATIVE off) for online_sigma_clip_seed_burnin."""
    if coverage is None:
        coverage = _full_coverage(burn_stack)
    had = _stacking_mod.HAS_NATIVE
    _stacking_mod.HAS_NATIVE = False
    try:
        return astro.online_sigma_clip_seed_burnin(burn_stack, coverage, sigma=sigma)
    finally:
        _stacking_mod.HAS_NATIVE = had


def _numpy_fold_frame(mean, m2, n_acc, frame, coverage, sigma=3.0):
    """Force the numpy path (HAS_NATIVE off) for online_sigma_clip_fold_frame."""
    had = _stacking_mod.HAS_NATIVE
    _stacking_mod.HAS_NATIVE = False
    try:
        return astro.online_sigma_clip_fold_frame(mean, m2, n_acc, frame, coverage, sigma=sigma)
    finally:
        _stacking_mod.HAS_NATIVE = had


def test_online_sigma_clip_seed_burnin_matches_numpy():
    d = _stack(n=10, seed=21)  # (K,H,W,C) burn-in window, with injected outliers
    cov = _full_coverage(d)
    mean_n, m2_n, nacc_n, rej_n = _numpy_seed_burnin(d, cov, sigma=3.0)
    mean_r, m2_r, nacc_r, rej_r = native.online_sigma_clip_seed_burnin(d, cov, 3.0)

    assert mean_r.shape == mean_n.shape == d.shape[1:]
    assert mean_r.dtype == np.float64
    np.testing.assert_allclose(mean_r, mean_n, atol=1e-6)
    np.testing.assert_allclose(m2_r, m2_n, atol=1e-3)
    np.testing.assert_allclose(nacc_r, nacc_n, atol=1e-9)
    assert rej_r == rej_n


def test_online_sigma_clip_fold_frame_matches_numpy():
    """Seed via burn-in, then fold several frames one at a time; native and
    numpy must agree at EVERY step, not just the final one, to catch
    accumulation-order bugs.

    Both kernels mutate mean/m2/n_acc IN PLACE (returning only the rejected
    count), so each side needs its own contiguous float64 copy of the seeded
    state to mutate across the loop."""
    d = _stack(n=20, seed=22)
    burn, rest = d[:10], d[10:]
    burn_cov = _full_coverage(burn)
    mean_n, m2_n, nacc_n, _ = _numpy_seed_burnin(burn, burn_cov, sigma=3.0)
    mean_r, m2_r, nacc_r, _ = native.online_sigma_clip_seed_burnin(burn, burn_cov, 3.0)
    np.testing.assert_allclose(mean_r, mean_n, atol=1e-6)

    mean_n = np.ascontiguousarray(mean_n, dtype=np.float64)
    m2_n = np.ascontiguousarray(m2_n, dtype=np.float64)
    nacc_n = np.ascontiguousarray(nacc_n, dtype=np.float64)
    mean_r = np.ascontiguousarray(mean_r, dtype=np.float64)
    m2_r = np.ascontiguousarray(m2_r, dtype=np.float64)
    nacc_r = np.ascontiguousarray(nacc_r, dtype=np.float64)

    H, W, C = d.shape[1:]
    coverage = np.ones((H, W), dtype=np.float32)
    for frame in rest:
        rej_n = _numpy_fold_frame(mean_n, m2_n, nacc_n, frame, coverage, sigma=3.0)
        rej_r = native.online_sigma_clip_fold_frame(mean_r, m2_r, nacc_r, frame, coverage, 3.0)
        np.testing.assert_allclose(mean_r, mean_n, atol=1e-6)
        np.testing.assert_allclose(m2_r, m2_n, atol=1e-3)
        np.testing.assert_allclose(nacc_r, nacc_n, atol=1e-9)
        assert rej_r == rej_n


def test_online_sigma_clip_seed_burnin_excludes_uncovered_samples():
    """Each burn-in frame has its own out-of-frame zero-fill region (a real
    concern: this session's actual Rosette Nebula run reported shifts up to
    132px). A pixel covered by only some of the K burn-in frames must seed
    its state from the covered samples alone -- zero-fill must not drag the
    median toward zero."""
    d = _stack(n=10, seed=25, outliers=False)
    H, W, C = d.shape[1:]
    coverage = np.ones((d.shape[0], H, W), dtype=np.float32)
    # Left half uncovered by every OTHER burn-in frame (only frame 0 covers
    # it) -- if zero-fill weren't excluded, the median there would collapse
    # toward 0 instead of the real ~1000 ADU background.
    coverage[1:, :, : W // 2] = 0.0
    d_masked = d.copy()
    d_masked[1:, :, : W // 2, :] = 0.0  # simulate the warp's zero-fill

    mean_r, m2_r, nacc_r, rej_r = native.online_sigma_clip_seed_burnin(d_masked, coverage, 3.0)
    mean_n, m2_n, nacc_n, rej_n = _numpy_seed_burnin(d_masked, coverage, sigma=3.0)

    # Native and numpy must agree...
    np.testing.assert_allclose(mean_r, mean_n, atol=1e-6)
    np.testing.assert_allclose(nacc_r, nacc_n, atol=1e-9)
    assert rej_r == rej_n
    # ...and the left-half mean must reflect the real background (~1000),
    # NOT be dragged toward the 9 zero-filled samples.
    assert mean_r[:, : W // 2, :].mean() > 500.0
    # Every pixel in that region only had 1 valid sample (frame 0) -- n_acc
    # there must be 1, not up to 10.
    np.testing.assert_allclose(nacc_r[:, : W // 2, :], 1.0)


def test_online_sigma_clip_fold_frame_respects_coverage():
    """A frame with a coverage mask False over part of the image must leave
    the running state untouched in the uncovered region.

    fold_frame mutates mean/m2/n_acc in place, so the pre-call state is
    snapshotted first to compare against."""
    d = _stack(n=12, seed=23)
    burn, frame = d[:10], d[10]
    mean, m2, n_acc, _ = native.online_sigma_clip_seed_burnin(burn, _full_coverage(burn), 3.0)
    mean = np.ascontiguousarray(mean, dtype=np.float64)
    m2 = np.ascontiguousarray(m2, dtype=np.float64)
    n_acc = np.ascontiguousarray(n_acc, dtype=np.float64)
    orig_mean, orig_m2, orig_nacc = mean.copy(), m2.copy(), n_acc.copy()

    H, W, C = d.shape[1:]
    coverage = np.ones((H, W), dtype=np.float32)
    coverage[:, : W // 2] = 0.0  # left half not covered by this frame's shift

    n_rej = native.online_sigma_clip_fold_frame(mean, m2, n_acc, frame, coverage, 3.0)

    np.testing.assert_array_equal(mean[:, : W // 2], orig_mean[:, : W // 2])
    np.testing.assert_array_equal(m2[:, : W // 2], orig_m2[:, : W // 2])
    np.testing.assert_array_equal(n_acc[:, : W // 2], orig_nacc[:, : W // 2])
    # Right half (covered) should generally change.
    assert not np.array_equal(mean[:, W // 2:], orig_mean[:, W // 2:])
    assert n_rej <= (H * (W - W // 2) * C)


def test_online_sigma_clip_streaming_matches_whole_array_kernel():
    """The split burn-in+fold kernels, run frame-at-a-time, must reproduce
    the already-validated whole-array online_sigma_clip_combine kernel
    (validated against synthetic ground truth + production batch
    sigma_clip_combine earlier) on the same stack -- a regression guard
    proving the split doesn't silently change the algorithm."""
    d = _stack(n=25, seed=24)
    burn_in = 10

    combined_whole, n_rej_whole, n_tot_whole = native.online_sigma_clip_combine(
        d, sigma=3.0, burn_in=burn_in)

    mean, m2, n_acc, n_rej_split = native.online_sigma_clip_seed_burnin(
        d[:burn_in], _full_coverage(d[:burn_in]), 3.0)
    mean = np.ascontiguousarray(mean, dtype=np.float64)
    m2 = np.ascontiguousarray(m2, dtype=np.float64)
    n_acc = np.ascontiguousarray(n_acc, dtype=np.float64)
    H, W, C = d.shape[1:]
    coverage = np.ones((H, W), dtype=np.float32)
    for frame in d[burn_in:]:
        rej = native.online_sigma_clip_fold_frame(mean, m2, n_acc, frame, coverage, 3.0)
        n_rej_split += rej

    assert n_tot_whole == d.shape[0]
    np.testing.assert_allclose(mean.astype(np.float32), combined_whole, atol=1e-3)
    assert n_rej_split == n_rej_whole


# ---------------------------------------------------------------------------
# Gram-matrix thin-SVD trick: gram_matrix_wide / small_times_wide (robust-PCA
# master calibration). _thin_svd_wide auto-dispatches to these with a bare
# try/except numpy fallback on any native failure -- without a forced
# native-vs-numpy comparison here, a broken kernel silently falls back and
# goes undetected by the rest of the suite (confirmed true before this test
# existed: monkeypatching the native call to raise produced zero failures).
# ---------------------------------------------------------------------------

def _numpy_thin_svd_wide(M):
    had = _robust_pca_mod._HAS_NATIVE
    _robust_pca_mod._HAS_NATIVE = False
    try:
        return _robust_pca_mod._thin_svd_wide(M)
    finally:
        _robust_pca_mod._HAS_NATIVE = had


def test_thin_svd_wide_native_matches_numpy_reconstruction():
    assert _robust_pca_mod._HAS_NATIVE  # sanity: this run actually has native available
    rng = np.random.default_rng(5)
    n, p = 9, 500  # wide (n << p), the real robust-PCA calibration-stack shape
    M = rng.normal(0.0, 5.0, (n, p))

    U_n, s_n, Vt_n = _robust_pca_mod._thin_svd_wide(M)
    recon_native = (U_n * s_n) @ Vt_n

    U_p, s_p, Vt_p = _numpy_thin_svd_wide(M)
    recon_numpy = (U_p * s_p) @ Vt_p

    # Singular vector signs aren't uniquely defined (see _thin_svd_wide's own
    # docstring), so compare the reconstruction and singular values, not U/Vt
    # directly.
    np.testing.assert_allclose(recon_native, recon_numpy, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(recon_native, M, atol=1e-6, rtol=1e-6)  # thin SVD is exact
    np.testing.assert_allclose(np.sort(s_n)[::-1], np.sort(s_p)[::-1], atol=1e-6, rtol=1e-6)


def test_gram_matrix_wide_matches_numpy():
    rng = np.random.default_rng(6)
    M = np.ascontiguousarray(rng.normal(0.0, 3.0, (7, 300)))
    got = np.asarray(native.gram_matrix_wide(M))
    want = M @ M.T
    np.testing.assert_allclose(got, want, atol=1e-6, rtol=1e-6)


def test_gram_matrix_wide_rejects_non_contiguous():
    rng = np.random.default_rng(7)
    M = rng.normal(0.0, 1.0, (300, 7)).T  # transpose view -- not C-contiguous
    assert not M.flags['C_CONTIGUOUS']
    with pytest.raises(ValueError):
        native.gram_matrix_wide(M)


def test_small_times_wide_matches_numpy():
    rng = np.random.default_rng(8)
    n, p = 6, 250
    small = np.ascontiguousarray(rng.normal(0.0, 1.0, (n, n)))
    data = np.ascontiguousarray(rng.normal(0.0, 1.0, (n, p)))
    got = np.asarray(native.small_times_wide(small, data))
    want = small @ data
    np.testing.assert_allclose(got, want, atol=1e-6, rtol=1e-6)


def test_small_times_wide_rejects_shape_mismatch():
    rng = np.random.default_rng(9)
    small = np.ascontiguousarray(rng.normal(0.0, 1.0, (5, 5)))
    data = np.ascontiguousarray(rng.normal(0.0, 1.0, (6, 250)))  # N mismatch
    with pytest.raises(ValueError):
        native.small_times_wide(small, data)


# ---------------------------------------------------------------------------
# continuum_scale_moments: single-pass central moments backing
# optimal_continuum_scale's closed-form skewness-vs-scale polynomial
# (src/channel_combine.py). Two-stage (mean, then central moments) design
# mirrors the numpy fallback exactly -- parity is checked directly against
# that fallback, not re-derived independently.
# ---------------------------------------------------------------------------

def test_continuum_scale_moments_matches_numpy():
    rng = np.random.default_rng(10)
    a = rng.normal(100.0, 20.0, 5000)
    b = rng.normal(50.0, 10.0, 5000)
    got = native.continuum_scale_moments(a, b)
    want = _channel_combine_mod._continuum_scale_moments_numpy(a, b)
    assert got[0] == want[0]  # n
    np.testing.assert_allclose(got[1:], want[1:], atol=1e-6, rtol=1e-6)


def test_continuum_scale_moments_reproduces_scipy_skew_at_several_scales():
    from scipy.stats import skew
    rng = np.random.default_rng(11)
    a = rng.normal(100.0, 20.0, 3000)
    b = rng.normal(50.0, 10.0, 3000)
    moments = native.continuum_scale_moments(a, b)
    scales = np.array([0.0, 0.5, 1.0, 1.7, 2.3, 3.0])
    got = _channel_combine_mod._skewness_from_moments(moments, scales)
    for s, g in zip(scales, got):
        want = skew(a - s * b, bias=False)
        assert abs(g - want) < 1e-6


def _numpy_ivw_combine_with_sigma(data, noise, sky=None, gain=None, weights=None):
    """Force the numpy tiled path for ivw_combine(..., return_sigma=True)."""
    had = _stacking_mod.HAS_NATIVE
    _stacking_mod.HAS_NATIVE = False
    try:
        return astro.ivw_combine(data, noise=noise, sky=sky, gain=gain,
                                 weights=weights, return_sigma=True)
    finally:
        _stacking_mod.HAS_NATIVE = had


def test_ivw_combine_with_sigma_native_matches_numpy():
    d = _stack(n=10, seed=30, outliers=False)
    noise = np.random.default_rng(31).uniform(2.0, 8.0, d.shape[0]).astype(np.float32)
    result_native, sigma_native = astro.ivw_combine(d, noise=noise, return_sigma=True)
    result_numpy, sigma_numpy = _numpy_ivw_combine_with_sigma(d, noise)
    np.testing.assert_allclose(result_native, result_numpy, atol=1e-3)
    np.testing.assert_allclose(sigma_native, sigma_numpy, atol=1e-4)


def test_ivw_combine_with_sigma_native_matches_analytic_wsum():
    n, h, w, c = 4, 12, 12, 1
    noise = np.array([2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    rng = np.random.default_rng(32)
    data = rng.normal(1000.0, 5.0, (n, h, w, c)).astype(np.float32)
    result, wsum = native.ivw_combine_with_sigma(data, noise, None, None, None)
    expected_wsum = sum(1.0 / nn ** 2 for nn in noise)
    np.testing.assert_allclose(np.asarray(wsum), expected_wsum, rtol=1e-5)


def test_continuum_scale_moments_rejects_length_mismatch():
    rng = np.random.default_rng(12)
    a = rng.normal(0.0, 1.0, 100)
    b = rng.normal(0.0, 1.0, 50)
    with pytest.raises(ValueError):
        native.continuum_scale_moments(a, b)
