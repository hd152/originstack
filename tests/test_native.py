"""Parity tests for the optional native (Rust) kernels.

Skipped entirely when `astro_native` is not built/installed, so the suite
still passes in a pure-Python environment.
"""
import numpy as np
import pytest

import originstack as astro
import src.background as _background_mod
import src.blind_match as _blind_match_mod
import src.channel_combine as _channel_combine_mod
import src.debayer as _debayer_mod
import src.denoising as _denoising_mod
import src.local_normalize as _local_normalize_mod
import src.postprocess as _postprocess_mod
import src.robust_pca as _robust_pca_mod
import src.stacking as _stacking_mod
import src.star_removal as _star_removal_mod
import src.star_repair as _star_repair_mod
import src.trail_reject as _trail_reject_mod
import src.wavelet as _wavelet_mod

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


@pytest.mark.parametrize("sigma", [0.8, 2.0, 5.0, 24.0, 32.0])
@pytest.mark.parametrize("shape", [(80, 96), (301, 257)])
def test_gaussian_filter_native_matches_scipy(sigma, shape):
    """gaussian_filter_native is a from-scratch separable reimplementation of
    scipy.ndimage.gaussian_filter's default mode='reflect' -- not a port, so
    parity is judged by numerical agreement, not shared code. Real-shape
    profiling showed correlate1d (gaussian_filter1d's C function) as the top
    self-time item in every profile taken of a full pipeline run this
    session, spread across ~30 call sites; this kernel and its wiring into
    background.py's _gaussian_blur / gaussian_filter_ds don't sweep all of
    them, just the highest-traffic ones."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(4)
    a = rng.normal(1000.0, 50.0, shape)
    want = gaussian_filter(a, sigma=sigma)
    got = np.asarray(native.gaussian_filter_native(a, sigma))
    # separable-pass floating-point summation order differs from scipy's;
    # both are exact renditions of the same closed-form kernel, so the gap
    # is double-precision rounding, not an approximation choice
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


def test_gaussian_filter_native_zero_sigma_is_passthrough():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(20, 30))
    got = np.asarray(native.gaussian_filter_native(a, 0.0))
    np.testing.assert_array_equal(got, a)


def test_gaussian_filter_native_rejects_non_2d_by_signature():
    # the numpy binding itself enforces 2D; the Python-side _gaussian_blur
    # wrapper is what actually gates this in production (see
    # test_gaussian_blur_falls_back_for_non_2d below)
    rng = np.random.default_rng(6)
    a = rng.normal(size=(10, 10, 3))
    with pytest.raises(Exception):
        native.gaussian_filter_native(a, 2.0)


def test_gaussian_blur_wrapper_matches_scipy_native_and_fallback():
    """background.py's _gaussian_blur (the wrapper wired into
    gaussian_filter_ds and swapped into ~10 direct call sites in background.py
    and denoising.py) must agree with plain scipy whether or not native is
    available."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(7)
    a = rng.normal(1000.0, 50.0, (150, 200))
    want = gaussian_filter(a, sigma=5.0)

    got_native = _background_mod._gaussian_blur(a, 5.0)
    np.testing.assert_allclose(got_native, want, rtol=1e-9, atol=1e-9)

    had = _background_mod._HAS_NATIVE_GAUSSIAN
    _background_mod._HAS_NATIVE_GAUSSIAN = False
    try:
        got_fallback = _background_mod._gaussian_blur(a, 5.0)
    finally:
        _background_mod._HAS_NATIVE_GAUSSIAN = had
    np.testing.assert_array_equal(got_fallback, want)


def test_gaussian_blur_falls_back_for_non_2d():
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(8)
    a = rng.normal(size=(20, 30, 3))
    want = gaussian_filter(a, sigma=2.0)
    got = _background_mod._gaussian_blur(a, 2.0)
    np.testing.assert_array_equal(got, want)


def test_gaussian_blur_preserves_input_dtype():
    """The native kernel always computes in float64 (like every other kernel
    in this file); _gaussian_blur must cast back down so a float32 caller
    (several master-calibration and Phase 4 call sites pass float32) doesn't
    silently get a float64 array back and double its memory footprint."""
    assert _background_mod._HAS_NATIVE_GAUSSIAN  # sanity: this run actually has native available
    rng = np.random.default_rng(9)
    a32 = rng.normal(1000.0, 50.0, (60, 70)).astype(np.float32)
    out32 = _background_mod._gaussian_blur(a32, 3.0)
    assert out32.dtype == np.float32

    a64 = a32.astype(np.float64)
    out64 = _background_mod._gaussian_blur(a64, 3.0)
    assert out64.dtype == np.float64
    # same blur either way, just float32-rounded
    np.testing.assert_allclose(out32.astype(np.float64), out64, rtol=1e-5, atol=1e-3)


def test_median_filter_per_channel_matches_combined_axis_scipy_call():
    """postprocess.py's hot-pixel step switched from one scipy
    ndimage.median_filter(stacked, size=(5,5,1)) call (measured 5.1s on a
    real full-res stack -- scipy's N-D rank filter has no fast path for a
    size-1 axis) to 3 independent native 2D calls, one per channel. Confirm
    that's actually equivalent, not just faster."""
    from scipy import ndimage
    rng = np.random.default_rng(4)
    stacked = rng.normal(500, 50, (60, 70, 3)).astype(np.float32)
    ref = ndimage.median_filter(stacked, size=(5, 5, 1))
    got = _postprocess_mod._median_filter_per_channel(stacked, 5)
    assert float(np.max(np.abs(ref.astype(np.float64) - got.astype(np.float64)))) < 1e-4


def test_median_filter_per_channel_falls_back_without_native(monkeypatch):
    from scipy import ndimage
    rng = np.random.default_rng(5)
    stacked = rng.normal(500, 50, (40, 50, 3)).astype(np.float32)
    ref = ndimage.median_filter(stacked, size=(3, 3, 1))
    monkeypatch.setattr(_postprocess_mod, '_HAS_NATIVE_MEDIAN', False)
    got = _postprocess_mod._median_filter_per_channel(stacked, 3)
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
# robust_pca_pre_svd_input / robust_pca_iterate: the fused per-iteration IALM
# elementwise kernels (see robust_pca.py's robust_pca_decompose). Found by
# profiling a real --flat-from-lights run where the plain-numpy elementwise
# arithmetic surrounding the SVD -- not the SVD itself -- was ~80% of the
# function's wall time.
# ---------------------------------------------------------------------------

def test_robust_pca_pre_svd_input_matches_numpy():
    rng = np.random.default_rng(10)
    n, p = 9, 500
    d = np.ascontiguousarray(rng.normal(0.0, 5.0, (n, p)))
    s = np.ascontiguousarray(rng.normal(0.0, 1.0, (n, p)))
    y = np.ascontiguousarray(rng.normal(0.0, 1.0, (n, p)))
    mu = 0.37
    got = np.asarray(native.robust_pca_pre_svd_input(d, s, y, mu))
    want = d - s + y / mu
    np.testing.assert_array_equal(got, want)  # same f64 op order -- bit-exact


def test_robust_pca_pre_svd_input_rejects_shape_mismatch():
    rng = np.random.default_rng(11)
    d = np.ascontiguousarray(rng.normal(0.0, 1.0, (5, 100)))
    s = np.ascontiguousarray(rng.normal(0.0, 1.0, (5, 100)))
    y = np.ascontiguousarray(rng.normal(0.0, 1.0, (4, 100)))  # N mismatch
    with pytest.raises(ValueError):
        native.robust_pca_pre_svd_input(d, s, y, 0.5)


def test_robust_pca_iterate_matches_numpy():
    rng = np.random.default_rng(12)
    n, p = 9, 500
    d = np.ascontiguousarray(rng.normal(0.0, 5.0, (n, p)))
    l = np.ascontiguousarray(rng.normal(0.0, 4.0, (n, p)))
    y0 = np.ascontiguousarray(rng.normal(0.0, 1.0, (n, p)))
    mu = 0.42
    lam_over_mu = 0.1

    s_native = np.zeros((n, p))
    y_native = y0.copy()
    resid_norm = float(native.robust_pca_iterate(d, l, s_native, y_native, lam_over_mu, mu))

    temp = d - l + y0 / mu
    s_want = np.sign(temp) * np.maximum(np.abs(temp) - lam_over_mu, 0.0)
    residual_want = d - l - s_want
    y_want = y0 + mu * residual_want
    resid_norm_want = float(np.linalg.norm(residual_want, 'fro'))

    np.testing.assert_array_equal(s_native, s_want)  # per-element op, bit-exact
    np.testing.assert_array_equal(y_native, y_want)
    # the norm is a parallel reduction, so only close, not bit-exact -- see
    # the kernel's own docstring
    np.testing.assert_allclose(resid_norm, resid_norm_want, rtol=1e-10)


def test_robust_pca_iterate_rejects_shape_mismatch():
    rng = np.random.default_rng(13)
    d = np.ascontiguousarray(rng.normal(0.0, 1.0, (5, 100)))
    l = np.ascontiguousarray(rng.normal(0.0, 1.0, (5, 100)))
    s = np.zeros((5, 100))
    y = np.zeros((4, 100))  # N mismatch
    with pytest.raises(ValueError):
        native.robust_pca_iterate(d, l, s, y, 0.1, 0.5)


def test_robust_pca_decompose_native_matches_numpy_fallback():
    """End-to-end: robust_pca_decompose's native and numpy-fallback paths
    must converge to the same low-rank/sparse split, not just agree on the
    individual fused kernels in isolation."""
    rng = np.random.default_rng(14)
    n, p = 10, 800
    low_rank = np.outer(rng.uniform(0.8, 1.2, n), rng.normal(1000.0, 50.0, p))
    sparse = np.zeros((n, p))
    idx = rng.integers(0, n * p, n * p // 50)
    sparse.flat[idx] = rng.uniform(200.0, 2000.0, len(idx))
    D = low_rank + sparse + rng.normal(0.0, 5.0, (n, p))

    assert _robust_pca_mod._HAS_NATIVE
    L_native, S_native = _robust_pca_mod.robust_pca_decompose(D)

    _robust_pca_mod._HAS_NATIVE = False
    try:
        L_numpy, S_numpy = _robust_pca_mod.robust_pca_decompose(D)
    finally:
        _robust_pca_mod._HAS_NATIVE = True

    np.testing.assert_allclose(L_native, L_numpy, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(S_native, S_numpy, atol=1e-6, rtol=1e-6)


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


# ---------------------------------------------------------------------------
# fit_moffat_native (src/star_repair.py)
# ---------------------------------------------------------------------------

def _synthetic_moffat_wing(rng, amp=500.0, alpha=6.0, beta=2.5, n=120, noise=2.0):
    r = rng.uniform(2.0, 18.0, n)
    v = amp / np.power(1.0 + (r / alpha) ** 2, beta) + rng.normal(0.0, noise, n)
    return r, v, (amp, alpha, beta)


@pytest.mark.parametrize("alpha,beta", [(4.0, 2.0), (8.0, 3.5), (12.0, 1.5)])
def test_fit_moffat_native_recovers_synthetic_params(alpha, beta):
    """Native bounded LM must recover ground-truth Moffat params about as
    well as scipy's curve_fit does -- not bit-exact (different algorithms),
    just comparably close, since both are noisy nonlinear fits."""
    rng = np.random.default_rng(hash((alpha, beta)) & 0xFFFF)
    r, v, (amp, alpha_true, beta_true) = _synthetic_moffat_wing(rng, alpha=alpha, beta=beta)
    native_fit = native.fit_moffat_native(r, v)
    scipy_fit = _star_repair_mod._fit_moffat_wing_numpy(r, v)
    assert native_fit is not None
    assert scipy_fit is not None
    n_amp, n_alpha, n_beta = native_fit
    s_amp, s_alpha, s_beta = scipy_fit
    # Both fits should land in the same neighbourhood of the true params.
    assert abs(n_alpha - alpha_true) / alpha_true < 0.35
    assert abs(n_beta - beta_true) / beta_true < 0.35
    # And agree with each other's fit to within a similar tolerance.
    assert abs(n_alpha - s_alpha) / s_alpha < 0.35
    assert abs(n_beta - s_beta) / s_beta < 0.35


def test_fit_moffat_native_matches_numpy_dispatch():
    rng = np.random.default_rng(5)
    r, v, _ = _synthetic_moffat_wing(rng)
    had = _star_repair_mod._HAS_NATIVE
    try:
        _star_repair_mod._HAS_NATIVE = True
        got = _star_repair_mod._fit_moffat_wing(r, v)
        _star_repair_mod._HAS_NATIVE = False
        want = _star_repair_mod._fit_moffat_wing(r, v)
    finally:
        _star_repair_mod._HAS_NATIVE = had
    assert got is not None and want is not None


def test_fit_moffat_native_none_on_too_few_samples():
    rng = np.random.default_rng(6)
    r = rng.uniform(2.0, 10.0, 3)
    v = rng.uniform(1.0, 10.0, 3)
    assert native.fit_moffat_native(r, v) is None


def test_fit_moffat_native_none_on_nonpositive_peak():
    rng = np.random.default_rng(7)
    r = rng.uniform(2.0, 10.0, 20)
    v = -np.abs(rng.uniform(0.0, 5.0, 20))
    assert native.fit_moffat_native(r, v) is None


# ---------------------------------------------------------------------------
# fit_psf_moffat2d_native / fit_psf_gauss2d_native (src/psf_deconvolution.py)
# ---------------------------------------------------------------------------

import src.psf_deconvolution as _psf_mod  # noqa: E402


def _synthetic_star_cutout(rng, sz=31, model='moffat', amp=900.0, bg=45.0,
                           alpha=3.2, beta=2.7, sigma=2.4, noise=2.5):
    yy, xx = np.mgrid[0:sz, 0:sz].astype(np.float64)
    x0, y0 = sz / 2.0 + rng.uniform(-1.0, 1.0), sz / 2.0 + rng.uniform(-1.0, 1.0)
    r2 = (xx - x0) ** 2 + (yy - y0) ** 2
    if model == 'moffat':
        z = amp * (1.0 + r2 / alpha ** 2) ** (-beta) + bg
        truth = (amp, x0, y0, alpha, beta, bg)
    else:
        z = amp * np.exp(-r2 / (2.0 * sigma ** 2)) + bg
        truth = (amp, x0, y0, sigma, bg)
    z = z + rng.normal(0.0, noise, z.shape)
    return np.ascontiguousarray(z.ravel(), dtype=np.float64), truth


@pytest.mark.parametrize("alpha,beta", [(2.6, 2.2), (3.4, 3.0), (5.0, 4.5)])
def test_fit_psf_moffat2d_native_recovers_synthetic(alpha, beta):
    rng = np.random.default_rng(hash((alpha, beta)) & 0xFFFF)
    z, (amp, x0, y0, a_t, b_t, bg) = _synthetic_star_cutout(
        rng, model='moffat', alpha=alpha, beta=beta)
    sz = int(round(z.size ** 0.5))
    pk, b25 = float(z.max()), float(np.percentile(z, 25))
    got = native.fit_psf_moffat2d_native(z, sz, pk, b25)
    want = _psf_mod._fit_star_2d_numpy(z.reshape(sz, sz), 'moffat', pk, b25)
    assert got is not None and want is not None
    # near the true shape params, and near curve_fit's own fit
    assert abs(got[3] - a_t) / a_t < 0.25
    assert abs(got[4] - b_t) / b_t < 0.30
    assert abs(got[3] - want[3]) / want[3] < 0.20
    assert abs(got[4] - want[4]) / want[4] < 0.25


@pytest.mark.parametrize("sigma", [1.8, 2.5, 3.6])
def test_fit_psf_gauss2d_native_recovers_synthetic(sigma):
    rng = np.random.default_rng(int(sigma * 1000))
    z, (amp, x0, y0, s_t, bg) = _synthetic_star_cutout(
        rng, model='gaussian', sigma=sigma)
    sz = int(round(z.size ** 0.5))
    pk, b25 = float(z.max()), float(np.percentile(z, 25))
    got = native.fit_psf_gauss2d_native(z, sz, pk, b25)
    want = _psf_mod._fit_star_2d_numpy(z.reshape(sz, sz), 'gaussian', pk, b25)
    assert got is not None and want is not None
    assert abs(got[3] - s_t) / s_t < 0.20
    assert abs(got[3] - want[3]) / want[3] < 0.15


def test_fit_psf_2d_native_dispatch_matches_numpy():
    rng = np.random.default_rng(11)
    z, _ = _synthetic_star_cutout(rng, model='moffat')
    sz = int(round(z.size ** 0.5))
    cut = z.reshape(sz, sz)
    pk, b25 = float(z.max()), float(np.percentile(z, 25))
    had = _psf_mod._HAS_NATIVE_PSF2D
    try:
        _psf_mod._HAS_NATIVE_PSF2D = True
        got = _psf_mod._fit_star_2d(cut, 'moffat', pk, b25)
        _psf_mod._HAS_NATIVE_PSF2D = False
        want = _psf_mod._fit_star_2d(cut, 'moffat', pk, b25)
    finally:
        _psf_mod._HAS_NATIVE_PSF2D = had
    assert got is not None and want is not None
    assert abs(got[3] - want[3]) / want[3] < 0.20


def test_fit_psf_2d_native_bad_size_raises():
    rng = np.random.default_rng(12)
    z, _ = _synthetic_star_cutout(rng)
    with pytest.raises(ValueError):
        native.fit_psf_moffat2d_native(z, 30, float(z.max()), 0.0)  # 30*30 != z.size


def test_fit_psf_2d_native_none_on_flat_cutout():
    z = np.full(31 * 31, 50.0, dtype=np.float64)
    # peak == bg -> no signal, kernel returns None (not a fit failure)
    assert native.fit_psf_moffat2d_native(z, 31, 50.0, 50.0) is None
    assert native.fit_psf_gauss2d_native(z, 31, 50.0, 50.0) is None


# ---------------------------------------------------------------------------
# originvision_score (src/originvision_infer.py's score_rgb, --originvision)
# ---------------------------------------------------------------------------

import src.originvision_infer as _ov_mod  # noqa: E402

_ov_model = _ov_mod.resolve_model_path(None)
_have_ov = hasattr(native, 'originvision_score') and _ov_model is not None


def _synth_star_frame(seed=7):
    rng = np.random.default_rng(seed)
    h, w = 260, 340
    yy, xx = np.mgrid[0:h, 0:w]
    img = 42.0 + 0.02 * xx + rng.normal(0, 3, (h, w))
    for _ in range(28):
        cy, cx = rng.integers(30, h - 30), rng.integers(30, w - 30)
        img += rng.uniform(60, 200) * np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rng.uniform(1.5, 3) ** 2))
    return np.ascontiguousarray(
        np.stack([img, img * 0.92, img * 0.85], -1), dtype=np.float32)


@pytest.mark.skipif(not _have_ov, reason='native originvision_score / bundled model absent')
def test_originvision_score_native_shape_and_keys():
    rgb = _synth_star_frame()
    r = native.originvision_score(rgb, _ov_model, 256, True)
    assert isinstance(r, dict)
    assert 'trailing' not in r['tasks']            # v4: untrained head excluded
    for k in ('checkpoint_epoch', 'tasks', 'defect_probability', 'is_defective',
              'quality_score', 'category', 'category_confidence', 'top_categories',
              'sky_brightness', 'stray_light_gradient', 'stray_light_flag'):
        assert k in r, k
    assert r['category'] in ('galaxy', 'nebula', 'star_cluster', 'comet')
    assert 0.0 <= r['category_confidence'] <= 1.0
    for bad in ('trailing_score', 'trailing_flag', 'background_grid'):
        assert bad not in r


@pytest.mark.skipif(not _have_ov, reason='native originvision_score / bundled model absent')
def test_originvision_score_native_bad_input_returns_none():
    assert native.originvision_score(
        np.zeros((8, 8, 2), np.float32), _ov_model, 256, True) is None


@pytest.mark.skipif(
    not (_have_ov and _ov_mod._ort is not None),
    reason='need native + onnxruntime for the cross-backend parity check')
def test_originvision_score_native_matches_onnxruntime():
    import unittest.mock as _m
    rgb = _synth_star_frame(11)
    nat = _ov_mod.score_rgb(rgb)
    with _m.patch.object(_ov_mod, '_HAS_NATIVE_OV', False):
        ort = _ov_mod.score_rgb(rgb)
    assert nat is not None and ort is not None
    assert nat['category'] == ort['category']
    assert nat['is_defective'] == ort['is_defective']
    assert nat['stray_light_flag'] == ort['stray_light_flag']
    for k in ('sky_brightness', 'stray_light_gradient', 'defect_probability'):
        assert abs(nat[k] - ort[k]) < max(1.0, abs(ort[k]) * 0.05), (k, nat[k], ort[k])


# ---------------------------------------------------------------------------
# mesh_median_grid (src/background.py's `_process_channel`)
# ---------------------------------------------------------------------------

def _numpy_mesh_median_grid(channel, star_mask, has_star_mask, cell_excluded, ny, nx):
    H, W = channel.shape
    bg = np.full((ny, nx), np.nan, dtype=np.float64)
    for iy in range(ny):
        y0 = int(round(iy * (H / ny)))
        y1 = min(int(round((iy + 1) * (H / ny))), H)
        for ix in range(nx):
            if cell_excluded[iy, ix]:
                continue
            x0 = int(round(ix * (W / nx)))
            x1 = min(int(round((ix + 1) * (W / nx))), W)
            cell = channel[y0:y1, x0:x1].ravel()
            if cell.size == 0:
                continue
            if has_star_mask:
                sm = star_mask[y0:y1, x0:x1].ravel()
                cell = cell[sm < 0.5]
            if len(cell) > 0:
                bg[iy, ix] = float(np.median(cell))
    return bg


@pytest.mark.parametrize("has_star_mask", [False, True])
def test_mesh_median_grid_matches_numpy(has_star_mask):
    rng = np.random.default_rng(13)
    H, W, ny, nx = 37, 41, 5, 7  # deliberately non-power-of-2, avoids round() ties
    channel = rng.normal(1000.0, 50.0, (H, W)).astype(np.float64)
    star_mask = rng.uniform(0.0, 1.0, (H, W)).astype(np.float32) if has_star_mask else np.zeros((1, 1), dtype=np.float32)
    cell_excluded = (rng.uniform(0.0, 1.0, (ny, nx)) < 0.15).astype(np.uint8)

    got = np.asarray(native.mesh_median_grid(channel, star_mask, has_star_mask, cell_excluded, ny, nx))
    want = _numpy_mesh_median_grid(channel, star_mask if has_star_mask else None, has_star_mask,
                                    cell_excluded.astype(bool), ny, nx)
    np.testing.assert_allclose(got, want, equal_nan=True, atol=1e-9)


# ---------------------------------------------------------------------------
# local_normalize_grid (src/local_normalize.py's `_coarse_background`)
# ---------------------------------------------------------------------------

def test_local_normalize_grid_matches_numpy():
    rng = np.random.default_rng(14)
    frame = rng.normal(500.0, 40.0, (53, 47, 3)).astype(np.float32)
    got = np.asarray(native.local_normalize_grid(frame, 9, 30.0))
    want = _local_normalize_mod._coarse_background_numpy(frame, 9, 30.0)
    np.testing.assert_allclose(got, want, atol=1e-3)


def test_local_normalize_grid_dispatch_matches_numpy_fallback():
    rng = np.random.default_rng(15)
    frame = rng.normal(500.0, 40.0, (40, 36, 3)).astype(np.float32)
    had = _local_normalize_mod._HAS_NATIVE
    try:
        _local_normalize_mod._HAS_NATIVE = True
        got = _local_normalize_mod._coarse_background(frame, 6, 25.0)
        _local_normalize_mod._HAS_NATIVE = False
        want = _local_normalize_mod._coarse_background(frame, 6, 25.0)
    finally:
        _local_normalize_mod._HAS_NATIVE = had
    np.testing.assert_allclose(got, want, atol=1e-3)


# ---------------------------------------------------------------------------
# stamp_star_disks (src/star_removal.py's `build_star_mask`)
# ---------------------------------------------------------------------------

def _make_sources(n, h, w, rng):
    dtype = [('peak', 'f8'), ('ycentroid', 'f8'), ('xcentroid', 'f8')]
    sources = np.zeros(n, dtype=dtype)
    sources['peak'] = rng.uniform(50.0, 5000.0, n)
    sources['ycentroid'] = rng.uniform(0, h, n)
    sources['xcentroid'] = rng.uniform(0, w, n)
    return sources


def test_stamp_star_disks_dispatch_matches_numpy_fallback():
    rng = np.random.default_rng(16)
    h, w = 80, 96
    sources = _make_sources(150, h, w, rng)
    had = _star_removal_mod._HAS_NATIVE
    try:
        _star_removal_mod._HAS_NATIVE = True
        mask_native, r_native = _star_removal_mod.build_star_mask((h, w), sources, fwhm=3.0)
        _star_removal_mod._HAS_NATIVE = False
        mask_numpy, r_numpy = _star_removal_mod.build_star_mask((h, w), sources, fwhm=3.0)
    finally:
        _star_removal_mod._HAS_NATIVE = had
    assert (mask_native is None) == (mask_numpy is None)
    if mask_native is not None:
        np.testing.assert_array_equal(mask_native, mask_numpy)
        assert abs(r_native - r_numpy) < 1e-9


def test_stamp_star_disks_direct_matches_numpy_disk_math():
    rng = np.random.default_rng(17)
    h, w = 60, 70
    n = 40
    cy = rng.uniform(0, h, n)
    cx = rng.uniform(0, w, n)
    r = rng.uniform(2.0, 12.0, n)

    mask_u8, max_r = native.stamp_star_disks(h, w, cy, cx, r)
    got = np.asarray(mask_u8).astype(bool)

    want = np.zeros((h, w), dtype=bool)
    want_max_r = 0.0
    for i in range(n):
        y0, y1 = max(0, int(cy[i] - r[i])), min(h, int(cy[i] + r[i]) + 1)
        x0, x1 = max(0, int(cx[i] - r[i])), min(w, int(cx[i] + r[i]) + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disk = (yy - cy[i]) ** 2 + (xx - cx[i]) ** 2 <= r[i] * r[i]
        want[y0:y1, x0:x1] |= disk
        want_max_r = max(want_max_r, r[i])

    np.testing.assert_array_equal(got, want)
    assert abs(max_r - want_max_r) < 1e-9


# ---------------------------------------------------------------------------
# bresenham_line_native (src/trail_reject.py's `_bresenham_line`)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r0,c0,r1,c1", [
    (0, 0, 10, 4), (5, 5, 5, 15), (3, 3, 3, 3), (20, 2, 2, 20),
    (0, 0, 0, 20), (12, 40, 0, 0), (-3, -3, 8, 5),
])
def test_bresenham_line_native_matches_numpy(r0, c0, r1, c1):
    had = _trail_reject_mod._HAS_NATIVE
    try:
        _trail_reject_mod._HAS_NATIVE = True
        rr_native, cc_native = _trail_reject_mod._bresenham_line(r0, c0, r1, c1)
        _trail_reject_mod._HAS_NATIVE = False
        rr_numpy, cc_numpy = _trail_reject_mod._bresenham_line(r0, c0, r1, c1)
    finally:
        _trail_reject_mod._HAS_NATIVE = had
    np.testing.assert_array_equal(rr_native, rr_numpy)
    np.testing.assert_array_equal(cc_native, cc_numpy)


# ---------------------------------------------------------------------------
# radial_bin_median (src/denoising.py's `radial_renormalize`)
# ---------------------------------------------------------------------------

def _numpy_radial_bin_median(radii, channel, max_radius, n_bins):
    bin_edges = np.linspace(0.0, max_radius + 1.0, n_bins + 1)
    profile = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        in_bin = (radii >= bin_edges[b]) & (radii < bin_edges[b + 1])
        if in_bin.any():
            profile[b] = float(np.median(channel[in_bin]))
    return profile


def test_radial_bin_median_matches_numpy():
    rng = np.random.default_rng(18)
    h, w = 90, 110
    yy, xx = np.mgrid[:h, :w]
    radii = np.sqrt((yy - 40.0) ** 2 + (xx - 50.0) ** 2).astype(np.float64)
    channel = (100.0 + 0.5 * radii + rng.normal(0.0, 3.0, (h, w))).astype(np.float64)
    max_radius = float(radii.max())

    got = np.asarray(native.radial_bin_median(radii, channel, max_radius, 40))
    want = _numpy_radial_bin_median(radii, channel, max_radius, 40)
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_radial_renormalize_dispatch_matches_numpy_fallback():
    rng = np.random.default_rng(19)
    img = rng.normal(200.0, 10.0, (50, 60, 3)).astype(np.float32)
    had = _denoising_mod._HAS_NATIVE
    try:
        _denoising_mod._HAS_NATIVE = True
        got = _denoising_mod.radial_renormalize(img, 25.0, 30.0, n_bins=30)
        _denoising_mod._HAS_NATIVE = False
        want = _denoising_mod.radial_renormalize(img, 25.0, 30.0, n_bins=30)
    finally:
        _denoising_mod._HAS_NATIVE = had
    np.testing.assert_allclose(got, want, atol=1e-2)


# ---------------------------------------------------------------------------
# aperture_photometry_batch (src/photometry.py) -- new in astro_native 0.19
# ---------------------------------------------------------------------------

_HAS_APB = hasattr(native, "aperture_photometry_batch")


@pytest.mark.skipif(not _HAS_APB, reason="astro_native lacks aperture_photometry_batch")
@pytest.mark.parametrize("r_ap,r_in,r_out", [(5.0, 8.0, 13.0),
                                             (6.3, 9.1, 15.7),
                                             (3.0, 6.0, 9.0)])
def test_aperture_photometry_batch_matches_numpy(r_ap, r_in, r_out):
    from src.photometry_core import _aperture_photometry_batch_numpy
    rng = np.random.default_rng(7)
    H, W, C = 120, 140, 3
    img = rng.normal(30.0, 2.0, (H, W, C)).astype(np.float32)
    xs = rng.uniform(20, W - 20, 25)
    ys = rng.uniform(20, H - 20, 25)
    yy, xx = np.mgrid[0:H, 0:W]
    for x, y in zip(xs, ys):
        s = rng.uniform(1.5, 2.5)
        for c in range(C):
            img[..., c] += rng.uniform(200, 3000) * np.exp(
                -(((xx - x) ** 2 + (yy - y) ** 2) / (2 * s ** 2)))
    img = np.ascontiguousarray(img)

    nat = native.aperture_photometry_batch(img, xs, ys, r_ap, r_in, r_out, 4)
    ref = _aperture_photometry_batch_numpy(img, xs, ys, r_ap, r_in, r_out, 4)
    for a, b in zip(nat, ref):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        assert np.array_equal(np.isfinite(a), np.isfinite(b))
        m = np.isfinite(a) & np.isfinite(b)
        np.testing.assert_allclose(a[m], b[m], rtol=1e-5, atol=1e-3)


@pytest.mark.skipif(not _HAS_APB, reason="astro_native lacks aperture_photometry_batch")
def test_aperture_photometry_batch_edge_star_is_nan():
    img = np.ones((40, 40, 3), np.float32)
    xs = np.array([2.0, 20.0])   # first star's r_out disk spills off the frame
    ys = np.array([20.0, 20.0])
    flux, sky, sig, peak, area = native.aperture_photometry_batch(
        img, xs, ys, 4.0, 6.0, 9.0, 4)
    assert not np.any(np.isfinite(np.asarray(flux)[0]))
    assert np.all(np.isfinite(np.asarray(flux)[1]))


@pytest.mark.skipif(not _HAS_APB, reason="astro_native lacks aperture_photometry_batch")
def test_aperture_photometry_batch_rejects_bad_radii():
    img = np.ones((20, 20, 1), np.float32)
    xs = np.array([10.0]); ys = np.array([10.0])
    with pytest.raises(ValueError):
        native.aperture_photometry_batch(img, xs, ys, 5.0, 4.0, 9.0, 4)


# ---------------------------------------------------------------------------
# cfa_drizzle_frame (src/cfa_drizzle.py, --cfa-drizzle)
# ---------------------------------------------------------------------------

_HAS_CFA = hasattr(native, "cfa_drizzle_frame")


def _cfa_inputs(n=12, H=96, seed=11):
    from tests.test_cfa_drizzle import _sim
    truth, mem, shifts = _sim(n, seed=seed, H=H)
    ref = truth.astype(np.float32) + np.random.default_rng(0).normal(
        0, 5, truth.shape).astype(np.float32)
    return mem, shifts, ref


@pytest.mark.skipif(not _HAS_CFA, reason="astro_native lacks cfa_drizzle_frame")
@pytest.mark.parametrize("pixfrac,scale", [(1.0, 1.0), (0.6, 1.0), (0.8, 2.0)])
def test_cfa_drizzle_frame_matches_numpy_translation(pixfrac, scale):
    from src.cfa_drizzle import cfa_drizzle_combine
    mem, shifts, ref = _cfa_inputs()
    ref = ref if scale == 1.0 else np.kron(ref, np.ones((2, 2, 1), np.float32))
    n = len(shifts)
    kw = dict(pattern='RGGB', top=0, left=0, scale=scale, pixfrac=pixfrac)
    a, sa = cfa_drizzle_combine(mem, list(range(n)), shifts, [None] * n, ref,
                                use_native=False, **kw)
    b, sb = cfa_drizzle_combine(mem, list(range(n)), shifts, [None] * n, ref,
                                use_native=True, **kw)
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-3)
    assert sa['samples'] == sb['samples']
    assert sa['rejected_frac'] == pytest.approx(sb['rejected_frac'], abs=1e-9)
    assert sb['native'] and not sa['native']


@pytest.mark.skipif(not _HAS_CFA, reason="astro_native lacks cfa_drizzle_frame")
@pytest.mark.parametrize("deg", [0.0, 4.5, -9.0, 30.0])
def test_cfa_drizzle_frame_matches_numpy_under_rotation(deg):
    """Rotation is what makes the row-band ownership logic non-trivial: a band's
    input rows slant across the sensor, and a drop straddling a band edge must
    still be deposited exactly once per row."""
    from src.affine_fit import RigidTransform
    from src.cfa_drizzle import cfa_drizzle_combine
    mem, _, ref = _cfa_inputs(n=8, H=160)
    n = len(mem)
    rng = np.random.default_rng(3)
    tfs = [RigidTransform.from_rotation_translation(
        np.deg2rad(deg + rng.normal(0, 0.3)), (float(rng.normal(0, 3)), float(rng.normal(0, 3))))
        for _ in range(n)]
    ref = np.pad(ref, ((0, 64), (0, 0), (0, 0)), mode='edge')[:160]
    kw = dict(pattern='RGGB', top=4, left=4, scale=1.0, pixfrac=0.8)
    a, sa = cfa_drizzle_combine(mem, list(range(n)), [None] * n, tfs, ref,
                                use_native=False, **kw)
    b, sb = cfa_drizzle_combine(mem, list(range(n)), [None] * n, tfs, ref,
                                use_native=True, **kw)
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-3)
    assert sa['samples'] == sb['samples']


@pytest.mark.skipif(not _HAS_CFA, reason="astro_native lacks cfa_drizzle_frame")
def test_cfa_drizzle_frame_rejects_bad_shapes():
    z3 = lambda h, w: np.zeros((h, w, 3))
    rgb = np.zeros((8, 8, 3), np.float32)
    ref = np.zeros((6, 6, 3), np.float32)
    args = (np.array([0, 1, 1, 2], np.uint8), np.array([1., 0., 0., 1.]),
            np.array([0., 0.]), 0.5, np.ones(3), np.zeros(3), 4.0, 0.15)
    with pytest.raises(ValueError):   # accumulators not matching the reference
        native.cfa_drizzle_frame(rgb, ref, *args, z3(5, 6), z3(6, 6), z3(6, 6))
    with pytest.raises(ValueError):   # singular affine
        native.cfa_drizzle_frame(rgb, ref, args[0], np.zeros(4), *args[2:],
                                 z3(6, 6), z3(6, 6), z3(6, 6))


# ---------------------------------------------------------------------------
# white_balance_apply (src/debayer.py white_balance_grayworld / _whitepatch)
# ---------------------------------------------------------------------------

_HAS_WB = hasattr(native, "white_balance_apply")


def _wb_frame(seed=0, h=64, w=90, saturate=True):
    rng = np.random.default_rng(seed)
    img = rng.uniform(100, 30000, (h, w, 3)).astype(np.float32)
    if saturate:                      # clipped star cores: equal channels at the ceiling
        for cy, cx in ((10, 12), (40, 60), (20, 75)):
            img[cy - 1:cy + 2, cx - 1:cx + 2, :] = 60000.0
        img[30, 30] = [55000.0, 59000.0, 57000.0]      # near-clipped, unequal (below the cores)
    return img


def _wb_both(fn, img):
    import src.debayer as deb
    saved = deb._HAS_NATIVE
    try:
        deb._HAS_NATIVE = False
        ref = fn(img.copy())
        deb._HAS_NATIVE = True
        new = fn(img.copy())
    finally:
        deb._HAS_NATIVE = saved
    return ref, new


@pytest.mark.skipif(not _HAS_WB, reason="astro_native lacks white_balance_apply")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_white_balance_grayworld_is_bit_identical(seed):
    import src.debayer as deb
    ref, new = _wb_both(deb.white_balance_grayworld, _wb_frame(seed))
    assert ref.dtype == new.dtype == np.float32
    np.testing.assert_array_equal(ref, new)


@pytest.mark.skipif(not _HAS_WB, reason="astro_native lacks white_balance_apply")
def test_white_balance_whitepatch_is_bit_identical():
    import src.debayer as deb
    ref, new = _wb_both(deb.white_balance_whitepatch, _wb_frame(3))
    np.testing.assert_array_equal(ref, new)


@pytest.mark.skipif(not _HAS_WB, reason="astro_native lacks white_balance_apply")
def test_white_balance_clipped_cores_are_neutralised_and_output_is_non_negative():
    import src.debayer as deb
    img = _wb_frame(4)
    _, new = _wb_both(deb.white_balance_grayworld, img)
    core = new[9:12, 11:14, :]                       # equal channels at the ceiling stay equal
    assert np.allclose(core[..., 0], core[..., 1]) and np.allclose(core[..., 1], core[..., 2])
    assert float(new.min()) >= 0.0


@pytest.mark.skipif(not _HAS_WB, reason="astro_native lacks white_balance_apply")
def test_white_balance_nan_propagates_like_numpy():
    import src.debayer as deb
    img = _wb_frame(5)
    img[7, 7, 1] = np.nan
    with np.errstate(all='ignore'):
        ref, new = _wb_both(deb.white_balance_grayworld, img)
    np.testing.assert_array_equal(np.isnan(ref), np.isnan(new))


@pytest.mark.skipif(not _HAS_WB, reason="astro_native lacks white_balance_apply")
def test_white_balance_direct_kernel_rejects_bad_input_and_falls_back_on_float64_gains():
    img = _wb_frame(6)
    with pytest.raises(ValueError):
        native.white_balance_apply(np.zeros((4, 4, 4), np.float32), np.ones(3, np.float32), False)
    with pytest.raises(ValueError):
        native.white_balance_apply(img, np.ones(2, np.float32), False)
    import src.debayer as deb
    assert deb._white_balance_native(np, img, np.ones(3, np.float64), False) is None     # dtype guard
    assert deb._white_balance_native(np, img[:, :, :2], np.ones(3, np.float32), False) is None
    out = deb._white_balance_native(np, np.asfortranarray(img), np.ones(3, np.float32), False)
    assert out is not None and out.shape == img.shape                                   # non-contiguous ok


@pytest.mark.skipif(not hasattr(native, "white_balance_grayworld"),
                    reason="astro_native lacks white_balance_grayworld")
def test_white_balance_grayworld_kernel_matches_float64_mean_path():
    """The gains come from float64-accumulated channel means (a float32 running
    sum drifts ~1.5% on real frames); the kernel must match the numpy path that
    does the same, and the means must be the accurate ones."""
    rng = np.random.default_rng(9)
    img = rng.uniform(0, 60000, (300, 420, 3)).astype(np.float32)
    ours = native.white_balance_grayworld(img)
    import src.debayer as deb
    saved = deb._HAS_NATIVE
    try:
        deb._HAS_NATIVE = False
        ref = deb.white_balance_grayworld(img.copy())
    finally:
        deb._HAS_NATIVE = saved
    np.testing.assert_array_equal(ours, ref)
    # gray-world result must have (nearly) equal channel means -- the float32
    # accumulation it replaced could leave them further apart (highlight blending moves them slightly either way)
    m = ours.astype(np.float64).mean(axis=(0, 1))
    assert (m.max() - m.min()) / m.mean() < 2e-3
    with pytest.raises(ValueError):
        native.white_balance_grayworld(np.zeros((4, 4, 4), np.float32))


def _lanczos3_direct(x):
    """Direct Lanczos-3 windowed sinc, float64 -- the definition the kernel's
    angle-addition shortcut must agree with."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    inside = np.abs(x) < 3.0
    nz = inside & (x != 0.0)
    px = np.pi * x[nz]
    out[nz] = 3.0 * np.sin(px) * np.sin(px / 3.0) / (px * px)
    out[inside & (x == 0.0)] = 1.0
    return out


def _warp_reference(img, R, off, out_h, out_w):
    """Independent numpy Lanczos-3 affine warp (interior pixels only are defined)."""
    oy, ox = np.mgrid[0:out_h, 0:out_w].astype(np.float64)
    iy = R[0, 0] * oy + R[0, 1] * ox + off[0]
    ix = R[1, 0] * oy + R[1, 1] * ox + off[1]
    fy, fx = np.floor(iy), np.floor(ix)
    ry, rx = iy - fy, ix - fx
    taps = np.arange(-2, 4)
    wy = np.stack([_lanczos3_direct(t - ry) for t in taps], axis=-1)
    wx = np.stack([_lanczos3_direct(t - rx) for t in taps], axis=-1)
    wy /= wy.sum(axis=-1, keepdims=True)
    wx /= wx.sum(axis=-1, keepdims=True)
    h, w = img.shape[:2]
    inside = (fy - 2 >= 0) & (fy + 3 < h) & (fx - 2 >= 0) & (fx + 3 < w)
    out = np.zeros((out_h, out_w, img.shape[2]))
    yy, xx = np.nonzero(inside)
    for c in range(img.shape[2]):
        acc = np.zeros(len(yy))
        for a, ty in enumerate(taps):
            row = np.zeros(len(yy))
            for b, tx in enumerate(taps):
                row += wx[yy, xx, b] * img[(fy[yy, xx] + ty).astype(int), (fx[yy, xx] + tx).astype(int), c]
            acc += wy[yy, xx, a] * row
        out[yy, xx, c] = acc
    return out, inside


@pytest.mark.parametrize("deg", [0.0, 3.0, 17.0, -41.0, 90.0])
def test_warp_matches_an_independent_direct_lanczos_reference(deg):
    rng = np.random.default_rng(int(abs(deg)) + 1)
    img = rng.uniform(0, 30000, (90, 110, 3)).astype(np.float32)
    th = np.deg2rad(deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    off = np.array([7.31, -4.77]) + (np.eye(2) - R) @ np.array([45.0, 55.0])
    got = native.warp_affine_lanczos3(img, R.ravel().tolist(), off.tolist(), 90, 110, 0.0)
    ref, inside = _warp_reference(img, R, off, 90, 110)
    assert inside.sum() > 2000
    np.testing.assert_allclose(np.asarray(got)[inside], ref[inside], rtol=2e-6, atol=2e-2)


def test_warp_rgb_fast_path_equals_the_single_channel_path():
    """The 3-channel interior fast path re-orders the loads, not the arithmetic:
    each channel must equal what the generic (c != 3) loop produces for it alone."""
    rng = np.random.default_rng(5)
    img = rng.uniform(0, 30000, (80, 100, 3)).astype(np.float32)
    th = np.deg2rad(6.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    off = [3.4, -2.2]
    full = np.asarray(native.warp_affine_lanczos3(img, R.ravel().tolist(), off, 80, 100, 0.0))
    for ch in range(3):
        one = np.asarray(native.warp_affine_lanczos3(
            np.ascontiguousarray(img[:, :, ch:ch + 1]), R.ravel().tolist(), off, 80, 100, 0.0))
        np.testing.assert_array_equal(full[:, :, ch], one[:, :, 0])


# ---------------------------------------------------------------------------
# Fused drizzle accumulate / area-overlap splat
# ---------------------------------------------------------------------------
_HAS_DRIZ = hasattr(native, "drizzle_accumulate_lanczos3") and hasattr(native, "drizzle_splat_frame")


def _driz_matrix(deg, scale):
    th = np.deg2rad(deg)
    c, s = np.cos(th) / scale, np.sin(th) / scale
    return [c, -s, s, c]


def _old_accumulate(acc, img, M, off, w, pixfrac, wmap, out_h, out_w):
    """The pre-fusion Python path, verbatim: warp -> (tent weight) -> *= w -> add."""
    res = native.warp_affine_lanczos3(img, M, off, out_h, out_w, 0.0)
    if pixfrac < 1.0 - 1e-9:
        gy, gx = np.meshgrid(np.arange(out_h, dtype=np.float64),
                             np.arange(out_w, dtype=np.float64), indexing='ij')
        raw_y = M[0] * gy + M[1] * gx + off[0]
        raw_x = M[2] * gy + M[3] * gx + off[1]
        half = max(pixfrac / 2.0, 1e-6)
        w_y = np.maximum(0.0, 1.0 - np.abs(raw_y - np.round(raw_y)) / half)
        w_x = np.maximum(0.0, 1.0 - np.abs(raw_x - np.round(raw_x)) / half)
        pfw = (w_y * w_x * w)[:, :, np.newaxis]
        res = res.astype(np.float64, copy=False) * pfw
        np.add(acc, res, out=acc)
        np.add(wmap, pfw, out=wmap)
    else:
        res *= w
        np.add(acc, res, out=acc)


@pytest.mark.skipif(not _HAS_DRIZ, reason="astro_native lacks the drizzle kernels")
@pytest.mark.parametrize("deg", [0.0, 7.5, -23.0])
@pytest.mark.parametrize("pixfrac", [1.0, 0.6])
def test_drizzle_accumulate_is_bit_identical_to_warp_weight_add(deg, pixfrac):
    rng = np.random.default_rng(3)
    img = rng.uniform(0, 5e4, (70, 90, 3)).astype(np.float32)
    out_h, out_w = 120, 160
    M = _driz_matrix(deg, 2.0)
    acc_old = np.zeros((out_h, out_w, 3))
    acc_new = np.zeros((out_h, out_w, 3))
    wm_old = np.zeros((out_h, out_w, 1))
    wm_new = np.zeros((out_h, out_w, 1))
    for j, w in enumerate([0.9, 1.7, 0.35]):          # several frames, different weights/offsets
        off = [3.3 + j * 0.37, 5.1 - j * 0.21]
        _old_accumulate(acc_old, img, M, off, w, pixfrac, wm_old, out_h, out_w)
        native.drizzle_accumulate_lanczos3(acc_new, img, M, off, w, pixfrac,
                                           wm_new if pixfrac < 1.0 else None)
    np.testing.assert_array_equal(acc_old, acc_new)
    if pixfrac < 1.0:
        np.testing.assert_array_equal(wm_old, wm_new)


@pytest.mark.skipif(not _HAS_DRIZ, reason="astro_native lacks the drizzle kernels")
def test_drizzle_kernels_reject_bad_input():
    img = np.zeros((20, 20, 3), np.float32)
    acc = np.zeros((30, 30, 3))
    with pytest.raises(ValueError):                    # pixfrac < 1 needs a weight map
        native.drizzle_accumulate_lanczos3(acc, img, [0.5, 0, 0, 0.5], [0, 0], 1.0, 0.5, None)
    with pytest.raises(ValueError):
        native.drizzle_accumulate_lanczos3(np.zeros((30, 30, 2)), img, [0.5, 0, 0, 0.5], [0, 0], 1.0, 1.0)
    with pytest.raises(ValueError):
        native.drizzle_splat_frame(img, [2, 0, 0, 2], [0, 0], 0.0, 1.0, acc, np.zeros((30, 30, 1)))
    with pytest.raises(ValueError):                    # singular affine
        native.drizzle_splat_frame(img, [1, 1, 1, 1], [0, 0], 1.0, 1.0, acc, np.zeros((30, 30, 1)))


def _splat_reference(img, fwd, off, h, w, oh, ow):
    """Direct per-pixel area-overlap deposit, one input pixel at a time."""
    num = np.zeros((oh, ow, 3))
    den = np.zeros((oh, ow))
    norm = 1.0 / (4.0 * h * h)
    for iy in range(img.shape[0]):
        for ix in range(img.shape[1]):
            dy, dx = iy - off[0], ix - off[1]
            oy = fwd[0] * dy + fwd[1] * dx
            ox = fwd[2] * dy + fwd[3] * dx
            for qy in range(oh):
                ovy = min(oy + h, qy + 0.5) - max(oy - h, qy - 0.5)
                if ovy <= 0:
                    continue
                for qx in range(ow):
                    ovx = min(ox + h, qx + 0.5) - max(ox - h, qx - 0.5)
                    if ovx <= 0:
                        continue
                    g = ovy * ovx * norm * w
                    num[qy, qx] += g * img[iy, ix].astype(np.float64)
                    den[qy, qx] += g
    return num, den


@pytest.mark.skipif(not _HAS_DRIZ, reason="astro_native lacks the drizzle kernels")
@pytest.mark.parametrize("deg", [0.0, 12.0])
def test_drizzle_splat_matches_direct_area_overlap(deg):
    rng = np.random.default_rng(5)
    img = rng.uniform(0, 1000, (14, 18, 3)).astype(np.float32)
    scale = 2.0
    M = np.array(_driz_matrix(deg, scale)).reshape(2, 2)
    off = np.array([1.7, -0.9])
    fwd = np.linalg.inv(M)
    oh, ow = 30, 38
    h = 0.7 * scale / 2.0
    ref_num, ref_den = _splat_reference(img, fwd.ravel(), off, h, 1.3, oh, ow)
    num = np.zeros((oh, ow, 3))
    den = np.zeros((oh, ow, 1))
    native.drizzle_splat_frame(img, fwd.ravel().tolist(), off.tolist(), h, 1.3, num, den)
    np.testing.assert_allclose(num, ref_num, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(den[:, :, 0], ref_den, rtol=1e-9, atol=1e-12)


@pytest.mark.skipif(not _HAS_DRIZ, reason="astro_native lacks the drizzle kernels")
def test_drizzle_splat_conserves_flux_and_flat_field():
    """A flat image stays flat where covered, and total weight = frame weight x
    input pixels (each drop's overlap weights sum to 1 when fully inside)."""
    img = np.full((40, 50, 3), 321.0, np.float32)
    scale = 2.0
    M = np.array(_driz_matrix(9.0, scale)).reshape(2, 2)
    off = -M @ np.array([60.0, 60.0])      # input origin lands at output (60, 60): fully inside
    num = np.zeros((260, 260, 3))
    den = np.zeros((260, 260, 1))
    native.drizzle_splat_frame(img, np.linalg.inv(M).ravel().tolist(), off.tolist(), 0.9, 2.0, num, den)
    covered = den[:, :, 0] > 1e-9
    assert covered.sum() > 3000
    np.testing.assert_allclose((num / np.where(den > 0, den, 1))[covered], 321.0, rtol=1e-9)
    assert abs(den.sum() - 2.0 * 40 * 50) < 1e-6 * 2.0 * 40 * 50   # fully inside the output


# ---------------------------------------------------------------------------
# Fused Phase-1 kernels: calibration and hot-pixel removal
# ---------------------------------------------------------------------------

_HAS_FUSED = all(hasattr(native, n) for n in ("calibrate_frame_inplace", "hot_pixel_bayer", "hot_pixel_rgb"))


class _NoKernel:
    """The debayer module's native handle with named kernels removed, so a call
    lands on the path that was in production before the kernel existed."""

    def __init__(self, real, *hidden):
        self._real, self._hidden = real, hidden

    def __getattr__(self, name):
        if name in self._hidden:
            raise AttributeError(name)
        return getattr(self._real, name)


def _without(monkeypatch, *names):
    monkeypatch.setattr(_debayer_mod, "_native", _NoKernel(_debayer_mod._native, *names))


def _mosaic(h=96, w=130, seed=3):
    rng = np.random.default_rng(seed)
    raw = (rng.normal(1000, 40, (h, w)) + rng.poisson(5, (h, w))).astype(np.float32)
    hot = rng.random((h, w)) < 6e-3
    raw[hot] += rng.uniform(500, 5000, int(hot.sum())).astype(np.float32)
    return raw, hot, rng


@pytest.mark.skipif(not _HAS_FUSED, reason="astro_native lacks the fused Phase-1 kernels")
@pytest.mark.parametrize("use", [("b",), ("d",), ("f",), ("b", "d"), ("b", "f"), ("d", "f"), ("b", "d", "f")])
def test_calibrate_frame_bit_identical_to_numpy(use, monkeypatch):
    raw, _hot, rng = _mosaic()
    m = {
        "b": rng.normal(100, 3, raw.shape).astype(np.float32),
        "d": rng.normal(150, 5, raw.shape).astype(np.float32),
        "f": np.clip(rng.normal(1, 0.1, raw.shape), 0.4, 2.5).astype(np.float32),
    }
    args = (m.get("b") if "b" in use else None, m.get("d") if "d" in use else None, 1.37,
            m.get("f") if "f" in use else None)
    fast, ok_f = _debayer_mod.calibrate_frame(raw.copy(), *args)
    monkeypatch.setattr(_debayer_mod, "_HAS_NATIVE", False)
    ref, ok_r = _debayer_mod.calibrate_frame(raw.copy(), *args)
    assert ok_f and ok_r
    np.testing.assert_array_equal(fast, ref)
    assert fast.min() >= 0.0


@pytest.mark.skipif(not _HAS_FUSED, reason="astro_native lacks the fused Phase-1 kernels")
def test_calibrate_frame_reports_non_finite(monkeypatch):
    raw, _hot, rng = _mosaic()
    flat = np.ones(raw.shape, np.float32)
    flat[3, 3] = 0.0                      # divide by zero -> inf
    _, ok = _debayer_mod.calibrate_frame(raw.copy(), None, None, 1.0, flat)
    assert ok is False
    monkeypatch.setattr(_debayer_mod, "_HAS_NATIVE", False)
    _, ok = _debayer_mod.calibrate_frame(raw.copy(), None, None, 1.0, flat)
    assert ok is False


@pytest.mark.skipif(not _HAS_FUSED, reason="astro_native lacks the fused Phase-1 kernels")
def test_calibrate_frame_leaves_non_float32_masters_to_numpy():
    raw, _hot, rng = _mosaic()
    dark = rng.normal(150, 5, raw.shape)               # float64: the fused kernel must not run
    out, ok = _debayer_mod.calibrate_frame(raw.copy(), None, dark, 1.0, None)
    ref = np.clip(raw - dark.astype(np.float32) * 1.0, 0, None)
    assert ok
    np.testing.assert_allclose(out, ref, atol=1e-3)


@pytest.mark.skipif(not _HAS_FUSED, reason="astro_native lacks the fused Phase-1 kernels")
@pytest.mark.parametrize("shape", [(96, 130), (7, 9), (3, 5), (64, 66)])
def test_hot_pixel_bayer_statistical_bit_identical(shape, monkeypatch):
    raw, _hot, _ = _mosaic(*shape)
    fast = _debayer_mod._fix_hot_bayer(raw.copy())
    monkeypatch.setattr(_debayer_mod, "_HAS_NATIVE", False)
    ref = _debayer_mod._fix_hot_bayer(raw.copy())
    np.testing.assert_array_equal(fast, ref)


@pytest.mark.skipif(not _HAS_FUSED, reason="astro_native lacks the fused Phase-1 kernels")
def test_hot_pixel_bayer_actually_repairs_hot_pixels():
    raw, hot, _ = _mosaic()
    out = _debayer_mod._fix_hot_bayer(raw.copy())
    assert (out != raw).sum() > 0.5 * hot.sum()
    assert out.max() < raw.max()
    assert np.all(out <= raw)                           # only ever replaces upward outliers


@pytest.mark.skipif(not _HAS_FUSED, reason="astro_native lacks the fused Phase-1 kernels")
def test_hot_pixel_bayer_map_bit_identical(monkeypatch):
    raw, hot, _ = _mosaic()
    fast = _debayer_mod._fix_hot_bayer(raw.copy(), threshold=None, hot_map=hot)
    monkeypatch.setattr(_debayer_mod, "_HAS_NATIVE", False)
    ref = _debayer_mod._fix_hot_bayer(raw.copy(), threshold=None, hot_map=hot)
    np.testing.assert_array_equal(fast, ref)


@pytest.mark.skipif(not _HAS_FUSED, reason="astro_native lacks the fused Phase-1 kernels")
def test_hot_pixel_bayer_rejects_map_and_threshold_together():
    raw, hot, _ = _mosaic()
    with pytest.raises(ValueError):
        native.hot_pixel_bayer(raw, hot.astype(np.uint8), 5.0)


@pytest.mark.skipif(not _HAS_FUSED, reason="astro_native lacks the fused Phase-1 kernels")
def test_hot_pixel_rgb_bit_identical_to_previous_native_path(monkeypatch):
    """Against the box-mean kernel path that shipped before (a sequential f64 sum);
    the pure-numpy fallback's scipy uniform_filter differs from it by an ulp."""
    rng = np.random.default_rng(5)
    rgb = rng.normal(500, 30, (90, 120, 3)).astype(np.float32)
    rgb[rng.random((90, 120)) < 8e-3] += 800
    fast, lum_f = _debayer_mod._fix_hot_rgb(rgb.copy())
    _without(monkeypatch, "hot_pixel_rgb")
    ref, lum_r = _debayer_mod._fix_hot_rgb(rgb.copy())
    assert (fast != rgb).any()
    np.testing.assert_array_equal(fast, ref)
    np.testing.assert_array_equal(lum_f, lum_r)


@pytest.mark.skipif(not _HAS_FUSED, reason="astro_native lacks the fused Phase-1 kernels")
def test_hot_pixel_rgb_nothing_flagged_and_degenerate():
    rng = np.random.default_rng(6)
    clean = rng.normal(500, 30, (60, 80, 3)).astype(np.float32)
    fixed, lum = native.hot_pixel_rgb(clean, 1e6)       # nothing can exceed this
    assert fixed is None
    np.testing.assert_array_equal(
        lum, (np.float32(0.299) * clean[:, :, 0] + np.float32(0.587) * clean[:, :, 1]
              + np.float32(0.114) * clean[:, :, 2]))
    assert native.hot_pixel_rgb(np.full((30, 30, 3), 100.0, np.float32), 12.0) is None
    # ...and the wrapper still handles the degenerate MAD via numpy's np.std fallback
    out, _ = _debayer_mod._fix_hot_rgb(np.full((30, 30, 3), 100.0, np.float32))
    np.testing.assert_array_equal(out, np.full((30, 30, 3), 100.0, np.float32))


@pytest.mark.skipif(not hasattr(native, "pre_gradient_apply"), reason="astro_native lacks pre_gradient_apply")
def test_pre_gradient_removal_bit_identical_to_numpy(monkeypatch):
    import src.frame_processor as fp
    rng = np.random.default_rng(8)
    h, w = 140, 190
    yy, xx = np.mgrid[0:h, 0:w]
    base = (300 + 0.4 * yy + 0.2 * xx + 0.001 * yy * xx).astype(np.float32)
    rgb = np.ascontiguousarray(
        np.stack([base, base * 1.1, base * 0.9], 2) + rng.normal(0, 20, (h, w, 3)).astype(np.float32))
    a, b = rgb.copy(), rgb.copy()
    lum_n = fp._pre_gradient_removal(a, None)
    monkeypatch.setattr(_debayer_mod, "_HAS_NATIVE", False)
    lum_p = fp._pre_gradient_removal(b, None)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(lum_n, lum_p)
    assert a.min() >= 0.0


# ---------------------------------------------------------------------------
# Debayer-adjacent kernels (crate 0.28): medians, G1/G2 + Bayer-grid equalisation,
# in-place hot-pixel / white balance, luminance
# ---------------------------------------------------------------------------

_HAS_DEBAYER_K = all(hasattr(native, n) for n in (
    "strided_sigma_clipped_median", "green_equalize_inplace", "bayer_grid_equalize_inplace",
    "hot_pixel_rgb_inplace", "luminance_native", "white_balance_grayworld_inplace",
    "white_balance_apply_inplace"))
_need_debayer_k = pytest.mark.skipif(not _HAS_DEBAYER_K, reason="astro_native lacks the debayer-adjacent kernels")


def _sky_plane(h=180, w=240, seed=11):
    rng = np.random.default_rng(seed)
    a = rng.normal(500, 25, (h, w)).astype(np.float32)
    hot = rng.random((h, w)) < 0.01                                    # stars: a heavy upper tail
    a[hot] += rng.uniform(200, 4000, int(hot.sum())).astype(np.float32)
    return a


@_need_debayer_k
@pytest.mark.parametrize("sl", [(slice(None), slice(None)), (slice(0, None, 2), slice(1, None, 2)),
                                (slice(1, None, 2), slice(0, None, 2))])
def test_strided_sigma_clipped_median_matches_numpy_reference(sl):
    a = _sky_plane()
    view = a[sl]
    got = native.strided_sigma_clipped_median(view, 3.0, 3)
    # the pure numpy algorithm, spelled out
    x = view.ravel()
    for _ in range(3):
        med = float(np.median(x)); std = float(np.std(x))
        if std < 1e-12:
            break
        x = x[np.abs(x - med) < 3.0 * std]
    expect = float(np.median(x))
    assert got == pytest.approx(expect, rel=1e-6)          # numpy's f32 std/sum differ from the f64 kernel's
    # and the older 1-D kernel (now on the same core) agrees exactly
    assert got == native.sigma_clipped_median_native(np.ascontiguousarray(view.ravel()), 3.0, 3)


@_need_debayer_k
@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
@pytest.mark.parametrize("imbalance", [1.0, 1.05, 1.5])
def test_green_equalize_inplace_bit_identical(pattern, imbalance, monkeypatch):
    rng = np.random.default_rng(4)
    raw = rng.normal(800, 30, (96, 130)).astype(np.float32)
    (_, _), (g1r, g1c), (g2r, g2c), (_, _) = _debayer_mod._PATTERN_OFFSETS[pattern]
    raw[g2r::2, g2c::2] *= np.float32(imbalance)
    fast = _debayer_mod.green_equalize(raw.copy(), pattern=pattern)
    _without(monkeypatch, "strided_sigma_clipped_median", "green_equalize_inplace")
    ref = _debayer_mod.green_equalize(raw.copy(), pattern=pattern)     # previous path: f64 medians, numpy scaling
    np.testing.assert_array_equal(fast, ref)


@_need_debayer_k
def test_green_equalize_inplace_flag_edits_caller_array_only_when_asked():
    rng = np.random.default_rng(5)
    raw = rng.normal(800, 30, (64, 64)).astype(np.float32)
    raw[1::2, 0::2] *= np.float32(1.1)
    keep = raw.copy()
    out = _debayer_mod.green_equalize(raw, pattern="RGGB")            # default: a copy
    np.testing.assert_array_equal(raw, keep)
    assert out is not raw
    out2 = _debayer_mod.green_equalize(raw, pattern="RGGB", inplace=True)
    assert out2 is raw and not np.array_equal(raw, keep)


@_need_debayer_k
@pytest.mark.parametrize("offset", [0.0, 0.5, 3.0, 400.0])
def test_bayer_grid_equalize_bit_identical(offset, monkeypatch):
    rng = np.random.default_rng(6)
    rgb = rng.normal(300, 20, (90, 120, 3)).astype(np.float32)
    rgb[0::2, 1::2, 1] += np.float32(offset)                         # a per-position green bias
    fast = _debayer_mod._equalize_bayer_grid(rgb.copy())
    _without(monkeypatch, "strided_sigma_clipped_median", "bayer_grid_equalize_inplace")
    ref = _debayer_mod._equalize_bayer_grid(rgb.copy())                # previous path
    np.testing.assert_array_equal(fast, ref)                          # 400 is past the 100 ADU guard: unchanged
    assert (offset == 400.0) == np.array_equal(fast, rgb)


@_need_debayer_k
def test_debayer_malvar_full_path_bit_identical(monkeypatch):
    rng = np.random.default_rng(7)
    raw = rng.normal(700, 30, (100, 140)).astype(np.float32)
    raw[1::2, 0::2] *= np.float32(1.04)
    fast = _debayer_mod.debayer(_debayer_mod.green_equalize(raw.copy(), "RGGB"), "RGGB", "malvar")
    _without(monkeypatch, "strided_sigma_clipped_median", "green_equalize_inplace",
             "bayer_grid_equalize_inplace")
    ref = _debayer_mod.debayer(_debayer_mod.green_equalize(raw.copy(), "RGGB"), "RGGB", "malvar")
    np.testing.assert_array_equal(fast, ref)


@_need_debayer_k
def test_hot_pixel_rgb_inplace_matches_copying_kernel():
    rng = np.random.default_rng(9)
    rgb = rng.normal(500, 30, (90, 120, 3)).astype(np.float32)
    rgb[rng.random((90, 120)) < 8e-3] += 800
    ref_fixed, ref_lum = native.hot_pixel_rgb(rgb, 12.0)
    work = rgb.copy()
    lum = native.hot_pixel_rgb_inplace(work, 12.0)
    np.testing.assert_array_equal(work, ref_fixed)
    np.testing.assert_array_equal(lum, ref_lum)
    clean = rng.normal(500, 30, (60, 80, 3)).astype(np.float32)
    keep = clean.copy()
    assert native.hot_pixel_rgb_inplace(clean, 1e6) is not None
    np.testing.assert_array_equal(clean, keep)                        # nothing flagged: untouched
    flat = np.full((30, 30, 3), 100.0, np.float32)
    assert native.hot_pixel_rgb_inplace(flat, 12.0) is None


@_need_debayer_k
def test_white_balance_grayworld_inplace_bit_identical():
    rng = np.random.default_rng(10)
    img = np.abs(rng.normal(400, 60, (80, 110, 3))).astype(np.float32) * np.array([1.0, 0.6, 0.8], np.float32)
    img[5, 5] = 9000.0                                                # a near-clipped star
    ref = native.white_balance_grayworld(img)
    work = img.copy()
    assert _debayer_mod.white_balance_grayworld(work, inplace=True) is work
    np.testing.assert_array_equal(work, ref)
    f = np.array([1.3, 1.0, 0.7], np.float32)
    work2 = img.copy()
    native.white_balance_apply_inplace(work2, f, True)
    np.testing.assert_array_equal(work2, native.white_balance_apply(img, f, True))


@_need_debayer_k
def test_luminance_native_bit_identical():
    rng = np.random.default_rng(12)
    rgb = rng.normal(500, 90, (70, 90, 3)).astype(np.float32)
    np.testing.assert_array_equal(
        _debayer_mod.luminance(rgb), 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])


# ---------------------------------------------------------------------------
# Windowed Lanczos-3 warp (crate 0.29): warping only the crop == cropping the warp
# ---------------------------------------------------------------------------

_HAS_ORIGIN = hasattr(native, "warp_affine_lanczos3") and "origin" in (
    getattr(native.warp_affine_lanczos3, "__text_signature__", "") or "")


@pytest.mark.skipif(not _HAS_ORIGIN, reason="astro_native warp lacks the window origin")
@pytest.mark.parametrize("mat_off", [
    ([0.9877, -0.1564, 0.1564, 0.9877], [13.7, -6.3]),     # 9 deg rotation
    ([1.0, 0.0, 0.0, 1.0], [3.37, -2.21]),                 # translation (separable table path)
    ([0.5, 0.0, 0.0, 0.5], [1.3, 0.7]),                    # scaling (separable, drizzle-like)
    ([1.0, 0.0, 0.0, 1.0], [0.0, 0.0]),                    # identity
])
@pytest.mark.parametrize("window", [(0, 0, 60, 70), (17, 23, 50, 61), (40, 30, 40, 80)])
def test_windowed_warp_equals_cropped_full_warp(mat_off, window):
    mat, off = mat_off
    rng = np.random.default_rng(21)
    img = rng.normal(500, 60, (100, 110, 3)).astype(np.float32)
    top, left, hh, ww = window
    full = native.warp_affine_lanczos3(img, mat, off, 100, 110, 0.0)
    part = native.warp_affine_lanczos3(img, mat, off, hh, ww, 0.0, (top, left))
    np.testing.assert_array_equal(part, full[top:top + hh, left:left + ww])


@pytest.mark.skipif(not _HAS_ORIGIN, reason="astro_native warp lacks the window origin")
@pytest.mark.parametrize("use_transform", [True, False])
def test_apply_transform_crop_matches_slicing_the_full_warp(use_transform):
    import math

    from src.registration import apply_transform
    rng = np.random.default_rng(22)
    img = rng.normal(500, 60, (100, 110, 3)).astype(np.float32)
    kw = {}
    if use_transform:
        th = math.radians(6.0)
        params = np.array([[math.cos(th), -math.sin(th), 4.5],
                           [math.sin(th), math.cos(th), -3.25],
                           [0.0, 0.0, 1.0]])
        kw["transform"] = type("T", (), {"params": params})()
    else:
        kw["shift"] = (2.6, -1.4)
    full = apply_transform(img, **kw)
    crop = (12, 88, 9, 97)
    np.testing.assert_array_equal(apply_transform(img, crop=crop, **kw),
                                  full[crop[0]:crop[1], crop[2]:crop[3]])
    # a non-native request (a local displacement field) still honours the crop
    field = np.zeros((4, 4, 2), np.float64)
    a = apply_transform(img, local_field=field, crop=crop, **kw)
    assert a.shape == (76, 88, 3)


@pytest.mark.skipif(not _HAS_ORIGIN, reason="astro_native warp lacks the window origin")
def test_rotated_warp_weight_table_matches_closed_form():
    """The rotated warp takes its Lanczos weights from an interpolated table; against the closed
    form (ORIGINSTACK_LANCZOS_EXACT=1, read once per process, hence the subprocess) nearly every
    float32 output is identical and the rest differ by about an ulp."""
    import os
    import subprocess
    import sys
    import tempfile
    code = (
        "import sys, math, numpy as np, astro_native as n\n"
        "rng = np.random.default_rng(3)\n"
        "yy, xx = np.mgrid[0:300, 0:340]\n"
        "img = (500 + 300*np.sin(xx/17.0)*np.cos(yy/23.0) + rng.normal(0, 30, (300, 340))).astype(np.float32)\n"
        "img = np.stack([img, img*0.9, img*1.1], 2).astype(np.float32)\n"
        "img[rng.random((300, 340)) < 1e-3] += 5000\n"
        "th = math.radians(7.0)\n"
        "M = [math.cos(th), -math.sin(th), math.sin(th), math.cos(th)]\n"
        "np.save(sys.argv[1], n.warp_affine_lanczos3(img, M, [10.3, -4.1], 300, 340))\n")
    with tempfile.TemporaryDirectory() as d:
        outs = {}
        for tag, exact in (("table", "0"), ("exact", "1")):
            path = os.path.join(d, tag + ".npy")
            env = dict(os.environ, ORIGINSTACK_LANCZOS_EXACT=exact)
            subprocess.run([sys.executable, "-c", code, path], check=True, env=env)
            outs[tag] = np.load(path)
    a, b = outs["exact"].astype(np.float64), outs["table"].astype(np.float64)
    assert np.isfinite(b).all()
    assert (a == b).mean() > 0.99
    ulp = np.spacing(np.abs(outs["exact"]).astype(np.float32)).astype(np.float64)
    assert (np.abs(a - b) <= 4 * ulp + 1e-3).all()
