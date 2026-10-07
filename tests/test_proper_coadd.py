"""Proper image coaddition (src/proper_coadd.py, --proper-coadd)."""
import numpy as np
import pytest

from src import proper_coadd as pc

_NATIVE = pc._native is not None and hasattr(pc._native, 'proper_coadd_prep')




@pytest.fixture(scope='module')
def field():
    rng = np.random.default_rng(1)
    H, W, N = 240, 300, 24
    yy, xx = np.mgrid[:H, :W].astype(float)
    ys, xs = rng.uniform(20, H - 20, 70), rng.uniform(20, W - 20, 70)
    fl = rng.lognormal(9, 0.8, 70)

    def render(sig):
        p = pc.moffat_params(sig, sig, 0.0, 3.0)
        norm = pc._moffat_grid(*np.mgrid[-60:61, -60:61].astype(float), p).sum()
        img = np.zeros((H, W))
        for y, x, f in zip(ys, xs, fl):
            sl = (slice(int(y) - 25, int(y) + 26), slice(int(x) - 25, int(x) + 26))
            img[sl] += f * pc._moffat_grid(yy[sl] - y, xx[sl] - x, p) / norm
        return img
    sigmas = rng.uniform(2.5, 5.0, N)          # per-frame FWHM, px
    Fs = rng.uniform(0.8, 1.1, N)
    frames = np.stack([F * render(s) + 200 + rng.normal(0, 6.0, (H, W)) for s, F in zip(sigmas, Fs)])
    A = np.ascontiguousarray(frames.astype(np.float32)[..., None].repeat(3, -1))
    A[5, 100, 100, :] += 5000.0                                     # a cosmic ray
    return A, Fs


def _fwhm(img):
    stars = pc.select_psf_stars(img[..., 1], 5.0)
    p, _ = pc.fit_psf(img[..., 1], stars)
    return pc.psf_fwhm(p)


def test_sharper_than_the_weighted_mean_and_flux_preserved(field):
    A, Fs = field
    w = Fs ** 2
    wmean = (A / Fs[:, None, None, None] * w[:, None, None, None]).sum(0) / w.sum()
    # a mean, like the pipeline's sigma-clip stack (a median of star profiles of
    # different widths is not flux-preserving, so it cannot be the flux yardstick)
    ref = A.mean(0)
    ref[100, 100] = np.median(A[:, 100, 100], 0)       # clipped there, as sigma-clip would be
    out = pc.proper_coadd(A, ref, fwhm=5.0, verbose=False)
    assert out is not None and out.shape == ref.shape and out.dtype == np.float32
    assert _fwhm(out) < _fwhm(wmean)              # MTF = weighted RMS >= weighted mean
    # flux: the zero-frequency term is the w-weighted mean of M_j / F_j whatever the PSF
    # model; F_j is relative to the median frame, so the total equals the truth (wmean,
    # F = 1) times the median transparency
    def total(img):
        g = img[..., 1].astype(np.float64)
        return (g - np.median(g)).sum()
    assert abs(total(out) / (total(wmean) * np.median(Fs)) - 1.0) < 0.01
    # the cosmic ray in one frame is gone
    assert out[100, 100, 1] < np.median(out[..., 1]) + 30


def test_no_stars_falls_back():
    rng = np.random.default_rng(2)
    A = rng.normal(100, 5, (5, 64, 80, 3)).astype(np.float32)
    assert pc.proper_coadd(A, A.mean(0), verbose=False) is None


@pytest.mark.skipif(not _NATIVE, reason='astro_native lacks proper_coadd kernels')
def test_native_prep_bit_identical_to_numpy():
    rng = np.random.default_rng(3)
    H, W, C, pad = 50, 70, 3, 8
    fr = rng.normal(500, 40, (H, W, C)).astype(np.float32)
    fr[10, 10] += 9000
    ref_sig = rng.normal(0, 30, (C, H, W)).astype(np.float32)
    ref_sig[:, 20:23, 30:33] += 4000
    sky, sig = [480.0, 510.5, 495.25], [35.0, 41.0, 38.5]
    a = np.zeros((C, H + 2 * pad, W + 2 * pad + 3), np.float32)
    b = a.copy()
    na = pc._native.proper_coadd_prep(fr, ref_sig, a, float(np.float32(0.93)),
                                      [float(np.float32(v)) for v in sky],
                                      [float(np.float32(v ** 2)) for v in sig],
                                      pc._REJECT_K, pc._SIGNAL_FRAC, pad)
    nb = pc._prep_numpy(fr, ref_sig, b, np.float32(0.93), sky, sig, H, W) if pad == pc._PAD else None
    if nb is None:                                  # the numpy mirror uses the module pad
        b2 = np.zeros((C, H + 2 * pc._PAD, W + 2 * pc._PAD), np.float32)
        nb = pc._prep_numpy(fr, ref_sig, b2, np.float32(0.93), sky, sig, H, W)
        b[:, pad:pad + H, pad:pad + W] = b2[:, pc._PAD:pc._PAD + H, pc._PAD:pc._PAD + W]
    assert na == nb >= C                               # at least the spike in every channel
    np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(not _NATIVE, reason='astro_native lacks proper_coadd kernels')
def test_native_accum_matches_numpy():
    """One accumulate step from zero: f64 arithmetic, then a single f32 rounding."""
    rng = np.random.default_rng(4)
    fm = (rng.normal(size=(20, 17)) + 1j * rng.normal(size=(20, 17))).astype(np.complex64)
    ph = (rng.normal(size=(20, 17)) + 1j * rng.normal(size=(20, 17))).astype(np.complex64)
    num = np.zeros((20, 17), np.complex64)
    den = np.zeros((20, 17), np.float32)
    pc._native.proper_coadd_accum32(num.view(np.float32), den, fm.view(np.float32), ph.view(np.float32), 0.7, 1.3)
    want = 0.7 * fm.astype(np.complex128) * np.conj(ph.astype(np.complex128))
    np.testing.assert_array_equal(num, want.astype(np.complex64))
    np.testing.assert_array_equal(den, (1.3 * np.abs(ph.astype(np.complex128)) ** 2).astype(np.float32))


@pytest.mark.skipif(not _NATIVE, reason='astro_native lacks proper_coadd kernels')
def test_native_and_numpy_paths_agree(field, monkeypatch):
    A, _ = field
    ref = np.median(A, 0)
    a = pc.proper_coadd(A, ref, fwhm=5.0, verbose=False)
    monkeypatch.setattr(pc, '_native', None)
    b = pc.proper_coadd(A, ref, fwhm=5.0, verbose=False)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-3 * float(np.std(a)))


def test_on_by_default_with_a_single_opt_out():
    """One store_false action (the desktop form keys on dest; two actions sharing it break that)."""
    from src.cli import build_parser
    p = build_parser()
    assert p.parse_args(['-d', 'x']).proper_coadd is True
    assert p.parse_args(['-d', 'x', '--no-proper-coadd']).proper_coadd is False
    acts = [a for a in p._actions if a.dest == 'proper_coadd']
    assert len(acts) == 1 and acts[0].option_strings == ['--no-proper-coadd']


@pytest.mark.skipif(not _NATIVE, reason='astro_native lacks proper_coadd kernels')
def test_accum32_matches_numpy_f64_accumulation():
    """Several frames into float32 accumulators vs the numpy fallback's float64 sums."""
    rng = np.random.default_rng(5)
    n64, d64 = np.zeros((20, 17), np.complex128), np.zeros((20, 17))
    n32, d32 = np.zeros((20, 17), np.complex64), np.zeros((20, 17), np.float32)
    for i in range(3):
        fm = (rng.normal(size=(20, 17)) + 1j * rng.normal(size=(20, 17))).astype(np.complex64)
        ph = (rng.normal(size=(20, 17)) + 1j * rng.normal(size=(20, 17))).astype(np.complex64)
        wn, wf = 0.7 + i, 1.3
        n64 += wn * (fm * np.conj(ph))
        d64 += wf * (ph.real.astype(np.float64) ** 2 + ph.imag.astype(np.float64) ** 2)
        pc._native.proper_coadd_accum32(n32.view(np.float32), d32, fm.view(np.float32), ph.view(np.float32), wn, wf)
    np.testing.assert_allclose(n32, n64, rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(d32, d64, rtol=1e-6)


def test_combine_is_deterministic(field):
    """Per-thread accumulators summed in a fixed order: identical output run to run."""
    A, _ = field
    ref = A.mean(0)
    a = pc.proper_coadd(A, ref, fwhm=5.0, verbose=False)
    b = pc.proper_coadd(A, ref, fwhm=5.0, verbose=False)
    np.testing.assert_array_equal(a, b)


def test_frame_sky_noise_native_matches_numpy():
    """astro_native.frame_sky_noise == _sky / _noise per channel, bit for bit."""
    an = pytest.importorskip("astro_native")
    if not hasattr(an, "frame_sky_noise"):
        pytest.skip("astro_native without frame_sky_noise")
    from src import proper_coadd as pc
    rng = np.random.default_rng(3)
    cases = [rng.normal(1000, 30, (203, 417, 3)).astype(np.float32),
             np.round(rng.normal(50, 4, (64, 90, 3))).astype(np.float32),     # ties
             rng.normal(0, 1, (40, 50, 3)).astype(np.float32)]                # < 1000 diffs
    fr = rng.normal(500, 10, (120, 160, 3)).astype(np.float32)
    fr[rng.random(fr.shape) < 0.05] = np.nan
    fr[:8, :, 1] = np.inf
    cases.append(fr)
    for fr in cases:
        sky, sig = an.frame_sky_noise(fr)
        for c in range(3):
            ref_sky, ref_sig = pc._sky(fr[..., c]), pc._noise(fr[..., c])
            assert sky[c] == ref_sky
            assert (sig[c] == ref_sig) or (np.isnan(sig[c]) and np.isnan(ref_sig))


_TILED = _NATIVE and pc._tiled_ok(np.zeros((1, 1, 1, 3), np.float32))


@pytest.mark.skipif(not _TILED, reason='astro_native lacks the tiled proper_coadd kernels')
def test_prep_rows_bit_identical_to_prep():
    """proper_coadd_prep_rows (bands of rows, batches of frames) == proper_coadd_prep."""
    rng = np.random.default_rng(6)
    H, W, C, pad = 37, 70, 3, 8
    frs = [rng.normal(500, 40, (H, W, C)).astype(np.float32) for _ in range(2)]
    frs[1][10, 10] += 9000
    ref_sig = rng.normal(0, 30, (C, H, W)).astype(np.float32)
    ref_sig[:, 20:23, 30:33] += 4000
    Fs, skies, sig2s = [1.0, 0.9], [[480.0, 510.5, 495.25], [470.0, 505.0, 490.5]], [[1225.0, 1681.0, 1482.25]] * 2
    PW = W + 2 * pad + 3
    full = [np.zeros((C, H + 2 * pad, PW), np.float32) for _ in frs]
    n_full = sum(pc._native.proper_coadd_prep(fr, ref_sig, b, F, sk, s2, 5.0, 0.15, pad)
                 for fr, b, F, sk, s2 in zip(frs, full, Fs, skies, sig2s))
    n_rows = 0
    for y0 in range(0, H, 5):
        nb = min(5, H - y0)
        band = np.full((2, C, nb, PW), np.nan, np.float32)            # every element is written
        n_rows += pc._native.proper_coadd_prep_rows(frs, ref_sig, band, y0, Fs, sum(skies, []),
                                                    sum(sig2s, []), 5.0, 0.15, pad)
        for b in range(2):
            np.testing.assert_array_equal(band[b], full[b][:, pad + y0:pad + y0 + nb])
    assert n_rows == n_full > 0


@pytest.mark.skipif(not _TILED, reason='astro_native lacks the tiled proper_coadd kernels')
def test_scatter_tiles_and_accum32_tile_match_accum32():
    """proper_coadd_scatter_tiles is a pure copy into (tile, row, column) blocks, and
    proper_coadd_accum32_tile on those blocks == proper_coadd_accum32 frame by frame."""
    rng = np.random.default_rng(7)
    C, PH, NF, CB, B = 3, 11, 37, 8, 2
    NT = -(-NF // CB)
    spec = (rng.normal(size=(B, C + 1, PH, NF)) + 1j * rng.normal(size=(B, C + 1, PH, NF))).astype(np.complex64)
    tiles = np.zeros((B * (C + 1), NT, PH, CB), np.complex64)
    for r0 in range(0, PH, 4):                                    # scattered in row bands
        rows = np.ascontiguousarray(spec[:, :, r0:r0 + 4].reshape(B * (C + 1), -1, NF))
        pc._native.proper_coadd_scatter_tiles(rows.view(np.float32), tiles.view(np.float32),
                                              list(range(B * (C + 1))), r0)
    flat = tiles.transpose(0, 2, 1, 3).reshape(B, C + 1, PH, NT * CB)
    np.testing.assert_array_equal(flat[..., :NF], spec)
    wn, wf = rng.uniform(0.5, 2, B * C), rng.uniform(0.5, 2, B * C)
    n_ref = np.zeros((C, PH, NF), np.complex64)
    d_ref = np.zeros((C, PH, NF), np.float32)
    for b in range(B):
        for c in range(C):
            pc._native.proper_coadd_accum32(n_ref[c].view(np.float32), d_ref[c], spec[b, c].view(np.float32),
                                            spec[b, C].view(np.float32), wn[b * C + c], wf[b * C + c])
    num = np.zeros((C, NT, PH, CB), np.complex64)
    den = np.zeros((C, NT, PH, CB), np.float32)
    t5 = tiles.reshape(B, C + 1, NT, PH, CB)
    for t in range(NT):
        nb = min(CB, NF - t * CB)
        blk = np.ascontiguousarray(t5[:, :, t, :, :nb])
        pc._native.proper_coadd_accum32_tile(num.view(np.float32), den, blk.view(np.float32), t,
                                             list(wn), list(wf))
    np.testing.assert_array_equal(num.transpose(0, 2, 1, 3).reshape(C, PH, -1)[..., :NF], n_ref)
    np.testing.assert_array_equal(den.transpose(0, 2, 1, 3).reshape(C, PH, -1)[..., :NF], d_ref)


@pytest.mark.skipif(not _TILED, reason='astro_native lacks the tiled proper_coadd kernels')
@pytest.mark.parametrize('batch', [1, 2])
def test_tiled_combine_bit_identical(field, monkeypatch, tmp_path, batch):
    """The tiled combine (_combine_tiled) gives exactly the per-frame path's output, from
    RAM and from a memmap, including a frame skipped for unusable noise."""
    A, _ = field
    ref = A.mean(0)
    real_sky_noise = pc._sky_noise

    def sky_noise(fr):                                     # frame 3's noise is unusable
        sky, sig = real_sky_noise(fr)
        if np.array_equal(fr, A[3]):
            sig = np.array([np.nan] * len(sig))
        return sky, sig
    monkeypatch.setattr(pc, '_sky_noise', sky_noise)
    monkeypatch.setattr(pc, '_BATCH', batch)
    real_ok = pc._tiled_ok

    def run(arr, tiled):
        monkeypatch.setattr(pc, '_tiled_ok', real_ok if tiled else (lambda a: False))
        st = {}
        return pc.proper_coadd(arr, ref, fwhm=5.0, verbose=False, stats=st), st
    want, st_want = run(A, False)
    assert st_want['frames'] == A.shape[0] - 1
    got, st_got = run(A, True)
    np.testing.assert_array_equal(got, want)
    assert st_got == st_want
    mm = np.memmap(tmp_path / 'aligned.dat', np.float32, 'w+', shape=A.shape)
    mm[:] = A
    got_mm, _ = run(mm, True)
    np.testing.assert_array_equal(got_mm, want)
    del mm
