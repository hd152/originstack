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
    rng = np.random.default_rng(4)
    fm = (rng.normal(size=(20, 17)) + 1j * rng.normal(size=(20, 17))).astype(np.complex64)
    ph = (rng.normal(size=(20, 17)) + 1j * rng.normal(size=(20, 17))).astype(np.complex64)
    num = np.zeros((20, 17), np.complex128)
    den = np.zeros((20, 17))
    pc._native.proper_coadd_accum(num.view(np.float64), den, fm.view(np.float32), ph.view(np.float32), 0.7, 1.3)
    want = 0.7 * fm.astype(np.complex128) * np.conj(ph.astype(np.complex128))
    np.testing.assert_allclose(num, want, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(den, 1.3 * np.abs(ph.astype(np.complex128)) ** 2, rtol=1e-12)


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
