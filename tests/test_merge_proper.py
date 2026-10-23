"""--merge-proper: nights weighted per spatial frequency by their PSF (merge.proper_merge)."""
import numpy as np
from scipy import ndimage as ndi

from src.merge import proper_merge
from src.proper_coadd import fit_psf, psf_fwhm, select_psf_stars

H, W = 512, 640
SKY = 100.0


def _truth(seed=0, n=45):
    rng = np.random.default_rng(seed)
    ys = rng.uniform(24, H - 24, n)
    xs = rng.uniform(24, W - 24, n)
    fl = rng.uniform(2e3, 2e4, n)
    return ys, xs, fl


def _night(truth, sigma_psf, noise, seed):
    ys, xs, fl = truth
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    img = np.full((H, W), SKY)
    for y, x, f in zip(ys, xs, fl):
        img += f / (2 * np.pi * sigma_psf ** 2) * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma_psf ** 2))
    rng = np.random.default_rng(seed)
    rgb = np.stack([img] * 3, -1) + rng.normal(0, noise, (H, W, 3))
    return rgb.astype(np.float32)


def _members(imgs, noises, valids=None):
    valids = valids or [np.ones((H, W), bool)] * len(imgs)
    return [(im, v, np.full(3, 1.0 / s ** 2)) for im, v, s in zip(imgs, valids, noises)]


def _ivw(members):
    num = sum(np.where(v[..., None], im, 0) * w for im, v, w in members)
    den = sum(np.where(v[..., None], 1.0, 0) * w for _, v, w in members)
    return (num / den).astype(np.float32)


def _fwhm(img):
    lum = img.mean(-1)
    p, _ = fit_psf(lum, select_psf_stars(lum, 3.0))
    return psf_fwhm(p)


def _hp_noise(img):
    lum = img.mean(-1)
    hp = lum - ndi.gaussian_filter(lum, 2.0)
    sky = ndi.gaussian_filter(lum, 3) < SKY + 2
    d = hp[sky]
    return 1.4826 * np.median(np.abs(d - np.median(d)))


def _two_nights():
    t = _truth()
    sharp = _night(t, 1.0, 3.0, seed=1)        # sharp, short: noisy
    soft = _night(t, 2.2, 1.5, seed=2)         # soft, long: deep
    return t, [sharp, soft], [3.0, 1.5]


def test_sharper_than_the_weighted_mean_at_the_same_noise_and_flux():
    t, imgs, noises = _two_nights()
    m = _members(imgs, noises)
    ivw = _ivw(m)
    pm = proper_merge(m, ivw)
    assert pm is not None
    assert _fwhm(pm) < 0.9 * _fwhm(ivw)
    assert abs(_hp_noise(pm) / _hp_noise(ivw) - 1.0) < 0.15
    ys, xs, fl = t
    yy, xx = np.mgrid[0:H, 0:W]
    for y, x, f in list(zip(ys, xs, fl))[:20]:
        ap = (yy - y) ** 2 + (xx - x) ** 2 < 12 ** 2
        got = float((pm[..., 0][ap] - SKY).sum())
        ref = float((ivw[..., 0][ap] - SKY).sum())
        assert abs(got / ref - 1.0) < 0.03


def test_identical_nights_give_the_weighted_mean():
    t = _truth()
    a = _night(t, 1.6, 2.0, seed=3)
    m = _members([a, a.copy()], [2.0, 2.0])
    ivw = _ivw(m)
    pm = proper_merge(m, ivw)
    assert pm is not None
    assert np.max(np.abs(pm - ivw)) < 1e-2 * float(ivw.max())


def test_only_the_common_footprint_changes():
    _, imgs, noises = _two_nights()
    half = np.zeros((H, W), bool)
    half[:, : int(W * 0.7)] = True
    m = _members(imgs, noises, [np.ones((H, W), bool), half])
    ivw = _ivw(m)
    pm = proper_merge(m, ivw)
    assert pm is not None and np.all(np.isfinite(pm))
    np.testing.assert_array_equal(pm[:, int(W * 0.7):], ivw[:, int(W * 0.7):])


def test_no_stars_means_no_proper_merge():
    rng = np.random.default_rng(0)
    flat = [(SKY + rng.normal(0, 2.0, (H, W, 3))).astype(np.float32) for _ in range(2)]
    m = _members(flat, [2.0, 2.0])
    assert proper_merge(m, _ivw(m)) is None


def test_cli_flag():
    from src.cli import parse_args
    assert parse_args(["-d", "x", "-o", "y.fits", "--merge-proper"]).merge_proper is True
    assert parse_args(["-d", "x", "-o", "y.fits"]).merge_proper is False
