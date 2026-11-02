"""src/psf_field.py: the width estimator, strip strengths and the equalising second pass."""
import numpy as np

from src import psf_field as pf

H, W = 600, 900
SIGMA = 2.0


def _field(fwhm_of, n=420, seed=2, axis='x'):
    rng = np.random.default_rng(seed)
    img = np.full((H, W), 200.0)
    for _ in range(n):
        y, x = rng.uniform(25, H - 25), rng.uniform(25, W - 25)
        a = rng.uniform(900, 3500)
        fw = fwhm_of(x if axis == 'x' else y)
        sig = fw / 2.3548
        y0, y1, x0, x1 = int(y) - 18, int(y) + 19, int(x) - 18, int(x) + 19
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        img[y0:y1, x0:x1] += a * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sig ** 2))
    img += rng.normal(0, SIGMA, img.shape)
    return np.repeat(img[..., None], 3, axis=2).astype(np.float32)


def test_star_fwhm_recovers_gaussians_and_rejects_empty_stamps():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:60, :60].astype(np.float64)
    for fwhm in (2.8, 3.5, 5.0, 6.5):
        sig = fwhm / 2.3548
        img = 100 + 3000 * np.exp(-((yy - 30.3) ** 2 + (xx - 29.6) ** 2) / (2 * sig ** 2)) + rng.normal(0, 3, (60, 60))
        est = pf.star_fwhm(img, 30.3, 29.6)
        assert est is not None and abs(est - fwhm) / fwhm < 0.05
    assert pf.star_fwhm(np.zeros((60, 60)), 30, 30) is None
    assert pf.star_fwhm(np.ones((60, 60)), 3, 3) is None


def _sharpen(seg, strength):
    """Stand-in for the network: an unsharp mask narrows every star by a fixed ratio."""
    from scipy import ndimage
    seg = np.asarray(seg, np.float32)
    blur = ndimage.gaussian_filter(seg, (1.4, 1.4, 0))
    return (seg + 1.1 * (seg - blur)).astype(np.float32)


def _width_by_strip(img, n=4):
    from src.debayer import luminance
    from src.quality import detect_stars_auto
    lum = luminance(img).astype(np.float32)
    src = detect_stars_auto(lum, SIGMA, background=float(np.median(lum)))
    idx = pf.bright_unsaturated(lum, src, 1500, SIGMA)
    ys = np.asarray(src['ycentroid'], float)[idx]
    xs = np.asarray(src['xcentroid'], float)[idx]
    return pf._strip_widths(lum, xs, ys, np.linspace(0, lum.shape[1], n + 1))


def test_equalise_tilt_brings_wide_strips_to_the_sharp_end_without_overshooting():
    img = _field(lambda x: 4.8 - 2.0 * x / W)                # 4.8 px on the left, 2.8 on the right
    before = _width_by_strip(img)
    out, info = pf.equalise_tilt(img, _sharpen, SIGMA)
    assert info is not None and info['axis'] == 'x'
    assert info['blends'][0] > info['blends'][-1] and max(info['blends']) <= 1.0
    after = _width_by_strip(out)
    assert np.nanmax(before) / np.nanmin(before) > 1.4
    assert np.nanmax(after) / np.nanmin(after) < 0.95 * np.nanmax(before) / np.nanmin(before)  # flatter
    assert np.nanmin(after) > 0.85 * np.nanmin(before)        # nothing sharpened past the sharp end
    assert out.shape == img.shape and np.isfinite(out).all()
    # flux of the (bright) stars is kept: the stub conserves it and the blend is linear
    assert abs(float(out.sum() - img.sum())) / float(img.sum()) < 0.01


def test_identity_deconvolver_changes_nothing():
    img = _field(lambda x: 4.8 - 2.0 * x / W)
    out, info = pf.equalise_tilt(img, lambda seg, k: seg, SIGMA)
    assert info is not None and all(b == 0.0 for b in info['blends'])
    assert np.allclose(out, img, atol=1e-3)


def test_declines_without_a_tilt_and_handles_a_vertical_gradient():
    flat = _field(lambda x: 3.3)
    out, info = pf.equalise_tilt(flat, _sharpen, SIGMA)
    assert info is None and out is flat
    vert = _field(lambda y: 4.8 - 2.0 * y / H, axis='y')
    seen = []

    def spy(seg, strength):
        seen.append(seg.shape)
        return _sharpen(seg, strength)
    out, info = pf.equalise_tilt(vert, spy, SIGMA)
    assert info is not None and info['axis'] == 'y'
    assert seen and all(shape[1] == W for shape in seen)          # strips reach the network as (h, W, 3)
    assert out.shape == vert.shape
