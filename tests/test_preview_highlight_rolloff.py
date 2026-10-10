"""Preview highlight roll-off for extended bright regions, and the local-contrast
RGB rebuild near zero luminance (both found on a real Orion Nebula stack)."""
import numpy as np

import src.io_fits as iof
from src.denoising import multiscale_local_contrast


def _sky(h=400, w=600, seed=0):
    rng = np.random.default_rng(seed)
    return (500.0 + rng.normal(0, 20.0, (h, w, 3))).astype(np.float32)


def _add_star(img, y, x, peak, fwhm=3.0):
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    s = fwhm / 2.355
    img += (peak * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * s * s)))[..., None].astype(np.float32)


def test_extended_core_keeps_structure():
    img = _sky()
    yy, xx = np.mgrid[:400, :600]
    r = np.hypot(yy - 200, xx - 300)
    # a broad nebula and a bright core with a ripple well above the 99.5% white point
    core = 20000.0 * np.exp(-(r / 60.0) ** 2) + 80000.0 * np.exp(-(r / 25.0) ** 2) * (
        1.0 + 0.3 * np.sin(xx / 3.0))
    img += core[..., None].astype(np.float32)
    out = iof.render_preview_float(img, 'ghs')
    lum = out @ np.array([0.299, 0.587, 0.114])
    inner = r < 15
    assert np.mean(lum[inner] >= 0.999) < 0.5          # not a flat white blob
    assert np.std(lum[inner]) > 0.01                   # the ripple survives


def test_star_cores_do_not_trigger_rolloff():
    img = _sky()
    rng = np.random.default_rng(1)
    for _ in range(300):
        _add_star(img, rng.uniform(5, 395), rng.uniform(5, 595), rng.uniform(2e3, 6e4))
    lum = img.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    med, sig = iof._sky_stats(lum.astype(np.float32))
    white, _ = iof._preview_white(lum, med, sig, 99.5, med)
    assert iof._extended_highlight_top(lum, white) is None


def test_local_contrast_finite_near_zero_luminance():
    # Sky centred on zero (as after DBE): a dark lane where luminance ~0 while
    # red is positive and blue negative, beside bright nebulosity.
    rng = np.random.default_rng(2)
    h, w = 300, 300
    img = rng.normal(0, 5.0, (h, w, 3)).astype(np.float32)
    yy, xx = np.mgrid[:h, :w]
    neb = 3000.0 * np.exp(-((xx - 150) / 50.0) ** 2)
    img += neb[..., None].astype(np.float32)
    lane = (np.abs(xx - 150) < 4)
    img[lane, 0] = 400.0
    img[lane, 2] = -400.0 * 0.299 / 0.114
    img[lane, 1] = 0.0
    out = multiscale_local_contrast(img)
    assert np.all(np.isfinite(out))
    assert out.max() < 10 * img.max()
