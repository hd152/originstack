"""multiscale_local_contrast must not carve a dark moat round a saturated star, and must leave
frames without such a star exactly as they were (src/denoising.py, _LC_TOWER)."""
import numpy as np
import pytest

from src import denoising as dn


def _scene(peak, n=400, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n].astype(np.float64)
    sky = 20.0 + rng.normal(0, 2.0, (n, n))
    neb = (60.0 * np.exp(-((yy - 120) ** 2 + (xx - 300) ** 2) / (2 * 60.0 ** 2))
           + 40.0 * np.exp(-((yy - 300) ** 2 + (xx - 120) ** 2) / (2 * 70.0 ** 2)))
    r = np.hypot(yy - 200, xx - 200)
    star = peak / (1 + (r / 2.2) ** 2) ** 1.6                  # Moffat with extended wings
    lum = sky + neb + star
    return np.stack([lum, lum * 0.95, lum * 0.9], -1).astype(np.float32), r


def _ratios(img, out, r, bands):
    return [float(out[..., 1][(r >= a) & (r < b)].mean() / img[..., 1][(r >= a) & (r < b)].mean())
            for a, b in bands]


def test_saturated_star_leaves_no_moat():
    img, r = _scene(60000.0)
    out = dn.multiscale_local_contrast(img, strength=0.7, scale_fwhm=2.8)
    rat = _ratios(img, out, r, [(12, 16), (16, 22), (22, 30), (30, 45)])
    assert min(rat[:3]) > 0.99          # the wings are not eaten (0.89-0.95 before the fix)
    assert rat[3] > 0.95


def test_the_scene_does_reproduce_the_moat_without_the_fade(monkeypatch):
    monkeypatch.setattr(dn, '_LC_TOWER', 1e30)
    img, r = _scene(60000.0)
    out = dn.multiscale_local_contrast(img, strength=0.7, scale_fwhm=2.8)
    assert _ratios(img, out, r, [(22, 30)])[0] < 0.95


def test_frame_without_a_towering_core_is_bit_identical(monkeypatch):
    img, _ = _scene(300.0)              # a bright star, but nothing 4x the above-sky 99th percentile
    on = dn.multiscale_local_contrast(img, strength=0.7, scale_fwhm=2.8)
    monkeypatch.setattr(dn, '_LC_TOWER', 1e30)
    off = dn.multiscale_local_contrast(img, strength=0.7, scale_fwhm=2.8)
    np.testing.assert_array_equal(on, off)


def test_fade_is_local_far_field_keeps_its_enhancement(monkeypatch):
    img, r = _scene(60000.0)
    on = dn.multiscale_local_contrast(img, strength=0.7, scale_fwhm=2.8)
    monkeypatch.setattr(dn, '_LC_TOWER', 1e30)
    off = dn.multiscale_local_contrast(img, strength=0.7, scale_fwhm=2.8)
    far = r > 120                        # the two nebula blobs sit out there
    assert np.abs(on[far] - off[far]).max() < 0.02 * np.abs(off[far]).max()
