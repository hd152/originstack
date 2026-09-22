"""The statistical Bayer hot-pixel fix must not clip stars.

Each colour plane is half resolution, so a star ~4 px wide is ~2 px wide there, and its peak
pixel stands far above its 3x3 plane median: the detector used to replace the peak of every
bright star by that median, in every frame. On a real 114-frame session that made the stack's
stars 18% wider and its noise 6-11% higher than with the step switched off. A flagged pixel is
now kept when an adjacent mosaic pixel (in another colour plane) is elevated too: a hot pixel is
a single-sensor-pixel event, a star lifts its neighbours.
"""
import numpy as np
import pytest

from src import debayer as D

NOISE = 30.0
SKY = 1000.0


def _frame(seed=0, shape=(120, 160), star=(60, 80), amp=1200.0, fwhm=4.4, n_hot=24, hot_amp=2500.0):
    rng = np.random.default_rng(seed)
    img = rng.normal(SKY, NOISE, shape).astype(np.float32)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    sig = fwhm / 2.3548
    img += (amp * np.exp(-((yy - star[0] + 0.3) ** 2 + (xx - star[1] - 0.2) ** 2) / (2 * sig ** 2))).astype(np.float32)
    hot = []
    while len(hot) < n_hot:
        y, x = int(rng.integers(6, shape[0] - 6)), int(rng.integers(6, shape[1] - 6))
        if abs(y - star[0]) > 14 or abs(x - star[1]) > 14:
            img[y, x] += hot_amp
            hot.append((y, x))
    return img, hot


def _star_flux(img, star=(60, 80), r=6):
    y, x = star
    return float((img[y - r:y + r + 1, x - r:x + r + 1] - SKY).sum())


@pytest.fixture(params=['numpy', 'native'])
def backend(request, monkeypatch):
    if request.param == 'numpy':
        monkeypatch.setattr(D, '_HAS_NATIVE', False)
    elif not (D._HAS_NATIVE and 'star_support' in (D._native.hot_pixel_bayer.__text_signature__ or '')):
        pytest.skip('astro_native without the star_support argument')
    return request.param


def test_star_survives_and_hot_pixels_do_not(backend):
    raw, hot = _frame()
    out = D._fix_hot_bayer(raw.copy())
    assert _star_flux(out) == pytest.approx(_star_flux(raw), rel=0.02), 'star flux was clipped'
    assert out[60, 80] > 0.95 * raw[60, 80] or out[59:62, 79:82].max() > 0.95 * raw[59:62, 79:82].max()
    removed = sum(out[y, x] < raw[y, x] - 0.7 * 2500 for y, x in hot)
    assert removed >= 0.9 * len(hot), f'only {removed} of {len(hot)} hot pixels removed'


def test_without_the_test_the_star_core_is_clipped(backend):
    """Documents the bug the test exists for; if this stops failing, the protection test above
    has stopped proving anything."""
    raw, _ = _frame()
    out = D._fix_hot_bayer(raw.copy(), star_support=None)
    assert _star_flux(out) < 0.9 * _star_flux(raw)


def test_a_hot_pixel_touching_a_star_wing_is_still_only_spared_if_neighbours_are_lifted(backend):
    """A hot pixel far from any star is removed however bright the star elsewhere is."""
    raw, hot = _frame(amp=4000.0, n_hot=10, hot_amp=6000.0)
    out = D._fix_hot_bayer(raw.copy())
    assert all(out[y, x] < raw[y, x] - 0.7 * 6000 for y, x in hot)
    assert _star_flux(out) == pytest.approx(_star_flux(raw), rel=0.02)


@pytest.mark.parametrize('shape', [(96, 130), (64, 66), (80, 91)])
def test_native_and_numpy_agree_bit_for_bit(shape, monkeypatch):
    if not (D._HAS_NATIVE and 'star_support' in (D._native.hot_pixel_bayer.__text_signature__ or '')):
        pytest.skip('astro_native without the star_support argument')
    raw, _ = _frame(shape=shape, star=(shape[0] // 2, shape[1] // 2), n_hot=min(6, shape[0] // 3))
    fast = D._fix_hot_bayer(raw.copy())
    monkeypatch.setattr(D, '_HAS_NATIVE', False)
    ref = D._fix_hot_bayer(raw.copy())
    np.testing.assert_array_equal(fast, ref)


def test_flat_or_nan_planes_are_left_alone(backend):
    flat = np.full((32, 32), 500.0, np.float32)
    np.testing.assert_array_equal(D._fix_hot_bayer(flat.copy()), flat)
    nan = flat.copy()
    nan[5, 5] = np.nan
    out = D._fix_hot_bayer(nan.copy())
    np.testing.assert_array_equal(np.nan_to_num(out), np.nan_to_num(nan))
