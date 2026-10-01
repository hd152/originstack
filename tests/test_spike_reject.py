"""--spike-reject: sharp mosaic spikes go, stars (even undersampled ones) stay.

Also pins the native ``spike_reject_bayer`` kernel bit-for-bit to its numpy
mirror, including ties, NaNs and tiny frames.
"""
import numpy as np
import pytest

from src import debayer as db


def _sky(shape=(200, 240), seed=0):
    rng = np.random.default_rng(seed)
    img = 1000.0 + rng.normal(0, 20.0, shape)
    # a Bayer checkerboard of plane offsets, like a real unequalised mosaic
    img[0::2, 1::2] += 400.0
    img[1::2, 0::2] += 400.0
    img[1::2, 1::2] -= 150.0
    return img.astype(np.float32)


def _add_star(img, y, x, fwhm, peak):
    s = fwhm / 2.3548
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    img += (peak * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * s * s))).astype(np.float32)


@pytest.fixture(params=['native', 'numpy'])
def backend(request, monkeypatch):
    if request.param == 'native':
        if not (db._HAS_NATIVE and hasattr(db._native, 'spike_reject_bayer')):
            pytest.skip('astro_native spike_reject_bayer not built')
    else:
        monkeypatch.setattr(db, '_HAS_NATIVE', False)
    return request.param


def test_single_and_paired_hits_are_removed(backend):
    img = _sky()
    clean = img.copy()
    img[50, 60] += 3000.0                          # single-pixel hit
    img[120, 101] += 2500.0                        # two adjacent pixels (different planes)
    img[120, 102] += 2200.0
    img[150, 30] += 1800.0                         # vertical pair
    img[151, 30] += 1600.0
    out, n = db.remove_spikes_bayer(img)
    assert n == 5
    for y, x in ((50, 60), (120, 101), (120, 102), (150, 30), (151, 30)):
        assert abs(out[y, x] - clean[y, x]) < 120.0      # back to ~sky (a few sigma)
    untouched = np.ones(img.shape, bool)
    untouched[[50, 120, 120, 150, 151], [60, 101, 102, 30, 30]] = False
    assert np.array_equal(out[untouched], img[untouched])


@pytest.mark.parametrize('fwhm', [2.2, 3.0, 4.5, 7.0])
def test_stars_are_kept(backend, fwhm):
    img = _sky(seed=1)
    for k, (y, x) in enumerate([(40.3, 50.7), (100.0, 100.0), (160.6, 180.2), (70.5, 200.5)]):
        _add_star(img, y, x, fwhm, peak=[2000.0, 20000.0, 60000.0, 400.0][k])
    out, n = db.remove_spikes_bayer(img)
    assert n == 0
    assert np.array_equal(out, img)


def test_hot_pixel_star_support_rule_misses_a_pair_that_this_catches():
    """The Bayer hot-pixel step keeps any pixel with a lifted 1-px neighbour (its
    star test), so an adjacent two-pixel hit survives it; the spike step does not."""
    img = _sky(seed=2)
    img[80, 81] += 3000.0
    img[80, 82] += 3000.0
    after_hot = db.remove_hot_pixels_bayer(img)
    assert after_hot[80, 81] > 3000.0
    out, _ = db.remove_spikes_bayer(after_hot)
    assert out[80, 81] < 2000.0 and out[80, 82] < 2000.0


def test_mono_or_rgb_input_is_returned_unchanged():
    rgb = np.ones((10, 10, 3), np.float32)
    out, n = db.remove_spikes_bayer(rgb)
    assert out is rgb and n == 0


@pytest.mark.skipif(not (db._HAS_NATIVE and hasattr(db._native, 'spike_reject_bayer')),
                    reason='astro_native spike_reject_bayer not built')
def test_native_spike_reject_bayer_matches_numpy_bit_for_bit():
    rng = np.random.default_rng(7)
    for t in range(200):
        shape = (int(rng.integers(1, 48)), int(rng.integers(1, 48)))
        r = rng.normal(100, 10, shape).astype(np.float32)
        if t % 3 == 0:
            r = np.round(r / 5) * 5                              # ties in the medians
        for _ in range(int(rng.integers(0, 6))):
            r[rng.integers(0, shape[0]), rng.integers(0, shape[1])] += rng.uniform(50, 5000)
        if t % 17 == 0:
            r[0, 0] = np.nan                                     # plane left alone
        a, na = db._native.spike_reject_bayer(r, 5.0, 5.0, 0.15)
        b, nb = db._spike_reject_bayer_numpy(r, 5.0, 5.0, 0.15)
        assert na == nb
        np.testing.assert_array_equal(a, b)
    img = _sky((300, 400), seed=9)
    img[10, 10] += 5000
    a, na = db._native.spike_reject_bayer(img, 5.0, 5.0, 0.15)
    b, nb = db._spike_reject_bayer_numpy(img, 5.0, 5.0, 0.15)
    assert na == nb == 1
    np.testing.assert_array_equal(a, b)


def test_frame_processor_applies_it_only_when_asked(tmp_path):
    from astropy.io import fits

    from src.frame_processor import _process_single_frame
    img = _sky((128, 160), seed=4)
    img[60, 70] += 8000.0          # a pair: the hot-pixel step's star-support test keeps it
    img[60, 71] += 8000.0
    p = tmp_path / 'Light_001.fits'
    fits.PrimaryHDU(img).writeto(p)
    kw = dict(header={}, masters={}, debayer_method='malvar', white_balance='none',
              skip_quality=True, session_bayer='RGGB')
    off = _process_single_frame(str(p), **kw)
    on = _process_single_frame(str(p), spike_reject=True, **kw)
    assert on['rgb'][60, 70].max() < off['rgb'][60, 70].max() - 1000.0
    assert 'spike_reject' in on['timings']
