"""The luminance-based RGB hot-pixel pass is skipped once hot pixels were fixed on the
mosaic: on undersampled stars it flattened the peak of nearly every bright star."""
import numpy as np

import src.frame_processor as FP


def _mosaic(seed=0, shape=(96, 128)):
    rng = np.random.default_rng(seed)
    m = rng.normal(1000, 20, shape).astype(np.float32)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    for y, x in ((30.3, 40.6), (60.7, 90.2), (20.1, 100.5)):      # FWHM ~2.4 px stars
        m += (30000 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.0 ** 2))).astype(np.float32)
    return m


def test_rgb_pass_skipped_for_a_mosaic(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("RGB hot-pixel pass ran after the mosaic pass")
    monkeypatch.setattr(FP, 'remove_hot_pixels_rgb_with_lum', boom)
    res = FP._process_single_frame('x.fits', {}, {}, 'rcd', 'none', session_bayer='RGGB',
                                   preloaded_data=(_mosaic(), {}), skip_quality=True)
    assert res['error'] is None
    rgb = res['rgb']
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    np.testing.assert_allclose(res['lum'], lum, rtol=1e-6)
    assert lum.max() > 15000                         # star peaks intact


def test_rgb_pass_still_runs_for_pre_debayered_input(monkeypatch):
    calls = []
    real = FP.remove_hot_pixels_rgb_with_lum
    monkeypatch.setattr(FP, 'remove_hot_pixels_rgb_with_lum',
                        lambda rgb, **k: calls.append(1) or real(rgb, **k))
    rgb = np.repeat(_mosaic()[..., None], 3, -1)
    res = FP._process_single_frame('x.fits', {}, {}, 'rcd', 'none',
                                   preloaded_data=(rgb, {}), skip_quality=True)
    assert res['error'] is None and calls
