"""Windowed correlation below the coarsest pyramid level (registration.py)."""
import numpy as np
import pytest
import scipy.fft as sfft

from src import registration as rg


def _field(seed=0, shape=(200, 260)):
    rng = np.random.default_rng(seed)
    img = rng.normal(0, 1, shape).astype(np.float32)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    m = min(10, min(shape) // 4)
    for _ in range(40):
        cy, cx = rng.uniform(m, shape[0] - m), rng.uniform(m, shape[1] - m)
        img += (50 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 6.0)).astype(np.float32)
    return img


def _fft_corr_at(ref, img, dy, dx):
    """Value of the zero-padded FFT correlation the pyramid used at lag (dy, dx)."""
    h, w = ref.shape
    ph, pw = sfft.next_fast_len(2 * h), sfft.next_fast_len(2 * w)
    c = sfft.irfft2(sfft.rfft2(ref.astype(np.float64), s=(ph, pw))
                    * np.conj(sfft.rfft2(img.astype(np.float64), s=(ph, pw))), s=(ph, pw))
    return c[dy % ph, dx % pw]


def test_window_values_equal_the_fft_correlation():
    ref, img = _field(1), _field(2)
    c = rg._xcorr_window_numpy(ref, img, 2)
    for i, dy in enumerate(range(-2, 3)):
        for j, dx in enumerate(range(-2, 3)):
            assert c[i, j] == pytest.approx(_fft_corr_at(ref, img, dy, dx), rel=1e-6, abs=1e-3)


@pytest.mark.skipif(not (rg.HAS_NATIVE and hasattr(rg._native, 'xcorr_window')),
                    reason='astro_native xcorr_window not built')
def test_native_xcorr_window_matches_numpy():
    for shape, r in (((200, 260), 2), ((17, 31), 3), ((64, 64), 0)):
        ref, img = _field(3, shape), _field(4, shape)
        a = np.asarray(rg._native.xcorr_window(ref, img, r))
        b = rg._xcorr_window_numpy(ref, img, r)
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-6)


@pytest.mark.parametrize('shift', [(0, 0), (3, -5), (17, 22), (-41, 9)])
def test_pyramid_shift_matches_the_full_fft_path(shift, monkeypatch):
    base = _field(5, (512, 640))
    ref = base
    img = np.roll(base, shift, axis=(0, 1))
    prep = rg.prepare_ref_pyramid(ref)
    got = rg.calculate_shift_pyramid_pref(prep, img)
    monkeypatch.setattr(rg, '_PYRAMID_WINDOW', 0)    # a 1x1 window is all edge: always FFT
    want = rg.calculate_shift_pyramid_pref(prep, img)
    assert got == want


def test_edge_peak_falls_back_to_fft(monkeypatch):
    calls = []
    real = rg.sfft.rfft2
    monkeypatch.setattr(rg.sfft, 'rfft2', lambda *a, **k: calls.append(1) or real(*a, **k))
    monkeypatch.setattr(rg, '_xcorr_window',
                        lambda ref, img, r: np.pad(np.zeros((1, 1)), r, constant_values=-1.0) * 0
                        + np.eye(2 * r + 1)[::-1])            # max on the window edge
    base = _field(6, (256, 320))
    prep = rg.prepare_ref_pyramid(base)
    n0 = len(calls)
    rg.calculate_shift_pyramid_pref(prep, base)
    assert len(calls) - n0 >= len([p for p in prep if p is not None])   # FFT at every level
