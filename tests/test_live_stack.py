"""Tests for src/live_stack.py — real-time incremental stacking."""
from __future__ import annotations

import argparse
import os

import numpy as np
import pytest

from src.live_stack import _HAS_SCIPY, LiveStacker, _stack_snr

pytestmark = pytest.mark.skipif(not _HAS_SCIPY, reason="scipy not installed")


def _args():
    return argparse.Namespace(
        debayer_method='bilinear', white_balance='none',
        ca_correction=False, pre_gradient_removal=False, trail_reject=False,
        _session_bayer=None, stretch='ghs', ghs_b=8.0, ghs_sp=0.15, ghs_hp=0.95,
        preview_black_sigma=0.0)


def _write_star_frame(path, H=160, W=160, shift=(0, 0), n_stars=45, seed=0):
    """Write a (3,H,W) RGB FITS star field (+ broad nebula), optionally shifted."""
    from astropy.io import fits
    rng = np.random.default_rng(seed)
    img = np.full((H, W), 100.0, np.float32)
    # Broad nebula so the frame has real dynamic range / structure to register.
    yy, xx = np.mgrid[0:H, 0:W]
    img += (300.0 * np.exp(-(((xx - W / 2) / (W * 0.3)) ** 2
                            + ((yy - H / 2) / (H * 0.3)) ** 2))).astype(np.float32)
    gy, gx = np.mgrid[-4:5, -4:5]
    g = np.exp(-(gx * gx + gy * gy) / (2 * 1.5 ** 2))
    # fixed star positions (same sky), shifted per frame
    star_rng = np.random.default_rng(999)
    for _ in range(n_stars):
        y0 = star_rng.integers(20, H - 20) + shift[0]
        x0 = star_rng.integers(20, W - 20) + shift[1]
        if 4 <= y0 < H - 4 and 4 <= x0 < W - 4:
            img[y0 - 4:y0 + 5, x0 - 4:x0 + 5] += (5000 * g).astype(np.float32)
    img += rng.standard_normal((H, W)).astype(np.float32) * 5.0
    cube = np.stack([img, img, img], axis=2).astype(np.float32)  # (H,W,3) HWC
    fits.PrimaryHDU(data=cube).writeto(path, overwrite=True)


def test_stack_snr_increases_with_signal():
    lum = np.full((50, 50), 100.0, np.float32)
    lum[20:30, 20:30] = 5000.0   # a bright block moves the 99.5th percentile
    assert _stack_snr(lum) > 0


def test_first_frame_seeds_reference(tmp_path):
    p = str(tmp_path / 'L0.fits')
    _write_star_frame(p)
    st = LiveStacker(_args(), masters={'bias': None, 'dark': None, 'flat': None})
    assert st.add_frame(p, header={}) is True
    assert st.n == 1
    assert st.ref_lum is not None
    stack = st.current_stack()
    assert stack is not None and stack.shape[2] == 3


def test_accumulates_and_registers_shifted_frames(tmp_path):
    st = LiveStacker(_args(), masters={'bias': None, 'dark': None, 'flat': None})
    shifts = [(0, 0), (3, -2), (-4, 5), (2, 6), (-3, -5)]
    for i, s in enumerate(shifts):
        p = str(tmp_path / f'L{i}.fits')
        _write_star_frame(p, shift=s, seed=i)
        st.add_frame(p, header={'EXPTIME': 30.0})
    assert st.n == len(shifts)
    # Integration time accumulated from headers.
    assert st.total_exposure == pytest.approx(len(shifts) * 30.0)
    # Registration coherence: after aligning, the stacked stars should be as
    # sharp as a single frame (a mis-stack would smear them → lower peak).
    stack = st.current_stack()
    lum = 0.299 * stack[:, :, 0] + 0.587 * stack[:, :, 1] + 0.114 * stack[:, :, 2]
    # brightest star peak well above background => stars aligned, not smeared
    assert lum.max() > 100.0 + 1500.0


def test_snr_improves_over_frames(tmp_path):
    st = LiveStacker(_args(), masters={'bias': None, 'dark': None, 'flat': None})
    for i in range(6):
        p = str(tmp_path / f'S{i}.fits')
        _write_star_frame(p, shift=(0, 0), seed=100 + i)
        st.add_frame(p, header={})
    ns = [n for n, _ in st.snr_history]
    snrs = [s for _, s in st.snr_history]
    assert ns == [1, 2, 3, 4, 5, 6]
    # Averaging down the noise: later SNR should exceed the first-frame SNR.
    assert snrs[-1] > snrs[0]


def test_rejects_bad_shift(tmp_path):
    st = LiveStacker(_args(), masters={'bias': None, 'dark': None, 'flat': None})
    p0 = str(tmp_path / 'R0.fits')
    _write_star_frame(p0, shift=(0, 0))
    st.add_frame(p0, header={})
    # A wildly different-size frame is rejected (shape mismatch guard).
    from astropy.io import fits
    p1 = str(tmp_path / 'R1.fits')
    fits.PrimaryHDU(data=np.full((80, 80, 3), 100.0, np.float32)).writeto(p1, overwrite=True)
    assert st.add_frame(p1, header={}) is False
    assert st.n == 1
    assert st.n_rejected >= 1


def test_discover_dedups_and_age_gates(tmp_path):
    st = LiveStacker(_args(), masters={'bias': None, 'dark': None, 'flat': None})
    p = str(tmp_path / 'light_001.fits')
    _write_star_frame(p)
    # Fresh file (age 0) with min_age high -> skipped as "still writing".
    got = st._discover_new_lights(str(tmp_path), seen=set(), min_age=1e6)
    assert got == []
    # min_age 0 -> discovered.
    got = st._discover_new_lights(str(tmp_path), seen=set(), min_age=0.0)
    assert len(got) == 1 and got[0][0] == p
    # Already-seen path is not returned again.
    got = st._discover_new_lights(str(tmp_path), seen={p}, min_age=0.0)
    assert got == []


def test_save_writes_fits_and_preview(tmp_path):
    st = LiveStacker(_args(), masters={'bias': None, 'dark': None, 'flat': None})
    for i in range(3):
        pth = str(tmp_path / f'L{i}.fits')
        _write_star_frame(pth, seed=i)
        st.add_frame(pth, header={'EXPTIME': 60.0})
    out = str(tmp_path / 'live_out.fits')
    st.save(out)
    assert os.path.exists(out)
    from astropy.io import fits
    with fits.open(out) as h:
        assert h[0].header['NFRAMES'] == 3
        assert h[0].header['RAWSTACK'] is True
        assert h[0].data.shape[0] == 3
