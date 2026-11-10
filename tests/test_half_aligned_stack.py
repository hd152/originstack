"""The float16 aligned stack (``frame_store.HalfFloatStack``, ``--aligned-stack half``):
accuracy against float32, RAM vs disk bit-identity, the read paths the combine and the
proper coadd use, and when the pipeline picks it."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.frame_store import FrameStore, HalfFloatStack


def _frames(n=6, H=48, W=64, C=3, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    star = 40000.0 * np.exp(-((yy - 20) ** 2 + (xx - 30) ** 2) / 4.0)
    out = []
    for k in range(n):
        f = (12000.0 + 50.0 * k + star[..., None] + rng.normal(0, 500.0, (H, W, C))).astype(np.float32)
        out.append(f)
    return out


def _store(mode, frames):
    st = FrameStore(mode)
    raw, _ = st.create('test_half_', 'float16', (len(frames),) + frames[0].shape)
    hs = HalfFloatStack(raw)
    for j, f in enumerate(frames):
        hs.write_frame(j, f.copy())
    return st, hs


def test_round_trip_error_is_far_below_the_noise():
    frames = _frames()
    st, hs = _store('ram', frames)
    try:
        for j, f in enumerate(frames):
            back = hs[j]
            assert back.dtype == np.float32 and back.flags.c_contiguous
            err = back.astype(np.float64) - f
            # relative to the distance from the frame's sky level: 2^-11 at most
            dist = np.abs(f - np.median(f[::16, ::16])) + 1.0
            assert np.max(np.abs(err) / dist) <= 2.0 ** -11 + 1e-7
            sky = np.abs(f - np.median(f)) < 2000
            assert np.std(err[sky]) < 0.25           # against 500 ADU of noise
    finally:
        st.cleanup()


def test_ram_and_disk_give_the_same_bits():
    frames = _frames(seed=3)
    s1, ram = _store('ram', frames)
    s2, disk = _store('disk', frames)
    try:
        assert isinstance(disk.raw, np.memmap) and not isinstance(ram.raw, np.memmap)
        for j in range(len(frames)):
            np.testing.assert_array_equal(ram[j], disk[j])
        np.testing.assert_array_equal(ram.read_band(5, 30), disk.read_band(5, 30))
    finally:
        s1.cleanup()
        s2.cleanup()


def test_read_paths_agree():
    frames = _frames(n=5, seed=4)
    st, hs = _store('ram', frames)
    try:
        full = np.stack([hs[j] for j in range(5)])
        np.testing.assert_array_equal(hs.read_band(7, 33), full[:, 7:33])
        np.testing.assert_array_equal(hs.read_band(7, 33, workers=1), full[:, 7:33])
        np.testing.assert_array_equal(hs[1:4], full[1:4])
        np.testing.assert_array_equal(hs[2, 10:20, 5:9], full[2, 10:20, 5:9])
        np.testing.assert_array_equal(hs[-1], full[4])
        lazy = hs.lazy_frame(3)
        np.testing.assert_array_equal(lazy[10:20, 5:9], full[3, 10:20, 5:9])
        np.testing.assert_array_equal(np.asarray(lazy), full[3])
        f32 = hs.to_float32(st)
        np.testing.assert_array_equal(np.asarray(f32), full)
    finally:
        st.cleanup()


def test_nan_inf_and_extreme_ranges_survive():
    f = _frames(n=1)[0]
    f[0, 0, 0] = np.nan
    f[1, 1, 1] = np.inf
    f[2, 2, 2] = 3.0e6                       # above float16's range before scaling
    st, hs = _store('ram', [f])
    try:
        back = hs[0]
        assert np.isnan(back[0, 0, 0]) and np.isinf(back[1, 1, 1])
        assert abs(back[2, 2, 2] - 3.0e6) <= 3.0e6 * 2.0 ** -11
        tiny = np.full_like(f, 1.0e-3)
        tiny[3, 3] = 2.0e-3
        st2, hs2 = _store('ram', [tiny])
        assert abs(float(hs2[0][3, 3, 0]) - 2.0e-3) < 2.0e-3 * 2.0 ** -10
        st2.cleanup()
    finally:
        st.cleanup()


def test_native_fused_combine_on_the_half_stack():
    """The default combine reads the half stack in bands; the result equals combining the
    decoded frames in one call."""
    from src import stacking
    if not stacking._native_usable(np.zeros((1, 1, 1, 3), np.float32)):
        pytest.skip("native module not built")
    from src.stacking import _combine_in_bands, _native
    frames = _frames(n=9, seed=5)
    st, hs = _store('disk', frames)
    try:
        dec = np.ascontiguousarray(np.stack([hs[j] for j in range(len(frames))]))
        q = np.ones((len(frames), 4, 4), np.float32)
        w = np.linspace(0.8, 1.0, len(frames)).astype(np.float32)
        H, W = dec.shape[1:3]

        def fused(block, r0):
            return _native.patch_weighted_sigma_combine_fast(block, q, w, 3.0, 3, True,
                                                             (float(H), float(W), float(r0), 0.0))
        old = stacking._BAND_BYTES
        stacking._BAND_BYTES = 9 * 10 * W * 3 * 4          # forces several bands
        try:
            banded = _combine_in_bands(hs, fused, SimpleNamespace(), 'test')
        finally:
            stacking._BAND_BYTES = old
        np.testing.assert_array_equal(banded, fused(dec, 0))
    finally:
        st.cleanup()


def _star_field(n=10, H=256, W=320, seed=6):
    """Frames with ~40 isolated stars of varying width, so proper coadd measures a PSF."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    pos = [(y, x) for y in range(30, H - 30, 36) for x in range(30, W - 30, 40)]
    amps = rng.uniform(3000, 30000, len(pos))
    out = []
    for k in range(n):
        s = 1.2 + 0.25 * (k % 4)
        img = np.full((H, W), 12000.0, np.float32)
        for (y, x), amp in zip(pos, amps):
            img += amp * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * s * s))
        f = img[..., None] * np.array([0.9, 1.0, 1.1], np.float32) + rng.normal(0, 150.0, (H, W, 3))
        out.append(f.astype(np.float32))
    return out


def test_proper_coadd_reads_the_half_stack_exactly():
    """Proper coadd on the half stack equals proper coadd on the same stack decoded to
    float32, bit for bit: the lazy measuring pass and the combine pass read what a float32
    array of the decoded values holds (and, with ``_PSF_FLOOR``, within a fraction of a
    percent of the noise of the coadd of the original float32 frames)."""
    from src import proper_coadd as pc
    frames = _star_field()
    st, hs = _store('ram', frames)
    try:
        dec = np.stack([hs[j] for j in range(len(frames))])
        ref = dec.mean(0)
        a = pc.proper_coadd(hs, ref, fwhm=3.5, verbose=False)
        b = pc.proper_coadd(dec, ref, fwhm=3.5, verbose=False)
        assert a is not None and b is not None, "the test field must give the coadd a PSF"
        np.testing.assert_array_equal(a, b)
    finally:
        st.cleanup()


def _args(**kw):
    base = dict(aligned_stack='half', stack_method='sigma_clip', local_normalize=False,
                moving_objects=False, moving_objects_stack=False, gpu_coadd=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_when_the_half_stack_is_used():
    from src.stacking import HAS_NATIVE, _half_stack_wanted
    q = [np.ones((2, 2))] * 4
    if not HAS_NATIVE:
        assert not _half_stack_wanted(_args(), q, 4, None)
        return
    assert _half_stack_wanted(_args(), q, 4, None)
    assert _half_stack_wanted(_args(stack_method='winsorized'), q, 4, None)
    assert not _half_stack_wanted(_args(aligned_stack='full'), q, 4, None)
    assert not _half_stack_wanted(_args(stack_method='median'), q, 4, None)
    assert not _half_stack_wanted(_args(), None, 4, None)                  # no patch maps
    assert not _half_stack_wanted(_args(), q, 4, [None] * 4)               # elastic
    for flag in ('local_normalize', 'moving_objects', 'moving_objects_stack', 'gpu_coadd'):
        assert not _half_stack_wanted(_args(**{flag: True}), q, 4, None), flag


def test_flag_default_and_fingerprint():
    from src import cli
    from src.checkpoint import stack_fingerprint
    a = cli.parse_args(['-d', 'x', '-o', 'y.fits'])
    assert a.aligned_stack == 'half'
    b = cli.parse_args(['-d', 'x', '-o', 'y.fits', '--aligned-stack', 'full'])
    # a checkpoint made with one storage must not be reused with the other
    assert stack_fingerprint(a, [])['p23'] != stack_fingerprint(b, [])['p23']


def _coadd_perturbation_rms(floor, monkeypatch):
    """rms change of a wide-PSF (FWHM ~5.9 px) proper coadd when 0.2 ADU of noise is added
    to every frame, in units of the stack's own noise."""
    from src import proper_coadd as pc
    monkeypatch.setattr(pc, '_PSF_FLOOR', floor)
    rng = np.random.default_rng(11)
    H, W = 256, 320
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    pos = [(y, x) for y in range(30, H - 30, 36) for x in range(30, W - 30, 40)]
    amps = rng.uniform(5000, 40000, len(pos))
    frames = []
    for k in range(10):
        s = 2.3 + 0.2 * (k % 3)
        img = np.full((H, W), 12000.0, np.float32)
        for (y, x), amp in zip(pos, amps):
            img += amp * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * s * s))
        frames.append((img[..., None] + rng.normal(0, 150.0, (H, W, 3))).astype(np.float32))
    st = np.stack(frames)
    ref = st.mean(0)
    a = pc.proper_coadd(st, ref, fwhm=6.0, verbose=False)
    b = pc.proper_coadd(st + rng.normal(0, 0.2, st.shape).astype(np.float32), ref, fwhm=6.0, verbose=False)
    assert a is not None and b is not None
    sky = a[20:60, 20:60, 1]
    noise = np.std(sky - np.roll(sky, 1, 1)) / np.sqrt(2)
    return float(np.std(b - a) / noise)


def test_proper_coadd_is_stable_to_sub_noise_input_changes(monkeypatch):
    """Wide stars put the PSF spectrum below float32 rounding at high frequencies; without
    ``_PSF_FLOOR`` the coadd weighted its frames there by rounding noise, so a 0.2 ADU change
    of the input moved the output by a fifth of its noise (real Whirlpool, 60 frames)."""
    with_floor = _coadd_perturbation_rms(1e-5, monkeypatch)
    without = _coadd_perturbation_rms(0.0, monkeypatch)
    assert with_floor < 0.01
    assert without > 10 * with_floor          # the test can see the problem it guards against
