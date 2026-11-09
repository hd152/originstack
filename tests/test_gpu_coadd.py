"""GPU proper coadd (src/gpu_coadd.rs, src/proper_coadd._combine_gpu, --gpu-coadd).

The kernel tests need an astro_native built with the ``gpu`` feature (default since 0.66.0) and a usable adapter and skip
otherwise; the fallback, option and fingerprint tests run everywhere, so the default CI covers the
Python side of the feature without a GPU."""
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from src import proper_coadd as pc
from src.models import RunCancelled

_NAT = pc._native
_HAS_GPU_BUILD = _NAT is not None and hasattr(_NAT, 'gpu_coadd_begin')
_GPU = _HAS_GPU_BUILD and bool(_NAT.gpu_available())
needs_gpu = pytest.mark.skipif(not _GPU, reason='astro_native has no gpu feature or no GPU adapter')


# --- helpers ---------------------------------------------------------------------------------

def _moffat_stamp(fwhm, r=pc._PSF_R, beta=3.0):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    a = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    k = (1 + (xx ** 2 + yy ** 2) / a ** 2) ** -beta
    return (k / k.sum()).astype(np.float32)


def _reference_accumulate(aligned, ref_sig, frames, stamps, F, sky, s2, wn, wf, ph, pw, pad=pc._PAD,
                          k=pc._REJECT_K, frac=pc._SIGNAL_FRAC, f64=True):
    """irfft2(num / sqrt(den)) cropped, with plain numpy / scipy (float64 transforms by default)."""
    import scipy.fft as sfft
    N, H, W, C = aligned.shape
    r = (stamps[0].shape[0] - 1) // 2
    dt = np.float64 if f64 else np.float32
    num = np.zeros((C, ph, pw // 2 + 1), np.complex128)
    den = np.zeros((C, ph, pw // 2 + 1))
    for j in frames:
        P = np.zeros((ph, pw), dt)
        kk = stamps[j].astype(dt)
        P[:r + 1, :r + 1] = kk[r:, r:]
        P[:r + 1, -r:] = kk[r:, :r]
        P[-r:, :r + 1] = kk[:r, r:]
        P[-r:, -r:] = kk[:r, :r]
        Ph = sfft.rfft2(P)
        for c in range(C):
            e = ref_sig[c] * np.float32(F[j])
            v = aligned[j][..., c] - np.float32(sky[j][c])
            t = np.maximum(e, 0) * np.float32(frac)
            t = t * t + np.float32(s2[j][c])
            bad = np.abs(v - e) > np.float32(k) * np.sqrt(t)
            plane = np.zeros((ph, pw), dt)
            plane[pad:pad + H, pad:pad + W] = np.where(bad, e, v)
            S = sfft.rfft2(plane)
            num[c] += wn[j][c] * (S * np.conj(Ph))
            den[c] += wf[j][c] * np.abs(Ph) ** 2
    out = np.empty((C, H, W))
    for c in range(C):
        R = num[c] / np.sqrt(np.maximum(den[c], 1e-300))
        out[c] = sfft.irfft2(R, s=(ph, pw))[pad:pad + H, pad:pad + W]
    return out


def _small_problem(seed=5, K=5, H=50, W=70, delta=False):
    rng = np.random.default_rng(seed)
    ref_sig = (rng.normal(0, 5, (3, H, W)) + 50).astype(np.float32)
    aligned = np.empty((K, H, W, 3), np.float32)
    stamps, F, sky, s2, wn, wf = [], [], [], [], [], []
    for j in range(K):
        fr = (1000 + rng.normal(0, 30, (H, W, 3))).astype(np.float32)
        fr[::13, ::11] += 3000                               # outliers the rejection must act on
        aligned[j] = fr
        if delta:
            st = np.zeros((2 * pc._PSF_R + 1,) * 2, np.float32)
            st[pc._PSF_R, pc._PSF_R] = 1.0
        else:
            st = _moffat_stamp(rng.uniform(2.5, 4.5))
        stamps.append(st)
        f = float(rng.uniform(0.7, 1.2))
        F.append(f)
        sky.append([1000.0, 1001.0, 999.0])
        sg = [30.0, 31.0, 29.0]
        s2.append([v * v for v in sg])
        wn.append([f / v ** 2 for v in sg])
        wf.append([f * f / v ** 2 for v in sg])
    return aligned, ref_sig, stamps, F, sky, s2, wn, wf


def _gpu_accumulate(aligned, ref_sig, stamps, F, sky, s2, wn, wf, ph, pw):
    K = aligned.shape[0]
    f32 = lambda x: [float(np.float32(v)) for v in x]            # noqa: E731
    _NAT.gpu_coadd_begin(ref_sig, float(pc._REJECT_K), float(pc._SIGNAL_FRAC), pc._PAD, pc._PSF_R, ph, pw)
    _NAT.gpu_coadd_add(aligned, list(range(K)), f32(F), [f32(s) for s in sky], [f32(s) for s in s2],
                       wn, wf, np.ascontiguousarray(np.stack(stamps)))
    return np.asarray(_NAT.gpu_coadd_finish())


def _rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


# --- kernels (need a GPU) ----------------------------------------------------------------------

@needs_gpu
def test_gpu_reports_an_adapter():
    assert _NAT.gpu_available() is True
    assert isinstance(_NAT.gpu_info(), str) and _NAT.gpu_info()


@needs_gpu
@pytest.mark.parametrize('shape', [(32, 64), (64, 32), (128, 128), (512, 1024)])
@pytest.mark.parametrize('inverse', [False, True])
def test_fft2_matches_numpy(shape, inverse):
    rng = np.random.default_rng(3)
    z = (rng.normal(0, 1000, shape) + 1j * rng.normal(0, 1000, shape)).astype(np.complex64)
    zin = np.ascontiguousarray(z.view(np.float32).reshape(*shape, 2))
    g = _NAT.gpu_fft2(zin, inverse)
    g = g[..., 0] + 1j * g[..., 1]
    ref = np.fft.ifft2(z.astype(np.complex128)) if inverse else np.fft.fft2(z.astype(np.complex128))
    assert _rms(np.abs(g - ref)) / _rms(np.abs(ref)) < 1e-5


@needs_gpu
def test_fft2_rejects_bad_planes():
    with pytest.raises(ValueError):
        _NAT.gpu_fft2(np.zeros((48, 64, 2), np.float32), False)         # not a power of two
    with pytest.raises(ValueError):
        _NAT.gpu_fft2(np.zeros((32, 64, 3), np.float32), False)         # not complex pairs


@needs_gpu
@pytest.mark.parametrize('delta', [False, True])
def test_coadd_matches_float64_reference(delta):
    """A real PSF and a delta function: the PSF-precision regression. Packing the PSF with a
    data plane in one complex FFT left 25-64% noise-like error with a real PSF."""
    prob = _small_problem(delta=delta)
    aligned, ref_sig, stamps, F, sky, s2, wn, wf = prob
    ph, pw = 128, 128
    gpu = _gpu_accumulate(*prob, ph, pw)
    truth = _reference_accumulate(aligned, ref_sig, range(len(F)), stamps, F, sky, s2, wn, wf, ph, pw)
    cpu32 = _reference_accumulate(aligned, ref_sig, range(len(F)), stamps, F, sky, s2, wn, wf, ph, pw, f64=False)
    scale = _rms(truth)
    assert _rms(gpu - truth) / scale < 1e-4
    # no worse than the CPU path's own float32 arithmetic, within a small factor
    assert _rms(gpu - truth) <= 5 * _rms(cpu32 - truth) + 1e-5 * scale


@needs_gpu
def test_coadd_non_square_planes_and_a_frame_subset():
    prob = _small_problem(K=4, H=40, W=100)
    aligned, ref_sig, stamps, F, sky, s2, wn, wf = prob
    gpu = _gpu_accumulate(*prob, 128, 256)
    truth = _reference_accumulate(aligned, ref_sig, range(4), stamps, F, sky, s2, wn, wf, 128, 256)
    assert _rms(gpu - truth) / _rms(truth) < 1e-4


@needs_gpu
def test_session_errors_leave_nothing_behind(monkeypatch):
    aligned, ref_sig, *_ = _small_problem()
    # a plane too small for frame + pad
    with pytest.raises(ValueError):
        _NAT.gpu_coadd_begin(ref_sig, 5.0, 0.15, pc._PAD, pc._PSF_R, 64, 64)
    with pytest.raises(ValueError):                      # ... and no session survived it
        _NAT.gpu_coadd_finish()
    # 2-channel data
    with pytest.raises(ValueError):
        _NAT.gpu_coadd_begin(ref_sig[:2], 5.0, 0.15, pc._PAD, pc._PSF_R, 128, 128)
    # a begin / abort / finish sequence
    _NAT.gpu_coadd_begin(ref_sig, 5.0, 0.15, pc._PAD, pc._PSF_R, 128, 128)
    _NAT.gpu_coadd_abort()
    with pytest.raises(ValueError):
        _NAT.gpu_coadd_finish()
    _NAT.gpu_coadd_abort()                               # harmless when there is no session


@needs_gpu
def test_over_budget_fails_cleanly_before_allocating():
    code = textwrap.dedent('''
        import numpy as np, astro_native as n
        ref = np.zeros((3, 50, 70), np.float32)
        try:
            n.gpu_coadd_begin(ref, 5.0, 0.15, 32, 24, 128, 128)
        except RuntimeError as e:
            print("RuntimeError", "budget" in str(e))
        try:
            n.gpu_coadd_finish()
        except ValueError:
            print("no session")
    ''')
    import os
    env = dict(os.environ, ORIGINSTACK_GPU_MAX_MB='1')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env, timeout=120)
    assert 'RuntimeError True' in out.stdout and 'no session' in out.stdout, out.stdout + out.stderr


@pytest.mark.skipif(not _HAS_GPU_BUILD, reason='astro_native has no gpu feature')
def test_gpu_can_be_disabled_for_a_process():
    code = 'import astro_native as n; print(n.gpu_available(), n.gpu_info())'
    import os
    env = dict(os.environ, ORIGINSTACK_GPU_DISABLE='1')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env, timeout=120)
    assert out.stdout.strip() == 'False None', out.stdout + out.stderr


# --- proper_coadd end to end (needs a GPU) -----------------------------------------------------

def _field():
    """The synthetic star field of tests/test_proper_coadd.py (240 x 300: not powers of two, so the
    GPU planes are padded to 512 x 512); returns the aligned stack and its mean as the reference."""
    rng = np.random.default_rng(1)
    H, W, N = 240, 300, 24
    yy, xx = np.mgrid[:H, :W].astype(float)
    ys, xs = rng.uniform(20, H - 20, 70), rng.uniform(20, W - 20, 70)
    fl = rng.lognormal(9, 0.8, 70)

    def render(sig):
        p = pc.moffat_params(sig, sig, 0.0, 3.0)
        norm = pc._moffat_grid(*np.mgrid[-60:61, -60:61].astype(float), p).sum()
        img = np.zeros((H, W))
        for y, x, f in zip(ys, xs, fl):
            sl = (slice(int(y) - 25, int(y) + 26), slice(int(x) - 25, int(x) + 26))
            img[sl] += f * pc._moffat_grid(yy[sl] - y, xx[sl] - x, p) / norm
        return img
    sig = rng.uniform(2.5, 5.0, N)
    Fs = rng.uniform(0.8, 1.1, N)
    fr = np.stack([F * render(s) + 200 + rng.normal(0, 6.0, (H, W)) for s, F in zip(sig, Fs)])
    A = np.ascontiguousarray(fr.astype(np.float32)[..., None].repeat(3, -1))
    A[5, 100, 100, :] += 5000.0                                     # a cosmic ray
    ref = A.mean(0)
    ref[100, 100] = np.median(A[:, 100, 100], 0)                    # clipped there, as sigma-clip would be
    return A, ref


@needs_gpu
def test_proper_coadd_gpu_matches_cpu():
    A, ref = _field()
    cpu = pc.proper_coadd(A, ref, fwhm=5.0, verbose=False, stats={})
    gstats = {}
    gpu = pc.proper_coadd(A, ref, fwhm=5.0, verbose=False, stats=gstats, gpu=True)
    assert cpu is not None and gpu is not None and gpu.shape == cpu.shape and gpu.dtype == np.float32
    noise = 1.4826 * np.median(np.abs(cpu[..., 1] - np.median(cpu[..., 1])))
    assert _rms(gpu - cpu) < 1e-3 * noise
    assert float(np.abs(gpu - cpu).max()) < 0.05 * noise
    assert gstats['frames'] == A.shape[0]
    assert gstats['poisson_factor']
    assert gpu[100, 100, 1] < np.median(gpu[..., 1]) + 30                 # the cosmic ray is gone too


# --- fallback and options (run everywhere) ------------------------------------------------------

class _FakeNative:
    """The real astro_native with the GPU entry points replaced: records calls, fails where told."""

    def __init__(self, real, fail_at=None, cancel=None):
        self._real, self.fail_at, self.cancel, self.calls = real, fail_at, cancel, []

    def __getattr__(self, name):
        return getattr(self._real, name)

    gpu_available = staticmethod(lambda: True)
    gpu_info = staticmethod(lambda: 'fake adapter')

    def gpu_coadd_begin(self, *a):
        self.calls.append('begin')
        if self.fail_at == 'begin':
            raise RuntimeError('boom at begin')

    def gpu_coadd_add(self, *a):
        self.calls.append('add')
        if self.cancel is not None:
            self.cancel.set()                     # the user cancels while the device is busy
        if self.fail_at == 'add':
            raise RuntimeError('boom at add')

    def gpu_coadd_finish(self):
        self.calls.append('finish')
        raise RuntimeError('boom at finish')

    def gpu_coadd_abort(self):
        self.calls.append('abort')


@pytest.mark.skipif(pc._native is None, reason='astro_native is not built')
@pytest.mark.parametrize('fail_at', ['begin', 'add', 'finish'])
def test_a_gpu_failure_falls_back_to_the_cpu_result(monkeypatch, fail_at):
    A, ref = _field()
    cpu = pc.proper_coadd(A, ref, fwhm=5.0, verbose=False)
    fake = _FakeNative(pc._native, fail_at)
    monkeypatch.setattr(pc, '_native', fake)
    out = pc.proper_coadd(A, ref, fwhm=5.0, verbose=False, gpu=True)
    assert out is not None and np.array_equal(out, cpu)             # the CPU path ran, unchanged
    assert fake.calls[0] == 'begin' and fake.calls[-1] == 'abort'   # the session was released


@pytest.mark.skipif(pc._native is None, reason='astro_native is not built')
def test_cancellation_propagates_and_releases_the_session(monkeypatch):
    import threading
    A, ref = _field()
    ev = threading.Event()
    fake = _FakeNative(pc._native, None, cancel=ev)
    monkeypatch.setattr(pc, '_native', fake)
    with pytest.raises(RunCancelled):
        pc.proper_coadd(A, ref, fwhm=5.0, verbose=False, gpu=True, cancel_event=ev)
    assert 'abort' in fake.calls


@pytest.fixture(autouse=True)
def _allow_integrated(monkeypatch):
    """The kernel tests must run on a machine whose only GPU is integrated."""
    monkeypatch.setenv('ORIGINSTACK_GPU_ALLOW_INTEGRATED', '1')


def test_integrated_gpu_is_declined_unless_asked_for(monkeypatch):
    A = np.zeros((3, 40, 50, 3), np.float32)
    info = {'v': 'AMD Radeon(TM) Graphics (Vulkan, IntegratedGpu)'}
    fake = type('N', (), {'gpu_coadd_begin': None, 'gpu_available': staticmethod(lambda: True),
                          'gpu_info': staticmethod(lambda: info['v'])})()
    monkeypatch.setattr(pc, '_native', fake)
    monkeypatch.setenv('ORIGINSTACK_GPU_ALLOW_INTEGRATED', '0')
    monkeypatch.delenv('ORIGINSTACK_GPU_ADAPTER', raising=False)
    assert 'integrated' in pc._gpu_unusable(A, 3, 40, 50)
    monkeypatch.setenv('ORIGINSTACK_GPU_ALLOW_INTEGRATED', '1')
    assert pc._gpu_unusable(A, 3, 40, 50) is None
    monkeypatch.setenv('ORIGINSTACK_GPU_ALLOW_INTEGRATED', '0')
    monkeypatch.setenv('ORIGINSTACK_GPU_ADAPTER', 'AMD')          # naming the adapter opts in
    assert pc._gpu_unusable(A, 3, 40, 50) is None
    monkeypatch.delenv('ORIGINSTACK_GPU_ADAPTER')
    info['v'] = 'NVIDIA GeForce RTX 3050 Laptop GPU (Vulkan, DiscreteGpu)'
    assert pc._gpu_unusable(A, 3, 40, 50) is None


def test_unusable_stacks_are_declined_with_a_reason(monkeypatch):
    A = np.zeros((3, 40, 50, 3), np.float32)
    monkeypatch.setattr(pc, '_native', None)
    assert 'no GPU support' in pc._gpu_unusable(A, 3, 40, 50)
    fake = type('N', (), {'gpu_coadd_begin': None, 'gpu_available': staticmethod(lambda: False)})()
    monkeypatch.setattr(pc, '_native', fake)
    assert 'adapter' in pc._gpu_unusable(A, 3, 40, 50)
    fake.gpu_available = staticmethod(lambda: True)
    assert '3-channel' in pc._gpu_unusable(A, 1, 40, 50)
    assert 'float32' in pc._gpu_unusable(A.astype(np.float64), 3, 40, 50)
    assert 'larger' in pc._gpu_unusable(A, 3, 5000, 50)
    assert pc._gpu_unusable(A, 3, 40, 50) is None
    assert pc._gpu_plane(1990) == 2048 and pc._gpu_plane(2945) == 4096 and pc._gpu_plane(1) == 64


def test_the_option_is_off_by_default_and_part_of_the_checkpoint_fingerprint():
    from src import checkpoint as ck
    from src.cli import parse_args
    assert parse_args(['-d', 'x']).gpu_coadd is False
    assert parse_args(['-d', 'x', '--gpu-coadd']).gpu_coadd is True
    a = ck.stack_fingerprint(parse_args(['-d', 'x']), [])
    b = ck.stack_fingerprint(parse_args(['-d', 'x', '--gpu-coadd']), [])
    assert a['p23'] != b['p23'] and 'gpu_coadd' in b['p23']          # a stale CPU checkpoint is not reused
    assert a['p1'] == b['p1']
