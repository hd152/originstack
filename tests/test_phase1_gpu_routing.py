"""--use-gpu: Phase 1 goes to the CPU process pool unless VRAM fits a GPU worker per core.

Measured on a real 4 GB card: the VRAM-capped GPU thread path (7 workers) ran
54 s against ~37 s for all 16 CPU workers, because only calibration, the
hot-pixel map and white balance dispatch to the GPU.
"""
import argparse

import pytest

from src import frame_processor as fp


class _FakeGpu:
    def __init__(self, active, workers):
        self.active = active
        self._workers = workers

    def max_gpu_workers(self, per_worker_mb, reserve_mb=512.0):
        return self._workers


def _args(**kw):
    base = dict(gpu_phase1='auto', parallel=0)
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def cores(monkeypatch):
    monkeypatch.setattr(fp.os, 'cpu_count', lambda: 16)


@pytest.mark.parametrize('workers,mode,parallel,expect', [
    (7, 'auto', 0, False),      # 4 GB card: VRAM-capped -> CPU pool
    (16, 'auto', 0, True),      # a card that fits one worker per core keeps the GPU path
    (40, 'auto', 0, True),
    (7, 'on', 0, True),         # forced
    (40, 'off', 0, False),
    (7, 'auto', 1, True),       # sequential: one GPU-calibrated thread beats one CPU thread
])
def test_decision(monkeypatch, cores, workers, mode, parallel, expect):
    monkeypatch.setattr(fp, 'get_gpu', lambda: _FakeGpu(True, workers))
    args = _args(gpu_phase1=mode, parallel=parallel)
    assert fp.phase1_uses_gpu(args) is expect
    assert args._phase1_gpu is expect


def test_no_gpu_means_no_gpu_path(monkeypatch, cores):
    monkeypatch.setattr(fp, 'get_gpu', lambda: _FakeGpu(False, 99))
    args = _args(gpu_phase1='on')
    assert fp.phase1_uses_gpu(args) is False


def test_decision_is_remembered_for_the_reload_and_cfa_probe(monkeypatch, cores):
    gpu = _FakeGpu(True, 7)
    monkeypatch.setattr(fp, 'get_gpu', lambda: gpu)
    args = _args()
    assert fp.phase1_uses_gpu(args) is False
    gpu._workers = 64                       # VRAM freed later in the run
    assert fp.phase1_uses_gpu(args) is False


def test_cli_flag_parses():
    from src.cli import parse_args
    assert parse_args(['-d', 'x']).gpu_phase1 == 'auto'
    assert parse_args(['-d', 'x', '--gpu-phase1', 'off']).gpu_phase1 == 'off'


def test_cfa_probe_calibrates_on_the_cpu(monkeypatch):
    """The probe measures what the CPU pool workers will debayer, so it must not
    take the GPU calibration path in the main process (which returns a cupy
    array the probe cannot use, silently disabling session CFA)."""
    seen = []
    monkeypatch.setattr(fp, '_process_single_frame',
                        lambda *a, **k: seen.append(k.get('allow_gpu')) or {'error': 'x'})
    monkeypatch.setattr(fp, 'phase1_uses_gpu', lambda a: False)
    args = argparse.Namespace(session_cfa_eq=True, debayer_method='malvar',
                              white_balance='none', _session_bayer='RGGB')
    frames = [argparse.Namespace(path=f'f{i}.fits') for i in range(20)]
    fp._measure_session_cfa(frames, {}, args)
    assert seen and all(v is False for v in seen)


def test_session_ca_probe_survives_a_device_array_debayer(monkeypatch):
    # With --use-gpu the main process's debayer returns a cupy array; the probe
    # must bring it to host, or every sample is lost and each Phase 1 worker
    # measures CA per frame (3.3 s/frame on a real session).
    import argparse

    import numpy as np

    import src.frame_processor as fp
    from src.models import FrameInfo

    class Device:                       # stands in for a cupy array
        def __init__(self, a):
            self.a = a

        def __array__(self, *a, **k):
            raise TypeError('Implicit conversion to a NumPy array is not allowed')

    class FakeGpu:
        active = True

        def to_host(self, x):
            return x.a if isinstance(x, Device) else x

    raw = np.random.default_rng(0).normal(100, 5, (64, 64)).astype(np.float32)
    monkeypatch.setattr(fp, 'get_gpu', lambda: FakeGpu())
    monkeypatch.setattr(fp, 'load_frame', lambda p: (raw, {'BAYERPAT': 'RGGB'}))
    monkeypatch.setattr(fp, 'green_equalize', lambda d, pattern='RGGB': d)
    monkeypatch.setattr(fp, 'debayer', lambda d, pattern='RGGB', method='bilinear':
                        Device(np.repeat(d[..., None], 3, axis=2)))
    seen = []
    monkeypatch.setattr(fp, 'measure_chromatic_aberration',
                        lambda rgb, **k: seen.append(type(rgb)) or {0: (0.0, 0.0), 2: (0.0, 0.0)})
    args = argparse.Namespace(ca_correction=True, _session_bayer='RGGB')
    frames = [FrameInfo(path=f'f{i}.fits', type='light', header={}) for i in range(20)]
    out = fp._measure_session_ca(frames, args)
    assert seen and all(t is np.ndarray for t in seen)
    assert out is not None
