"""Band-by-band combine of a disk-backed aligned stack (stacking._combine_in_bands)
and proper coadd's progress/cancel ticker.

An 849-frame session spent minutes in single calls with no output (the aligned
stack was 60 GB on disk); banding must give exactly the one-call result.
"""
import argparse
import threading

import numpy as np
import pytest

import src.stacking as st
from src.models import RunCancelled


def _memmap_stack(tmp_path, n=12, h=40, w=48, seed=21):
    rng = np.random.default_rng(seed)
    d = rng.normal(1000.0, 20.0, (n, h, w, 3)).astype(np.float32)
    d[3, 10:13, 20:23] += 5000.0                      # outliers to reject
    mm = np.memmap(tmp_path / 'aligned.dat', dtype=np.float32, mode='w+', shape=d.shape)
    mm[:] = d
    return mm


@pytest.mark.skipif(not (st.HAS_NATIVE and hasattr(st._native, 'patch_weighted_sigma_combine_fast')),
                    reason='native kernel not built')
def test_banded_fused_combine_matches_one_call(tmp_path, monkeypatch):
    mm = _memmap_stack(tmp_path)
    rng = np.random.default_rng(6)
    qm = np.ascontiguousarray(rng.uniform(0.2, 1.0, (12, 8, 8)), dtype=np.float32)
    gw = rng.uniform(0.5, 1.5, 12).astype(np.float32)
    geom = (56.0, 64.0, 9.0, 10.0)

    def fused(block, r0):
        return st._native.patch_weighted_sigma_combine_fast(
            block, qm, gw, 3.0, 3, True, (geom[0], geom[1], geom[2] + r0, geom[3]))

    whole = fused(np.ascontiguousarray(mm), 0)
    # ~7 rows per band: several bands, the last one short
    monkeypatch.setattr(st, '_BAND_BYTES', 12 * 48 * 3 * 4 * 7)
    banded = st._combine_in_bands(mm, fused, argparse.Namespace(), 'Combining')
    np.testing.assert_array_equal(banded, whole)


def test_ram_stack_is_one_call(tmp_path):
    arr = np.zeros((4, 10, 10, 3), np.float32)
    calls = []
    st._combine_in_bands(arr, lambda b, r0: calls.append((b.shape, r0)) or b[0],
                         argparse.Namespace(), 'x')
    assert calls == [((4, 10, 10, 3), 0)]


def test_cancel_between_bands(tmp_path, monkeypatch):
    mm = _memmap_stack(tmp_path)
    monkeypatch.setattr(st, '_BAND_BYTES', 12 * 48 * 3 * 4 * 5)
    ev = threading.Event()
    seen = []

    def combine(block, r0):
        seen.append(r0)
        ev.set()                      # the user presses Cancel during the first band
        return block[0]
    with pytest.raises(RunCancelled):
        st._combine_in_bands(mm, combine, argparse.Namespace(_cancel_event=ev), 'x')
    assert seen == [0]


def test_proper_coadd_ticker_reports_and_cancels(monkeypatch):
    from src import proper_coadd as pc
    from src.ui_events import get_ui_events
    seen = []
    monkeypatch.setattr(get_ui_events(), 'progress', lambda label, d, t: seen.append((d, t)))
    ev = threading.Event()
    tick = pc._progress_ticker('Proper coadd', 3, ev, verbose=False)
    tick()
    tick()
    assert seen == [(1, 3), (2, 3)]
    ev.set()
    with pytest.raises(RunCancelled):
        tick()
