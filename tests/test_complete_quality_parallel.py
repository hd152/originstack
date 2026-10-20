"""pipeline._complete_accepted_quality_metrics runs the post-gate diagnostics in threads:
the same metrics as one frame at a time, and it stops on Cancel."""
import argparse
import copy
import threading

import numpy as np
import pytest

from src import pipeline
from src.models import FrameInfo, RunCancelled
from src.quality import complete_quality_metrics, compute_quality_metrics


def _frames(n=6, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:256, 0:320].astype(np.float32)
    lums = []
    for _ in range(n):
        img = 1000 + rng.normal(0, 20, (256, 320)).astype(np.float32)
        for _ in range(25):
            y, x = rng.uniform(10, 246), rng.uniform(10, 310)
            img += rng.uniform(500, 5000) * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 4.5)
        lums.append(img.astype(np.float32))
    mem = np.stack(lums)
    frames = [FrameInfo(path=f"f{i}.fits", type="light", header={},
                        metrics=compute_quality_metrics(mem[i], level='gate')) for i in range(n)]
    return frames, mem


def test_parallel_matches_one_frame_at_a_time():
    frames, mem = _frames()
    serial = copy.deepcopy(frames)
    for f, i in zip(serial, range(len(serial))):
        complete_quality_metrics(mem[i], f.metrics)
    args = argparse.Namespace(advanced_metrics=True)
    pipeline._complete_accepted_quality_metrics(frames, list(range(len(frames))), mem, args)
    for a, b in zip(frames, serial):
        assert a.metrics.keys() == b.metrics.keys()
        for k in a.metrics:
            va, vb = a.metrics[k], b.metrics[k]
            if isinstance(va, np.ndarray):
                np.testing.assert_array_equal(va, vb)
            else:
                assert va == vb or (va != va and vb != vb), k          # NaN == NaN


def test_stops_on_cancel():
    frames, mem = _frames(n=3, seed=1)
    ev = threading.Event()
    ev.set()
    args = argparse.Namespace(advanced_metrics=True, _cancel_event=ev)
    with pytest.raises(RunCancelled):
        pipeline._complete_accepted_quality_metrics(frames, list(range(3)), mem, args)
