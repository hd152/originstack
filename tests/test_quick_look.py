"""The desktop app's quick look after registration (src/quick_look.py)."""
from types import SimpleNamespace

import numpy as np
import pytest

from src import quick_look as ql
from src.registration import apply_transform
from src.ui_events import UIEvents


def _stars(h=160, w=200, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.full((h, w, 3), 50.0, np.float32)
    for _ in range(25):
        cy, cx, a = rng.uniform(30, h - 30), rng.uniform(30, w - 30), rng.uniform(300, 2000)
        g = a * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.8 ** 2))
        img += g[..., None]
    return img


def _session(n=6, shifts_px=None):
    base = _stars()
    shifts_px = shifts_px or [(0.0, 0.0), (3.0, -2.0), (-4.0, 5.0), (6.0, 1.0), (-2.0, -6.0), (1.0, 4.0)][:n]
    frames = np.stack([apply_transform(base, shift=s) for s in shifts_px])
    return base, frames, shifts_px


def _align_shifts(shifts_px, frames):
    """Whichever sign aligns the frames: the one that makes the aligned mean sharpest."""
    best = None
    for sign in (1, -1):
        sh = [(sign * dy, sign * dx) for dy, dx in shifts_px]
        crop = (20, 140, 20, 180)
        pos = list(range(len(frames)))
        img = ql.quick_look_image(frames, pos, pos, sh, [None] * len(frames), crop)
        if best is None or img.max() > best[1]:
            best = (sh, img.max())
    return best[0]


def test_aligned_mean_is_sharper_than_the_unaligned_mean():
    _, frames, shifts_px = _session()
    sh = _align_shifts(shifts_px, frames)
    crop = (20, 140, 20, 180)
    pos = list(range(len(frames)))
    aligned = ql.quick_look_image(frames, pos, pos, sh, [None] * len(frames), crop)
    plain = frames[:, 20:140, 20:180].mean(axis=0)
    assert aligned.shape == (120, 160, 3)
    assert aligned.max() > 1.4 * plain.max()


def test_large_frames_are_binned_to_preview_size():
    big = np.random.default_rng(0).uniform(0, 100, (2, 2400, 3000, 3)).astype(np.float32)
    img = ql.quick_look_image(big, [0, 1], [0, 1], [(0.0, 0.0)] * 2, [None] * 2, (0, 2400, 0, 3000))
    assert max(img.shape[:2]) <= ql.QUICK_LOOK_MAX_DIM and img.shape[2] == 3


def test_pick_frames_takes_the_sharpest_in_session_order():
    final = [SimpleNamespace(metrics={'fwhm': f}) for f in (5.0, 3.0, 0.0, 4.0, 2.5)]
    assert ql.pick_frames(final, n=3) == [1, 3, 4]       # 0.0 (unmeasured) sorts last


def test_nothing_happens_without_the_desktop_app(monkeypatch):
    ui = UIEvents()                                       # inactive, like a CLI run
    monkeypatch.setattr('src.ui_events.get_ui_events', lambda: ui)
    called = []
    monkeypatch.setattr(ql, 'quick_look_image', lambda *a, **k: called.append(1))
    assert ql.publish_quick_look(None, [], [], [], [], (0, 1, 0, 1), SimpleNamespace()) is False
    assert not called


def test_published_as_a_named_preview_when_the_ui_is_attached(monkeypatch):
    _, frames, shifts_px = _session()
    ui = UIEvents()
    ui.attach()
    monkeypatch.setattr('src.ui_events.get_ui_events', lambda: ui)
    final = [SimpleNamespace(metrics={'fwhm': 3.0 + i}) for i in range(len(frames))]
    ok = ql.publish_quick_look(frames, final, list(range(len(frames))), [(0.0, 0.0)] * len(frames),
                               [None] * len(frames), (20, 140, 20, 180), SimpleNamespace())
    assert ok is True
    assert ui.named_jpeg('quick-look')


def test_a_failure_is_swallowed_but_a_cancel_is_not(monkeypatch):
    from src.models import RunCancelled
    ui = UIEvents()
    ui.attach()
    monkeypatch.setattr('src.ui_events.get_ui_events', lambda: ui)
    monkeypatch.setattr(ql, 'quick_look_image', lambda *a, **k: (_ for _ in ()).throw(ValueError('x')))
    assert ql.publish_quick_look(None, [SimpleNamespace(metrics={})], [0], [(0, 0)], [None],
                                 (0, 9, 0, 9), SimpleNamespace()) is False
    monkeypatch.setattr(ql, 'quick_look_image', lambda *a, **k: (_ for _ in ()).throw(RunCancelled()))
    with pytest.raises(RunCancelled):
        ql.publish_quick_look(None, [SimpleNamespace(metrics={})], [0], [(0, 0)], [None],
                              (0, 9, 0, 9), SimpleNamespace())
