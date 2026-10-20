"""Comet mode: the comet is found by its motion against the stars, each frame keeps
its own field-rotation correction, and the comet + stars composite is built from
layers. Regressions from real sessions: on SWAN the "nucleus" was a bright star
(the comet stack was aligned on it and every other star arced round it); on
Lemmon (360 px of motion) the old Gaussian blend showed two heads."""
from types import SimpleNamespace

import numpy as np

from src.affine_fit import RigidTransform
from src.registration import (
    _aligned_to_raw_yx,
    _comet_frame_transform,
    _comet_motion_track,
)
from src.stacking import blend_comet_star_stacks
from src.transparency import to_aligned_yx


def _rigid(deg, ty, tx):
    a = np.radians(deg)
    p = np.eye(3)
    p[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    p[0, 2], p[1, 2] = tx, ty
    return RigidTransform(p)


def test_aligned_to_raw_inverts_to_aligned():
    tf = _rigid(7.0, 12.0, -30.0)
    yx = np.array([[100.0, 250.0], [800.0, 40.0]])
    np.testing.assert_allclose(_aligned_to_raw_yx(to_aligned_yx(yx, tf, None), tf, None), yx)
    np.testing.assert_allclose(_aligned_to_raw_yx(to_aligned_yx(yx, None, (3.0, -4.0)),
                                                  None, (3.0, -4.0)), yx)


def test_comet_transform_keeps_rotation_and_lands_the_nucleus():
    tf = _rigid(-9.0, 55.0, 392.0)
    raw_nucleus = np.array([600.0, 1500.0])
    aligned = to_aligned_yx(raw_nucleus, tf, None)
    ref_pos = aligned - np.array([40.0, -120.0])        # the comet moved since the reference
    shift, new_tf = _comet_frame_transform(tf, (55.0, 392.0), aligned - ref_pos)
    np.testing.assert_allclose(to_aligned_yx(raw_nucleus, new_tf, None), ref_pos, atol=1e-9)
    np.testing.assert_allclose(new_tf.params[:2, :2], tf.params[:2, :2])   # rotation kept
    # a shift-only star registration composes the same way
    sh, none_tf = _comet_frame_transform(None, (5.0, -2.0), np.array([1.0, 3.0]))
    assert none_tf is None and sh == (4.0, -5.0)


def _blob(H, W, y, x, amp, sigma):
    yy, xx = np.mgrid[:H, :W]
    return amp * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2))


def test_motion_finds_the_moving_comet_not_the_brightest_blob():
    H, W, n = 160, 220, 24
    rng = np.random.default_rng(0)
    stars = np.zeros((H, W))
    for _ in range(40):
        stars += _blob(H, W, rng.uniform(15, H - 15), rng.uniform(15, W - 15),
                       rng.uniform(200, 2000), 1.2)
    stars += _blob(H, W, 120, 40, 6000.0, 4.0)           # brighter and broader than the comet
    frames, final = [], []
    for i in range(n):
        cy, cx = 60.0 + 1.5 * i, 70.0 + 2.5 * i           # moves ~35 x 58 px over the session
        img = 1000.0 + stars + _blob(H, W, cy, cx, 900.0, 5.0) + rng.normal(0, 8, (H, W))
        frames.append(img.astype(np.float32))
        final.append(SimpleNamespace(header={'DATE-OBS': f'2025-10-30T19:{6 + i // 6:02d}:{(i % 6) * 10:02d}'},
                                     path=f'f{i}.fits'))
    mem_lum = np.stack(frames)
    got = _comet_motion_track(final, list(range(n)), mem_lum, [(0.0, 0.0)] * n, [None] * n)
    assert got is not None
    first, last = got['positions'][0], got['positions'][-1]
    assert np.hypot(first[0] - 60.0, first[1] - 70.0) < 2.0
    assert np.hypot(last[0] - (60.0 + 1.5 * (n - 1)), last[1] - (70.0 + 2.5 * (n - 1))) < 2.0


def test_no_moving_object_returns_none():
    H, W, n = 120, 160, 12
    rng = np.random.default_rng(1)
    base = 1000.0 + _blob(H, W, 60, 80, 900.0, 5.0)
    mem_lum = np.stack([(base + rng.normal(0, 8, (H, W))).astype(np.float32) for _ in range(n)])
    final = [SimpleNamespace(header={}, path=f'f{i}.fits') for i in range(n)]
    assert _comet_motion_track(final, list(range(n)), mem_lum, [(0.0, 0.0)] * n, [None] * n) is None


def test_composite_has_one_comet_head_and_the_stars():
    H, W = 120, 200
    rng = np.random.default_rng(2)
    sky = 500.0 + rng.normal(0, 5, (H, W))
    star_pos = ((20, 30), (90, 170), (100, 60))
    stars = sum(_blob(H, W, y, x, 3000.0, 1.3) for y, x in star_pos)
    comet = _blob(H, W, 60, 100, 5000.0, 6.0)
    # the comet's displacement from its reference position, frame by frame
    track = np.array([(0.0, dx) for dx in np.linspace(-60, 60, 61)])
    # star-aligned stack: sharp stars, the comet smeared along its track
    smear = sum(_blob(H, W, 60, 100 + dx, 5000.0, 6.0) for _, dx in track) / len(track)
    star_stack = np.repeat((sky + stars + smear)[..., None], 3, axis=2).astype(np.float32)
    # comet-aligned stack: the comet sharp at its reference position, stars rejected
    comet_stack = np.repeat((sky + comet)[..., None], 3, axis=2).astype(np.float32)
    lum = blend_comet_star_stacks(star_stack, comet_stack, track=track).mean(axis=2)
    assert lum[60, 100] > 4500.0                          # one head, at full strength
    for x in (45, 70, 130, 155):
        assert abs(lum[60, x] - 500.0) < 40.0             # no trail, no second head
    for y, x in star_pos:
        assert lum[y, x] > 3000.0                         # the stars are in
    far = lum[:, :20]
    assert abs(float(np.median(far)) - 500.0) < 5.0       # sky unchanged away from the comet
