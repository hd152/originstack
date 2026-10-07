"""--full-field: coverage geometry, rectangle growth and the outer combine."""
from types import SimpleNamespace

import numpy as np
import pytest

from src import full_field as ff
from src.registration import apply_transform, calc_common_crop


def _rot(deg, tx=0.0, ty=0.0, cy=0.0, cx=0.0):
    a = np.deg2rad(deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    c = np.array([cx, cy])
    t = c - R @ c + np.array([tx, ty])
    m = np.eye(3)
    m[:2, :2] = R
    m[:2, 2] = t
    return SimpleNamespace(params=m)


@pytest.mark.parametrize("shift,transform", [
    ((5.3, -7.8), None),
    (None, _rot(3.0, 4.0, -2.0, cy=60, cx=80)),
    (None, _rot(-6.0, -3.0, 5.0, cy=60, cx=80)),
])
def test_coverage_matches_the_warp(shift, transform):
    """A pixel frame_coverage calls covered is fully inside the warped frame:
    warping an all-ones image gives 1 there."""
    H, W = 120, 160
    ones = np.ones((H, W, 3), np.float32)
    warped = apply_transform(ones, shift=shift, transform=transform, crop=(0, H, 0, W))
    cov = ff.frame_coverage(shift, transform, np.arange(H), np.arange(W), H, W)
    assert cov.mean() > 0.6
    np.testing.assert_allclose(warped[cov], 1.0, atol=1e-4)
    # and well outside the footprint the warp is empty
    far = ~ff.frame_coverage(shift, transform, np.arange(H), np.arange(W), H, W, margin=-4)
    assert np.all(np.abs(warped[far]) < 1e-6)


def test_common_crop_is_fully_covered():
    H, W = 200, 300
    shifts = [None] * 5
    transforms = [_rot(d, cy=100, cx=150) for d in (-4, -2, 0, 2, 4)]
    top, bottom, left, right = calc_common_crop([(0, 0)] * 5, (H, W), transforms=transforms)
    cnt = ff.coverage_count(shifts, transforms, np.arange(top, bottom),
                            np.arange(left, right), H, W)
    # the common crop sits inside every frame (up to the warp-edge margin)
    assert (cnt == 5).mean() > 0.97


def test_grow_rectangle_stops_at_uncovered():
    ok = np.zeros((10, 12), bool)
    ok[2:9, 1:10] = True
    assert ff.grow_rectangle(ok, (4, 5, 4, 6)) == (2, 8, 1, 9)
    ok[2, 5] = False
    assert ff.grow_rectangle(ok, (4, 5, 4, 6)) == (3, 8, 1, 9)


def test_rectangle_contains_core_and_meets_threshold():
    H, W = 200, 300
    n = 12
    transforms = [_rot(a, cy=100, cx=150) for a in np.linspace(-8, 8, n)]
    shifts = [None] * n
    core = calc_common_crop([(0, 0)] * n, (H, W), transforms=transforms)
    rect = ff.choose_rectangle(shifts, transforms, H, W, core, frac=0.5)
    t, b, l, r = rect
    assert t <= core[0] and b >= core[1] and l <= core[2] and r >= core[3]
    assert (b - t) * (r - l) > 1.1 * (core[1] - core[0]) * (core[3] - core[2])
    cnt = ff.coverage_count(shifts, transforms, np.arange(t, b), np.arange(l, r), H, W)
    assert np.percentile(cnt, 0.5) >= 0.5 * n - 1


def test_extend_recovers_scene_outside_the_core():
    """Translated noisy frames of a known scene: the extension reproduces the
    scene outside the core and leaves the core's interior untouched."""
    rng = np.random.default_rng(3)
    H, W, n, pad = 160, 224, 10, 20
    yy, xx = np.mgrid[0:H + 2 * pad, 0:W + 2 * pad]
    scene = (100 + 0.3 * yy + 0.2 * xx).astype(np.float32)
    offs = [tuple(int(v) for v in rng.integers(-15, 16, 2)) for _ in range(n)]
    frames = []
    for dy, dx in offs:
        f = scene[pad + dy:pad + dy + H, pad + dx:pad + dx + W][..., None].repeat(3, -1)
        frames.append(np.ascontiguousarray(f + rng.normal(0, 1.0, (H, W, 3)), np.float32))
    # registered onto frame 0's grid: output(y) = frame_k(y - s_k), s_k = off_k - off_0
    shifts = [(float(dy - offs[0][0]), float(dx - offs[0][1])) for dy, dx in offs]
    truth = scene[pad + offs[0][0]:pad + offs[0][0] + H, pad + offs[0][1]:pad + offs[0][1] + W]
    core = calc_common_crop(shifts, (H, W))
    aligned = np.stack([apply_transform(f, shift=s, crop=core) for f, s in zip(frames, shifts)])
    core_stack = aligned.mean(0).astype(np.float32)
    args = SimpleNamespace(rejection_sigma=3.0, rejection_iters=3)
    res = ff.extend_full_field(core_stack, frames, list(range(n)), shifts, [None] * n,
                               H, W, 3, core, np.ones(n), args, frac=0.5)
    assert res is not None
    out, rect, coverage = res
    t, b, l, r = rect
    ct, cb, cl, cr = core
    assert (b - t) * (r - l) > (cb - ct) * (cr - cl)
    outer = np.ones((b - t, r - l), bool)
    outer[ct - t:cb - t, cl - l:cr - l] = False
    err = out[..., 1][outer] - truth[t:b, l:r][outer]
    assert abs(float(np.median(err))) < 0.3
    assert float(np.std(err)) < 1.0
    assert coverage[outer].min() >= 5
    assert np.all(coverage[~outer] == n)
    # the core's interior (beyond the feather band) is the core stack itself
    fe = min(ff.FEATHER_PX, (cb - ct) // 4, (cr - cl) // 4) + 1
    np.testing.assert_array_equal(out[ct - t + fe:cb - t - fe, cl - l + fe:cr - l - fe],
                                  core_stack[fe:-fe, fe:-fe])


def test_gain_fit_is_not_diluted_by_noise():
    """Two noisy copies of a mostly-flat sky with a few stars: the fitted gain is the
    true one. The unsmoothed fit on sky pixels this replaced read ~0.5 on real data."""
    rng = np.random.default_rng(2)
    H, W = 64, 2000
    yy, xx = np.mgrid[:H, :W].astype(float)
    truth = np.full((H, W), 1000.0)
    for y, x in zip(rng.uniform(5, H - 5, 40), rng.uniform(5, W - 5, 40)):
        truth += 2000 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 2.0 ** 2))
    src = (truth + rng.normal(0, 60, truth.shape)).astype(np.float32)
    ref = (1.3 * truth - 200 + rng.normal(0, 60, truth.shape)).astype(np.float32)
    g, o = ff._fit_gain_offset(src, ref)
    assert abs(g - 1.3) < 0.03
    assert abs(o + 200) < 40


def test_sky_drift_through_the_session_does_not_offset_the_edges():
    """The sky changes through the night while the pointing drifts, so the frames
    that reach an edge come from one end of the session. Without per-frame sky
    matching the edge took that subset's sky level (real Sunflower session:
    sky fell 27%, the added strip came out ~9 noise sigma dark)."""
    rng = np.random.default_rng(5)
    H, W, n, pad = 160, 224, 16, 24
    yy, xx = np.mgrid[0:H + 2 * pad, 0:W + 2 * pad]
    scene = (1000 + 0.2 * yy + 0.1 * xx).astype(np.float32)
    offs = [(0, int(round(-20 + 40 * k / (n - 1)))) for k in range(n)]  # drift left->right
    sky = [400.0 * (1 - k / (n - 1)) for k in range(n)]                  # sky falls
    frames = []
    for (dy, dx), sk in zip(offs, sky):
        f = scene[pad + dy:pad + dy + H, pad + dx:pad + dx + W][..., None].repeat(3, -1) + sk
        frames.append(np.ascontiguousarray(f + rng.normal(0, 2.0, (H, W, 3)), np.float32))
    ref_off = offs[n // 2]
    shifts = [(float(dy - ref_off[0]), float(dx - ref_off[1])) for dy, dx in offs]
    core = calc_common_crop(shifts, (H, W))
    aligned = np.stack([apply_transform(f, shift=s, crop=core) for f, s in zip(frames, shifts)])
    core_stack = aligned.mean(0).astype(np.float32)
    truth = scene[pad + ref_off[0]:pad + ref_off[0] + H,
                  pad + ref_off[1]:pad + ref_off[1] + W] + float(np.mean(sky))
    args = SimpleNamespace(rejection_sigma=3.0, rejection_iters=3)
    out, (t, b, l, r), _ = ff.extend_full_field(core_stack, frames, list(range(n)), shifts,
                                                [None] * n, H, W, 3, core, np.ones(n), args,
                                                frac=0.5)
    ct, cb, cl, cr = core
    assert l < cl and r > cr
    for cols in (slice(0, cl - l), slice(cr - l, r - l)):          # left and right strips
        err = out[ct - t:cb - t, cols, 1] - truth[ct:cb, l:r][:, cols]
        assert abs(float(np.median(err))) < 5.0, float(np.median(err))
