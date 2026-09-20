"""Session-wide radial distortion: recovery, significance, and the applied warp."""
from types import SimpleNamespace

import numpy as np
from scipy import ndimage

from src.distortion import (
    RadialModel,
    _rigid,
    build_displacement_fields,
    collect_pairs,
    fit_radial_distortion,
    format_summary,
    is_significant,
)
from src.registration import apply_transform

_H, _W = 800, 1100
_DT = np.dtype([('xcentroid', 'f8'), ('ycentroid', 'f8'), ('flux', 'f8')])


def _stars(yx):
    a = np.zeros(len(yx), _DT)
    a['ycentroid'], a['xcentroid'], a['flux'] = yx[:, 0], yx[:, 1], 5000.0
    return a


def _rot(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s], [s, c]])


def _tf_from_rigid(R, t):
    """A registration transform whose to_aligned_yx() is  p -> R p + t  (inverse of
    the convention apply_transform reads: it stores R^T and t)."""
    m = np.eye(3)
    m[:2, :2] = R.T
    m[1, 2], m[0, 2] = t[0], t[1]
    return SimpleNamespace(params=m)


def _session(a1, n_frames=24, n_stars=260, seed=0, a2=0.0):
    rng = np.random.default_rng(seed)
    true = RadialModel(a1, a2, ((_H - 1) / 2, (_W - 1) / 2), 0.5 * np.hypot(_H, _W))
    u_ref = np.column_stack([rng.uniform(20, _H - 20, n_stars), rng.uniform(20, _W - 20, n_stars)])
    p_ref = true.distort(u_ref)
    ref_stars = _stars(p_ref)
    final, shifts, tfs, truth = [], [], [], []
    for _ in range(n_frames):
        R = _rot(rng.uniform(-9, 9))
        t = rng.uniform(-40, 40, 2)
        u_frame = (u_ref - t) @ R                       # R^T (u - t) row-wise
        p_frame = true.distort(u_frame)
        inside = ((p_frame[:, 0] > 5) & (p_frame[:, 0] < _H - 5)
                  & (p_frame[:, 1] > 5) & (p_frame[:, 1] < _W - 5))
        # the registration a rigid fit of the *distorted* positions would give
        Rf, tf = _rigid(p_frame[inside], p_ref[inside])
        final.append(SimpleNamespace(metrics={'_star_sources': _stars(p_frame[inside]
                                                                        + rng.normal(0, 0.05, (inside.sum(), 2)))}))
        shifts.append((0.0, 0.0))
        tfs.append(_tf_from_rigid(Rf, tf))
        truth.append((p_frame, inside))
    return true, ref_stars, final, shifts, tfs, p_ref, truth


def test_model_round_trips():
    m = RadialModel(0.03, -0.004, (400.0, 550.0), 680.0)
    p = np.random.default_rng(1).uniform(0, 1000, (200, 2))
    assert np.allclose(m.distort(m.undistort(p)), p, atol=1e-6)
    assert np.allclose(RadialModel(0, 0, (0, 0), 1).undistort(p), p)


def test_recovers_the_distortion_coefficient_and_removes_the_residual():
    true, ref, final, shifts, tfs, _, _ = _session(a1=0.03, n_frames=40)
    pairs = collect_pairs(final, shifts, tfs, ref, tol=6.0)
    assert sum(p is not None for p in pairs) >= 35
    fit = fit_radial_distortion(pairs, (_H, _W))
    assert abs(fit['model'].a1 - 0.03) < 0.004
    assert fit['rms_rigid_px'] > 0.6                     # rigid registration really is off
    assert fit['rms_model_px'] < 0.2
    assert np.isfinite(fit['heldout_model_px']) and fit['heldout_model_px'] < 0.25
    assert is_significant(fit)
    assert 'held-out' in format_summary(fit, True) and 'applied' in format_summary(fit, True)


def test_no_distortion_is_not_reported_as_significant():
    _, ref, final, shifts, tfs, _, _ = _session(a1=0.0, seed=3)
    fit = fit_radial_distortion(collect_pairs(final, shifts, tfs, ref, tol=6.0), (_H, _W))
    assert abs(fit['model'].a1) < 0.003
    assert not is_significant(fit)
    assert 'not significant' in format_summary(fit, False)


def test_too_few_frames_or_stars_gives_none():
    _, ref, final, shifts, tfs, _, _ = _session(a1=0.03, n_frames=3)
    assert fit_radial_distortion(collect_pairs(final, shifts, tfs, ref, tol=6.0), (_H, _W)) is None
    empty = collect_pairs([SimpleNamespace(metrics={})] * 5, [(0, 0)] * 5, [None] * 5, ref)
    assert empty == [None] * 5 and fit_radial_distortion(empty, (_H, _W)) is None


def _render(pts, amp=5000.0, sigma=1.8):
    img = np.zeros((_H, _W), np.float32)
    yy, xx = np.mgrid[:_H, :_W]
    for y, x in pts:
        y0, y1 = int(max(y - 12, 0)), int(min(y + 13, _H))
        x0, x1 = int(max(x - 12, 0)), int(min(x + 13, _W))
        gy, gx = np.mgrid[y0:y1, x0:x1]
        img[y0:y1, x0:x1] += amp * np.exp(-((gy - y) ** 2 + (gx - x) ** 2) / (2 * sigma ** 2))
    return img[:, :, None]


def _centroid_errors(out, targets):
    errs = []
    for y, x in targets:
        iy, ix = int(round(y)), int(round(x))
        if not (10 <= iy < _H - 10 and 10 <= ix < _W - 10):
            continue
        win = out[iy - 6:iy + 7, ix - 6:ix + 7, 0]
        if win.sum() < 1e3:
            continue
        cy, cx = ndimage.center_of_mass(win)
        errs.append(np.hypot(cy + iy - 6 - y, cx + ix - 6 - x))
    return np.array(errs)


def test_the_built_fields_line_stars_up_through_apply_transform():
    """End to end: render a distorted frame, warp it with its registration alone
    and with registration + the model's field, and measure where the stars land
    against where the reference has them. This is the check that the field's
    sign and axis conventions really match the elastic-warp path."""
    true, ref, final, shifts, tfs, p_ref, truth = _session(a1=0.06, seed=5, n_frames=40)
    fit = fit_radial_distortion(collect_pairs(final, shifts, tfs, ref, tol=6.0), (_H, _W))
    fields = build_displacement_fields(fit, shifts, tfs, (_H, _W))
    assert len(fields) == len(final) and all(f is not None for f in fields)
    worse, better = [], []
    for j in (0, 7, 15, 25, 33):
        p_frame, inside = truth[j]
        img = _render(p_frame[inside])
        rigid_only = apply_transform(img, transform=tfs[j])
        with_field = apply_transform(img, transform=tfs[j], local_field=fields[j])
        targets = p_ref[inside]
        worse.append(np.median(_centroid_errors(rigid_only, targets)))
        better.append(np.median(_centroid_errors(with_field, targets)))
    assert np.mean(worse) > 0.5                       # distortion really misplaces stars
    assert np.mean(better) < 0.15 * np.mean(worse)    # and the field puts them back
    assert np.mean(better) < 0.1
