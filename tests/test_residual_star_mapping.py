"""The residual check maps each frame's own stars into aligned space instead of warping
the frame and detecting again (registration._stars_to_aligned, _match_frame_stars).
The mapping must be the exact inverse of the warp the old path applied."""
import numpy as np
import pytest
from scipy.spatial import cKDTree

import src.registration as reg
from src.affine_fit import RigidTransform


def _rot(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return RigidTransform(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]))


def _star(shape, y, x, sigma=1.5, amp=1000.0):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return amp * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2))


def _peak(img):
    """Position (y, x) of a single Gaussian: the first moment over the whole image
    (a window centred on the brightest pixel pulls it toward that pixel by ~0.2 px)."""
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    t = img.sum()
    return (yy * img).sum() / t, (xx * img).sum() / t


@pytest.mark.parametrize("rot_deg", [0.0, 3.0, -6.0])
def test_mapping_is_the_inverse_of_the_residual_warp(rot_deg):
    shape = (200, 260)
    frame_yx = np.array([[90.3, 120.7]])
    lum = _star(shape, *frame_yx[0]).astype(np.float32)
    shift = (7.25, -11.5)
    tf = None
    if rot_deg:
        tf = _rot(rot_deg)
    warped = reg._residual_warp(lum, shift, tf)
    want = np.array(_peak(np.asarray(warped, np.float64)))
    got = reg._stars_to_aligned(frame_yx, shift, tf)[0]
    np.testing.assert_allclose(got, want, atol=0.05)


def _catalog(yx):
    out = np.zeros(len(yx), dtype=[('xcentroid', 'f8'), ('ycentroid', 'f8'), ('flux', 'f8')])
    out['ycentroid'], out['xcentroid'], out['flux'] = yx[:, 0], yx[:, 1], 1000.0
    return out


def test_correct_transform_matches_and_a_wrong_one_does_not():
    rng = np.random.default_rng(0)
    shape = (300, 400)
    ref_yx = np.c_[rng.uniform(20, 280, 40), rng.uniform(20, 380, 40)]
    tf = _rot(4.0)
    shift = (5.0, -8.0)
    # frame positions: the warp maps aligned o to frame i = M @ (o - shift)
    M = tf.params[:2, :2]
    frame_yx = (M @ (ref_yx - np.array(shift)).T).T
    lum = np.zeros(shape, np.float32)
    ref_tree = cKDTree(ref_yx[:, ::-1])
    good = reg._match_frame_stars(lum, shift, tf, ref_tree, ref_yx[:, ::-1], 1.0, _catalog(frame_yx))
    assert good is not None
    rms = np.sqrt(np.mean(np.sum((good[0] - good[1]) ** 2, axis=1)))
    assert len(good[0]) >= 30 and rms < 1e-6
    bad = reg._match_frame_stars(lum, (shift[0] + 3.0, shift[1]), tf, ref_tree, ref_yx[:, ::-1], 1.0,
                                 _catalog(frame_yx))
    rms_bad = np.sqrt(np.mean(np.sum((bad[0] - bad[1]) ** 2, axis=1)))
    assert rms_bad > 2.5                                     # a 3 px error shows as ~3 px


def test_shift_only_mapping_and_a_singular_transform():
    yx = np.array([[10.0, 10.0], [50.0, 50.0], [95.0, 95.0]])
    out = reg._stars_to_aligned(yx, (20.0, 20.0), None)
    np.testing.assert_allclose(out, yx + 20.0)              # shift only: o = i + shift
    assert reg._stars_to_aligned(yx, (0.0, 0.0),
                                 RigidTransform(np.diag([0.0, 1.0, 1.0]))) is None
