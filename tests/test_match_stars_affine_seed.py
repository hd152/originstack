"""match_stars_affine's seed is calculate_shift's (dy, dx) -- the shift that
moves the frame onto the reference, i.e. ref = img + shift. It used to
subtract the seed, predicting every star 2*|shift| away, so any seed larger
than ~AFFINE_MATCH_RADIUS/2 failed and the frame dropped to translation-only
(field-rotation smear) whenever the blind matcher also failed."""
import numpy as np
import pytest

from src.registration import calculate_shift, match_stars_affine


def _table(xy):
    return [{'xcentroid': float(x), 'ycentroid': float(y)} for x, y in xy]


def test_seed_from_calculate_shift_recovers_rotation_and_shift():
    rng = np.random.default_rng(1)
    ref_xy = np.column_stack([rng.uniform(40, 560, 60), rng.uniform(40, 360, 60)])
    ang = np.radians(0.5)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    img_xy = ref_xy @ rot.T + [-22.0, 37.0]          # frame drifted 37 px down, 22 left
    # The seed calculate_shift would return: (dy, dx) with ref = img + seed.
    seed = tuple((ref_xy - img_xy).mean(axis=0)[::-1])
    tf = match_stars_affine(_table(ref_xy), _table(img_xy), initial_shift=seed)
    assert tf is not None
    mapped = img_xy @ tf.params[:2, :2].T + tf.params[:2, 2]
    np.testing.assert_allclose(mapped, ref_xy, atol=1e-6)


def test_seed_convention_matches_calculate_shift():
    """Pins the convention the fix relies on, on rendered images."""
    h, w = 200, 300
    yy, xx = np.mgrid[:h, :w]
    ref_xy = np.array([[60.0, 50.0], [200.0, 80.0], [120.0, 150.0], [250.0, 170.0]])

    def render(xy):
        im = np.full((h, w), 10.0, np.float32)
        for x, y in xy:
            im += 500 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 4.5)
        return im

    dy, dx = calculate_shift(render(ref_xy), render(ref_xy + [-12.0, 18.0]), verbose=False)
    assert dy == pytest.approx(-18.0, abs=0.1)
    assert dx == pytest.approx(12.0, abs=0.1)
