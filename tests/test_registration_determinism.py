"""match_stars_affine is reproducible: RANSAC is seeded, so the same stars give the same transform."""
import numpy as np
import pytest

from src import registration as rg


def _stars(pts):
    return [{'xcentroid': x, 'ycentroid': y} for x, y in pts]


@pytest.mark.skipif(not rg.HAS_SKIMAGE_TRANSFORM, reason='affine matching unavailable')
def test_match_stars_affine_is_reproducible():
    rng = np.random.default_rng(3)
    ref = rng.uniform(50, 950, (60, 2))
    th = np.deg2rad(0.4)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    img = (ref - 500) @ R.T + 500 + [3.2, -1.7] + rng.normal(0, 1.2, ref.shape)  # residuals near the 2 px cut
    img[:12] = rng.uniform(50, 950, (12, 2))          # plus outliers: unseeded draws disagree
    runs = [rg.match_stars_affine(_stars(ref), _stars(img), (1.7, -3.2)) for _ in range(5)]
    assert runs[0] is not None
    for m in runs[1:]:
        np.testing.assert_array_equal(np.asarray(m.params), np.asarray(runs[0].params))


def test_match_stars_affine_passes_a_seed(monkeypatch):
    seen = {}

    def fake(src, dst, **kw):
        seen.update(kw)
        return None, None
    monkeypatch.setattr(rg, 'fit_rigid_ransac', fake)
    monkeypatch.setattr(rg, 'HAS_SKIMAGE_TRANSFORM', True)
    pts = _stars([(10.0 * i, 7.0 * i + 3) for i in range(1, 8)])
    rg.match_stars_affine(pts, pts, (0.0, 0.0))
    assert seen.get('seed') is not None
