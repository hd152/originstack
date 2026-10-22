"""--merge-consistency: merge.consistency_combine rejects what only one night shows."""
import numpy as np

from src.merge import consistency_combine

H, W = 160, 200


def _scene(rng):
    yy, xx = np.mgrid[:H, :W].astype(np.float64)
    img = np.full((H, W, 3), 100.0)
    for _ in range(25):                                   # stars
        y, x, f = rng.uniform(10, H - 10), rng.uniform(10, W - 10), rng.uniform(200, 3000)
        img += (f * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.5 ** 2)))[:, :, None]
    img += (40 * np.exp(-((yy - 80) ** 2 + (xx - 100) ** 2) / (2 * 25.0 ** 2)))[:, :, None]   # nebula
    return img


def _blob(y, x, peak, s=1.0):
    yy, xx = np.mgrid[:H, :W].astype(np.float64)
    return (peak * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * s * s)))[:, :, None] * np.ones(3)


def _flux(img, y, x, r=5):
    yy, xx = np.mgrid[:H, :W]
    m = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
    ring = ((yy - y) ** 2 + (xx - x) ** 2 >= 9 ** 2) & ((yy - y) ** 2 + (xx - x) ** 2 < 12 ** 2)
    return float((img[m, 1] - np.median(img[ring, 1])).sum())


def _empty_spot(base, margin=20):
    """(y, x) of the faintest smoothed spot: artefacts on real structure are kept by design
    (the other nights show something there), so the tests put them on empty sky."""
    from scipy import ndimage as ndi
    sm = ndi.uniform_filter(base[:, :, 1], 25)[margin:-margin, margin:-margin]
    y, x = np.unravel_index(int(np.argmin(sm)), sm.shape)
    return y + margin, x + margin


def _nights(n, sigma=5.0, seed=0):
    rng = np.random.default_rng(seed)
    base = _scene(rng)
    return base, [base + rng.normal(0, sigma, base.shape) for _ in range(n)]


def _members(imgs, valid=None, sigma=5.0):
    valid = valid or [np.ones((H, W), bool)] * len(imgs)
    return [(im.astype(np.float32), v, np.full(3, 1.0 / sigma ** 2)) for im, v in zip(imgs, valid)]


def _plain(imgs):
    return np.mean(imgs, axis=0)


def test_one_night_artefact_removed_shared_faint_source_kept():
    base, (a, b) = _nights(2)
    gy, gx = _empty_spot(base)
    fy, fx = (gy + 50) % (H - 40) + 20, (gx + 80) % (W - 40) + 20
    faint = _blob(fy, fx, 6.0, 2.0)                      # ~1.2 sigma per pixel, in both nights
    a, b = a + faint, b + faint
    glint = _blob(gy, gx, 500.0)                         # 100 sigma, night a only
    clean, _ = consistency_combine(_members([a, b]))
    out, frac = consistency_combine(_members([a + glint, b]))
    plain = _plain([a + glint, b])
    g = _flux(glint, gy, gx)
    assert _flux(plain, gy, gx) - _flux(_plain([a, b]), gy, gx) > 0.4 * g
    assert abs(_flux(out, gy, gx) - _flux(clean, gy, gx)) < 0.1 * g        # the glint's share left
    assert _flux(out, fy, fx) / _flux(plain, fy, fx) > 0.95
    away = np.ones((H, W), bool)
    away[gy - 15:gy + 16, gx - 15:gx + 16] = False
    assert np.abs(out - clean)[away].std() < 0.3        # elsewhere unchanged, within noise
    assert 0 < frac < 0.05


def test_identical_scene_is_the_plain_mean():
    _, (a, b) = _nights(2, seed=3)
    out, frac = consistency_combine(_members([a, b]))
    d = out - _plain([a, b])
    assert np.abs(d).mean() < 0.3 and frac < 0.005      # chance flags on noise: rare and local


def test_partial_footprint_leaves_no_seam():
    _, (a, b, c) = _nights(3, seed=5)
    valid_c = np.zeros((H, W), bool)
    valid_c[:, :W // 2] = True
    c = np.where(valid_c[:, :, None], c, 0.0)              # zero fill outside its footprint
    out, _ = consistency_combine(_members([a, b, c], [np.ones((H, W), bool)] * 2 + [valid_c]))
    plain = np.where(valid_c[:, :, None], (a + b + c) / 3, (a + b) / 2)
    seam = np.abs(out - plain)[:, W // 2 - 6:W // 2 + 6]
    assert seam.mean() < 1.0 and seam.max() < 20


def test_three_nights_reject_the_odd_one():
    base, (a, b, c) = _nights(3, seed=7)
    gy, gx = _empty_spot(base)
    glint = _blob(gy, gx, 300.0)
    clean, _ = consistency_combine(_members([a, b, c]))
    out, _ = consistency_combine(_members([a, b + glint, c]))
    assert abs(_flux(out, gy, gx) - _flux(clean, gy, gx)) < 0.1 * _flux(glint, gy, gx)
