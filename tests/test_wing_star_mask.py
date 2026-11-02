"""Brightness-scaled star masks for background sampling (background.wing_aware_star_mask).

The 4 px FWHM blob that DBE gets at every star ends long before a bright star's r^-2.5 wing
does. The wing mask extends each bright star's mask to where its own radial profile reaches
``Config.WING_MASK_K`` pixel sigmas above the local sky.
"""
import numpy as np

from src.background import (
    _pixel_noise_sigma,
    _star_wing_radius,
    dynamic_background_extraction,
    wing_aware_star_mask,
)
from src.models import Config
from src.quality import generate_star_mask

H, W = 640, 800
SIGMA = 5.0


def _sources(stars, flux=6000.0):
    src = np.zeros(len(stars), dtype=[('xcentroid', 'f8'), ('ycentroid', 'f8'),
                                      ('flux', 'f8'), ('peak', 'f8')])
    for i, (cy, cx) in enumerate(stars):
        src[i] = (cx, cy, flux - i, flux)
    return src


def _frame(stars, wing_b, core=6000.0, seed=5, gradient=True):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    sig = np.zeros((H, W))
    for cy, cx in stars:
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        sig += core * np.exp(-r2 / (2 * 1.7 ** 2)) + wing_b * (1 + r2 / 9.0) ** -1.25
    sky = 1000 + (0.15 * xx + 0.1 * yy if gradient else 0.0)
    return sig, (sky + sig + rng.normal(0, SIGMA, (H, W))).astype(np.float32)


def _analytic_radius(wing_b, sigma, k):
    """Radius where wing_b (1 + r^2/9)^-1.25 falls to k sigma."""
    return 3.0 * np.sqrt((wing_b / (k * sigma)) ** 0.8 - 1.0)


def test_noise_estimate_ignores_gradient_and_stars():
    _, img = _frame([(200, 300)], wing_b=800.0)
    assert abs(_pixel_noise_sigma(img) - SIGMA) < 0.3


def test_wing_radius_matches_the_wing_profile():
    _, img = _frame([(300, 400)], wing_b=3000.0)
    k = Config.WING_MASK_K
    r = _star_wing_radius(img, 300.0, 400.0, SIGMA, k, 120, 4.0)
    expect = _analytic_radius(3000.0, SIGMA, k)
    assert 20 < expect < 120
    assert 0.7 * expect < r < 1.3 * expect


def test_brighter_wing_gives_larger_radius():
    radii = []
    for b in (500.0, 2000.0, 8000.0):
        _, img = _frame([(300, 400)], wing_b=b)
        radii.append(_star_wing_radius(img, 300.0, 400.0, SIGMA, 0.5, 120, 4.0))
    assert radii[0] < radii[1] < radii[2]


def test_gradient_does_not_change_the_radius():
    _, flat = _frame([(300, 400)], wing_b=3000.0, gradient=False)
    _, tilt = _frame([(300, 400)], wing_b=3000.0, gradient=True)
    a = _star_wing_radius(flat, 300.0, 400.0, SIGMA, 0.5, 120, 4.0)
    b = _star_wing_radius(tilt, 300.0, 400.0, SIGMA, 0.5, 120, 4.0)
    assert abs(a - b) <= 6


def test_star_without_a_wing_keeps_the_compact_mask():
    stars = [(300, 400)]
    _, img = _frame(stars, wing_b=0.0, core=2000.0)
    src = _sources(stars, 2000.0)
    base = generate_star_mask((H, W), src, fwhm=4.0)
    out = wing_aware_star_mask(img, src, base)
    assert (out >= 0.5).sum() <= 1.1 * (base >= 0.5).sum() + 30


def test_mask_only_grows_and_marks_the_wing():
    stars = [(200, 250), (450, 600)]
    _, img = _frame(stars, wing_b=4000.0)
    src = _sources(stars)
    base = generate_star_mask((H, W), src, fwhm=4.0)
    out = wing_aware_star_mask(img, src, base)
    assert out.dtype == np.float32 and out.shape == base.shape
    assert (out >= base).all()
    yy, xx = np.mgrid[0:H, 0:W]
    expect = _analytic_radius(4000.0, SIGMA / np.sqrt(1.0), Config.WING_MASK_K)
    inside = np.hypot(yy - 200, xx - 250) < 0.6 * expect
    assert (out[inside] >= 0.5).all()
    far = np.hypot(yy - 200, xx - 250) > 1.8 * expect
    far &= np.hypot(yy - 450, xx - 600) > 1.8 * expect
    assert (out[far] < 0.5).all()


def test_area_budget_and_edge_stars_and_empty_sources():
    rng = np.random.default_rng(1)
    stars = [(float(rng.uniform(0, H)), float(rng.uniform(0, W))) for _ in range(60)]
    _, img = _frame(stars[:3], wing_b=8000.0)
    src = _sources(stars)
    out = wing_aware_star_mask(img, src, None)
    assert (out >= 0.5).mean() <= Config.WING_MASK_MAX_AREA_FRAC + 0.05
    assert np.array_equal(wing_aware_star_mask(img, None, np.zeros((H, W), np.float32)),
                          np.zeros((H, W), np.float32))
    edge = _sources([(2, 3), (H - 1, W - 2)])
    wing_aware_star_mask(img, edge, None)          # windows clipped at the border: no crash


def test_dbe_with_wing_mask_leaves_less_wing_in_the_background_model():
    stars = [(150, 200), (420, 560), (300, 380), (520, 150), (130, 640)]
    sig, img0 = _frame(stars, wing_b=80000.0)
    rng = np.random.default_rng(7)
    img = np.stack([img0 * (0.9 + 0.1 * c) + rng.normal(0, 0.0, img0.shape) for c in range(3)],
                   axis=2).astype(np.float32)
    lum = img.mean(axis=2)
    src = _sources(stars)
    base = generate_star_mask((H, W), src, fwhm=4.0)
    wing = wing_aware_star_mask(lum, src, base)
    yy, xx = np.mgrid[0:H, 0:W]
    far = np.ones((H, W), bool)
    for cy, cx in stars:
        far &= np.hypot(yy - cy, xx - cx) > 120

    def run(mask):
        out = dynamic_background_extraction(img.copy(), star_mask=mask)
        err = out - sig[..., None] * np.array([0.9, 1.0, 1.1])
        err -= np.median(err, axis=(0, 1))
        ann = np.zeros((H, W), bool)
        for cy, cx in stars:
            r = np.hypot(yy - cy, xx - cx)
            ann |= (r > 10) & (r < 80)
        return float(err[ann].mean()), float(np.sqrt((err[far] ** 2).mean()))

    ann_base, far_base = run(base)
    ann_wing, far_wing = run(wing)
    # the compact mask lets the wing into the sky model: it digs a moat around the stars
    assert ann_base < 0
    assert ann_wing > ann_base
    assert far_wing <= far_base
