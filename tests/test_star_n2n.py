"""Noisier2Noise star removal (src/star_n2n.py) and its tools: injection model, the
correction formula, an identity network, the bank builder and the evaluation metrics.

No trained model is needed: the network checks use a zero output layer (f(z) = z), which
must give back the input exactly, and an oracle in place of the network for the formula.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
import eval_star_n2n as ev  # noqa: E402
import gen_starless_bank as gb  # noqa: E402

from src import nn_denoise as nd  # noqa: E402
from src import star_n2n as sn  # noqa: E402


def _synthetic(seed=0, h=320, w=448, n=220):
    """Linear (H, W, 3) ADU image: sky 200 + gradient, a nebula blob, Gaussian stars with a
    power-law flux distribution (three clipped at 4000 ADU), sky-limited noise sigma 5."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.empty((h, w, 3), np.float32)
    neb = 60 * np.exp(-((yy - h / 2) ** 2 + (xx - w / 3) ** 2) / (2 * 60.0 ** 2))
    for c in range(3):
        img[..., c] = 200 + 0.02 * xx + neb * (1.0, 0.6, 0.8)[c]
    flux = 300 * (1 - rng.random(n)) ** (-1 / 1.2)
    flux = np.minimum(flux, 60000)
    ys, xs = rng.uniform(8, h - 8, n), rng.uniform(8, w - 8, n)
    for f, y, x in zip(flux, ys, xs):
        y0, y1, x0, x1 = int(y) - 8, int(y) + 9, int(x) - 8, int(x) + 9
        p = np.exp(-((yy[y0:y1, x0:x1] - y) ** 2 + (xx[y0:y1, x0:x1] - x) ** 2) / (2 * 1.5 ** 2))
        p /= 2 * np.pi * 1.5 ** 2
        col = (1.0, 0.9, 0.7)
        for c in range(3):
            img[y0:y1, x0:x1, c] += f * col[c] * p
    img += rng.normal(0, 5, img.shape).astype(np.float32)
    return np.minimum(img, 4000.0).astype(np.float32)


def _zero_out_weights(path, seed=0):
    """Weights for the nn_denoise U-Net with random hidden layers and a zero output layer,
    so forward() returns 0 and denoise_z(z) == z."""
    rng = np.random.default_rng(seed)
    f, c = 24, 3
    shapes = {'e0a': (f, c), 'e0b': (f, f), 'e1a': (2 * f, f), 'e1b': (2 * f, 2 * f),
              'ma': (4 * f, 2 * f), 'mb': (4 * f, 4 * f), 'd1a': (2 * f, 6 * f), 'd1b': (2 * f, 2 * f),
              'd0a': (f, 3 * f), 'd0b': (f, f)}
    w = {}
    for k, (o, i) in shapes.items():
        w[k + '.weight'] = (0.1 * rng.standard_normal((o, i, 3, 3))).astype(np.float32)
        w[k + '.bias'] = np.zeros(o, np.float32)
    w['out.weight'] = np.zeros((c, f, 1, 1), np.float32)
    w['out.bias'] = np.zeros(c, np.float32)
    np.savez(path, **w)
    return str(path)


@pytest.fixture(scope='module')
def field_and_z():
    z, _, _, B = sn.normalise_image(_synthetic())
    return sn.StarField(z, B), z, B


def test_star_field_models_the_image(field_and_z):
    field, _, _ = field_and_z
    assert field.ok
    assert field.n_donors >= 50
    assert field.saturated.sum() >= 1                 # the clipped stars are found and kept out
    sums = field.stamps.sum((1, 2))
    np.testing.assert_allclose(sums, 1.0, rtol=1e-4)  # unit flux per channel
    assert np.isclose(field.density.sum(), 1.0)


def test_inject_count_flux_and_noise(field_and_z):
    field, _, B = field_and_z
    rng = np.random.default_rng(3)
    s, ys, xs, flux = field.inject(rng, alpha=1.0)
    n = field.n_donors
    assert abs(len(ys) - n) < 5 * np.sqrt(n)          # Poisson(alpha * donors)
    # total injected flux (edge clipping only loses a little)
    assert 0.8 < s.sum() / flux.sum() <= 1.05
    # photon noise is there: far from any injected star the layer is exactly 0
    assert (s == 0).mean() > 0.3
    assert len(field.inject(rng, alpha=0.0)[1]) == 0


def test_correction_formula_with_oracle(monkeypatch):
    """With f = E[y|z] for a known split, the estimate is exactly the starless sky."""
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (64, 64, 3)).astype(np.float32)
    stars = np.abs(rng.normal(0, 20, (64, 64, 3))).astype(np.float32)
    z = x + stars
    for alpha in (0.5, 1.0):
        # given z, a fraction alpha / (1 + alpha) of the stars is the injected layer
        monkeypatch.setattr(sn.nd, 'denoise_z', lambda zz, w, a=alpha: zz - a / (1 + a) * stars)
        np.testing.assert_allclose(sn.starless_z(z, None, alpha), x, atol=1e-4)


def test_identity_network_returns_input(tmp_path):
    img = _synthetic(seed=2)
    path = _zero_out_weights(tmp_path / 'zero.npz')
    starless, stars = sn.remove_stars_nn(img, model_path=path, alpha=0.5)
    np.testing.assert_allclose(starless, img, atol=1e-3 * float(np.abs(img).max()))
    assert np.abs(stars).max() < 1e-3 * float(np.abs(img).max())
    nd._weights_cache.pop(path, None)


def test_non_finite_pixels_pass_through(tmp_path):
    img = _synthetic(seed=4)
    img[10:20, 10:20] = np.nan
    path = _zero_out_weights(tmp_path / 'zero.npz')
    starless, stars = sn.remove_stars_nn(img, model_path=path)
    assert np.isnan(starless[10:20, 10:20]).all()
    assert np.isfinite(stars).all()
    nd._weights_cache.pop(path, None)


def test_bank_patches():
    rng = np.random.default_rng(0)
    got, field = gb.session_patches(_synthetic(seed=5), rng, per_session=12, alpha=1.0, realisations=3)
    za, zb, w = got
    assert len(za) == len(zb) == len(w) == 12
    assert za[0].shape == (gb.PATCH, gb.PATCH, 3) and za[0].dtype == np.float16
    diff = np.stack([a.astype(np.float32) - b.astype(np.float32) for a, b in zip(za, zb)])
    assert diff.sum() > 0                             # input = target + injected stars
    assert all(np.isfinite(x.astype(np.float32)).all() and (x > 0).all() for x in w)


def test_bank_skips_starless_image():
    rng = np.random.default_rng(0)
    flat = np.random.default_rng(1).normal(200, 5, (320, 448, 3)).astype(np.float32)
    got, field = gb.session_patches(flat, rng, per_session=4, alpha=1.0, realisations=1)
    assert got is None and not field.ok


def test_eval_metrics_on_identity_model(tmp_path):
    """An identity model removes nothing: injected and real stars stay (fraction ~1), faint
    discs are kept (~1), and the sky does not move."""
    p = tmp_path / 'stack.npy'
    np.save(p, _synthetic(seed=6))
    w = nd.load_weights(_zero_out_weights(tmp_path / 'zero.npz'))
    r = ev.evaluate(str(p), w, alpha=1.0, test_alpha=0.3, seed=0)
    assert r['inject_left']['all'] == pytest.approx(1.0, abs=0.05)
    assert r['real_left']['all'] == pytest.approx(1.0, abs=0.05)
    assert r['faint_kept'] == pytest.approx(1.0, abs=0.05)
    assert abs(r['sky_shift']) < 1e-3 and r['sky_mad'] < 1e-3
