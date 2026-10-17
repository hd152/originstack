"""Learned star removal (src/nn_starless.py): the forward pass and the guarantees of the
runtime path, with small random weights (independent of the trained model)."""
import numpy as np
import pytest

from src import nn_denoise as nd
from src import nn_starless as ns
from src import star_removal


def _weights(depth, widths, mid, seed=0, scale=0.05):
    rng = np.random.default_rng(seed)
    w = {}

    def conv(name, i, o):
        w[name + '.weight'] = (rng.standard_normal((o, i, 3, 3)) * scale).astype(np.float32)
        w[name + '.bias'] = (rng.standard_normal(o) * scale).astype(np.float32)

    prev = 3
    for k in range(depth):
        conv(f'e{k}a', prev, widths[k])
        conv(f'e{k}b', widths[k], widths[k])
        prev = widths[k]
    conv('ma', prev, mid)
    conv('mb', mid, mid)
    up = mid
    for k in reversed(range(depth)):
        conv(f'd{k}a', up + widths[k], widths[k])
        conv(f'd{k}b', widths[k], widths[k])
        up = widths[k]
    w['out.weight'] = (rng.standard_normal((3, widths[0], 1, 1)) * scale).astype(np.float32)
    w['out.bias'] = np.zeros(3, np.float32)
    return w


def test_depth_two_is_the_denoiser_network():
    w = _weights(2, (8, 16), 32)
    x = np.random.default_rng(1).normal(0, 1, (3, 32, 48)).astype(np.float32)
    assert ns.depth_of(w) == 2
    np.testing.assert_array_equal(ns.forward(x, w), nd.forward(x, w))


def test_forward_shape_at_depth_four():
    w = _weights(4, (8, 8, 16, 16), 16)
    x = np.random.default_rng(2).normal(0, 1, (3, 64, 80)).astype(np.float32)
    assert ns.forward(x, w).shape == (3, 64, 80)


@pytest.fixture
def model(tmp_path):
    p = tmp_path / 'w.npz'
    np.savez(p, **_weights(4, (8, 8, 16, 16), 16, seed=3, scale=0.3))
    return str(p)


def _field(h=200, w=300, seed=4):
    rng = np.random.default_rng(seed)
    img = rng.normal(500.0, 10.0, (h, w, 3)).astype(np.float32)
    yy, xx = np.mgrid[:h, :w]
    for _ in range(30):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        img += (rng.uniform(100, 5000) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 4.5))[..., None]
    return img


def test_only_light_is_removed_and_layers_sum_to_the_input(model):
    img = _field()
    starless, stars = ns.remove_stars_nn(img, model)
    assert starless.shape == img.shape and starless.dtype == np.float32
    assert np.all(starless <= img)
    assert np.all(stars >= 0)
    # exact up to float32 rounding of the larger layer (random weights make it large)
    tol = 4 * np.finfo(np.float32).eps * max(np.abs(starless).max(), np.abs(img).max())
    np.testing.assert_allclose(starless + stars, img, rtol=0, atol=tol)


def test_non_finite_pixels_pass_through(model):
    img = _field()
    img[10, 20, 1] = np.nan
    img[50, 60, :] = np.inf
    starless, stars = ns.remove_stars_nn(img, model)
    assert np.isnan(starless[10, 20, 1]) and np.isinf(starless[50, 60]).all()
    assert stars[10, 20, 1] == 0 and np.all(stars[50, 60] == 0)
    assert np.isfinite(starless[100, 100]).all()


def test_image_sizes_not_a_multiple_of_the_tile(model, monkeypatch):
    monkeypatch.setattr(ns, 'TILE', 64)
    monkeypatch.setattr(ns, 'MARGIN', 16)
    img = _field(131, 157)
    starless, _ = ns.remove_stars_nn(img, model)
    assert starless.shape == img.shape and np.isfinite(starless).all()


def test_missing_weights_fall_back_to_the_inpaint(monkeypatch):
    monkeypatch.setattr(ns, 'model_available', lambda path=None: False)
    img = _field()
    out, n = star_removal.remove_stars(img, None, 3.0, method='nn')
    # no detections were passed, so the inpaint has nothing to do
    assert n == 0 and np.array_equal(out, img)
