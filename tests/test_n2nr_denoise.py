"""Noisier2Noise denoiser training data and inference (tools/train_nn_denoise.py --mode n2nr,
src/nn_denoise.py with ``_n2nr_alpha`` weights)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
import train_nn_denoise as tnd  # noqa: E402

from src import nn_denoise as nd  # noqa: E402


def _quarters(tmp_path, sigma=10.0, seed=0):
    """Four quarter-stacks of one sky with independent noise (sky + a ramp + a blob)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:128, 0:160].astype(np.float32)
    sky = 500 + 0.5 * xx + 300 * np.exp(-((yy - 64) ** 2 + (xx - 80) ** 2) / 200.0)
    sky = np.repeat(sky[..., None], 3, axis=2).astype(np.float32)
    for i in range(4):
        np.save(tmp_path / f"q{i}.npy", sky + rng.normal(0, sigma, sky.shape).astype(np.float32))
    return sky


def test_contrasts_are_noise_like_the_full_stacks(tmp_path):
    sky = _quarters(tmp_path)
    pairs = tnd.n2nr_pairs(str(tmp_path), alpha=1.0)
    assert len(pairs) == 3
    full = pairs[0][1]
    full_noise = full - sky                                  # sigma / 2
    for z, f in pairs:
        np.testing.assert_array_equal(f, full)              # the target is the full-depth stack
        n = z - f                                            # alpha * contrast
        assert abs(float(n.mean())) < 0.1                    # no signal left in it
        assert abs(float(n.std()) / float(full_noise.std()) - 1.0) < 0.05
        assert abs(np.corrcoef(n.ravel(), full_noise.ravel())[0, 1]) < 0.05


def test_alpha_scales_the_injected_noise(tmp_path):
    _quarters(tmp_path, seed=1)
    a1 = tnd.n2nr_pairs(str(tmp_path), alpha=1.0)
    a5 = tnd.n2nr_pairs(str(tmp_path), alpha=0.5)
    for (z1, f), (z5, _) in zip(a1, a5):
        np.testing.assert_allclose(z5 - f, 0.5 * (z1 - f), atol=1e-3)


def _zero_net(path, alpha=None):
    rng = np.random.default_rng(0)
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
    if alpha is not None:
        w['_n2nr_alpha'] = np.array([alpha], np.float32)
    np.savez(path, **w)
    return str(path)


def test_identity_network_with_alpha_returns_the_input(tmp_path):
    """f(y) = y gives ((1 + a^2) y - y) / a^2 = y: the formula itself adds nothing."""
    rng = np.random.default_rng(2)
    img = (500 + rng.normal(0, 10, (96, 128, 3))).astype(np.float32)
    path = _zero_net(tmp_path / 'z.npz', alpha=0.5)
    out = nd.nn_denoise(img, model_path=path, restore_sigma=0)
    np.testing.assert_allclose(out, img, atol=1e-3)
    nd._weights_cache.pop(path, None)


def test_alpha_formula_applies_only_to_n2nr_weights(tmp_path, monkeypatch):
    """f(y) = y / 2 in noise units: plain weights return it, alpha 1 returns 2 f - y = 0."""
    rng = np.random.default_rng(3)
    img = (500 + rng.normal(0, 10, (96, 128, 3))).astype(np.float32)
    monkeypatch.setattr(nd, 'denoise_z', lambda z, w: z * np.float32(0.5))
    for alpha, expect in ((None, 0.5), (1.0, 0.0)):
        path = _zero_net(tmp_path / f'h{alpha}.npz', alpha=alpha)
        grid, sigma = nd.normalise(img)
        bg = nd.background_at(grid, 0, 0, *img.shape[:2])
        z = (img - bg) / sigma.astype(np.float32)
        out = nd.nn_denoise(img, model_path=path, restore_sigma=0)
        got = (out - bg) / sigma.astype(np.float32)
        np.testing.assert_allclose(got, expect * z, atol=1e-3)
        nd._weights_cache.pop(path, None)
