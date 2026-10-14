"""--denoiser nn (src/nn_denoise.py): numpy U-Net forward pass, tiling, normalisation, CLI.

The torch model in tools/train_nn_denoise.py is checked against ``forward`` at export time
(torch is not a test dependency); these pin the numpy side: its convolution is torch's
(cross-correlation, zero padding), tiling does not change the result, and the
normalisation is undone exactly.
"""
import numpy as np
import pytest
from scipy import ndimage as ndi

from src import nn_denoise as nd
from src.cli import parse_args


def _random_weights(f=4, c=3, scale=0.15, seed=0):
    rng = np.random.default_rng(seed)
    shapes = {'e0a': (f, c), 'e0b': (f, f), 'e1a': (2 * f, f), 'e1b': (2 * f, 2 * f),
              'ma': (4 * f, 2 * f), 'mb': (4 * f, 4 * f), 'd1a': (2 * f, 6 * f), 'd1b': (2 * f, 2 * f),
              'd0a': (f, 3 * f), 'd0b': (f, f)}
    w = {}
    for k, (o, i) in shapes.items():
        w[k + '.weight'] = rng.normal(0, scale, (o, i, 3, 3)).astype(np.float32)
        w[k + '.bias'] = rng.normal(0, 0.05, o).astype(np.float32)
    w['out.weight'] = rng.normal(0, scale, (c, f, 1, 1)).astype(np.float32)
    w['out.bias'] = np.zeros(c, np.float32)
    return w


def test_conv_is_torch_style_cross_correlation_with_zero_padding():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 9, 11)).astype(np.float32)
    w = rng.normal(size=(3, 2, 3, 3)).astype(np.float32)
    b = rng.normal(size=3).astype(np.float32)
    got = nd._conv3x3(x, w, b, act=False)
    ref = np.stack([sum(ndi.correlate(x[c], w[o, c], mode='constant') for c in range(2)) + b[o]
                    for o in range(3)])
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


def test_leaky_relu():
    out = nd._conv3x3(np.full((1, 4, 4), -1.0, np.float32), np.ones((1, 1, 3, 3), np.float32),
                      np.zeros(1, np.float32))
    assert out[0, 1, 1] == pytest.approx(-9 * nd.LEAK)


def test_tiling_matches_one_big_tile(monkeypatch):
    w = _random_weights()
    z = np.random.default_rng(2).normal(size=(300, 420, 3)).astype(np.float32)
    monkeypatch.setattr(nd, 'TILE', 128)
    tiled = nd.denoise_z(z, w)
    monkeypatch.setattr(nd, 'TILE', 512)
    whole = nd.denoise_z(z, w)
    np.testing.assert_allclose(tiled, whole, rtol=1e-4, atol=1e-4)


def test_zero_output_layer_is_identity_and_strength_zero_returns_input(tmp_path):
    w = _random_weights()
    w['out.weight'][:] = 0
    path = str(tmp_path / 'm.npz')
    np.savez(path, **w)
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[:200, :260]
    img = (1000 + 2 * xx + yy)[..., None] * np.array([1.0, 1.1, 0.9]) + rng.normal(0, 20, (200, 260, 3))
    img = img.astype(np.float32)
    np.testing.assert_allclose(nd.nn_denoise(img, model_path=path), img, rtol=1e-5, atol=1e-2)
    w2 = _random_weights()
    np.savez(path, **w2)
    nd._weights_cache.clear()
    np.testing.assert_array_equal(nd.nn_denoise(img, strength=0.0, model_path=path), img)


def test_normalisation_gives_unit_noise_around_zero():
    rng = np.random.default_rng(4)
    yy, xx = np.mgrid[:512, :640]
    sky = (500 + 0.8 * xx + 0.3 * yy)[..., None] * np.ones(3)
    img = (sky + rng.normal(0, [10.0, 20.0, 40.0], (512, 640, 3))).astype(np.float32)
    grid, sigma = nd.normalise(img)
    np.testing.assert_allclose(sigma, [10, 20, 40], rtol=0.05)
    z = (img - nd.background_at(grid, 0, 0, 512, 640)) / sigma.astype(np.float32)
    inner = z[160:-160, 160:-160]          # the mesh smoothing bends a steep ramp near the edges
    assert np.abs(np.median(inner, axis=(0, 1))).max() < 0.05
    np.testing.assert_allclose(inner.std(axis=(0, 1)), 1.0, rtol=0.05)


def test_nan_pixels_pass_through(tmp_path):
    w = _random_weights()
    path = str(tmp_path / 'm.npz')
    np.savez(path, **w)
    img = np.random.default_rng(5).normal(100, 5, (130, 140, 3)).astype(np.float32)
    img[10, 20, 1] = np.nan
    out = nd.nn_denoise(img, model_path=path)
    assert np.isnan(out[10, 20, 1]) and np.isfinite(np.delete(out.ravel(), 10 * 140 * 3 + 20 * 3 + 1)).all()


@pytest.mark.skipif(not nd.model_available(), reason="bundled model not present")
def test_bundled_model_lowers_noise_and_keeps_stars():
    from tests._helpers import star_field
    rng = np.random.default_rng(6)
    stars = [(rng.uniform(20, 236), rng.uniform(20, 300), rng.uniform(2000, 20000)) for _ in range(25)]
    clean = star_field((256, 320), stars, 1.5, bg=800.0, noise=0.0, rng=rng, dtype=np.float64)
    noisy = clean + rng.normal(0, 30, clean.shape)
    img = np.stack([noisy] * 3, -1).astype(np.float32)
    out = nd.nn_denoise(img)
    err_in = np.abs(img[..., 1] - clean).mean()
    err_out = np.abs(out[..., 1] - clean).mean()
    assert err_out < 0.7 * err_in
    flux = lambda a: sum(a[int(y) - 6:int(y) + 7, int(x) - 6:int(x) + 7].sum() - 169 * 800  # noqa: E731
                         for y, x, _ in stars)
    assert flux(out[..., 1]) == pytest.approx(flux(clean), rel=0.03)


def test_cli_selects_only_the_nn_denoiser():
    a = parse_args(['-d', 'x', '-o', 'y.fits', '--denoiser', 'nn'])
    assert a.denoise_nn is True
    assert not (a.denoise_curvelet or a.denoise_acdnr or a.denoise_bilateral or a.denoise_aniso)
    assert 'denoise_nn' in a._explicit_cli_dests
    assert parse_args(['-d', 'x', '-o', 'y.fits']).denoise_nn is False


def test_preview_stretch_uses_the_noise_before_denoising():
    """--denoiser nn removes ~90% of the sky noise; the preview stretch is solved from it,
    so Phase 4 records the ratio and the preview scales its measured sigma back up."""
    import argparse

    from src.io_fits import _sky_stats, preview_stretch_kw, render_preview_float
    rng = np.random.default_rng(9)
    img = (500 + rng.normal(0, 5, (200, 200, 3))).astype(np.float32)
    img[90:110, 90:110] += 400
    lum = img.mean(-1)
    assert _sky_stats(lum, 4.0)[1] == pytest.approx(4.0 * _sky_stats(lum)[1])
    assert preview_stretch_kw(None)['noise_scale'] == 1.0
    assert preview_stretch_kw(argparse.Namespace(_preview_noise_scale=7.5))['noise_scale'] == 7.5
    soft = render_preview_float(img, stretch='noisefloor', noise_scale=10.0)
    hard = render_preview_float(img, stretch='noisefloor')
    assert soft[:50, :50].std() < 0.5 * hard[:50, :50].std()  # sky noise far less amplified
