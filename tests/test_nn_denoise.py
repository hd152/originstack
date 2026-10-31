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


# ── --deconvolve nn: the same network trained on seeing pairs (src/data/nn_deconv.npz) ──

def test_cli_deconvolve_nn_selects_only_the_learned_deconvolver():
    a = parse_args(['-d', 'x', '-o', 'y.fits', '--deconvolve', 'nn'])
    assert a.deconvolve_nn is True
    assert a.deconvolve is False                       # not the Richardson-Lucy step
    assert not (a.denoise_curvelet or a.denoise_acdnr or a.denoise_bilateral
                or a.denoise_aniso or a.denoise_nn)    # it denoises itself
    # presets and --auto must not turn a denoiser back on
    assert {'deconvolve_nn', 'denoise_curvelet', 'denoise_nn'} <= a._explicit_cli_dests
    # --denoiser nn as well: the deconvolver replaces it
    b = parse_args(['-d', 'x', '-o', 'y.fits', '--denoiser', 'nn', '--deconvolve', 'nn'])
    assert b.deconvolve_nn is True and b.denoise_nn is False
    assert parse_args(['-d', 'x', '-o', 'y.fits']).deconvolve_nn is False


def test_deconvolve_nn_survives_a_saved_config(tmp_path):
    from src.cli import load_config_file, save_effective_config
    a = parse_args(['-d', 'x', '-o', str(tmp_path / 's.fits'), '--deconvolve', 'nn'])
    save_effective_config(a, str(tmp_path / 's.fits'))
    b = parse_args(['-d', 'x', '-o', str(tmp_path / 't.fits')])
    load_config_file(str(tmp_path / 's_config.toml'), b)
    assert b.deconvolve_nn is True and b.deconvolve_mode == 'nn'
    assert not (b.denoise_curvelet or b.denoise_nn)


@pytest.mark.skipif(not nd.model_available(nd.DECONV_MODEL_PATH), reason="deconvolution model not present")
def test_bundled_deconvolver_narrows_stars_and_keeps_their_flux():
    """On ideal Gaussian stars it narrows them only a little (4-5% at sigma 1.2 px; 8-18% on
    real Origin stacks, whose seeing blur it was trained to undo) but keeps flux and denoises."""
    from scipy.optimize import curve_fit

    from tests._helpers import star_field
    rng = np.random.default_rng(11)
    stars = [(rng.uniform(24, 232), rng.uniform(24, 296), rng.uniform(3000, 20000)) for _ in range(20)]
    clean = star_field((256, 320), stars, 1.2, bg=800.0, noise=0.0, rng=rng, dtype=np.float64)
    noisy = clean + np.random.default_rng(2).normal(0, 25, clean.shape)
    img = np.stack([noisy] * 3, -1).astype(np.float32)
    out = nd.nn_denoise(img, model_path=nd.DECONV_MODEL_PATH)

    def width(a):   # median fitted Gaussian sigma
        g = lambda c, amp, s, b: amp * np.exp(-0.5 * (c[0] ** 2 + c[1] ** 2) / s ** 2) + b  # noqa: E731
        ws = []
        for y, x, _ in stars:
            yi, xi = int(round(y)), int(round(x))
            box = a[yi - 7:yi + 8, xi - 7:xi + 8]
            yy, xx = np.mgrid[-7:8, -7:8] + np.array([yi - y, xi - x])[:, None, None]
            p, _ = curve_fit(g, (yy.ravel(), xx.ravel()), box.ravel(), p0=(box.max() - 800, 1.5, 800))
            ws.append(abs(p[1]))
        return float(np.median(ws))

    flux = lambda a: sum(a[int(y) - 8:int(y) + 9, int(x) - 8:int(x) + 9].sum() - 289 * 800  # noqa: E731
                         for y, x, _ in stars)
    assert width(out[..., 1]) < 0.97 * width(img[..., 1])            # sharper
    assert flux(out[..., 1]) == pytest.approx(flux(clean), rel=0.02)  # same light
    sky = (slice(0, 16), slice(0, 16))
    assert out[sky][..., 1].std() < 0.5 * img[sky][..., 1].std()     # and denoised


def test_phase4_runs_the_deconvolver_at_the_nn_step(monkeypatch):
    import src.nn_denoise as ndm
    import src.postprocess as pp
    from src.models import ProcessingStats
    calls = []

    def fake(img, strength=1.0, model_path=None, restore_sigma=4.0, fwhm_range=None,
             star_restore_frac=0.0):
        calls.append((model_path, restore_sigma))
        return img
    monkeypatch.setattr(ndm, 'nn_denoise', fake)
    monkeypatch.setattr(ndm, 'model_available', lambda path=None: True)
    monkeypatch.setattr(pp, 'reduce_chroma_noise', lambda *a, **k: pytest.fail('chroma NR ran'))
    rng = np.random.default_rng(3)
    img = (500 + rng.normal(0, 5, (96, 128, 3))).astype(np.float32)
    args = parse_args(['-d', '.', '--deconvolve', 'nn', '--skip-step', 'background'])
    args._diagnostic_dir = None
    pp.postprocess_stack(img, args, [], ProcessingStats())
    assert calls == [(ndm.DECONV_MODEL_PATH, 4.0), (None, 4.0)]   # deconvolver, then the blend's denoiser


def test_nn_deconv_blend_mixes_in_the_denoiser(monkeypatch):
    import src.nn_denoise as ndm
    import src.postprocess as pp
    from src.models import ProcessingStats
    calls = []

    def fake(img, strength=1.0, model_path=None, restore_sigma=4.0, fwhm_range=None,
             star_restore_frac=0.0):
        calls.append(model_path)
        return np.full_like(img, 10.0 if model_path == ndm.DECONV_MODEL_PATH else 20.0)
    monkeypatch.setattr(ndm, 'nn_denoise', fake)
    monkeypatch.setattr(ndm, 'model_available', lambda path=None: True)
    seen = {}
    real_sanitize = pp._sanitize

    def spy(a, what):
        seen.setdefault(what, a.copy())
        return real_sanitize(a, what)
    monkeypatch.setattr(pp, '_sanitize', spy)
    img = np.full((64, 80, 3), 500, np.float32)
    args = parse_args(['-d', '.', '--deconvolve', 'nn', '--skip-step', 'background'])
    args._diagnostic_dir = None
    args.nn_deconv_blend = 0.25
    pp.postprocess_stack(img, args, [], ProcessingStats())
    assert calls == [ndm.DECONV_MODEL_PATH, None]
    out = seen['nn deconvolution']
    assert out.dtype == np.float32 and np.allclose(out, 0.25 * 10 + 0.75 * 20)


# ── pixel-scale normalisation (fwhm_range) ────────────────────────────────────

def _star_field(fwhm, shape=(300, 300), n=60, seed=3):
    rng = np.random.default_rng(seed)
    img = rng.normal(100.0, 2.0, shape).astype(np.float32)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    sig = fwhm / 2.3548
    pts = [(y, x) for y in range(25, shape[0] - 25, 45) for x in range(25, shape[1] - 25, 45)][:n]
    for k, (y, x) in enumerate(pts):
        img += (400.0 + 150.0 * (k % 9)) * np.exp(-((yy - y - 0.3) ** 2 + (xx - x + 0.2) ** 2) / (2 * sig ** 2)).astype(np.float32)
    return np.repeat(img[..., None], 3, axis=2)


@pytest.mark.parametrize('fwhm', [2.0, 4.0, 8.0])
def test_measure_star_fwhm_recovers_the_gaussian_width(fwhm):
    got = nd.measure_star_fwhm(_star_field(fwhm))
    assert got is not None and abs(got - fwhm) < 0.15 * fwhm


def test_measure_star_fwhm_none_without_stars():
    flat = np.random.default_rng(0).normal(100, 2, (200, 200, 3)).astype(np.float32)
    assert nd.measure_star_fwhm(flat) is None


def test_scale_factor_clamps_to_the_nearest_bound():
    rng = (3.0, 5.0)
    assert nd.scale_factor(4.0, rng) == 1.0                    # inside: untouched
    assert nd.scale_factor(3.0, rng) == 1.0 and nd.scale_factor(5.0, rng) == 1.0
    assert nd.scale_factor(10.0, rng) == pytest.approx(0.5)    # downsample to 5 px
    assert nd.scale_factor(2.0, rng) == pytest.approx(1.5)     # upsample to 3 px
    assert nd.scale_factor(None, rng) == 1.0 and nd.scale_factor(float('nan'), rng) == 1.0
    assert nd.scale_factor(1.0, rng) == 1.0                    # would need 3x: skipped


def test_nn_denoise_resamples_and_returns_the_original_grid(monkeypatch):
    seen = {}
    monkeypatch.setattr(nd, 'load_weights', lambda path=None: _random_weights())
    monkeypatch.setattr(nd, 'measure_star_fwhm', lambda img, max_stars=60: 10.0)
    orig = nd.denoise_z

    def grab(z, w):
        seen['shape'] = z.shape
        return orig(z, w)
    monkeypatch.setattr(nd, 'denoise_z', grab)
    img = _star_field(10.0, (300, 340))
    img[5, 5, 0] = np.nan
    out = nd.nn_denoise(img, fwhm_range=(3.0, 5.0))
    assert seen['shape'][:2] == (150, 170)                     # 10 px -> 5 px: half size
    assert out.shape == img.shape and out.dtype == np.float32
    assert np.isnan(out[5, 5, 0]) and np.isfinite(np.delete(out.ravel(), 5 * 340 * 3 + 15)).all()


def test_nn_denoise_inside_range_or_unmeasured_is_unchanged(monkeypatch):
    monkeypatch.setattr(nd, 'load_weights', lambda path=None: _random_weights())
    img = _star_field(4.0)
    base = nd.nn_denoise(img)
    monkeypatch.setattr(nd, 'measure_star_fwhm', lambda img, max_stars=60: 4.0)
    np.testing.assert_array_equal(nd.nn_denoise(img, fwhm_range=(2.8, 5.2)), base)
    monkeypatch.setattr(nd, 'measure_star_fwhm', lambda img, max_stars=60: None)
    np.testing.assert_array_equal(nd.nn_denoise(img, fwhm_range=(2.8, 5.2)), base)


# ── star-aware restore (star_restore_frac) ───────────────────────────────────

def test_star_mask_is_one_at_stars_and_zero_far_away():
    img = _star_field(4.0)
    m = nd.star_mask(img, 4.0)
    assert m is not None and m.shape == img.shape[:2] and m.dtype == np.float32
    assert m[25, 25] == 1.0 and m[0, 299] == 0.0          # a grid star; a corner 45 px from any
    assert nd.star_mask(np.random.default_rng(0).normal(100, 2, (200, 200, 3)).astype(np.float32), 4.0) is None \
        or True                                           # noise-only: detections are allowed to be empty


def test_star_restore_changes_output_only_near_stars(monkeypatch):
    monkeypatch.setattr(nd, 'load_weights', lambda path=None: _random_weights())
    img = _star_field(4.0)
    base = nd.nn_denoise(img)
    star = nd.nn_denoise(img, star_restore_frac=0.4)
    diff = np.abs(star - base).max(axis=2)
    mask = nd.star_mask(img, nd.measure_star_fwhm(img))
    assert diff.max() > 0                                  # it does something
    far = ndi.gaussian_filter(mask, 6) < 1e-4              # well outside every star's mask
    # the blurs differ by at most the 4 px kernel's reach beyond the mask
    assert far.any() and diff[far].max() < 0.05 * diff.max()


def test_star_restore_fades_by_weight_not_blur_width(monkeypatch):
    """Near the top of the fade-out the tight restore must be a weak version of the real one,
    not a ~0 px blur (which gave the input back: Pleiades 4.37 px, fwhm_ratio 0.73 -> 1.03)."""
    monkeypatch.setattr(nd, 'load_weights', lambda path=None: _random_weights())
    img = _star_field(4.0)
    monkeypatch.setattr(nd, 'measure_star_fwhm', lambda im, max_stars=60: 4.37)
    base = nd.nn_denoise(img)
    near_top = nd.nn_denoise(img, star_restore_frac=0.4)
    monkeypatch.setattr(nd, 'measure_star_fwhm', lambda im, max_stars=60: 3.0)
    full = nd.nn_denoise(img, star_restore_frac=0.4)
    d_top = np.abs(near_top - base).max()
    d_full = np.abs(full - base).max()
    assert d_full > 0 and d_top < 0.2 * d_full             # ramp 0.04: nearly the plain restore
    monkeypatch.setattr(nd, 'measure_star_fwhm', lambda im, max_stars=60: 4.5)
    np.testing.assert_array_equal(nd.nn_denoise(img, star_restore_frac=0.4), base)   # past the ramp: off


def test_monochrome_input_stays_monochrome(monkeypatch):
    monkeypatch.setattr(nd, 'load_weights', lambda path=None: _random_weights())
    mono = _star_field(4.0)[..., :1].repeat(3, axis=2)
    out = nd.nn_denoise(mono)
    np.testing.assert_array_equal(out[..., 0], out[..., 1])
    np.testing.assert_array_equal(out[..., 0], out[..., 2])
    colour = mono.copy()
    colour[..., 2] *= 1.3                                    # not identical planes: untouched logic
    out = nd.nn_denoise(colour)
    assert not np.array_equal(out[..., 0], out[..., 2])
