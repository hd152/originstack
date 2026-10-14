"""Preview stretches noisefloor, oklab and clahe (src/io_fits.py)."""
import numpy as np
import pytest

import src.io_fits as iof
from src.models import Config

W = np.array([0.299, 0.587, 0.114])


def _scene(noise=20.0, h=400, w=600, seed=0):
    """Sky + a reddish nebula with ripples + a red, a blue and a white star."""
    rng = np.random.default_rng(seed)
    img = (500.0 + rng.normal(0, noise, (h, w, 3))).astype(np.float32)
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - 200, xx - 300)
    neb = 300.0 * np.exp(-(r / 90.0) ** 2) * (1 + 0.4 * np.sin(xx / 6.0))
    img += (neb[..., None] * np.array([1.0, 0.5, 0.6])).astype(np.float32)
    for y, x, peak, col in [(60, 80, 30000, (1.0, 0.4, 0.2)),
                            (330, 500, 20000, (0.3, 0.5, 1.0)),
                            (200, 300, 8000, (1.0, 1.0, 1.0))]:
        g = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.3 ** 2))
        img += ((peak * g)[..., None] * np.array(col)).astype(np.float32)
    sky = (r > 250) & (np.hypot(yy - 60, xx - 80) > 15) & (np.hypot(yy - 330, xx - 500) > 15)
    return img, r, sky


def _assert_display(out):
    assert np.all(np.isfinite(out)) and out.min() >= 0.0 and out.max() <= 1.0


def _sat(p):
    return (p.max(axis=-1) - p.min(axis=-1)) / np.maximum(p.max(axis=-1), 1e-9)


# ── noisefloor ─────────────────────────────────────────────────────────────

def _galaxy(noise, h=400, w=600, seed=0):
    """Mostly sky (so the sky sigma estimate is the true one) + a compact
    bright galaxy, putting the white point far above 50 sigma."""
    rng = np.random.default_rng(seed)
    img = (500.0 + rng.normal(0, noise, (h, w, 3))).astype(np.float32)
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - 200, xx - 300)
    img += (20000.0 * np.exp(-(r / 25.0) ** 2))[..., None].astype(np.float32)
    return img, r > 150


def test_noisefloor_holds_sky_noise_at_target_across_noise_levels():
    stds, meds = [], []
    for noise in (10.0, 40.0):
        img, sky = _galaxy(noise)
        original = img.copy()
        lum = iof.render_preview_float(img, 'noisefloor') @ W
        np.testing.assert_array_equal(img, original)
        stds.append(np.std(lum[sky]) * 255)
        meds.append(np.median(lum[sky]) * 255)
    # The target is in units of the *estimated* sky sigma (_sky_stats, which
    # reads ~30% high at the lowest noise here), so allow a band around it.
    for s in stds:
        assert 0.6 * Config.PREVIEW_NOISE_LEVELS < s < 1.5 * Config.PREVIEW_NOISE_LEVELS
    assert abs(meds[0] - meds[1]) < 3.0
    # Steadier than the fixed ghs curve over a 4x change in noise.
    ghs = []
    for noise in (10.0, 40.0):
        img, sky = _galaxy(noise)
        ghs.append(np.std((iof.render_preview_float(img, 'ghs') @ W)[sky]) * 255)
    assert max(stds) / min(stds) < 1.3 < ghs[1] / ghs[0]


def test_noise_limited_b_meets_target_amplitude():
    sky, sigma, black, white, hp = 500.0, 10.0, 475.0, 20000.0, 0.95
    b = iof._noise_limited_b(sky, sigma, black, white, hp)
    span = white - black
    x, d = (sky - black) / span, sigma / span
    amp = 0.5 * (np.arcsinh(b * (x + d)) - np.arcsinh(b * (x - d))) / np.arcsinh(b * hp)
    assert amp * 255 == pytest.approx(Config.PREVIEW_NOISE_LEVELS, rel=1e-3)


def test_noise_limited_b_is_linear_when_noise_already_visible():
    # White only 20 sigma above sky: a linear map already shows the noise large.
    assert iof._noise_limited_b(500.0, 10.0, 475.0, 700.0, 0.95) == 0.0


# ── oklab ──────────────────────────────────────────────────────────────────

def test_oklab_grey_matches_ghs():
    img, _r, _sky = _scene()
    grey = np.repeat((img @ W.astype(np.float32))[..., None], 3, axis=2).astype(np.float32)
    np.testing.assert_allclose(iof.render_preview_float(grey, 'oklab'),
                               iof.render_preview_float(grey, 'ghs'), atol=2e-4)


def test_oklab_star_cores_brighter_and_still_coloured():
    img, r, _sky = _scene()
    g = iof.render_preview_float(img, 'ghs')
    o = iof.render_preview_float(img, 'oklab')
    _assert_display(o)
    for y, x in ((60, 80), (330, 500)):
        # ghs keeps the ratios by dimming the core to fit the gamut
        assert o[y, x] @ W > g[y, x] @ W + 0.1
        assert _sat(o[y, x]) > 0.3
        assert np.argmax(o[y, x]) == np.argmax(g[y, x])          # same dominant hue
    # mid-tones keep the ghs preview's colour
    neb = (r > 20) & (r < 80)
    assert abs(_sat(o[neb]).mean() - _sat(g[neb]).mean()) < 0.02


def test_oklab_roundtrip():
    rng = np.random.default_rng(3)
    rgb = rng.uniform(0.0, 1.0, (50, 50, 3))
    np.testing.assert_allclose(iof._oklab_to_linear(iof._linear_to_oklab(rgb)), rgb, atol=1e-9)


# ── clahe ──────────────────────────────────────────────────────────────────

def test_clahe_lifts_nebula_contrast_but_not_sky_or_stars():
    img, r, sky = _scene()
    g = iof.render_preview_float(img, 'ghs')
    c = iof.render_preview_float(img, 'clahe')
    _assert_display(c)
    gl, cl = g @ W, c @ W
    assert abs(cl[sky].mean() - gl[sky].mean()) * 255 < 0.5
    neb = (r > 20) & (r < 80)
    assert cl[neb].std() > 1.2 * gl[neb].std()
    np.testing.assert_allclose(c[60, 80], g[60, 80], atol=0.02)


def test_clahe_single_tile_is_monotone():
    ramp = np.tile(np.linspace(0.0, 1.0, 512), (64, 1))
    out = iof._clahe(ramp, tiles=1, clip=2.0)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.all(np.diff(out, axis=1) >= -1e-6)


def test_clahe_has_no_tile_seams():
    rng = np.random.default_rng(4)
    yy, xx = np.mgrid[:512, :512]
    img = np.clip(0.3 + 0.2 * np.sin(xx / 40.0) * np.cos(yy / 55.0)
                  + rng.normal(0, 0.02, (512, 512)), 0, 1)
    out = iof._clahe(img, tiles=8, clip=2.0)
    jumps = np.abs(np.diff(out, axis=1)).mean(axis=0)
    edges = [64 * k - 1 for k in range(1, 8)]    # columns across a tile boundary
    assert jumps[edges].max() < 1.5 * np.median(jumps)


def test_clahe_finite_on_sky_centred_data():
    # After DBE the sky sits at zero; luminance ~0 with channels of opposite sign.
    rng = np.random.default_rng(2)
    img = rng.normal(0, 5.0, (300, 300, 3)).astype(np.float32)
    yy, xx = np.mgrid[:300, :300]
    img += (3000.0 * np.exp(-((xx - 150) / 50.0) ** 2))[..., None].astype(np.float32)
    img[:, 140:148, 0] = 400.0
    img[:, 140:148, 2] = -400.0 * 0.299 / 0.114
    _assert_display(iof.render_preview_float(img, 'clahe'))


@pytest.mark.parametrize('color', ['preserve', 'channel'])
@pytest.mark.parametrize('name', ['noisefloor', 'oklab', 'clahe'])
def test_new_stretches_valid_in_both_colour_modes(name, color):
    img, _r, _sky = _scene()
    original = img.copy()
    _assert_display(iof.render_preview_float(img, name, color=color))
    np.testing.assert_array_equal(img, original)


@pytest.mark.parametrize('name', ['noisefloor', 'oklab', 'clahe'])
def test_new_stretch_flags_parse(name):
    from src import cli
    assert cli.parse_args(['-d', '.', '--stretch', name]).stretch == name
