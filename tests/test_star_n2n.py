"""Noisier2Noise star removal (src/star_n2n.py) and its tools: injection model, the
correction formula, an identity network, the bank builder and the evaluation metrics.

No trained model is needed: the network checks use a zero output layer (f(z) = z), which
must give back the input exactly, and an oracle in place of the network for the formula.
"""
import os
import sys

import numpy as np
import pytest
from scipy import ndimage as ndi

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
    # total injected flux: copies of the donors, out to where their wings vanish
    assert 0.9 < s.sum() / flux.sum() <= 1.1
    # copies only: far from any injected star the layer is exactly 0
    assert (s == 0).mean() > 0.3
    assert len(field.inject(rng, alpha=0.0)[1]) == 0


def test_real_discs_are_zero_mean_noise_at_the_donors(field_and_z):
    field, _, _ = field_and_z
    d = field.real_discs(np.random.default_rng(4))
    lum = d.mean(-1)
    assert abs(float(d.sum())) < 0.05 * float(np.abs(d).sum())     # noise, not flux
    on = np.zeros(lum.shape, bool)
    on[field.donor_y, field.donor_x] = True
    assert (lum[on] != 0).all()                                     # one disc on every donor
    far = ~ndi.binary_dilation(on, iterations=sn.COPY_R_MAX + 1)
    assert (lum[far] == 0).all()


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
    # tta=0: an identity network passes the run-time noise discs straight through; the
    # hole guard would raise noise pixels inside star footprints
    starless, stars = sn.remove_stars_nn(img, model_path=path, alpha=0.5, tta=0, guard=False,
                                         saturated=False)
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
    r = ev.evaluate(str(p), w, alpha=1.0, test_alpha=0.3, seed=0, tta=0, post=False)
    assert r['inject_left']['all'] == pytest.approx(1.0, abs=0.05)
    assert r['real_left']['all'] == pytest.approx(1.0, abs=0.05)
    assert r['faint_kept'] == pytest.approx(1.0, abs=0.05)
    assert abs(r['sky_shift']) < 1e-3 and r['sky_mad'] < 1e-3


def test_guard_fills_holes_and_leaves_the_sky(field_and_z):
    field, z, _ = field_and_z
    x = np.zeros_like(z)
    y0, x0, q = field.donor_y[0], field.donor_x[0], field.donor_r[0]
    x[y0 - 1:y0 + 2, x0 - 1:x0 + 2] = -500.0                 # an inverted core
    x[5, 5] = -3.0                                           # sky noise far from any star
    star = np.zeros(z.shape[:2], bool)
    star[field.donor_y, field.donor_x] = True
    far = ~ndi.binary_dilation(star, iterations=sn.COPY_R_MAX + 5)
    g = sn.guard_holes(x, field)
    assert g[y0, x0].min() >= -sn.GUARD_SIGMA - 1e-6        # floored at ring p10 (0) - 1
    np.testing.assert_array_equal(g[far], x[far])           # nothing changes off the stars
    assert (g >= x).all()                                    # it only ever raises


def test_fill_saturated_removes_the_clipped_stars_only(field_and_z):
    field, z, _ = field_and_z
    assert field.saturated.any()
    out = sn.fill_saturated(z, z, field)
    sy, sx = field.det_y[field.saturated], field.det_x[field.saturated]
    for y, x in zip(sy, sx):                                # the core now looks like its surroundings
        yy, xx = np.mgrid[-24:25, -24:25]
        ring = out[y - 24:y + 25, x - 24:x + 25][np.hypot(yy, xx) > 16]
        assert z[y, x].mean() > 50                          # it was a bright star...
        assert abs(out[y - 1:y + 2, x - 1:x + 2].mean() - np.median(ring)) < 5   # ...now sky
    free = np.ones(z.shape[:2], bool)
    free[sy, sx] = False
    far = ndi.distance_transform_edt(free) > sn.HALO_R + 1
    np.testing.assert_array_equal(out[far], z[far])         # nothing else is touched


# ── --remove-stars-method nn ─────────────────────────────────────────────────

def test_cli_remove_stars_method():
    from src.cli import parse_args
    assert parse_args(['-d', 'x', '-o', 'y.fits']).remove_stars_method == 'inpaint'
    a = parse_args(['-d', 'x', '-o', 'y.fits', '--remove-stars', '--remove-stars-method', 'nn'])
    assert a.remove_stars and a.remove_stars_method == 'nn'


def test_remove_stars_method_survives_a_saved_config(tmp_path):
    from src.cli import load_config_file, parse_args, save_effective_config
    a = parse_args(['-d', 'x', '-o', str(tmp_path / 's.fits'), '--remove-stars', '--remove-stars-method', 'nn'])
    save_effective_config(a, str(tmp_path / 's.fits'))
    b = parse_args(['-d', 'x', '-o', str(tmp_path / 't.fits')])
    load_config_file(str(tmp_path / 's_config.toml'), b)
    assert b.remove_stars and b.remove_stars_method == 'nn'


def _phase4(tmp_path, monkeypatch, method, available=True):
    import src.postprocess as pp
    from src.cli import parse_args
    from src.models import ProcessingStats
    calls = {'nn': [], 'inpaint': 0, 'sidecar': []}

    def fake_nn(img, *a, **k):
        calls['nn'].append(img.copy())
        return img * 0.5, img * 0.5

    def fake_inpaint(img, sources, fwhm, **k):
        calls['inpaint'] += 1
        return img, 1
    monkeypatch.setattr(sn, 'remove_stars_nn', fake_nn)
    monkeypatch.setattr(sn, 'model_available', lambda path=None: available)
    monkeypatch.setattr('src.star_removal.remove_stars', fake_inpaint)
    monkeypatch.setattr(pp, '_save_sidecar_fits', lambda img, out, suffix: calls['sidecar'].append(suffix))
    img = _synthetic(seed=8)
    args = parse_args(['-d', '.', '-o', str(tmp_path / 's.fits'), '--remove-stars',
                       '--remove-stars-method', method, '--skip-step', 'background'])
    args._diagnostic_dir = None
    pp.postprocess_stack(img.copy(), args, [], ProcessingStats())
    return img, calls


def test_phase4_nn_removes_stars_from_the_linear_stack(tmp_path, monkeypatch):
    img, calls = _phase4(tmp_path, monkeypatch, 'nn')
    assert len(calls['nn']) == 1
    np.testing.assert_array_equal(calls['nn'][0], img)       # Phase 4's input, not its output
    assert calls['inpaint'] == 0 and calls['sidecar'].count('_starless') == 1


def test_phase4_nn_falls_back_to_inpaint_without_the_model(tmp_path, monkeypatch):
    _, calls = _phase4(tmp_path, monkeypatch, 'nn', available=False)
    assert calls['nn'] == [] and calls['inpaint'] == 1


@pytest.mark.skipif(not sn.model_available(), reason="star removal model not present")
def test_bundled_model_removes_unsaturated_stars():
    img = _synthetic(seed=9)
    starless, stars = sn.remove_stars_nn(img)
    assert np.isfinite(starless).all()
    np.testing.assert_allclose(starless + stars, img, rtol=1e-5, atol=1e-3)
    # the share left depends on which sky discs are drawn: 0.20-0.40 between disc seeds on
    # this small, crowded image, so the check averages four
    z, _, _, B = sn.normalise_image(img)
    f = sn.StarField(z, B)
    w = nd.load_weights(sn.MODEL_PATH)
    fin = sn.aperture_flux(z, f.donor_y, f.donor_x).mean(1)
    bright = fin > np.quantile(fin, 0.5)
    left = []
    for seed in range(4):
        fout = sn.aperture_flux(sn.starless_z(z, w, 1.0, f, 2, seed=seed), f.donor_y, f.donor_x).mean(1)
        left.append(np.median(fout[bright] / fin[bright]))
    assert np.mean(left) < 0.4                               # most of each star is gone


def test_starless_process_nn_splits_with_the_linear_sky_level(tmp_path, monkeypatch):
    """--starless-process with --remove-stars-method nn: the split before the denoisers uses
    the network, with B measured on Phase 4's (linear) input; the later splits keep the
    inpaint. The layers still sum back to the input."""
    import src.postprocess as pp
    from src.cli import parse_args
    from src.models import ProcessingStats
    seen = []

    def fake_nn(img, *a, sky_level=None, **k):
        seen.append(sky_level)
        return img * 0.25, img * 0.75
    monkeypatch.setattr(sn, 'remove_stars_nn', fake_nn)
    monkeypatch.setattr(sn, 'model_available', lambda path=None: True)
    img = _synthetic(seed=10)
    expect = sn.linear_sky_level(img)
    args = parse_args(['-d', '.', '-o', str(tmp_path / 's.fits'), '--starless-process',
                       '--remove-stars-method', 'nn', '--skip-step', 'background'])
    args._diagnostic_dir = None
    pp.postprocess_stack(img.copy(), args, [], ProcessingStats())
    assert len(seen) == 1                                    # only the pre-denoise split
    np.testing.assert_allclose(seen[0], expect)
    assert (np.asarray(seen[0]) > 1).all()                   # a real sky, not the subtracted ~0

    starless, stars = pp._split_starless(img, np.zeros(1), [], args, nn_sky=expect)
    np.testing.assert_allclose(starless + stars, img, rtol=1e-6)


def test_fill_saturated_removes_diffraction_spikes(field_and_z):
    """Four radial spikes on the saturated star: the result matches the spike-free one."""
    field, z, _ = field_and_z
    (y, x), = zip(field.det_y[field.saturated], field.det_x[field.saturated])
    h, w, _ = z.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot(yy - y, xx - x)
    spikes = np.zeros((h, w), np.float32)
    for ang in (0.0, 90.0, 180.0, 270.0):
        t = np.radians(ang)
        d = np.abs(-(xx - x) * np.sin(t) + (yy - y) * np.cos(t))      # distance to the line
        along = (xx - x) * np.cos(t) + (yy - y) * np.sin(t)              # position along it
        spikes += 30.0 * np.exp(-r / 40.0) * np.exp(-d ** 2 / 2.0) * (along > 8) * (r < 100)
    zs = z + spikes[..., None]
    clean = sn.fill_saturated(z, z, field)
    got = sn.fill_saturated(zs, zs, field)
    ring = (r > 15) & (r < 80)
    left = (got - clean).mean(-1)[ring]
    injected = spikes[ring]
    assert left.sum() < 0.2 * injected.sum()                 # >= 80% of the spike light gone
    assert np.abs(left).max() < 0.5 * injected.max()
    far = r > 2 * sn.HALO_R + 2
    np.testing.assert_array_equal(got[far], zs[far])         # nothing beyond the star touched


def test_spike_subtraction_stays_finite_when_every_angle_is_masked():
    """A ring of other saturated stars masks whole radii: no NaN may leak into the image."""
    rng = np.random.default_rng(5)
    out = rng.normal(0, 1, (200, 200, 3)).astype(np.float32)
    ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    others = np.c_[100 + 50 * np.sin(ang), 100 + 50 * np.cos(ang)]
    before = out.copy()
    sn._subtract_spikes(out, 100, 100, 10.0, 95.0, others)
    assert np.isfinite(out).all()
    assert (out <= before + 1e-6).all()                      # it only ever subtracts



def test_saturated_star_in_bright_emission_is_left_alone():
    """Orion: saturated stars inside the bright nebula core had the nebula cut out
    in discs and coloured rings by the halo subtraction and core fill."""
    from src.star_n2n import SAT_SKIP_BG, _ring_level
    H, W = 200, 200
    out = np.zeros((H, W, 3), np.float32)
    assert _ring_level(out, 100, 100) == 0.0
    out += SAT_SKIP_BG * 2.0
    assert _ring_level(out, 100, 100) > SAT_SKIP_BG


def test_footprint_colour_follows_the_star_per_radius(field_and_z):
    """A star with a bluer core and a redder halo keeps that colour per radius in the star
    layer; one colour for the whole star gave blue rims and red-yellow cores. Where the
    star is too faint to say, the copy's integrated colour is used."""
    r = 12
    yy, xx = np.mgrid[-r - 5:r + 6, -r - 5:r + 6].astype(np.float32)
    rr = np.hypot(yy, xx)
    core = 400.0 * np.exp(-rr ** 2 / (2 * 1.5 ** 2))
    halo = 40.0 * np.exp(-rr ** 2 / (2 * 4.0 ** 2))
    cut = np.stack([core * 0.8 + halo * 1.4, core + halo, core * 1.2 + halo * 0.6], -1)
    cut += np.random.default_rng(0).normal(0, 1, cut.shape).astype(np.float32)
    cf = np.array([1.0, 1.0, 1.0])
    col = sn._radial_colour(cut, rr, r, cf)
    assert col.shape == cut.shape
    np.testing.assert_allclose(col.sum(-1), 1.0, atol=1e-5)
    assert col[r + 5, r + 5, 2] > col[r + 5, r + 5, 0]                # blue core
    assert col[r + 5, r + 5 + 7, 0] > col[r + 5, r + 5 + 7, 2]        # red halo
    np.testing.assert_allclose(col[0, 0], cf / 3, atol=1e-6)          # outside: integrated colour


def test_disc_sources_avoid_structured_sky():
    """Sky discs come from structure-free sky: a disc cut from bright nebula carries its
    texture into the output (Orion: red and green discs around stars on dark sky)."""
    img = _synthetic(seed=3, h=480, w=640)
    yy, xx = np.mgrid[0:480, 0:640].astype(np.float32)
    ripples = 40.0 * (1 + np.sin(xx / 6.0) * np.sin(yy / 7.0)) * (xx > 400)
    img = img + ripples[..., None].astype(np.float32)
    z, _, _, B = sn.normalise_image(img)
    f = sn.StarField(z, B)
    assert f.ok and len(f._sky_x) > 0
    assert np.mean(f._sky_x > 420) < 0.05


def test_residual_core_is_refilled_off_centre_and_faint_stars_are_left(field_and_z):
    """A lopsided positive crescent left at a donor (what the network leaves at the brightest
    stars) goes back to the sky level with its noise; a noise-level bump (a faint star's tail)
    does not trigger the gate; negatives and everything off the donors stay as they are."""
    field, z, _ = field_and_z
    rng = np.random.default_rng(5)
    x = rng.standard_normal(z.shape).astype(np.float32)
    (y0, x0), (y1, x1) = (field.donor_y[0], field.donor_x[0]), (field.donor_y[1], field.donor_x[1])
    x[y0 - 2:y0 + 3, x0 - 4:x0 - 1] += 150.0                 # crescent left of the centre
    x[y0, x0 + 1] = -40.0                                    # an inverted pixel, not a core
    x[y1, x1] += 2.0                                         # a faint star's tail
    out = sn.remove_residual_cores(x, field)
    assert x[y0 - 2:y0 + 3, x0 - 4:x0 - 1].mean() > 100
    assert abs(out[y0 - 2:y0 + 3, x0 - 4:x0 - 1].mean()) < 3  # the crescent is sky again...
    assert out[y0 - 2:y0 + 3, x0 - 4:x0 - 1].std() > 0.5      # ...with noise, not a flat fill
    assert out[y0, x0 + 1, 0] == x[y0, x0 + 1, 0]            # negatives are guard_holes' business
    np.testing.assert_array_equal(out[y1 - 3:y1 + 4, x1 - 3:x1 + 4], x[y1 - 3:y1 + 4, x1 - 3:x1 + 4])
    star = np.zeros(z.shape[:2], bool)
    star[field.donor_y, field.donor_x] = True
    far = ~ndi.binary_dilation(star, iterations=sn.CORE_R + sn.CORE_RING + 8)
    np.testing.assert_array_equal(out[far], x[far])


def test_starless_z_cores_switch(field_and_z, monkeypatch):
    field, z, _ = field_and_z
    calls = []
    monkeypatch.setattr(sn, 'remove_residual_cores', lambda x, f, seed=0: calls.append(1) or x)
    monkeypatch.setattr(sn.nd, 'denoise_z', lambda zz, w: zz * 0.5)
    sn.starless_z(z, None, field=field, tta=1, saturated=False)
    assert calls == [1]
    sn.starless_z(z, None, field=field, tta=1, saturated=False, cores=False)
    sn.starless_z(z, None, field=field, tta=1, saturated=False, guard=False)
    assert calls == [1]
