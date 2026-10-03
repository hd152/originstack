"""Phase 4 image-quality changes (2026-10): colour-preserving stretch,
noise-relative anisotropic diffusion threshold, Gaia colour calibration of the
linear stack before Phase 4."""
import argparse

import numpy as np
import pytest

from src.denoising import generalized_hyperbolic_stretch
from src.io_fits import colour_preserving_stretch, render_preview_float, render_preview_uint8

LUMA = np.array([0.299, 0.587, 0.114])


def _coloured_stars(H=128, seed=0):
    """Sky + noise + a red, a blue and a yellow star of different brightness."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :H]
    img = np.full((H, H, 3), 500.0)
    stars = [((30, 30), 4000.0, (1.0, 0.6, 0.35)),     # red, moderate
             ((90, 40), 9000.0, (0.45, 0.7, 1.0)),     # blue, bright
             ((60, 100), 2500.0, (1.0, 0.9, 0.6))]     # yellow, faint
    for (cy, cx), amp, col in stars:
        prof = amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.8 ** 2))
        img += prof[..., None] * np.array(col)
    return (img + rng.normal(0, 10, img.shape)).astype(np.float32), stars


# ---------------------------------------------------------------- stretch

def test_preserve_keeps_rgb_ratios_and_luminance():
    rgb = np.array([[[0.30, 0.20, 0.10], [0.05, 0.10, 0.30]]], np.float32) * 100
    black, white, sky = 0.0, 100.0, 0.1
    lum = rgb @ LUMA
    T = generalized_hyperbolic_stretch(lum, b=6, SP=0.1, black_point=black, white_point=white)
    out = colour_preserving_stretch(rgb, T, black, white, sky)
    for i in range(2):
        r_in = rgb[0, i] / rgb[0, i].sum()
        r_out = out[0, i] / out[0, i].sum()
        np.testing.assert_allclose(r_out, r_in, atol=1e-5)
    np.testing.assert_allclose(out[0] @ LUMA, T[0], atol=1e-5)


def test_preserve_over_range_keeps_hue_and_saturation():
    # A saturated red whose luminance maps near 1: R would exceed 1.
    rgb = np.array([[[95.0, 20.0, 10.0]]], np.float32)
    T = np.array([[0.95]])
    out = colour_preserving_stretch(rgb, T, 0.0, 100.0, 0.1)
    assert out.max() == pytest.approx(1.0) and out.min() >= 0.0
    np.testing.assert_allclose(out[0, 0] / out[0, 0].sum(), rgb[0, 0] / rgb[0, 0].sum(),
                               atol=1e-5)
    # A core at the white point keeps its colour instead of turning white
    core = colour_preserving_stretch(np.array([[[45.0, 70.0, 100.0]]], np.float32),
                                     np.array([[1.0]]), 0.0, 70.0, 0.1)
    assert core[0, 0, 2] == pytest.approx(1.0) and core[0, 0, 0] < 0.5


def test_preserve_greys_the_noise_floor():
    rgb = np.array([[[1.0, -1.0, 0.5]]], np.float32)       # just above black, noise
    T = np.array([[0.002]])
    out = colour_preserving_stretch(rgb, T, 0.0, 100.0, sky_sigma=5.0)
    assert np.ptp(out[0, 0]) < 1e-3                        # no colour speckle


def test_preserve_keeps_star_colours_channel_mode_shifts_them():
    img, stars = _coloured_stars()
    lum = img @ LUMA.astype(np.float32)
    black, white = float(np.median(lum)), float(lum.max())
    kw = dict(b=8.0, SP=0.1, black_point=black, white_point=white)
    pres = colour_preserving_stretch(img, generalized_hyperbolic_stretch(lum, **kw),
                                     black, white, 10.0)
    chan = np.stack([generalized_hyperbolic_stretch(img[..., c], **kw) for c in range(3)], 2)
    sky = np.median(img, axis=(0, 1))
    for (cy, cx), _amp, _col in stars:
        for dx, tol in ((0, None), (3, 0.05)):             # core (gamut-limited), wing
            lin = img[cy, cx + dx] - sky
            ci_lin = np.log(lin[2] / lin[0])
            ci_p = np.log(pres[cy, cx + dx, 2] / pres[cy, cx + dx, 0])
            ci_c = np.log(chan[cy, cx + dx, 2] / chan[cy, cx + dx, 0])
            assert abs(ci_p - ci_lin) < abs(ci_c - ci_lin)
            if tol is not None:
                assert abs(ci_p - ci_lin) < tol


def test_channel_mode_is_the_old_per_channel_curve():
    img, _ = _coloured_stars()
    out = render_preview_uint8(img, stretch='ghs', ghs_b=8.0, ghs_sp=0.1, color='channel')
    lum = img @ LUMA.astype(np.float32)
    from src.io_fits import _sky_stats
    med, _ = _sky_stats(lum)
    white = float(np.percentile(lum, 99.5))
    ref = np.stack([generalized_hyperbolic_stretch(img[..., c], b=8.0, SP=0.1, LP=0.0, HP=0.95,
                                                   black_point=med, white_point=white)
                    for c in range(3)], axis=2)
    np.testing.assert_array_equal(out, np.clip(ref * 255, 0, 255).astype(np.uint8))


def test_stretch_color_flag_parses():
    from src import cli
    assert cli.parse_args(['-d', '.']).stretch_color == 'preserve'
    assert cli.parse_args(['-d', '.', '--stretch-color', 'channel']).stretch_color == 'channel'


# ---------------------------------------------------------------- aniso kappa

def test_aniso_kappa_scales_with_noise():
    from src.postprocess import _aniso_kappa
    rng = np.random.default_rng(3)
    a = (100 + rng.normal(0, 10, (128, 128, 3))).astype(np.float32)
    b = (100 + rng.normal(0, 40, (128, 128, 3))).astype(np.float32)
    ka, kb = _aniso_kappa(a, None), _aniso_kappa(b, None)
    assert 3.0 < kb / ka < 5.0
    assert _aniso_kappa(a, 2.0) == pytest.approx(4 * _aniso_kappa(a, 0.5))


def test_old_saved_aniso_kappa_is_ignored(tmp_path):
    from src import cli
    cfg = tmp_path / 'c.toml'
    cfg.write_text('aniso_kappa = 30.0\n')
    args = cli.parse_args(['-d', '.'])
    cli.load_config_file(str(cfg), args)
    assert not hasattr(args, 'aniso_kappa')
    assert args.aniso_kappa_sigma == 1.0


# ---------------------------------------------------------------- colour calibration

def test_colour_calibration_flags():
    from src import cli
    assert cli.parse_args(['-d', '.']).color_calibrate is True
    assert cli.parse_args(['-d', '.', '--no-color-calibrate']).color_calibrate is False
    assert cli.parse_args(['-d', '.', '--color-calibrate']).color_calibrate is True
    assert cli.parse_args(['-d', '.']).color_calibrate_method == 'solar'


def _star_field_with_colour_law(n=60, H=600, seed=5, gain=(0.8, 1.0, 1.4)):
    """Stars whose true B/R follows a Gaia-like colour law, seen through
    channel gains the calibration has to undo."""
    from src.photometry import GaiaMatch
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :H]
    img = np.full((H, H, 3), 200.0)
    xs = rng.uniform(30, H - 30, n)
    ys = rng.uniform(30, H - 30, n)
    bp_rp = rng.uniform(0.2, 1.8, n)
    amps = rng.uniform(800, 4000, n)
    for x, y, c, a in zip(xs, ys, bp_rp, amps):
        # true flux ratios: B/R and G/R fall with redder BP-RP; 1 at solar 0.82
        br = 10 ** (-0.4 * 1.1 * (c - 0.82))
        gr = 10 ** (-0.4 * 0.5 * (c - 0.82))
        prof = a * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.5 ** 2))
        img += prof[..., None] * (np.array([1.0, gr, br]) * np.array(gain))
    img += rng.normal(0, 3, img.shape)
    gm = GaiaMatch(source_id=np.arange(n), ra=np.zeros(n), dec=np.zeros(n), x=xs, y=ys,
                   g=np.full(n, 12.0), bp=12.0 + bp_rp / 2, rp=12.0 - bp_rp / 2,
                   det_peak=amps * 1.2, fwhm=3.5, ap_radius=6, r_in=9.0,
                   r_out=15.0, field_ra=0.0, field_dec=0.0, plate_scale=1.0)
    return img.astype(np.float32), gm, gain


def test_solar_fit_makes_g2v_white(monkeypatch):
    import src.photometry as ph
    from src import color_calibrate as cc
    img, gm, gain = _star_field_with_colour_law()
    monkeypatch.setattr(ph, 'match_gaia_field', lambda *a, **k: gm)
    res = cc.fit_channel_scales_solar(img, header=None)
    assert res is not None
    (sr, sg, sb), info = res
    assert sg == 1.0
    np.testing.assert_allclose(sr * gain[0], sg * gain[1], rtol=0.03)
    np.testing.assert_allclose(sb * gain[2], sg * gain[1], rtol=0.03)
    assert info['n'] >= 15


def test_solar_fit_declines_with_too_few_stars(monkeypatch):
    import src.photometry as ph
    from src import color_calibrate as cc
    img, gm, _ = _star_field_with_colour_law(n=8)
    monkeypatch.setattr(ph, 'match_gaia_field', lambda *a, **k: gm)
    assert cc.fit_channel_scales_solar(img, header=None) is None


def test_colour_calibrate_stack_scales_both_arrays(monkeypatch):
    from src import color_calibrate as cc
    from src import pipeline
    monkeypatch.setattr(cc, 'calibrate_linear_stack', cc.calibrate_linear_stack.__wrapped__)
    monkeypatch.setattr(cc, 'fit_channel_scales_solar',
                        lambda img, header, verbose=False: ((0.5, 1.0, 2.0), dict(
                            n=20, slope_br=1.0, slope_gr=0.5, scatter_br=0.05)))
    stacked = np.ones((4, 4, 3), np.float32)
    fits_copy = stacked.copy()
    args = argparse.Namespace(color_calibrate=True, color_calibrate_method='solar',
                              verbose=False, _stack_wcs={'CTYPE1': ('RA---TAN', '')})
    pipeline._colour_calibrate_stack(args, stacked, fits_copy)
    np.testing.assert_array_equal(stacked[0, 0], [0.5, 1.0, 2.0])
    np.testing.assert_array_equal(fits_copy[0, 0], [0.5, 1.0, 2.0])
    assert args._colcal_scales == (0.5, 1.0, 2.0)
    from astropy.io import fits
    h = fits.Header()
    pipeline._colcal_header(h, args._colcal_scales)
    assert h['COLCAL'] is True and h['COLCAL_B'] == 2.0


def test_colour_calibrate_stack_leaves_data_alone_when_off_or_failing(monkeypatch):
    from src import color_calibrate as cc
    from src import pipeline
    stacked = np.ones((4, 4, 3), np.float32)
    args = argparse.Namespace(color_calibrate=False, verbose=False,
                              _stack_wcs={'CTYPE1': ('RA---TAN', '')})
    pipeline._colour_calibrate_stack(args, stacked)
    assert args._colcal_scales is None and stacked.max() == 1.0

    monkeypatch.setattr(cc, 'calibrate_linear_stack', cc.calibrate_linear_stack.__wrapped__)

    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(cc, 'fit_channel_scales_solar', boom)
    args.color_calibrate, args.color_calibrate_method = True, 'solar'
    pipeline._colour_calibrate_stack(args, stacked)
    assert args._colcal_scales is None and stacked.max() == 1.0
    args._stack_wcs = None
    pipeline._colour_calibrate_stack(args, stacked)
    assert args._colcal_scales is None


# ---------------------------------------------------------------- star reduction

def _star_grid(H=300, sigma=1.6, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :H]
    img = np.full((H, H, 3), 500.0)
    pos = [(40 + 45 * (i // 6), 40 + 45 * (i % 6)) for i in range(30)]
    for y, x in pos:
        img += (3000 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2)))[..., None] \
            * np.array([1.0, 0.8, 0.6])
    img = (img + rng.normal(0, 10, img.shape)).astype(np.float32)
    from astropy.table import Table
    src = Table({'xcentroid': np.array([p[1] for p in pos], float),
                 'ycentroid': np.array([p[0] for p in pos], float),
                 'flux': np.ones(len(pos))})
    return img, src, pos


def test_shrink_stars_narrows_without_blurring_or_touching_noise():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
    from common_star_fwhm import _gauss_fit

    from src.denoising import reduce_stars, shrink_stars
    img, src, pos = _star_grid()
    lum = lambda a: a @ LUMA  # noqa: E731
    out = shrink_stars(img, src, 3.77, amount=0.5)
    f0 = np.array([_gauss_fit(lum(img), x, y)[0] for y, x in pos])
    f1 = np.array([_gauss_fit(lum(out), x, y)[0] for y, x in pos])
    p0 = np.array([lum(img)[y, x] for y, x in pos])
    p1 = np.array([lum(out)[y, x] for y, x in pos])
    assert np.median(f1 / f0) == pytest.approx(1 / np.sqrt(1.5), abs=0.03)
    assert np.median(p1 / p0) > 0.97                       # peak kept
    sky = (slice(0, 20), slice(0, 20))
    np.testing.assert_array_equal(out[sky], img[sky])      # noise untouched away from stars
    for y, x in pos[:5]:                                   # colour kept
        r0 = (img[y, x] - 500)[2] / (img[y, x] - 500)[0]
        r1 = (out[y, x] - 500)[2] / (out[y, x] - 500)[0]
        assert r1 == pytest.approx(r0, rel=0.05)
    # The old blur-blend widened the same stars
    from src.quality import generate_star_mask
    mask = generate_star_mask(img.shape[:2], src, fwhm=4.0)
    old = reduce_stars(img, mask, reduction_factor=0.5, blur_sigma=1.5)
    f_old = np.array([_gauss_fit(lum(old), x, y)[0] for y, x in pos])
    assert np.median(f_old / f0) > 1.0


def test_shrink_stars_noop_cases():
    from src.denoising import shrink_stars
    img, src, _ = _star_grid()
    assert shrink_stars(img, src, 3.8, amount=0.0) is img
    assert shrink_stars(img, None, 3.8) is img
    faint = np.full_like(img, 500.0)
    assert shrink_stars(faint, src, 3.8) is faint          # nothing above 5 sigma


def test_gray_locus_skipped_after_gaia_calibration(monkeypatch):
    import src.postprocess as pp
    from src import cli
    calls = []
    monkeypatch.setattr(pp, 'photometric_color_calibrate',
                        lambda img, src, verbose=False: (calls.append(1) or img, None))
    img, _ = _coloured_stars()
    args = cli.parse_args(['-d', '.', '--skip-step', 'background'])
    args.photometric_calibration = True
    args._diagnostic_dir = None
    from src.models import ProcessingStats
    args._colcal_scales = (0.9, 1.0, 1.1)
    pp.postprocess_stack(img.copy(), args, [], ProcessingStats())
    assert calls == []
    args._colcal_scales = None
    pp.postprocess_stack(img.copy(), args, [], ProcessingStats())
    assert calls == [1]


def test_from_stack_reads_recorded_calibration():
    from astropy.io import fits

    from src.pipeline import prepare_linear_for_phase4
    h = fits.Header()
    h['COLCAL'] = True
    h['COLCAL_R'], h['COLCAL_G'], h['COLCAL_B'] = 0.8, 1.0, 1.2
    args = argparse.Namespace(color_calibrate=True)
    img = np.ones((4, 4, 3), np.float32)
    prepare_linear_for_phase4(args, img, h)
    assert args._colcal_scales == (0.8, 1.0, 1.2)
    assert img.max() == 1.0                                # not scaled twice


def test_early_cache_key_changes_with_pixels(tmp_path):
    from src import cli
    from src.postprocess import _early_cache_key
    f = tmp_path / 's.fits'
    f.write_bytes(b'x')
    args = cli.parse_args(['--from-stack', str(f)])
    img = np.ones((32, 32, 3), np.float32)
    k1 = _early_cache_key(img, args, set())
    scaled = img * np.array([0.8, 1.0, 1.2], np.float32)
    assert _early_cache_key(scaled, args, set()) != k1
    assert _early_cache_key(img.copy(), args, set()) == k1
