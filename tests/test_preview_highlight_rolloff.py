"""Preview highlight roll-off for extended bright regions, and the local-contrast
RGB rebuild near zero luminance (both found on a real Orion Nebula stack)."""
import numpy as np

import src.io_fits as iof
from src.denoising import multiscale_local_contrast


def _sky(h=400, w=600, seed=0):
    rng = np.random.default_rng(seed)
    return (500.0 + rng.normal(0, 20.0, (h, w, 3))).astype(np.float32)


def _add_star(img, y, x, peak, fwhm=3.0):
    s = fwhm / 2.355
    radius = int(np.ceil(8 * s))
    y0, y1 = max(0, int(y - radius)), min(img.shape[0], int(y + radius) + 1)
    x0, x1 = max(0, int(x - radius)), min(img.shape[1], int(x + radius) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    img[y0:y1, x0:x1] += (
        peak * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * s * s))
    )[..., None].astype(np.float32)


def test_extended_core_keeps_structure():
    img = _sky()
    yy, xx = np.mgrid[:400, :600]
    r = np.hypot(yy - 200, xx - 300)
    # a broad nebula and a bright core with a ripple well above the 99.5% white point
    core = 20000.0 * np.exp(-(r / 60.0) ** 2) + 80000.0 * np.exp(-(r / 25.0) ** 2) * (
        1.0 + 0.3 * np.sin(xx / 3.0))
    img += core[..., None].astype(np.float32)
    out = iof.render_preview_float(img, 'ghs')
    lum = out @ np.array([0.299, 0.587, 0.114])
    inner = r < 15
    assert np.mean(lum[inner] >= 0.999) < 0.5          # not a flat white blob
    assert np.std(lum[inner]) > 0.01                   # the ripple survives


def test_star_cores_do_not_trigger_rolloff():
    img = _sky()
    rng = np.random.default_rng(1)
    for _ in range(300):
        _add_star(img, rng.uniform(5, 395), rng.uniform(5, 595), rng.uniform(2e3, 6e4))
    lum = img.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    med, sig = iof._sky_stats(lum.astype(np.float32))
    white, _ = iof._preview_white(lum, med, sig, 99.5, med)
    assert iof._extended_highlight_top(lum, white) is None


def test_local_contrast_finite_near_zero_luminance():
    # Sky centred on zero (as after DBE): a dark lane where luminance ~0 while
    # red is positive and blue negative, beside bright nebulosity.
    rng = np.random.default_rng(2)
    h, w = 300, 300
    img = rng.normal(0, 5.0, (h, w, 3)).astype(np.float32)
    yy, xx = np.mgrid[:h, :w]
    neb = 3000.0 * np.exp(-((xx - 150) / 50.0) ** 2)
    img += neb[..., None].astype(np.float32)
    lane = (np.abs(xx - 150) < 4)
    img[lane, 0] = 400.0
    img[lane, 2] = -400.0 * 0.299 / 0.114
    img[lane, 1] = 0.0
    out = multiscale_local_contrast(img)
    assert np.all(np.isfinite(out))
    assert out.max() < 10 * img.max()


def test_frame_filling_nebula_wings_are_lifted():
    # A bright core well above the white point plus faint wings filling the frame
    # (Orion): the wings must not render near-black.
    img = _sky()
    yy, xx = np.mgrid[:400, :600]
    r = np.hypot(yy - 200, xx - 300)
    img += (40000.0 * np.exp(-(r / 30.0) ** 2) + 600.0 * np.exp(-(r / 120.0) ** 2))[..., None].astype(np.float32)
    out = iof.render_preview_float(img, 'ghs', ghs_b=5.64, ghs_sp=0.115)
    lum = out @ np.array([0.299, 0.587, 0.114])
    wings = (r > 70) & (r < 100)
    assert np.median(lum[wings]) > 0.15


def test_no_extended_highlight_keeps_the_curve():
    img = _sky()
    _add_star(img, 100, 100, 3e4)
    lum = img.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    med, sig = iof._sky_stats(lum.astype(np.float32))
    white, _ = iof._preview_white(lum, med, sig, 99.5, med)
    assert iof._extended_highlight_top(lum, white) is None


def test_masked_stretch_boosts_faint_nebula_without_changing_fits_input():
    rng = np.random.default_rng(12)
    h, w = 300, 400
    img = (500.0 + rng.normal(0.0, 8.0, (h, w, 3))).astype(np.float32)
    yy, xx = np.mgrid[:h, :w]
    nebula = 150.0 * np.exp(-(((xx - 200.0) / 70.0) ** 2
                              + ((yy - 150.0) / 50.0) ** 2))
    img += nebula[..., None].astype(np.float32)
    _add_star(img, 150, 200, 40000.0, fwhm=4.0)
    original = img.copy()

    standard = iof.render_preview_float(img, 'ghs', ghs_b=6.0, ghs_sp=0.1,
                                        black_sigma=-0.5)
    masked = iof.render_preview_float(img, 'masked', ghs_b=6.0, ghs_sp=0.1,
                                      black_sigma=-0.5)

    faint_region = (nebula > 50.0) & (nebula < 90.0)
    sky_region = nebula < 2.0
    luma = np.array([0.299, 0.587, 0.114])
    standard_lum = standard @ luma
    masked_lum = masked @ luma

    assert np.mean(masked_lum[faint_region]) > np.mean(standard_lum[faint_region]) + 0.02
    assert np.mean(masked_lum[sky_region]) < np.mean(standard_lum[sky_region]) + 0.01
    assert np.all(np.isfinite(masked))
    assert masked.min() >= 0.0 and masked.max() <= 1.0
    np.testing.assert_array_equal(img, original)


def test_masked_stretch_flag_parses():
    from src import cli
    assert cli.parse_args(['-d', '.', '--stretch', 'masked']).stretch == 'masked'


def test_multiscale_stretch_lifts_faint_filaments_without_raising_sky():
    rng = np.random.default_rng(23)
    h, w = 300, 400
    img = (500.0 + rng.normal(0.0, 8.0, (h, w, 3))).astype(np.float32)
    yy, xx = np.mgrid[:h, :w]
    broad_nebula = 100.0 * np.exp(-(((xx - 200.0) / 85.0) ** 2
                                    + ((yy - 150.0) / 65.0) ** 2))
    filament = 180.0 * np.exp(-((yy - (150.0 + 0.18 * (xx - 200.0))) / 2.0) ** 2)
    filament *= (np.abs(xx - 200.0) < 100.0)
    img += (broad_nebula + filament)[..., None].astype(np.float32)
    original = img.copy()

    standard = iof.render_preview_float(img, 'ghs', ghs_b=6.0, ghs_sp=0.1,
                                        black_sigma=-0.5)
    multiscale = iof.render_preview_float(img, 'multiscale', ghs_b=6.0, ghs_sp=0.1,
                                          black_sigma=-0.5)
    luma = np.array([0.299, 0.587, 0.114])
    linear_lum = img @ luma
    sky, sky_sigma = iof._sky_stats(linear_lum)
    black = sky - 0.5 * sky_sigma
    white, _ = iof._preview_white(linear_lum, sky, sky_sigma, 99.5, black)
    diffuse_weight, filament_weight = iof._multiscale_stretch_weights(
        linear_lum, sky, sky_sigma, white)
    standard_lum = standard @ luma
    multiscale_lum = multiscale @ luma
    sky = broad_nebula < 2.0
    filament_region = (filament > 80.0) & (broad_nebula > 20.0)
    diffuse_region = (broad_nebula > 45.0) & (filament < 1.0)

    assert np.mean(diffuse_weight[diffuse_region]) > 0.5
    assert np.mean(filament_weight[filament_region]) > 0.5
    assert np.mean(multiscale_lum[diffuse_region] - standard_lum[diffuse_region]) > 0.005
    assert np.mean(multiscale_lum[filament_region] - standard_lum[filament_region]) > 0.02
    assert np.mean(multiscale_lum[sky] - standard_lum[sky]) < 0.01
    assert np.all(np.isfinite(multiscale))
    assert multiscale.min() >= 0.0 and multiscale.max() <= 1.0
    np.testing.assert_array_equal(img, original)


def test_multiscale_stretch_flag_parses():
    from src import cli
    assert cli.parse_args(['-d', '.', '--stretch', 'multiscale']).stretch == 'multiscale'


def test_seti_stretch_maps_median_to_target_and_preserves_input():
    values = np.linspace(0.0, 1.0, 1001, dtype=np.float32)
    stretched = iof._statistical_stretch(values, float(np.median(values)))
    np.testing.assert_allclose(np.median(stretched), 0.2, atol=1e-6)

    ramp = np.linspace(100.0, 1000.0, 400, dtype=np.float32)
    image = np.broadcast_to(ramp[None, :, None], (200, 400, 3)).copy()
    original = image.copy()
    preview = iof.render_preview_float(image, stretch='seti')
    assert np.all(np.isfinite(preview))
    assert preview.min() >= 0.0 and preview.max() <= 1.0
    np.testing.assert_allclose(np.median(preview @ np.array([0.299, 0.587, 0.114])),
                               0.2, atol=1e-5)
    np.testing.assert_array_equal(image, original)


def test_seti_channel_stretch_returns_valid_preview():
    ramp = np.linspace(100.0, 1000.0, 400, dtype=np.float32)
    image = np.broadcast_to(ramp[None, :, None], (200, 400, 3)).copy()
    original = image.copy()

    preview = iof.render_preview_float(image, stretch='seti', color='channel')

    assert np.all(np.isfinite(preview))
    assert preview.min() >= 0.0 and preview.max() <= 1.0
    np.testing.assert_array_equal(image, original)


def test_seti_stretch_flag_parses():
    from src import cli
    assert cli.parse_args(['-d', '.', '--stretch', 'seti']).stretch == 'seti'
