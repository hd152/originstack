"""The stretch editor's numbers (src/stretch_editor.py) and the UIEvents methods behind it:
live_stretch / stretch_state / reset_stretch / restretch(params)."""
import numpy as np
import pytest

from src import stretch_editor as se
from src.io_fits import luminance, render_preview_rgb_float, to_uint8
from src.ui_events import UIEvents


def _scene(h=480, w=640, seed=3):
    """Linear RGB: a sky with correlated (smoothed) noise, a faint nebula and some stars."""
    rng = np.random.default_rng(seed)
    from scipy.ndimage import gaussian_filter
    sky = np.stack([0.02 + 0.004 * gaussian_filter(rng.standard_normal((h, w)), 1.2) * 3
                    for _ in range(3)], axis=-1).astype(np.float32)
    yy, xx = np.mgrid[:h, :w]
    neb = np.exp(-(((yy - h / 2) / 40) ** 2 + ((xx - w / 2) / 60) ** 2)).astype(np.float32)
    img = sky + 0.6 * neb[..., None] * np.array([1.0, 0.6, 0.4], dtype=np.float32)
    for _ in range(60):
        y, x = rng.integers(10, h - 10), rng.integers(10, w - 10)
        amp = rng.uniform(0.2, 1.0)
        img += (amp * np.exp(-(((yy - y) ** 2 + (xx - x) ** 2) / 3.0)))[..., None].astype(np.float32)
    return img


class TestControls:
    def test_each_stretch_lists_only_what_it_reads(self):
        assert set(se.controls_for('ghs')) == set(se.PARAM_KEYS)
        assert se.controls_for('oklab') == se.controls_for('ghs')
        assert se.controls_for('arcsinh') == ('black_sigma', 'white_pct')
        assert se.controls_for('linear') == ()
        assert se.controls_for('nonsense') == ()

    def test_a_listed_control_changes_the_picture_an_unlisted_one_does_not(self):
        img = _scene()
        for stretch in ('ghs', 'arcsinh', 'noisefloor', 'seti'):
            base = render_preview_rgb_float(img, stretch=stretch, nebula_auto=False)
            for key, alt in (('ghs_b', 14.0), ('ghs_sp', 0.3), ('ghs_hp', 0.8),
                             ('black_sigma', 2.0), ('white_pct', 99.0)):
                out = render_preview_rgb_float(img, stretch=stretch, nebula_auto=False, **{key: alt})
                changed = not np.array_equal(base, out)
                assert changed == (key in se.controls_for(stretch)), (stretch, key)

    def test_clamp_drops_unknown_and_limits(self):
        out = se.clamp_params({'ghs_b': 99, 'white_pct': 10, 'bogus': 1})
        assert out == {'ghs_b': 25.0, 'white_pct': 95.0}


class TestNebulaLift:
    def test_auto_lift_overrides_a_weak_strength_manual_keeps_it(self):
        img = _scene()
        info_auto, info_manual = {}, {}
        a = render_preview_rgb_float(img, stretch='ghs', ghs_b=2.0, info=info_auto)
        b = render_preview_rgb_float(img, stretch='ghs', ghs_b=2.0, nebula_auto=False,
                                     info=info_manual)
        assert info_auto['lifted'] and info_auto['ghs_b'] > 2.0 and info_auto['ghs_sp'] == 0.0
        assert info_manual['ghs_b'] == 2.0 and 'lifted' not in info_manual
        assert a.mean() > b.mean()

    def test_default_render_is_unchanged_by_the_new_arguments(self):
        img = _scene()
        assert np.array_equal(render_preview_rgb_float(img, stretch='ghs'),
                              render_preview_rgb_float(img, stretch='ghs', nebula_auto=True, info={}))


class TestWhitePercentile:
    def test_default_is_unchanged(self):
        img = _scene()
        assert np.array_equal(render_preview_rgb_float(img, stretch='ghs'),
                              render_preview_rgb_float(img, stretch='ghs', white_pct=99.5))

    def test_a_lower_white_point_brightens(self):
        img = _scene()
        a = render_preview_rgb_float(img, stretch='ghs', white_pct=99.9)
        b = render_preview_rgb_float(img, stretch='ghs', white_pct=98.0)
        assert b.mean() > a.mean()


class TestHistogram:
    def test_axis_roundtrip(self):
        x = np.array([-5.0, 0.0, 3.0, 40.0, 1e4])
        assert np.allclose(se.from_axis(se.to_axis(x)), x)

    def test_subsample_keeps_the_noise_an_average_loses_it(self):
        from src.ui_events import _decimate
        rng = np.random.default_rng(1)
        img = (0.02 + 0.003 * rng.standard_normal((900, 1200, 3))).astype(np.float32)
        full = se.sky_of(luminance(img))[1]
        sub = se.sky_of(luminance(se.subsample(img, 400)))[1]
        avg = se.sky_of(luminance(_decimate(img, 400)))[1]
        assert sub == pytest.approx(full, rel=0.1)
        assert avg < 0.5 * full                       # why the editor does not average

    def test_subsample_small_image_is_returned_as_is(self):
        img = _scene(h=100, w=120)
        assert se.subsample(img, 400) is img

    def test_histogram_shape_and_normalisation(self):
        img = _scene()
        lum = luminance(img)
        med, sigma = se.sky_of(lum)
        h = se.linear_histogram(img, lum, med, sigma, bins=100)
        assert h['counts'].shape == (4, 100) and h['edges'].shape == (101,)
        assert h['counts'].max() == pytest.approx(1.0) and h['counts'].min() >= 0
        peak_bin = int(np.argmax(h['counts'][3]))
        assert abs(0.5 * (h['edges'][peak_bin] + h['edges'][peak_bin + 1])) < 1.0   # sky sits near 0 sigma

    def test_tone_curve_rises_for_a_stretch(self):
        img = _scene()
        lum = luminance(img)
        med, sigma = se.sky_of(lum)
        out = render_preview_rgb_float(img, stretch='ghs')
        xs, ys = se.tone_curve(lum, luminance(out), med, sigma)
        assert len(xs) == len(ys) > 10
        assert np.all(np.diff(xs) > 0)
        assert ys[-1] > ys[0] and np.all(np.diff(ys) > -0.05)       # monotone to within noise


class TestMarkers:
    def test_black_marker_roundtrip(self):
        med, sigma = 0.02, 0.003
        for b in (-2.0, 0.0, 1.5, 6.0):
            u = se.marker_axis(med + b * sigma, med, sigma)
            assert se.black_sigma_at(u) == pytest.approx(b, abs=1e-6)

    def test_black_marker_is_clamped(self):
        assert se.black_sigma_at(50.0) == se.PARAM_RANGES['black_sigma'][1]
        assert se.black_sigma_at(-50.0) == se.PARAM_RANGES['black_sigma'][0]

    def test_white_marker_roundtrip(self):
        img = _scene()
        lum = luminance(img)
        med, sigma = se.sky_of(lum)
        for pct in (98.0, 99.5, 99.9):
            v = se.white_value(lum, med, sigma, pct, 0.0)
            u = se.marker_axis(v, med, sigma)
            assert se.white_pct_at(u, lum, med, sigma) == pytest.approx(pct, abs=0.05)

    def test_kw_for_render_merges_and_clamps(self):
        kw = se.kw_for_render({'stretch': 'ghs', 'color': 'preserve', 'ghs_b': 8.0},
                              {'ghs_b': 12.0, 'junk': 1}, stretch='arcsinh')
        assert kw['stretch'] == 'arcsinh' and kw['ghs_b'] == 12.0
        assert 'junk' not in kw and kw['white_pct'] == 99.5


# ── UIEvents ──────────────────────────────────────────────────────────


@pytest.fixture
def ui(tmp_path):
    u = UIEvents()
    u.attach()
    img = _scene()
    kw = {'stretch': 'ghs', 'ghs_b': 8.0, 'ghs_sp': 0.15, 'ghs_hp': 0.95, 'black_sigma': 0.0,
          'color': 'preserve', 'noise_scale': 1.0, 'anchor': None}
    with u._lock:
        u._final_src = {'rgb': img, 'starless': None, 'path': str(tmp_path / 'final.jpg'),
                        'kw': dict(kw), 'kw0': dict(kw), 'exports': ()}
    return u


class TestLiveStretch:
    def test_returns_a_picture_a_histogram_and_markers(self, ui, tmp_path):
        r = ui.live_stretch()
        assert r['jpeg'][:2] == b'\xff\xd8'
        assert r['hist']['counts'].shape[0] == 4
        assert r['markers']['black'] < r['markers']['white']
        assert not (tmp_path / 'final.jpg').exists()           # nothing written

    def test_settings_are_kept_and_reset_restores_the_run_values(self, ui):
        ui.live_stretch({'ghs_b': 15.0, 'black_sigma': 2.0})
        st = ui.stretch_state()
        assert st['ghs_b'] == 15.0 and st['black_sigma'] == 2.0
        back = ui.reset_stretch()
        assert back['ghs_b'] == 8.0 and back['black_sigma'] == 0.0

    def test_stronger_stretch_is_brighter(self, ui):
        from src.qt_app.widgets import qimage_from_bytes
        pytest.importorskip('PySide6')
        from src.qt_app.viewer import image_to_array
        manual = {'nebula_auto': False}
        weak = image_to_array(qimage_from_bytes(ui.live_stretch({'ghs_b': 2.0, **manual})['jpeg'])).mean()
        strong = image_to_array(qimage_from_bytes(ui.live_stretch({'ghs_b': 18.0, **manual})['jpeg'])).mean()
        assert strong > weak

    def test_the_cache_is_reused_between_calls(self, ui):
        ui.live_stretch()
        first = ui._final_src['_edit'][1400]
        ui.live_stretch({'ghs_b': 3.0})
        assert ui._final_src['_edit'][1400] is first
        ui.live_stretch(max_dim=500)                   # another size has its own entry
        assert set(ui._final_src['_edit']) == {1400, 500}

    def test_live_matches_the_full_resolution_render(self, ui):
        """The reduction must look like what Apply writes: compare the mean level of the live
        render with the full image's, for settings that depend on the noise and percentiles."""
        params = {'black_sigma': 1.5, 'white_pct': 99.0, 'ghs_b': 10.0}
        live = ui.live_stretch(params, max_dim=320)
        kw = se.kw_for_render(ui._final_src['kw0'], params)
        full = render_preview_rgb_float(ui._final_src['rgb'], **kw)
        small = se.subsample(full, 320)
        from src.qt_app.viewer import image_to_array
        from src.qt_app.widgets import qimage_from_bytes
        pytest.importorskip('PySide6')
        got = image_to_array(qimage_from_bytes(live['jpeg'])).astype(float) / 255.0
        assert got.shape[:2] == small.shape[:2]
        assert abs(got.mean() - to_uint8(small).mean() / 255.0) < 0.03

    def test_reports_what_the_nebula_lift_did(self, ui):
        r = ui.live_stretch({'ghs_b': 2.0})
        assert r['effective']['lifted'] and r['effective']['ghs_b'] > 2.0
        r = ui.live_stretch({'ghs_b': 2.0, 'nebula_auto': False})
        assert r['effective']['ghs_b'] == 2.0
        assert ui.stretch_state()['nebula_auto'] is False

    def test_no_preview_raises(self):
        u = UIEvents()
        with pytest.raises(RuntimeError):
            u.live_stretch()
        assert u.stretch_state() is None and u.reset_stretch() is None

    def test_restretch_with_params_writes_the_file_and_keeps_them(self, ui, tmp_path):
        data = ui.restretch('ghs', 'preserve', {'ghs_b': 12.0, 'white_pct': 99.2})
        assert data[:2] == b'\xff\xd8' and (tmp_path / 'final.jpg').exists()
        st = ui.stretch_state()
        assert st['ghs_b'] == 12.0 and st['white_pct'] == 99.2

    def test_restretch_without_params_is_the_old_call(self, ui, tmp_path):
        ui.restretch('linear', 'channel')
        assert ui._final_src['kw']['stretch'] == 'linear' and (tmp_path / 'final.jpg').exists()


@pytest.mark.parametrize('params', [{}, {'ghs_b': 12.0, 'black_sigma': 1.0, 'white_pct': 99.2},
                                    {'nebula_auto': False, 'ghs_b': 5.0}])
def test_the_sampled_render_equals_the_full_render_at_the_same_pixels(params):
    """The property the live preview relies on: stretching every f-th pixel gives what
    stretching everything and then taking every f-th pixel gives."""
    img = _scene(h=900, w=1200)
    kw = se.kw_for_render({'stretch': 'ghs', 'color': 'preserve'}, params)
    small = se.subsample(img, 400)
    a = render_preview_rgb_float(small, pixel_scale=small.size / img.size, **kw)
    b = se.subsample(render_preview_rgb_float(img, **kw), 400)
    assert a.shape == b.shape
    assert np.abs(a - b).mean() < 0.003


def _compact_nebula(h=1200, w=1600, seed=5):
    """A flat sky with one small, very bright extended object (like M57) and a few stars."""
    rng = np.random.default_rng(seed)
    img = (0.02 + 0.003 * rng.standard_normal((h, w, 3))).astype(np.float32)
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - h / 2, xx - w / 2)
    ring = np.exp(-((r - 30) / 8.0) ** 2).astype(np.float32)           # ~2.6k px above white
    img += 3.0 * ring[..., None] * np.array([0.3, 1.0, 0.5], dtype=np.float32)
    for _ in range(25):
        y, x = rng.integers(20, h - 20), rng.integers(20, w - 20)
        img += (0.4 * np.exp(-(((yy - y) ** 2 + (xx - x) ** 2) / 3.0)))[..., None].astype(np.float32)
    return img


def test_a_small_bright_target_is_judged_the_same_on_the_reduced_copy():
    """The roll-off counts pixels; without the area ratio a compact nebula fell under the
    1000-pixel floor on the reduction and was clipped where the full render rolls it off."""
    img = _compact_nebula()
    small = se.subsample(img, 400)
    scale = small.size / img.size
    kw = {'stretch': 'ghs', 'color': 'preserve'}
    full = se.subsample(render_preview_rgb_float(img, **kw), 400)
    right = render_preview_rgb_float(small, pixel_scale=scale, **kw)
    wrong = render_preview_rgb_float(small, **kw)
    err_right, err_wrong = np.abs(right - full).mean(), np.abs(wrong - full).mean()
    assert err_right < 0.004
    assert err_wrong > 3 * err_right


class TestFullView:
    def test_full_size_render_has_the_real_shape_and_is_cached(self, ui):
        a = ui.full_view()
        h, w = ui._final_src['rgb'].shape[:2]
        assert a['rgb8'].shape == (h, w, 3) and a['rgb8'].dtype == np.uint8 and a['size'] == (w, h)
        assert ui.full_view() is a                              # same settings: not rendered again
        ui.live_stretch({'ghs_b': 3.0, 'nebula_auto': False})   # a different stretch renders again
        assert ui.full_view() is not a

    def test_it_equals_the_apply_render(self, ui):
        ui.live_stretch({'ghs_b': 11.0, 'black_sigma': 1.0})
        kw = se.kw_for_render(ui._final_src['kw'])
        ref = to_uint8(render_preview_rgb_float(ui._final_src['rgb'], **kw))
        assert np.array_equal(ui.full_view()['rgb8'], ref)

    def test_very_large_images_come_back_reduced_but_say_their_real_size(self, ui):
        h, w = ui._final_src['rgb'].shape[:2]
        out = ui.full_view(max_pixels=(h * w) // 4)
        assert out['size'] == (w, h) and out['rgb8'].shape[0] * out['rgb8'].shape[1] <= (h * w) // 4 + h + w

    def test_final_size_and_no_preview(self, ui):
        h, w = ui._final_src['rgb'].shape[:2]
        assert ui.final_size() == (w, h)
        u = UIEvents()
        assert u.final_size() is None
        with pytest.raises(RuntimeError):
            u.full_view()

    def test_linear_sample_reports_value_and_sigma_above_the_sky(self, ui):
        rgb = ui._final_src['rgb']
        brightest = np.unravel_index(int(np.argmax(rgb.sum(axis=2))), rgb.shape[:2])
        y, x = int(brightest[0]), int(brightest[1])
        s = ui.sample_linear(x, y)
        assert s['rgb'] == pytest.approx(tuple(float(v) for v in rgb[y, x]))
        assert s['sigma'] > 50                                  # a star is far above the sky
        sky = ui.sample_linear(2, 2)
        assert abs(sky['sigma']) < 6
        assert ui.sample_linear(-1, 0) is None and ui.sample_linear(0, rgb.shape[0]) is None

    def test_a_new_stack_drops_the_caches(self, ui):
        ui.full_view()
        ui.sample_linear(0, 0)
        assert '_full' in ui._final_src and '_sky' in ui._final_src
