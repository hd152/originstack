"""The Qt window: form model, Setup widgets, viewer, post-processing panel, main window.

Runs on Qt's offscreen platform (no display needed); skipped without PySide6.
"""
import io
import os

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
pytest.importorskip('PySide6')

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope='module')
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _jpeg(color=(200, 60, 20), size=(64, 48)) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new('RGB', size, color).save(buf, 'JPEG')
    return buf.getvalue()


# ── FormModel ─────────────────────────────────────────────────────────


class TestFormModel:
    def test_untouched_form_submits_nothing(self, qapp):
        from src.qt_app.form_model import FormModel
        assert FormModel().read_form() == {}

    def test_changed_values_only_and_reset(self, qapp):
        from src.qt_app.form_model import FormModel
        m = FormModel()
        m.set('stack_method', 'median')
        m.set('use_gpu', True)
        assert m.read_form() == {'stack_method': 'median', 'use_gpu': True}
        m.reset('stack_method')
        assert m.read_form() == {'use_gpu': True}

    def test_setting_the_default_back_is_not_a_change(self, qapp):
        from src.qt_app.form_model import FormModel
        m = FormModel()
        default = m.initial['stack_method']
        m.set('stack_method', 'median')
        m.set('stack_method', default)
        assert m.read_form() == {}

    def test_matches_the_cli_builder(self, qapp):
        """What the form submits must be accepted by build_argv_from_form."""
        from src.desktop_control import build_argv_from_form
        from src.qt_app.form_model import FormModel
        m = FormModel()
        m.set('stack_method', 'median')
        m.set('directory', 'lights')
        argv = build_argv_from_form(m.read_form())
        assert '--stack-method' in argv and 'median' in argv

    def test_target_cards_follow_the_fields(self, qapp):
        from src.qt_app.form_model import FormModel
        m = FormModel()
        assert m.selected_target() == 'auto'
        m.pick_target('galaxy')
        assert (m.get('target_type'), m.selected_target()) == ('galaxy', 'galaxy')
        m.pick_target('comet')
        assert m.get('comet_mode') is True and m.selected_target() == 'comet'
        m.pick_target('auto')
        assert m.read_form() == {}

    def test_run_mode(self, qapp):
        from src.qt_app.form_model import FormModel
        m = FormModel()
        assert m.selected_mode() == 'full'
        m.pick_mode('quick')
        assert m.get('preset') == 'quick' and m.selected_mode() == 'quick'

    def test_changed_settings_ignore_folders_and_pickers(self, qapp):
        from src.qt_app.form_model import FormModel
        m = FormModel()
        m.set('output', 'x.fits')
        m.pick_target('galaxy')
        assert m.changed_settings() == []
        m.set('stack_method', 'median')
        assert m.changed_settings() == ['Stack method']

    def test_preset_roundtrip_leaves_folders(self, qapp):
        from src.qt_app.form_model import FormModel
        m = FormModel()
        m.set('stack_method', 'median')
        m.set('output', 'a.fits')
        assert m.save_preset('mine')
        assert 'output' not in m.presets()['mine']
        m.reset('stack_method')
        m.set('use_gpu', True)
        m.apply_preset(m.presets()['mine'])
        assert m.get('stack_method') == 'median'
        assert m.get('use_gpu') is False            # not in the preset: back to its default
        assert m.get('output') == 'a.fits'          # folders stay
        assert m.delete_preset('mine') and 'mine' not in m.presets()

    def test_empty_preset_name_is_refused(self, qapp):
        from src.qt_app.form_model import FormModel
        assert FormModel().save_preset('  ') is False

    def test_describe_run_without_folder(self, qapp):
        from src.qt_app.form_model import FormModel
        rows = dict(FormModel().describe_run())
        assert 'Choose the folder' in rows['Frames']
        assert 'Auto-detect' in rows['Target']

    def test_scan_counts_frames(self, qapp, tmp_path):
        from src.qt_app.form_model import FormModel
        m = FormModel()
        m.set('directory', str(tmp_path))            # empty folder
        assert m.data_summary == '' and m.count_text == 'no frames found'
        m.set('directory', str(tmp_path / 'missing'))
        assert m.count_text == ''

    def test_remember_and_restore_folders(self, qapp, tmp_path):
        from src.qt_app.form_model import FormModel
        m = FormModel()
        m.set('directory', str(tmp_path))
        m.set('output', str(tmp_path / 'stack.fits'))
        m.remember_folders()
        m2 = FormModel()
        m2.restore_folders()
        assert m2.get('directory') == str(tmp_path)
        assert m2.get('output') == str(tmp_path)     # a .fits path is remembered as its folder


# ── Setup form widgets ────────────────────────────────────────────────


@pytest.fixture(scope='module')
def form(qapp):
    from src.qt_app.setup_form import SetupForm
    return SetupForm()


class TestSetupForm:
    def test_every_field_has_a_widget(self, form):
        assert set(form.fields) == set(form.model.fields)

    def test_editing_a_widget_updates_the_model(self, form):
        fv = form.fields['stack_method']
        fv.edit.setCurrentText('median')
        assert form.model.get('stack_method') == 'median'
        assert fv.dot.text() == '●' and fv.reset.text() == 'reset'
        form.model.reset('stack_method')
        assert fv.edit.currentText() == form.model.initial['stack_method']
        assert fv.dot.text() == ''

    def test_checkbox_and_text_fields(self, form):
        gpu = form.fields['use_gpu'].edit
        gpu.setChecked(True)
        assert form.model.get('use_gpu') is True
        form.model.reset('use_gpu')
        assert not gpu.isChecked()
        form.fields['parallel'].edit.setText('4')
        form.fields['parallel'].edit.textEdited.emit('4')
        assert form.model.get('parallel') == '4'
        form.model.reset('parallel')
        assert form.fields['parallel'].edit.text() == form.model.initial['parallel']

    def test_additional_options_toggle_and_count(self, form):
        form.toggle_advanced()
        assert form.advanced.isVisibleTo(form) and '▾' in form.toggle_btn.text()
        dest = next(d for d, f in form.fields.items()
                    if f.group is not None and d not in ('target_type', 'comet_mode', 'preset')
                    and form.model.fields[d]['kind'] == 'bool_true')
        form.model.set(dest, not form.model.initial[dest])
        assert '1 changed' in form.toggle_btn.text()
        form.model.reset(dest)
        form.toggle_advanced()
        assert not form.advanced.isVisibleTo(form)

    def test_search_finds_a_setting_across_groups(self, form):
        form.search.setText('drizzle')
        names = [form.nav.item(i).text() for i in range(form.nav.count())]
        assert names and all('(' in n for n in names)
        assert form.advanced.isVisibleTo(form)           # a search opens the section
        form.search.setText('zzzz-no-such-setting')
        assert form.nav.count() == 0 and form.no_match.isVisibleTo(form)
        form.search.setText('')
        form.toggle_advanced()

    def test_only_changed_lists_changed_expert_fields(self, form):
        expert = next(d for d, f in form.fields.items() if f.group and f.expert
                      and form.model.fields[d]['kind'] == 'bool_true')
        form.model.set(expert, not form.model.initial[expert])
        form.changed_only.setChecked(True)
        assert form.fields[expert].widgets[0].isVisibleTo(form)
        form.changed_only.setChecked(False)
        form.model.reset(expert)
        form.toggle_advanced()

    def test_cards_select_and_badge_the_suggestion(self, form):
        form.model.pick_target('nebula')
        assert form.goal.cards['nebula'].property('selected') == 'true'
        assert form.goal.cards['auto'].property('selected') == 'false'
        form.model.suggested = 'galaxy'
        form.goal.refresh()
        assert form.goal.cards['galaxy'].badge.text() == 'SUGGESTED'
        form.model.suggested = None
        form.model.pick_target('auto')

    def test_presets_combo(self, form):
        form.model.set('stack_method', 'median')
        form.model.save_preset('qt-test')
        form.refresh_presets(select='qt-test')
        assert form.preset_combo.currentText() == 'qt-test'
        form.model.reset('stack_method')
        form._load_preset()
        assert form.model.get('stack_method') == 'median'
        form._delete_preset()
        assert form.preset_combo.findText('qt-test') < 0
        form.model.reset('stack_method')


# ── viewer ────────────────────────────────────────────────────────────


class TestViewer:
    def test_load_zoom_and_clear(self, qapp):
        from src.qt_app.viewer import PreviewPane
        p = PreviewPane()
        p.resize(400, 300)
        p.show()
        assert not p.has_image()
        p.load_slot(_jpeg(), 'final', 'Final')
        assert p.has_image() and p.current_slug == 'final' and p.caption() == 'Final'
        assert p.view.image_size() == (64, 48)
        fit = p.view.scale_factor()
        p.one_to_one()
        assert p.view.scale_factor() == pytest.approx(1.0)
        p.fit()
        assert p.view.scale_factor() == pytest.approx(fit)
        p.clear()
        assert not p.has_image() and p.current_slug == ''

    def test_wheel_zoom_is_bounded(self, qapp):
        from PySide6.QtCore import QPoint, QPointF, Qt
        from PySide6.QtGui import QWheelEvent

        from src.qt_app.viewer import PreviewPane
        p = PreviewPane()
        p.resize(400, 300)
        p.show()
        p.load_slot(_jpeg(size=(400, 300)), 'a', 'a')
        fit = p.view.scale_factor()
        for _i in range(80):
            p.view.wheelEvent(QWheelEvent(QPointF(100, 100), QPointF(100, 100), QPoint(), QPoint(0, 120),
                                          Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                                          Qt.ScrollPhase.NoScrollPhase, False))
        assert p.view.scale_factor() <= 40.0 + 1e-6
        for _i in range(200):
            p.view.wheelEvent(QWheelEvent(QPointF(100, 100), QPointF(100, 100), QPoint(), QPoint(0, -120),
                                          Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                                          Qt.ScrollPhase.NoScrollPhase, False))
        assert p.view.scale_factor() >= fit * 0.5 - 1e-6

    def test_same_size_keeps_the_zoom_a_new_size_refits(self, qapp):
        from src.qt_app.viewer import PreviewPane
        p = PreviewPane()
        p.resize(400, 300)
        p.show()
        p.load_slot(_jpeg(size=(200, 100)), 'a', 'a')
        p.one_to_one()
        p.load_slot(_jpeg((10, 10, 10), size=(200, 100)), 'b', 'b')
        assert p.view.scale_factor() == pytest.approx(1.0)
        p.load_slot(_jpeg(size=(100, 100)), 'c', 'c')
        assert p.view.scale_factor() != pytest.approx(1.0)

    def test_compare_paints_without_error(self, qapp):
        from src.qt_app.viewer import PreviewPane
        p = PreviewPane()
        p.resize(300, 200)
        p.show()
        p.load_slot(_jpeg((255, 0, 0)), 'a', 'a')
        p.set_compare_slot(_jpeg((0, 0, 255)), 'b')
        p.set_compare(True)
        p.set_wipe(0.5)
        from PySide6.QtCore import QPointF
        v = p.view
        left_pt = v.mapFromScene(QPointF(8, 24))             # inside the picture, left of the wipe
        right_pt = v.mapFromScene(QPointF(56, 24))           # inside the picture, right of it
        img = v.viewport().grab().toImage()
        left, right = img.pixelColor(left_pt), img.pixelColor(right_pt)
        assert left.red() > left.blue() and right.blue() > right.red()
        p.set_wipe(0.0)
        assert v.viewport().grab().toImage().pixelColor(left_pt).blue() > 200

    def test_bad_bytes_are_ignored(self, qapp):
        from src.qt_app.viewer import PreviewPane
        p = PreviewPane()
        p.load_slot(b'not an image', 'x', 'x')
        assert not p.has_image()

    def test_histogram_counts_the_picture(self, qapp):
        import numpy as np

        from src.qt_app.viewer import histogram_bins, image_to_array
        from src.qt_app.widgets import qimage_from_bytes
        arr = image_to_array(qimage_from_bytes(_jpeg((255, 0, 0))))
        assert arr.shape == (48, 64, 3) and arr[..., 0].min() > 240
        counts = histogram_bins(arr)
        assert counts.shape == (3, 64)
        assert counts[0, -1] == counts[0].sum() and counts[1, 0] == counts[1].sum()
        assert isinstance(counts, np.ndarray)

    def test_histogram_toggle(self, qapp):
        from src.qt_app.viewer import PreviewPane
        p = PreviewPane()
        p.show()
        p.load_slot(_jpeg(), 'a', 'a')
        p.hist_box.setChecked(True)
        assert p.hist.isVisible() and p.hist._counts is not None
        p.hist_box.setChecked(False)
        assert not p.hist.isVisible()


# ── post-processing panel ─────────────────────────────────────────────


class _FakeUI:
    def __init__(self):
        self.reruns = []
        self.stretches = []
        self.lives = []

    def postprocess_steps(self):
        return [{'id': 'hot_pixel', 'label': 'Hot pixels', 'on': True},
                {'id': 'wavelet', 'label': 'Wavelet denoise', 'on': True},
                {'id': 'scnr', 'label': 'SCNR', 'on': False}]

    def rerun_postprocess(self, off):
        self.reruns.append(list(off))
        return b'JPEG'

    def starless_view(self, on):
        return b'STARLESS' if on else b'JPEG'

    def restretch(self, stretch, color, params=None):
        self.stretches.append((stretch, color, params))
        return b'RESTRETCHED'

    def compare_postprocess_step(self, step_id, cancel=None):
        return b'OFF', b'ON'

    # the stretch editor
    state = {'stretch': 'ghs', 'color': 'preserve', 'ghs_b': 8.0, 'ghs_sp': 0.15, 'ghs_hp': 0.95,
             'black_sigma': 0.0, 'white_pct': 99.5, 'nebula_auto': True}

    def stretch_state(self):
        return dict(self.state)

    def reset_stretch(self):
        return dict(self.state)

    def live_stretch(self, params=None, stretch=None, color=None, max_dim=1400, persist=True):
        self.lives.append((dict(params or {}), stretch, color))
        edges = np.linspace(-2.0, 6.0, 61)
        counts = np.random.default_rng(0).random((4, 60))
        lum = np.linspace(0.0, 1.0, 1000)
        lifted = bool((params or {}).get('nebula_auto', True))
        return {'jpeg': _jpeg((9, 9, 9)), 'hist': {'edges': edges, 'counts': counts},
                'curve': (np.linspace(-1, 5, 20), np.linspace(0, 1, 20)),
                'markers': {'black': 0.0, 'white': 4.0}, 'kw': {},
                'effective': {'ghs_b': 14.0, 'ghs_sp': 0.0, 'lifted': lifted},
                'lum': lum, 'med': 0.0, 'sigma': 0.1}


class _FakeRM:
    running = False

    def is_running(self):
        return self.running


def _snap(path='out.fits', status='ok'):
    return {'restretch': {'path': path, 'stretch': 'ghs', 'color': 'preserve'},
            'postprocess': True, 'run_status': status}


def _wait(app, cond, seconds=5.0):
    import time
    end = time.time() + seconds
    while time.time() < end and not cond():
        app.processEvents()
        time.sleep(0.01)
    return cond()


class TestPostProcessPanel:
    @pytest.fixture
    def panel(self, qapp):
        from src.qt_app import post_panel
        post_panel.PP_DEBOUNCE_MS = 20
        p = post_panel.PostProcessPanel(_FakeUI(), _FakeRM())
        p.resize(700, 300)
        p.show()
        return p

    def test_hidden_until_a_run_finishes(self, panel):
        panel.sync({'restretch': None, 'postprocess': False, 'run_status': 'idle'})
        assert not panel.pp_group.isVisible()
        panel.sync(_snap(status='running'))
        assert not panel.pp_group.isVisible()

    def test_one_box_per_step_ticked_as_applied(self, panel):
        panel.sync(_snap())
        assert panel.pp_group.isVisible()
        assert {i: b.isChecked() for i, b in panel._pp_boxes.items()} == \
            {'hot_pixel': True, 'wavelet': True, 'scnr': False}
        assert panel.cmp_combo.count() == 3

    def test_unticking_a_step_reruns_phase4_with_the_off_list(self, qapp, panel):
        panel.sync(_snap())
        got = []
        panel.image_ready.connect(lambda data, slug, cap: got.append((data, slug, cap)))
        panel._pp_boxes['wavelet'].setChecked(False)
        assert _wait(qapp, lambda: bool(got))
        assert panel.ui.reruns == [['wavelet', 'scnr']]
        assert got[0][1] == 'final' and 'post-processed' in got[0][2]

    def test_a_burst_of_clicks_runs_once(self, qapp, panel):
        panel.sync(_snap())
        got = []
        panel.image_ready.connect(lambda *a: got.append(a))
        for i in ('hot_pixel', 'wavelet', 'hot_pixel'):
            panel._pp_boxes[i].toggle()
        assert _wait(qapp, lambda: bool(got))
        _wait(qapp, lambda: False, 0.2)
        assert len(panel.ui.reruns) == 1

    def test_starless_toggle(self, qapp, panel):
        panel.sync(_snap())
        got = []
        panel.image_ready.connect(lambda data, slug, cap: got.append((data, cap)))
        panel.starless_box.setChecked(True)
        assert _wait(qapp, lambda: bool(got))
        assert got[-1] == (b'STARLESS', 'Final (starless)')

    def test_with_live_off_changes_wait_for_apply(self, qapp, panel):
        panel.sync(_snap())
        panel.live_box.setChecked(False)
        panel._pp_boxes['wavelet'].setChecked(False)
        _wait(qapp, lambda: False, 0.2)
        assert panel.ui.reruns == [] and 'not applied' in panel.pp_status.text()
        got = []
        panel.image_ready.connect(lambda *a: got.append(a))
        panel.apply_btn.click()
        assert _wait(qapp, lambda: bool(got))
        assert panel.ui.reruns == [['wavelet', 'scnr']]

    def test_nothing_runs_while_a_stack_is_running(self, qapp, panel):
        panel.sync(_snap())
        panel.rm.running = True
        panel._pp_boxes['wavelet'].setChecked(False)
        _wait(qapp, lambda: False, 0.2)
        assert panel.ui.reruns == []

    def test_a_new_run_rebuilds_the_boxes(self, panel):
        panel.sync(_snap('a.fits'))
        first = list(panel._pp_boxes.values())
        panel.sync(_snap('b.fits'))
        assert list(panel._pp_boxes.values()) != first


# ── stretch editor ────────────────────────────────────────────────────


@pytest.fixture
def editor(qapp):
    from src.qt_app import stretch_panel
    stretch_panel.LIVE_DEBOUNCE_MS = 5
    e = stretch_panel.StretchEditor(_FakeUI(), _FakeRM(), ['linear', 'arcsinh', 'ghs', 'seti'],
                                    ['preserve', 'channel'])
    e.resize(700, 420)
    e.show()
    return e


class TestStretchEditor:
    def test_unavailable_until_a_run_finishes(self, editor):
        editor.sync({'restretch': None, 'run_status': 'running'})
        assert not editor.available

    def test_loads_the_runs_settings_and_draws_the_levels(self, qapp, editor):
        seen = []
        editor.availability_changed.connect(seen.append)
        editor.sync(_snap())
        assert editor.available and seen == [True]
        assert editor.stretch_combo.currentText() == 'ghs'
        assert editor.rows['ghs_b'].value() == 8.0 and editor.auto_box.isChecked()
        assert _wait(qapp, lambda: editor.levels.hist is not None)
        assert editor.levels.black == 0.0 and editor.levels.white == 4.0

    def test_the_initial_levels_do_not_replace_the_picture(self, qapp, editor):
        shown = []
        editor.live_image.connect(lambda *a: shown.append(a))
        editor.sync(_snap())
        assert _wait(qapp, lambda: editor.levels.hist is not None)
        _wait(qapp, lambda: False, 0.1)
        assert shown == []

    def test_a_slider_move_renders_a_live_picture_with_that_setting(self, qapp, editor):
        editor.sync(_snap())
        _wait(qapp, lambda: editor.levels.hist is not None)
        shown = []
        editor.live_image.connect(lambda data, cap: shown.append((data, cap)))
        editor.rows['ghs_b'].spin.setValue(12.0)
        assert _wait(qapp, lambda: bool(shown))
        params, stretch, color = editor.ui.lives[-1]
        assert params['ghs_b'] == 12.0 and stretch == 'ghs' and color == 'preserve'
        assert shown[-1][1] == 'Final (live preview)'

    def test_spin_and_slider_stay_in_step(self, editor):
        row = editor.rows['black_sigma']
        row.spin.setValue(2.5)
        assert row.slider.value() == int(round((2.5 - row._lo) / row._step))
        row.slider.setValue(row.slider.value() + 10)
        assert row.value() == pytest.approx(3.5)
        seen = []
        row.edited.connect(lambda k, v: seen.append(v))
        row.set_value(2.0)                          # programmatic: no edit reported
        assert seen == []

    def test_a_burst_of_edits_shows_only_the_last(self, qapp, editor):
        editor.sync(_snap())
        _wait(qapp, lambda: editor.levels.hist is not None)
        shown = []
        editor.live_image.connect(lambda *a: shown.append(a))
        for v in (3.0, 6.0, 9.0, 12.0, 15.0):
            editor.rows['ghs_b'].spin.setValue(v)
        assert _wait(qapp, lambda: bool(shown))
        _wait(qapp, lambda: False, 0.2)
        assert editor.ui.lives[-1][0]['ghs_b'] == 15.0
        assert len(editor.ui.lives) < 6

    def test_controls_follow_the_stretch(self, editor):
        editor.stretch_combo.setCurrentText('arcsinh')
        editor.stretch_combo.activated.emit(editor.stretch_combo.currentIndex())
        assert [k for k, r in editor.rows.items() if r.slider.isEnabled()] == ['black_sigma', 'white_pct']
        assert not editor.auto_box.isEnabled()
        editor.stretch_combo.setCurrentText('linear')
        editor.stretch_combo.activated.emit(editor.stretch_combo.currentIndex())
        assert not any(r.slider.isEnabled() for r in editor.rows.values())
        editor.stretch_combo.setCurrentText('ghs')
        editor.stretch_combo.activated.emit(editor.stretch_combo.currentIndex())
        assert all(r.slider.isEnabled() for r in editor.rows.values())

    def test_dragging_the_markers_sets_black_and_white(self, qapp, editor):
        editor.sync(_snap())
        _wait(qapp, lambda: editor.levels.hist is not None)
        from src import stretch_editor as se
        editor.levels.black_moved.emit(float(se.to_axis(2.0)))
        assert editor.rows['black_sigma'].value() == pytest.approx(2.0, abs=0.06)
        editor.levels.white_moved.emit(editor.levels.white)
        assert 95.0 <= editor.rows['white_pct'].value() <= 99.99

    def test_turning_auto_lift_off_keeps_what_it_chose(self, qapp, editor):
        editor.sync(_snap())
        editor.rows['ghs_b'].spin.setValue(3.0)
        assert _wait(qapp, lambda: 'Auto-lift set strength 14.0' in editor.status.text())
        editor.auto_box.setChecked(False)
        assert editor.rows['ghs_b'].value() == 14.0 and editor.rows['ghs_sp'].value() == 0.0
        assert _wait(qapp, lambda: editor.ui.lives[-1][0].get('nebula_auto') is False)

    def test_apply_saves_with_the_settings(self, qapp, editor):
        editor.sync(_snap())
        editor.rows['white_pct'].spin.setValue(99.0)
        got = []
        editor.image_ready.connect(lambda data, slug, cap: got.append((data, slug, cap)))
        editor.apply_btn.click()
        assert _wait(qapp, lambda: bool(got))
        stretch, color, params = editor.ui.stretches[-1]
        assert (stretch, color) == ('ghs', 'preserve') and params['white_pct'] == 99.0
        assert got[0][:2] == (b'RESTRETCHED', 'final')
        assert 'out.fits' in editor.status.text()

    def test_reset_puts_the_runs_values_back_and_shows_them(self, qapp, editor):
        editor.sync(_snap())
        editor.rows['ghs_b'].spin.setValue(20.0)
        shown = []
        editor.live_image.connect(lambda *a: shown.append(a))
        editor.reset_btn.click()
        assert editor.rows['ghs_b'].value() == 8.0
        assert _wait(qapp, lambda: bool(shown))

    def test_nothing_renders_while_a_stack_is_running(self, qapp, editor):
        editor.sync(_snap())
        _wait(qapp, lambda: editor.levels.hist is not None)
        n = len(editor.ui.lives)
        editor.rm.running = True
        editor.rows['ghs_b'].spin.setValue(11.0)
        _wait(qapp, lambda: False, 0.15)
        assert len(editor.ui.lives) == n

    def test_goes_away_with_the_run(self, editor):
        editor.sync(_snap())
        seen = []
        editor.availability_changed.connect(seen.append)
        editor.sync({'restretch': None, 'run_status': 'running'})
        assert not editor.available and seen == [False] and editor.levels.hist is None


class TestLevelsView:
    def _view(self, qapp):
        from src.qt_app.levels import LevelsView
        v = LevelsView()
        v.resize(400, 140)
        v.show()
        edges = np.linspace(-2.0, 6.0, 41)
        v.set_data({'edges': edges, 'counts': np.random.default_rng(1).random((4, 40))},
                   (np.linspace(-1, 5, 10), np.linspace(0, 1, 10)), 0.0, 4.0)
        return v

    @staticmethod
    def _mouse(kind, x):
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QMouseEvent
        return QMouseEvent(kind, QPointF(x, 60), QPointF(x, 60), Qt.MouseButton.LeftButton,
                           Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)

    def test_paints_with_and_without_data(self, qapp):
        from src.qt_app.levels import LevelsView
        empty = LevelsView()
        empty.resize(300, 120)
        assert not empty.grab().isNull()
        assert not self._view(qapp).grab().isNull()

    def test_dragging_a_marker_reports_axis_positions_and_never_crosses(self, qapp):
        from PySide6.QtCore import QEvent
        v = self._view(qapp)
        got = []
        v.black_moved.connect(lambda u: got.append(('b', u)))
        v.white_moved.connect(lambda u: got.append(('w', u)))
        v.mousePressEvent(self._mouse(QEvent.Type.MouseButtonPress, v._x(0.0)))
        v.mouseMoveEvent(self._mouse(QEvent.Type.MouseMove, v._x(2.0)))
        v.mouseMoveEvent(self._mouse(QEvent.Type.MouseMove, v._x(5.5)))      # past the white marker
        v.mouseReleaseEvent(self._mouse(QEvent.Type.MouseButtonRelease, v._x(5.5)))
        assert got[0][0] == 'b' and got[-1][1] == pytest.approx(4.0)         # held at the white marker
        assert v._drag is None

    def test_a_press_away_from_a_marker_does_nothing(self, qapp):
        from PySide6.QtCore import QEvent
        v = self._view(qapp)
        got = []
        v.black_moved.connect(got.append)
        v.white_moved.connect(got.append)
        v.mousePressEvent(self._mouse(QEvent.Type.MouseButtonPress, v._x(2.0)))
        assert got == [] and v._drag is None

    def test_disabled_markers_cannot_be_grabbed(self, qapp):
        v = self._view(qapp)
        v.set_enabled_markers(False, False)
        assert v._marker_at(v._x(0.0)) is None


# ── main window ───────────────────────────────────────────────────────


@pytest.fixture(scope='module')
def win(qapp):
    from src.qt_app.main_window import MainWindow
    w = MainWindow()
    w.resize(1300, 850)
    w.show()
    yield w
    w._timer.stop()
    w.ui.detach()


class TestMainWindow:
    def test_starts_idle_with_the_plan_shown(self, win):
        assert win.start_btn.isEnabled() and not win.cancel_btn.isEnabled()
        assert win.summary_group.title() == 'THIS RUN'
        assert win.plan_grid.count() >= 8

    def test_example_picture_follows_the_target_card(self, win):
        win.model.pick_target('galaxy')
        assert win.preview.current_slug == 'example' and 'galaxy' in win.preview.caption().lower()
        win.model.pick_target('auto')

    def test_frame_sorting_cycles(self, win):
        rows = [('b', 1.0, 5, 3, 4), ('a', 2.0, 4, 9, 3), ('c', '', 6, 1, 5)]
        win._frame_sort = None
        assert win._sorted_frame_rows(rows) == rows
        win._sort_frames('score')                        # quality columns: highest first
        assert win._frame_sort == ('score', True)
        assert [r[0] for r in win._sorted_frame_rows(rows)] == ['a', 'b', 'c']
        win._sort_frames('score')
        assert win._frame_sort == ('score', False)
        win._sort_frames('score')
        assert win._frame_sort is None
        win._sort_frames('frame')                        # names: A to Z first
        assert [r[0] for r in win._sorted_frame_rows(rows)] == ['a', 'b', 'c']
        win._sort_frames('frame')
        win._sort_frames('frame')

    def test_preflight_with_no_folder(self, win):
        win.model.set('directory', '')
        assert all(lbl.text().startswith('○') for lbl in win.preflight_labels.values())

    def test_dropping_a_folder_sets_the_lights(self, win, tmp_path):
        from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
        from PySide6.QtGui import QDropEvent
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(tmp_path))])
        ev = QDropEvent(QPointF(10, 10), Qt.DropAction.CopyAction, mime,
                        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        win.dropEvent(ev)
        assert os.path.normpath(win.model.get('directory')) == os.path.normpath(str(tmp_path))
        win.model.set('directory', '')

    def test_single_letter_shortcuts_do_not_steal_typing(self, win):
        """F, 1 and C act on the picture only; the Setup fields still get those keys."""
        from PySide6.QtGui import QShortcut
        singles = [s for s in win.findChildren(QShortcut) if len(s.key().toString()) == 1]
        assert singles and all(s.parent() is win.preview for s in singles)

    def test_refresh_follows_milestones_and_fills_frames(self, win, qapp):
        win.ui.run_started()
        win.ui._set_named_slot('quick-look', 'Quick look', _jpeg())
        win._refresh()
        assert win.view_combo.count() == 1 and win.preview.current_slug == 'quick-look'
        win.ui._set_named_slot('final', 'Final', _jpeg((10, 200, 10)))
        win._refresh()
        assert win.view_combo.count() == 2 and win.preview.current_slug == 'final'
        win.compare_box.setChecked(True)
        assert win.compare_combo.isEnabled() and win.wipe.isVisibleTo(win)
        win.compare_box.setChecked(False)
        assert not win.wipe.isVisibleTo(win)

    def test_close_confirms_while_running(self, win, monkeypatch):
        from PySide6.QtGui import QCloseEvent
        from PySide6.QtWidgets import QMessageBox
        monkeypatch.setattr(win.rm, 'is_running', lambda: True)
        monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: QMessageBox.StandardButton.No)
        ev = QCloseEvent()
        win.closeEvent(ev)
        assert not ev.isAccepted()
        monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: QMessageBox.StandardButton.Yes)
        ev = QCloseEvent()
        win.closeEvent(ev)
        assert ev.isAccepted()


def test_done_status_is_applied_even_when_the_run_thread_outlives_the_snapshot(qapp, monkeypatch):
    """The finished snapshot can arrive while the run thread still counts as running; the
    window must show Done on a later poll, not stay on Running."""
    from src.qt_app.main_window import MainWindow
    w = MainWindow()
    w._timer.stop()
    state = {'running': True}
    monkeypatch.setattr(w.rm, 'is_running', lambda: state['running'])
    w.ui.run_started()
    w.ui.run_finished('ok')
    w._refresh()                                    # new snapshot, thread still winding down
    assert not w.start_btn.isEnabled() and w.header_status.text() == 'Running…'
    state['running'] = False
    w._refresh()                                    # same snapshot version
    assert w.start_btn.isEnabled()
    assert w.status_label.text() == 'Done' and w.header_status.text() == 'Complete'


def test_typing_a_folder_scans_once_after_the_pause(qapp, tmp_path, monkeypatch):
    import time

    from src.qt_app import setup_form
    monkeypatch.setattr(setup_form, 'RESCAN_DEBOUNCE_MS', 30)
    f = setup_form.SetupForm()
    f._rescan_timer.setInterval(30)
    scans = []
    f.model.scanned.connect(lambda: scans.append(1))
    fv = f.fields['directory']
    for part in ('C', str(tmp_path)[:6], str(tmp_path)):
        fv.edit.setText(part)
        fv.edit.textEdited.emit(part)
    assert scans == []                               # nothing scanned while typing
    end = time.time() + 2
    while not scans and time.time() < end:
        qapp.processEvents()
        time.sleep(0.01)
    assert len(scans) == 1 and f.model.get('directory') == str(tmp_path)
    f.model.set('directory', '')                      # programmatic sets still scan at once
    assert len(scans) == 2


def test_stale_live_result_after_a_new_run_is_dropped(qapp):
    from src.qt_app import stretch_panel
    e = stretch_panel.StretchEditor(_FakeUI(), _FakeRM(), ['ghs'], ['preserve'])
    shown = []
    e.live_image.connect(lambda *a: shown.append(a))
    e.available = False                               # a new run reset the editor
    e._on_live(1, 'ok', _FakeUI().live_stretch({}), True)
    assert shown == [] and e._last is None and e.levels.hist is None


# ── theme ─────────────────────────────────────────────────────────────


class TestTheme:
    @pytest.mark.parametrize('mode', ['dark', 'light'])
    def test_text_colours_clear_wcag_aa(self, qapp, mode):
        from src.qt_app import theme
        pal = theme.PALETTES[mode]
        for fg in ('TEXT', 'TEXT_DIM', 'TEXT_FAINT', 'ACCENT', 'ACCENT2', 'LOG_FG', 'SUCCESS', 'BAD'):
            for bg in ('BG', 'PANEL', 'WELL'):
                assert theme.contrast(pal[fg], pal[bg]) >= 4.5, (mode, fg, bg)
        assert theme.contrast(pal['ACCENT_TEXT'], pal['ACCENT']) >= 4.5

    @pytest.mark.parametrize('mode', ['dark', 'light'])
    def test_stylesheet_builds_and_draws_its_icons(self, qapp, mode):
        import os

        from src.qt_app import theme
        try:
            assert theme.apply_mode(mode) == mode and theme.BG == theme.PALETTES[mode]['BG']
            css = theme.stylesheet()
            assert theme.PALETTES[mode]['ACCENT'] in css
            icons = theme.make_icons()
            assert all(os.path.getsize(p) > 0 for p in icons.values())
            assert icons['down'] in css and icons['check'] in css
        finally:
            theme.apply_mode('dark')

    def test_mode_resolution(self, qapp):
        from src.qt_app import theme
        assert theme.resolve_mode('light') == 'light' and theme.resolve_mode('dark') == 'dark'
        assert theme.resolve_mode('auto') in ('dark', 'light')
        assert theme.apply_mode('nonsense') == 'dark'

    def test_window_paints_in_light_mode(self, qapp):
        from src.qt_app import theme
        from src.qt_app.main_window import MainWindow
        theme.apply_mode('light')
        try:
            qapp.setStyleSheet(theme.stylesheet())
            w = MainWindow()
            w._timer.stop()
            w.resize(1200, 800)
            w.show()
            img = w.grab().toImage()
            assert img.pixelColor(img.width() - 5, img.height() // 2).lightness() > 128
        finally:
            theme.apply_mode('dark')
            qapp.setStyleSheet(theme.stylesheet())


# ── remembered state and menus ────────────────────────────────────────


class TestWindowState:
    def test_recent_folders_newest_first_unique_and_capped(self, qapp):
        from src.qt_app import window_state as ws
        for i in range(12):
            ws.remember_folder(f'/data/{i}')
        ws.remember_folder('/data/5')
        got = ws.recent_folders()
        assert got[0] == '/data/5' and len(got) == ws.MAX_RECENT and len(set(got)) == len(got)

    def test_encode_decode_roundtrip(self, qapp):
        from PySide6.QtCore import QByteArray

        from src.qt_app import window_state as ws
        raw = QByteArray(b'\x00\x01state\xff')
        assert bytes(ws.decode(ws.encode(raw))) == bytes(raw)
        ws.decode('not base64 !!')                              # never raises

    def test_load_survives_garbage(self, qapp):
        from src.desktop_control import save_app_setting
        from src.qt_app import window_state as ws
        save_app_setting('qt_window', 'garbage')
        assert ws.load() == {}


class TestMenusAndLayout:
    def test_menus_exist(self, win):
        titles = [a.text().replace('&', '') for a in win.menuBar().actions()]
        assert titles == ['File', 'View', 'Help']

    def test_tools_collapse_and_maximise(self, win):
        win.set_tools_collapsed(True)
        assert win.tools.maximumHeight() < 100 and win.collapse_btn.text() == '▴'
        assert not win.act_tools.isChecked()
        win.set_tools_collapsed(False)
        assert win.tools.maximumHeight() > 1000 and win.act_tools.isChecked()
        win.set_picture_maximised(True)
        assert not win.tools.isVisible() and win.act_max.isChecked()
        win.set_picture_maximised(False)

    def test_save_and_restore_state(self, win):
        from src.qt_app import window_state as ws
        win.set_tools_collapsed(True)
        win.form.goal.set_cards_open(False)
        win._save_state()
        saved = ws.load()
        assert saved['tools_collapsed'] is True and saved['goals_open'] is False and saved['geometry']
        win.set_tools_collapsed(False)
        win.form.goal.set_cards_open(True)
        win._restore_state()
        assert win._tools_collapsed is True and win.form.goal.cards_open is False
        win.set_tools_collapsed(False)
        win.form.goal.set_cards_open(True)

    def test_recent_menu_lists_existing_folders_only(self, win, tmp_path):
        from src.qt_app import window_state as ws
        ws.remember_folder(str(tmp_path))
        ws.remember_folder(str(tmp_path / 'gone'))
        win._refresh_recent_menu()
        texts = [a.text() for a in win.recent_menu.actions()]
        assert str(tmp_path) in texts and str(tmp_path / 'gone') not in texts

    def test_cards_fold_to_one_line(self, win):
        g = win.form.goal
        win.model.pick_target('nebula')
        g.set_cards_open(False)
        assert not g.cards_box.isVisibleTo(win.form) and g.cards_summary.text() == 'Nebula'
        assert g.cards_toggle.text().startswith('▸')
        g.set_cards_open(True)
        assert g.cards_box.isVisibleTo(win.form) and g.cards_summary.text() == ''
        win.model.pick_target('auto')

    def test_phase_times_text(self, win):
        snap = {'phases': {1: {'title': 'Quality', 'elapsed': 12.4}, 2: {'title': 'Stacking', 'elapsed': 31.0},
                           3: {'title': 'Post', 'elapsed': None}}}
        assert win._phase_times(snap) == 'Quality 12 s · Stacking 31 s'
        assert win._phase_times({}) == ''

    def test_shortcuts_dialog_lists_every_shortcut(self, qapp):
        from PySide6.QtWidgets import QLabel

        from src.qt_app.help_dialog import SHORTCUTS, ShortcutsDialog
        d = ShortcutsDialog()
        texts = {lbl.text() for lbl in d.findChildren(QLabel)}
        assert all(keys in texts for keys, _w, _where in SHORTCUTS)

    def test_the_finished_run_closes_the_last_phase(self):
        from src.ui_events import UIEvents
        u = UIEvents()
        u.attach()
        u.run_started()
        u.phase(1, 'Quality')
        u.run_finished('ok')
        assert u.snapshot()['phases'][1]['elapsed'] is not None


# ── editor: reset, undo, presets, drag, peek ──────────────────────────


class TestEditorExtras:
    def test_double_click_resets_a_slider_and_the_mark_follows(self, qapp, editor):
        editor.sync(_snap())
        row = editor.rows['ghs_b']
        assert not row.mark.isVisibleTo(editor)
        row.spin.setValue(15.0)
        assert row.mark.isVisibleTo(editor)
        row.slider.double_clicked.emit()
        assert row.value() == 8.0 and not row.mark.isVisibleTo(editor)
        row.spin.setValue(3.0)
        row.label.double_clicked.emit()
        assert row.value() == 8.0
        row.spin.setValue(4.0)
        row.mark.clicked.emit()
        assert row.value() == 8.0

    def test_undo_and_redo_walk_the_edit_history(self, qapp, editor):
        editor.sync(_snap())
        editor.rows['ghs_b'].spin.setValue(12.0)
        editor._commit_timer.stop()
        editor._commit()
        editor.rows['black_sigma'].spin.setValue(2.0)
        editor._commit_timer.stop()
        editor._commit()
        assert editor.undo_btn.isEnabled() and not editor.redo_btn.isEnabled()
        editor.undo()
        assert editor.rows['black_sigma'].value() == 0.0 and editor.rows['ghs_b'].value() == 12.0
        editor.undo()
        assert editor.rows['ghs_b'].value() == 8.0 and not editor.undo_btn.isEnabled()
        editor.redo()
        editor.redo()
        assert editor.rows['black_sigma'].value() == 2.0 and not editor.redo_btn.isEnabled()

    def test_an_edit_after_undo_drops_the_redo_branch(self, qapp, editor):
        editor.sync(_snap())
        for v in (10.0, 12.0):
            editor.rows['ghs_b'].spin.setValue(v)
            editor._commit_timer.stop()
            editor._commit()
        editor.undo()
        editor.rows['ghs_sp'].spin.setValue(0.3)
        editor._commit_timer.stop()
        editor._commit()
        assert not editor.redo_btn.isEnabled()

    def test_a_burst_of_edits_is_one_undo_step(self, qapp, editor, monkeypatch):
        from src.qt_app import stretch_panel
        monkeypatch.setattr(stretch_panel, 'COMMIT_MS', 40)
        editor.sync(_snap())
        n = len(editor._history)
        for v in (9.0, 10.0, 11.0, 12.0):
            editor.rows['ghs_b'].spin.setValue(v)
        _wait(qapp, lambda: len(editor._history) > n)
        assert len(editor._history) == n + 1

    def test_dragging_renders_small_then_full_on_release(self, qapp, editor):
        from src.qt_app import stretch_panel
        calls = []
        orig = editor.ui.live_stretch
        editor.ui.live_stretch = lambda p=None, s=None, c=None, m=1400, persist=True: (
            calls.append(m), orig(p, s, c, m, persist))[1]
        editor.sync(_snap())
        _wait(qapp, lambda: editor.levels.hist is not None)
        calls.clear()
        editor.rows['ghs_b'].drag_started.emit()
        editor.rows['ghs_b'].spin.setValue(11.0)
        _wait(qapp, lambda: stretch_panel.DRAG_DIM in calls)
        editor.rows['ghs_b'].drag_finished.emit()
        _wait(qapp, lambda: calls and calls[-1] == stretch_panel.FULL_DIM)
        assert calls[0] == stretch_panel.DRAG_DIM and calls[-1] == stretch_panel.FULL_DIM

    def test_saved_stretches(self, qapp, editor, monkeypatch):
        from PySide6.QtWidgets import QInputDialog
        editor.sync(_snap())
        editor.rows['ghs_b'].spin.setValue(13.0)
        monkeypatch.setattr(QInputDialog, 'getText', lambda *a, **k: ('Deep', True))
        editor.preset_save.click()
        assert editor.preset_combo.currentText() == 'Deep'
        editor.rows['ghs_b'].spin.setValue(2.0)
        editor.preset_combo.activated.emit(editor.preset_combo.currentIndex())
        assert editor.rows['ghs_b'].value() == 13.0
        editor.preset_delete.click()
        assert editor.preset_combo.findText('Deep') < 0

    def test_hold_to_peek_shows_the_runs_stretch_and_keeps_the_settings(self, qapp, editor):
        editor.sync(_snap())
        _wait(qapp, lambda: editor.levels.hist is not None)
        editor.rows['ghs_b'].spin.setValue(18.0)
        _wait(qapp, lambda: False, 0.15)
        shown = []
        editor.live_image.connect(lambda data, cap: shown.append(cap))
        editor.peek(True)
        assert _wait(qapp, lambda: 'Final (as the run made it)' in shown)
        editor.peek(False)
        assert _wait(qapp, lambda: shown[-1] == 'Final (live preview)')
        assert editor.rows['ghs_b'].value() == 18.0

    def test_dragging_the_symmetry_marker_sets_the_symmetry_point(self, qapp, editor):
        editor.sync(_snap())
        editor.auto_box.setChecked(False)
        _wait(qapp, lambda: editor.levels.sym is not None)
        xb, xw = editor._axis_xs()
        from src import stretch_editor as se
        editor._on_sym_dragged(float(se.to_axis(xb + 0.25 * (xw - xb))))
        assert editor.rows['ghs_sp'].value() == pytest.approx(0.25, abs=0.01)

    def test_marker_labels_follow_the_values(self, editor):
        editor.rows['black_sigma'].spin.setValue(1.5)
        editor.rows['white_pct'].spin.setValue(99.2)
        assert editor.levels.labels['black'] == '1.5σ' and editor.levels.labels['white'] == '99.20%'

    def test_log_scale_toggle(self, editor):
        editor.log_box.setChecked(False)
        assert not editor.levels.log_y
        editor.log_box.setChecked(True)
        assert editor.levels.log_y

    def test_the_saved_message_is_not_overwritten_by_a_late_render(self, qapp, editor):
        editor.sync(_snap())
        _wait(qapp, lambda: editor.levels.hist is not None)
        editor._sticky = True
        editor.status.setText('Saved out.fits')
        editor._request(show=True)
        _wait(qapp, lambda: not editor._busy)
        assert editor.status.text() == 'Saved out.fits'
        editor.rows['ghs_b'].spin.setValue(5.0)             # an edit releases it
        assert not editor._sticky


class TestLevelsExtras:
    def test_readout_and_labels(self, qapp):
        from PySide6.QtCore import QEvent

        from src.qt_app.levels import LevelsView
        v = LevelsView()
        v.resize(400, 140)
        v.show()
        edges = np.linspace(-2.0, 6.0, 41)
        v.set_data({'edges': edges, 'counts': np.ones((4, 40))}, None, 0.0, 4.0, 2.0)
        v.set_lum(np.linspace(0, 10, 1000), 0.0, 1.0)
        assert v.readout() == ''
        v._hover_x = v._x(0.0)
        assert 'σ above the sky' in v.readout()
        v.set_marker_labels(black='1σ', white='99%')
        v.set_enabled_markers(True, True, sym=True)
        assert not v.grab().isNull()
        v.leaveEvent(QEvent(QEvent.Type.Leave))
        assert v.readout() == ''

    def test_the_symmetry_marker_stays_between_black_and_white(self, qapp):
        from PySide6.QtCore import QEvent
        v = TestLevelsView()._view(qapp)
        v.set_enabled_markers(True, True, sym=True)
        v.sym = 2.0
        got = []
        v.sym_moved.connect(got.append)
        v.mousePressEvent(TestLevelsView._mouse(QEvent.Type.MouseButtonPress, v._x(2.0)))
        v.mouseMoveEvent(TestLevelsView._mouse(QEvent.Type.MouseMove, v._x(9.0)))
        v.mouseReleaseEvent(TestLevelsView._mouse(QEvent.Type.MouseButtonRelease, v._x(9.0)))
        assert got and got[-1] == pytest.approx(4.0)            # held at the white marker


# ── viewer extras ─────────────────────────────────────────────────────


class TestViewerExtras:
    def _pane(self, qapp):
        from src.qt_app.viewer import PreviewPane
        p = PreviewPane()
        p.resize(400, 300)
        p.show()
        p.load_slot(_jpeg((200, 40, 40), size=(200, 150)), 'a', 'a')
        return p

    def test_pixel_inspector_reads_the_pixel(self, qapp):
        from PySide6.QtCore import QPointF
        p = self._pane(qapp)
        p.inspect_box.setChecked(True)
        pt = p.view.mapFromScene(QPointF(50, 50))
        p.view._emit_pixel(QPointF(pt))
        text = p.pixel_label.text()
        import re
        x, y = (int(v) for v in re.match(r'x (\d+)\s+y (\d+)', text).groups())
        assert abs(x - 50) <= 1 and abs(y - 50) <= 1 and 'R 20' in text    # mapFromScene rounds to a whole view pixel
        p.view._emit_pixel(QPointF(-500, -500))
        assert p.pixel_label.text() == ''
        p.inspect_box.setChecked(False)
        assert p.pixel_label.text() == ''

    def test_minimap_only_when_zoomed_in(self, qapp):
        p = self._pane(qapp)
        assert p.view._minimap_rect() is None
        p.view._zoom_by(4.0)
        mm = p.view._minimap_rect()
        assert mm is not None and mm.right() <= p.view.viewport().width()
        assert not p.view.viewport().grab().isNull()
        p.view.show_minimap = False
        assert p.view._minimap_rect() is None

    def test_clicking_the_minimap_recentres(self, qapp):
        from PySide6.QtCore import QEvent, QPointF, Qt
        from PySide6.QtGui import QMouseEvent
        p = self._pane(qapp)
        p.view._zoom_by(6.0)
        mm = p.view._minimap_rect()
        target = QPointF(mm.left() + 3, mm.top() + 3)              # the top-left corner of the picture
        p.view.mousePressEvent(QMouseEvent(QEvent.Type.MouseButtonPress, target, target,
                                           Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                                           Qt.KeyboardModifier.NoModifier))
        centre = p.view.mapToScene(p.view.viewport().rect().center())
        assert centre.x() < 60 and centre.y() < 60
        p.view.mouseReleaseEvent(QMouseEvent(QEvent.Type.MouseButtonRelease, target, target,
                                             Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
                                             Qt.KeyboardModifier.NoModifier))

    def test_the_wipe_line_can_be_dragged_on_the_picture(self, qapp):
        from PySide6.QtCore import QEvent, QPointF, Qt
        from PySide6.QtGui import QMouseEvent
        p = self._pane(qapp)
        p.set_compare_slot(_jpeg((0, 0, 255), size=(200, 150)), 'b')
        p.set_compare(True)
        p.set_wipe(0.5)
        got = []
        p.wipe_dragged.connect(got.append)
        x = p.view._wipe_x()

        def ev(kind, px, buttons=Qt.MouseButton.LeftButton):
            pt = QPointF(px, 100)
            return QMouseEvent(kind, pt, pt, Qt.MouseButton.LeftButton, buttons,
                               Qt.KeyboardModifier.NoModifier)
        p.view.mousePressEvent(ev(QEvent.Type.MouseButtonPress, x + 2))
        p.view.mouseMoveEvent(ev(QEvent.Type.MouseMove, x + 80))
        p.view.mouseReleaseEvent(ev(QEvent.Type.MouseButtonRelease, x + 80, Qt.MouseButton.NoButton))
        assert got and got[-1] == pytest.approx((x + 80) / p.view.viewport().width(), abs=0.01)
        assert p.view.wipe_frac == got[-1] and not p.view._wipe_drag

    def test_holding_b_reports_down_and_up_once(self, qapp):
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtGui import QKeyEvent
        p = self._pane(qapp)
        got = []
        p.hold_original.connect(got.append)
        no = Qt.KeyboardModifier.NoModifier
        p.view.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_B, no))
        p.view.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_B, no, 'b', True))   # auto-repeat
        p.view.keyReleaseEvent(QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_B, no))
        assert got == [True, False]

    def test_empty_state_shows_the_hint(self, qapp):
        from src.qt_app.viewer import PreviewPane
        p = PreviewPane()
        p.resize(500, 300)
        p.set_empty_hint('Drop a folder here')
        p.show()
        assert p.view.empty_hint == 'Drop a folder here' and not p.grab().isNull()


# ── 1:1 viewing ───────────────────────────────────────────────────────


class TestFullSize:
    def test_a_reduced_copy_is_measured_in_real_pixels(self, qapp):
        from src.qt_app.viewer import PreviewPane
        p = PreviewPane()
        p.resize(500, 400)
        p.show()
        p.load_slot(_jpeg(size=(100, 75)), 'final', 'x', logical_size=(400, 300))
        v = p.view
        assert v.image_size() == (400, 300) and v.item_scale() == 4.0
        p.one_to_one()
        assert v.scale_factor() == pytest.approx(1.0) and p.zoom_label.text() == '100%'
        assert v.needs_detail()                          # 1 screen pixel per real pixel = 4 per shown pixel

    def test_aspect_mismatch_ignores_the_logical_size(self, qapp):
        from src.qt_app.viewer import PreviewPane
        p = PreviewPane()
        p.load_slot(_jpeg(size=(100, 75)), 'final', 'x', logical_size=(400, 100))
        assert p.view.image_size() == (100, 75)

    def test_swapping_in_the_full_copy_keeps_zoom_and_position(self, qapp):
        from PySide6.QtCore import QPointF

        from src.qt_app.viewer import PreviewPane
        p = PreviewPane()
        p.resize(500, 400)
        p.show()
        p.load_slot(_jpeg(size=(100, 75)), 'final', 'x', logical_size=(400, 300))
        p.one_to_one()
        p.view.centerOn(QPointF(300, 200))
        centre = p.view.mapToScene(p.view.viewport().rect().center())
        full = np.zeros((300, 400, 3), np.uint8)
        full[..., 0] = 200
        p.load_array(full, (400, 300))
        assert p.view.item_scale() == 1.0 and not p.view.needs_detail()
        assert p.view.scale_factor() == pytest.approx(1.0)
        after = p.view.mapToScene(p.view.viewport().rect().center())
        assert abs(after.x() - centre.x()) < 1.5 and abs(after.y() - centre.y()) < 1.5

    def test_pixel_inspector_reports_real_pixels_and_linear_values(self, qapp):
        from PySide6.QtCore import QPointF

        from src.qt_app.viewer import PreviewPane
        p = PreviewPane()
        p.resize(500, 400)
        p.show()
        p.load_slot(_jpeg((200, 40, 40), size=(100, 75)), 'final', 'x', logical_size=(400, 300))
        p.linear_sampler = lambda x, y: {'rgb': (0.5, 0.25, 0.125), 'sigma': 12.34} if (x, y) == (200, 100) else None
        p.inspect_box.setChecked(True)
        p.view._emit_pixel(QPointF(p.view.mapFromScene(QPointF(200.5, 100.5))))
        text = p.pixel_label.text()
        assert text.startswith('x 200  y 100') and 'linear 0.5 0.25 0.125 (+12.3σ)' in text

    def test_zooming_past_the_pixels_asks_for_more_detail_once_it_settles(self, qapp):
        from src.qt_app.viewer import PreviewPane
        p = PreviewPane()
        p.resize(500, 400)
        p.show()
        asked = []
        p.detail_wanted.connect(lambda: asked.append(1))
        p.load_slot(_jpeg(size=(1000, 750)), 'final', 'x', logical_size=(4000, 3000))
        assert asked == []                               # fitted: the copy is already finer than the screen
        p.view._zoom_by(8.0)
        assert asked


class TestMainWindowFullSize:
    @pytest.fixture
    def stack(self, win):
        """A final image in UIEvents behind the window."""
        import numpy as np
        rng = np.random.default_rng(0)
        img = (0.02 + 0.003 * rng.standard_normal((300, 400, 3))).astype(np.float32)
        img[100:110, 150:160] += 0.5
        kw = {'stretch': 'ghs', 'ghs_b': 8.0, 'ghs_sp': 0.15, 'ghs_hp': 0.95, 'black_sigma': 0.0,
              'color': 'preserve', 'noise_scale': 1.0, 'anchor': None}
        with win.ui._lock:
            win.ui._final_src = {'rgb': img, 'starless': None, 'path': 'x.jpg', 'kw': dict(kw),
                                 'kw0': dict(kw), 'exports': ()}
        yield img
        with win.ui._lock:
            win.ui._final_src = None

    def test_zooming_in_loads_the_full_size_picture(self, win, stack, qapp):
        data = win.ui.live_stretch(max_dim=100)['jpeg']        # a 100 px copy of a 400 px image
        win._show_picture(data, 'final', 'Final')
        assert win.preview.view.image_size() == (400, 300) and win.preview.view.item_scale() > 3
        win.preview.view._zoom_by(8.0)
        assert _wait(qapp, lambda: win.preview.view.item_scale() == 1.0, 8.0)
        assert win.preview.view.image_size() == (400, 300)

    def test_a_picture_change_during_the_load_discards_it(self, win, stack, qapp):
        win._show_picture(win.ui.live_stretch(max_dim=100)['jpeg'], 'final', 'Final')
        gen = win._detail_gen
        win._show_picture(win.ui.live_stretch(max_dim=100)['jpeg'], 'live', 'Live')
        win._on_detail_ready(gen, np.zeros((300, 400, 3), np.uint8), (400, 300))     # the stale one
        assert win.preview.view.item_scale() > 3 and win._detail_done != gen

    def test_other_slots_are_not_treated_as_the_final_image(self, win, stack):
        win._show_picture(_jpeg(size=(100, 75)), 'quick-look', 'q')
        assert win.preview.view.image_size() == (100, 75)
        assert win._sample_linear(1, 1) is None

    def test_linear_sampling_follows_the_final_picture(self, win, stack):
        win._show_picture(win.ui.live_stretch(max_dim=100)['jpeg'], 'final', 'Final')
        s = win._sample_linear(155, 105)
        assert s is not None and s['sigma'] > 20


# ── frame-quality graph and the review ────────────────────────────────


def _frame_rows(n=12):
    return [{'name': f'light_{i:03d}.fits', 'score': 40.0 + i, 'fwhm': 3.0 + (i % 4) / 2,
             'snr': 10.0 + i / 3, 'stars': 20 + i, 'ok': i != 5, 'reason': 'blurry' if i == 5 else ''}
            for i in range(n)]


def _mouse_ev(kind, pos, modifiers=None, button=None):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QMouseEvent
    return QMouseEvent(kind, pos, pos, Qt.MouseButton.LeftButton,
                       button if button is not None else Qt.MouseButton.LeftButton,
                       modifiers or Qt.KeyboardModifier.NoModifier)


class TestFrameGraph:
    @pytest.fixture
    def graph(self, qapp):
        from src.qt_app.frame_graph import FrameGraph
        g = FrameGraph()
        g.resize(500, 200)
        g.show()
        g.set_rows(_frame_rows())
        return g

    def _click(self, g, i, modifiers=None):
        from PySide6.QtCore import QEvent, Qt
        pt = g._point(i)
        g.mousePressEvent(_mouse_ev(QEvent.Type.MouseButtonPress, pt, modifiers))
        g.mouseReleaseEvent(_mouse_ev(QEvent.Type.MouseButtonRelease, pt, modifiers, Qt.MouseButton.NoButton))

    def test_paints_with_and_without_frames(self, qapp, graph):
        from src.qt_app.frame_graph import FrameGraph
        assert not graph.grab().isNull()
        empty = FrameGraph()
        empty.resize(300, 150)
        assert not empty.grab().isNull()

    def test_a_click_picks_and_announces_the_frame(self, graph):
        got, sel = [], []
        graph.frame_clicked.connect(got.append)
        graph.selection_changed.connect(sel.append)
        self._click(graph, 3)
        assert got == ['light_003.fits'] and graph.selected == {'light_003.fits'} and sel[-1] == graph.selected

    def test_shift_click_adds_to_the_selection(self, graph):
        from PySide6.QtCore import Qt
        self._click(graph, 2)
        self._click(graph, 6, Qt.KeyboardModifier.ShiftModifier)
        assert graph.selected == {'light_002.fits', 'light_006.fits'}

    def test_dragging_a_box_picks_the_frames_inside(self, graph):
        from PySide6.QtCore import QEvent, QPointF, Qt
        a, b = graph._point(2), graph._point(7)
        top_left = QPointF(a.x() - 4, 0)
        bottom_right = QPointF(b.x() + 4, graph.height())
        graph.mousePressEvent(_mouse_ev(QEvent.Type.MouseButtonPress, top_left))
        graph.mouseMoveEvent(_mouse_ev(QEvent.Type.MouseMove, bottom_right))
        graph.mouseReleaseEvent(_mouse_ev(QEvent.Type.MouseButtonRelease, bottom_right, None, Qt.MouseButton.NoButton))
        assert graph.selected == {f'light_{i:03d}.fits' for i in range(2, 8)}

    def test_a_click_on_empty_space_clears_the_selection(self, graph):
        from PySide6.QtCore import QEvent, QPointF, Qt
        self._click(graph, 4)
        far = QPointF(graph._plot().left() + 2, graph._plot().top() + 2)
        graph.mousePressEvent(_mouse_ev(QEvent.Type.MouseButtonPress, far))
        graph.mouseReleaseEvent(_mouse_ev(QEvent.Type.MouseButtonRelease, far, None, Qt.MouseButton.NoButton))
        assert graph.selected == set()

    def test_the_measure_changes_the_vertical_range(self, graph):
        lo, hi = graph._range()
        graph.set_measure('fwhm')
        lo2, hi2 = graph._range()
        assert (lo, hi) != (lo2, hi2) and hi2 < 10
        graph.set_measure('stars')
        assert graph._range()[1] > 25

    def test_tooltip_text_names_the_reason(self, graph):
        graph.set_excluded({'light_002.fits'})
        assert 'Left out by you' in graph.describe(2)
        assert 'blurry' in graph.describe(5)
        assert 'light_000.fits' in graph.describe(0) and 'FWHM' in graph.describe(0)

    def test_rows_that_vanish_leave_the_selection(self, graph):
        graph.set_selected({'light_001.fits', 'gone.fits'})
        graph.set_rows(_frame_rows(3))
        assert graph.selected == {'light_001.fits'}


class TestFramesPanel:
    @pytest.fixture
    def panel(self, qapp):
        from src.qt_app.frame_graph import FramesPanel
        p = FramesPanel()
        p.resize(520, 360)
        p.show()
        p.set_rows(_frame_rows())
        return p

    def test_exclude_include_and_clear(self, panel):
        got = []
        panel.excluded_changed.connect(got.append)
        assert not panel.exclude_btn.isEnabled()
        panel.graph.set_selected({'light_001.fits', 'light_002.fits'})
        assert panel.exclude_btn.isEnabled() and not panel.include_btn.isEnabled()
        panel.exclude_btn.click()
        assert got[-1] == {'light_001.fits', 'light_002.fits'} and panel.excluded() == got[-1]
        assert panel.include_btn.isEnabled() and not panel.exclude_btn.isEnabled()
        assert '2 left out of 12' in panel.count_label.text()
        panel.graph.set_selected({'light_001.fits'})
        panel.include_btn.click()
        assert got[-1] == {'light_002.fits'}
        panel.clear_btn.click()
        assert got[-1] == set() and not panel.clear_btn.isEnabled()
        assert panel.count_label.text() == '12 frames'

    def test_setting_the_exclusions_from_outside_is_quiet(self, panel):
        got = []
        panel.excluded_changed.connect(got.append)
        panel.set_excluded({'light_004.fits'})
        assert got == [] and panel.excluded() == {'light_004.fits'} and panel.restack_btn.isEnabled()

    def test_review_mode_shows_the_banner(self, panel):
        assert not panel.banner.isVisibleTo(panel) and panel.restack_btn.isVisibleTo(panel)
        panel.set_review_mode(True)
        assert panel.banner.isVisibleTo(panel) and not panel.restack_btn.isVisibleTo(panel)
        got = []
        panel.continue_clicked.connect(lambda: got.append(1))
        panel.continue_btn.click()
        assert got == [1]

    def test_measure_choice_drives_the_graph(self, panel):
        panel.metric_combo.setCurrentIndex(1)
        panel.metric_combo.activated.emit(1)
        assert panel.graph.measure == 'fwhm'

    def test_setting_text_round_trip(self):
        from src.qt_app.frame_graph import names_to_setting, setting_to_names
        assert names_to_setting({'b.fits', 'a.fits'}) == 'a.fits,b.fits'
        assert setting_to_names('a.fits, b.fits') == {'a.fits', 'b.fits'}


class TestMainWindowReview:
    def test_the_excluded_setting_and_the_graph_stay_in_step(self, win):
        win.model.set('exclude_frames', 'a.fits,b.fits')
        assert win.frames_panel.excluded() == {'a.fits', 'b.fits'}
        win.frames_panel.graph.set_rows([{'name': 'a.fits', 'score': 1}, {'name': 'c.fits', 'score': 2}])
        win.frames_panel.graph.set_selected({'c.fits'})
        win.frames_panel.exclude_btn.click()
        assert win.model.get('exclude_frames') == 'a.fits,b.fits,c.fits'
        win.model.set('exclude_frames', '')
        assert win.frames_panel.excluded() == set()
        win.frames_panel.graph.set_rows([])

    def test_a_run_waiting_for_review_shows_the_graph_and_continues(self, win, qapp, monkeypatch):
        sent = []
        monkeypatch.setattr(win.rm, 'review_continue', lambda names: sent.append(set(names)))
        win.ui.run_started()
        win.ui.review_request(_frame_rows(6))
        win._refresh()
        assert win._reviewing and win.frames_panel.banner.isVisibleTo(win.frames_panel)
        assert win.tools.currentIndex() == win._tab_index['frames'] and win.tools.isVisible()
        assert len(win.frames_panel.graph.rows) == 6 and win.status_label.text() == 'Paused: review the frames'
        win.frames_panel.graph.set_selected({'light_002.fits'})
        win.frames_panel.exclude_btn.click()
        win.frames_panel.continue_btn.click()
        assert sent == [{'light_002.fits'}] and win.model.get('exclude_frames') == 'light_002.fits'
        win.ui.review_done()
        win._refresh()
        assert not win._reviewing and not win.frames_panel.banner.isVisibleTo(win.frames_panel)
        win.model.set('exclude_frames', '')
        win.ui.run_finished('cancelled')

    def test_a_click_on_the_graph_selects_the_table_row(self, win):
        rows = [('light_000.fits', 1, 1, 1, 1, True), ('light_001.fits', 2, 2, 2, 2, True)]
        win._fill_frames_table(rows)
        win._on_graph_frame('light_001.fits')
        assert win._pinned_frame == 'light_001.fits'
        assert win.frames_panel.graph.focus == 'light_001.fits'
        win._clear_frames_table()

    def test_start_passes_the_review_choice_to_the_run(self, win, monkeypatch):
        seen = {}
        monkeypatch.setattr(win.rm, 'start', lambda form: seen.update(form=form, review=win.rm.review_frames) or {'ok': True})
        win.review_box.setChecked(True)
        win.model.set('directory', '')
        win._on_start()
        assert seen['review'] is True
        from src.desktop_control import load_app_setting
        assert load_app_setting('review_frames') is True
        win.review_box.setChecked(False)
        win.start_btn.setEnabled(True)

    def test_exclusions_go_into_the_run_arguments(self, win):
        win.model.set('exclude_frames', 'light_001.fits,light_002.fits')
        from src.desktop_control import build_argv_from_form
        argv = build_argv_from_form(win.model.read_form())
        assert '--exclude-frames' in argv and 'light_001.fits,light_002.fits' in argv
        win.model.set('exclude_frames', '')


class TestReviewCountdown:
    @pytest.fixture
    def panel(self, qapp):
        from src.qt_app.frame_graph import FramesPanel
        p = FramesPanel()
        p.resize(520, 360)
        p.show()
        p.set_rows(_frame_rows())
        yield p
        p.stop_countdown()

    def test_the_default_is_five_seconds(self):
        from src.qt_app import frame_graph
        assert frame_graph.REVIEW_COUNTDOWN_S == 5

    def test_it_counts_down_and_continues_by_itself(self, panel):
        got = []
        panel.continue_clicked.connect(lambda: got.append(1))
        panel.set_review_mode(True, countdown=3)
        assert panel.counting() and 'Continuing in 3 s' in panel.banner_text.text()
        assert panel.stay_btn.isVisibleTo(panel)
        panel.tick()
        assert 'Continuing in 2 s' in panel.banner_text.text() and got == []
        panel.tick()
        assert 'Continuing in 1 s' in panel.banner_text.text()
        panel.tick()
        assert got == [1] and not panel.counting()

    def test_the_default_countdown_is_the_five_second_one(self, panel):
        panel.set_review_mode(True)
        assert 'Continuing in 5 s' in panel.banner_text.text()

    def test_touching_the_graph_keeps_it_paused(self, panel):
        from PySide6.QtCore import QEvent
        got = []
        panel.continue_clicked.connect(lambda: got.append(1))
        panel.set_review_mode(True, countdown=2)
        panel.graph.mousePressEvent(_mouse_ev(QEvent.Type.MouseButtonPress, panel.graph._point(2)))
        assert not panel.counting() and not panel.stay_btn.isVisibleTo(panel)
        assert 'Paused for your review. Pick the frames' in panel.banner_text.text()
        for _i in range(5):
            panel.tick()                      # a late tick after the stop must not continue the run
        assert got == []

    def test_stay_paused_stops_it_and_continue_still_works(self, panel):
        got = []
        panel.continue_clicked.connect(lambda: got.append(1))
        panel.set_review_mode(True, countdown=4)
        panel.stay_btn.click()
        assert not panel.counting() and got == []
        panel.continue_btn.click()
        assert got == [1]

    def test_picking_and_excluding_stop_the_countdown(self, panel):
        panel.set_review_mode(True, countdown=4)
        panel.graph.set_selected({'light_001.fits'})
        panel.exclude_btn.click()
        assert not panel.counting()
        panel.set_review_mode(True, countdown=4)
        panel.metric_combo.activated.emit(1)
        assert not panel.counting()

    def test_leaving_review_mode_stops_it(self, panel):
        panel.set_review_mode(True, countdown=4)
        panel.set_review_mode(False)
        assert not panel.counting() and not panel.banner.isVisibleTo(panel)

    def test_the_window_starts_it_and_a_table_click_stops_it(self, win, monkeypatch):
        sent = []
        monkeypatch.setattr(win.rm, 'review_continue', lambda names: sent.append(set(names)))
        win.ui.run_started()
        win.ui.review_request(_frame_rows(4))
        win._refresh()
        assert win.frames_panel.counting()
        win._fill_frames_table([('light_000.fits', 1, 1, 1, 1, True)])
        win._on_graph_frame('light_000.fits')
        assert not win.frames_panel.counting()
        win.ui.review_done()
        win._refresh()
        win._clear_frames_table()
        win.ui.run_finished('cancelled')

    def test_the_window_continues_when_the_countdown_ends(self, win, monkeypatch):
        sent = []
        monkeypatch.setattr(win.rm, 'review_continue', lambda names: sent.append(set(names)))
        win.ui.run_started()
        win.ui.review_request(_frame_rows(4))
        win._refresh()
        for _i in range(5):
            win.frames_panel.tick()
        assert sent == [set()]
        win.ui.review_done()
        win._refresh()
        win.ui.run_finished('cancelled')


# ── taskbar progress ──────────────────────────────────────────────────


class TestTaskbarProgress:
    @staticmethod
    def _fake(monkeypatch):
        """A TaskbarProgress that is 'available' and records the COM calls it would make."""
        from src.qt_app import taskbar
        tb = taskbar.TaskbarProgress.__new__(taskbar.TaskbarProgress)
        tb.hwnd, tb.available, tb._ptr, tb._vtable, tb._state, tb._value = 7, True, None, None, None, None
        calls = []
        monkeypatch.setattr(tb, '_call', lambda slot, types, *a: calls.append((slot, [getattr(x, 'value', x) for x in a])),
                            raising=False)
        return tb, calls, taskbar

    def test_no_op_without_a_taskbar(self, qapp):
        from src.qt_app.taskbar import TaskbarProgress
        tb = TaskbarProgress()
        tb.available = False
        tb.set_window(5)
        tb.set_value(1, 2)
        tb.clear()                                   # nothing raises

    def test_value_sets_normal_state_once_then_only_changes(self, qapp, monkeypatch):
        tb, calls, mod = self._fake(monkeypatch)
        tb.set_value(1, 10)
        tb.set_value(1, 10)
        tb.set_value(2, 10)
        assert calls == [(mod._SET_PROGRESS_STATE, [mod.NORMAL]),
                         (mod._SET_PROGRESS_VALUE, [1, 10]),
                         (mod._SET_PROGRESS_VALUE, [2, 10])]

    def test_unknown_total_pulses_and_value_is_clamped(self, qapp, monkeypatch):
        tb, calls, mod = self._fake(monkeypatch)
        tb.set_value(0, 0)
        assert calls == [(mod._SET_PROGRESS_STATE, [mod.INDETERMINATE])]
        tb.set_value(99, 10)
        assert calls[-1] == (mod._SET_PROGRESS_VALUE, [10, 10])

    def test_clear_forgets_the_value(self, qapp, monkeypatch):
        tb, calls, mod = self._fake(monkeypatch)
        tb.set_value(3, 10)
        tb.clear()
        tb.set_value(3, 10)
        assert [c for c in calls if c[0] == mod._SET_PROGRESS_VALUE] == [(mod._SET_PROGRESS_VALUE, [3, 10])] * 2

    def test_a_new_window_resets_the_memo(self, qapp, monkeypatch):
        tb, calls, mod = self._fake(monkeypatch)
        tb.set_value(3, 10)
        tb.set_window(8)
        tb.set_value(3, 10)
        assert len(calls) == 4

    @pytest.mark.skipif(os.name != 'nt', reason='ITaskbarList3 is Windows-only')
    def test_the_real_com_object_accepts_the_calls(self, qapp):
        from PySide6.QtWidgets import QWidget

        from src.qt_app.taskbar import TaskbarProgress
        w = QWidget()
        w.show()
        tb = TaskbarProgress(int(w.winId()))
        assert tb.available
        tb.set_value(3, 10)
        tb.set_value(0, 0)
        tb.clear()
        w.close()

    def test_the_window_shows_run_progress(self, win, monkeypatch):
        from src.qt_app import taskbar
        seen = []
        monkeypatch.setattr(win.taskbar, 'available', True)
        monkeypatch.setattr(win.taskbar, 'set_window', lambda h: None)
        monkeypatch.setattr(win.taskbar, 'set_value', lambda d, t: seen.append(('v', d, t)))
        monkeypatch.setattr(win.taskbar, 'set_state', lambda s: seen.append(('s', s)))
        monkeypatch.setattr(win.taskbar, 'clear', lambda: seen.append('clear'))
        win._update_taskbar({'progress': {'done': 4, 'total': 9}, 'run_status': 'running'}, True)
        win._reviewing = True
        win._update_taskbar({'progress': {'done': 4, 'total': 9}, 'run_status': 'running'}, True)
        win._reviewing = False
        win._update_taskbar({'progress': {}, 'run_status': 'ok'}, False)
        assert seen == [('v', 4, 9), ('s', taskbar.PAUSED), 'clear']


# ── ported from the tkinter window's tests (frames table, update check, languages) ──


def _frames_rgb(seed=0):
    return np.random.default_rng(seed).normal(100.0, 5.0, (64, 96, 3)).astype(np.float32)


def _publish_frames(ui, names, thumbs=None):
    for n in names:
        ui.frame_metrics(n, {'score': 50.0, 'snr': 9.0, 'star_count': 100, 'fwhm': 3.0})
    for n in (names if thumbs is None else thumbs):
        ui.frame_preview(n, _frames_rgb(seed=len(n)))


def _table_row(win, name):
    for r in range(win.frames_table.rowCount()):
        if win.frames_table.item(r, 0).text() == name:
            return r
    raise AssertionError(f'{name} is not in the table')


class TestFramesTablePicking:
    def test_picking_a_row_shows_that_frame(self, win):
        win.ui.run_started()
        _publish_frames(win.ui, ['light_001.fits', 'light_002.fits'])
        win._refresh()
        win.frames_table.selectRow(_table_row(win, 'light_002.fits'))
        assert win.preview.current_slug == 'frame:light_002.fits'
        win.frames_table.selectRow(_table_row(win, 'light_001.fits'))
        assert win.preview.current_slug == 'frame:light_001.fits'
        win.ui.run_finished('cancelled')

    def test_the_picked_frame_survives_later_updates(self, win):
        win.ui.run_started()
        _publish_frames(win.ui, ['light_001.fits'])
        win._refresh()
        win.frames_table.selectRow(_table_row(win, 'light_001.fits'))
        _publish_frames(win.ui, ['light_002.fits', 'light_003.fits'])
        win.ui.preview(_frames_rgb(), 'Aligned stack', slot='aligned', min_interval=0.0)
        win._refresh()
        assert win.preview.current_slug == 'frame:light_001.fits'       # no jump to the milestone
        assert [i.row() for i in win.frames_table.selectedIndexes()][:1] == [_table_row(win, 'light_001.fits')]
        win.ui.run_finished('cancelled')

    def test_a_milestone_pick_goes_back_to_following_the_run(self, win):
        win.ui.run_started()
        _publish_frames(win.ui, ['light_001.fits'])
        win.ui.preview(_frames_rgb(), 'Aligned stack', slot='aligned', min_interval=0.0)
        win._refresh()
        win.frames_table.selectRow(_table_row(win, 'light_001.fits'))
        assert win._pinned_frame == 'light_001.fits'
        win.view_combo.setCurrentIndex(win.view_combo.count() - 1)
        win._on_view_selected()                          # what picking in the box does
        assert win._pinned_frame is None and not win.frames_table.selectedItems()
        win.ui.preview(_frames_rgb(seed=9), 'Final', slot='final', min_interval=0.0)
        win._refresh()
        assert win.preview.current_slug == 'final'
        win.ui.run_finished('cancelled')

    def test_a_frame_without_a_thumbnail_says_so(self, win):
        win.ui.run_started()
        _publish_frames(win.ui, ['light_001.fits', 'rejected.fits'], thumbs=['light_001.fits'])
        win._refresh()
        win.frames_table.selectRow(_table_row(win, 'light_001.fits'))
        shown = win.preview.current_slug
        win.frames_table.selectRow(_table_row(win, 'rejected.fits'))
        assert 'rejected.fits' in win.preview.caption() and 'no preview' in win.preview.caption()
        assert win.preview.current_slug == shown
        win.ui.run_finished('cancelled')


class TestUpdateCheckWiring:
    def test_no_check_started_when_disabled(self, qapp, monkeypatch):
        import threading
        from unittest import mock

        from src.qt_app.main_window import MainWindow
        monkeypatch.setattr('src.utils.should_check_for_update', lambda: False)
        started = []
        monkeypatch.setattr(threading, 'Thread', lambda *a, **k: started.append(1) or mock.MagicMock())
        w = MainWindow()
        try:
            assert not started and w.update_label.text() == ''
        finally:
            w._timer.stop()
            w.ui.detach()

    def test_label_updates_when_a_newer_version_is_found(self, qapp, monkeypatch):
        import threading

        from src.qt_app.main_window import MainWindow
        monkeypatch.setattr('src.utils.should_check_for_update', lambda: True)
        real_thread, started = threading.Thread, []

        def track(*args, **kwargs):
            t = real_thread(*args, **kwargs)
            started.append(t)
            return t
        monkeypatch.setattr(threading, 'Thread', track)
        monkeypatch.setattr('src.net_query.check_for_update',
                            lambda *a, **k: {'version': '9.9.9', 'url': 'https://example.invalid/new'})
        w = MainWindow()
        try:
            started[0].join(timeout=2)
            assert not started[0].is_alive()
            for _i in range(50):                       # the signal is queued to the GUI thread
                qapp.processEvents()
                if '9.9.9' in w.update_label.text():
                    break
            assert '9.9.9' in w.update_label.text()
            assert w._update_url == 'https://example.invalid/new'
        finally:
            w._timer.stop()
            w.ui.detach()


@pytest.mark.parametrize('code', [c for c in __import__('src.i18n', fromlist=['LANGUAGES']).LANGUAGES if c != 'en'])
def test_setup_form_builds_in_every_language(code, qapp):
    from src import i18n
    from src.qt_app.setup_form import SetupForm
    i18n.set_language(code)
    try:
        form = SetupForm()
        form.resize(600, 700)
        form.show()
        assert form.fields
        # Translated text never reaches what Start submits.
        assert form.model.read_form() == {}
        form.close()
    finally:
        i18n.set_language('en')


def test_merge_previous_stacks_one_click(qapp, tmp_path, monkeypatch):
    from src.qt_app.form_model import FormModel
    from src.qt_app.setup_form import SetupForm
    m = FormModel()
    stack = tmp_path / 'M101_stacked.fits'
    stack.write_bytes(b'')
    monkeypatch.setattr(m, '_find_previous', lambda d, n: [
        {'path': str(stack), 'frames': 40, 'date': '2026-01-01'}])
    m.previous_stacks = m._find_previous('x', 'M101')
    assert not m.merge_previous()
    m.set_merge_previous(True)
    assert m.get('merge') == str(stack) and m.merge_previous()
    assert m.read_form()['merge'] == str(stack)
    m.set_merge_previous(False)
    assert m.get('merge') == ''
    # a hand-typed merge list is not cleared by unticking
    m.set('merge', 'other.fits')
    m.set_merge_previous(False)
    assert m.get('merge') == 'other.fits'
    form = SetupForm(m)
    form._refresh_merge_box()
    assert not form.merge_box.isHidden()
    assert '40' in form.merge_box.text()
