"""The desktop app's RECENT FRAMES table: picking a row shows that frame in the preview.

The per-frame thumbnail strip is gone; every processed frame gets a thumbnail (made from a
decimated copy, so it costs a fraction of stretching the full frame) and the table rows
look theirs up by name.
"""
import io

import numpy as np
import pytest

from src.ui_events import UIEvents, _decimate


def _rgb(h=64, w=96, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(100.0, 5.0, (h, w, 3))).astype(np.float32)


class TestThumbnails:
    def test_decimate_averages_blocks_and_fits_max_dim(self):
        a = np.zeros((20, 30, 3), dtype=np.float32)
        a[:10, :10] = 4.0
        small = _decimate(a, 10)             # factor 3
        assert small.shape == (6, 10, 3)
        assert small.dtype == np.float32
        assert small[0, 0, 0] == pytest.approx(4.0)       # a block inside the bright square
        assert small[-1, -1, 0] == 0.0

    def test_decimate_leaves_small_images_alone(self):
        a = _rgb(40, 50)
        assert _decimate(a, 64) is a or np.array_equal(_decimate(a, 64), a)

    def test_frame_jpeg_for_returns_the_newest_thumbnail_of_that_name(self):
        ui = UIEvents()
        ui.attach()
        ui.frame_preview('a.fits', _rgb(seed=1))
        ui.frame_preview('b.fits', _rgb(seed=2))
        first = ui.frame_jpeg_for('a.fits')
        ui.frame_preview('a.fits', _rgb(seed=3))          # same name again: the newer one wins
        assert first[:2] == b'\xff\xd8'
        assert ui.frame_jpeg_for('a.fits') != first
        assert ui.frame_jpeg_for('b.fits')[:2] == b'\xff\xd8'
        assert ui.frame_jpeg_for('missing.fits') is None

    def test_thumbnail_of_a_large_frame_is_at_most_max_dim(self):
        from PIL import Image
        ui = UIEvents()
        ui.attach()
        ui.frame_preview('big.fits', _rgb(1200, 1800), max_dim=512)
        img = Image.open(io.BytesIO(ui.frame_jpeg_for('big.fits')))
        assert max(img.size) <= 512

    def test_default_is_a_thumbnail_per_frame(self):
        from src.frame_processor import _publish_frame_thumb

        class _Args:
            pass

        class _Wv:
            active = True

            def __init__(self):
                self.names = []

            def frame_preview(self, name, rgb, args=None):
                self.names.append(name)

        wv, counter = _Wv(), [0]
        for i in range(4):
            _publish_frame_thumb(wv, _Args(), f'f{i}', _rgb(8, 8), counter)
        assert wv.names == ['f0', 'f1', 'f2', 'f3']       # no `ui_frame_every` set: every frame

    def test_cli_default(self):
        from src.cli import parse_args
        assert parse_args(['-d', 'x']).ui_frame_every == 1


def _app(monkeypatch, root):
    from src import desktop_app
    monkeypatch.setattr('src.utils.should_check_for_update', lambda: False)
    return desktop_app.App(root)


def _row(app, name):
    for item in app.frames_tree.get_children():
        if app.frames_tree.set(item, 'frame') == name:
            return item
    raise AssertionError(f'{name} is not in the table')


@pytest.fixture
def app(tk_root, monkeypatch):
    app = _app(monkeypatch, tk_root)
    ui = UIEvents()
    ui.attach()
    app.ui = ui
    yield app, ui, tk_root
    for after in (getattr(app, '_pp_timer', None),):
        if after:
            tk_root.after_cancel(after)


def _publish(ui, names, thumbs=None):
    for n in names:
        ui.frame_metrics(n, {'score': 50.0, 'snr': 9.0, 'star_count': 100, 'fwhm': 3.0})
    for n in (names if thumbs is None else thumbs):
        ui.frame_preview(n, _rgb(seed=len(n)))


def test_the_thumbnail_strip_is_gone(app):
    a, _, _ = app
    assert not hasattr(a, 'frame_strip')


def test_picking_a_row_shows_that_frame(app):
    a, ui, root = app
    _publish(ui, ['light_001.fits', 'light_002.fits'])
    a._refresh()
    a.frames_tree.selection_set(_row(a, 'light_002.fits'))
    root.update()
    assert a.preview.current_slug == 'frame:light_002.fits'
    assert a.preview.caption_var.get() == 'light_002.fits'
    a.frames_tree.selection_set(_row(a, 'light_001.fits'))
    root.update()
    assert a.preview.current_slug == 'frame:light_001.fits'


def test_the_picked_frame_survives_later_updates(app):
    a, ui, root = app
    _publish(ui, ['light_001.fits'])
    a._refresh()
    a.frames_tree.selection_set(_row(a, 'light_001.fits'))
    root.update()
    # new frames arrive and a new milestone is published: the preview must not jump away
    _publish(ui, ['light_002.fits', 'light_003.fits'])
    ui.preview(_rgb(), 'Aligned stack', slot='aligned', min_interval=0.0)
    a._refresh()
    root.update()
    assert a.preview.current_slug == 'frame:light_001.fits'
    assert a.frames_tree.selection() == (_row(a, 'light_001.fits'),)      # still highlighted


def test_picking_a_milestone_goes_back_to_following_the_run(app):
    a, ui, root = app
    _publish(ui, ['light_001.fits'])
    ui.preview(_rgb(), 'Aligned stack', slot='aligned', min_interval=0.0)
    a._refresh()
    a.frames_tree.selection_set(_row(a, 'light_001.fits'))
    root.update()
    assert a._pinned_frame == 'light_001.fits'
    a.view_combo.set(a.view_combo['values'][-1])
    a._on_view_selected()
    root.update()
    assert a._pinned_frame is None
    assert a.frames_tree.selection() == ()
    ui.preview(_rgb(seed=9), 'Final', slot='final', min_interval=0.0)
    a._refresh()
    assert a.preview.current_slug == 'final'                      # following the run again


def test_a_frame_without_a_thumbnail_says_so(app):
    a, ui, root = app
    _publish(ui, ['light_001.fits', 'rejected.fits'], thumbs=['light_001.fits'])
    a._refresh()
    a.frames_tree.selection_set(_row(a, 'light_001.fits'))
    root.update()
    shown = a.preview.current_slug
    a.frames_tree.selection_set(_row(a, 'rejected.fits'))
    root.update()
    assert 'rejected.fits' in a.preview.caption_var.get()
    assert 'no preview' in a.preview.caption_var.get()
    assert a.preview.current_slug == shown                         # the image itself is unchanged


def test_unchanged_rows_do_not_rebuild_the_table(app):
    a, ui, root = app
    _publish(ui, ['light_001.fits'])
    a._refresh()
    item = _row(a, 'light_001.fits')
    ui.log('something happened')                                   # a state change, no new frame
    a._refresh()
    assert _row(a, 'light_001.fits') == item
