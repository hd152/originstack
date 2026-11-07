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
