"""Leaving frames out of a stack: ``--exclude-frames``, the pause between quality analysis and
registration (``apply_frame_review`` / ``RunManager._review_hook``), and the review state the app
reads from ``UIEvents``."""
import os
import sys
import threading
import time

import pytest

from src.frame_processor import apply_frame_exclusions, apply_frame_review, parse_frame_names
from src.models import FrameInfo, ProcessingStats, RunCancelled


def _lights(n=5):
    return [FrameInfo(path=os.path.join('data', f'light_{i:03d}.fits'), type='light', header={},
                      metrics={'score': 50.0 + i, 'snr': 10.0, 'star_count': 20, 'fwhm': 3.0 + i / 10})
            for i in range(n)]


class TestParse:
    def test_separators_and_blanks(self):
        assert parse_frame_names('a.fits, b.fits;c.fits\nd.fits,,  ') == {'a.fits', 'b.fits', 'c.fits', 'd.fits'}
        assert parse_frame_names(['a', 'b']) == {'a', 'b'}
        assert parse_frame_names('') == set() and parse_frame_names(None) == set()

    def test_exclusions_by_file_name(self, capsys):
        lights = _lights()
        kept = apply_frame_exclusions(lights, 'light_001.fits,light_003.fits,nothing.fits')
        assert [os.path.basename(f.path) for f in kept] == ['light_000.fits', 'light_002.fits', 'light_004.fits']
        assert 'Leaving out 2 frame(s)' in capsys.readouterr().out

    def test_no_names_is_the_same_list(self):
        lights = _lights()
        assert apply_frame_exclusions(lights, '') is lights


class TestReview:
    def test_dropped_frames_become_rejected_and_the_stats_follow(self):
        lights = _lights()
        stats = ProcessingStats()
        reasons = {}
        final = list(lights)
        got = apply_frame_review(final, lights, reasons, stats,
                                 lambda f, l, r: {'light_001.fits', 'light_004.fits'})
        assert [os.path.basename(f.path) for f in got] == ['light_000.fits', 'light_002.fits', 'light_003.fits']
        assert reasons[lights[1].path] == 'excluded by you (frame review)'
        assert not lights[1].accepted and lights[0].accepted
        assert (stats.accepted_frames, stats.rejected_frames) == (3, 2)

    def test_no_hook_or_nothing_dropped_changes_nothing(self):
        lights, stats = _lights(), ProcessingStats()
        assert apply_frame_review(lights, lights, {}, stats, None) is lights
        assert apply_frame_review(lights, lights, {}, stats, lambda *a: set()) is lights

    def test_the_hook_sees_every_light_and_the_reasons(self):
        lights = _lights()
        seen = {}

        def hook(final, all_lights, reasons):
            seen.update(n_final=len(final), n_all=len(all_lights), reasons=dict(reasons))
            return set()
        apply_frame_review(lights[:3], lights, {'x': 'why'}, ProcessingStats(), hook)
        assert seen == {'n_final': 3, 'n_all': 5, 'reasons': {'x': 'why'}}


class TestRunManagerReview:
    @pytest.fixture
    def rm(self):
        from src.desktop_control import RunManager
        from src.ui_events import get_ui_events
        ui = get_ui_events()
        ui.attach()
        ui.run_started()
        yield RunManager()
        ui.detach()

    def _hook_in_thread(self, rm, lights):
        out = {}

        def run():
            try:
                out['dropped'] = rm._review_hook(list(lights[:4]), lights, {lights[4].path: 'blurry'})
            except RunCancelled as e:
                out['cancelled'] = str(e)
        t = threading.Thread(target=run)
        t.start()
        return t, out

    @staticmethod
    def _wait(cond, seconds=3.0):
        end = time.time() + seconds
        while time.time() < end and not cond():
            time.sleep(0.01)
        return cond()

    def test_waits_publishes_the_frames_and_returns_the_dropped_names(self, rm):
        from src.ui_events import get_ui_events
        ui = get_ui_events()
        lights = _lights()
        t, out = self._hook_in_thread(rm, lights)
        assert self._wait(lambda: ui.snapshot()['review']['waiting'])
        rows = ui.snapshot()['review']['frames']
        assert [r['name'] for r in rows] == [f'light_{i:03d}.fits' for i in range(5)]
        assert rows[0]['ok'] and not rows[4]['ok'] and rows[4]['reason'] == 'blurry'
        assert 'dropped' not in out                           # still waiting
        rm.review_continue({'light_002.fits'})
        t.join(3)
        assert out['dropped'] == {'light_002.fits'}
        assert not ui.snapshot()['review']['waiting']

    def test_a_cancel_while_waiting_stops_the_run(self, rm):
        t, out = self._hook_in_thread(rm, _lights())
        time.sleep(0.1)
        rm._cancel_event.set()
        t.join(3)
        assert 'frame review' in out['cancelled']

    def test_the_hook_is_only_set_when_asked(self, rm):
        rm.review_frames = False
        assert rm.review_frames is False


class TestEndToEnd:
    """Two real runs on the synthetic dataset (6 small lights), a few seconds each."""

    @pytest.fixture
    def dataset(self, tmp_path):
        """What tools/create_synthetic.py writes (it only writes into the repo): 3 darks, 2 flats
        and 6 lights with small dithers."""
        import numpy as np
        from astropy.io import fits
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
        from create_synthetic import make_star_image
        d = tmp_path / 'synthetic_data'
        d.mkdir()
        for i in range(3):
            fits.writeto(str(d / f'dark_{i:03d}.fit'),
                         np.random.default_rng(i).normal(50.0, 1.5, (256, 256)).astype(np.float32))
        yy, xx = np.indices((256, 256))
        r = np.sqrt((yy - 128) ** 2 + (xx - 128) ** 2)
        for i in range(2):
            fits.writeto(str(d / f'flat_{i:03d}.fit'), (1000.0 * (1.0 - 0.0003 * r)).astype(np.float32))
        for i, (dy, dx) in enumerate([(0, 0), (2, -2), (-3, 3), (1, 4), (4, 1), (-2, -3)]):
            img = np.roll(np.roll(make_star_image((256, 256), n_stars=30, seed=42 + i, bg=1000.0), dy, 0), dx, 1)
            hdr = fits.Header()
            hdr['BAYERPAT'] = 'RGGB'
            fits.writeto(str(d / f'light_{i:03d}.fit'), img.astype(np.float32), header=hdr)
        return str(d)

    @staticmethod
    def _stack(directory, out, extra=(), hook=None):
        from src.cli import apply_post_parse_setup, parse_args, process_directory
        args = parse_args(['-d', directory, '-o', out, '--stack-method', 'median', '-j', '1',
                           *extra])
        apply_post_parse_setup(args)
        args._review_hook = hook
        process_directory(args.directory, args.output, args)

    def test_exclude_frames_leaves_the_named_light_out(self, dataset, tmp_path, capsys):
        self._stack(dataset, str(tmp_path / 'a.fits'), ['--exclude-frames', 'light_000.fit'])
        out = capsys.readouterr().out
        assert 'Leaving out 1 frame(s) named in --exclude-frames' in out
        assert '5 lights' in out and (tmp_path / 'a.fits').exists()

    def test_the_review_pause_drops_frames_before_registration(self, dataset, tmp_path, capsys):
        calls = []

        def hook(final, lights, reasons):
            calls.append((len(final), len(lights)))
            return {'light_001.fit'}
        self._stack(dataset, str(tmp_path / 'b.fits'), hook=hook)
        out = capsys.readouterr().out
        assert len(calls) == 1 and calls[0][1] == 6
        assert 'Frame review: 1 frame(s) left out by you' in out and (tmp_path / 'b.fits').exists()
