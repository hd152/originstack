"""Choosing between pooling sessions and stacking them separately.

Pooling every subfolder's lights into one stack used to be unconditional, on
the reasoning that a multi-night directory is the common case. That is exactly
the case where pooling is most expensive: on an alt-az mount the field rotates
as the target tracks, so sessions at different hour angles are rotated
relative to each other, and one common crop across all of them discards the
corners -- 26.8 degrees and 42% of the frame, measured across five real Lagoon
sessions.

The mode is now predicted from metadata alone, which is free and happens
before any stacking. These tests cover the decision, not the stacking.
"""

import io
import json
import os
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from unittest import mock

import numpy as np

from src.cli import _ROTATION_SPLIT_THRESHOLD_DEG, _predict_rotation_spread, _want_combine_sessions

try:
    from astropy.io import fits
    HAS_ASTROPY = True
except Exception:
    HAS_ASTROPY = False


def _write_session(root, name, times, ra_deg=271.0, dec_deg=-24.36,
                   lat=33.83, lon=-117.79, with_gps=True, with_wcs=True,
                   include_flat=False, tz='-0700'):
    """A minimal session directory: info.json plus dated light frames."""
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)

    info = {'StackedInfo': {}}
    si = info['StackedInfo']
    si['dateTime'] = times[0] + '-0700'
    if with_wcs:
        # has_wcs needs the field size and image dimensions too, not just the
        # pointing -- it is "can a WCS be built", not "is there an RA/Dec".
        si['celestial'] = {'first': np.radians(ra_deg),
                           'second': np.radians(dec_deg)}
        si['fovX'] = np.radians(1.27)
        si['fovY'] = np.radians(0.85)
        si['imageWidth'] = 3056
        si['imageHeight'] = 2048
    if with_gps:
        si['gps'] = {'latitude': lat, 'longitude': lon, 'altitude': 0.0}
    with open(os.path.join(d, 'info.json'), 'w', encoding='utf-8') as fh:
        json.dump(info, fh)

    for i, t in enumerate(times):
        hdu = fits.PrimaryHDU(data=np.ones((8, 8), dtype=np.float32))
        hdu.header['DATE-OBS'] = t
        hdu.header['TIMEZONE'] = tz
        hdu.header['EXPTIME'] = 30.0
        hdu.writeto(os.path.join(d, f'Light{i:04d}.fits'), overwrite=True)

    if include_flat:
        # A real session carries calibration frames beside the lights, and a
        # flat's DATE-OBS is '0-00-00T00:00:00'. Globbing *.fits sweeps it up
        # and poisons the time span, so the classifier must exclude it.
        flat = fits.PrimaryHDU(data=np.ones((8, 8), dtype=np.float32))
        flat.header['DATE-OBS'] = '0-00-00T00:00:00'
        flat.header['IMAGETYP'] = 'flat'
        flat.writeto(os.path.join(d, 'flat.fits'), overwrite=True)
    return d


def _args(**kw):
    base = dict(mosaic=False, hierarchical=False, combine_sessions=False,
                _explicit_cli_dests=set())
    base.update(kw)
    return Namespace(**base)


@unittest.skipUnless(HAS_ASTROPY, "astropy required")
class TestPredictRotationSpread(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_one_short_session_spans_little_rotation(self):
        d = _write_session(self.root, 's1',
                           ['2026-08-31T20:40:35', '2026-08-31T20:50:35'])
        spread = _predict_rotation_spread([d])
        self.assertIsNotNone(spread)
        self.assertLess(spread, _ROTATION_SPLIT_THRESHOLD_DEG)

    def test_sessions_hours_apart_span_a_large_rotation(self):
        a = _write_session(self.root, 'a',
                           ['2026-08-31T20:40:35', '2026-08-31T21:10:35'])
        b = _write_session(self.root, 'b',
                           ['2026-08-31T23:40:35', '2026-09-01T00:10:35'])
        spread = _predict_rotation_spread([a, b])
        self.assertIsNotNone(spread)
        self.assertGreater(spread, _ROTATION_SPLIT_THRESHOLD_DEG)

    def test_calibration_frames_do_not_poison_the_time_span(self):
        """A flat's DATE-OBS is '0-00-00T00:00:00'.

        Reading it as a light-frame timestamp makes the parallactic angle
        unparseable and the whole prediction return None, which silently
        falls back to pooling -- the behaviour this is meant to replace.
        """
        d = _write_session(self.root, 'withcal',
                           ['2026-08-31T20:40:35', '2026-08-31T20:50:35'],
                           include_flat=True)
        spread = _predict_rotation_spread([d])
        self.assertIsNotNone(spread, "calibration frames must be excluded")
        self.assertLess(spread, _ROTATION_SPLIT_THRESHOLD_DEG)

    def test_returns_none_without_gps(self):
        d = _write_session(self.root, 'nogps',
                           ['2026-08-31T20:40:35'], with_gps=False)
        self.assertIsNone(_predict_rotation_spread([d]))

    def test_returns_none_without_a_session_solve(self):
        d = _write_session(self.root, 'nowcs',
                           ['2026-08-31T20:40:35'], with_wcs=False)
        self.assertIsNone(_predict_rotation_spread([d]))

    def test_returns_none_for_a_directory_with_no_lights(self):
        d = os.path.join(self.root, 'empty')
        os.makedirs(d, exist_ok=True)
        self.assertIsNone(_predict_rotation_spread([d]))

    def test_works_without_astropys_iers_tables(self):
        """Regression: the packaged exe strips astropy_iers_data, so
        Time.sidereal_time("apparent") raised FileNotFoundError('finals2000A.all')
        there -- swallowed by a fail-soft except, so the prediction returned
        None and the exe silently always pooled while a source checkout
        split. Same directory, two different stacks. Sidereal time is now
        closed-form, so a broken astropy must not matter."""
        a = _write_session(self.root, 'a',
                           ['2026-08-31T20:40:35', '2026-08-31T21:10:35'])
        b = _write_session(self.root, 'b',
                           ['2026-08-31T23:40:35', '2026-09-01T00:10:35'])
        expected = _predict_rotation_spread([a, b])
        self.assertIsNotNone(expected)

        from astropy.time import Time
        with mock.patch.object(Time, 'sidereal_time',
                               side_effect=FileNotFoundError('finals2000A.all')):
            frozen = _predict_rotation_spread([a, b])

        self.assertIsNotNone(frozen, "must not depend on the IERS tables")
        self.assertAlmostEqual(frozen, expected, places=6)

    def test_a_timezone_that_is_not_an_offset_is_declined_not_misparsed(self):
        """'PDT' appended to DATE-OBS is unparseable; the prediction used to
        vanish silently. Now it is declined, and the reason is logged."""
        d = _write_session(self.root, 'badtz',
                           ['2026-08-31T20:40:35', '2026-08-31T20:50:35'], tz='PDT')
        with self.assertLogs('originstack', level='DEBUG') as logs:
            self.assertIsNone(_predict_rotation_spread([d]))
        self.assertTrue(any('TIMEZONE' in m for m in logs.output))

    def test_a_colon_separated_offset_is_accepted(self):
        d = _write_session(self.root, 'colon',
                           ['2026-08-31T20:40:35', '2026-08-31T20:50:35'], tz='-07:00')
        self.assertIsNotNone(_predict_rotation_spread([d]))

    def test_a_missing_timezone_is_read_as_utc(self):
        d = _write_session(self.root, 'notz',
                           ['2026-08-31T20:40:35', '2026-08-31T20:50:35'], tz='')
        self.assertIsNotNone(_predict_rotation_spread([d]))


@unittest.skipUnless(HAS_ASTROPY, "astropy required")
class TestModeSelection(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.close = [
            _write_session(self.root, 'c1', ['2026-08-31T20:40:35',
                                             '2026-08-31T20:45:35']),
            _write_session(self.root, 'c2', ['2026-08-31T20:50:35',
                                             '2026-08-31T20:55:35']),
        ]
        self.far = [
            _write_session(self.root, 'f1', ['2026-08-31T20:40:35',
                                             '2026-08-31T21:10:35']),
            _write_session(self.root, 'f2', ['2026-08-31T23:40:35',
                                             '2026-09-01T00:10:35']),
        ]

    def tearDown(self):
        self._tmp.cleanup()

    def test_rotated_sessions_are_stacked_separately(self):
        self.assertFalse(_want_combine_sessions(_args(), self.far))

    def test_closely_spaced_sessions_are_pooled(self):
        self.assertTrue(_want_combine_sessions(_args(), self.close))

    def test_explicit_combine_sessions_wins_over_the_prediction(self):
        args = _args(combine_sessions=True,
                     _explicit_cli_dests={'combine_sessions'})
        self.assertTrue(_want_combine_sessions(args, self.far))

    def test_explicit_hierarchical_wins_on_unrotated_data(self):
        self.assertFalse(_want_combine_sessions(_args(hierarchical=True),
                                                self.close))

    def test_mosaic_always_needs_separate_panels(self):
        self.assertFalse(_want_combine_sessions(_args(mosaic=True), self.close))

    def test_unreadable_metadata_keeps_the_previous_pooling_behaviour(self):
        """An unreadable session must not silently change how stacks build."""
        bare = os.path.join(self.root, 'bare')
        os.makedirs(bare, exist_ok=True)
        self.assertTrue(_want_combine_sessions(_args(), [bare, bare]))

    def test_a_single_subfolder_is_pooled(self):
        self.assertTrue(_want_combine_sessions(_args(), self.far[:1]))

    def test_a_saved_config_that_asks_for_pooling_is_honoured(self):
        """Regression: a --config file recording combine_sessions = true was
        silently overridden by the rotation heuristic on replay, because the
        explicit-flag check only consulted _explicit_cli_dests, which is built
        from argv alone. The value is what matters, wherever it came from."""
        args = _args(combine_sessions=True)          # no _explicit_cli_dests entry
        self.assertTrue(_want_combine_sessions(args, self.far))

    def test_the_default_false_does_not_pin_a_replayed_run_to_splitting(self):
        """A saved config also records the *default* False. Only True is a
        choice; False must still leave the decision to the data."""
        self.assertTrue(_want_combine_sessions(_args(combine_sessions=False), self.close))
        self.assertFalse(_want_combine_sessions(_args(combine_sessions=False), self.far))

    def test_a_failed_prediction_says_so(self):
        """Pooling on unreadable metadata is right, but doing it silently is
        indistinguishable from the rotation check being broken."""
        bare = os.path.join(self.root, 'bare')
        os.makedirs(bare, exist_ok=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertTrue(_want_combine_sessions(_args(), [bare, bare]))
        self.assertIn('Could not predict field rotation', buf.getvalue())
        self.assertIn('--hierarchical', buf.getvalue(),
                      "the message should say how to override it")

    def test_a_successful_prediction_states_the_measured_spread(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            _want_combine_sessions(_args(), self.far)
        self.assertIn('deg of field rotation', buf.getvalue())


if __name__ == '__main__':
    unittest.main()
