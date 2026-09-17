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

import json
import os
import unittest
from argparse import Namespace

import numpy as np

from src.cli import _ROTATION_SPLIT_THRESHOLD_DEG, _predict_rotation_spread, _want_combine_sessions

try:
    from astropy.io import fits
    HAS_ASTROPY = True
except Exception:
    HAS_ASTROPY = False


def _write_session(root, name, times, ra_deg=271.0, dec_deg=-24.36,
                   lat=33.83, lon=-117.79, with_gps=True, with_wcs=True,
                   include_flat=False):
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
        hdu.header['TIMEZONE'] = '-0700'
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


if __name__ == '__main__':
    unittest.main()
