"""Tests for the collection quality sweep (--quality-sweep)."""
import argparse
import os
import tempfile
import unittest

import numpy as np
from astropy.io import fits

from src.frame_discovery import discover_frames
from src.quality_sweep import REJECT_SUFFIX, _walk_light_folders, run_quality_sweep, undo_quality_sweep


def _write_light(path, good=True, seed=0, H=128, W=160):
    """Bayer mosaic light: star-rich (good) or nearly blank (bad)."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W), 1000.0)
    if good:
        star_rng = np.random.default_rng(42)
        yy, xx = np.mgrid[0:H, 0:W]
        for _ in range(25):
            cy, cx = star_rng.uniform(10, H - 10), star_rng.uniform(10, W - 10)
            img += star_rng.uniform(3000, 9000) * np.exp(
                -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.6 ** 2))
        img += rng.normal(0, 20, (H, W))
    else:
        img += rng.normal(0, 1.0, (H, W))  # blank: no stars, tiny contrast
    hdu = fits.PrimaryHDU(data=img.astype(np.float32))
    hdu.header['BAYERPAT'] = 'RGGB'
    hdu.header['EXPTIME'] = 10.0
    hdu.writeto(path, overwrite=True)


def _args(**kw):
    d = dict(apply=False, parallel=1, verbose=False, quality_report=None,
             quality_filter=True, quality_threshold=50.0,
             max_ellipticity=0.5)
    d.update(kw)
    return argparse.Namespace(**d)


def _make_tree(root):
    """Two nested sessions; session A has 5 good + 2 bad lights and a dark."""
    a = os.path.join(root, 'M51', '2026-07-01')
    b = os.path.join(root, 'M42')
    os.makedirs(a)
    os.makedirs(b)
    for i in range(5):
        _write_light(os.path.join(a, f'Light{i:04d}.fits'), good=True, seed=i)
    _write_light(os.path.join(a, 'Light9998.fits'), good=False, seed=90)
    _write_light(os.path.join(a, 'Light9999.fits'), good=False, seed=91)
    _write_light(os.path.join(a, 'dark_0001.fits'), good=False, seed=92)
    for i in range(3):
        _write_light(os.path.join(b, f'Light{i:04d}.fits'), good=True, seed=100 + i)
    return a, b


class TestQualitySweep(unittest.TestCase):

    def test_walker_finds_nested_lights_and_ignores_darks(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = _make_tree(td)
            folders = dict(_walk_light_folders(td))
            self.assertIn(a, folders)
            self.assertIn(b, folders)
            names = [os.path.basename(f.path) for f in folders[a]]
            self.assertEqual(len(names), 7)  # dark excluded
            self.assertNotIn('dark_0001.fits', names)

    def test_dry_run_renames_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            a, _ = _make_tree(td)
            before = sorted(os.listdir(a))
            rc = run_quality_sweep(td, _args(apply=False))
            self.assertEqual(rc, 0)
            self.assertEqual(sorted(os.listdir(a)), before)

    def test_apply_flags_blank_frames_and_hides_them(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = _make_tree(td)
            rc = run_quality_sweep(td, _args(apply=True))
            self.assertEqual(rc, 0)
            names = set(os.listdir(a))
            # The two blank frames must be flagged; the five good ones kept.
            self.assertIn('Light9998.fits' + REJECT_SUFFIX, names)
            self.assertIn('Light9999.fits' + REJECT_SUFFIX, names)
            for i in range(5):
                self.assertIn(f'Light{i:04d}.fits', names)
            # Flagged files invisible to discovery now.
            lights = discover_frames(a).get('light', [])
            self.assertEqual(len(lights), 5)
            # Good-only folder untouched.
            self.assertEqual(len(discover_frames(b).get('light', [])), 3)

    def test_undo_restores_everything(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = _make_tree(td)
            before_a = sorted(os.listdir(a))
            run_quality_sweep(td, _args(apply=True))
            rc = undo_quality_sweep(td)
            self.assertEqual(rc, 0)
            self.assertEqual(sorted(os.listdir(a)), before_a)

    def test_csv_report(self):
        with tempfile.TemporaryDirectory() as td:
            a, _ = _make_tree(td)
            csv_path = os.path.join(td, 'report.csv')
            run_quality_sweep(td, _args(quality_report=csv_path))
            lines = open(csv_path, encoding='utf-8').read().strip().split('\n')
            self.assertEqual(lines[0], 'filename,snr,fwhm,star_count,'
                                       'quality_score,accepted,rejection_reason')
            self.assertEqual(len(lines), 1 + 10)  # 7 + 3 lights


if __name__ == '__main__':
    unittest.main()
