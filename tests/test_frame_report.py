"""--frame-report: verdict + plain reason per light, from data the gates already produce."""
import argparse
import csv
import json
import os

import numpy as np

from src.frame_processor import quality_gate
from src.frame_report import build_frame_report, emit_frame_report
from src.models import FrameInfo, ProcessingStats
from src.ui_events import UIEvents


def _frame(i, fwhm=3.0, stars=100, score=100.0, snr=10.0, **kw):
    m = dict(score=score, snr=snr, contrast=10.0, dynamic_range=100.0, fwhm=fwhm,
             star_count=stars, background=1000.0, ellipticity=0.1)
    m.update(kw)
    return FrameInfo(path=f'/x/light_{i:02d}.fits', type='light', header={}, metrics=m)


def _args(**kw):
    d = dict(verbose=False, quality_filter=True, quality_threshold=50.0,
             max_ellipticity=0.5, frame_report=True)
    d.update(kw)
    return argparse.Namespace(**d)


def _session():
    lights = [_frame(i) for i in range(12)]
    lights[3] = _frame(3, fwhm=7.1)                       # kept, but flagged
    lights[5] = _frame(5, stars=0)                        # hard limit
    lights[7] = _frame(7, score=10.0)                     # percentile gate
    lights[9] = _frame(9, ellipticity=0.8)                # kept, flagged
    lights[10].metrics['phase2_drop'] = 'registration residual 4.00 px RMS > 1.50 px'
    return lights


def _run(lights, args):
    reasons = {}
    stats = ProcessingStats(total_frames=len(lights))
    final = quality_gate(lights, args, reasons, stats)
    final = [f for f in final if not f.metrics.get('phase2_drop')]
    return reasons, final


def test_rows_verdicts_and_reasons():
    lights = _session()
    reasons, final = _run(lights, _args())
    rows = {r['filename']: r for r in build_frame_report(lights, reasons, final)}
    assert len(rows) == 12
    assert rows['light_00.fits']['verdict'] == 'kept'
    assert rows['light_05.fits']['stage'] == 'hard_limit'
    assert rows['light_05.fits']['reason'] == 'no stars detected'
    assert rows['light_07.fits']['stage'] == 'percentile'
    assert rows['light_07.fits']['reason'].startswith('score 10.0 <')
    assert rows['light_10.fits']['stage'] == 'registration'
    assert 'residual 4.00 px' in rows['light_10.fits']['reason']
    n3 = rows['light_03.fits']
    assert n3['verdict'] == 'kept' and 'FWHM 7.1 px > 1.5x median' in n3['notes']
    assert 'ellipticity 0.80 > 0.5' in rows['light_09.fits']['notes']
    assert rows['light_00.fits']['notes'] == ''


def test_report_changes_no_gate_outcome():
    a, b = _session(), _session()
    ra, fa = _run(a, _args())
    build_frame_report(b, {}, [], 0.5)    # building the report must not touch frames
    rb, fb = _run(b, _args())
    assert ra == rb
    assert [f.path for f in fa] == [f.path for f in fb]
    assert [f.accepted for f in a] == [f.accepted for f in b]


def test_files_and_ui_event(tmp_path):
    lights = _session()
    reasons, final = _run(lights, _args())
    ui = UIEvents()
    ui.active = True
    import src.ui_events as uie
    old, uie._ui_events = uie._ui_events, ui
    try:
        out = str(tmp_path / 'stack.fits')
        emit_frame_report(lights, reasons, final, _args(verbose=True), out)
    finally:
        uie._ui_events = old
    with open(tmp_path / 'stack_frames.csv', newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 12 and rows[5]['verdict'] == 'rejected'
    data = json.loads((tmp_path / 'stack_frames.json').read_text(encoding='utf-8'))
    assert data['summary']['hard_limit'] == 1 and data['summary']['kept'] == len(final)
    assert len(ui.snapshot()['frame_report']) == 12
    assert not np.isnan([r['fwhm'] or 0 for r in data['frames']]).any()


def test_flag_registered_and_off_by_default():
    import sys

    from src.cli import parse_args
    old = sys.argv
    try:
        sys.argv = ['originstack', '-d', '.']
        assert parse_args().frame_report is False
        sys.argv = ['originstack', '-d', '.', '--frame-report']
        assert parse_args().frame_report is True
    finally:
        sys.argv = old
