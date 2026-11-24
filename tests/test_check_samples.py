"""tools/check_samples.py: log parsing, the comparison, stack metrics and baseline adoption."""
import importlib.util
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

_spec = importlib.util.spec_from_file_location(
    'check_samples', Path(__file__).resolve().parents[1] / 'tools' / 'check_samples.py')
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)

LOG = """
  Light0021.fits: score=77  SNR=1.7  stars=48  FWHM=3.2  sharpness=0
  Light0008.fits: score=83  SNR=1.7  stars=52  FWHM=2.4  sharpness=0
  Light0017.fits: score=79  SNR=1.7  stars=48  FWHM=2.6  sharpness=0
  [OK] Accepted: 43/45 (95.6%)
  Target: Flame Nebula [Emission Nebula]  conf=100%  source=session
WARNING [debayer]: Bayer pattern GBRG: G1/G2 medians differ 1.70x
  Integration time: 7.5 minutes
  Total time: 1m 8.9s
"""


def _stack_fits(path, fwhm=3.5, noise=3.0, seed=0):
    rng = np.random.default_rng(seed)
    H, W = 400, 500
    yy, xx = np.mgrid[:H, :W].astype(np.float64)
    img = np.full((H, W), 200.0)
    sig = fwhm / 2.3548
    for _ in range(120):
        y, x = rng.uniform(20, H - 20), rng.uniform(20, W - 20)
        y0, y1, x0, x1 = int(y) - 15, int(y) + 16, int(x) - 15, int(x) + 16
        gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        img[y0:y1, x0:x1] += rng.uniform(600, 3000) * np.exp(-((gy - y) ** 2 + (gx - x) ** 2) / (2 * sig ** 2))
    cube = np.stack([img + rng.normal(0, noise, img.shape) for _ in range(3)]).astype(np.float32)
    hdr = fits.Header()
    hdr['RAWSTACK'] = True
    hdr['NFRAMES'] = 45
    fits.PrimaryHDU(cube, hdr).writeto(path, overwrite=True)


def test_parse_log():
    r = cs.parse_log(LOG)
    assert r['accepted'] == 43 and r['total'] == 45
    assert r['target'] == 'Flame Nebula' and r['target_type'] == 'Emission Nebula'
    assert r['integration_min'] == 7.5 and r['frame_fwhm'] == 2.6
    assert r['seconds'] == 68.9 and r['warnings'] == 1
    assert cs.parse_log('nothing here') == {'warnings': 0}


def test_compare_flags_real_differences_only():
    base = {'target': 'A', 'target_type': 'Galaxy', 'accepted': 45, 'total': 45, 'integration_min': 15.0,
            'frame_fwhm': 3.0, 'stack_fwhm': 3.5, 'stack_noise': 20.0}
    same = dict(base, accepted=44, frame_fwhm=3.2, stack_noise=21.5)       # within tolerance
    assert cs.compare(base, same) == []
    bad = dict(base, accepted=40, target_type='Star Field', stack_fwhm=4.5, integration_min=14.0)
    text = ' | '.join(cs.compare(base, bad))
    assert 'target_type' in text and 'frames accepted' in text and 'stack_fwhm' in text and 'integration_min' in text
    assert any('missing' in p for p in cs.compare(base, {k: v for k, v in base.items() if k != 'stack_noise'}))


def test_stack_metrics_measure_width_and_noise(tmp_path):
    p = tmp_path / 's.fits'
    _stack_fits(str(p), fwhm=3.5, noise=3.0)
    m = cs.stack_metrics(str(p))
    assert abs(m['stack_fwhm'] - 3.5) < 0.4
    assert 1.5 < m['stack_noise'] < 4.5
    wide = tmp_path / 'w.fits'
    _stack_fits(str(wide), fwhm=5.5)
    assert cs.stack_metrics(str(wide))['stack_fwhm'] > m['stack_fwhm'] + 1.0


def test_stack_metrics_read_the_colour_calibration(tmp_path):
    plain = tmp_path / 'p.fits'
    _stack_fits(str(plain))
    assert cs.stack_metrics(str(plain))['colour_calibrated'] is False
    cal = tmp_path / 'c.fits'
    _stack_fits(str(cal))
    with fits.open(cal, mode='update') as h:
        h[0].header['COLCAL'] = True
        for c, v in zip('RGB', (0.69, 1.0, 1.0134)):
            h[0].header[f'COLCAL_{c}'] = v
    m = cs.stack_metrics(str(cal))
    assert m['colour_calibrated'] is True and m['colour_scales'] == [0.69, 1.0, 1.0134]


def test_parse_log_reads_why_colour_calibration_did_not_run():
    assert cs.parse_log(LOG + "\n  Colour calibration: not enough well-measured Gaia stars -- skipped\n")[
        'colour_note'] == 'not enough well-measured Gaia stars'
    assert cs.parse_log(LOG + "\n  WARNING: colour calibration failed: timed out\n")['colour_note'] == 'failed: timed out'
    assert cs.parse_log("  Offline mode: no network requests will be made")['colour_note'] == 'offline'
    assert 'colour_note' not in cs.parse_log(LOG + "\n  Colour calibration: R x0.690 G x1.000 B x1.013 (6 Gaia stars)\n")


def test_missing_colour_calibration_skips_the_noise_not_the_rest():
    """The weekly check's M51 noise went 90.38 <-> 105.26 on one commit: Gaia was not reached."""
    base = {'accepted': 43, 'total': 45, 'stack_fwhm': 2.48, 'stack_noise': 90.38}     # pre-flag baseline
    uncal = dict(base, stack_noise=105.26, colour_calibrated=False, colour_note='failed: timed out')
    assert cs.compare(base, uncal) == []
    note = cs.colour_mismatch(base, uncal)
    assert 'did not run' in note and 'timed out' in note and '90.38 -> 105.26' in note
    # the other numbers are still compared
    assert any('stack_fwhm' in p for p in cs.compare(base, dict(uncal, stack_fwhm=3.5)))
    # a calibrated run against the same baseline: noise compared as before
    assert cs.colour_mismatch(base, dict(base, colour_calibrated=True)) is None
    assert any('stack_noise' in p for p in cs.compare(base, dict(base, stack_noise=105.26, colour_calibrated=True)))
    # both uncalibrated (e.g. a session Gaia never calibrates): compared
    flagged = dict(base, colour_calibrated=False)
    assert cs.colour_mismatch(flagged, dict(flagged)) is None
    assert any('stack_noise' in p for p in cs.compare(flagged, dict(flagged, stack_noise=130.0)))


def test_adopt_builds_a_baseline_from_logs_and_stacks(tmp_path, capsys):
    (tmp_path / '04_demo.log').write_text(LOG, encoding='utf-8')
    _stack_fits(str(tmp_path / '04_demo.fits'))
    (tmp_path / '_summary.txt').write_text('ignored', encoding='utf-8')
    out = tmp_path / 'base.json'
    assert cs.main(['--adopt', str(tmp_path), '--baseline', str(out)]) == 0
    data = json.loads(out.read_text())
    e = data['sessions']['04_demo']
    assert e['accepted'] == 43 and e['target_type'] == 'Emission Nebula' and 'stack_fwhm' in e
    assert set(data['sessions']) == {'04_demo'} and 'OriginStack' in data['note']
    assert cs.compare(e, e) == []
