"""tools/make_sample_session.py: location scrubbing of a shareable sample session."""
import importlib.util
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

_spec = importlib.util.spec_from_file_location(
    'make_sample_session', Path(__file__).resolve().parents[1] / 'tools' / 'make_sample_session.py')
ms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ms)

LAT, LON = 33.8258129999915, -117.78890269997


def _write_fits(path, kind='LIGHT', seed=0):
    h = fits.Header()
    h['IMAGETYP'] = kind
    h['CAMERA'] = 'Origin178-932f562381e97ce70'
    h['RA'] = 202.4
    if kind == 'LIGHT':
        h['LATITUDE'], h['LONGITUD'], h['ALTITUDE'] = LAT, LON, 0.0
        h['AIRMASS'] = 1.34
        h['SITELAT'] = LAT
        h['OBSGEO-B'] = LAT
    data = np.random.default_rng(seed).integers(0, 60000, (16, 24)).astype(np.uint16)
    fits.PrimaryHDU(data, h).writeto(path)


def _session(tmp_path, n=6):
    src = tmp_path / 'session'
    src.mkdir()
    for i in range(n):
        _write_fits(src / f'Light{i + 1:04d}.fits', seed=i)
    _write_fits(src / 'flat.fits', 'FLAT')
    (src / 'FinalStackedMaster.tiff').write_bytes(b'II*\x00' + f'{LAT:.6f}'.encode())
    (src / 'info.json').write_text(json.dumps({'StackedInfo': {
        'gps': {'latitude': LAT, 'longitude': LON, 'altitude': 0.0},
        'uuid': 'abc', 'objectName': 'Whirlpool Galaxy'}}))
    (src / 'dark.json').write_text(json.dumps({'cameraID': 'Origin178-932f562381e97ce70', 'iso': 200}))
    return src


def test_is_site_key():
    for k in ('LATITUDE', 'longitud', 'ALTITUDE', 'AIRMASS', 'SITELAT', 'SITELONG', 'OBSGEO-B'):
        assert ms.is_site_key(k)
    for k in ('RA', 'DEC', 'EXPTIME', 'DATE-OBS', 'TIMEZONE', 'CAMERA'):
        assert not ms.is_site_key(k)


def test_scrub_json_drops_gps_uuid_and_serial():
    out = ms.scrub_json({'StackedInfo': {'gps': {'latitude': 1}, 'uuid': 'x', 'iso': 500},
                         'cameraID': 'Origin178-932f562381e97ce70'})
    assert 'gps' not in out['StackedInfo'] and 'uuid' not in out['StackedInfo']
    assert out['StackedInfo']['iso'] == 500
    assert out['cameraID'] == 'Origin178-' + ms.FAKE_UNIT


def test_pick_evenly():
    names = [str(i) for i in range(100)]
    got = ms.pick_evenly(names, 5)
    assert got == ['0', '25', '50', '74', '99'] or (len(got) == 5 and got[0] == '0' and got[-1] == '99')
    assert ms.pick_evenly(names[:3], 5) == names[:3]


def test_build_is_clean_and_keeps_pixels(tmp_path):
    src = _session(tmp_path)
    dst = tmp_path / 'sample'
    rep = ms.build(str(src), str(dst), 4)
    assert rep['lights'] == 4
    assert ms.verify(str(src), str(dst), rep) == []
    assert not (dst / 'FinalStackedMaster.tiff').exists()
    h = fits.getheader(dst / 'Light0001.fits')
    assert 'LATITUDE' not in h and 'SITELAT' not in h and 'OBSGEO-B' not in h and 'AIRMASS' not in h
    assert h['RA'] == 202.4 and h['CAMERA'] == 'Origin178-' + ms.FAKE_UNIT
    info = json.loads((dst / 'info.json').read_text())
    assert 'gps' not in info['StackedInfo']


def test_verify_catches_a_leftover(tmp_path):
    src = _session(tmp_path)
    dst = tmp_path / 'sample'
    rep = ms.build(str(src), str(dst), 4)
    (dst / 'notes.json').write_text(json.dumps({'where': f'{abs(LAT):.3f}'}))
    assert any('contains' in p for p in ms.verify(str(src), str(dst), rep))
