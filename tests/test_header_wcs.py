"""Stack WCS for captures without an info.json solve (Seestar, Dwarf, other apps).

The first sub is solved from its header's pointing keywords (2x2-binned, so a
Bayer mosaic needs no debayer) and its WCS carried onto the stack through the
same registration mapping the info.json path uses. On the S50 samples this put
Gaia stars within 0.03-0.16 px of the stack's stars and turned on the Gaia
colour calibration, which had been skipped ("no stack WCS").
"""
from types import SimpleNamespace

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from src import local_solve, pipeline
from src.frame_discovery import discover_frames, is_os_metadata
from src.local_solve import LocalSolution, solve_raw_frame
from src.models import FrameInfo


def _seestar_header():
    h = fits.Header()
    for k, v in (('RA', 211.025), ('DEC', 54.2217), ('FOCALLEN', 250), ('XPIXSZ', 2.9),
                 ('YPIXSZ', 2.9), ('IMAGETYP', 'Light'), ('EXPTIME', 10.0),
                 ('INSTRUME', 'Seestar S50'), ('BAYERPAT', 'GRBG')):
        h[k] = v
    return h


def _wcs(cards):
    h = fits.Header()
    for k, (v, _c) in cards.items():
        h[k] = v
    return WCS(h, naxis=2)


def test_raw_frame_solution_is_carried_from_binned_to_native_pixels(monkeypatch):
    rot = np.radians(52.0)
    s = 4.741 / 3600.0
    cd_b = s * np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    binned = LocalSolution(crval=(210.80, 54.34), crpix=(270.5, 480.5), cd=cd_b,
                           n_match=43, rms_px=0.44, scale_arcsec=4.741)
    seen = {}

    def fake_solve(img, hint, fetch=True, verbose=False):
        seen['shape'], seen['scale'] = img.shape, hint.scale
        return binned
    monkeypatch.setattr(local_solve, 'solve_local', fake_solve)

    raw = np.zeros((1920, 1080), np.uint16)
    cards = solve_raw_frame(raw, _seestar_header())
    assert seen['shape'] == (960, 540)
    native_scale = 2.9 / 1000.0 / 250 * 206264.806
    assert abs(seen['scale'] - 2 * native_scale) < 1e-6

    wb, wn = _wcs(binned.header_cards()), _wcs(cards)
    rng = np.random.default_rng(1)
    xy = np.column_stack([rng.uniform(0, 1080, 20), rng.uniform(0, 1920, 20)])
    # native pixel p lies at binned (p - 0.5) / 2 (a binned pixel spans native 2b, 2b+1)
    np.testing.assert_allclose(wn.all_pix2world(xy, 0), wb.all_pix2world((xy - 0.5) / 2.0, 0),
                               atol=1e-9)


def test_raw_frame_without_pointing_or_scale_is_not_solved(monkeypatch):
    monkeypatch.setattr(local_solve, 'solve_local', lambda *a, **k: 1 / 0)
    h = _seestar_header()
    del h['FOCALLEN']
    assert solve_raw_frame(np.zeros((200, 200)), h) is None
    h = _seestar_header()
    del h['RA'], h['DEC']
    assert solve_raw_frame(np.zeros((200, 200)), h) is None


def _cards():
    return {'CTYPE1': ('RA---TAN', ''), 'CTYPE2': ('DEC--TAN', ''),
            'CRVAL1': (210.8, ''), 'CRVAL2': (54.34, ''),
            'CRPIX1': (540.5, ''), 'CRPIX2': (960.5, ''),
            'CD1_1': (-6.6e-4, ''), 'CD1_2': (0.0, ''), 'CD2_1': (0.0, ''), 'CD2_2': (6.6e-4, '')}


def test_stack_wcs_from_the_first_sub_without_session_info(monkeypatch):
    lights = [FrameInfo(path=f'L{i}.fit', type='light',
                        header={'DATE-OBS': f'2026-09-14T22:{i:02d}:00'}) for i in range(3)]
    solved = []
    monkeypatch.setattr(pipeline, '_solve_first_sub',
                        lambda frame, args: solved.append(frame.path) or _cards())
    refined = []
    monkeypatch.setattr(local_solve, 'solve_header', lambda *a, **k: refined.append(1) or True)
    args = SimpleNamespace(_session_info=None, verbose=False)
    shifts = [(0.0, 0.0), (5.0, -3.0), (10.0, -6.0)]
    pipeline._settle_stack_wcs(args, lights, lights, shifts, [None] * 3, 7, 11,
                               np.zeros((1900, 1060, 3), np.float32))
    assert solved == ['L0.fit']
    assert refined == []            # a solved sub needs no second solve on the stack
    cards = args._stack_wcs
    assert cards['WCSORIG'][0] == 'local_solve'
    # the crop moved the reference pixel; the scale is unchanged
    assert abs(cards['CRPIX1'][0] - (540.5 - 11)) < 1e-6
    assert abs(cards['CRPIX2'][0] - (960.5 - 7)) < 1e-6
    assert abs(cards['CD2_2'][0] - 6.6e-4) < 1e-12


def test_no_header_solve_with_wcs_refine_off(monkeypatch):
    monkeypatch.setattr(pipeline, '_solve_first_sub', lambda *a: 1 / 0)
    lights = [FrameInfo(path='L0.fit', type='light', header={})]
    args = SimpleNamespace(_session_info=None, wcs_refine=False)
    pipeline._settle_stack_wcs(args, lights, lights, [(0.0, 0.0)], [None], 0, 0,
                               np.zeros((10, 10, 3), np.float32))
    assert getattr(args, '_stack_wcs', None) is None


def test_stack_wcs_reaches_the_output_header_without_session_info():
    from src.cli import parse_args
    from src.io_fits import populate_fits_header
    from src.models import ProcessingStats
    args = parse_args(['-d', 'x', '-o', 'y.fits', '--no-auto'])
    args._session_info = None
    args._stack_wcs = _cards()
    f = FrameInfo(path='a.fit', type='light', header={'DATE-OBS': '2026-09-14T22:27:35', 'EXPTIME': 10.0})
    hdr = fits.Header()
    populate_fits_header(hdr, [f], ProcessingStats(), args, (3, 8, 8), [(0.0, 0.0)],
                         {'bias': None, 'dark': None, 'flat': None})
    assert hdr['CTYPE1'] == 'RA---TAN' and hdr['CRVAL1'] == 210.8


def test_macos_metadata_files_are_not_frames(tmp_path):
    assert is_os_metadata('._Light_M 31_10.0s_IRCUT_20260915-002755.fit')
    assert is_os_metadata('__MACOSX')
    assert not is_os_metadata('Light_M 31_10.0s_IRCUT_20260915-002755.fit')
    hdu = fits.PrimaryHDU(np.zeros((8, 8), np.uint16), header=_seestar_header())
    hdu.writeto(tmp_path / 'Light_M 31_10.0s_IRCUT_20260915-002755.fit')
    # a Finder zip's AppleDouble twin: a resource fork with the frame's name
    (tmp_path / '._Light_M 31_10.0s_IRCUT_20260915-002755.fit').write_bytes(b'\x00\x05\x16\x07' + bytes(216))
    frames = discover_frames(str(tmp_path))
    assert [f.path.split('\\')[-1].split('/')[-1] for f in frames['light']] == \
        ['Light_M 31_10.0s_IRCUT_20260915-002755.fit']
