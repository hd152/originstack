"""Frame types from the usual IMAGETYP spellings of capture software, and dark flats kept
out of the darks."""
import pytest

from src.frame_discovery import _imagetyp_kind, classify_frame


@pytest.mark.parametrize('value, kind', [
    ('DARK', 'dark'), ('Dark Frame', 'dark'), ('dark', 'dark'),
    ('FLAT', 'flat'), ('Flat Field', 'flat'), ('Flat Frame', 'flat'),
    ('BIAS', 'bias'), ('Bias Frame', 'bias'), ('Offset', 'bias'), ('ZERO', 'bias'),
    ('LIGHT', 'light'), ('Light Frame', 'light'), ('Object', 'light'),
    ('Dark Flat', 'darkflat'), ('DARKFLAT', 'darkflat'), ('Flat Dark', 'darkflat'),
    (None, None), ('', None), ('Something', None)])
def test_imagetyp_spellings(value, kind):
    assert _imagetyp_kind(value) == kind


@pytest.mark.parametrize('imagetyp, kind', [('Dark Frame', 'dark'), ('Flat Field', 'flat'),
                                            ('Bias Frame', 'bias'), ('Light Frame', 'light')])
def test_header_type_with_a_plain_file_name(imagetyp, kind):
    assert classify_frame('x/2026-10-16_001.fits', {'IMAGETYP': imagetyp, 'EXPTIME': 10}) == kind


@pytest.mark.parametrize('name, hdr', [('x/DarkFlat_001.fits', {}), ('x/dark_flat_5s.fits', {}),
                                       ('x/flat-dark-01.fits', {}),
                                       ('x/frame_001.fits', {'IMAGETYP': 'Dark Flat', 'EXPTIME': 1})])
def test_dark_flats_are_not_darks(name, hdr):
    assert classify_frame(name, hdr) == 'darkflat'


def test_origin_flat_labelled_light_stays_a_flat():
    # real Origin flats: the file name says flat, IMAGETYP says LIGHT
    assert classify_frame('x/flat.fits', {'IMAGETYP': 'LIGHT', 'EXPTIME': 1}) == 'flat'


def test_unchanged_cases():
    assert classify_frame('x/Light0001.fits', {'IMAGETYP': 'LIGHT', 'EXPTIME': 20}) == 'light'
    assert classify_frame('x/dark0009.fits', {}) == 'dark'
    assert classify_frame('x/a.fits', {'EXPTIME': 0}) == 'bias'
    assert classify_frame('x/a.fits', {'EXPTIME': 10}) == 'light'


def test_discovery_leaves_dark_flats_out(tmp_path):
    import numpy as np
    from astropy.io import fits

    from src.frame_discovery import discover_frames
    for name, typ in [('Light_001.fits', 'Light Frame'), ('a.fits', 'Dark Frame'),
                      ('b.fits', 'Dark Flat'), ('c.fits', 'Flat Field')]:
        h = fits.Header(); h['IMAGETYP'] = typ; h['EXPTIME'] = 5.0
        fits.PrimaryHDU(np.zeros((4, 4), np.uint16), header=h).writeto(tmp_path / name)
    f = discover_frames(str(tmp_path))
    assert {k: len(v) for k, v in f.items()} == {'light': 1, 'dark': 1, 'flat': 1, 'bias': 0}
