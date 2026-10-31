"""FITS loading beyond the plain 2-D Bayer frame: tile-compressed .fits.fz, 3-plane cubes,
and monochrome frames (src/io_fits.py)."""
import numpy as np
import pytest
from astropy.io import fits

from src import io_fits
from src.frame_discovery import discover_frames


def _write(path, data, **cards):
    hdu = fits.PrimaryHDU(data)
    for k, v in cards.items():
        hdu.header[k] = v
    hdu.writeto(path, overwrite=True)


def _write_fz(path, data, **cards):
    hdu = fits.CompImageHDU(data)
    for k, v in cards.items():
        hdu.header[k] = v
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)


@pytest.mark.parametrize('cards,expected', [
    ({'BAYERPAT': 'RGGB'}, None),                    # Origin
    ({'BAYERPAT': 'GRBG', 'FILTER': 'IRCUT'}, None),  # Seestar-like: a pattern wins over a filter
    ({'BAYERPAT': 'MONO'}, 'header'),
    ({'BAYERPAT': 'none'}, 'header'),
    ({'COLORTYP': 'NONE'}, 'header'),
    ({'FILTER': 'rp'}, 'filter'),
    ({'FILTER': ''}, None),                          # Origin writes an empty filter, no BAYERPAT on old files
    ({}, None),                                      # headerless FITS stays a Bayer frame
    ({'FILTER': 'NONE'}, None),
])
def test_mono_reason(cards, expected):
    assert io_fits.mono_reason(cards) == expected


def test_bayer_frame_is_untouched(tmp_path):
    d = np.arange(64 * 48, dtype=np.float32).reshape(48, 64)
    p = str(tmp_path / 'b.fits')
    _write(p, d, BAYERPAT='RGGB')
    got, hdr = io_fits.load_fits(p)
    assert got.shape == (48, 64) and hdr['BAYERPAT'] == 'RGGB'
    np.testing.assert_array_equal(got, d)


def test_mono_frame_is_replicated_to_three_channels(tmp_path):
    d = np.random.default_rng(0).normal(100, 5, (40, 60)).astype(np.float32)
    p = str(tmp_path / 'm.fits')
    _write(p, d, FILTER='V')
    got, _ = io_fits.load_fits(p)
    assert got.shape == (40, 60, 3)
    for c in range(3):
        np.testing.assert_array_equal(got[..., c], d)


def test_colour_cube_comes_back_height_width_channels(tmp_path):
    cube = np.random.default_rng(1).normal(50, 3, (3, 30, 44)).astype(np.float32)
    p = str(tmp_path / 'c.fits')
    _write(p, cube)
    got, _ = io_fits.load_fits(p)
    assert got.shape == (30, 44, 3)
    np.testing.assert_array_equal(got[..., 1], cube[1])


def test_tile_compressed_frame_reads_pixels_and_header_from_the_extension(tmp_path):
    d = np.random.default_rng(2).integers(0, 3000, (36, 52)).astype(np.int16)
    p = str(tmp_path / 'f.fits.fz')
    _write_fz(p, d, FILTER='gp', EXPTIME=30.0, OBJECT='M51')
    got, hdr = io_fits.load_fits(p)
    assert got.shape == (36, 52, 3)                           # mono by its filter
    np.testing.assert_array_equal(got[..., 0], d.astype(np.float32))
    assert hdr['FILTER'] == 'gp' and hdr['EXPTIME'] == 30.0 and hdr['OBJECT'] == 'M51'
    assert 'ZIMAGE' not in hdr and 'XTENSION' not in hdr
    assert io_fits._read_fits_header(p)['OBJECT'] == 'M51'
    assert io_fits.read_fits_be16(p) is None


def test_mono_frame_takes_the_bitpix16_fast_path_as_a_plane(tmp_path):
    """The raw-block fast path returns the 2-D plane (the frame processor replicates it after
    calibration); a Bayer frame still goes through it unchanged."""
    p = str(tmp_path / 'm16.fits')
    _write(p, np.zeros((20, 20), np.int16), FILTER='V')
    fast = io_fits.read_fits_be16(p)
    assert fast is not None and fast[0].shape == (20, 20)
    q = str(tmp_path / 'b16.fits')
    _write(q, np.zeros((20, 20), np.int16), BAYERPAT='RGGB')
    assert io_fits.read_fits_be16(q) is not None


def test_replicate_mono_false_returns_the_plane(tmp_path):
    d = np.random.default_rng(3).normal(100, 5, (30, 40)).astype(np.float32)
    p = str(tmp_path / 'm.fits')
    _write(p, d, FILTER='V')
    plane, hdr = io_fits.load_frame(p, replicate_mono=False)
    assert plane.shape == (30, 40) and io_fits.mono_reason(hdr)
    assert io_fits.load_frame(p)[0].shape == (30, 40, 3)          # the default is unchanged


def test_discovery_finds_fits_fz(tmp_path):
    _write_fz(str(tmp_path / 'a-light.fits.fz'), np.zeros((16, 16), np.int16), FILTER='rp', EXPTIME=60.0)
    found = discover_frames(str(tmp_path))
    assert [f.path.endswith('a-light.fits.fz') for f in found['light']] == [True]
