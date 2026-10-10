"""A multi-session run writes its combined stack into the folder holding the
sessions; the next run on that folder must still see the sessions, not stack
that one file as a light (found on a real Orion Nebula folder)."""
import numpy as np
import pytest
from astropy.io import fits

from src.background import extract_background
from src.cli import _root_has_input_fits
from src.frame_discovery import classify_frame, discover_frames
from src.merge import seed_reference_header


def _write(path, data, **keys):
    hdu = fits.PrimaryHDU(data.astype(np.float32))
    for k, v in keys.items():
        hdu.header[k] = v
    hdu.writeto(path)


def test_rawstack_without_creator_is_skipped():
    assert classify_frame('x/Orion Nebula_stacked.fits', {'RAWSTACK': True}) == 'skip'


def test_combined_header_is_marked_as_ours():
    hdr = fits.Header()
    seed_reference_header(hdr, {})
    assert classify_frame('x/combined.fits', dict(hdr)) == 'skip'


def test_root_with_only_a_previous_stack_is_not_single_folder(tmp_path):
    _write(tmp_path / 'Orion Nebula_stacked.fits', np.zeros((3, 8, 8)), RAWSTACK=True)
    (tmp_path / 'session1').mkdir()
    _write(tmp_path / 'session1' / 'Light_0001.fits', np.zeros((8, 8)), IMAGETYP='Light')
    assert not _root_has_input_fits(str(tmp_path))
    assert discover_frames(str(tmp_path))['light'] == []


def test_root_with_lights_is_single_folder(tmp_path):
    _write(tmp_path / 'stacked.fits', np.zeros((3, 8, 8)), RAWSTACK=True)
    _write(tmp_path / 'Light_0001.fits', np.zeros((8, 8)), IMAGETYP='Light')
    assert _root_has_input_fits(str(tmp_path))


@pytest.mark.parametrize('shape', [(20, 400), (400, 20), (20, 20)])
def test_background_mesh_one_cell_high_or_wide(shape):
    rng = np.random.default_rng(0)
    img = (100.0 + rng.normal(0, 1, shape)).astype(np.float32)
    bg = extract_background(img, mesh_size=64)
    assert bg.shape == shape
    assert np.all(np.isfinite(bg))
    assert abs(float(np.median(bg)) - 100.0) < 1.0
