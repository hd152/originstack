import numpy as np
from astropy.io import fits

from src.previous_stacks import find_previous_stacks


def _stack(path, obj, raw=True, frames=40, date="2026-01-01T20:00:00"):
    hdr = fits.Header()
    hdr["OBJECT"] = obj
    hdr["RAWSTACK"] = raw
    hdr["NFRAMES"] = frames
    hdr["DATE-OBS"] = date
    fits.PrimaryHDU(np.zeros((3, 4, 4), np.float32), hdr).writeto(path)


def test_finds_matching_raw_stacks_oldest_first(tmp_path):
    _stack(tmp_path / "b_stacked.fits", "Pinwheel Galaxy", date="2026-02-07T19:00:00")
    _stack(tmp_path / "a_stacked.fits", "pinwheel_galaxy", date="2026-01-26T19:00:00", frames=25)
    _stack(tmp_path / "c_stacked.fits", "Trifid Nebula")
    _stack(tmp_path / "d_stacked.fits", "Pinwheel Galaxy", raw=False)
    (tmp_path / "notes_stacked.fits").write_bytes(b"not fits")
    got = find_previous_stacks("Pinwheel galaxy", [str(tmp_path)])
    assert [g["frames"] for g in got] == [25, 40]
    assert got[0]["path"].endswith("a_stacked.fits")


def test_exclude_and_empty_target(tmp_path):
    _stack(tmp_path / "a_stacked.fits", "M101")
    p = str(tmp_path / "a_stacked.fits")
    assert find_previous_stacks("M101", [str(tmp_path)], exclude=[p]) == []
    assert find_previous_stacks("", [str(tmp_path)]) == []
    assert find_previous_stacks("M101", [str(tmp_path / "missing")]) == []
