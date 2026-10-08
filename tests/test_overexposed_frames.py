"""Overexposed subs are rejected on their raw values.

A comet session shot in bright twilight had a raw median of 65535 and 52% of its
pixels clipped. The existing gate runs after calibration and white balance (it
rejected only above 95% 'at max'), so the frames were stacked and every clipped star
core came out magenta or green depending on which channel had clipped."""
import numpy as np
import pytest

from src.frame_processor import raw_saturated_fraction


def _be16(phys: np.ndarray, bzero: float) -> np.ndarray:
    """The block read_fits_be16 returns: stored int16 values, big-endian bytes,
    viewed as native uint16."""
    stored = (phys.astype(np.int32) - int(bzero)).astype(np.int16)
    return stored.astype('>i2').view(np.uint16)


def _mosaic(H=120, W=160, sky=20000.0):
    rng = np.random.default_rng(0)
    return np.clip(rng.normal(sky, 300, (H, W)), 0, 65535)


def test_clean_frame_reads_near_zero():
    raw = _be16(_mosaic(), 32768.0)
    assert raw_saturated_fraction(raw, 32768.0) < 0.001


def test_clipped_green_sites_only_are_detected():
    """Only G clipped (R/B below full scale): every-4th-pixel sampling visits one
    colour site and missed this entirely on real data."""
    phys = _mosaic(sky=40000.0)
    phys[0::2, 1::2] = 65535    # RGGB: G1
    phys[1::2, 0::2] = 65535    # G2
    f = raw_saturated_fraction(_be16(phys, 32768.0), 32768.0)
    assert 0.45 < f < 0.55


def test_fully_saturated_frame():
    phys = np.full((90, 90), 65535.0)
    assert raw_saturated_fraction(_be16(phys, 32768.0), 32768.0) == pytest.approx(1.0)


def test_integer_and_float_paths():
    phys = _mosaic()
    phys[:60] = 65535
    assert raw_saturated_fraction(phys.astype(np.uint16)) == pytest.approx(0.5, abs=0.02)
    assert raw_saturated_fraction(phys.astype(np.float32)) == pytest.approx(0.5, abs=0.02)
    # float data whose maximum is not 16-bit full scale: full scale unknown
    assert raw_saturated_fraction((phys / 65535).astype(np.float32)) is None
