"""Colour calibration's deeper Gaia query (``color_calibrate._merge_gaia_matches``)."""
import numpy as np

from src.color_calibrate import _merge_gaia_matches
from src.photometry import GaiaMatch


def _gm(x, y, g):
    z = np.zeros(len(x))
    return GaiaMatch(source_id=np.arange(len(x)), ra=z, dec=z, x=np.asarray(x, float),
                     y=np.asarray(y, float), g=np.asarray(g, float), bp=z, rp=z,
                     det_peak=z, fwhm=3.0, ap_radius=5, r_in=8.0, r_out=14.0,
                     field_ra=0.0, field_dec=0.0, plate_scale=1.0)


def test_fainter_slice_adds_only_new_detections():
    a = _gm([10, 100], [10, 100], [12, 13])
    b = _gm([11, 300], [10, 300], [14, 14.5])      # the first lands on a's detection
    m = _merge_gaia_matches(a, b)
    np.testing.assert_array_equal(m.x, [10, 100, 300])
    np.testing.assert_array_equal(m.g, [12, 13, 14.5])
    assert m.fwhm == a.fwhm and m.ap_radius == a.ap_radius
