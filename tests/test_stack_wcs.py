"""Session info.json WCS: orientation, and carrying it onto the registered stack.

The orientation numbers are a real measurement, not a convention: a Gaia
solve (src/local_solve.py) of the Sunflower session's own stack gave the CD
matrix pinned below, from the same info.json values. The previous textbook
east-left form mirrored the sky (0 of 42 Gaia stars within 2 px).
"""
import math
from types import SimpleNamespace

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from src.session_info import SessionInfo, build_wcs_keywords, stack_wcs_keywords
from src.transparency import to_aligned_yx


def _sunflower():
    return SessionInfo(ra_rad=3.4725225931238355, dec_rad=0.7333140653118461,
                       fov_x_rad=0.02189285686932017, fov_y_rad=0.014671975601117918,
                       orientation_rad=0.12073042282491844,
                       image_width=3056, image_height=2048)


def _cd(kw):
    return np.array([[kw['CD1_1'][0], kw['CD1_2'][0]], [kw['CD2_1'][0], kw['CD2_2'][0]]])


def test_orientation_matches_a_gaia_solve_of_the_session():
    solved = np.array([[4.07207228e-04, -4.91606594e-05],
                       [4.90392539e-05, 4.07057311e-04]])
    np.testing.assert_allclose(_cd(build_wcs_keywords(_sunflower())), solved, rtol=0.01, atol=2e-7)
    assert np.linalg.det(_cd(build_wcs_keywords(_sunflower()))) > 0


def _hdr(kw, shape=None):
    h = fits.Header()
    for k, (v, _c) in kw.items():
        h[k] = v
    return WCS(h)


def test_stack_wcs_follows_rotation_translation_and_crop():
    si = _sunflower()
    phi = math.radians(2.3)
    R = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
    params = np.eye(3)
    params[:2, :2] = R
    params[:2, 2] = [37.0, -21.0]
    tr = SimpleNamespace(params=params)
    top, left = 64, 41

    def to_stack(yx):
        return to_aligned_yx(yx, tr, None) - np.array([top, left], float)

    raw = _hdr(build_wcs_keywords(si))
    stk = _hdr(stack_wcs_keywords(si, to_stack, (1900, 2970)))
    rng = np.random.default_rng(0)
    yx = np.column_stack([rng.uniform(200, 1800, 50), rng.uniform(300, 2700, 50)])
    sky_raw = raw.all_pix2world(yx[:, ::-1], 0)
    st = to_stack(yx)
    sky_stk = stk.all_pix2world(st[:, ::-1], 0)
    d = np.hypot((sky_raw[:, 0] - sky_stk[:, 0]) * math.cos(si.dec_rad),
                 sky_raw[:, 1] - sky_stk[:, 1]) * 3600
    assert d.max() < 0.05          # arcsec; ~0.03 px at 1.48"/px


def test_identity_mapping_only_moves_crpix():
    si = _sunflower()
    raw = build_wcs_keywords(si)
    out = stack_wcs_keywords(si, lambda yx: np.asarray(yx) - [10.0, 20.0], (2000, 3000))
    assert out['CRPIX1'][0] == raw['CRPIX1'][0] - 20.0
    assert out['CRPIX2'][0] == raw['CRPIX2'][0] - 10.0
    np.testing.assert_allclose(_cd(out), _cd(raw), rtol=1e-12)


def test_no_session_wcs_gives_nothing():
    assert stack_wcs_keywords(SessionInfo(), lambda yx: yx, (10, 10)) == {}
