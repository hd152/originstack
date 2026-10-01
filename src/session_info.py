"""Parse per-session info.json written by capture apps.

Public API
----------
load_session_info(directory) -> Optional[SessionInfo]
build_wcs_keywords(si)       -> dict   (empty if data insufficient)
stack_wcs_keywords(si, ...)  -> dict   (the same WCS mapped onto a registered, cropped stack)
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Optional

_log = logging.getLogger("originstack")

_INFO_FILENAMES = ('info.json', 'session.json', 'capture.json')


@dataclass
class SessionInfo:
    """Parsed capture-app session metadata."""
    object_name: Optional[str] = None
    ra_rad: Optional[float] = None
    dec_rad: Optional[float] = None
    fov_x_rad: Optional[float] = None
    fov_y_rad: Optional[float] = None
    orientation_rad: Optional[float] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    bayer: Optional[str] = None
    filter_name: Optional[str] = None
    telescope: Optional[str] = None
    mount: Optional[str] = None
    reducer: Optional[str] = None
    exposure: Optional[float] = None
    iso: Optional[int] = None
    temperature: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    stacked_depth: Optional[int] = None
    total_duration_ms: Optional[float] = None
    date_time: Optional[str] = None
    stretch_background: Optional[float] = None
    stretch_strength: Optional[float] = None

    @property
    def has_wcs(self) -> bool:
        return all(v is not None for v in (
            self.ra_rad, self.dec_rad,
            self.fov_x_rad, self.fov_y_rad,
            self.image_width, self.image_height,
        ))

    @property
    def has_gps(self) -> bool:
        return self.latitude is not None and self.longitude is not None


def load_session_info(directory: str) -> Optional[SessionInfo]:
    """Search *directory* for an info.json and parse it.

    Returns a SessionInfo on success, None if not found or malformed.
    """
    for fname in _INFO_FILENAMES:
        path = os.path.join(directory, fname)
        if os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    raw = json.load(fh)
                return _parse(raw, path)
            except Exception as exc:
                _log.debug("Failed to parse %s: %s", path, exc)
    return None


def _parse(raw: dict, path: str) -> Optional['SessionInfo']:
    d = raw.get('StackedInfo', raw)

    si = SessionInfo()

    si.object_name = _str_or_none(d.get('objectName'))
    si.date_time = _str_or_none(d.get('dateTime'))
    si.image_width = _int_or_none(d.get('imageWidth'))
    si.image_height = _int_or_none(d.get('imageHeight'))
    si.stacked_depth = _int_or_none(d.get('stackedDepth'))
    si.total_duration_ms = _float_or_none(d.get('totalDurationMs'))
    si.stretch_background = _float_or_none(d.get('stretchBackground'))
    si.stretch_strength = _float_or_none(d.get('stretchStrength'))

    bayer_raw = _str_or_none(d.get('bayer'))
    si.bayer = bayer_raw.upper() if bayer_raw else None

    si.filter_name = _str_or_none(d.get('filter'))
    si.telescope = _str_or_none(d.get('telescope'))
    si.mount = _str_or_none(d.get('mount'))
    si.reducer = _str_or_none(d.get('reducer'))
    si.fov_x_rad = _float_or_none(d.get('fovX'))
    si.fov_y_rad = _float_or_none(d.get('fovY'))
    si.orientation_rad = _float_or_none(d.get('orientation'))

    cel = d.get('celestial')
    if isinstance(cel, dict):
        si.ra_rad = _float_or_none(cel.get('first'))
        si.dec_rad = _float_or_none(cel.get('second'))

    gps = d.get('gps')
    if isinstance(gps, dict):
        si.latitude = _float_or_none(gps.get('latitude'))
        si.longitude = _float_or_none(gps.get('longitude'))
        si.altitude = _float_or_none(gps.get('altitude'))

    cp = d.get('captureParams')
    if isinstance(cp, dict):
        si.exposure = _float_or_none(cp.get('exposure'))
        si.iso = _int_or_none(cp.get('iso'))
        si.temperature = _float_or_none(cp.get('temperature'))

    _log.debug(
        "Loaded session info from %s: object=%r ra=%.4f dec=%.4f bayer=%s",
        path, si.object_name, si.ra_rad or 0.0, si.dec_rad or 0.0, si.bayer,
    )
    return si


def build_wcs_keywords(si: 'SessionInfo') -> dict:
    """Build a FITS-compatible WCS keyword dict from session info.

    Uses a gnomonic (TAN) projection centred on the image centre.
    Pixel coordinates are 1-based (FITS convention).
    Returns an empty dict if *si* lacks sufficient data.
    """
    if not si.has_wcs:
        return {}

    ra_deg = math.degrees(si.ra_rad)
    dec_deg = math.degrees(si.dec_rad)

    psx = math.degrees(si.fov_x_rad) / si.image_width   # deg/pixel
    psy = math.degrees(si.fov_y_rad) / si.image_height

    crpix1 = si.image_width / 2.0 + 0.5
    crpix2 = si.image_height / 2.0 + 0.5

    theta = si.orientation_rad if si.orientation_rad is not None else 0.0
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # TAN CD matrix for the Origin's FITS pixel grid: RA increases with +X and
    # Dec with +Y (determinant +psx*psy), rotated by theta:
    #   CD1_1 = psx cos  CD1_2 = -psy sin
    #   CD2_1 = psx sin  CD2_2 =  psy cos
    # Measured, not assumed: plate-solving raw Origin frames against Gaia
    # (src/local_solve.py) gives this orientation on every session tried (Crab,
    # Sunflower: centre within 3", position angle -theta). The previous form used
    # the textbook east-left CDELT1 = -psx, i.e. a mirrored sky -- correct at the
    # centre and ~4500" wrong at the corners.
    cd1_1 = psx * cos_t
    cd1_2 = -psy * sin_t
    cd2_1 = psx * sin_t
    cd2_2 = psy * cos_t

    return {
        'CTYPE1':   ('RA---TAN', 'WCS axis 1: RA, gnomonic projection'),
        'CTYPE2':   ('DEC--TAN', 'WCS axis 2: Dec, gnomonic projection'),
        'CRVAL1':   (ra_deg,     'RA at reference pixel (degrees)'),
        'CRVAL2':   (dec_deg,    'Dec at reference pixel (degrees)'),
        'CRPIX1':   (crpix1,     'X reference pixel (1-based)'),
        'CRPIX2':   (crpix2,     'Y reference pixel (1-based)'),
        'CD1_1':    (cd1_1,      'WCS CD matrix [1,1]'),
        'CD1_2':    (cd1_2,      'WCS CD matrix [1,2]'),
        'CD2_1':    (cd2_1,      'WCS CD matrix [2,1]'),
        'CD2_2':    (cd2_2,      'WCS CD matrix [2,2]'),
        'EQUINOX':  (2000.0,     'Equinox of coordinates (J2000)'),
        'ORIENTAT': (math.degrees(theta), 'Position angle of north (degrees)'),
        'WCSORIG': ('session_info', 'WCS source: session info.json'),
    }


def stack_wcs_keywords(si: 'SessionInfo', to_stack, shape_hw) -> dict:
    """``build_wcs_keywords`` carried onto a registered, cropped stack.

    The info.json solve describes the session's *first* sub (measured: its
    centre matches a Gaia solve of Light0001 to 3" and later subs drift away
    from it), while the stack lives on the registration reference's grid,
    cropped to the common region. *to_stack* maps native (row, col) of that
    first sub to stack (row, col) -- the registration transform minus the crop
    (``transparency.to_aligned_yx`` convention). The local Jacobian at the
    sub's centre carries the CD matrix across, so a rotation between that sub
    and the reference is honoured. Returns {} when *si* has no WCS.
    """
    kw = build_wcs_keywords(si)
    if not kw:
        return {}
    import numpy as np
    cx, cy = kw['CRPIX1'][0] - 1.0, kw['CRPIX2'][0] - 1.0
    d = 100.0
    src = np.array([[cy, cx], [cy, cx + d], [cy + d, cx]])
    dst = np.asarray(to_stack(src), dtype=np.float64)
    # J maps native (dx, dy) to stack (dx, dy)
    J = np.column_stack([(dst[1] - dst[0])[::-1] / d, (dst[2] - dst[0])[::-1] / d])
    cd = np.array([[kw['CD1_1'][0], kw['CD1_2'][0]], [kw['CD2_1'][0], kw['CD2_2'][0]]])
    cd_out = cd @ np.linalg.inv(J)
    out = dict(kw)
    out['CRPIX1'] = (float(dst[0][1]) + 1.0, 'X reference pixel (1-based)')
    out['CRPIX2'] = (float(dst[0][0]) + 1.0, 'Y reference pixel (1-based)')
    for k, v in (('CD1_1', cd_out[0, 0]), ('CD1_2', cd_out[0, 1]),
                 ('CD2_1', cd_out[1, 0]), ('CD2_2', cd_out[1, 1])):
        out[k] = (float(v), kw[k][1])
    out['WCSORIG'] = ('session_info', 'WCS source: session info.json, mapped to the stack')
    return out


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _str_or_none(val) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


def _float_or_none(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _int_or_none(val) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None
