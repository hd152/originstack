"""Tests for src/observing_geometry.py and utils.header_get_first."""
from __future__ import annotations

import math

import pytest

from src.observing_geometry import airmass, airmass_kasten_young, altaz, parallactic_angle_deg, zenith_angle_deg
from src.utils import header_get_first

# ---------------------------------------------------------------------------
# airmass_kasten_young (pure function, no astropy)
# ---------------------------------------------------------------------------

def test_airmass_at_zenith_is_one():
    assert airmass_kasten_young(90.0) == pytest.approx(1.0, abs=1e-3)


def test_airmass_grows_toward_horizon():
    a30 = airmass_kasten_young(60.0)   # 30 deg zenith angle
    a60 = airmass_kasten_young(30.0)   # 60 deg zenith angle
    assert 1.0 < a30 < a60
    assert a30 == pytest.approx(1.0 / math.cos(math.radians(30.0)), rel=0.02)


def test_airmass_none_below_horizon():
    assert airmass_kasten_young(2.0) is None
    assert airmass_kasten_young(-10.0) is None


# ---------------------------------------------------------------------------
# altaz / airmass / zenith / parallactic (need astropy)
# ---------------------------------------------------------------------------

_SITE = dict(lat_deg=40.0, lon_deg=-105.0, height_m=1600.0)


def _transit_time(ra_deg):
    """A UTC time near which ``ra_deg`` transits at the test site."""
    import astropy.units as u
    from astropy.coordinates import EarthLocation
    from astropy.time import Time
    loc = EarthLocation(lat=_SITE["lat_deg"] * u.deg, lon=_SITE["lon_deg"] * u.deg)
    t = Time("2026-03-20T00:00:00")
    for _ in range(48):
        lst = t.sidereal_time("apparent", longitude=loc.lon).deg
        if abs(((lst - ra_deg + 180.0) % 360.0) - 180.0) < 4.0:
            return t.isot
        t = t + 30 * u.min
    return None


def test_altaz_and_airmass_at_transit():
    ra, dec = 180.0, 40.0                       # dec == site latitude -> near zenith
    when = _transit_time(ra)
    assert when is not None
    aa = altaz(ra, dec, when_iso=when, **_SITE)
    assert aa is not None
    alt, _az = aa
    assert alt > 80.0                            # essentially overhead
    X = airmass(ra, dec, when_iso=when, **_SITE)
    assert X is not None and X == pytest.approx(1.0, abs=0.05)
    assert zenith_angle_deg(ra, dec, when_iso=when, **_SITE) == pytest.approx(90.0 - alt)


def test_parallactic_angle_zero_at_meridian():
    ra, dec = 180.0, 10.0
    when = _transit_time(ra)
    q = parallactic_angle_deg(ra, dec, _SITE["lat_deg"], _SITE["lon_deg"], when)
    assert q is not None
    assert abs(q) < 5.0                          # on the meridian, q ~ 0


def test_geometry_returns_none_on_garbage_time():
    assert altaz(180.0, 0.0, when_iso="not-a-time", **_SITE) is None
    assert airmass(180.0, 0.0, when_iso="not-a-time", **_SITE) is None


# ---------------------------------------------------------------------------
# header_get_first
# ---------------------------------------------------------------------------

def test_header_get_first_picks_first_present():
    h = {"GAIN": "1.5", "EGAIN": "2.0"}
    assert header_get_first(h, ("EGAIN", "GAIN"), cast=float) == 2.0
    assert header_get_first(h, ("MISSING", "GAIN"), cast=float) == 1.5


def test_header_get_first_skips_uncastable():
    h = {"A": "oops", "B": "3.0"}
    assert header_get_first(h, ("A", "B"), cast=float) == 3.0


def test_header_get_first_default_and_none_header():
    assert header_get_first({}, ("X",), default=7) == 7
    assert header_get_first(None, ("X",)) is None
    assert header_get_first({"X": None}, ("X",), default="d") == "d"
