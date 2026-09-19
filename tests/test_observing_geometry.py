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


# ---------------------------------------------------------------------------
# UTC-offset timestamps (Celestron Origin)
# ---------------------------------------------------------------------------

_ORIGIN_LOCAL = "2026-08-31T20:40:35-0700"      # as an Origin header writes it
_SAME_UTC = "2026-09-01T03:40:35"               # the identical instant
_TZ_SITE = (33.83, -117.79)
_TZ_TARGET = (271.0, -24.36)                        # the Lagoon


def test_offset_timestamps_agree_with_their_utc_equivalent():
    """Regression: every function here returned None on real Origin data.

    ``astropy.time.Time`` rejects an ISO string with a numeric UTC offset,
    which is exactly the format a Celestron Origin writes to DATE-OBS and
    info.json. Because this module fails soft by design, nothing errored --
    it silently dropped the airmass extinction term from --photometry (a real
    photometric error, absorbed into the zero point) and the auto-derived
    zenith angle for --fix-atmospheric-dispersion, on the one camera this
    project was written for.
    """
    local = altaz(*_TZ_TARGET, *_TZ_SITE, 0.0, _ORIGIN_LOCAL)
    utc = altaz(*_TZ_TARGET, *_TZ_SITE, 0.0, _SAME_UTC)

    assert local is not None, "an Origin timestamp must not return None"
    assert utc is not None
    assert local[0] == pytest.approx(utc[0], abs=1e-9)
    assert local[1] == pytest.approx(utc[1], abs=1e-9)


@pytest.mark.parametrize("spelling", [
    "2026-09-01T03:40:35",          # naive, already UTC
    "2026-09-01T03:40:35Z",
    "2026-09-01T03:40:35+00:00",
    "2026-08-31T20:40:35-0700",     # Origin: offset without a colon
    "2026-08-31T20:40:35-07:00",    # offset with a colon
])
def test_every_spelling_of_one_instant_gives_the_same_geometry(spelling):
    reference = airmass(*_TZ_TARGET, *_TZ_SITE, 0.0, _SAME_UTC)
    assert reference is not None
    assert airmass(*_TZ_TARGET, *_TZ_SITE, 0.0, spelling) == pytest.approx(reference, rel=1e-9)


def test_offset_is_converted_not_discarded():
    """Stripping '-0700' instead of converting it is seven hours of error."""
    honoured = altaz(*_TZ_TARGET, *_TZ_SITE, 0.0, _ORIGIN_LOCAL)
    stripped = altaz(*_TZ_TARGET, *_TZ_SITE, 0.0, "2026-08-31T20:40:35")

    assert honoured is not None and stripped is not None
    # Seven hours of Earth rotation is ~105 degrees of hour angle; the two
    # readings must not be mistaken for each other.
    assert abs(honoured[0] - stripped[0]) > 5.0


def test_parallactic_angle_accepts_an_offset_timestamp():
    assert parallactic_angle_deg(*_TZ_TARGET, *_TZ_SITE, _ORIGIN_LOCAL) == pytest.approx(
        parallactic_angle_deg(*_TZ_TARGET, *_TZ_SITE, _SAME_UTC), abs=1e-9)


def test_zenith_angle_accepts_an_offset_timestamp():
    assert zenith_angle_deg(*_TZ_TARGET, *_TZ_SITE, 0.0, _ORIGIN_LOCAL) == pytest.approx(
        zenith_angle_deg(*_TZ_TARGET, *_TZ_SITE, 0.0, _SAME_UTC), abs=1e-9)


# ---------------------------------------------------------------------------
# No dependency on astropy's IERS tables
# ---------------------------------------------------------------------------
#
# packaging/originstack.spec strips astropy_iers_data from the frozen app. Any
# call that needs UT1-UTC then raises FileNotFoundError('finals2000A.all'),
# which every function here swallows (fail-soft by design) and turns into None.
# --photometry's airmass term, --fix-atmospheric-dispersion's zenith angle and
# the default-path rotation prediction all silently vanished in the exe while
# working in a source checkout.

def _astropy_reference_lst_parallactic(ra, dec, lat, lon, when_iso):
    """Parallactic angle from astropy's *apparent* sidereal time -- the ground
    truth the closed form is compared against. Skips when astropy cannot
    evaluate that date offline (which is itself the failure being tested for)."""
    pytest.importorskip("astropy")
    import astropy.units as u
    from astropy.coordinates import EarthLocation
    from astropy.time import Time
    from astropy.utils.iers import conf

    from src.observing_geometry import _as_utc_iso
    try:
        with conf.set_temp("auto_max_age", None), conf.set_temp("auto_download", False):
            loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg)
            lst = Time(_as_utc_iso(when_iso), location=loc).sidereal_time("apparent").deg
    except Exception as exc:                                  # pragma: no cover
        pytest.skip(f"astropy cannot evaluate this date offline: {exc}")
    ha = math.radians((lst - ra + 180.0) % 360.0 - 180.0)
    d, phi = math.radians(dec), math.radians(lat)
    return math.degrees(math.atan2(math.sin(ha),
                                   math.tan(phi) * math.cos(d) - math.sin(d) * math.cos(ha)))


@pytest.mark.parametrize("when,ra,dec", [
    ("2026-08-31T20:40:32-0700", 270.9, -24.4),
    ("2026-03-01T02:00:00", 83.8, -5.4),
    ("2026-12-15T23:10:00+0000", 10.7, 41.3),
    ("2026-06-02T05:31:00-0700", 202.5, 47.2),
])
def test_closed_form_parallactic_angle_matches_astropy(when, ra, dec):
    """Mean vs apparent sidereal time differs by the equation of the equinoxes
    (~18 arcsec); measured worst case over these was 0.0024 deg. The consumers
    resolve degrees (the rotation split threshold is 3), so 0.01 is generous."""
    lat, lon = 37.77, -122.42
    closed = parallactic_angle_deg(ra, dec, lat, lon, when)
    truth = _astropy_reference_lst_parallactic(ra, dec, lat, lon, when)
    assert closed is not None
    assert abs((closed - truth + 180.0) % 360.0 - 180.0) < 0.01


def test_parallactic_angle_does_not_call_astropys_sidereal_time(monkeypatch):
    pytest.importorskip("astropy")
    from astropy.time import Time

    def boom(*a, **k):
        raise FileNotFoundError("finals2000A.all")

    monkeypatch.setattr(Time, "sidereal_time", boom)
    assert parallactic_angle_deg(270.9, -24.4, 33.83, -117.79, _ORIGIN_LOCAL) is not None


def test_altaz_falls_back_to_closed_form_when_astropy_is_missing(monkeypatch):
    """Without the fallback this returned None in the frozen build."""
    pytest.importorskip("astropy")
    import sys

    with_astropy = altaz(*_TZ_TARGET, *_TZ_SITE, 0.0, _ORIGIN_LOCAL)
    assert with_astropy is not None

    monkeypatch.setitem(sys.modules, "astropy.coordinates", None)   # import raises
    fallback = altaz(*_TZ_TARGET, *_TZ_SITE, 0.0, _ORIGIN_LOCAL)

    assert fallback is not None, "must not return None just because astropy is absent"
    # Mean-equinox-of-date against J2000 input carries the documented ~0.35 deg
    # precession offset; irrelevant to airmass, so allow a degree.
    assert fallback[0] == pytest.approx(with_astropy[0], abs=1.0)
    assert fallback[1] == pytest.approx(with_astropy[1], abs=1.0)


def test_altaz_falls_back_when_the_astropy_transform_itself_fails(monkeypatch):
    """The IERS failure does not surface as an ImportError: astropy imports
    fine and then raises inside the transform."""
    pytest.importorskip("astropy")
    from astropy.coordinates import SkyCoord

    def boom(*a, **k):
        raise FileNotFoundError("finals2000A.all")

    monkeypatch.setattr(SkyCoord, "transform_to", boom)
    assert altaz(*_TZ_TARGET, *_TZ_SITE, 0.0, _ORIGIN_LOCAL) is not None
    assert airmass(*_TZ_TARGET, *_TZ_SITE, 0.0, _ORIGIN_LOCAL) is not None


def test_garbage_time_still_returns_none_from_the_fallback(monkeypatch):
    pytest.importorskip("astropy")
    import sys

    monkeypatch.setitem(sys.modules, "astropy.coordinates", None)
    assert altaz(180.0, 0.0, when_iso="not-a-time", **_SITE) is None


# ---------------------------------------------------------------------------
# parse_timestamp (shared by observing_geometry and sky_model)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "2026-09-01T03:40:35", "2026-09-01T03:40:35Z", "2026-09-01T03:40:35+00:00",
    "2026-08-31T20:40:35-0700", "2026-08-31T20:40:35-07:00",
    "2026-09-01 03:40:35",
])
def test_parse_timestamp_normalises_every_spelling_to_naive_utc(text):
    from src.utils import parse_timestamp
    dt = parse_timestamp(text)
    assert dt is not None and dt.tzinfo is None
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second) == (2026, 9, 1, 3, 40, 35)


@pytest.mark.parametrize("bad", ["", None, "not a timestamp", "0-00-00T00:00:00"])
def test_parse_timestamp_returns_none_for_garbage(bad):
    from src.utils import parse_timestamp
    assert parse_timestamp(bad) is None
