"""Origin lights stamp DATE-OBS in local time with the offset in a separate
TIMEZONE keyword. Reading DATE-OBS alone (parse_timestamp treats an
offset-less time as UTC) put every time-dependent result seven hours out:
a --fix-atmospheric-dispersion zenith angle of 150 deg (below the horizon,
so the correction went the wrong way) and light-curve MJDs/airmasses off."""
import types

import pytest

from src.observing_geometry import zenith_angle_deg
from src.photometry_timeseries import _frame_time_iso, _to_mjd
from src.utils import obs_time_utc_iso


@pytest.mark.parametrize('hdr, expected', [
    ({'DATE-OBS': '2026-08-31T20:40:32', 'TIMEZONE': '-0700'}, '2026-09-01T03:40:32'),
    ({'DATE-OBS': '2026-08-31T20:40:32', 'TIMEZONE': '-07:00'}, '2026-09-01T03:40:32'),
    ({'DATE-OBS': '2026-08-31T20:40:32-0700', 'TIMEZONE': '-0700'}, '2026-09-01T03:40:32'),  # not twice
    ({'DATE-OBS': '2026-08-31T20:40:32Z', 'TIMEZONE': '-0700'}, '2026-08-31T20:40:32'),
    ({'DATE-OBS': '2026-08-31T20:40:32', 'TIMEZONE': 'PDT'}, '2026-08-31T20:40:32'),        # unusable
    ({'DATE-OBS': '2026-08-31T20:40:32'}, '2026-08-31T20:40:32'),
    ({'DATE-OBS': '2026-08-31', 'TIMEZONE': '-0700'}, '2026-08-31T00:00:00'),
])
def test_obs_time_applies_timezone(hdr, expected):
    assert obs_time_utc_iso(hdr) == expected


def test_obs_time_fallback_is_normalised_for_astropy():
    iso = obs_time_utc_iso({}, fallback='2026-08-31T20:40:32-0700')
    assert iso == '2026-09-01T03:40:32'
    assert _to_mjd(iso) == pytest.approx(61284.15315, abs=1e-4)


def test_frame_time_uses_timezone_and_session_fallback():
    frame = types.SimpleNamespace(header={'DATE-OBS': '2026-08-31T20:40:32', 'TIMEZONE': '-0700'})
    assert _frame_time_iso(frame, None, 0, 1) == '2026-09-01T03:40:32'
    # No per-frame DATE-OBS: interpolate from info.json's offset-bearing start.
    # This used to hand '...-0700' to astropy Time, fail, and blank every MJD.
    si = types.SimpleNamespace(date_time='2026-08-31T20:40:32-0700', total_duration_ms=600_000)
    iso = _frame_time_iso(types.SimpleNamespace(header={}), si, 1, 2)
    assert iso.startswith('2026-09-01T03:50:32')
    assert _to_mjd(iso) == _to_mjd(iso)   # not NaN


def test_zenith_angle_is_none_below_the_horizon():
    # Antares from 34N, 118W: up in the evening (local), down 7 h later.
    ra, dec, lat, lon = 247.35, -26.43, 34.0, -118.0
    assert zenith_angle_deg(ra, dec, lat, lon, 0.0, '2026-06-15T05:30:00') is not None
    assert zenith_angle_deg(ra, dec, lat, lon, 0.0, '2026-06-15T15:30:00') is None


def test_stacked_header_keeps_timezone():
    from astropy.io import fits

    from src.cli import parse_args
    from src.io_fits import populate_fits_header
    from src.models import FrameInfo, ProcessingStats
    f = FrameInfo(path='a.fits', type='light',
                  header={'DATE-OBS': '2026-08-31T20:40:32', 'TIMEZONE': '-0700', 'EXPTIME': 10.0})
    hdr = fits.Header()
    populate_fits_header(hdr, [f], ProcessingStats(),
                         parse_args(['-d', 'x', '-o', 'y.fits', '--no-auto']), (3, 8, 8),
                         [(0.0, 0.0)], {'bias': None, 'dark': None, 'flat': None})
    assert obs_time_utc_iso(hdr) == '2026-09-01T03:40:32'
