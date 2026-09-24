"""Site + target -> sky-position geometry (alt/az, airmass, parallactic
angle) from an observation time.

One place for the "lat/long + RA/Dec + UTC -> where is it on the sky" math
that both the photometry airmass term and (optionally) the atmospheric
dispersion corrector need. astropy is imported lazily; every function
returns None rather than raising when inputs are missing or unparseable,
so callers can fall back cleanly.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from src.utils import parse_timestamp


def _as_utc_iso(when_iso: str) -> Optional[str]:
    """Normalise a timestamp to a UTC string astropy's Time will accept.

    ``astropy.time.Time`` rejects an ISO string carrying a numeric UTC offset
    (``2026-08-31T20:40:35-0700``), which is exactly the format a Celestron
    Origin ``info.json`` and its FITS headers use. Every function in this
    module then returned None on that data, silently dropping the airmass
    extinction term from --photometry and the auto-derived zenith angle from
    --fix-atmospheric-dispersion. Because they all fail soft by design,
    nothing errored -- the pipeline just quietly did less on the one camera
    this project was written for.
    """
    if not when_iso:
        return None
    dt = parse_timestamp(when_iso)
    if dt is None:
        return str(when_iso).strip()   # let astropy try it unchanged
    return dt.isoformat()


def _altaz_closed_form(ra_deg: float, dec_deg: float, lat_deg: float,
                       lon_deg: float,
                       when_iso: str) -> Optional[Tuple[float, float]]:
    """(alt, az) with no astropy and no IERS tables -- see ``altaz``."""
    try:
        from src.sky_model import altaz_from_equatorial, gmst_deg, julian_date
    except Exception:  # pragma: no cover - sky_model is not optional
        return None
    jd = julian_date(when_iso)
    if jd is None:
        return None
    lst = (gmst_deg(jd) + float(lon_deg)) % 360.0
    alt, az = altaz_from_equatorial(ra_deg, dec_deg, lat_deg, lst)
    return float(alt), float(az)


def altaz(ra_deg: float, dec_deg: float, lat_deg: float, lon_deg: float,
          height_m: float, when_iso: str) -> Optional[Tuple[float, float]]:
    """(altitude_deg, azimuth_deg) of the target at ``when_iso`` (UTC ISO),
    or None.

    astropy is tried first (it carries the target from J2000 to the equinox of
    date properly), but a closed-form fallback runs when astropy is absent or
    its IERS tables are -- which is the case inside the packaged app, where
    ``packaging/originstack.spec`` strips ``astropy_iers_data``. Without the
    fallback every caller here returned None in the frozen build: --photometry
    silently dropped its airmass extinction term and
    --fix-atmospheric-dispersion silently lost its auto-derived zenith angle.
    The fallback works in mean equinox of date against J2000 input, so it
    carries the ~0.35 deg precession offset documented in ``sky_model`` --
    irrelevant to airmass, which varies on degree scales.
    """
    try:
        import astropy.units as u
        from astropy.coordinates import AltAz, EarthLocation, SkyCoord
        from astropy.time import Time
    except Exception:
        return _altaz_closed_form(ra_deg, dec_deg, lat_deg, lon_deg, when_iso)
    try:
        loc = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg,
                            height=(height_m or 0.0) * u.m)
        when = _as_utc_iso(when_iso)
        if when is None:
            return None
        frame = AltAz(obstime=Time(when), location=loc)
        aa = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg).transform_to(frame)
        return float(aa.alt.deg), float(aa.az.deg)
    except Exception:
        return _altaz_closed_form(ra_deg, dec_deg, lat_deg, lon_deg, when_iso)


def airmass_kasten_young(alt_deg: float) -> Optional[float]:
    """Airmass from altitude via Kasten & Young (1989) -- finite and
    accurate toward the horizon, unlike a plain ``sec z``. None below ~3deg
    altitude (refraction model breaks down)."""
    if alt_deg is None or alt_deg <= 3.0:
        return None
    z = 90.0 - float(alt_deg)
    X = 1.0 / (math.cos(math.radians(z))
               + 0.50572 * (96.07995 - z) ** (-1.6364))
    if not math.isfinite(X) or X <= 0.9:
        return None
    # The additive term makes the formula bottom out a hair below 1.0 at the
    # exact zenith; clamp rather than reject.
    return max(float(X), 1.0)


def airmass(ra_deg: float, dec_deg: float, lat_deg: float, lon_deg: float,
            height_m: float, when_iso: str) -> Optional[float]:
    """Airmass of the target at the field centre for ``when_iso``, or None."""
    aa = altaz(ra_deg, dec_deg, lat_deg, lon_deg, height_m, when_iso)
    if aa is None:
        return None
    return airmass_kasten_young(aa[0])


def zenith_angle_deg(ra_deg: float, dec_deg: float, lat_deg: float,
                     lon_deg: float, height_m: float,
                     when_iso: str) -> Optional[float]:
    """Zenith angle (90 - altitude), or None -- also None when the target is
    at or below the horizon, which only happens with a wrong time or site
    (tan(z) changes sign past 90 deg, so a dispersion correction built on it
    would shift the channels the wrong way)."""
    aa = altaz(ra_deg, dec_deg, lat_deg, lon_deg, height_m, when_iso)
    if aa is None or aa[0] <= 0.0:
        return None
    return 90.0 - aa[0]


def parallactic_angle_deg(ra_deg: float, dec_deg: float, lat_deg: float,
                          lon_deg: float, when_iso: str) -> Optional[float]:
    """Astronomical parallactic angle q (degrees, measured from north
    towards east) -- the angle between the hour circle and the vertical
    circle through the target. NOTE: this is *not* yet the on-detector
    "toward zenith" direction; the caller must still add the image's
    north position angle. None on failure.

    Local sidereal time comes from ``sky_model.gmst_deg`` (closed form), not
    ``astropy.time.Time.sidereal_time``. That is deliberate and load-bearing:
    apparent sidereal time needs UT1-UTC from the IERS earth-orientation
    tables, which ``packaging/originstack.spec`` strips from the frozen build
    -- so the astropy path raised FileNotFoundError inside the packaged app,
    was swallowed by the ``except`` below, and returned None. Since
    ``cli._predict_rotation_spread`` calls this on the *default* path, the exe
    and a source checkout silently produced different stacks from the same
    directory. The closed form has no such dependency.

    Mean rather than apparent sidereal time costs at most the equation of the
    equinoxes (~18 arcsec), far below anything either caller resolves.
    """
    try:
        from src.sky_model import gmst_deg, julian_date
    except Exception:  # pragma: no cover - sky_model is not optional
        return None
    try:
        jd = julian_date(when_iso)
        if jd is None:
            return None
        lst = (gmst_deg(jd) + float(lon_deg)) % 360.0
        ha = math.radians((lst - ra_deg + 180.0) % 360.0 - 180.0)  # [-pi, pi]
        dec = math.radians(dec_deg)
        phi = math.radians(lat_deg)
        q = math.atan2(math.sin(ha),
                       math.tan(phi) * math.cos(dec) - math.sin(dec) * math.cos(ha))
        return float(math.degrees(q))
    except Exception:
        return None
