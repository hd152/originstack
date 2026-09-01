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


def altaz(ra_deg: float, dec_deg: float, lat_deg: float, lon_deg: float,
          height_m: float, when_iso: str) -> Optional[Tuple[float, float]]:
    """(altitude_deg, azimuth_deg) of the target at ``when_iso`` (UTC ISO),
    or None."""
    try:
        import astropy.units as u
        from astropy.coordinates import AltAz, EarthLocation, SkyCoord
        from astropy.time import Time
    except Exception:
        return None
    try:
        loc = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg,
                            height=(height_m or 0.0) * u.m)
        frame = AltAz(obstime=Time(str(when_iso)), location=loc)
        aa = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg).transform_to(frame)
        return float(aa.alt.deg), float(aa.az.deg)
    except Exception:
        return None


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
    """Zenith angle (90 - altitude), or None."""
    aa = altaz(ra_deg, dec_deg, lat_deg, lon_deg, height_m, when_iso)
    return None if aa is None else 90.0 - aa[0]


def parallactic_angle_deg(ra_deg: float, dec_deg: float, lat_deg: float,
                          lon_deg: float, when_iso: str) -> Optional[float]:
    """Astronomical parallactic angle q (degrees, measured from north
    towards east) -- the angle between the hour circle and the vertical
    circle through the target. NOTE: this is *not* yet the on-detector
    "toward zenith" direction; the caller must still add the image's
    north position angle. None on failure.
    """
    try:
        import astropy.units as u
        import numpy as np
        from astropy.coordinates import EarthLocation
        from astropy.time import Time
    except Exception:
        return None
    try:
        loc = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg)
        lst = Time(str(when_iso), location=loc).sidereal_time("apparent").deg
        ha = math.radians((lst - ra_deg + 180.0) % 360.0 - 180.0)  # [-pi, pi]
        dec = math.radians(dec_deg)
        phi = math.radians(lat_deg)
        q = math.atan2(math.sin(ha),
                       math.tan(phi) * math.cos(dec) - math.sin(dec) * math.cos(ha))
        return float(np.degrees(q))
    except Exception:
        return None
