"""Physics-based sky background model (``--bg-method physical``).

Every background extractor in this pipeline -- and in every other stacker --
is *blind*: mesh, DBE and the starlet method all fit a free-form surface to
the image and call whatever they fit "the background". A free-form surface
with hundreds of degrees of freedom cannot distinguish a light-pollution
gradient from a frame-filling nebula, so it subtracts both. That failure is
not hypothetical: it removed over half the nebulosity from a real Lagoon
session, and the pipeline's current mitigation is a heuristic in
``auto_settings.py`` that skips the sky-residual passes for extended targets
-- treating the symptom.

This module models the sky from first principles instead. The night sky's
brightness at a given pixel is a sum of physically distinct components whose
*spatial shape* is fixed by geometry (where the moon is, where the zenith is,
where the ecliptic runs), leaving only a scalar amplitude per component free:

    sky(x, y) = c0
              + c_air  * airglow(z(x, y))
              + c_moon * moonlight(rho(x, y), z(x, y))
              + c_zodi * zodiacal(lambda(x, y), beta(x, y))
              + c_lp   * light_pollution(az(x, y), z(x, y))

Five or six free parameters against a geometry-determined basis. A nebula is
not a member of that basis and, unlike a mesh, the model has nowhere to put
one -- it structurally cannot absorb astrophysical signal. It also explains
itself: the fitted coefficients say *what* the gradient was made of ("62% of
your gradient is moonglow"), which no surface fit can.

Ephemerides are computed here in closed form rather than via astropy:
``EarthLocation``/``AltAz`` transforms pull in IERS earth-orientation tables
that [packaging/originstack.spec](packaging/originstack.spec) deliberately
excludes from the packaged app, and sky-brightness modelling needs
degree-level accuracy, not milliarcseconds. Same reasoning that put
``net_query.py`` (vs astroquery) and ``wavelet.py`` (vs pywt) in this
codebase. Sun position is Meeus' low-precision solar formula; moon position is
the standard truncated lunar series with twelve longitude and five latitude
perturbation terms. Both were validated against astropy's own ephemeris
(``tests/test_sky_model.py``): **0.009 deg for the sun, 0.05 deg for the
moon** over 2024-2027.

Frame convention: these return coordinates of the **mean equinox of date**,
which is the frame hour angle and alt/az are defined in. WCS pixel
coordinates are J2000, so pixel-to-moon separations carry the accumulated
precession offset (~0.35 deg in 2026) -- deliberately not corrected, because
every quantity here varies on 10-degree scales and a third of a degree moves
the Krisciunas-Schaefer scattering function by well under a percent. That
0.35 deg is also exactly the trap this module's validation caught twice: it
is what a J2000-vs-of-date comparison looks like, and it must not be mistaken
for an ephemeris error (the real one, found the same way, was 19.6 deg --
Schlyter's lunar elements are epoched at 1999-12-31.0, not J2000.0).

References:
  Krisciunas & Schaefer (1991), PASP 103, 1033 -- moonlight scattering model.
  van Rhijn (1921) -- airglow layer path-length enhancement.
  Meeus, *Astronomical Algorithms*, 2nd ed., ch. 25 (sun), ch. 47 (moon).
  Kasten & Young (1989) -- airmass (via ``observing_geometry``).
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_log = logging.getLogger("originstack")

_DEG = math.pi / 180.0
_J2000 = 2451545.0


# ---------------------------------------------------------------------------
# Time and frame conversions (closed form, no IERS)
# ---------------------------------------------------------------------------

def julian_date(when: str) -> Optional[float]:
    """Julian Date from an ISO-8601 timestamp, or None if unparseable.

    **Timezone offsets are honoured, not stripped.** A Celestron Origin
    ``info.json`` records local time with an offset
    (``2026-08-31T20:40:32-0700``), and the naive readings of that string are
    both wrong in ways that are invisible downstream: failing to parse it at
    all silently disables the whole physical sky model on exactly the data it
    was written for, and discarding the ``-0700`` puts the timestamp seven
    hours out, which moves the moon most of the way across the sky. Offsets
    are converted to UTC; a naive timestamp is assumed to already be UTC.
    """
    if not when:
        return None
    text = str(when).strip()

    dt = None
    # fromisoformat covers 'Z', '+HH:MM' and (3.11+) '+HHMM' in one shot.
    try:
        dt = _dt.datetime.fromisoformat(text.replace('Z', '+00:00'))
    except ValueError:
        naive = text.replace('Z', '').replace('T', ' ')
        for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%d %H:%M', '%Y-%m-%d'):
            try:
                dt = _dt.datetime.strptime(naive, fmt)
                break
            except ValueError:
                continue
    if dt is None:
        return None

    if dt.tzinfo is not None:
        dt = dt.astimezone(_dt.timezone.utc).replace(tzinfo=None)

    y, m = dt.year, dt.month
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4          # Gregorian calendar correction
    day_frac = (dt.hour + dt.minute / 60.0
                + (dt.second + dt.microsecond * 1e-6) / 3600.0) / 24.0
    return (math.floor(365.25 * (y + 4716))
            + math.floor(30.6001 * (m + 1))
            + dt.day + day_frac + b - 1524.5)


def gmst_deg(jd: float) -> float:
    """Greenwich mean sidereal time in degrees."""
    return (280.46061837 + 360.98564736629 * (jd - _J2000)) % 360.0


def obliquity_deg(jd: float) -> float:
    """Mean obliquity of the ecliptic."""
    return 23.4392911 - 3.563e-7 * (jd - _J2000)


def ecliptic_to_equatorial(lon_deg: float, lat_deg: float,
                           eps_deg: float) -> Tuple[float, float]:
    """Ecliptic (lon, lat) -> equatorial (RA, Dec), all degrees."""
    lon, lat, eps = lon_deg * _DEG, lat_deg * _DEG, eps_deg * _DEG
    sin_dec = (math.sin(lat) * math.cos(eps)
               + math.cos(lat) * math.sin(eps) * math.sin(lon))
    dec = math.asin(max(-1.0, min(1.0, sin_dec)))
    ra = math.atan2(math.sin(lon) * math.cos(eps) - math.tan(lat) * math.sin(eps),
                    math.cos(lon))
    return (math.degrees(ra) % 360.0, math.degrees(dec))


def equatorial_to_ecliptic(ra_deg, dec_deg, eps_deg):
    """Equatorial (RA, Dec) -> ecliptic (lon, lat). Array-safe."""
    ra = np.asarray(ra_deg, dtype=np.float64) * _DEG
    dec = np.asarray(dec_deg, dtype=np.float64) * _DEG
    eps = eps_deg * _DEG
    sin_lat = np.sin(dec) * np.cos(eps) - np.cos(dec) * np.sin(eps) * np.sin(ra)
    lat = np.arcsin(np.clip(sin_lat, -1.0, 1.0))
    lon = np.arctan2(np.sin(ra) * np.cos(eps) + np.tan(dec) * np.sin(eps),
                     np.cos(ra))
    return np.degrees(lon) % 360.0, np.degrees(lat)


def altaz_from_equatorial(ra_deg, dec_deg, lat_deg: float, lst_deg: float):
    """Equatorial -> (altitude, azimuth) in degrees. Array-safe.

    Azimuth is measured from north through east, matching the convention in
    ``observing_geometry.altaz``.
    """
    ra = np.asarray(ra_deg, dtype=np.float64)
    dec = np.asarray(dec_deg, dtype=np.float64) * _DEG
    ha = (lst_deg - ra) * _DEG
    phi = lat_deg * _DEG

    sin_alt = np.sin(dec) * np.sin(phi) + np.cos(dec) * np.cos(phi) * np.cos(ha)
    alt = np.arcsin(np.clip(sin_alt, -1.0, 1.0))
    az = np.arctan2(-np.sin(ha) * np.cos(dec),
                    np.sin(dec) * np.cos(phi) - np.cos(dec) * np.sin(phi) * np.cos(ha))
    return np.degrees(alt), np.degrees(az) % 360.0


def angular_separation_deg(ra1, dec1, ra2_deg: float, dec2_deg: float):
    """Great-circle separation, degrees. Array-safe in the first pair.

    Uses the haversine form rather than the ``acos`` dot product, which loses
    precision catastrophically at the small separations that matter most here
    (the moon a few degrees off the field).
    """
    ra1 = np.asarray(ra1, dtype=np.float64) * _DEG
    dec1 = np.asarray(dec1, dtype=np.float64) * _DEG
    ra2, dec2 = ra2_deg * _DEG, dec2_deg * _DEG
    d_ra, d_dec = ra1 - ra2, dec1 - dec2
    h = (np.sin(d_dec / 2.0) ** 2
         + np.cos(dec1) * np.cos(dec2) * np.sin(d_ra / 2.0) ** 2)
    return np.degrees(2.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0))))


# ---------------------------------------------------------------------------
# Sun and moon ephemerides
# ---------------------------------------------------------------------------

def sun_position(jd: float) -> Tuple[float, float, float]:
    """(RA, Dec, ecliptic longitude) of the sun, degrees, mean equinox of date.

    Meeus ch. 25 low-precision formula; measured against astropy at 0.009 deg
    over 2024-2027.
    """
    n = jd - _J2000
    mean_lon = (280.460 + 0.9856474 * n) % 360.0
    mean_anom = ((357.528 + 0.9856003 * n) % 360.0) * _DEG
    ecl_lon = (mean_lon + 1.915 * math.sin(mean_anom)
               + 0.020 * math.sin(2.0 * mean_anom)) % 360.0
    ra, dec = ecliptic_to_equatorial(ecl_lon, 0.0, obliquity_deg(jd))
    return ra, dec, ecl_lon


def moon_position(jd: float) -> Tuple[float, float, float, float]:
    """(RA, Dec, ecliptic longitude, ecliptic latitude) of the moon, degrees,
    mean equinox of date.

    Truncated lunar series with twelve longitude and five latitude
    perturbation terms; measured against astropy at 0.05 deg over 2024-2027,
    which shifts the moon-field separation by the same amount -- negligible
    against the Krisciunas-Schaefer scattering function, which varies slowly
    in ``rho``.
    """
    # These lunar elements are epoched at 1999-12-31.0 TDT (JD 2451543.5), NOT
    # J2000.0. Using the J2000 day number here puts the mean anomaly 1.5 days
    # x 13.065 deg/day ~= 19.6 deg ahead, which lands the moon roughly two
    # zodiac signs away -- caught by validating against astropy's ephemeris.
    d = jd - 2451543.5

    # Orbital elements
    node = (125.1228 - 0.0529538083 * d) * _DEG     # ascending node
    incl = 5.1454 * _DEG
    argp = (318.0634 + 0.1643573223 * d) * _DEG     # argument of perigee
    ecc = 0.054900
    mean_anom = ((115.3654 + 13.0649929509 * d) % 360.0) * _DEG

    # Kepler, iterated -- the moon's eccentricity is small so this converges
    # in a couple of passes.
    ecc_anom = mean_anom + ecc * math.sin(mean_anom) * (1.0 + ecc * math.cos(mean_anom))
    for _ in range(6):
        delta = ((ecc_anom - ecc * math.sin(ecc_anom) - mean_anom)
                 / (1.0 - ecc * math.cos(ecc_anom)))
        ecc_anom -= delta
        if abs(delta) < 1e-10:
            break

    xv = math.cos(ecc_anom) - ecc
    yv = math.sqrt(1.0 - ecc * ecc) * math.sin(ecc_anom)
    true_anom = math.atan2(yv, xv)
    dist = math.hypot(xv, yv)                        # earth radii (a = 1 here)

    # Position in the ecliptic frame
    u_arg = true_anom + argp
    xh = dist * (math.cos(node) * math.cos(u_arg)
                 - math.sin(node) * math.sin(u_arg) * math.cos(incl))
    yh = dist * (math.sin(node) * math.cos(u_arg)
                 + math.cos(node) * math.sin(u_arg) * math.cos(incl))
    zh = dist * math.sin(u_arg) * math.sin(incl)

    lon = math.degrees(math.atan2(yh, xh))
    lat = math.degrees(math.atan2(zh, math.hypot(xh, yh)))

    # Perturbations. The solar elements here share the lunar epoch above --
    # mixing epochs between the two bodies would reintroduce the same class of
    # offset the day number just fixed.
    sun_mean_anom = ((356.0470 + 0.9856002585 * d) % 360.0) * _DEG
    sun_arg_peri = 282.9404 + 4.70935e-5 * d
    sun_mean_lon = sun_arg_peri + math.degrees(sun_mean_anom)

    moon_mean_lon = math.degrees(node + argp + mean_anom)
    elong = (moon_mean_lon - sun_mean_lon) * _DEG          # mean elongation D
    arg_lat = math.radians(moon_mean_lon) - node           # argument of latitude F
    ma, sa, dd, ff = mean_anom, sun_mean_anom, elong, arg_lat

    lon += (-1.274 * math.sin(ma - 2 * dd)        # evection
            + 0.658 * math.sin(2 * dd)            # variation
            - 0.186 * math.sin(sa)                # yearly equation
            - 0.059 * math.sin(2 * ma - 2 * dd)
            - 0.057 * math.sin(ma - 2 * dd + sa)
            + 0.053 * math.sin(ma + 2 * dd)
            + 0.046 * math.sin(2 * dd - sa)
            + 0.041 * math.sin(ma - sa)
            - 0.035 * math.sin(dd)                # parallactic equation
            - 0.031 * math.sin(ma + sa)
            - 0.015 * math.sin(2 * ff - 2 * dd)
            + 0.011 * math.sin(ma - 4 * dd))
    lat += (-0.173 * math.sin(ff - 2 * dd)
            - 0.055 * math.sin(ma - ff - 2 * dd)
            - 0.046 * math.sin(ma + ff - 2 * dd)
            + 0.033 * math.sin(ff + 2 * dd)
            + 0.017 * math.sin(2 * ma + ff))

    lon %= 360.0
    ra, dec = ecliptic_to_equatorial(lon, lat, obliquity_deg(jd))
    return ra, dec, lon, lat


def moon_phase_angle_deg(jd: float) -> float:
    """Sun-moon elongation as seen from earth, i.e. the phase angle ``alpha``
    used by the Krisciunas-Schaefer brightness model. 0 = full, 180 = new."""
    _, _, sun_lon = sun_position(jd)
    _, _, moon_lon, moon_lat = moon_position(jd)
    elong = angular_separation_deg(
        *ecliptic_to_equatorial(moon_lon, moon_lat, obliquity_deg(jd)),
        *ecliptic_to_equatorial(sun_lon, 0.0, obliquity_deg(jd)))
    return float(180.0 - float(elong))


def moon_illuminated_fraction(jd: float) -> float:
    """Fraction of the lunar disc illuminated, 0 (new) to 1 (full)."""
    return float((1.0 + math.cos(moon_phase_angle_deg(jd) * _DEG)) / 2.0)


# ---------------------------------------------------------------------------
# Sky brightness components (relative shape, not absolute calibration)
# ---------------------------------------------------------------------------

def van_rhijn(zenith_angle_deg_arr, layer_km: float = 90.0,
              earth_radius_km: float = 6378.0):
    """Airglow path-length enhancement toward the horizon (van Rhijn 1921).

    The emitting layer sits at ~90 km; a line of sight at zenith angle ``z``
    crosses more of it than one at the zenith by
    ``1/sqrt(1 - (R/(R+h))^2 sin^2 z)``. This is why the sky is genuinely
    brighter near the horizon even from a dark site -- a real, geometry-fixed
    gradient that blind surface fitters happily confuse with light pollution.
    """
    z = np.asarray(zenith_angle_deg_arr, dtype=np.float64) * _DEG
    ratio = earth_radius_km / (earth_radius_km + layer_km)
    denom = 1.0 - (ratio * np.sin(z)) ** 2
    return 1.0 / np.sqrt(np.clip(denom, 1e-6, None))


def moonlight_brightness(rho_deg, moon_alt_deg: float, target_zenith_deg,
                         phase_angle_deg: float, k_extinction: float = 0.172):
    """Krisciunas & Schaefer (1991) scattered-moonlight surface brightness.

    Returns brightness in nanolamberts (their ``B_moon``); only the *relative*
    spatial shape is used by the fit, so the absolute scale and the caller's
    ADU units never have to be reconciled.

    Args:
        rho_deg: Angular separation between the moon and each pixel.
        moon_alt_deg: Moon altitude. At or below the horizon there is no
            scattered moonlight and this returns zeros.
        target_zenith_deg: Per-pixel zenith angle.
        phase_angle_deg: ``alpha``, 0 at full moon.
        k_extinction: Atmospheric extinction coefficient (mag/airmass); 0.172
            is the Krisciunas-Schaefer V-band default.
    """
    rho = np.asarray(rho_deg, dtype=np.float64)
    if moon_alt_deg is None or moon_alt_deg <= 0.0:
        return np.zeros_like(rho)

    alpha = abs(float(phase_angle_deg))
    # Illuminance of the moon outside the atmosphere.
    illuminance = 10.0 ** (-0.4 * (3.84 + 0.026 * alpha + 4.0e-9 * alpha ** 4))

    # Scattering function: Rayleigh term + aerosol/Mie term.
    rho_r = rho * _DEG
    scatter = (10.0 ** 5.36 * (1.06 + np.cos(rho_r) ** 2)
               + 10.0 ** (6.15 - rho / 40.0))

    x_moon = _airmass_array(90.0 - float(moon_alt_deg))
    x_target = _airmass_array(target_zenith_deg)

    return (scatter * illuminance
            * 10.0 ** (-0.4 * k_extinction * x_moon)
            * (1.0 - 10.0 ** (-0.4 * k_extinction * x_target)))


def _airmass_array(zenith_angle_deg_arr):
    """Kasten & Young (1989) airmass, vectorised and horizon-safe."""
    z = np.clip(np.asarray(zenith_angle_deg_arr, dtype=np.float64), 0.0, 89.0)
    return 1.0 / (np.cos(z * _DEG) + 0.50572 * (96.07995 - z) ** -1.6364)


def zodiacal_brightness(helio_ecliptic_lon, ecliptic_lat):
    """Coarse zodiacal-light brightness in relative units.

    A smooth analytic stand-in for the tabulated Levasseur-Regourd surface:
    the zodiacal cloud brightens strongly toward the sun (small helio-ecliptic
    longitude) and toward the ecliptic plane. Treated as a *shape* only -- the
    fit's scalar amplitude absorbs the normalisation -- so what matters is
    that the gradient runs along the ecliptic, which this captures and a blind
    surface fit has no way to know about.
    """
    lon = np.asarray(helio_ecliptic_lon, dtype=np.float64) % 360.0
    lon = np.where(lon > 180.0, 360.0 - lon, lon)     # fold to [0, 180]
    lat = np.abs(np.asarray(ecliptic_lat, dtype=np.float64))

    lon_term = 1.0 + 8.0 * np.exp(-lon / 35.0)        # strong toward the sun
    lat_term = np.exp(-lat / 22.0)                    # concentrated to the plane
    return lon_term * lat_term


def light_pollution_brightness(az_deg, zenith_angle_deg_arr,
                               source_az_deg: float = 0.0):
    """Ground-source skyglow: brighter low, and brighter toward the source.

    A dome of scattered light centred on a town has two robust features -- it
    falls off with altitude, and it is asymmetric in azimuth toward the
    source. The azimuth of the dominant source is fitted (see
    ``fit_sky_model``) rather than assumed, because the observer rarely knows
    it in usable form.
    """
    z = np.asarray(zenith_angle_deg_arr, dtype=np.float64)
    az = np.asarray(az_deg, dtype=np.float64)
    horizon_term = _airmass_array(z)
    delta_az = (az - float(source_az_deg)) * _DEG
    return horizon_term * (1.0 + 0.5 * np.cos(delta_az))


# ---------------------------------------------------------------------------
# Basis construction and fitting
# ---------------------------------------------------------------------------

class SkyGeometry:
    """Per-pixel observing geometry for one frame."""

    def __init__(self, zenith_angle, azimuth, moon_sep, moon_alt,
                 helio_lon, ecl_lat, jd, phase_angle):
        self.zenith_angle = zenith_angle
        self.azimuth = azimuth
        self.moon_sep = moon_sep
        self.moon_alt = moon_alt
        self.helio_lon = helio_lon
        self.ecl_lat = ecl_lat
        self.jd = jd
        self.phase_angle = phase_angle


def build_geometry(wcs, shape: Tuple[int, int], lat_deg: float, lon_deg: float,
                   when_iso: str, step: int = 16) -> Optional[SkyGeometry]:
    """Per-pixel observing geometry over an image, via its WCS.

    Sampled on a ``step``-pixel grid and bilinearly upsampled: these fields
    vary on degree scales across a field that is typically under a degree
    wide, so evaluating them per-pixel would be pure waste.
    """
    jd = julian_date(when_iso)
    if jd is None or wcs is None:
        return None

    h, w = int(shape[0]), int(shape[1])
    ys = np.arange(0, h, step, dtype=np.float64)
    xs = np.arange(0, w, step, dtype=np.float64)
    if ys[-1] != h - 1:
        ys = np.append(ys, h - 1)
    if xs[-1] != w - 1:
        xs = np.append(xs, w - 1)
    grid_x, grid_y = np.meshgrid(xs, ys)

    try:
        ra, dec = wcs.all_pix2world(grid_x, grid_y, 0)
    except Exception as exc:
        _log.debug("physical sky model: WCS projection failed (%s)", exc)
        return None
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    if not np.isfinite(ra).all() or not np.isfinite(dec).all():
        return None

    lst = (gmst_deg(jd) + lon_deg) % 360.0
    alt, az = altaz_from_equatorial(ra, dec, lat_deg, lst)
    zenith = 90.0 - alt

    moon_ra, moon_dec, _, _ = moon_position(jd)
    moon_alt, _ = altaz_from_equatorial(np.array([moon_ra]), np.array([moon_dec]),
                                        lat_deg, lst)
    moon_sep = angular_separation_deg(ra, dec, moon_ra, moon_dec)

    eps = obliquity_deg(jd)
    ecl_lon, ecl_lat = equatorial_to_ecliptic(ra, dec, eps)
    _, _, sun_lon = sun_position(jd)
    helio_lon = (ecl_lon - sun_lon) % 360.0

    def _up(arr):
        return _bilinear_upsample(arr, ys, xs, h, w)

    return SkyGeometry(
        zenith_angle=_up(zenith), azimuth=_up(az), moon_sep=_up(moon_sep),
        moon_alt=float(moon_alt[0]), helio_lon=_up(helio_lon),
        ecl_lat=_up(ecl_lat), jd=jd, phase_angle=moon_phase_angle_deg(jd))


def _bilinear_upsample(values: np.ndarray, ys: np.ndarray, xs: np.ndarray,
                       h: int, w: int) -> np.ndarray:
    """Bilinearly expand a coarse (len(ys), len(xs)) grid to (h, w)."""
    out_y = np.arange(h, dtype=np.float64)
    out_x = np.arange(w, dtype=np.float64)

    iy = np.clip(np.searchsorted(ys, out_y, side='right') - 1, 0, len(ys) - 2)
    ix = np.clip(np.searchsorted(xs, out_x, side='right') - 1, 0, len(xs) - 2)
    ty = ((out_y - ys[iy]) / np.maximum(ys[iy + 1] - ys[iy], 1e-9))[:, None]
    tx = ((out_x - xs[ix]) / np.maximum(xs[ix + 1] - xs[ix], 1e-9))[None, :]

    v00 = values[np.ix_(iy, ix)]
    v01 = values[np.ix_(iy, ix + 1)]
    v10 = values[np.ix_(iy + 1, ix)]
    v11 = values[np.ix_(iy + 1, ix + 1)]
    return ((1 - ty) * ((1 - tx) * v00 + tx * v01)
            + ty * ((1 - tx) * v10 + tx * v11))


def amp_glow_basis(shape: Tuple[int, int],
                   decay_frac: float = 0.28) -> Tuple[List[np.ndarray], List[str]]:
    """Per-corner exponential glow, in **sensor** coordinates.

    Amp glow is electroluminescence from readout electronics at the edge of
    the sensor: brightest in one corner, falling off exponentially, and fixed
    to the detector rather than the sky. One term per corner lets the
    non-negative fit pick whichever corner (or pair) the camera actually
    glows from, without being told.

    Note what is deliberately **absent** here: a vignetting term. Vignetting
    is radially symmetric about the optical axis, and observers centre their
    targets, so a free centred radial term is nebula-shaped -- it would
    reopen exactly the failure this module exists to prevent, and
    ``_corner_gradient`` cannot catch it because a radial pattern leaves all
    four corners equal by construction. It is also largely redundant:
    ``frame_processor`` already divides by the master flat when one exists,
    which is what a flat is for. Measuring residual vignetting honestly needs
    the same star compared at different sensor positions across a dithered
    session -- see ``fit_sky_model_multi`` -- not a shape fitted to one
    stacked background. Corner terms carry none of that risk: they peak at an
    edge, so they cannot absorb a centred object.
    """
    h, w = int(shape[0]), int(shape[1])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ny, nx = yy / max(h - 1, 1), xx / max(w - 1, 1)
    scale = max(float(decay_frac), 1e-3)

    terms, names = [], []
    for cy, cx, label in ((0.0, 0.0, 'tl'), (0.0, 1.0, 'tr'),
                          (1.0, 0.0, 'bl'), (1.0, 1.0, 'br')):
        dist = np.hypot(ny - cy, nx - cx)
        terms.append(np.exp(-dist / scale))
        names.append(f'amp_glow_{label}')
    return terms, names


def build_basis(geom: SkyGeometry,
                lp_source_az_deg: float = 0.0,
                instrumental: bool = False) -> Tuple[np.ndarray, List[str]]:
    """Stack the physical component maps into a design matrix.

    Returns ``(basis, names)`` where ``basis`` is ``(n_terms, H, W)``. Each
    term is normalised to unit mean so the fitted coefficients are directly
    comparable as "how much of the gradient is this component".

    ``instrumental`` adds detector-fixed terms (see ``amp_glow_basis``). They
    are off by default because they are only separable from the sky terms
    when the sky geometry actually moves between exposures -- on a single
    stacked frame both are just smooth surfaces and adding more of them only
    makes an already ill-conditioned fit worse.
    """
    terms, names = [], []

    terms.append(np.ones_like(geom.zenith_angle))
    names.append('constant')

    terms.append(van_rhijn(geom.zenith_angle))
    names.append('airglow')

    moon = moonlight_brightness(geom.moon_sep, geom.moon_alt,
                                geom.zenith_angle, geom.phase_angle)
    if np.ptp(moon) > 0:
        terms.append(moon)
        names.append('moonlight')

    terms.append(zodiacal_brightness(geom.helio_lon, geom.ecl_lat))
    names.append('zodiacal')

    terms.append(light_pollution_brightness(geom.azimuth, geom.zenith_angle,
                                            lp_source_az_deg))
    names.append('light_pollution')

    if instrumental:
        inst_terms, inst_names = amp_glow_basis(geom.zenith_angle.shape)
        terms.extend(inst_terms)
        names.extend(inst_names)

    basis = np.stack([_unit_mean(t) for t in terms], axis=0)
    return basis, names


def _unit_mean(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    mean = float(np.mean(arr))
    return arr / mean if abs(mean) > 1e-12 else arr


def fit_sky_model(channel: np.ndarray, basis: np.ndarray,
                  mask: Optional[np.ndarray] = None,
                  n_iter: int = 3, clip_sigma: float = 2.5
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Robust least-squares fit of ``channel`` to the physical basis.

    Two things make this fit refuse to absorb astrophysical signal, and both
    are load-bearing:

    **Non-negative coefficients.** Physically, no sky component emits negative
    light. Numerically, this is what stops the fit from eating a nebula.
    Across a real 1-2 degree field the component maps vary by only a few
    percent and are nearly collinear (condition number ~1e5), so *unbounded*
    least squares can build a centrally-peaked bump out of large, cancelling
    positive and negative multiples of near-identical ramps -- measured at
    -25% signal preservation on a synthetic frame-filling nebula, i.e. worse
    than doing nothing. Constraining every coefficient to be >= 0 removes that
    cancellation freedom entirely: a non-negative combination of monotonic
    ramps is still a monotonic ramp, and cannot have an interior maximum.

    **Upward-only sigma clipping.** Stars and nebulosity sit *above* the sky,
    so only the high tail is rejected. Symmetric clipping would drag the
    fitted background up into the signal and subtract real flux.

    Returns ``(coefficients, model_image)``.
    """
    flat_basis = basis.reshape(basis.shape[0], -1).T      # (npix, nterms)
    flat_data = np.asarray(channel, dtype=np.float64).ravel()

    good = np.isfinite(flat_data)
    if mask is not None:
        good &= ~np.asarray(mask, dtype=bool).ravel()
    if good.sum() < flat_basis.shape[1] * 4:
        raise ValueError("not enough unmasked background pixels to fit the sky model")

    coeffs = np.zeros(flat_basis.shape[1], dtype=np.float64)
    for _ in range(max(1, n_iter)):
        coeffs = _nnls(flat_basis[good], flat_data[good])
        residual = flat_data - flat_basis @ coeffs
        scale = float(np.std(residual[good]))
        if not np.isfinite(scale) or scale <= 0:
            break
        good &= residual < clip_sigma * scale             # upward-only rejection
        if good.sum() < flat_basis.shape[1] * 4:
            break

    model = (flat_basis @ coeffs).reshape(channel.shape)
    return coeffs, model


def _nnls(design: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Non-negative least squares, with a plain-lstsq clamp as a fallback."""
    try:
        from scipy.optimize import nnls
        coeffs, _ = nnls(design, data)
        return np.asarray(coeffs, dtype=np.float64)
    except Exception as exc:                              # pragma: no cover
        _log.debug("nnls unavailable (%s); clamping an unbounded fit", exc)
        coeffs, *_ = np.linalg.lstsq(design, data, rcond=None)
        return np.maximum(np.asarray(coeffs, dtype=np.float64), 0.0)


# Above this condition number the component maps are too nearly parallel for
# the split between them to mean anything. Measured: a realistic 1.5 deg field
# lands around 1e5, a 0.04 deg field is worse still. Removal stays valid
# either way (the *span* is correct), only the attribution is unidentifiable.
_ATTRIBUTION_MAX_COND = 1.0e3

# The fit must cut the corner-to-corner background spread to at least this
# fraction of its original value to be worth applying. Set just below 1.0 --
# the bar is "measurably better", not "dramatically better", because a blind
# extractor is standing by and is the right tool whenever this one isn't.
_MIN_GRADIENT_IMPROVEMENT = 0.95

# Field of view (degrees, longest axis) below which the sky components carry
# too little structure to be worth fitting. From a measured condition-number
# sweep -- see the gate in remove_physical_sky. This is roughly 300 mm on
# APS-C; shorter lenses qualify, telescopes generally do not.
_MIN_FIELD_OF_VIEW_DEG = 5.0


def _field_of_view_deg(wcs, shape: Tuple[int, int]) -> Optional[float]:
    """Angular size of the longest image axis, or None if the WCS won't say.

    Always goes through ``proj_plane_pixel_scales`` rather than reading
    ``wcs.wcs.cdelt``. A real session WCS -- a Celestron Origin solve, for
    one -- carries its scale in a **CD matrix** and leaves ``cdelt`` at
    ``[1, 1]``, so reading cdelt directly returns the image size in *pixels*
    dressed up as degrees: 2958 "degrees" for a 2958-pixel-wide frame, which
    sails past any sanity threshold instead of tripping it. A fallback
    guarded on "cdelt looks unset" never fires either, because 1.0 is
    perfectly finite and positive.
    """
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales
        scales = np.abs(np.asarray(proj_plane_pixel_scales(wcs), dtype=np.float64))
        if scales.size < 2 or not np.all(np.isfinite(scales)) or not np.all(scales > 0):
            return None
        h, w = int(shape[0]), int(shape[1])
        return float(max(h * scales[1], w * scales[0]))
    except Exception:
        return None


def _corner_gradient(image: np.ndarray) -> float:
    """Corner-to-corner spread of the background, a scalar flatness measure.

    Medians of the four corner eighths: far enough out to be sky on a
    centre-framed target, and a median so stars in a corner don't move it.
    """
    arr = np.asarray(image, dtype=np.float64)
    lum = arr.mean(axis=-1) if arr.ndim == 3 else arr
    h, w = lum.shape
    hy, hx = max(h // 8, 1), max(w // 8, 1)
    corners = [float(np.median(lum[:hy, :hx])), float(np.median(lum[:hy, -hx:])),
               float(np.median(lum[-hy:, :hx])), float(np.median(lum[-hy:, -hx:]))]
    return max(corners) - min(corners)


def basis_condition(basis: np.ndarray) -> float:
    """Condition number of the design matrix, for attribution gating."""
    flat = basis.reshape(basis.shape[0], -1).T
    try:
        return float(np.linalg.cond(flat))
    except Exception:                                     # pragma: no cover
        return float('inf')


def fit_sky_model_multi(channels: Sequence[np.ndarray],
                        geometries: Sequence['SkyGeometry'],
                        masks: Optional[Sequence[Optional[np.ndarray]]] = None,
                        lp_source_az_deg: float = 0.0,
                        n_iter: int = 3, clip_sigma: float = 2.5):
    """Joint fit across several exposures, separating sky from detector.

    This is what makes the decomposition identifiable. On a single frame the
    sky components are nearly collinear (condition ~1e5 at a 1-degree field),
    so the split between them is meaningless -- and a detector-fixed term is
    indistinguishable from a sky one, because both are just smooth surfaces.

    Across a session they separate, because they move differently:

      * **Sky** terms are fixed to the *sky*. As the target tracks, its
        zenith angle, azimuth and moon separation all change, so each sky
        component's map is different in every exposure.
      * **Detector** terms are fixed to the *sensor*. Amp glow sits in the
        same corner in every exposure regardless of where the telescope
        points.

    So the solve shares one amplitude per component across all frames, while
    letting each frame contribute its own geometry. A component that changes
    with the sky and one that does not can then be told apart, which no
    amount of cleverness on a single stacked frame can do.

    Args:
        channels: One 2-D luminance plane per exposure. All the same shape.
        geometries: The matching ``SkyGeometry`` per exposure, each built
            from that exposure's own timestamp.
        masks: Optional per-exposure boolean masks of pixels to exclude
            (stars, the target itself).

    Returns:
        ``(coefficients, names, models)`` -- one shared coefficient vector,
        the term names, and the per-exposure model images it implies.
    """
    if len(channels) != len(geometries):
        raise ValueError("need one geometry per channel")
    if len(channels) < 2:
        raise ValueError("multi-frame separation needs at least two exposures")

    bases, names = [], None
    for geom in geometries:
        basis, this_names = build_basis(geom, lp_source_az_deg, instrumental=True)
        if names is None:
            names = this_names
        elif this_names != names:
            # A moonlight term appears only while the moon is above the
            # horizon, so a session spanning moonrise would otherwise stack
            # design matrices with different columns.
            raise ValueError("frames disagree on which components are present; "
                             "split the session at moonrise/moonset")
        bases.append(basis)

    n_terms = bases[0].shape[0]
    rows, targets = [], []
    for i, (basis, channel) in enumerate(zip(bases, channels)):
        flat = basis.reshape(n_terms, -1).T
        data = np.asarray(channel, dtype=np.float64).ravel()
        good = np.isfinite(data)
        if masks is not None and masks[i] is not None:
            good &= ~np.asarray(masks[i], dtype=bool).ravel()
        rows.append(flat[good])
        targets.append(data[good])

    design = np.concatenate(rows, axis=0)
    target = np.concatenate(targets, axis=0)
    if design.shape[0] < n_terms * 4:
        raise ValueError("not enough unmasked background pixels to fit the sky model")

    keep = np.ones(design.shape[0], dtype=bool)
    coeffs = np.zeros(n_terms, dtype=np.float64)
    for _ in range(max(1, n_iter)):
        coeffs = _nnls(design[keep], target[keep])
        residual = target - design @ coeffs
        scale = float(np.std(residual[keep]))
        if not np.isfinite(scale) or scale <= 0:
            break
        keep &= residual < clip_sigma * scale      # upward-only, as single-frame
        if keep.sum() < n_terms * 4:
            break

    models = [np.tensordot(coeffs, b, axes=(0, 0)) for b in bases]
    return coeffs, names, models


def describe_fit(coeffs: np.ndarray, names: List[str],
                 condition: Optional[float] = None) -> str:
    """Human-readable breakdown of what the gradient was made of.

    Honest about identifiability: over a typical field of view the component
    maps differ by only a few percent and are nearly collinear, so while the
    *fit* is well-posed (non-negativity pins it down) the *split* between
    components is not -- the solver can load the whole gradient onto whichever
    term it likes. When the design matrix is that ill-conditioned this says so
    instead of reporting a confident-looking percentage breakdown that means
    nothing. Attribution becomes meaningful on wide fields, or when one
    component dominates geometrically (a bright moon close to the field).
    """
    varying = [(n, float(c)) for n, c in zip(names, coeffs) if n != 'constant']
    total = sum(abs(c) for _, c in varying)
    if total <= 0:
        return "sky model: flat (no significant gradient components)"

    if condition is not None and condition > _ATTRIBUTION_MAX_COND:
        return ("sky model: gradient removed; component split not identifiable "
                f"at this field size (basis condition {condition:.0e})")

    parts = [f"{n} {100.0 * abs(c) / total:.0f}%"
             for n, c in sorted(varying, key=lambda t: -abs(t[1])) if abs(c) / total > 0.01]
    return "sky model: " + ", ".join(parts)


def remove_physical_sky(image: np.ndarray, wcs, lat_deg: float, lon_deg: float,
                        when_iso: str, mask: Optional[np.ndarray] = None,
                        lp_source_az_deg: float = 0.0
                        ) -> Optional[Dict]:
    """Fit and subtract the physical sky model, per channel.

    Returns ``None`` (rather than raising) when the geometry can't be built
    (no WCS, no timestamp, no site coordinates) **or when the fitted model
    does not actually flatten the background**, so callers fall back to a
    blind extractor exactly as they do for every other optional input.

    That second check is not defensive padding -- it is what makes this
    honest on real data. Measured on a real 1-degree Lagoon field: the zenith
    angle varies by 0.94 deg across the whole frame and the azimuth by 1.5
    deg, so every component map is essentially constant, the fit has nothing
    to grip, and subtracting it made the corner-to-corner gradient *worse*
    (67 -> 111 ADU) where DBE removed 68% of it. Sweeping the light-pollution
    azimuth through all 360 degrees moved the residual by less than 0.01 ADU,
    confirming the basis has no discriminating power at that field size.

    This is structural rather than a tuning problem: physical sky components
    vary on ten-degree scales, so across a typical deep-sky field the real
    gradient is dominated by *instrumental* effects -- vignetting, amp glow,
    filter gradients -- which a model of the sky cannot represent by
    construction. The model earns its place on wide fields; on narrow ones it
    must stand aside for an extractor that can fit what is actually there.
    """
    geom = build_geometry(wcs, image.shape[:2], lat_deg, lon_deg, when_iso)
    if geom is None:
        return None

    # Field-size gate. Measured sweep of the design-matrix condition number
    # against field of view (moon down, so sky terms only):
    #
    #     0.5 deg -> 5.1e5    2 deg -> 3.2e4    10 deg -> 4.2e2
    #     1.0 deg -> 1.3e5    5 deg -> 9.5e2    40 deg -> 6.3e1
    #
    # The components only become separable around 5 degrees, and below that
    # they carry too little structure to fit a real gradient at all. A close
    # bright moon does not rescue it: with a full moon 52 deg up and 5 deg
    # from the target, the moonlight term still varies by only 5% across a
    # 1-degree frame (condition 9.2e5). So this is a property of the field,
    # not of the night, and it is worth failing fast rather than fitting
    # noise and then discovering it via the improvement check below.
    fov = _field_of_view_deg(wcs, image.shape[:2])
    if fov is not None and fov < _MIN_FIELD_OF_VIEW_DEG:
        _log.debug("physical sky model: %.2f deg field is below the %.1f deg "
                   "where sky components become separable; falling back",
                   fov, _MIN_FIELD_OF_VIEW_DEG)
        return None

    basis, names = build_basis(geom, lp_source_az_deg)

    out = np.array(image, dtype=np.float32, copy=True)
    models, all_coeffs = [], []

    n_ch = image.shape[2] if image.ndim == 3 else 1
    for c in range(n_ch):
        channel = image[:, :, c] if image.ndim == 3 else image
        coeffs, model = fit_sky_model(channel, basis, mask)
        # Preserve the channel's own sky level: subtract only the *varying*
        # part, so this stays a gradient remover and does not also silently
        # re-zero the pedestal that later Phase 4 steps expect to be there.
        varying = model - float(np.median(model))
        if image.ndim == 3:
            out[:, :, c] = channel - varying
        else:
            out = channel - varying
        models.append(model)
        all_coeffs.append(coeffs)

    # Verify the model actually helped before handing it back.
    before = _corner_gradient(image)
    after = _corner_gradient(out)
    if not (after < before * _MIN_GRADIENT_IMPROVEMENT):
        _log.debug("physical sky model: corner gradient %.3f -> %.3f, not an "
                   "improvement; falling back to a blind extractor",
                   before, after)
        return None

    mean_coeffs = np.mean(np.stack(all_coeffs, axis=0), axis=0)
    condition = basis_condition(basis)
    return {
        'image': out,
        'model': np.stack(models, axis=-1) if image.ndim == 3 else models[0],
        'coefficients': mean_coeffs,
        'names': names,
        'condition': condition,
        'description': describe_fit(mean_coeffs, names, condition),
        'moon_altitude_deg': geom.moon_alt,
        'moon_phase_angle_deg': geom.phase_angle,
        'moon_illuminated_fraction': float((1.0 + math.cos(geom.phase_angle * _DEG)) / 2.0),
    }
