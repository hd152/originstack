"""Direct-HTTP replacements for every astroquery service this codebase used:
Gaia DR3 (TAP), VizieR (TAP), SIMBAD (TAP), astrometry.net (upload/solve
REST API), and JPL Horizons (REST ephemeris API). stdlib-only (urllib +
json), no astroquery dependency.

Every query function here returns None (or an empty/failure sentinel) on
any network, HTTP, or parse error -- the same best-effort, non-fatal
contract the astroquery-backed code already had throughout this codebase:
these are all optional enrichment features (plate-solve object ID,
photometric calibration catalogues, target-name inference, comet
ephemeris) that already degraded gracefully to a non-networked fallback
path when astroquery/network was unavailable. That contract is unchanged.

Gaia, VizieR, and SIMBAD are all standard IVOA TAP (Table Access Protocol)
services, so they share one query path (``tap_query``) that POSTs an ADQL
query to the service's ``/sync`` endpoint and parses the standard
TAP-JSON response (``{"metadata": [...], "data": [[...], ...]}``) into an
``astropy.table.Table`` -- the same column-access interface
(``table["colname"]``, ``len(table)``, boolean-mask indexing) astroquery's
results already had, so call sites needed no changes beyond the query
call itself.

Every endpoint/schema here (Gaia TAP, VizieR TAP, SIMBAD TAP, JPL Horizons,
astrometry.net login) was validated against the real, live services
(2026-07-30) -- not just documentation. tests/test_net_query.py covers
response-parsing logic against mocked payloads for fast, offline CI; the
live check is a one-off, not part of the automated suite (no network
access assumed at test time).
"""
from __future__ import annotations

import io
import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_USER_AGENT = "originstack-net_query/1.0"
_DEFAULT_TIMEOUT = 30.0


def _ssl_context() -> ssl.SSLContext:
    """Default SSL context, with certifi's CA bundle layered on top when
    available. Some Python installs (seen firsthand on Windows: the default
    OpenSSL cafile points at a path that doesn't exist) ship with no usable
    default trust store, which would otherwise make every query in this
    module fail with CERTIFICATE_VERIFY_FAILED -- indistinguishable from a
    genuine network-down condition. certifi is not a hard dependency (it's
    a very common transitive one); this only uses it if already installed,
    same as every other optional-dependency pattern in this codebase."""
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(cafile=certifi.where())
    except Exception:
        pass
    return ctx


# ---------------------------------------------------------------------------
# Low-level HTTP
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: float = _DEFAULT_TIMEOUT) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
        return resp.read()


def _http_post_form(url: str, fields: Dict[str, str],
                    timeout: float = _DEFAULT_TIMEOUT) -> bytes:
    body = urllib.parse.urlencode(fields).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
        return resp.read()


def _http_post_multipart(url: str, fields: Dict[str, str], file_field: str,
                         file_name: str, file_bytes: bytes,
                         timeout: float = _DEFAULT_TIMEOUT) -> bytes:
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(
            (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
             f'\r\n\r\n{value}\r\n').encode("utf-8")
        )
    parts.append(
        (f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
         f'filename="{file_name}"\r\nContent-Type: application/octet-stream'
         f'\r\n\r\n').encode("utf-8")
        + file_bytes + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)
    req = urllib.request.Request(url, data=body, headers={
        "User-Agent": _USER_AGENT,
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    })
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
        return resp.read()


# ---------------------------------------------------------------------------
# TAP (IVOA Table Access Protocol) -- shared by Gaia, VizieR, SIMBAD
# ---------------------------------------------------------------------------

def tap_query(sync_url: str, adql: str, timeout: float = 60.0):
    """Run a synchronous ADQL query against a TAP ``/sync`` endpoint.

    Returns an ``astropy.table.Table`` (same column-access interface
    astroquery's TAP-backed results had) or ``None`` on any failure.
    """
    from astropy.table import Table
    try:
        raw = _http_post_form(sync_url, {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "json",
            "QUERY": adql,
        }, timeout=timeout)
        payload = json.loads(raw.decode("utf-8"))
        colnames = [c["name"] for c in payload["metadata"]]
        rows = payload["data"]
        columns: Dict[str, list] = {name: [] for name in colnames}
        for row in rows:
            for name, val in zip(colnames, row):
                columns[name].append(val)
        arrays: Dict[str, np.ndarray] = {}
        for name in colnames:
            vals = columns[name]
            try:
                cleaned = [np.nan if v is None else v for v in vals]
                arrays[name] = np.array(cleaned, dtype=float)
            except (TypeError, ValueError):
                arrays[name] = np.array(
                    ["" if v is None else v for v in vals], dtype=object)
        return Table(arrays)
    except Exception:
        return None


_GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
_VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
_SIMBAD_TAP = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"


def gaia_cone_search(ra_deg: float, dec_deg: float, radius_deg: float,
                     columns: List[str], max_rows: int = 500,
                     require_not_null: Optional[List[str]] = None):
    """Gaia DR3 cone search. Returns an astropy Table or None."""
    cols = ", ".join(columns)
    where_extra = ""
    if require_not_null:
        conds = " AND ".join(f"{c} IS NOT NULL" for c in require_not_null)
        where_extra = f" AND {conds}"
    adql = (
        f"SELECT TOP {int(max_rows)} {cols} FROM gaiadr3.gaia_source "
        f"WHERE 1=CONTAINS(POINT('ICRS',ra,dec),"
        f"CIRCLE('ICRS',{ra_deg},{dec_deg},{radius_deg})){where_extra}"
    )
    return tap_query(_GAIA_TAP, adql)


def vizier_cone_search(ra_deg: float, dec_deg: float, radius_deg: float,
                       catalog: str, columns: List[str], max_rows: int = 500):
    """VizieR catalogue cone search (e.g. catalog='II/246/out' for 2MASS PSC).
    Returns an astropy Table or None."""
    cols = ", ".join(columns)
    adql = (
        f'SELECT TOP {int(max_rows)} {cols} FROM "{catalog}" '
        f"WHERE 1=CONTAINS(POINT('ICRS',RAJ2000,DEJ2000),"
        f"CIRCLE('ICRS',{ra_deg},{dec_deg},{radius_deg}))"
    )
    return tap_query(_VIZIER_TAP, adql)


def simbad_cone_search(ra_deg: float, dec_deg: float, radius_deg: float = 0.5):
    """SIMBAD cone search for the nearest catalogued object. Returns an
    astropy Table with (main_id, otype) columns, or None."""
    adql = (
        "SELECT main_id, otype FROM basic "
        f"WHERE 1=CONTAINS(POINT('ICRS',ra,dec),"
        f"CIRCLE('ICRS',{ra_deg},{dec_deg},{radius_deg}))"
    )
    return tap_query(_SIMBAD_TAP, adql)


def simbad_name_lookup(name: str):
    """Resolve a SIMBAD identifier/alias to its (main_id, otype). Returns
    an astropy Table with (main_id, otype) columns, or None."""
    escaped = name.replace("'", "''")
    adql = (
        "SELECT basic.main_id, basic.otype FROM ident "
        "JOIN basic ON ident.oidref = basic.oid "
        f"WHERE ident.id = '{escaped}'"
    )
    return tap_query(_SIMBAD_TAP, adql)


# ---------------------------------------------------------------------------
# astrometry.net (nova.astrometry.net) upload/solve REST API
# ---------------------------------------------------------------------------

_ASTROMETRY_API = "https://nova.astrometry.net/api/"


def astrometry_net_login(api_key: str, timeout: float = 30.0) -> Optional[str]:
    """Return a session key, or None on failure."""
    try:
        raw = _http_post_form(
            _ASTROMETRY_API + "login",
            {"request-json": json.dumps({"apikey": api_key})},
            timeout=timeout)
        r = json.loads(raw.decode("utf-8"))
        if r.get("status") == "success":
            return r["session"]
    except Exception:
        pass
    return None


def astrometry_net_upload(session: str, fits_path: str,
                          timeout: float = 120.0, **solve_params) -> Optional[str]:
    """Upload a FITS file for solving. Returns a submission id, or None."""
    payload: Dict[str, Any] = {
        "session": session,
        "publicly_visible": "n",
        "allow_modifications": "n",
        "allow_commercial_use": "n",
    }
    payload.update({k: v for k, v in solve_params.items() if v is not None})
    try:
        with open(fits_path, "rb") as f:
            file_bytes = f.read()
        raw = _http_post_multipart(
            _ASTROMETRY_API + "upload", {"request-json": json.dumps(payload)},
            "file", "image.fits", file_bytes, timeout=timeout)
        r = json.loads(raw.decode("utf-8"))
        if r.get("status") == "success":
            return str(r["subid"])
    except Exception:
        pass
    return None


def astrometry_net_poll_submission(subid: str, deadline: float,
                                   poll_interval: float = 5.0) -> Optional[str]:
    """Poll a submission until it has a job id. Returns the job id, or
    None on timeout/failure."""
    while time.time() < deadline:
        try:
            raw = _http_get(_ASTROMETRY_API + f"submissions/{subid}", timeout=30)
            r = json.loads(raw.decode("utf-8"))
            jobs = r.get("jobs") or []
            if jobs and jobs[0] is not None:
                return str(jobs[0])
        except Exception:
            pass
        time.sleep(poll_interval)
    return None


def astrometry_net_poll_job(job_id: str, deadline: float,
                            poll_interval: float = 5.0) -> Optional[bool]:
    """Poll a job until it succeeds or fails. Returns True/False, or None
    on timeout."""
    while time.time() < deadline:
        try:
            raw = _http_get(_ASTROMETRY_API + f"jobs/{job_id}", timeout=30)
            r = json.loads(raw.decode("utf-8"))
            status = r.get("status")
            if status == "success":
                return True
            if status == "failure":
                return False
        except Exception:
            pass
        time.sleep(poll_interval)
    return None


def astrometry_net_fetch_wcs(job_id: str, timeout: float = 30.0):
    """Fetch the solved WCS as an astropy.io.fits.Header, or None."""
    from astropy.io import fits
    try:
        raw = _http_get(f"https://nova.astrometry.net/wcs_file/{job_id}", timeout=timeout)
        with fits.open(io.BytesIO(raw)) as hdul:
            return hdul[0].header.copy()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# JPL Horizons REST ephemeris API
# ---------------------------------------------------------------------------

_HORIZONS_API = "https://ssd.jpl.nasa.gov/api/horizons.api"


def horizons_ephemeris(designation: str, iso_time: str,
                       location: Optional[Any] = None,
                       timeout: float = 30.0) -> Optional[Tuple[float, float]]:
    """Single-instant astrometric (RA, Dec) in degrees for a small body
    (comet/asteroid designation) at ``iso_time`` (UTC, 'YYYY-MM-DDTHH:MM:SS').

    ``location``: None for geocentric, a dict {'lon','lat','elevation'}
    (deg, deg, km -- same convention astroquery.jplhorizons.Horizons used)
    for a topocentric site, or an MPC observatory code string.

    Returns None on any failure (unparseable response, network error,
    object not found).
    """
    try:
        try:
            t0 = datetime.strptime(iso_time, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            t0 = datetime.fromisoformat(iso_time)
    except Exception:
        return None
    t1 = t0 + timedelta(minutes=1)

    params: Dict[str, str] = {
        "format": "json",
        "COMMAND": f"'{designation}'",
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "OBSERVER",
        "START_TIME": f"'{t0.strftime('%Y-%m-%d %H:%M')}'",
        "STOP_TIME": f"'{t1.strftime('%Y-%m-%d %H:%M')}'",
        "STEP_SIZE": "'1m'",
        "QUANTITIES": "'1'",
        "ANG_FORMAT": "'DEG'",
        "CSV_FORMAT": "YES",
    }
    if isinstance(location, dict):
        params["CENTER"] = "'coord@399'"
        params["COORD_TYPE"] = "'GEODETIC'"
        params["SITE_COORD"] = (
            f"'{location['lon']},{location['lat']},{location['elevation']}'")
    elif isinstance(location, str) and location:
        params["CENTER"] = f"'{location}@399'"
    else:
        params["CENTER"] = "'500@399'"

    url = _HORIZONS_API + "?" + urllib.parse.urlencode(params)
    try:
        raw = _http_get(url, timeout=timeout)
        payload = json.loads(raw.decode("utf-8"))
        text = payload.get("result", "") or ""
        if "$$SOE" not in text or "$$EOE" not in text:
            return None
        block = text.split("$$SOE", 1)[1].split("$$EOE", 1)[0].strip()
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            return None
        # CSV row: "Date_UT,  , R.A._(ICRF), DEC_(ICRF)" (QUANTITIES=1,
        # ANG_FORMAT=DEG gives decimal-degree RA/Dec directly; the blank
        # field is Horizons' solar/lunar-presence flag column).
        fields = [p.strip() for p in lines[0].split(",")]
        nums = []
        for p in fields[1:]:
            try:
                nums.append(float(p))
            except ValueError:
                continue
        if len(nums) < 2:
            return None
        return nums[0], nums[1]
    except Exception:
        return None
