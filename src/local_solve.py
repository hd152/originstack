"""Offline plate solving against a local Gaia star index (``--plate-solver local``).

No external binary, no API key and -- once the index covers the field -- no
network. Two parts:

**Star index.** Gaia DR3 stars in fixed sky tiles (5 deg Dec bands, each cut
into ~5 deg RA cells), the brightest ``TILE_MAX_STARS`` per tile below
``TILE_MAG_LIMIT``, one ``.npy`` per tile under ``star_index_dir()``. A tile
the solver needs and does not have is fetched from Gaia's TAP service and kept
when the run is online, so any field solved once solves offline afterwards;
``tools/build_star_index.py`` fills a region or the whole sky ahead of time
(~1650 tiles, ~80 MB) for a machine that is never online. ~120 stars/deg^2 --
about 100 in an Origin-sized (1.3 x 0.8 deg) field.

**Solver.** Hint-driven, not all-sky blind: a position hint (an existing WCS in
the header -- e.g. the Origin ``info.json`` session solve -- ``RA``/``DEC``/
``OBJCTRA`` keywords, or a caller-supplied centre) and, when known, a pixel
scale (WCS, ``FOCALLEN``+``XPIXSZ``). Catalogue stars around each candidate
centre are projected onto the tangent plane at the trial scale and matched to
the image's brightest stars with ``blind_match.match_rigid_unknown_rotation``
(the rotation-agnostic native matcher ``--merge`` uses), for both parities.
A hit is refined to a full affine (``CD`` matrix) by iterated nearest-neighbour
least squares, re-centred on the image centre, and accepted only with enough
inliers at a small residual. Candidate centres cover ``search_radius`` in
half-field steps, and the scale is swept in 2 % steps across its uncertainty,
so a coarse hint (an object name's coordinates, a header scale off by a few
percent) still solves -- each trial is a few ms.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.utils import safe_print

_log = logging.getLogger('originstack')

BAND_DEG = 5.0
TILE_MAG_LIMIT = 15.0
TILE_MAX_STARS = 3000
_N_BANDS = int(round(180.0 / BAND_DEG))

IMG_STARS = 40            # brightest image stars used for the pattern match
MIN_INLIERS = 8
MAX_RMS_PX = 1.5
SCALE_STEP = 1.02         # trial-scale ratio; the matcher tolerates ~1% in pair distances


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def star_index_dir() -> str:
    """``$ORIGINSTACK_STAR_INDEX``, else a per-user cache directory."""
    env = os.environ.get('ORIGINSTACK_STAR_INDEX')
    if env:
        return env
    if os.name == 'nt':
        base = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~')
        return os.path.join(base, 'OriginStack', 'star_index')
    base = os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache')
    return os.path.join(base, 'originstack', 'star_index')


def _n_ra(band: int) -> int:
    dec_c = -90.0 + (band + 0.5) * BAND_DEG
    return max(1, int(round(360.0 * math.cos(math.radians(dec_c)) / BAND_DEG)))


def tile_bounds(band: int, r: int) -> Tuple[float, float, float, float]:
    """(ra_lo, ra_hi, dec_lo, dec_hi) of tile (band, r), degrees."""
    n = _n_ra(band)
    return (r * 360.0 / n, (r + 1) * 360.0 / n,
            -90.0 + band * BAND_DEG, -90.0 + (band + 1) * BAND_DEG)


def all_tiles() -> List[Tuple[int, int]]:
    return [(b, r) for b in range(_N_BANDS) for r in range(_n_ra(b))]


def tiles_for_cone(ra: float, dec: float, radius: float) -> List[Tuple[int, int]]:
    """Tiles that may hold stars within *radius* deg of (ra, dec)."""
    lo, hi = max(-90.0, dec - radius), min(90.0, dec + radius)
    b0 = max(0, int((lo + 90.0) // BAND_DEG))
    b1 = min(_N_BANDS - 1, int((hi + 90.0) // BAND_DEG))
    out = []
    for b in range(b0, b1 + 1):
        n = _n_ra(b)
        _, _, blo, bhi = tile_bounds(b, 0)
        # widest RA half-extent of the cone inside this band
        dmax = max(abs(max(lo, blo)), abs(min(hi, bhi)))
        c = math.cos(math.radians(min(dmax, 89.999)))
        if lo <= -89.999 or hi >= 89.999 or radius >= 90.0 or radius / max(c, 1e-6) >= 180.0:
            out.extend((b, r) for r in range(n))
            continue
        half = math.degrees(math.asin(min(1.0, math.sin(math.radians(radius)) / c)))
        width = 360.0 / n
        for r in range(n):
            ra_lo = r * width
            # distance from the cone's RA to the tile's RA interval, with wrap
            d = (ra - ra_lo) % 360.0
            if d <= width or (360.0 - d) <= half or (d - width) <= half:
                out.append((b, r))
    return out


def _tile_path(band: int, r: int, root: Optional[str] = None) -> str:
    return os.path.join(root or star_index_dir(), f'b{band:02d}_r{r:03d}.npy')


def fetch_tile(band: int, r: int, root: Optional[str] = None,
               timeout: float = 120.0) -> Optional[np.ndarray]:
    """Query Gaia DR3 for one tile and store it. Returns the (N, 3) array
    (ra, dec, G) or None when the query failed (offline, network error)."""
    from src import net_query
    ra_lo, ra_hi, dec_lo, dec_hi = tile_bounds(band, r)
    # ADQL for Gaia's TAP service; only float-formatted bounds are interpolated.
    adql = (f"SELECT TOP {TILE_MAX_STARS} ra, dec, phot_g_mean_mag "  # nosec B608
            f"FROM gaiadr3.gaia_source WHERE ra >= {ra_lo:.9f} AND ra < {ra_hi:.9f} "
            f"AND dec >= {dec_lo:.9f} AND dec < {dec_hi:.9f} "
            f"AND phot_g_mean_mag < {TILE_MAG_LIMIT} ORDER BY phot_g_mean_mag")
    try:
        net_query._require_online(net_query._GAIA_TAP)
    except net_query.OfflineError:
        return None
    tb = net_query.tap_query(net_query._GAIA_TAP, adql, timeout=timeout)
    if tb is None:
        return None
    arr = np.column_stack([np.asarray(tb['ra'], float), np.asarray(tb['dec'], float),
                           np.asarray(tb['phot_g_mean_mag'], float)]) if len(tb) else np.zeros((0, 3))
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    path = _tile_path(band, r, root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp.npy'
    np.save(tmp, arr.astype(np.float64))
    os.replace(tmp, path)
    return arr


def load_tile(band: int, r: int, root: Optional[str] = None) -> Optional[np.ndarray]:
    p = _tile_path(band, r, root)
    if not os.path.exists(p):
        return None
    try:
        return np.load(p)
    except Exception as e:
        _log.debug("star index tile %s unreadable: %s", p, e)
        return None


def cone_catalog(ra: float, dec: float, radius: float, fetch: bool = True,
                 root: Optional[str] = None) -> Tuple[np.ndarray, float]:
    """Index stars within *radius* deg of (ra, dec), brightest first, and the
    fraction of the needed tiles that were available. Missing tiles are fetched
    (and cached) when *fetch* and the run is online."""
    tiles = tiles_for_cone(ra, dec, radius)
    parts, have = [], 0
    for b, r in tiles:
        t = load_tile(b, r, root)
        if t is None and fetch:
            t = fetch_tile(b, r, root)
        if t is not None:
            have += 1
            if len(t):
                parts.append(t)
    cov = have / len(tiles) if tiles else 0.0
    if not parts:
        return np.zeros((0, 3)), cov
    cat = np.concatenate(parts)
    sep = angular_sep_deg(cat[:, 0], cat[:, 1], ra, dec)
    cat = cat[sep <= radius]
    return cat[np.argsort(cat[:, 2], kind='stable')], cov


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def angular_sep_deg(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    s = (np.sin((d2 - d1) / 2) ** 2
         + np.cos(d1) * np.cos(d2) * np.sin((r2 - r1) / 2) ** 2)
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(s, 0, 1))))


def project_tan(ra, dec, ra0: float, dec0: float):
    """Gnomonic standard coordinates (xi, eta), degrees, about (ra0, dec0)."""
    a, d = np.radians(ra), np.radians(dec)
    a0, d0 = math.radians(ra0), math.radians(dec0)
    cosc = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - a0)
    xi = np.cos(d) * np.sin(a - a0) / cosc
    eta = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a - a0)) / cosc
    return np.degrees(xi), np.degrees(eta)


def deproject_tan(xi, eta, ra0: float, dec0: float):
    x, y = np.radians(xi), np.radians(eta)
    a0, d0 = math.radians(ra0), math.radians(dec0)
    rho = np.hypot(x, y)
    c = np.arctan(rho)
    with np.errstate(invalid='ignore', divide='ignore'):
        dec = np.arcsin(np.cos(c) * np.sin(d0)
                        + np.where(rho > 0, y * np.sin(c) * np.cos(d0) / np.where(rho > 0, rho, 1), 0))
        ra = a0 + np.arctan2(x * np.sin(c),
                             rho * np.cos(d0) * np.cos(c) - y * np.sin(d0) * np.sin(c))
    return np.degrees(ra) % 360.0, np.degrees(dec)


# ---------------------------------------------------------------------------
# Hints
# ---------------------------------------------------------------------------

@dataclass
class SolveHint:
    ra: float                       # deg
    dec: float                      # deg
    search_radius: float = 0.0      # deg; how far the true centre may be from (ra, dec)
    scale: Optional[float] = None   # arcsec/px
    scale_tol: float = 0.05         # fractional uncertainty of *scale*
    source: str = ''


def _sexagesimal(val, hours: bool) -> Optional[float]:
    """Degrees from a numeric keyword (already degrees) or a sexagesimal string
    (hours for RA when *hours*)."""
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(val)
    except (TypeError, ValueError):
        pass
    try:
        txt = str(val).strip()
        sign = -1.0 if txt.startswith('-') else 1.0
        for ch in ':hmsd\'"':
            txt = txt.replace(ch, ' ')
        parts = [abs(float(p)) for p in txt.replace('+', ' ').replace('-', ' ').split()]
        v = parts[0] + (parts[1] / 60.0 if len(parts) > 1 else 0.0)             + (parts[2] / 3600.0 if len(parts) > 2 else 0.0)
        return sign * v * (15.0 if hours else 1.0)
    except (IndexError, ValueError):
        return None


def hint_from_header(header, shape: Tuple[int, int]) -> Optional[SolveHint]:
    """Position/scale hint from a FITS header: an existing TAN WCS (centre of the
    image, its scale; searched a field-width around, since a capture-app solve
    of the raw frame can be off after cropping), else pointing keywords."""
    h, w = shape
    scale = None
    try:
        if header.get('CD1_1') is not None:
            cd = np.array([[float(header['CD1_1']), float(header.get('CD1_2', 0.0))],
                           [float(header.get('CD2_1', 0.0)), float(header['CD2_2'])]])
            scale = math.sqrt(abs(np.linalg.det(cd))) * 3600.0
        elif header.get('CDELT1') is not None:
            scale = abs(float(header['CDELT1'])) * 3600.0
    except (TypeError, ValueError, KeyError):
        scale = None
    if scale is None:
        try:
            fl, px = float(header.get('FOCALLEN') or 0), float(header.get('XPIXSZ') or 0)
            if fl > 0 and px > 0:
                scale = px / 1000.0 / fl * 206264.806 * float(header.get('XBINNING') or 1)
        except (TypeError, ValueError):
            scale = None
    ctype = str(header.get('CTYPE1', ''))
    if 'TAN' in ctype and header.get('CRVAL1') is not None and scale:
        try:
            from astropy.wcs import WCS
            wcs = WCS(header, naxis=2)
            ra, dec = wcs.all_pix2world([[(w - 1) / 2.0, (h - 1) / 2.0]], 0)[0]
            fov = scale * max(h, w) / 3600.0
            return SolveHint(float(ra) % 360.0, float(dec), search_radius=0.5 * fov,
                             scale=scale, scale_tol=0.03, source='header WCS')
        except Exception as e:
            _log.debug("header WCS hint failed: %s", e)
    for rk, dk, hours in (('RA', 'DEC', False), ('OBJCTRA', 'OBJCTDEC', True),
                          ('RA_OBJ', 'DEC_OBJ', False)):
        if header.get(rk) is not None and header.get(dk) is not None:
            ra = _sexagesimal(header[rk], hours)
            dec = _sexagesimal(header[dk], False)
            if ra is not None and dec is not None:
                fov = (scale or 2.0) * max(h, w) / 3600.0
                return SolveHint(ra % 360.0, dec, search_radius=max(1.0, fov),
                                 scale=scale, scale_tol=0.05 if scale else 0.0,
                                 source=f'{rk}/{dk} keywords')
    return None


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

@dataclass
class LocalSolution:
    crval: Tuple[float, float]
    crpix: Tuple[float, float]      # FITS 1-based
    cd: np.ndarray                  # 2x2, deg/px
    n_match: int
    rms_px: float
    scale_arcsec: float

    def header_cards(self) -> Dict[str, tuple]:
        return {
            'CTYPE1': ('RA---TAN', 'WCS axis 1: RA, gnomonic projection'),
            'CTYPE2': ('DEC--TAN', 'WCS axis 2: Dec, gnomonic projection'),
            'CRVAL1': (self.crval[0], 'RA at reference pixel (deg)'),
            'CRVAL2': (self.crval[1], 'Dec at reference pixel (deg)'),
            'CRPIX1': (self.crpix[0], 'X reference pixel (1-based)'),
            'CRPIX2': (self.crpix[1], 'Y reference pixel (1-based)'),
            'CD1_1': (float(self.cd[0, 0]), 'WCS CD matrix [1,1]'),
            'CD1_2': (float(self.cd[0, 1]), 'WCS CD matrix [1,2]'),
            'CD2_1': (float(self.cd[1, 0]), 'WCS CD matrix [2,1]'),
            'CD2_2': (float(self.cd[1, 1]), 'WCS CD matrix [2,2]'),
            'CUNIT1': ('deg', ''), 'CUNIT2': ('deg', ''),
            'EQUINOX': (2000.0, 'Equinox of coordinates'),
            'RADESYS': ('ICRS', 'Gaia DR3 reference frame'),
            'WCSORIG': ('local_solve', 'WCS source: OriginStack local Gaia index'),
            'PLTNSTAR': (self.n_match, 'Stars matched by the plate solve'),
            'PLTRMS': (round(self.rms_px, 3), 'Plate solve residual RMS (px)'),
        }


def _image_stars(lum: np.ndarray) -> np.ndarray:
    from src.star_detect import detect_stars_matched_filter
    img = np.ascontiguousarray(lum, dtype=np.float32)
    img = np.nan_to_num(img - np.float32(np.nanmedian(img)))
    # the detector's default threshold is tuned for registration on deep stacks; a
    # shallow or nebula-filled image can leave a handful of stars, so step it down
    # (as registration.registration_stars does for thin catalogues)
    s = None
    for k in (22.0, 12.0, 8.0, 6.0):
        s = detect_stars_matched_filter(img, k_confirm=k)
        if s is not None and len(s) >= 3 * IMG_STARS:
            break
    if s is None or len(s) == 0:
        return np.zeros((0, 3))
    xy = np.column_stack([s['xcentroid'], s['ycentroid'], s['flux']]).astype(float)
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    xy = xy[np.argsort(-xy[:, 2])]
    # A very bright star's halo, spikes and split core detect as a cluster of
    # "stars" that outrank the real field (a mag-1.7 star put 14 of the top 40
    # within 211 px of itself, and the field did not solve): drop fainter
    # detections within a brightness-scaled radius of each bright one.
    if len(xy) > 10:
        f_ref = float(np.median(xy[:min(len(xy), 100), 2]))
        keep = np.ones(len(xy), bool)
        for i in range(len(xy)):
            if not keep[i]:
                continue
            ratio = xy[i, 2] / max(f_ref, 1e-12)
            r = 10.0 if ratio < 10.0 else 25.0 * math.sqrt(ratio)
            d = np.hypot(xy[i + 1:, 0] - xy[i, 0], xy[i + 1:, 1] - xy[i, 1])
            keep[i + 1:] &= d > r
        xy = xy[keep]
    # drop the very brightest few: saturated cores centroid badly
    return xy[min(3, len(xy) // 10):]


def _as_sources(x, y, flux) -> np.ndarray:
    a = np.zeros(len(x), dtype=[('xcentroid', 'f8'), ('ycentroid', 'f8'), ('flux', 'f8')])
    a['xcentroid'], a['ycentroid'], a['flux'] = x, y, flux
    return a


def _nn_pairs(p: np.ndarray, q: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """Mutual nearest neighbours of point sets p -> q within *tol*."""
    from scipy.spatial import cKDTree
    if len(p) == 0 or len(q) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    d, j = cKDTree(q).query(p, distance_upper_bound=tol)
    ok = np.isfinite(d)
    i = np.nonzero(ok)[0]
    j = j[ok]
    _, back = cKDTree(p).query(q[j])
    keep = back == i
    return i[keep], j[keep]


def _fit_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 affine mapping src (n,2) -> dst (n,2), least squares."""
    A = np.column_stack([src, np.ones(len(src))])
    coef, *_ = np.linalg.lstsq(A, dst, rcond=None)
    M = np.eye(3)
    M[:2, :2] = coef[:2].T
    M[:2, 2] = coef[2]
    return M


def _apply(M: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ M[:2, :2].T + M[:2, 2]


def _trial(img_xy: np.ndarray, cat: np.ndarray, ra0: float, dec0: float,
           scale: float, parity: int, center: np.ndarray, n_cat: int):
    """One (centre, scale, parity) attempt. Returns (affine image->tangent-plane
    arcsec, n_inliers, rms_px) or None."""
    from src.blind_match import match_rigid_unknown_rotation
    xi, eta = project_tan(cat[:, 0], cat[:, 1], ra0, dec0)
    # catalogue in trial pixel units about the image centre
    u = parity * xi * 3600.0 / scale
    v = -eta * 3600.0 / scale                    # image rows grow downward-ish; parity covers the rest
    cat_px = np.column_stack([u, v])[:n_cat]
    cflux = 10.0 ** (-0.4 * cat[:n_cat, 2])
    src = img_xy[:IMG_STARS]
    rel = src[:, :2] - center
    res = match_rigid_unknown_rotation(_as_sources(rel[:, 0], rel[:, 1], src[:, 2]),
                                       _as_sources(cat_px[:, 0], cat_px[:, 1], cflux),
                                       max_stars=max(len(src), n_cat), pixel_tol=6.0,
                                       dist_rel_tol=0.012, min_inliers=MIN_INLIERS)
    if res is None:
        return None
    M = np.asarray(res.params, float)
    # refine: similarity guess -> affine by iterated mutual-NN least squares, on
    # every detected image star and the full catalogue at this trial scale
    all_rel = img_xy[:, :2] - center
    all_cat = np.column_stack([parity * xi * 3600.0 / scale, -eta * 3600.0 / scale])
    tol = 6.0
    i = j = None
    for _ in range(6):
        i, j = _nn_pairs(_apply(M, all_rel), all_cat, tol)
        if len(i) < MIN_INLIERS:
            return None
        M = _fit_affine(all_rel[i], all_cat[j])
        tol = max(2.0, tol * 0.6)
    r = _apply(M, all_rel[i]) - all_cat[j]
    rms = float(np.sqrt(np.mean(np.sum(r * r, axis=1))))
    # back to tangent-plane arcsec: undo the trial scale/parity
    to_arcsec = np.diag([parity * scale, -scale, 1.0])
    return to_arcsec @ M, len(i), rms, (i, j)


def _scales(hint: SolveHint) -> List[float]:
    if hint.scale and hint.scale_tol > 0:
        lo, hi = hint.scale * (1 - hint.scale_tol), hint.scale * (1 + hint.scale_tol)
    elif hint.scale:
        return [hint.scale]
    else:
        lo, hi = 0.3, 30.0
    n = int(math.ceil(math.log(hi / lo) / math.log(SCALE_STEP)))
    mid = math.sqrt(lo * hi)
    grid = sorted({mid * SCALE_STEP ** k for k in range(-n // 2 - 1, n // 2 + 2)
                   if lo * 0.99 <= mid * SCALE_STEP ** k <= hi * 1.01})
    return sorted(grid, key=lambda s: abs(math.log(s / (hint.scale or mid))))


def solve_local(lum: np.ndarray, hint: SolveHint, fetch: bool = True,
                root: Optional[str] = None, verbose: bool = False,
                time_budget_s: float = 120.0) -> Optional[LocalSolution]:
    """Plate-solve *lum* (2-D) near *hint*. Returns the solution or None."""
    import time
    t0 = time.time()
    h, w = lum.shape
    img_xy = _image_stars(lum)
    if len(img_xy) < MIN_INLIERS:
        if verbose:
            safe_print(f"  [local solve] only {len(img_xy)} stars detected -- cannot solve")
        return None
    center = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    scales = _scales(hint)
    max_scale = max(scales)
    half_diag = 0.5 * math.hypot(w, h) * max_scale / 3600.0
    field_min = min(w, h) * min(scales) / 3600.0
    # candidate centres: the hint, then rings at half-field steps out to the search radius
    step = max(0.5 * field_min, 0.05)
    cands = [(hint.ra, hint.dec)]
    nr = int(math.ceil(hint.search_radius / step)) if hint.search_radius > 0 else 0
    for k in range(1, nr + 1):
        n_on = max(6, int(round(2 * math.pi * k)))
        for q in range(n_on):
            a = 2 * math.pi * q / n_on
            ra, dec = deproject_tan(k * step * math.cos(a), k * step * math.sin(a),
                                    hint.ra, hint.dec)
            cands.append((float(ra), float(dec)))
    big, cov = cone_catalog(hint.ra, hint.dec, hint.search_radius + half_diag * 1.2,
                            fetch=fetch, root=root)
    if verbose:
        safe_print(f"  [local solve] {len(img_xy)} image stars, {len(big)} index stars "
                   f"({cov * 100:.0f}% tile coverage), {len(cands)} centre(s) x "
                   f"{len(scales)} scale(s) x 2 parities; hint from {hint.source or 'caller'}")
    if len(big) < MIN_INLIERS:
        return None
    n_cat = int(np.clip(1.5 * min(len(img_xy), IMG_STARS), 30, 80))
    best = None
    for ra0, dec0 in cands:
        if time.time() - t0 > time_budget_s:
            break
        sep = angular_sep_deg(big[:, 0], big[:, 1], ra0, dec0)
        cat = big[sep <= half_diag * 1.05]
        if len(cat) < MIN_INLIERS:
            continue
        for scale in scales:
            for parity in (1, -1):
                if time.time() - t0 > time_budget_s:
                    break
                try:
                    r = _trial(img_xy, cat, ra0, dec0, scale, parity, center, n_cat)
                except Exception as e:           # a degenerate trial must not end the search
                    _log.debug("local solve trial failed: %s", e)
                    r = None
                if r is None:
                    continue
                M, n, rms, _pairs = r
                need = max(MIN_INLIERS, int(0.25 * min(len(img_xy), len(cat))))
                if n >= need and rms <= MAX_RMS_PX:
                    if best is None or n > best[1]:
                        best = (M, n, rms, ra0, dec0, cat)
                    if n >= 0.5 * min(len(img_xy), len(cat)):
                        break
            if best is not None and best[1] >= 0.5 * min(len(img_xy), len(best[5])):
                break
        if best is not None:
            break
    if best is None:
        if verbose:
            safe_print(f"  [local solve] no match ({time.time() - t0:.1f}s)")
        return None
    M, n, rms, ra0, dec0, cat = best
    # re-centre the tangent point on the image centre and refit once (the trial
    # centre can be a field-width off; a TAN projection about the wrong point
    # bends the mapping away from the affine the fit assumes)
    xi_c, eta_c = _apply(M, np.zeros((1, 2)))[0] / 3600.0
    ra_c, dec_c = deproject_tan(xi_c, eta_c, ra0, dec0)
    ra_c, dec_c = float(ra_c), float(dec_c)
    all_rel = img_xy[:, :2] - center
    for _ in range(2):
        xi, eta = project_tan(big[:, 0], big[:, 1], ra_c, dec_c)
        cat_as = np.column_stack([xi, eta]) * 3600.0
        guess = M.copy()
        guess[:2, 2] = 0.0
        i, j = _nn_pairs(_apply(guess, all_rel), cat_as, 3.0 * np.sqrt(abs(np.linalg.det(M[:2, :2]))))
        if len(i) < MIN_INLIERS:
            break
        M = _fit_affine(all_rel[i], cat_as[j])
        resid = _apply(M, all_rel[i]) - cat_as[j]
        px = np.sqrt(abs(np.linalg.det(M[:2, :2])))
        rms = float(np.sqrt(np.mean(np.sum(resid * resid, axis=1)))) / px
        n = len(i)
        xi_c, eta_c = _apply(M, np.zeros((1, 2)))[0] / 3600.0
        ra_c, dec_c = (float(v) for v in deproject_tan(xi_c, eta_c, ra_c, dec_c))
    cd = M[:2, :2] / 3600.0
    # FITS: CD maps (pixel - CRPIX) -> (xi, eta); xi grows with RA
    sol = LocalSolution(crval=(ra_c % 360.0, dec_c), crpix=(center[0] + 1.0, center[1] + 1.0),
                        cd=cd, n_match=n, rms_px=rms,
                        scale_arcsec=float(np.sqrt(abs(np.linalg.det(cd))) * 3600.0))
    if verbose:
        safe_print(f"  [local solve] solved in {time.time() - t0:.1f}s: RA {sol.crval[0]:.5f} "
                   f"Dec {sol.crval[1]:+.5f}, {sol.scale_arcsec:.3f}\"/px, "
                   f"{n} stars, rms {rms:.2f} px")
    return sol


_WCS_KEYS_TO_CLEAR = ('CDELT1', 'CDELT2', 'CROTA1', 'CROTA2', 'PC1_1', 'PC1_2', 'PC2_1',
                      'PC2_2', 'ORIENTAT', 'A_ORDER', 'B_ORDER', 'AP_ORDER', 'BP_ORDER')


def solve_header(lum: np.ndarray, header, verbose: bool = False,
                 hint: Optional[SolveHint] = None, fetch: bool = True) -> bool:
    """Solve *lum* and write a TAN WCS into *header*. Returns success."""
    hint = hint or hint_from_header(header, lum.shape)
    if hint is None:
        if verbose:
            safe_print("  [local solve] no position hint (no WCS, RA/DEC or OBJCTRA in the "
                       "header) -- the local solver needs one")
        return False
    sol = solve_local(lum, hint, fetch=fetch, verbose=verbose)
    if sol is None:
        return False
    for k in _WCS_KEYS_TO_CLEAR:
        if k in header:
            del header[k]
    for k, (v, c) in sol.header_cards().items():
        header[k] = (v, c)
    header['PLTSOLVD'] = (True, 'Plate solving successful')
    header['PLTSOLVR'] = ('local', 'Plate solver used (Gaia DR3 index)')
    return True


def build_index(tiles: Sequence[Tuple[int, int]], root: Optional[str] = None,
                refresh: bool = False, progress=None) -> Tuple[int, int]:
    """Fetch *tiles* into the index. Returns (fetched, failed)."""
    ok = bad = 0
    for k, (b, r) in enumerate(tiles):
        if not refresh and os.path.exists(_tile_path(b, r, root)):
            continue
        if fetch_tile(b, r, root) is None:
            bad += 1
        else:
            ok += 1
        if progress:
            progress(k + 1, len(tiles), ok, bad)
    return ok, bad
