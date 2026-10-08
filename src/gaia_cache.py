"""Local cache of Gaia DR3 cone queries (``net_query.gaia_cone_search``).

Colour calibration, ``--photometry`` and ``--photometry-timeseries`` each ask
Gaia for the brightest stars of the field, 8-12 s a query over the network.
The same target comes back on other nights, through ``--from-stack``
reprocessing and through several of those steps in one run, so every result
is kept under ``star_index_dir()/gaia_cones`` and reused when -- and only
when -- the cached rows are what the server would return (the same stars in
the same brightness order; stars with exactly equal G may come back in another
order, as the server's own order among ties depends on the query):

- the cached query was brightest-first (``ORDER BY phot_g_mean_mag``) with
  the same ``require_not_null`` columns, holds every requested column, and its
  ``min_mag`` is absent or no fainter than the request's;
- its cone contains the requested cone;
- either it was not truncated by its ``TOP n`` (it holds every star of its
  cone), or at least ``max_rows`` of its rows fall in the requested cone and
  magnitude range -- then they are exactly the request's brightest
  ``max_rows``.

Anything else goes to the network, and that result is cached in turn. A query
always fetches ``ra``, ``dec`` and ``phot_g_mean_mag`` as well, so later
requests for a smaller cone or a fainter slice can be filtered. Off with
``$ORIGINSTACK_NO_GAIA_CACHE``. A hit works under ``--offline`` too: nothing
leaves the machine.
"""
from __future__ import annotations

import glob
import json
import logging
import math
import os
import time
import uuid
from typing import List, Optional

import numpy as np

_log = logging.getLogger('originstack')

ORDER_COLUMN = 'phot_g_mean_mag'
KEY_COLUMNS = ('ra', 'dec', ORDER_COLUMN)
MAX_ENTRIES = 400


def enabled() -> bool:
    return not os.environ.get('ORIGINSTACK_NO_GAIA_CACHE')


def cache_dir() -> str:
    from src.local_solve import star_index_dir
    return os.path.join(star_index_dir(), 'gaia_cones')


def _sep_deg(ra1, dec1, ra2, dec2) -> float:
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    c = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _sep_deg_arr(ra, dec, ra0, dec0) -> np.ndarray:
    r, d = np.radians(ra), np.radians(dec)
    r0, d0 = math.radians(ra0), math.radians(dec0)
    c = np.sin(d) * math.sin(d0) + np.cos(d) * math.cos(d0) * np.cos(r - r0)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def _entries():
    for path in glob.glob(os.path.join(cache_dir(), '*.npz')):
        try:
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z['__meta__']))
            yield path, meta
        except Exception:
            continue


def padded(radius: float, max_rows: int):
    """Cone radius and row count to fetch on a miss: ~5% wider (night-to-night
    pointing offsets on the Origin are under 0.03 deg) and 25% more rows, so the
    request is still answered from the entry and a revisit lands inside it."""
    return radius * 1.05 + 0.01, int(math.ceil(max_rows * 1.25))


def _meta(ra, dec, radius, columns, max_rows, n, require_not_null, min_mag) -> dict:
    return dict(ra=float(ra), dec=float(dec), radius=float(radius), columns=list(columns),
                order_by=ORDER_COLUMN, require_not_null=sorted(set(require_not_null or [])),
                min_mag=None if min_mag is None else float(min_mag),
                truncated=n >= int(max_rows), created=time.time())


def _answer(cols: dict, meta: dict, ra, dec, radius, columns, max_rows,
            require_not_null, min_mag):
    """The request answered from one entry, or None if this entry cannot."""
    if meta.get('order_by') != ORDER_COLUMN:
        return None
    if meta.get('require_not_null') != sorted(set(require_not_null or [])):
        return None
    if not set(columns) <= set(meta.get('columns', [])):
        return None
    cmin = meta.get('min_mag')
    if cmin is not None and (min_mag is None or min_mag < cmin):
        return None
    if _sep_deg(ra, dec, meta['ra'], meta['dec']) + radius > meta['radius'] + 1e-9:
        return None
    sel = _sep_deg_arr(np.asarray(cols['ra'], float), np.asarray(cols['dec'], float), ra, dec) <= radius
    if min_mag is not None:
        # The server compares its float32 (REAL) column with the double literal;
        # the JSON carries the float32's shortest decimal, so round back to float32
        # first -- a star exactly at min_mag (the previous slice's faintest) otherwise
        # goes the other way.
        g = np.asarray(cols[ORDER_COLUMN], float).astype(np.float32).astype(np.float64)
        sel &= g > float(min_mag)
    idx = np.flatnonzero(sel)
    if len(idx) >= max_rows:
        idx = idx[:max_rows]               # rows are stored brightest first
    elif meta.get('truncated', True):
        return None                         # the cone may hold stars this entry never saw
    from astropy.table import Table
    return Table({c: np.asarray(cols[c])[idx] for c in columns})


def lookup(ra: float, dec: float, radius: float, columns: List[str], max_rows: int,
           require_not_null: Optional[List[str]], min_mag: Optional[float],
           entry_table=None, entry_rows: Optional[int] = None,
           entry_radius: Optional[float] = None, entry_centre=None):
    """The cached answer to a brightest-first Gaia cone query, or None.
    ``entry_table`` (with its query's row cap, radius and centre): answer from
    that table only, e.g. the padded fetch just made."""
    if entry_table is not None:
        cols = {c: entry_table[c] for c in entry_table.colnames}
        meta = _meta(entry_centre[0], entry_centre[1], entry_radius, entry_table.colnames,
                     entry_rows, len(entry_table), require_not_null, min_mag)
        return _answer(cols, meta, ra, dec, radius, columns, max_rows, require_not_null, min_mag)
    if not enabled():
        return None
    for path, meta in _entries():
        try:
            with np.load(path, allow_pickle=False) as z:
                cols = {k: z[k] for k in z.files if k != '__meta__'}
        except Exception:
            continue
        ans = _answer(cols, meta, ra, dec, radius, columns, max_rows, require_not_null, min_mag)
        if ans is None:
            continue
        try:
            os.utime(path)                  # most recently used survives pruning
        except OSError:
            pass
        _log.debug("Gaia cache hit: %s (%d rows)", os.path.basename(path), len(ans))
        return ans
    return None


def store(table, ra: float, dec: float, radius: float, max_rows: int,
          require_not_null: Optional[List[str]], min_mag: Optional[float]) -> None:
    """Keep a brightest-first query result (``table`` must hold ``KEY_COLUMNS``;
    ``radius``/``max_rows`` are the query's own)."""
    if not enabled() or table is None:
        return
    try:
        d = cache_dir()
        os.makedirs(d, exist_ok=True)
        cols = {}
        for c in table.colnames:
            a = np.asarray(table[c])
            cols[c] = a.astype(str) if a.dtype == object else a
        meta = _meta(ra, dec, radius, table.colnames, max_rows, len(table),
                     require_not_null, min_mag)
        tmp = os.path.join(d, f'.{uuid.uuid4().hex}.npz')
        np.savez_compressed(tmp, __meta__=np.array(json.dumps(meta)), **cols)
        os.replace(tmp, os.path.join(d, f'{uuid.uuid4().hex}.npz'))
        _prune(d)
    except Exception as exc:          # a cache must never fail a run
        _log.debug("Gaia cache store failed: %s", exc)


def _prune(d: str) -> None:
    files = sorted(glob.glob(os.path.join(d, '*.npz')), key=os.path.getmtime)
    for p in files[:max(0, len(files) - MAX_ENTRIES)]:
        try:
            os.remove(p)
        except OSError:
            pass
