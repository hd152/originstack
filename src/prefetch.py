"""Fetch the sky catalogues a run will need while the CPU is busy stacking.

After Phase 3 a run waits on the network twice: ``pipeline._settle_stack_wcs`` fetches the
Gaia index tiles under the field (``local_solve``; ~3.7 s per cold tile) and colour
calibration queries a Gaia cone (``color_calibrate.query_gaia_stars``; ~3.1 s cold). Both
land in on-disk caches, and the field is known before Phase 1 (the capture app's
``info.json`` solve, or the pointing keywords). ``start`` therefore fills those caches in a
background thread; ``join`` waits for it before the first use, so the main thread never
races it on a half-written tile. Nothing here changes a result: the main thread asks for
the same catalogues and finds them cached, and every failure is logged and ignored.
"""
import logging
import math
import threading
from typing import List, Optional, Tuple

_log = logging.getLogger("originstack")

# Slack on the cone so the cache still contains the cone the stack's refined WCS asks for:
# colour calibration queries 1.1 x the stack's corner distance (the stack can be wider than
# the first sub's field once --full-field extends it) and the refined centre moves a little.
_RADIUS_MARGIN = 1.25
# The cache answers a smaller cone from a larger one only when it then still holds the rows
# the caller asks for (``gaia_cache``: a truncated entry is the brightest rows of the *big*
# cone), so ask for more than colour calibration's 500.
_GAIA_ROWS_FACTOR = 2
# How long the main thread waits for the index tiles before fetching them itself. A slow archive
# (a cold tile took 3.7 s, 14 s and 64 s on three runs on one day) must not hold the run up: after
# this the main thread fetches too, and tile writes are safe against that (unique temp names).
_JOIN_TIMEOUT_S = 10.0


def field_from_session(args, lights) -> Optional[Tuple[float, float, float]]:
    """``(ra, dec, radius)`` in degrees of the field the lights cover, or None."""
    si = getattr(args, '_session_info', None)
    if si is not None and si.has_wcs:
        diag = math.hypot(math.degrees(si.fov_x_rad), math.degrees(si.fov_y_rad))
        return (math.degrees(si.ra_rad) % 360.0, math.degrees(si.dec_rad), 0.5 * diag)
    for f in lights or []:
        hdr = getattr(f, 'header', None) or {}
        try:
            h, w = int(hdr.get('NAXIS2') or 0), int(hdr.get('NAXIS1') or 0)
            if not (h and w):
                continue
            from src.local_solve import hint_from_header
            hint = hint_from_header(hdr, (h, w))
        except Exception:
            _log.debug("prefetch.field_from_session: header hint failed", exc_info=True)
            continue
        if hint is not None and hint.scale:
            return (hint.ra, hint.dec, 0.5 * hint.scale * math.hypot(h, w) / 3600.0)
        break
    return None


def _cone_query(method: str) -> dict:
    """The arguments of the first Gaia cone query the colour calibration ``method`` sends: the
    default (``solar``) goes through ``photometry.match_gaia_field``; ``colorindex``/``spcc`` through
    ``color_calibrate.query_gaia_stars``. Rows are multiplied (``_GAIA_ROWS_FACTOR``). The later,
    deeper queries (a magnitude cut taken from the stack) cannot be known up front."""
    if method in ('colorindex', 'spcc'):
        from src.color_calibrate import GAIA_COLUMNS, GAIA_MAX_STARS, GAIA_NOT_NULL
        return dict(columns=list(GAIA_COLUMNS), max_rows=_GAIA_ROWS_FACTOR * GAIA_MAX_STARS,
                    require_not_null=list(GAIA_NOT_NULL))
    from src.photometry import GAIA_FIELD_COLUMNS, GAIA_FIELD_NOT_NULL, GAIA_FIELD_ROWS
    return dict(columns=list(GAIA_FIELD_COLUMNS), max_rows=_GAIA_ROWS_FACTOR * GAIA_FIELD_ROWS,
                require_not_null=list(GAIA_FIELD_NOT_NULL))


def _tiles_missing(ra: float, dec: float, radius: float) -> List[Tuple[int, int]]:
    from src import local_solve
    return [t for t in local_solve.tiles_for_cone(ra, dec, radius)
            if local_solve.load_tile(*t) is None]


def _run(field, want_tiles: bool, want_gaia: bool, cancel, tiles_done=None, method: str = 'solar') -> None:
    """Tiles first (``_settle_stack_wcs`` needs them right after Phase 3; ``tiles_done`` is set when
    that stage is over, however it ended), then the colour-calibration cone (needed in Phase 4)."""
    from src import net_query
    ra, dec, radius = field
    cancelled = (lambda: bool(cancel is not None and cancel.is_set()))
    try:
        if want_tiles:
            from src import local_solve
            for band, r in _tiles_missing(ra, dec, radius * _RADIUS_MARGIN):
                if cancelled() or net_query.is_offline():
                    return
                local_solve.fetch_tile(band, r)
    finally:
        if tiles_done is not None:
            tiles_done.set()
    if want_gaia and not cancelled():
        net_query.gaia_cone_search(ra, dec, radius * _RADIUS_MARGIN, **_cone_query(method))


def start(args, lights) -> Optional[threading.Thread]:
    """Start the background fetch (stored on ``args._catalog_prefetch``, with the event for its
    tile stage on ``args._catalog_prefetch_tiles``); None when there is nothing to fetch (offline,
    field unknown, nothing needs a catalogue)."""
    try:
        from src import net_query
        if getattr(args, 'offline', False) or net_query.is_offline():
            return None
        want_tiles = bool(getattr(args, 'wcs_refine', True))
        want_gaia = bool(getattr(args, 'color_calibrate', False))
        if not (want_tiles or want_gaia):
            return None
        field = field_from_session(args, lights)
        if field is None:
            return None

        tiles_done = threading.Event()

        def body():
            try:
                _run(field, want_tiles, want_gaia, getattr(args, '_cancel_event', None), tiles_done,
                     str(getattr(args, 'color_calibrate_method', 'solar') or 'solar'))
            except Exception:
                _log.debug("catalogue prefetch failed", exc_info=True)
            finally:
                tiles_done.set()

        th = threading.Thread(target=body, name='catalog-prefetch', daemon=True)
        th.start()
        args._catalog_prefetch = th
        args._catalog_prefetch_tiles = tiles_done
        return th
    except Exception:
        _log.debug("prefetch.start failed", exc_info=True)
        return None


def join(args, timeout: float = _JOIN_TIMEOUT_S) -> None:
    """Wait (at most ``timeout`` s) for the prefetch's tile stage before the main thread reads the
    index. The cone stage keeps running in the background: Phase 4 asks for it much later, and the
    Gaia cache answers it whenever it has landed."""
    ev = getattr(args, '_catalog_prefetch_tiles', None)
    if ev is not None:
        ev.wait(timeout)
    args._catalog_prefetch = None
    args._catalog_prefetch_tiles = None
