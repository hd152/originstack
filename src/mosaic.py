"""Mosaic stitching for multi-panel sessions (``--mosaic``, or detected automatically).

Workflow
--------
Each panel subfolder is stacked on its own through Phases 1-3 (Phase 4 deferred, as for
a multi-night combine). ``stitch_mosaic_panels`` then:

  1. reads every panel's linear stack and its WCS (the session solve refined against
     Gaia, or a plate solve);
  2. builds one TAN grid covering all panels (median pixel scale, the deepest panel's
     orientation);
  3. reprojects each panel onto it (an exact WCS-to-WCS map on a coarse grid,
     interpolated -- the TAN-to-TAN map is smooth -- then cubic resampling), keeping
     only the panel's own footprint;
  4. matches the panels to each other in their overlaps with one least-squares solve
     for all of them at once: a gain per panel and channel (transparency, exposure),
     then a plane per panel and channel (residual gradient), the deepest panel fixed;
  5. blends with feathered, inverse-noise-variance weights;
  6. writes a linear mosaic FITS (``RAWSTACK``, ``MOSAIC``); the caller runs Phase 4
     once on it, so background extraction and the stretch see one image, not N.

No dependency beyond numpy/scipy/astropy (``reproject`` used to be required and was not
in the packaged app, so ``--mosaic`` fell back to an ordinary combine there).

``predict_mosaic`` decides from metadata alone whether a folder of sessions is a mosaic:
sessions pointed more than ``_SAME_POINTING_FRAC`` of a field apart whose fields
overlap. The Celestron Origin's own Mosaic Mode (firmware 1.4.6084 and later) captures
and stacks every tile as a session; when the sessions carry its mosaic description
(``mosaic_manifest``: a shared ``MosaicUuid`` in ``info.json``, or a JSON tile list whose
UUIDs are the sessions' ``StackedInfo.uuid``), that decides exactly and missing tiles are
reported. The schema comes from the firmware; where it is stored on disk has not been
seen on real data yet, so pointing remains the fallback.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits

from src.utils import safe_print

try:
    from astropy.wcs import WCS
    HAS_WCS = True
except Exception:
    HAS_WCS = False

try:
    from scipy import ndimage as ndi
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# Sessions closer than this fraction of the (smaller) field are the same pointing:
# nights of one target differ by 7-70 px of 2048 (< 0.04), up to 0.15 when re-centred,
# and a mis-aimed Whirlpool session sat 0.32 off (0.25 made it a two-panel "mosaic" of
# nine nights); a mosaic tile steps ~0.6-0.9 (the Origin's M24 tiles: 0.80).
_SAME_POINTING_FRAC = 0.5
# Two pointings can overlap if they are closer than this fraction of the larger field.
_OVERLAP_FRAC = 0.95
_MAP_STEP = 32            # px between exact WCS evaluations of the reprojection map
_TRIM_PX = 4              # dropped from each panel's footprint edge (warp ringing, crop rim)
_FEATHER_FRAC = 0.12      # feather ramp, as a fraction of the panel's smaller side
_BLOCK = 16               # px; overlap differences are fitted on block medians
_MIN_OVERLAP_BLOCKS = 30
_TILE_KEYS = ('mosaicuuid', 'tilerow', 'tilecol', 'gridwidth', 'gridheight', 'tilesize',
              'tilestride', 'mosaicorientation', 'compositionuuid')


# ── detection ───────────────────────────────────────────────────────────────

def _walk_keys(obj, out: Dict[str, object]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in _TILE_KEYS:
                out[k.lower()] = v
            _walk_keys(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk_keys(v, out)


def session_pointing(d: str) -> Optional[dict]:
    """``{'ra', 'dec', 'fov_x', 'fov_y'}`` (radians) of a session folder from its
    ``info.json``, plus any Origin mosaic tile keys found in it, or None."""
    path = os.path.join(d, 'info.json')
    try:
        with open(path, encoding='utf-8') as fh:
            info = json.load(fh)
    except Exception:
        return None
    si = info.get('StackedInfo', info) if isinstance(info, dict) else {}
    try:
        cel = si['celestial']
        out = {'ra': float(cel['first']), 'dec': float(cel['second']),
               'fov_x': float(si['fovX']), 'fov_y': float(si['fovY'])}
    except Exception:
        return None
    if not (out['fov_x'] > 0 and out['fov_y'] > 0):
        return None
    tiles: Dict[str, object] = {}
    _walk_keys(info, tiles)
    if tiles:
        out['tile'] = tiles
    return out


def _separation(a: dict, b: dict) -> float:
    """Angular separation (radians)."""
    s = (math.sin(a['dec']) * math.sin(b['dec'])
         + math.cos(a['dec']) * math.cos(b['dec']) * math.cos(a['ra'] - b['ra']))
    return math.acos(max(-1.0, min(1.0, s)))


def session_uuid(d: str) -> Optional[str]:
    """The session's stacked-master UUID (``info.json`` ``StackedInfo.uuid``): what the
    Origin's mosaic tile list refers to each tile by."""
    try:
        with open(os.path.join(d, 'info.json'), encoding='utf-8') as fh:
            info = json.load(fh)
        u = (info.get('StackedInfo') or info).get('uuid')
        return str(u).lower() if u else None
    except Exception:
        return None


def _tile_entries(obj, out: list) -> None:
    """Every dict under ``obj`` that has TileRow and TileCol (an Origin mosaic tile)."""
    if isinstance(obj, dict):
        keys = {k.lower(): k for k in obj if isinstance(k, str)}
        if 'tilerow' in keys and 'tilecol' in keys:
            out.append(obj)
        for v in obj.values():
            _tile_entries(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _tile_entries(v, out)


def _uuids_in(obj) -> List[str]:
    """String values of any ``*uuid*`` key directly in ``obj`` (the tile list's own field
    name for the stacked master is not known from the firmware, only that it is a UUID)."""
    return [str(v).lower() for k, v in obj.items()
            if isinstance(k, str) and 'uuid' in k.lower() and isinstance(v, str) and v]


def _read_json(path: str):
    try:
        if os.path.getsize(path) > 4 << 20:
            return None
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except Exception:
        return None


def mosaic_manifest(subdirs: List[str]) -> Optional[dict]:
    """The Origin's own mosaic description, when the sessions carry one: ``{'mosaic':
    uuid or None, 'tiles': {dir: (row, col)}, 'grid': (w, h) or None, 'total': int or
    None, 'missing': [(row, col)], 'others': [dir]}``, else None.

    Two sources, both in the schema the firmware validates (``MosaicParameters``:
    CenterRa/CenterDec/Orientation/MosaicWidth/MosaicHeight/TileStride/FeatherRadius;
    ``MosaicMetadata``: GridWidth/GridHeight/TilesTotal/TilesCompleted; tiles: TileRow,
    TileCol and a stacked-master UUID): tile keys inside the sessions' own ``info.json``
    (sessions sharing a MosaicUuid are one mosaic), or any JSON file in the parent folder
    or a session folder listing tiles whose UUIDs are the sessions' ``StackedInfo.uuid``.
    Where the firmware stores it on disk has not been seen yet; this reads both places."""
    if len(subdirs) < 2:
        return None
    uuid_of = {d: session_uuid(d) for d in subdirs}
    dir_of = {u: d for d, u in uuid_of.items() if u}
    tiles: Dict[str, Tuple[int, int]] = {}
    info_tiles: Dict[str, Tuple[int, int]] = {}
    meta: Dict[str, object] = {}
    # (1) the sessions' own info.json
    by_mosaic: Dict[str, List[str]] = {}
    for d in subdirs:
        p = session_pointing(d) or {}
        t = p.get('tile') or {}
        if 'mosaicuuid' in t:
            by_mosaic.setdefault(str(t['mosaicuuid']).lower(), []).append(d)
            if 'tilerow' in t and 'tilecol' in t:
                try:
                    info_tiles[d] = (int(t['tilerow']), int(t['tilecol']))
                except (TypeError, ValueError):
                    pass
            for k in ('gridwidth', 'gridheight'):
                if k in t:
                    meta[k] = t[k]
    mosaic_id = None
    if by_mosaic:
        mosaic_id, members = max(by_mosaic.items(), key=lambda kv: len(kv[1]))
        if len(members) < 2:
            mosaic_id = None           # every session its own "mosaic": nothing in common
        else:
            tiles.update({d: info_tiles[d] for d in members if d in info_tiles})
    # (2) tile lists in JSON files of the parent and session folders
    parent = os.path.dirname(os.path.normpath(subdirs[0]))
    files = [os.path.join(parent, f) for f in sorted(os.listdir(parent)) if f.lower().endswith('.json')] \
        if os.path.isdir(parent) else []
    for d in subdirs:
        files += [os.path.join(d, f) for f in sorted(os.listdir(d))
                  if f.lower().endswith('.json') and f.lower() != 'info.json']
    for f in files:
        obj = _read_json(f)
        if obj is None:
            continue
        entries: list = []
        _tile_entries(obj, entries)
        matched = 0
        for e in entries:
            keys = {k.lower(): k for k in e if isinstance(k, str)}
            try:
                rc = (int(e[keys['tilerow']]), int(e[keys['tilecol']]))
            except (TypeError, ValueError):
                continue
            for u in _uuids_in(e):
                if u in dir_of:
                    tiles.setdefault(dir_of[u], rc)
                    matched += 1
                    break
        if matched >= 2:
            found: Dict[str, object] = {}
            _walk_keys(obj, found)
            for k in ('gridwidth', 'gridheight'):
                if k in found:
                    meta[k] = found[k]
            _total = _find_key(obj, 'tilestotal')
            if _total is not None:
                meta['total'] = _total
            if mosaic_id is None:
                mid = _find_key(obj, 'mosaicuuid')
                mosaic_id = str(mid).lower() if mid else None
    if len(tiles) < 2 and mosaic_id is None:
        return None
    members = set(tiles)
    if mosaic_id is not None:
        members |= set(by_mosaic.get(mosaic_id, []))
    if len(members) < 2:
        return None
    grid = None
    try:
        if 'gridwidth' in meta and 'gridheight' in meta:
            grid = (int(meta['gridwidth']), int(meta['gridheight']))
    except (TypeError, ValueError):
        grid = None
    total = None
    try:
        total = int(meta['total']) if 'total' in meta else (grid[0] * grid[1] if grid else None)
    except (TypeError, ValueError):
        total = None
    missing = []
    if grid:
        have = set(tiles.values())
        missing = [(r, c) for r in range(grid[1]) for c in range(grid[0]) if (r, c) not in have]
    return {'mosaic': mosaic_id, 'tiles': {d: tiles.get(d) for d in sorted(members)},
            'grid': grid, 'total': total, 'missing': missing,
            'others': [d for d in subdirs if d not in members]}


def _find_key(obj, key: str):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() == key:
                return v
            r = _find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def predict_mosaic(subdirs: List[str]) -> Optional[dict]:
    """``{'clusters': [[dir, ...], ...], 'reason': str}`` when ``subdirs`` look like the
    panels of one mosaic, else None. The Origin's own mosaic description wins when the
    sessions carry one (``mosaic_manifest``: exact, with missing tiles reported);
    otherwise panels = groups of sessions at one pointing (several nights of the same
    tile stay one group), and it is a mosaic when there are at least two groups and every
    group overlaps another, so the panels form one connected sky area. Unreadable
    metadata, or groups that do not overlap (different targets), give None."""
    man = mosaic_manifest(subdirs)
    if man is not None:
        groups: Dict[object, List[str]] = {}
        for d, rc in man['tiles'].items():
            groups.setdefault(rc if rc is not None else d, []).append(d)
        reason = (f"Origin mosaic{' ' + man['mosaic'][:8] if man['mosaic'] else ''}: "
                  f"{len(groups)} tile(s) from {len(man['tiles'])} session(s)")
        if man['grid']:
            reason += f", grid {man['grid'][0]}x{man['grid'][1]}"
        if man['missing']:
            reason += f"; missing tile(s) {', '.join(f'[{r},{c}]' for r, c in man['missing'])}"
        if man['others']:
            reason += f"; {len(man['others'])} other session(s) not part of it"
        return {'clusters': list(groups.values()), 'reason': reason, 'exact': True,
                'manifest': man}
    pts = [(d, session_pointing(d)) for d in subdirs]
    if len(pts) < 2 or any(p is None for _, p in pts):
        return None
    fov_small = min(min(p['fov_x'], p['fov_y']) for _, p in pts)
    fov_large = max(max(p['fov_x'], p['fov_y']) for _, p in pts)
    clusters: List[List[Tuple[str, dict]]] = []
    for d, p in pts:
        for c in clusters:
            if _separation(c[0][1], p) < _SAME_POINTING_FRAC * fov_small:
                c.append((d, p))
                break
        else:
            clusters.append([(d, p)])
    tiled = any('tile' in p for _, p in pts)
    if len(clusters) < 2:
        return None
    # connectivity of the overlap graph between pointings
    n = len(clusters)
    seen, todo = {0}, [0]
    while todo:
        i = todo.pop()
        for j in range(n):
            if j not in seen and _separation(clusters[i][0][1], clusters[j][0][1]) < _OVERLAP_FRAC * fov_large:
                seen.add(j)
                todo.append(j)
    if len(seen) != n:
        return None
    steps = [_separation(clusters[i][0][1], clusters[j][0][1]) / fov_small
             for i in range(n) for j in range(i + 1, n)]
    reason = (f"{n} pointings {min(steps):.2f}-{max(steps):.2f} fields apart with "
              f"overlapping fields" + (" (Origin mosaic tile keys present)" if tiled else ""))
    return {'clusters': [[d for d, _ in c] for c in clusters], 'reason': reason}


# ── reprojection ────────────────────────────────────────────────────────────

def _load_panel(path: str) -> Tuple[np.ndarray, fits.Header]:
    """A panel FITS as (H, W, 3) float32 + header."""
    with fits.open(path, memmap=False) as hd:
        hdr = hd[0].header.copy()
        data = np.array(hd[0].data, dtype=np.float32)
    if data.ndim == 2:
        data = np.stack([data] * 3, axis=-1)
    elif data.ndim == 3 and data.shape[0] == 3:
        data = np.ascontiguousarray(np.moveaxis(data, 0, -1))
    return data, hdr


def _panel_wcs(hdr: fits.Header) -> "WCS":
    """A 2-axis celestial WCS from a panel header (built with naxis=2: the data is a
    (3, H, W) cube)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WCS(hdr, naxis=2)


def _has_wcs(hdr: fits.Header) -> bool:
    return bool(hdr.get('PLTSOLVD', False)) or all(k in hdr for k in ('CTYPE1', 'CRVAL1', 'CRPIX1'))


def _pixel_scale_deg(w: "WCS") -> float:
    m = w.pixel_scale_matrix
    return float(math.sqrt(abs(np.linalg.det(m))))


def _output_wcs(wcss: List["WCS"], shapes: List[Tuple[int, int]], ref: int) -> Tuple["WCS", Tuple[int, int]]:
    """A TAN grid covering every panel: the reference (deepest) panel's own projection --
    tangent point, scale, orientation, parity -- extended by whole pixels, so that panel
    lands on the grid without being resampled and only the others are interpolated."""
    r = wcss[ref]
    out = WCS(naxis=2)
    out.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    out.wcs.crval = list(r.wcs.crval)
    out.wcs.cd = r.pixel_scale_matrix
    out.wcs.crpix = list(r.wcs.crpix)
    xs, ys = [], []
    for w, (h, wd) in zip(wcss, shapes):
        t = np.linspace(0, 1, 9)
        ex = np.concatenate([t * (wd - 1), np.full(9, wd - 1.0), t[::-1] * (wd - 1), np.zeros(9)])
        ey = np.concatenate([np.zeros(9), t * (h - 1), np.full(9, h - 1.0), t[::-1] * (h - 1)])
        ra, dec = w.pixel_to_world_values(ex, ey)
        px, py = out.world_to_pixel_values(ra, dec)
        xs.append(px)
        ys.append(py)
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    # (1e-6: a panel edge at -1e-10 from rounding must not add an empty border row)
    x0, y0 = math.floor(float(xs.min()) + 1e-6), math.floor(float(ys.min()) + 1e-6)
    # shift by whole pixels so the minimum lands on pixel 0 (CRPIX is 1-based)
    out.wcs.crpix = [float(r.wcs.crpix[0]) - x0, float(r.wcs.crpix[1]) - y0]
    W = int(math.ceil(float(xs.max())) - x0 + 1)
    H = int(math.ceil(float(ys.max())) - y0 + 1)
    return out, (H, W)


class _Placed:
    """One panel resampled onto the output grid, over its own bounding box only."""

    def __init__(self, data, valid, y0, x0, sigma, depth, name):
        self.data, self.valid, self.y0, self.x0 = data, valid, y0, x0
        self.sigma, self.depth, self.name = sigma, depth, name
        self.gain = np.ones(3)
        self.plane = np.zeros((3, 3))           # per channel: a + b u + c v

    @property
    def box(self):
        return self.y0, self.y0 + self.data.shape[0], self.x0, self.x0 + self.data.shape[1]


def _reproject(img: np.ndarray, w_in: "WCS", w_out: "WCS", shape_out) -> Optional[Tuple[np.ndarray, np.ndarray, int, int]]:
    """(data, valid, y0, x0): ``img`` resampled onto ``w_out`` over its footprint's box."""
    h, wd, _ = img.shape
    t = np.linspace(0, 1, 9)
    ex = np.concatenate([t * (wd - 1), np.full(9, wd - 1.0), t[::-1] * (wd - 1), np.zeros(9)])
    ey = np.concatenate([np.zeros(9), t * (h - 1), np.full(9, h - 1.0), t[::-1] * (h - 1)])
    ra, dec = w_in.pixel_to_world_values(ex, ey)
    px, py = w_out.world_to_pixel_values(ra, dec)
    Ho, Wo = shape_out
    y0, y1 = max(0, int(math.floor(py.min())) - 1), min(Ho, int(math.ceil(py.max())) + 2)
    x0, x1 = max(0, int(math.floor(px.min())) - 1), min(Wo, int(math.ceil(px.max())) + 2)
    if y1 <= y0 or x1 <= x0:
        return None
    gy = np.unique(np.concatenate([np.arange(y0, y1, _MAP_STEP), [y1 - 1]])).astype(np.float64)
    gx = np.unique(np.concatenate([np.arange(x0, x1, _MAP_STEP), [x1 - 1]])).astype(np.float64)
    GX, GY = np.meshgrid(gx, gy)
    ra, dec = w_out.pixel_to_world_values(GX, GY)
    sx, sy = w_in.world_to_pixel_values(ra, dec)
    # the exact map on the coarse grid, interpolated to every output pixel
    fy = np.interp(np.arange(y0, y1), gy, np.arange(len(gy)))
    fx = np.interp(np.arange(x0, x1), gx, np.arange(len(gx)))
    FY, FX = np.meshgrid(fy, fx, indexing='ij')
    mx = ndi.map_coordinates(sx, [FY, FX], order=1, mode='nearest')
    my = ndi.map_coordinates(sy, [FY, FX], order=1, mode='nearest')
    finite = np.all(np.isfinite(img), axis=2)
    inside = (mx >= _TRIM_PX) & (mx <= wd - 1 - _TRIM_PX) & (my >= _TRIM_PX) & (my <= h - 1 - _TRIM_PX)
    ok = ndi.map_coordinates(finite.astype(np.float32), [my, mx], order=0, mode='constant', cval=0.0) > 0.5
    valid = inside & ok
    out = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
    for c in range(3):
        ch = np.where(finite, img[..., c], 0.0).astype(np.float64)
        out[..., c] = ndi.map_coordinates(ch, [my, mx], order=3, mode='nearest', prefilter=True)
    out[~valid] = 0.0
    return out, valid, y0, x0


def _noise_sigma(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-channel noise of a placed panel: MAD of lag-4 row differences (as nn_denoise)."""
    out = np.ones(3)
    v = valid[::2, 4:] & valid[::2, :-4]
    for c in range(3):
        x = img[::2, :, c]
        d = (x[:, 4:] - x[:, :-4])[v]
        if d.size > 100:
            out[c] = 1.4826 * np.median(np.abs(d - np.median(d))) / math.sqrt(2.0)
    return np.maximum(out, 1e-9)


def _block_median(a: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(values (gh, gw), ok mask) of ``a``'s block medians over valid pixels ``m``."""
    h, w = a.shape
    gh, gw = h // _BLOCK, w // _BLOCK
    if gh == 0 or gw == 0:
        return np.zeros((0, 0)), np.zeros((0, 0), bool)
    a = a[:gh * _BLOCK, :gw * _BLOCK].reshape(gh, _BLOCK, gw, _BLOCK).transpose(0, 2, 1, 3).reshape(gh, gw, -1)
    m = m[:gh * _BLOCK, :gw * _BLOCK].reshape(gh, _BLOCK, gw, _BLOCK).transpose(0, 2, 1, 3).reshape(gh, gw, -1)
    full = m.all(axis=-1)
    vals = np.where(full, np.median(np.where(m, a, 0.0), axis=-1), np.nan)
    return vals, full


def _overlap(a: _Placed, b: _Placed):
    """(slices into a, slices into b, origin y, x) of the two boxes' intersection."""
    ay0, ay1, ax0, ax1 = a.box
    by0, by1, bx0, bx1 = b.box
    y0, y1, x0, x1 = max(ay0, by0), min(ay1, by1), max(ax0, bx0), min(ax1, bx1)
    if y1 - y0 < 2 * _BLOCK or x1 - x0 < 2 * _BLOCK:
        return None
    sa = (slice(y0 - ay0, y1 - ay0), slice(x0 - ax0, x1 - ax0))
    sb = (slice(y0 - by0, y1 - by0), slice(x0 - bx0, x1 - bx0))
    return sa, sb, y0, x0


def _robust_plane(u, v, d, iters: int = 6):
    """(a, b, c) of d ~ a + b u + c v, Tukey-biweight IRLS."""
    A = np.column_stack([np.ones_like(u), u, v])
    w = np.ones_like(d)
    coef = np.zeros(3)
    for _ in range(iters):
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], d * sw, rcond=None)
        r = d - A @ coef
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
        t = r / (4.685 * s)
        w = np.where(np.abs(t) < 1, (1 - t * t) ** 2, 0.0)
    return coef


def _robust_gain(bi, bj, u, v, iters: int = 6) -> float:
    """r of bj ~ r bi + a + b u + c v (Tukey-biweight IRLS): the gain between two panels
    with their relative sky gradient fitted at the same time."""
    A = np.column_stack([bi, np.ones_like(u), u, v])
    w = np.ones_like(bj)
    coef = np.zeros(4)
    for _ in range(iters):
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], bj * sw, rcond=None)
        r = bj - A @ coef
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
        t = r / (4.685 * s)
        w = np.where(np.abs(t) < 1, (1 - t * t) ** 2, 0.0)
    return float(coef[0])


def _match(panels: List[_Placed], shape_out, ref: int, verbose: bool) -> None:
    """Set each panel's gain and plane by one global least-squares solve over all
    pairwise overlaps (the reference panel fixed at gain 1, plane 0)."""
    Ho, Wo = shape_out
    n = len(panels)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            ov = _overlap(panels[i], panels[j])
            if ov is None:
                continue
            sa, sb, y0, x0 = ov
            m = panels[i].valid[sa] & panels[j].valid[sb]
            if m.sum() < _MIN_OVERLAP_BLOCKS * _BLOCK * _BLOCK:
                continue
            pairs.append((i, j, sa, sb, y0, x0, m))
    if not pairs:
        return
    # 1) gains: per overlap, P_j ~ r P_i + plane, fitted jointly on block medians (a
    #    gain fitted before removing the pair's relative gradient was biased by it,
    #    ~1% on a synthetic 2x2 mosaic: visible seams on a bright galaxy), then the
    #    log ratios solved for all panels at once
    for c in range(3):
        rows, rhs, wts = [], [], []
        for i, j, sa, sb, y0, x0, m in pairs:
            ai, mi = _block_median(panels[i].data[sa][..., c], m)
            aj, _ = _block_median(panels[j].data[sb][..., c], m)
            ok = mi & np.isfinite(ai) & np.isfinite(aj)
            if ok.sum() < _MIN_OVERLAP_BLOCKS:
                continue
            si, sj = ai[ok], aj[ok]
            # the gain needs signal: blocks well above the overlap's sky in both
            bright = ((si > np.median(si) + 10.0 * panels[i].sigma[c] / _BLOCK)
                      & (sj > np.median(sj) + 10.0 * panels[j].sigma[c] / _BLOCK))
            if bright.sum() < 20:
                continue
            by, bx = np.nonzero(ok)
            u = (x0 + (bx + 0.5) * _BLOCK) / Wo - 0.5
            v = (y0 + (by + 0.5) * _BLOCK) / Ho - 0.5
            ratio = _robust_gain(si, sj, u, v)
            if not (0.3 < ratio < 3.0):
                continue
            row = np.zeros(n)
            row[i], row[j] = 1.0, -1.0           # log g_i - log g_j = log(ratio)
            rows.append(row)
            rhs.append(math.log(ratio))
            wts.append(math.sqrt(bright.sum()))
        lg = np.zeros(n)
        if rows:
            A = np.array(rows) * np.array(wts)[:, None]
            b = np.array(rhs) * np.array(wts)
            keep = [k for k in range(n) if k != ref]
            sol, *_ = np.linalg.lstsq(A[:, keep], b, rcond=None)
            lg[keep] = sol
        for k in range(n):
            panels[k].gain[c] = math.exp(lg[k])
    # 2) planes on the gain-corrected data: (g_i P_i - g_j P_j) ~ plane_i - plane_j
    for c in range(3):
        rows, rhs = [], []
        for i, j, sa, sb, y0, x0, m in pairs:
            di = panels[i].gain[c] * panels[i].data[sa][..., c] - panels[j].gain[c] * panels[j].data[sb][..., c]
            vals, ok = _block_median(di, m)
            ok &= np.isfinite(vals)
            if ok.sum() < _MIN_OVERLAP_BLOCKS:
                continue
            by, bx = np.nonzero(ok)
            u = (x0 + (bx + 0.5) * _BLOCK) / Wo - 0.5
            v = (y0 + (by + 0.5) * _BLOCK) / Ho - 0.5
            coef = _robust_plane(u, v, vals[ok])
            wt = math.sqrt(ok.sum())
            for k in range(3):
                row = np.zeros(3 * n)
                row[3 * i + k], row[3 * j + k] = wt, -wt
                rows.append(row)
                rhs.append(wt * coef[k])
        if not rows:
            continue
        keep = [q for q in range(3 * n) if q // 3 != ref]
        sol, *_ = np.linalg.lstsq(np.array(rows)[:, keep], np.array(rhs), rcond=None)
        full = np.zeros(3 * n)
        full[keep] = sol
        for k in range(n):
            panels[k].plane[c] = full[3 * k:3 * k + 3]
    if verbose:
        for p in panels:
            safe_print(f"    {p.name}: gain {np.round(p.gain, 3)}, plane offset "
                       f"{np.round(p.plane[:, 0], 1)}")


def _feather(valid: np.ndarray) -> np.ndarray:
    d = ndi.distance_transform_edt(valid)
    ramp = max(8.0, _FEATHER_FRAC * min(valid.shape))
    t = np.clip(d / ramp, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _blend(panels: List[_Placed], shape_out) -> Tuple[np.ndarray, np.ndarray]:
    """(mosaic (H, W, 3), coverage count): feathered inverse-noise-variance mean of the
    matched panels."""
    Ho, Wo = shape_out
    num = np.zeros((Ho, Wo, 3), np.float64)
    den = np.zeros((Ho, Wo, 3), np.float64)
    cov = np.zeros((Ho, Wo), np.int16)
    yy_cache = {}
    for p in panels:
        y0, y1, x0, x1 = p.box
        key = (y0, y1, x0, x1)
        if key not in yy_cache:
            v, u = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            yy_cache = {key: ((u + 0.5) / Wo - 0.5, (v + 0.5) / Ho - 0.5)}
        u, v = yy_cache[key]
        fw = _feather(p.valid)
        for c in range(3):
            a, b, cc = p.plane[c]
            corr = p.gain[c] * p.data[..., c] - (a + b * u + cc * v)
            w = fw / (p.gain[c] * p.sigma[c]) ** 2
            num[y0:y1, x0:x1, c] += w * corr
            den[y0:y1, x0:x1, c] += w
        cov[y0:y1, x0:x1] += p.valid
    out = np.where(den > 0, num / np.maximum(den, 1e-300), 0.0).astype(np.float32)
    return out, cov


# ── public API ──────────────────────────────────────────────────────────────

def stitch_mosaic_panels(panel_paths: List[str], output_path: str, args) -> bool:
    """Stitch per-panel linear stacks into ``output_path`` (a linear mosaic FITS; the
    caller post-processes it). Returns False -- caller falls back to an ordinary
    combine -- when a panel has no WCS or the panels do not overlap."""
    verbose = getattr(args, 'verbose', False)
    if not (HAS_WCS and HAS_SCIPY):
        safe_print("  WARNING: mosaic stitching needs astropy.wcs and scipy")
        return False
    imgs, hdrs = [], []
    for path in panel_paths:
        data, hdr = _load_panel(path)
        if not _has_wcs(hdr):
            safe_print(f"  WARNING: {os.path.basename(path)} has no usable WCS; a mosaic needs "
                       f"every panel solved (the session solve from info.json, or --plate-solve)")
            return False
        imgs.append(data)
        hdrs.append(hdr)
    depth = [float(h.get('INTGTIME') or h.get('NFRAMES') or 1.0) for h in hdrs]
    ref = int(np.argmax(depth))
    wcss = [_panel_wcs(h) for h in hdrs]
    w_out, shape_out = _output_wcs(wcss, [im.shape[:2] for im in imgs], ref)
    Ho, Wo = shape_out
    if Ho * Wo > 400e6:
        safe_print(f"  WARNING: mosaic grid {Ho}x{Wo} is implausibly large (panels too far "
                   f"apart?); not stitching")
        return False
    safe_print(f"  Mosaic grid: {Ho}x{Wo} px ({Ho * Wo / 1e6:.1f} Mpx), {len(imgs)} panels, "
               f"reference {os.path.basename(panel_paths[ref])}")
    placed: List[_Placed] = []
    for k, (im, w, path) in enumerate(zip(imgs, wcss, panel_paths)):
        got = _reproject(im, w, w_out, shape_out)
        if got is None:
            safe_print(f"  WARNING: {os.path.basename(path)} falls outside the mosaic grid")
            return False
        data, valid, y0, x0 = got
        placed.append(_Placed(data, valid, y0, x0, _noise_sigma(data, valid), depth[k],
                              os.path.basename(path)))
    imgs = None
    # every panel must overlap the rest (one connected sky area)
    n = len(placed)
    seen, todo = {ref}, [ref]
    while todo:
        i = todo.pop()
        for j in range(n):
            if j not in seen and _overlap(placed[i], placed[j]) is not None:
                seen.add(j)
                todo.append(j)
    if len(seen) != n:
        safe_print("  WARNING: the panels do not all overlap; not a mosaic")
        return False
    safe_print("  Matching panels in their overlaps (gain + gradient, one global solve)...")
    _match(placed, shape_out, ref, verbose)
    mosaic, cov = _blend(placed, shape_out)

    hdr = w_out.to_header()
    base = hdrs[ref]
    for key in ('TELESCOP', 'INSTRUME', 'FOCALLEN', 'XPIXSZ', 'YPIXSZ', 'OBSERVER', 'OBJECT',
                'DATE-OBS', 'BAYERPAT', 'CAMERA'):
        if key in base:
            try:
                hdr[key] = base[key]
            except Exception:
                pass
    hdr['RAWSTACK'] = (True, 'Linear stack (before post-processing)')
    hdr['MOSAIC'] = (True, 'Stitched from several panels')
    hdr['NPANELS'] = (len(placed), 'Number of mosaic panels')
    hdr['NFRAMES'] = (int(sum(int(h.get('NFRAMES') or 0) for h in hdrs)), 'Frames in all panels')
    hdr['INTGTIME'] = (float(max(depth)), '[s] deepest panel (panels cover different sky)')
    hdr['COVERAGE'] = (float(np.mean(cov > 0)), 'Fraction of the grid covered')
    hdr['CREATOR'] = 'originstack.py'
    for i, path in enumerate(panel_paths):
        try:
            hdr[f'PANEL{i:03d}'] = os.path.basename(path)[:68]
        except Exception:
            pass
    fits.PrimaryHDU(data=np.transpose(mosaic, (2, 0, 1)), header=hdr).writeto(output_path, overwrite=True)
    safe_print(f"  ✓ Mosaic: {os.path.basename(output_path)} ({Ho}x{Wo}, {len(placed)} panels, "
               f"{np.mean(cov > 0) * 100:.1f}% of the grid covered, {np.mean(cov > 1) * 100:.1f}% overlap)")
    return True
