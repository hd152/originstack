"""Object annotation (--annotate): circle and label bright stars and named
deep-sky objects (galaxies, nebulae, clusters) on a copy of the stretched
preview, using the FITS header's WCS (from --plate-solve or a session
info.json solve) and live SIMBAD cone-search queries (src/net_query.py).

Two separate queries, not one combined query -- SIMBAD's per-object V
magnitude (the useful "how bright is this star" signal) lives in a
separate `flux` table that most extended objects (galaxies, nebulae) don't
have a clean entry in; an inner join would silently drop them:
  - Bright stars: basic+flux(V) join, magnitude-capped, brightest first.
  - Named DSOs: basic table only, restricted to a curated set of object
    types AND a well-known catalog prefix (M/NGC/IC) in the name -- SIMBAD
    has thousands of obscure entries (molecular-cloud fragments, individual
    HII sub-condensations, ...) in a typical field; this keeps the output
    to the handful a human would actually call "the nebula"/"the galaxy".

Fails soft everywhere: no WCS -> skip with a message, network/query
failure -> skip with a message, never raises out of run_annotation.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils import safe_print

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None

# SIMBAD otype codes worth labeling as a "deep-sky object" -- galaxies,
# planetary/reflection nebulae, HII regions, supernova remnants, clusters,
# and generic interstellar matter. Deliberately excludes finer-grained
# codes (individual young stellar objects, molecular-cloud fragments, ...)
# that would flood a typical rich field with entries no human labels by
# eye. See src/annotation.py module docstring.
_DSO_OTYPES = ('G', 'PN', 'HII', 'SNR', 'Cl*', 'GlC', 'OpC', 'RNe', 'ISM')
_DSO_NAME_PREFIXES = ('M %', 'NGC %', 'IC %')


def _build_wcs(header) -> Optional[object]:
    """Return a 2D celestial astropy WCS from a FITS header, or None if the
    header has no WCS solution (plate-solve/session-info didn't run)."""
    if 'CTYPE1' not in header or 'CRVAL1' not in header:
        return None
    try:
        from astropy.wcs import WCS
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # FITSFixedWarning on minor header quirks
            wcs = WCS(header, naxis=2)
        if not wcs.has_celestial:
            return None
        return wcs
    except Exception as exc:
        safe_print(f"  Annotate: could not build WCS from header ({exc})")
        return None


def _field_center_and_radius(wcs, shape: Tuple[int, int]) -> Tuple[float, float, float]:
    """(ra_center_deg, dec_center_deg, radius_deg) covering the full frame,
    from the WCS's own pixel scale -- no assumption about instrument/FOV."""
    h, w = shape
    center = wcs.all_pix2world([[w / 2.0, h / 2.0]], 0)[0]
    ra_c, dec_c = float(center[0]), float(center[1])
    corners = wcs.all_pix2world([[0, 0], [w, 0], [0, h], [w, h]], 0)
    dists = [
        np.hypot((c[0] - ra_c) * np.cos(np.radians(dec_c)), c[1] - dec_c)
        for c in corners
    ]
    return ra_c, dec_c, float(max(dists))


def query_annotation_objects(
    ra_deg: float, dec_deg: float, radius_deg: float,
    star_mag_limit: float = 9.0, max_stars: int = 40, max_dso: int = 20,
) -> List[Dict]:
    """Live SIMBAD queries for bright stars + named DSOs in the given cone.
    Returns a list of dicts: {ra, dec, name, otype, kind}. Empty list (not
    an exception) on any query failure -- annotation degrades to "nothing
    found" rather than aborting the whole run.
    """
    from src.net_query import tap_query, _SIMBAD_TAP

    objects: List[Dict] = []

    star_adql = (
        f"SELECT TOP {int(max_stars)} basic.ra, basic.dec, basic.main_id, "  # nosec B608
        "basic.otype, flux.flux AS vmag FROM basic "
        "JOIN flux ON flux.oidref = basic.oid "
        f"WHERE flux.filter='V' AND flux.flux < {float(star_mag_limit)} "
        f"AND 1=CONTAINS(POINT('ICRS',basic.ra,basic.dec),"
        f"CIRCLE('ICRS',{float(ra_deg)},{float(dec_deg)},{float(radius_deg)})) "
        "ORDER BY vmag ASC"
    )
    try:
        t = tap_query(_SIMBAD_TAP, star_adql)
        if t is not None:
            for row in t:
                objects.append({
                    'ra': float(row['ra']), 'dec': float(row['dec']),
                    'name': str(row['main_id']).strip(),
                    'otype': str(row['otype']).strip(), 'kind': 'star',
                })
    except Exception as exc:
        safe_print(f"  Annotate: bright-star query failed ({exc})")

    otype_list = ",".join(f"'{o}'" for o in _DSO_OTYPES)
    name_clause = " OR ".join(f"basic.main_id LIKE '{p}'" for p in _DSO_NAME_PREFIXES)
    dso_adql = (
        f"SELECT TOP {int(max_dso)} basic.ra, basic.dec, basic.main_id, "  # nosec B608
        f"basic.otype FROM basic WHERE basic.otype IN ({otype_list}) "
        f"AND ({name_clause}) "
        f"AND 1=CONTAINS(POINT('ICRS',basic.ra,basic.dec),"
        f"CIRCLE('ICRS',{float(ra_deg)},{float(dec_deg)},{float(radius_deg)}))"
    )
    try:
        t = tap_query(_SIMBAD_TAP, dso_adql)
        if t is not None:
            for row in t:
                objects.append({
                    'ra': float(row['ra']), 'dec': float(row['dec']),
                    'name': str(row['main_id']).strip(),
                    'otype': str(row['otype']).strip(), 'kind': 'dso',
                })
    except Exception as exc:
        safe_print(f"  Annotate: named-object query failed ({exc})")

    return objects


def draw_annotations(preview_uint8: np.ndarray, wcs, objects: List[Dict]) -> Optional[np.ndarray]:
    """Draw a circle + label for each object at its projected pixel position
    on a copy of the (H,W,3) uint8 preview. Objects that project outside the
    frame (WCS solution edge, or a wide cone-search radius vs. a rectangular
    frame) are silently skipped. Returns None if Pillow is unavailable."""
    if Image is None:
        safe_print("  Annotate: Pillow not installed -- skipping")
        return None
    if not objects:
        return preview_uint8

    h, w = preview_uint8.shape[:2]
    img = Image.fromarray(preview_uint8)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    radii = {'star': 12, 'dso': 22}
    colors = {'star': (255, 210, 80), 'dso': (90, 200, 255)}

    world = np.array([[o['ra'], o['dec']] for o in objects])
    pix = wcs.all_world2pix(world, 0)

    for obj, (x, y) in zip(objects, pix):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        x, y = float(x), float(y)
        if not (0 <= x < w and 0 <= y < h):
            continue
        r = radii[obj['kind']]
        color = colors[obj['kind']]
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=2)
        draw.text((x + r + 3, y - 6), obj['name'], fill=color, font=font)

    return np.array(img)


def run_annotation(stacked: np.ndarray, header, output_path: str, args) -> bool:
    """Entry point for --annotate. Builds WCS from the header, queries
    SIMBAD for the frame's field of view, and writes
    ``<output>_annotated.jpg``. Returns True on success; failures (no WCS,
    network error, Pillow absent) print a message and return False -- never
    raises, so a bad plate solve or an offline machine doesn't fail the
    whole run over an optional extra.
    """
    wcs = _build_wcs(header)
    if wcs is None:
        safe_print("  Annotate: no WCS in header (needs --plate-solve or a "
                   "session solve) -- skipping")
        return False

    h, w = stacked.shape[:2]
    ra_c, dec_c, radius = _field_center_and_radius(wcs, (h, w))
    safe_print(f"  Annotate: querying SIMBAD ({radius:.2f} deg radius around "
               f"RA={ra_c:.3f} Dec={dec_c:.3f}) ...")

    star_mag_limit = float(getattr(args, 'annotate_mag_limit', 9.0))
    objects = query_annotation_objects(ra_c, dec_c, radius, star_mag_limit=star_mag_limit)
    if not objects:
        safe_print("  Annotate: no catalogued objects found in field")
        return False

    from src.io_fits import render_preview_uint8
    preview = render_preview_uint8(
        stacked, stretch=getattr(args, 'stretch', 'ghs'),
        ghs_b=float(getattr(args, 'ghs_b', 8.0)),
        ghs_sp=float(getattr(args, 'ghs_sp', 0.15)),
        ghs_hp=float(getattr(args, 'ghs_hp', 0.95)),
        black_sigma=float(getattr(args, 'preview_black_sigma', 0.0) or 0.0))
    if preview is None:
        safe_print("  Annotate: preview stretch unavailable -- skipping")
        return False

    annotated = draw_annotations(preview, wcs, objects)
    if annotated is None:
        return False

    out_path = f"{output_path.rsplit('.', 1)[0]}_annotated.jpg"
    n_stars = sum(1 for o in objects if o['kind'] == 'star')
    n_dso = sum(1 for o in objects if o['kind'] == 'dso')
    try:
        Image.fromarray(annotated).save(out_path, format='JPEG', quality=92)
    except Exception as exc:
        safe_print(f"  Annotate: failed to save {out_path} ({exc})")
        return False
    safe_print(f"  Annotate: {n_stars} stars, {n_dso} named objects -> {out_path}")
    return True
