"""Gaia DR3 astrometric distortion correction (--gaia-distortion-correction).

Different problem than --elastic-registration: that corrects frame-to-frame
*relative* drift (differential atmospheric refraction, field rotation)
against whichever frame was picked as reference. This corrects the
telescope's own *absolute* optical distortion (coma, field curvature) --
the same distortion pattern in every frame once they're all warped into
reference-frame pixel space, since it's a property of the optics/reference
geometry, not of any individual frame. Anchored to real sky positions
(Gaia DR3), not just internal frame-to-frame self-consistency.

Requires a WCS already present in the reference frame's header -- from the
capture software's own pre-imaging plate-solve (common: N.I.N.A., SGP,
TheSkyX, MaxIm all support this), or reused from a prior --merge chain.
This module does NOT trigger a new plate-solve itself (that only happens
late, on the final stack, in src/plate_solve.py) -- v1 limitation, fails
soft with a clear message when no WCS is present rather than silently
doing nothing.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.models import Config
from src.utils import safe_print

# Minimum cross-matched Gaia<->detected star pairs before attempting a fit --
# same order of magnitude as Config.LOCAL_WARP_MIN_STARS (the elastic-
# registration floor), since this feeds the exact same DBE-style local
# regression fitter.
GAIA_MIN_MATCHES = 12
# Max angular radius to query around the field center, and the cone-search
# row cap -- generous but bounded so a huge/misidentified field doesn't
# trigger a runaway query.
GAIA_MAX_ROWS = 500
# Cross-match tolerance in pixels between a Gaia-predicted position (from the
# existing WCS, not yet distortion-corrected) and a detected star centroid.
# Deliberately loose -- the WCS itself is approximate (that's the whole
# reason a distortion correction is being fit), so a tight tolerance would
# reject exactly the matches most informative about the distortion.
GAIA_MATCH_RADIUS_PX = 15.0


def _build_wcs(header: dict):
    """Return a 2D celestial astropy WCS from a FITS header dict, or None."""
    if not header or 'CTYPE1' not in header or 'CRVAL1' not in header:
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
    except Exception:
        return None


def _field_center_and_radius(wcs, H: int, W: int):
    """(ra_center_deg, dec_center_deg, radius_deg) covering the full frame --
    same approach as src/annotation.py's helper of the same purpose."""
    center = wcs.all_pix2world([[W / 2.0, H / 2.0]], 0)[0]
    ra_c, dec_c = float(center[0]), float(center[1])
    corners = wcs.all_pix2world([[0, 0], [W, 0], [0, H], [W, H]], 0)
    dists = [
        np.hypot((c[0] - ra_c) * np.cos(np.radians(dec_c)), c[1] - dec_c)
        for c in corners
    ]
    return ra_c, dec_c, float(max(dists))


def fit_gaia_distortion_field(header: dict, ref_stars, H: int, W: int) -> Optional[np.ndarray]:
    """Fit a session-wide (Gc, Gc, 2) optical distortion field from Gaia DR3
    reference-star positions vs. the reference frame's detected centroids.

    Returns ``None`` (fail soft) if: no WCS in ``header``, the Gaia query
    fails or returns too few rows, or fewer than ``GAIA_MIN_MATCHES``
    detected stars cross-match a Gaia prediction within
    ``GAIA_MATCH_RADIUS_PX``. Caller should proceed without this correction
    in any of those cases, same convention as
    ``registration.fit_displacement_field``'s below-star-floor ``None``.
    """
    wcs = _build_wcs(header)
    if wcs is None:
        safe_print("  --gaia-distortion-correction: no WCS in reference frame header "
                   "(needs a capture-software plate-solve or a --merge'd prior session) "
                   "-- skipping")
        return None

    if ref_stars is None or len(ref_stars) < GAIA_MIN_MATCHES:
        safe_print(f"  --gaia-distortion-correction: too few detected reference stars "
                   f"({0 if ref_stars is None else len(ref_stars)} < {GAIA_MIN_MATCHES}) "
                   f"-- skipping")
        return None

    try:
        ra_c, dec_c, radius_deg = _field_center_and_radius(wcs, H, W)
        from src.net_query import gaia_cone_search
        table = gaia_cone_search(ra_c, dec_c, radius_deg,
                                 columns=['ra', 'dec', 'phot_g_mean_mag'],
                                 max_rows=GAIA_MAX_ROWS,
                                 require_not_null=['ra', 'dec'])
    except Exception as exc:
        safe_print(f"  --gaia-distortion-correction: Gaia query failed ({exc}) -- skipping")
        return None

    if table is None or len(table) < GAIA_MIN_MATCHES:
        safe_print(f"  --gaia-distortion-correction: Gaia query returned "
                   f"{0 if table is None else len(table)} rows (< {GAIA_MIN_MATCHES}) "
                   f"-- skipping")
        return None

    try:
        gaia_radec = np.column_stack([np.asarray(table['ra'], dtype=np.float64),
                                      np.asarray(table['dec'], dtype=np.float64)])
        predicted_px = wcs.all_world2pix(gaia_radec, 0)  # (M, 2) x,y -- undistorted prediction
    except Exception as exc:
        safe_print(f"  --gaia-distortion-correction: WCS projection failed ({exc}) -- skipping")
        return None

    # Bounds-check: a star predicted well outside the frame can't have a
    # real detected counterpart -- drop before the nearest-neighbour match
    # so it can't spuriously pair with an edge star.
    in_bounds = ((predicted_px[:, 0] >= -GAIA_MATCH_RADIUS_PX)
                & (predicted_px[:, 0] < W + GAIA_MATCH_RADIUS_PX)
                & (predicted_px[:, 1] >= -GAIA_MATCH_RADIUS_PX)
                & (predicted_px[:, 1] < H + GAIA_MATCH_RADIUS_PX))
    predicted_px = predicted_px[in_bounds]
    if len(predicted_px) < GAIA_MIN_MATCHES:
        safe_print(f"  --gaia-distortion-correction: only {len(predicted_px)} Gaia stars "
                   f"land in-frame (< {GAIA_MIN_MATCHES}) -- skipping")
        return None

    try:
        det_xy = np.column_stack([
            np.asarray(ref_stars['xcentroid'], dtype=np.float64),
            np.asarray(ref_stars['ycentroid'], dtype=np.float64),
        ])
        from scipy.spatial import cKDTree
        tree = cKDTree(det_xy)
        dist, idx = tree.query(predicted_px, k=1, distance_upper_bound=GAIA_MATCH_RADIUS_PX)
        valid = np.isfinite(dist) & (idx < len(det_xy))
        n_matched = int(valid.sum())
        if n_matched < GAIA_MIN_MATCHES:
            safe_print(f"  --gaia-distortion-correction: only {n_matched} Gaia<->detected "
                       f"matches within {GAIA_MATCH_RADIUS_PX:.0f}px (< {GAIA_MIN_MATCHES}) "
                       f"-- skipping")
            return None

        matched_predicted = predicted_px[valid]
        matched_detected = det_xy[idx[valid]]
    except Exception as exc:
        safe_print(f"  --gaia-distortion-correction: cross-match failed ({exc}) -- skipping")
        return None

    # Reuses fit_displacement_field's exact DBE-style local-regression fit --
    # same (x,y)-pair-in/coarse-field-out contract, just fed
    # (Gaia-predicted, detected) instead of (reference-frame, this-frame):
    # dy = predicted[:,1] - detected[:,1], i.e. the field points FROM the
    # distorted detected position TOWARD the true undistorted one, matching
    # apply_transform's local_field convention exactly (same "ref minus
    # frame" semantics fit_displacement_field's own docstring describes).
    from src.registration import fit_displacement_field
    field = fit_displacement_field(matched_predicted, matched_detected, H, W)
    if field is None:
        safe_print(f"  --gaia-distortion-correction: {n_matched} matches, below "
                   f"Config.LOCAL_WARP_MIN_STARS ({Config.LOCAL_WARP_MIN_STARS}) needed "
                   f"for the local-regression fit -- skipping")
        return None

    safe_print(f"  Gaia distortion correction: {n_matched} matched stars, "
              f"max displacement {float(np.hypot(field[..., 0], field[..., 1]).max()):.2f}px")
    return field
