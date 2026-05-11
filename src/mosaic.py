"""Mosaic stitching for multi-panel astrophotography.

Workflow
--------
Each panel subfolder is stacked independently (phases 1–4 unchanged).
``stitch_mosaic_panels`` then:

  1. Verifies every panel has a plate-solved WCS (PLTSOLVD=True in header).
  2. Computes the optimal output WCS grid covering all panels.
  3. Reprojects each panel onto that grid, per-channel.
  4. Blends overlaps with distance-transform feathering weights.
  5. Matches per-panel sky backgrounds to minimise seams.
  6. Writes the final mosaic FITS + preview.

If ``reproject`` is not installed or any panel lacks a WCS, falls back to
the translation-based combine already in cli.py (returns False).

Dependencies
------------
Required (new):  pip install reproject
Already present: astropy, scipy, numpy
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
from astropy.io import fits

from src.utils import safe_print
from src.io_fits import save_preview_rgb


# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    from reproject import reproject_interp
    from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd
    HAS_REPROJECT = True
except Exception:
    HAS_REPROJECT = False

try:
    from astropy.wcs import WCS
    HAS_WCS = True
except Exception:
    HAS_WCS = False

try:
    from scipy.ndimage import distance_transform_edt
    HAS_EDT = True
except Exception:
    HAS_EDT = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_dependencies() -> str | None:
    """Return a human-readable error string if a required package is absent."""
    if not HAS_REPROJECT:
        return ("mosaic mode requires the 'reproject' package — "
                "install with: pip install reproject")
    if not HAS_WCS:
        return "mosaic mode requires astropy.wcs (install: pip install astropy)"
    return None


def _feather_weight(ch_data: np.ndarray) -> np.ndarray:
    """Distance-from-edge weight map for smooth seam blending.

    Each pixel in the output is proportional to its distance to the nearest
    zero / non-finite pixel in *ch_data*.  Panels therefore contribute most
    strongly at their centres and fade toward their edges, eliminating hard
    seams in overlap zones.
    """
    valid = (np.isfinite(ch_data) & (ch_data != 0)).astype(np.float32)
    if not HAS_EDT or valid.sum() == 0:
        return valid
    w = distance_transform_edt(valid).astype(np.float32)
    max_w = float(w.max())
    return (w / max_w) if max_w > 0 else valid


def _sky_floor(ch_data: np.ndarray, border_frac: float = 0.05) -> float:
    """Sigma-clipped median of border pixels — the per-channel sky pedestal."""
    H, W = ch_data.shape
    by = max(5, int(H * border_frac))
    bx = max(5, int(W * border_frac))
    border = np.concatenate([
        ch_data[:by, :].ravel(),
        ch_data[-by:, :].ravel(),
        ch_data[by:-by, :bx].ravel(),
        ch_data[by:-by, -bx:].ravel(),
    ])
    border = border[np.isfinite(border) & (border > 0)]
    if border.size == 0:
        return 0.0
    med = float(np.median(border))
    sigma = float(np.std(border))
    clipped = border[np.abs(border - med) < 3.0 * sigma]
    return float(np.median(clipped)) if clipped.size > 0 else med


def _normalise_backgrounds(
    panels: List[Tuple[np.ndarray, fits.Header]],
) -> List[Tuple[np.ndarray, fits.Header]]:
    """Subtract a per-panel, per-channel sky pedestal so all panels share a
    common zero-sky baseline before reprojection.

    This is a first-pass correction; ``reproject_and_coadd`` with
    ``match_background=True`` adds a second finer pass in overlap zones.
    """
    result = []
    for data_chw, hdr in panels:
        corrected = data_chw.copy()
        for c in range(3):
            sky = _sky_floor(data_chw[c])
            if sky != 0.0:
                corrected[c] -= sky
        result.append((corrected, hdr))
    return result


def _load_panel(path: str) -> Tuple[np.ndarray, fits.Header]:
    """Load a panel FITS into a (3, H, W) float32 array + header."""
    with fits.open(path, memmap=False) as hd:
        hdr = hd[0].header.copy()
        data = np.array(hd[0].data, dtype=np.float32)
    if data.ndim == 2:
        # Grayscale — promote to 3-channel
        data = np.stack([data, data, data], axis=0)
    elif data.ndim == 3 and data.shape[0] != 3:
        # (H, W, 3) → (3, H, W)
        data = np.transpose(data, (2, 0, 1))
    return data, hdr


def _panel_wcs(hdr: fits.Header) -> "WCS":
    """Extract a 2-axis celestial WCS from a panel header."""
    wcs = WCS(hdr)
    # Prefer the celestial sub-WCS; fall back to full WCS if .celestial is empty
    cel = wcs.celestial
    if cel.naxis == 2:
        return cel
    return wcs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def stitch_mosaic_panels(
    panel_paths: List[str],
    output_path: str,
    args,
) -> bool:
    """Reproject and coadd WCS-solved panel stacks into a single mosaic FITS.

    Parameters
    ----------
    panel_paths:
        Ordered list of per-panel stacked FITS paths (output of each
        subfolder's ``stack_target`` call).
    output_path:
        Destination path for the final mosaic FITS.
    args:
        Parsed CLI namespace — used for ``verbose``, ``stretch``, and GHS
        preview parameters.

    Returns
    -------
    True
        Mosaic written successfully.
    False
        Fell back to caller's translation-based combine (e.g. missing
        dependency or unsolved panel).
    """
    verbose = getattr(args, 'verbose', False)

    # --- Dependency check ------------------------------------------------
    err = _check_dependencies()
    if err:
        safe_print(f"  WARNING: {err}")
        return False

    # --- Load panels, verify WCS -----------------------------------------
    panels: List[Tuple[np.ndarray, fits.Header]] = []
    for path in panel_paths:
        data, hdr = _load_panel(path)
        if not hdr.get('PLTSOLVD', False):
            safe_print(
                f"  WARNING: {os.path.basename(path)} has no plate solution "
                f"(PLTSOLVD not set in header).\n"
                f"  Mosaic requires every panel to be plate-solved.  "
                f"Re-run with --mosaic --plate-solve to add WCS."
            )
            return False
        panels.append((data, hdr))

    safe_print(
        f"  All {len(panels)} panels plate-solved — "
        f"stitching via WCS reprojection"
    )

    # --- Per-panel sky normalisation -------------------------------------
    safe_print("  Normalising panel sky backgrounds...")
    panels = _normalise_backgrounds(panels)

    # --- Compute output WCS grid -----------------------------------------
    safe_print("  Computing optimal output WCS grid...")
    wcs_shapes = []
    for data_chw, hdr in panels:
        _, H, W = data_chw.shape
        wcs_shapes.append((_panel_wcs(hdr), (H, W)))

    wcs_out, shape_out = find_optimal_celestial_wcs(wcs_shapes, auto_rotate=False)
    Hm, Wm = shape_out
    safe_print(f"  Output mosaic: {Hm}×{Wm} px  ({Hm * Wm / 1e6:.1f} Mpx)")

    # --- Reproject per channel ------------------------------------------
    mosaic_chw = np.zeros((3, Hm, Wm), dtype=np.float32)
    footprint: np.ndarray | None = None
    channel_names = ['R', 'G', 'B']

    for c, ch in enumerate(channel_names):
        if verbose:
            safe_print(f"  Reprojecting channel {ch}...")

        inputs = []
        weights = []
        for data_chw, hdr in panels:
            ch_data = data_chw[c]                     # (H, W)
            inputs.append((ch_data, _panel_wcs(hdr)))
            weights.append(_feather_weight(ch_data))

        # match_background: additive per-panel offset fitting in overlap zones
        # (scipy already required; gracefully degrade for older reproject)
        try:
            ch_mosaic, ch_fp = reproject_and_coadd(
                inputs,
                wcs_out,
                shape_out=shape_out,
                input_weights=weights,
                reproject_function=reproject_interp,
                combine_function='mean',
                match_background=True,
            )
        except TypeError:
            # reproject < 0.9 — no match_background parameter
            ch_mosaic, ch_fp = reproject_and_coadd(
                inputs,
                wcs_out,
                shape_out=shape_out,
                input_weights=weights,
                reproject_function=reproject_interp,
                combine_function='mean',
            )

        mosaic_chw[c] = np.nan_to_num(ch_mosaic, nan=0.0).astype(np.float32)
        if footprint is None:
            footprint = ch_fp

    # --- Build output header --------------------------------------------
    out_hdr = wcs_out.to_header()
    # Carry over instrument/observer keywords from first panel
    base_hdr = panels[0][1]
    for key in ('TELESCOP', 'INSTRUME', 'FOCALLEN', 'XPIXSZ', 'YPIXSZ',
                'OBSERVER', 'OBJECT'):
        if key in base_hdr:
            try:
                out_hdr[key] = base_hdr[key]
            except Exception:
                pass
    out_hdr['NPANELS'] = (len(panels), 'Number of mosaic panels')
    out_hdr['CREATOR'] = 'astro_stack.py'
    out_hdr['COMBINED'] = True

    # Panel source paths (truncated to 68 chars each — FITS header limit)
    for i, path in enumerate(panel_paths):
        key = f'PANEL{i:03d}'
        try:
            out_hdr[key] = os.path.basename(path)[:68]
        except Exception:
            pass

    # --- Write FITS + preview -------------------------------------------
    primary = fits.PrimaryHDU(data=mosaic_chw, header=out_hdr)
    primary.writeto(output_path, overwrite=True)

    mosaic_hwc = np.transpose(mosaic_chw, (1, 2, 0))
    preview_path = os.path.splitext(output_path)[0] + '.jpg'
    save_preview_rgb(mosaic_hwc, preview_path,
                     stretch=getattr(args, 'stretch', 'ghs'),
                     ghs_b=getattr(args, 'ghs_b', 8.0),
                     ghs_sp=getattr(args, 'ghs_sp', 0.15),
                     ghs_hp=getattr(args, 'ghs_hp', 0.95))

    safe_print(
        f"  ✓ Mosaic: {os.path.basename(output_path)}  "
        f"({Hm}×{Wm}, {Hm * Wm / 1e6:.1f} Mpx, {len(panels)} panels)"
    )
    safe_print(f"  ✓ Preview: {os.path.basename(preview_path)}")

    if footprint is not None:
        coverage = float(np.mean(footprint > 0)) * 100.0
        overlap_px = int(np.sum(footprint > 1.0))
        safe_print(
            f"  Coverage: {coverage:.1f}% of output grid populated  "
            f"| Overlap zone: {overlap_px:,} px"
        )

    return True
