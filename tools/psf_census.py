"""Does the PSF vary across the field in a way that changes from frame to frame?

Tile-aware proper coaddition (a PSF and weight per frame *and tile*) only pays when some
frames are sharper in one part of the field and others in another -- "lucky patches" -- beyond
what the frame's overall seeing and the field's fixed shape already explain. This measures it:
the FWHM of the stars in each cell of a grid (default 2x2) in each of N calibrated frames,
then removes the per-frame mean (the frame's seeing) and the per-cell mean (the field's static
shape) and looks at what is left.

  excess = sqrt(max(residual_scatter^2 - measurement_noise^2, 0)) / mean FWHM

The measurement noise comes from splitting each cell's stars into two halves (by flux rank) and
comparing their FWHM. Gates from the research report (tile-aware coadd, PSF field):

  * excess below 3%     -> no lucky-patch signal: do not build tile-aware proper coadd;
  * static max/min cell FWHM below 1.05 -> a single PSF per frame describes the field.

Cells are in sensor coordinates (the frame as the camera wrote it), which is where the optics
are fixed; the stack's rotation does not enter.

Usage:
  python tools/psf_census.py SESSION_DIR [--frames 40] [--grid 2] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MIN_STARS_PER_CELL = 12


def census_statistics(F: np.ndarray, noise_diff: np.ndarray) -> Dict[str, float]:
    """Statistics of a (frames, cells) FWHM matrix.

    ``noise_diff`` holds, for the same cells and frames, the difference of the FWHM of two
    disjoint halves of the cell's stars (so ``std(noise_diff) / 2`` is the error of the full-cell
    estimate). NaN cells are dropped by removing every row or column with a gap larger than
    allowed (rows/columns with any NaN)."""
    keep_r = ~np.isnan(F).any(axis=1)
    F = F[keep_r]
    nd = noise_diff[keep_r]
    if F.shape[0] < 6 or F.shape[1] < 2:
        return {'n_frames': int(F.shape[0]), 'excess_rel': float('nan')}
    grand = float(F.mean())
    resid = F - F.mean(axis=1, keepdims=True) - F.mean(axis=0, keepdims=True) + grand
    n, m = F.shape
    dof = (n - 1) * (m - 1) / (n * m)                  # the two-way fit absorbed this share of the variance
    resid_var = float(np.mean(resid ** 2)) / dof
    noise_var = float(np.nanvar(nd)) / 4.0
    cell_means = F.mean(axis=0)
    return {
        'n_frames': int(n), 'n_cells': int(m), 'mean_fwhm': grand,
        'frame_scatter_rel': float(F.mean(axis=1).std() / grand),
        'static_max_over_min': float(cell_means.max() / cell_means.min()),
        'residual_rel': float(np.sqrt(resid_var) / grand),
        'noise_rel': float(np.sqrt(noise_var) / grand),
        'excess_rel': float(np.sqrt(max(resid_var - noise_var, 0.0)) / grand),
    }


def verdict(stats: Dict[str, float]) -> List[str]:
    out = []
    e = stats.get('excess_rel', float('nan'))
    if not np.isfinite(e):
        return ["not enough usable frames or stars for a verdict"]
    out.append(("lucky-patch signal present (%.1f%% >= 3%%): tile-aware proper coadd is worth building"
                if e >= 0.03 else
                "no lucky-patch signal (%.1f%% < 3%%): do not build tile-aware proper coadd") % (100 * e))
    r = stats['static_max_over_min']
    out.append(("static field variation %.3f >= 1.05: a position-dependent PSF would help"
                if r >= 1.05 else
                "static field variation %.3f < 1.05: one PSF per frame describes the field") % r)
    return out


def cell_fwhms(lum: np.ndarray, sources, grid: int) -> tuple:
    """(FWHM per cell, difference of two half-star FWHMs per cell) for one frame; NaN when a cell has few stars."""
    from src.quality import measure_fwhm
    H, W = lum.shape
    fw = np.full(grid * grid, np.nan)
    nd = np.full(grid * grid, np.nan)
    ys = np.asarray(sources['ycentroid'])
    xs = np.asarray(sources['xcentroid'])
    order = np.argsort(-np.asarray(sources['flux']))
    for gy in range(grid):
        for gx in range(grid):
            sel = [i for i in order if gy * H / grid <= ys[i] < (gy + 1) * H / grid
                   and gx * W / grid <= xs[i] < (gx + 1) * W / grid]
            if len(sel) < MIN_STARS_PER_CELL:
                continue
            sub = sources[sel]
            a, b = sub[0::2], sub[1::2]
            fa, fb = measure_fwhm(lum, a), measure_fwhm(lum, b)
            f_all = measure_fwhm(lum, sub)
            if f_all > 0 and fa > 0 and fb > 0:
                fw[gy * grid + gx] = f_all
                nd[gy * grid + gx] = fa - fb
    return fw, nd


def run(session_dir: str, n_frames: int, grid: int) -> Dict[str, float]:
    from src.cli import _build_masters
    from src.debayer import luminance
    from src.frame_discovery import discover_frames
    from src.frame_processor import _process_single_frame
    from src.models import ProcessingStats
    from src.quality import detect_stars_auto

    class _Args:                      # the few settings _build_masters reads
        master_method = 'median'
        auto = False
        dark_temp_model = False
        flat_from_lights = False
        vignette_map = None
        photometry = False
        photometry_timeseries = False
        session_dark_scale = False
        _explicit_cli_dests = set()

    frames = discover_frames(session_dir)
    lights = frames['light']
    masters = _build_masters(frames, ProcessingStats(), _Args())
    idx = np.unique(np.linspace(0, len(lights) - 1, min(n_frames, len(lights))).astype(int))
    rows, diffs = [], []
    for i in idx:
        res = _process_single_frame(lights[i].path, lights[i].header, masters, 'rcd', 'none',
                                    quality_level='quick', skip_quality=True)
        if res.get('error'):
            continue
        lum = luminance(res['rgb']).astype(np.float32)
        noise = 1.4826 * float(np.median(np.abs(lum - np.median(lum))))
        src = detect_stars_auto(lum, noise, background=float(np.median(lum)))
        if src is None or len(src) < grid * grid * MIN_STARS_PER_CELL:
            continue
        fw, nd = cell_fwhms(lum, src, grid)
        rows.append(fw)
        diffs.append(nd)
    if not rows:
        return {'n_frames': 0, 'excess_rel': float('nan')}
    return census_statistics(np.array(rows), np.array(diffs))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument('session')
    ap.add_argument('--frames', type=int, default=40)
    ap.add_argument('--grid', type=int, default=2)
    ap.add_argument('--json')
    a = ap.parse_args(argv)
    st = run(a.session, a.frames, a.grid)
    print(f"PSF census of {os.path.basename(os.path.normpath(a.session))}: {st.get('n_frames', 0)} frames, "
          f"{a.grid}x{a.grid} cells")
    for k in ('mean_fwhm', 'frame_scatter_rel', 'static_max_over_min', 'residual_rel', 'noise_rel',
              'excess_rel'):
        if k in st:
            print(f"  {k:22s} {st[k]:.4f}")
    for line in verdict(st):
        print("  ->", line)
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as fh:
            json.dump(st, fh, indent=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
