"""Dither-coverage uniformity diagnostic (--dither-report).

Drizzle output quality is bounded by how uniformly the sub-pixel dither
offsets sample the output grid -- a session where every frame's fractional
shift clusters near the same sub-pixel phase undersamples parts of the
output footprint even if every individual frame looks fine on its own.
Nothing else in this pipeline measures that directly. Diagnostic only (like
--aberration-report): no effect on the stack itself, fails soft on any
error.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from src.utils import safe_print

try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None
    ImageDraw = None


def compute_dither_coverage(shifts: List[Tuple[float, float]], grid: int = 16,
                            scale: float = 1.0) -> dict:
    """Coverage-count grid + uniformity stats for a set of per-frame
    (dy, dx) registration shifts.

    Bins each frame's sub-pixel phase (fractional part of ``dy * scale``,
    ``dx * scale`` -- ``scale`` matches whatever ``--drizzle-scale`` is
    active, since that's the output-pixel grid dither actually needs to
    fill) into a ``grid x grid`` histogram over one output-pixel cell. A
    uniformly-dithered session fills every bin close to evenly; a session
    with too little or correlated dithering (near-integer shifts, a
    repeating pattern) leaves some bins empty while others are overfilled.
    """
    if not shifts:
        return {'grid_counts': None, 'uniformity': 0.0, 'empty_frac': 1.0, 'n_frames': 0}

    phases_y = np.array([(dy * scale) % 1.0 for dy, dx in shifts])
    phases_x = np.array([(dx * scale) % 1.0 for dy, dx in shifts])
    bins_y = np.clip((phases_y * grid).astype(int), 0, grid - 1)
    bins_x = np.clip((phases_x * grid).astype(int), 0, grid - 1)

    hist = np.zeros((grid, grid), dtype=np.int64)
    np.add.at(hist, (bins_y, bins_x), 1)

    mean = float(hist.mean())
    std = float(hist.std())
    uniformity = max(0.0, 1.0 - std / mean) if mean > 0 else 0.0
    empty_frac = float(np.mean(hist == 0))

    return {
        'grid_counts': hist,
        'uniformity': float(min(uniformity, 1.0)),  # 1.0 = perfectly uniform
        'empty_frac': empty_frac,
        'n_frames': len(shifts),
    }


def _save_coverage_png(stats: dict, path: str) -> bool:
    if Image is None or stats.get('grid_counts') is None:
        return False
    hist = stats['grid_counts']
    g = hist.shape[0]
    cell = 24
    size = g * cell
    vmax = max(1, int(hist.max()))
    img = Image.new('RGB', (size, size), (20, 20, 30))
    draw = ImageDraw.Draw(img)
    for gy in range(g):
        for gx in range(g):
            v = hist[gy, gx] / vmax
            color = (int(30 + 200 * v), int(30 + 180 * v), int(60 + 120 * (1 - v)))
            draw.rectangle([gx * cell, gy * cell, (gx + 1) * cell - 1, (gy + 1) * cell - 1],
                          fill=color)
    try:
        img.save(path)
        return True
    except Exception:
        return False


def run_dither_report(shifts: List[Tuple[float, float]], output_path: str,
                      drizzle_scale: float = 1.0, grid: int = 16) -> Optional[dict]:
    """Entry point wired to ``--dither-report``: computes coverage stats,
    prints a summary, and writes ``<output>_dither.png`` when PIL is
    available. Fails soft (prints a message, returns ``None``) rather than
    raising, matching ``--aberration-report``'s contract.
    """
    try:
        stats = compute_dither_coverage(shifts, grid=grid, scale=drizzle_scale)
        if stats['n_frames'] == 0:
            safe_print("  Dither report: no shifts available -- skipping")
            return None
        safe_print(f"\n  Dither coverage: {stats['n_frames']} frames, "
                   f"uniformity={stats['uniformity']:.2f} (1.0=perfectly uniform), "
                   f"{stats['empty_frac'] * 100:.0f}% of {grid}x{grid} sub-pixel bins "
                   f"never sampled")
        if stats['uniformity'] < 0.5:
            safe_print("  WARNING: dither coverage is quite non-uniform -- drizzle "
                       "output quality may vary noticeably across the frame")
        png_path = os.path.splitext(output_path)[0] + '_dither.png'
        if _save_coverage_png(stats, png_path):
            safe_print(f"  Dither coverage map: {os.path.basename(png_path)}")
        return stats
    except Exception as e:
        safe_print(f"  Dither report failed ({e}) -- skipping")
        return None
