"""A rough first look at the stack, a few seconds after registration (desktop app only).

The first stack preview used to appear when Phase 3 finished its combine; on a long session that is
minutes of an empty preview. Registration already knows every frame's transform, so a handful of
the sharpest frames are warped into the common crop, binned to preview size and averaged: faint
and noisy, but the target, the framing and the colour balance are visible while the real stack is
still being made. It is published as its own named preview and replaced by the real one.

Shows nothing and costs nothing on a CLI run (``UIEvents.active`` is False there), and it never
touches the stack: it reads the aligned-input frames and writes only a JPEG to the UI.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from src.models import RunCancelled

log = logging.getLogger("originstack")

QUICK_LOOK_FRAMES = 8          # sharpest frames averaged
QUICK_LOOK_MAX_DIM = 1024      # the preview is encoded at this size anyway


def pick_frames(final: Sequence[Any], n: int = QUICK_LOOK_FRAMES) -> List[int]:
    """Positions in ``final`` of the ``n`` sharpest frames (smallest FWHM; frames without one last),
    in session order so the mean does not depend on how ties fall."""
    def key(j: int) -> float:
        fw = (getattr(final[j], 'metrics', None) or {}).get('fwhm')
        return float(fw) if fw and fw > 0 else float('inf')
    return sorted(sorted(range(len(final)), key=key)[:max(1, n)])


def quick_look_image(mem_rgb, final_indices: Sequence[int], positions: Sequence[int],
                     shifts: Sequence[Tuple[float, float]], transforms: Sequence[Optional[Any]],
                     crop: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """Mean of the chosen frames warped into ``crop`` and binned to ~``QUICK_LOOK_MAX_DIM`` px
    (float32 HWC, linear), or None when nothing could be warped."""
    from src.registration import apply_transform
    top, bottom, left, right = crop
    h, w = bottom - top, right - left
    if h < 8 or w < 8:
        return None
    b = max(1, -(-max(h, w) // QUICK_LOOK_MAX_DIM))     # ceil: never larger than the preview
    hb, wb = h // b, w // b
    total = None
    used = 0
    for j in positions:
        frame = np.ascontiguousarray(mem_rgb[final_indices[j]], dtype=np.float32)
        warped = apply_transform(frame, shift=shifts[j], transform=transforms[j], crop=crop)
        small = warped[:hb * b, :wb * b].reshape(hb, b, wb, b, -1).mean(axis=(1, 3), dtype=np.float32)
        total = small if total is None else total + small
        used += 1
    if not used:
        return None
    return (total / used).astype(np.float32)


def publish_quick_look(mem_rgb, final, final_indices, shifts, transforms, crop, args) -> bool:
    """Publish the quick look to the desktop UI; True when one was shown. Never raises (a failed
    preview must not fail a run), except for a cancel."""
    from src.ui_events import get_ui_events
    ui = get_ui_events()
    if not ui.active:
        return False
    try:
        positions = pick_frames(final)
        img = quick_look_image(mem_rgb, final_indices, positions, shifts, transforms, crop)
        if img is None:
            return False
        ui.preview(img, f'Quick look: {len(positions)} sharpest frames, rough', args,
                   slot='quick_look', min_interval=0.0)
        return True
    except RunCancelled:
        raise
    except Exception as exc:
        log.warning("quick look skipped: %s", exc)
        return False
