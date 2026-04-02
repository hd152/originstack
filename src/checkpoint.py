"""Session checkpoint/resume: save and restore pipeline state between phases."""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.models import FrameInfo, ProcessingStats
from src.utils import safe_print


def _checkpoint_dir(output_path: str) -> str:
    """Return checkpoint directory path based on output path."""
    return os.path.splitext(output_path)[0] + '_checkpoint'


def save_checkpoint(output_path: str, phase: int,
                    lights: List[FrameInfo],
                    final: Optional[List[FrameInfo]] = None,
                    shifts: Optional[List] = None,
                    dither_info: Optional[Dict] = None,
                    stats: Optional[ProcessingStats] = None) -> None:
    """Save pipeline state after a completed phase."""
    ckpt_dir = _checkpoint_dir(output_path)
    os.makedirs(ckpt_dir, exist_ok=True)

    state = {
        'phase': phase,
        'timestamp': time.time(),
        'n_lights': len(lights),
    }

    # Save frame info (paths, metrics, accepted status)
    frame_data = []
    for f in lights:
        fd = {
            'path': f.path,
            'type': f.type,
            'accepted': f.accepted,
            'shift': list(f.shift) if f.shift else [0.0, 0.0],
        }
        if f.metrics:
            # Exclude non-serializable items
            fd['metrics'] = {k: v for k, v in f.metrics.items()
                             if k != '_star_sources' and isinstance(v, (int, float, str, bool))}
        frame_data.append(fd)
    state['frames'] = frame_data

    if final is not None:
        state['final_indices'] = [i for i, f in enumerate(lights) if f in final]

    if shifts is not None:
        state['shifts'] = [list(s) if s else [0.0, 0.0] for s in shifts]

    if dither_info is not None:
        # Filter to serializable values
        state['dither_info'] = {k: v for k, v in dither_info.items()
                                if isinstance(v, (int, float, str, bool, list))}

    if stats is not None:
        state['stats'] = {
            'quality_time': stats.quality_time,
            'registration_time': stats.registration_time,
            'total_frames': stats.total_frames,
            'accepted_frames': stats.accepted_frames,
            'rejected_frames': stats.rejected_frames,
        }

    ckpt_path = os.path.join(ckpt_dir, 'checkpoint.json')
    with open(ckpt_path, 'w') as f:
        json.dump(state, f, indent=2)
    safe_print(f"  Checkpoint saved: phase {phase} complete")


def load_checkpoint(output_path: str) -> Optional[Dict]:
    """Load checkpoint if it exists. Returns None if no checkpoint found."""
    ckpt_dir = _checkpoint_dir(output_path)
    ckpt_path = os.path.join(ckpt_dir, 'checkpoint.json')

    if not os.path.exists(ckpt_path):
        return None

    try:
        with open(ckpt_path, 'r') as f:
            state = json.load(f)
        return state
    except Exception as e:
        safe_print(f"  WARNING: Checkpoint file corrupt or unreadable ({e}) — starting fresh")
        return None


def can_resume(output_path: str, lights: List[FrameInfo]) -> Tuple[bool, int, Optional[Dict]]:
    """Check if we can resume from a checkpoint.

    Returns (can_resume, completed_phase, checkpoint_data).
    Validates that frame paths match the current input.
    """
    state = load_checkpoint(output_path)
    if state is None:
        return False, 0, None

    # Validate frame paths match
    saved_paths = {f['path'] for f in state.get('frames', [])}
    current_paths = {f.path for f in lights}

    if saved_paths != current_paths:
        safe_print(f"  Checkpoint found but frame set changed — starting fresh")
        return False, 0, None

    phase = state.get('phase', 0)
    age_hours = (time.time() - state.get('timestamp', 0)) / 3600
    if age_hours > 24:
        safe_print(f"  Checkpoint found but too old ({age_hours:.0f}h) — starting fresh")
        return False, 0, None

    safe_print(f"  Checkpoint found: phase {phase} complete ({age_hours:.1f}h ago)")
    return True, phase, state


def restore_frame_state(lights: List[FrameInfo], state: Dict) -> List[FrameInfo]:
    """Restore frame metrics and accepted status from checkpoint."""
    frame_map = {fd['path']: fd for fd in state.get('frames', [])}

    for f in lights:
        fd = frame_map.get(f.path)
        if fd:
            f.accepted = fd.get('accepted', True)
            if fd.get('metrics'):
                f.metrics = fd['metrics']
            f.shift = tuple(fd.get('shift', [0.0, 0.0]))

    final_indices = state.get('final_indices', [])
    return [lights[i] for i in final_indices if i < len(lights)]


def cleanup_checkpoint(output_path: str) -> None:
    """Remove checkpoint files after successful completion."""
    ckpt_dir = _checkpoint_dir(output_path)
    if os.path.exists(ckpt_dir):
        try:
            import shutil
            shutil.rmtree(ckpt_dir)
        except Exception:
            pass
