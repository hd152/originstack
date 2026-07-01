"""Session checkpoint/resume: save and restore pipeline state between phases."""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.models import FrameInfo, ProcessingStats
from src.utils import safe_print

# Sentinel for _to_json — module-level so it's not re-created on every call.
_SKIP = object()


def _checkpoint_dir(output_path: str) -> str:
    return os.path.splitext(output_path)[0] + '_checkpoint'


def _raw_stack_path(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, 'raw_stack.npy')


def _ckpt_json_path(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, 'checkpoint.json')


def _transforms_path(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, 'transforms.npy')


def save_raw_stack(output_path: str, stacked: np.ndarray) -> None:
    """Save the pre-post-processing stacked array to the checkpoint directory."""
    ckpt_dir = _checkpoint_dir(output_path)
    os.makedirs(ckpt_dir, exist_ok=True)
    arr32 = stacked.astype(np.float32)
    np.save(_raw_stack_path(ckpt_dir), arr32)
    size_mb = arr32.nbytes / (1024 ** 2)
    safe_print(f"  Raw stack saved to checkpoint ({size_mb:.0f} MB)")


def load_raw_stack(output_path: str) -> Optional[np.ndarray]:
    """Load the pre-post-processing stacked array from the checkpoint directory."""
    path = _raw_stack_path(_checkpoint_dir(output_path))
    if not os.path.exists(path):
        return None
    try:
        arr = np.load(path)
        safe_print(f"  Loaded raw stack from checkpoint "
                   f"({arr.shape[0]}x{arr.shape[1]}x{arr.shape[2]}, "
                   f"{arr.nbytes / (1024**2):.0f} MB)")
        return arr
    except Exception as e:
        safe_print(f"  WARNING: Could not load raw stack ({e})")
        return None


def save_checkpoint(output_path: str, phase: int,
                    lights: List[FrameInfo],
                    final: Optional[List[FrameInfo]] = None,
                    shifts: Optional[List] = None,
                    transforms: Optional[List] = None,
                    dither_info: Optional[Dict] = None,
                    stats: Optional[ProcessingStats] = None,
                    crop: Optional[List[int]] = None) -> None:
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
            fd['metrics'] = {k: v for k, v in f.metrics.items()
                             if k != '_star_sources' and isinstance(v, (int, float, str, bool))}
        frame_data.append(fd)
    state['frames'] = frame_data

    if final is not None:
        final_ids = {id(f) for f in final}
        state['final_indices'] = [i for i, f in enumerate(lights) if id(f) in final_ids]

    if shifts is not None:
        state['shifts'] = [list(s) if s else [0.0, 0.0] for s in shifts]

    # Affine transforms are numpy arrays (or None); save separately as .npy
    if transforms is not None:
        _save_transforms(ckpt_dir, transforms)
        state['has_transforms'] = True

    if dither_info is not None:
        def _to_json(v):
            """Recursively convert v to a JSON-safe Python value, or return _SKIP."""
            if v is None or isinstance(v, (bool, str)):
                return v
            if isinstance(v, np.ndarray):
                return _SKIP
            if isinstance(v, np.integer):
                return int(v)
            if isinstance(v, np.floating):
                return float(v)
            if isinstance(v, (int, float)):
                return v
            if isinstance(v, (list, tuple)):
                out = []
                for item in v:
                    converted = _to_json(item)
                    if converted is _SKIP:
                        return _SKIP
                    out.append(converted)
                return out
            return _SKIP

        safe_dither: dict = {}
        for k, v in dither_info.items():
            converted = _to_json(v)
            if converted is not _SKIP:
                safe_dither[k] = converted
        state['dither_info'] = safe_dither

    if crop is not None:
        state['crop'] = [int(v) for v in crop]

    if stats is not None:
        state['stats'] = {
            'quality_time': stats.quality_time,
            'registration_time': stats.registration_time,
            'total_frames': stats.total_frames,
            'accepted_frames': stats.accepted_frames,
            'rejected_frames': stats.rejected_frames,
        }

    with open(_ckpt_json_path(ckpt_dir), 'w') as f:
        json.dump(state, f, indent=2)
    safe_print(f"  Checkpoint saved: phase {phase} complete")


class _RestoredTransform:
    """Minimal shim wrapping a saved 3×3 matrix as a .params-bearing transform.

    All downstream consumers (calc_common_crop, apply_transform, stacking,
    quality-map patching) only access .params, so a full skimage object is
    not required here.
    """
    __slots__ = ('params',)

    def __init__(self, matrix: np.ndarray) -> None:
        self.params = np.asarray(matrix, dtype=np.float64)


def _save_transforms(ckpt_dir: str, transforms: List) -> None:
    """Serialize a list of affine transform matrices (or None) to transforms.npy."""
    arr = np.empty(len(transforms), dtype=object)
    for i, t in enumerate(transforms):
        if t is None:
            arr[i] = None
        elif hasattr(t, 'params'):
            arr[i] = np.array(t.params, dtype=np.float64)
        else:
            arr[i] = np.array(t, dtype=np.float64)
    np.save(_transforms_path(ckpt_dir), arr, allow_pickle=True)


def load_transforms(output_path: str, n_frames: int) -> List:
    """Load affine transforms from checkpoint.

    Returns a list of length *n_frames* where each entry is a
    _RestoredTransform (has .params) or None.  Falls back to all-None if the
    file is absent or unreadable.
    """
    path = _transforms_path(_checkpoint_dir(output_path))
    if not os.path.exists(path):
        return [None] * n_frames
    try:
        arr = np.load(path, allow_pickle=True)
        if len(arr) != n_frames:
            safe_print(f"  WARNING: transforms checkpoint length mismatch "
                       f"({len(arr)} vs {n_frames}) — affine skipped")
            return [None] * n_frames
        return [_RestoredTransform(t) if t is not None else None for t in arr]
    except Exception as e:
        safe_print(f"  WARNING: Could not load transforms from checkpoint ({e}) — affine skipped")
        return [None] * n_frames


def load_checkpoint(output_path: str) -> Optional[Dict]:
    """Load checkpoint if it exists. Returns None if no checkpoint found."""
    ckpt_dir = _checkpoint_dir(output_path)
    path = _ckpt_json_path(ckpt_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
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

    saved_paths = {f['path'] for f in state.get('frames', [])}
    current_paths = {f.path for f in lights}

    if saved_paths != current_paths:
        safe_print(f"  Checkpoint found but frame set changed — starting fresh")
        return False, 0, None

    phase = state.get('phase', 0)
    age_hours = (time.time() - state.get('timestamp', 0)) / 3600
    if age_hours > 72:
        safe_print(f"  Checkpoint found but too old ({age_hours:.0f}h) — starting fresh")
        return False, 0, None

    # Phase 3 requires the raw stack array on disk — downgrade if missing
    if phase >= 3:
        if not os.path.exists(_raw_stack_path(_checkpoint_dir(output_path))):
            phase = 2
            safe_print(f"  Checkpoint found: phase 3 complete but no raw_stack.npy "
                       f"— resuming from phase 2 ({age_hours:.1f}h ago)")
        else:
            safe_print(f"  Checkpoint found: phase {phase} complete ({age_hours:.1f}h ago) "
                       f"— will skip phases 1-3 and re-run post-processing only")
    else:
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
