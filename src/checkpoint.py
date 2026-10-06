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

# What a checkpoint's saved state depends on, split by the phase that reads it.
# A checkpoint used to be matched on the set of light-frame paths alone, so
# re-running with a different --stack-method, rejection sigma, debayer method
# (or edited raw files) silently resumed from the old stack. The CLI groups
# are the source of truth for which flags feed which phase; the extra names
# are config-file/--auto-managed attributes with no flag of their own.
_P1_GROUPS = ('Frames & calibration (Phase 1)',)
_P23_GROUPS = ('Registration & stacking (Phases 2-3)', 'Comet mode')
_P1_EXTRA = ('cal_dir', 'vignette_map', 'use_gpu', 'gpu_phase1', 'auto', 'preset',
             'advanced_metrics', 'banding_amount', 'banding_sigma', 'pre_gradient_removal',
             'max_ellipticity')
_P23_EXTRA = ('consensus_ref', 'drizzle_psf_wiener_k', 'esd_max_outliers', 'esd_significance',
              'ibp_relax', 'ivw_gain', 'linear_fit_iters', 'linear_fit_sigma_high',
              'linear_fit_sigma_low', 'masked_correlation', 'moving_objects_threshold',
              'no_alignment_centrality', 'no_shift_outlier_filter', 'patch_registration',
              'percentile_high', 'percentile_low', 'rejection_estimator',
              'skip_phase_correlation', 'wavelet_combine_levels', 'weight_fwhm',
              'weight_noise', 'weight_snr', 'weight_stars',
              # Core flag, but --auto derives the stacking settings from it.
              'target_type')
# In the Phases 2-3 group but only read by Phase 4 / the preview.
_P23_IGNORE = ('error_aware_stretch', 'uncertainty_propagate', 'uncertainty_realizations')


def _group_dests(titles) -> List[str]:
    from src.cli import build_parser  # lazy: cli imports pipeline imports this module
    out: List[str] = []
    for g in build_parser()._action_groups:
        if g.title in titles:
            out.extend(a.dest for a in g._group_actions)
    return out


def stack_fingerprint(args, lights: List[FrameInfo]) -> Dict:
    """Settings and inputs a checkpoint is only valid for.

    ``inputs`` hashes each light's path, size and mtime; ``p1``/``p23`` hold the
    repr of every setting read by Phase 1 and by Phases 2-3. Computed before
    --auto mutates anything, so identical command lines give identical prints.
    """
    import hashlib

    h = hashlib.sha256()
    for f in sorted(lights, key=lambda x: x.path):
        base = f.path.split('::')[0]           # SER virtual frames share a file
        try:
            st = os.stat(base)
            h.update(f"{f.path}|{st.st_size}|{st.st_mtime_ns}\n".encode())
        except OSError:
            h.update(f"{f.path}|missing\n".encode())

    def _vals(names) -> Dict[str, str]:
        return {n: repr(getattr(args, n, None)) for n in sorted(set(names))}

    p1 = _vals(list(_group_dests(_P1_GROUPS)) + list(_P1_EXTRA))
    p23 = _vals([d for d in _group_dests(_P23_GROUPS) if d not in _P23_IGNORE]
                + list(_P23_EXTRA))
    return {'inputs': h.hexdigest(), 'p1': p1, 'p23': p23}


def _changed(saved: Dict[str, str], now: Dict[str, str]) -> List[str]:
    return sorted(k for k in set(saved) | set(now) if saved.get(k) != now.get(k))


def _checkpoint_dir(output_path: str) -> str:
    return os.path.splitext(output_path)[0] + '_checkpoint'


def _raw_stack_path(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, 'raw_stack.npy')


def _ckpt_json_path(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, 'checkpoint.json')


def _transforms_path(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, 'transforms.npy')


def _field_list_path(ckpt_dir: str, name: str) -> str:
    """Path for a per-frame ndarray-list checkpoint (displacement fields,
    patch quality maps) that cannot be JSON-serialised inside dither_info."""
    return os.path.join(ckpt_dir, f'{name}.npy')


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
                    crop: Optional[List[int]] = None,
                    fingerprint: Optional[Dict] = None) -> None:
    """Save pipeline state after a completed phase."""
    ckpt_dir = _checkpoint_dir(output_path)
    os.makedirs(ckpt_dir, exist_ok=True)

    state = {
        'phase': phase,
        'timestamp': time.time(),
        'n_lights': len(lights),
    }
    if fingerprint is not None:
        state['fingerprint'] = fingerprint

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
        # Persist the per-frame ndarray lists that JSON can't hold (elastic
        # displacement fields, patch quality maps) as .npy so a phase-2 resume
        # stacks with them, reproducing the uninterrupted result.
        for _field_name in ('displacement_fields', 'quality_maps'):
            _flds = dither_info.get(_field_name)
            if _flds is not None:
                try:
                    _save_field_list(ckpt_dir, _field_name, list(_flds))
                except Exception as _e:
                    safe_print(f"  WARNING: could not checkpoint {_field_name} ({_e})")

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


def _save_field_list(ckpt_dir: str, name: str, fields: List) -> None:
    """Serialize a per-frame list of ndarrays (or None) to <name>.npy.

    Used for displacement fields (--elastic-registration) and patch quality
    maps (--patch-registration): both are lists of small ndarrays that live in
    dither_info but are dropped by the JSON serialiser, so without this a
    phase-2 resume would silently stack without them (a different result than
    an uninterrupted run)."""
    arr = np.empty(len(fields), dtype=object)
    for i, fld in enumerate(fields):
        arr[i] = None if fld is None else np.asarray(fld)
    np.save(_field_list_path(ckpt_dir, name), arr, allow_pickle=True)


def load_field_list(output_path: str, name: str, n_frames: int) -> Optional[List]:
    """Load a per-frame ndarray-list checkpoint saved by ``_save_field_list``.

    Returns None when the file is absent (feature was off or not checkpointed)
    or on a length mismatch, so callers cleanly fall back to no fields."""
    path = _field_list_path(_checkpoint_dir(output_path), name)
    if not os.path.exists(path):
        return None
    try:
        arr = np.load(path, allow_pickle=True)
        if len(arr) != n_frames:
            safe_print(f"  WARNING: {name} checkpoint length mismatch "
                       f"({len(arr)} vs {n_frames}) — skipped")
            return None
        return [None if a is None else np.asarray(a) for a in arr]
    except Exception as e:
        safe_print(f"  WARNING: Could not load {name} from checkpoint ({e}) — skipped")
        return None


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


def can_resume(output_path: str, lights: List[FrameInfo],
               fingerprint: Optional[Dict] = None) -> Tuple[bool, int, Optional[Dict]]:
    """Check if we can resume from a checkpoint.

    Returns (can_resume, completed_phase, checkpoint_data).
    Validates that frame paths match the current input and, when both the
    checkpoint and the caller carry a ``stack_fingerprint``, that the light
    files and the settings each saved phase depends on are unchanged: a
    Phase 1 setting or an edited file starts fresh, a Phases 2-3 setting
    resumes after Phase 1 (its accepted-frame list is still valid).
    """
    state = load_checkpoint(output_path)
    if state is None:
        return False, 0, None

    saved_paths = {f['path'] for f in state.get('frames', [])}
    current_paths = {f.path for f in lights}

    if saved_paths != current_paths:
        safe_print("  Checkpoint found but frame set changed — starting fresh")
        return False, 0, None

    phase = state.get('phase', 0)
    age_hours = (time.time() - state.get('timestamp', 0)) / 3600
    saved_fp = state.get('fingerprint')
    verified = False
    if saved_fp is not None and fingerprint is not None:
        if saved_fp.get('inputs') != fingerprint['inputs']:
            safe_print("  Checkpoint found but light files changed on disk — starting fresh")
            return False, 0, None
        p1 = _changed(saved_fp.get('p1', {}), fingerprint['p1'])
        if p1:
            safe_print(f"  Checkpoint found but Phase 1 settings changed "
                       f"({', '.join(p1[:6])}{'...' if len(p1) > 6 else ''}) — starting fresh")
            return False, 0, None
        p23 = _changed(saved_fp.get('p23', {}), fingerprint['p23'])
        if p23 and phase >= 2:
            safe_print(f"  Checkpoint: registration/stacking settings changed "
                       f"({', '.join(p23[:6])}{'...' if len(p23) > 6 else ''}) — "
                       f"reusing Phase 1 only")
            phase = 1
        verified = True
    # A verified checkpoint is valid however old it is (--keep-checkpoint exists
    # to iterate on Phase 4 over days); an unverifiable legacy one expires.
    if age_hours > 72 and not verified:
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
