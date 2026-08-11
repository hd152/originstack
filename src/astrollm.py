"""astrollm integration: subprocess-based defect/quality/category scoring.

astrollm is a separately-trained model, still finishing its first real
training run. This is advisory/logging only -- results are stored on
FrameInfo.metrics['astrollm'] for visibility but never set f.accepted or
touch f.metrics['score'], and no frame is auto-dropped. Subprocess call
only, no network -- matches astrollm's local-only premise.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import numpy as np

from src.models import Config, FrameInfo
from src.utils import safe_print

logger = logging.getLogger('originstack')


def run_astrollm_infer(image_path: str, python_exe: str, script_path: str,
                       checkpoint_path: str,
                       timeout: float = Config.ASTROLLM_TIMEOUT_S) -> Optional[dict]:
    """Run astrollm's infer.py on one image and return its parsed JSON result.

    Returns None (logging a warning) on any failure -- bad exit code,
    timeout, missing binary, or unparseable stdout. Callers must treat None
    as "no score available", never as a rejection signal.
    """
    # infer.py is run with cwd=<its own dir> (so a relative --checkpoint
    # resolves against the astrollm repo, matching the documented
    # invocation) -- a relative image_path would resolve against that same
    # cwd instead of the caller's, so make it absolute first.
    cmd = [python_exe, script_path, '--checkpoint', checkpoint_path,
           '--image', os.path.abspath(image_path), '--json']
    try:
        proc = subprocess.run(cmd, cwd=os.path.dirname(script_path) or None,
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning(f"astrollm: timed out after {timeout:.0f}s on "
                       f"{os.path.basename(image_path)}")
        return None
    except (FileNotFoundError, OSError) as e:
        logger.warning(f"astrollm: could not launch subprocess for "
                       f"{os.path.basename(image_path)}: {e}")
        return None

    if proc.returncode != 0:
        logger.warning(f"astrollm: exit {proc.returncode} for "
                       f"{os.path.basename(image_path)}: "
                       f"{proc.stderr.strip()[-300:]}")
        return None

    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    if not lines:
        logger.warning(f"astrollm: empty stdout for {os.path.basename(image_path)}")
        return None
    try:
        return json.loads(lines[-1])
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"astrollm: could not parse JSON output for "
                       f"{os.path.basename(image_path)}: {e}")
        return None


def _astrollm_paths(args) -> Optional[tuple]:
    python_exe = getattr(args, 'astrollm_python', None)
    script_path = getattr(args, 'astrollm_script', None)
    checkpoint_path = getattr(args, 'astrollm_checkpoint', None)
    if not (python_exe and script_path and checkpoint_path):
        return None
    return python_exe, script_path, checkpoint_path


def score_lights_with_astrollm(lights: List[FrameInfo], args) -> None:
    """Score every accepted light frame with astrollm, advisory-only.

    Stores the raw result (or None on failure) at f.metrics['astrollm'].
    Logs a session-relative summary: frames flagged is_defective /
    stray_light_flag, and frames whose quality_score falls more than
    Config.ASTROLLM_OUTLIER_SIGMA below the session mean -- all log-only,
    matching astrollm's early/unvalidated integration status.
    """
    if not getattr(args, 'astrollm', False):
        return
    paths = _astrollm_paths(args)
    if paths is None:
        return
    python_exe, script_path, checkpoint_path = paths
    timeout = float(getattr(args, 'astrollm_timeout', Config.ASTROLLM_TIMEOUT_S))
    workers = max(1, int(getattr(args, 'astrollm_workers', 2)))

    targets = [f for f in lights if f.accepted]
    if not targets:
        return

    safe_print(f"\n  astrollm: scoring {len(targets)} frame(s) "
               f"({workers} worker(s))...")

    def _score(f: FrameInfo):
        return f, run_astrollm_infer(f.path, python_exe, script_path,
                                     checkpoint_path, timeout=timeout)

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_score, f) for f in targets]
        for fut in as_completed(futures):
            f, result = fut.result()
            if f.metrics is None:
                f.metrics = {}
            f.metrics['astrollm'] = result
            results[f.path] = result

    scored = {p: r for p, r in results.items() if r is not None}
    n_failed = len(targets) - len(scored)
    if n_failed:
        safe_print(f"  astrollm: {n_failed}/{len(targets)} frame(s) failed to score "
                   f"(see warnings above)")
    if not scored:
        return

    defective = [p for p, r in scored.items() if r.get('is_defective')]
    stray = [p for p, r in scored.items() if r.get('stray_light_flag')]
    if defective:
        safe_print(f"  astrollm: {len(defective)} frame(s) flagged is_defective "
                   f"(advisory -- not auto-dropped): "
                   + ", ".join(os.path.basename(p) for p in defective[:5])
                   + (", ..." if len(defective) > 5 else ""))
    if stray:
        safe_print(f"  astrollm: {len(stray)} frame(s) flagged stray_light "
                   f"(advisory -- not auto-dropped): "
                   + ", ".join(os.path.basename(p) for p in stray[:5])
                   + (", ..." if len(stray) > 5 else ""))

    scores = np.array([r.get('quality_score', 0.0) for r in scored.values()],
                      dtype=np.float64)
    paths_scored = list(scored.keys())
    if len(scores) >= 3:
        mean, std = float(np.mean(scores)), float(np.std(scores))
        if std > 1e-6:
            sigma_thresh = Config.ASTROLLM_OUTLIER_SIGMA
            below = [paths_scored[i] for i in range(len(scores))
                    if (mean - scores[i]) / std > sigma_thresh]
            if below:
                safe_print(f"  astrollm: {len(below)} frame(s) below-session-average "
                           f"quality_score (>{sigma_thresh:.1f}sigma, advisory only): "
                           + ", ".join(os.path.basename(p) for p in below[:5])
                           + (", ..." if len(below) > 5 else ""))


def score_master_with_astrollm(master_image_path: str, args,
                               inferred_type: Optional[str] = None) -> None:
    """Score the final stacked master with astrollm, advisory-only (log only).

    ``master_image_path`` must NOT be the pipeline's own main output FITS:
    astrollm's infer.py treats any ``.fits`` input as a raw undebayered
    single-plane Bayer light frame and runs it through its own cv2 debayer,
    which errors on our already-debayered (3, H, W) RGB cube. Callers must
    pass a rendered non-FITS image instead -- the TIFF export when present,
    else the preview JPEG (see src/pipeline.py's call site).

    Compares astrollm's predicted category against the pipeline's own
    metadata-based target inference and flags a mismatch as a possible
    misidentified/mislabeled session.
    """
    if not getattr(args, 'astrollm', False):
        return
    paths = _astrollm_paths(args)
    if paths is None:
        return
    python_exe, script_path, checkpoint_path = paths
    timeout = float(getattr(args, 'astrollm_timeout', Config.ASTROLLM_TIMEOUT_S))

    result = run_astrollm_infer(master_image_path, python_exe, script_path,
                                checkpoint_path, timeout=timeout)
    if result is None:
        safe_print("  astrollm: master scoring failed (see warning above)")
        return

    category = result.get('category')
    confidence = result.get('category_confidence', 0.0)
    safe_print(f"  astrollm (master): category={category} "
               f"conf={confidence:.0%}  "
               f"sky_brightness={result.get('sky_brightness', 0):.1f}  "
               f"stray_light_gradient={result.get('stray_light_gradient', 0):.1f}")

    if inferred_type and category and inferred_type != 'unknown':
        # astrollm's category head is a coarse 7-bucket taxonomy (galaxy,
        # nebula, star_cluster, comet, planet, star, other) while
        # inferred_type is fine-grained (emission_nebula, reflection_nebula,
        # planetary_nebula, globular_cluster, ...) -- an exact-string
        # compare would flag "nebula" vs "emission_nebula" as a mismatch
        # even though astrollm got it right. Match on shared word tokens
        # instead (e.g. both contain "nebula", or both contain "cluster").
        _inferred_words = set(inferred_type.replace('_', ' ').lower().split())
        _category_words = set(str(category).replace('_', ' ').lower().split())
        if not (_inferred_words & _category_words):
            logger.warning(
                f"astrollm: master category '{category}' does not match "
                f"pipeline-inferred target type '{inferred_type}' -- "
                f"possible misidentified/mislabeled session")
            safe_print(f"  astrollm WARNING: master category '{category}' "
                       f"vs inferred target type '{inferred_type}' mismatch")
