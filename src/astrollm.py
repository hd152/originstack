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
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np

from src.models import Config, FrameInfo
from src.utils import safe_print

logger = logging.getLogger('originstack')

_FITS_EXTS = ('.fits', '.fit', '.fts')


def _render_light_for_onnx(src_path: str) -> Tuple[str, bool]:
    """Debayer a raw light frame to a temp RGB image for ``infer_onnx.py``.

    The vendored ONNX entry point (``vendor/astrollm/infer_onnx.py``) reads
    TIFF/PNG/JPG only -- it dropped astropy, so it refuses FITS. OriginStack
    renders each raw frame itself here: a small score drift vs astrollm's own
    cv2 debayer, acceptable for an advisory-only signal.

    Returns ``(path, is_temp)``. Non-FITS input (the already-rendered stacked
    master) passes straight through. If rendering isn't possible -- unreadable
    frame, or neither ``tifffile`` nor ``Pillow`` available -- the original
    path is returned unchanged so ``infer_onnx.py`` rejects it and the frame
    simply goes unscored, rather than raising into the pipeline.
    """
    if not src_path.lower().endswith(_FITS_EXTS):
        return src_path, False
    try:
        from src.io_fits import load_frame
        data, hdr = load_frame(src_path)
        arr = np.asarray(data)
        if arr.ndim == 2:
            from src.debayer import debayer
            pattern = str(hdr.get('BAYERPAT') or hdr.get('COLORTYP')
                          or 'GBRG').strip().upper()
            rgb = debayer(arr.astype(np.float32), pattern=pattern)
        elif arr.ndim == 3:
            rgb = arr.astype(np.float32)
            if rgb.shape[0] in (1, 3) and rgb.shape[-1] not in (1, 3):
                rgb = np.moveaxis(rgb, 0, -1)          # (C,H,W) -> (H,W,C)
        else:
            return src_path, False

        lo, hi = float(np.nanmin(rgb)), float(np.nanmax(rgb))
        if hi <= lo:
            hi = lo + 1.0
        # 16-bit range pack only -- let infer_onnx.py apply the per-channel
        # percentile stretch it was trained on.
        u16 = np.clip((rgb - lo) / (hi - lo) * 65535.0, 0, 65535).astype(np.uint16)

        fd, out = tempfile.mkstemp(suffix='.tiff', prefix='astrollm_')
        os.close(fd)
        try:
            import tifffile
            tifffile.imwrite(out, u16)
            return out, True
        except Exception:
            pass
        try:
            from PIL import Image
            png = out[:-5] + '.png'
            Image.fromarray((u16 >> 8).astype(np.uint8), 'RGB').save(png)
            os.unlink(out)
            return png, True
        except Exception:
            for p in (out, out[:-5] + '.png'):
                try:
                    os.unlink(p)
                except OSError:
                    pass
            logger.warning("astrollm: no tifffile/Pillow to render "
                           f"{os.path.basename(src_path)} for ONNX inference")
            return src_path, False
    except Exception as e:
        logger.warning(f"astrollm: could not render {os.path.basename(src_path)} "
                       f"for ONNX inference: {e}")
        return src_path, False


def run_astrollm_infer(image_path: str, python_exe: str, script_path: str,
                       checkpoint_path: str,
                       timeout: float = Config.ASTROLLM_TIMEOUT_S) -> Optional[dict]:
    """Run astrollm's ``infer_onnx.py`` on one image, return its parsed JSON.

    A raw ``.fits`` frame is debayered to a temp image first (see
    ``_render_light_for_onnx``) since the ONNX entry point takes TIFF/PNG/JPG
    only. Returns None (logging a warning) on any failure -- bad exit code,
    timeout, missing binary, unrenderable input, or unparseable stdout.
    Callers must treat None as "no score available", never as a rejection.
    """
    render_path, is_temp = _render_light_for_onnx(image_path)
    # infer_onnx.py is run with cwd=<its own dir> so its relative --model
    # default / `from data.imageops import` resolve against the vendored
    # copy; a relative image path would resolve there too, so make it
    # absolute first.
    cmd = [python_exe, script_path, '--model', checkpoint_path,
           '--image', os.path.abspath(render_path), '--json']
    try:
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
    finally:
        if is_temp:
            try:
                os.unlink(render_path)
            except OSError:
                pass


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

    Gated on --astrollm-score-all, not just --astrollm: scoring every
    accepted frame costs ~8s/frame (subprocess + model-load overhead, not
    per-image compute) -- minutes on a large session. Checked here too,
    not just at the call site, so calling this function directly is never
    accidentally slow regardless of caller.
    """
    if not (getattr(args, 'astrollm', False) and getattr(args, 'astrollm_score_all', False)):
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


# astrollm's category head is a coarse 4-class taxonomy (galaxy, nebula,
# star_cluster, comet) -- only two map unambiguously onto one of the
# pipeline's own 7 target-type anchors (auto_settings.py's _TYPE_ANCHORS).
# "nebula" alone can't distinguish emission/reflection/planetary, and comet
# isn't a target the target-type blend-weight system covers at all (comet
# has its own separate --comet-mode). A wrong guess here would misdirect
# --auto's whole preset blend, so ambiguous categories intentionally get no
# mapping (no boost) rather than a guessed one -- unlike
# score_master_with_astrollm's mismatch warning above, which can afford to
# be fuzzy since a human reads it. (Older checkpoints had a 7-bucket head
# with planet/star/other; those keys are simply absent here, still None.)
_CATEGORY_TO_TARGET_TYPE = {
    'galaxy': 'galaxy',
    'star_cluster': 'globular_cluster',
}


def map_astrollm_category(category: Optional[str]) -> Optional[str]:
    """astrollm category -> one of auto_settings.py's target-type anchors,
    or None if there's no unambiguous mapping (see _CATEGORY_TO_TARGET_TYPE)."""
    if not category:
        return None
    return _CATEGORY_TO_TARGET_TYPE.get(str(category).lower())


def sample_session_priors(lights: List[FrameInfo], args) -> Optional[dict]:
    """Fast, session-level astrollm signal for --auto: a category (fed into
    the auto-advisor's existing prior_type/prior_confidence boost -- the
    same mechanism SIMBAD/header metadata inference already uses) and a
    defect flag (a defensive nudge toward --trail-reject + stronger chroma
    denoising in _apply_quality_settings, never a frame rejection).

    Deliberately scores a SMALL SAMPLE, not the whole session: one
    astrollm subprocess call still costs a few seconds, dominated by
    interpreter + onnxruntime startup and the per-frame debayer render
    rather than the ONNX inference itself -- scoring every accepted frame
    (score_lights_with_astrollm's job, a
    separate opt-in path) would add minutes to a session with 100+
    frames, which defeats the point of a fast pre-stacking signal. Mirrors
    frame_processor.py's _measure_session_ca: a few frames spread through
    the session (early/middle/late) rather than just the first one, on
    the same "this is a fixed property of the session, not a per-frame
    one" reasoning chromatic aberration already uses.

    Returns None if astrollm isn't configured, or every sampled call
    failed. Never touches f.accepted/f.metrics -- this is a session-level
    signal, computed separately from (and not a replacement for)
    score_lights_with_astrollm's own per-frame advisory scoring.
    """
    if not getattr(args, 'astrollm', False):
        return None
    paths = _astrollm_paths(args)
    if paths is None:
        return None
    python_exe, script_path, checkpoint_path = paths
    timeout = float(getattr(args, 'astrollm_timeout', Config.ASTROLLM_TIMEOUT_S))

    accepted = [f for f in lights if f.accepted]
    if not accepted:
        return None
    n = len(accepted)
    idxs = sorted({n // 6, n // 2, (5 * n) // 6})

    def _one(i: int):
        return run_astrollm_infer(accepted[i].path, python_exe, script_path,
                                  checkpoint_path, timeout=timeout)

    results = []
    try:
        with ThreadPoolExecutor(max_workers=len(idxs)) as ex:
            for fut in [ex.submit(_one, i) for i in idxs]:
                try:
                    r = fut.result()
                    if r is not None:
                        results.append(r)
                except Exception:
                    pass
    except Exception:
        return None
    if not results:
        return None

    # Category: majority vote across samples, ties broken by mean confidence.
    by_category: dict = {}
    for r in results:
        c = r.get('category')
        if c:
            by_category.setdefault(c, []).append(float(r.get('category_confidence', 0.0)))
    category, confidence = None, 0.0
    if by_category:
        category = max(by_category, key=lambda c: (len(by_category[c]), np.mean(by_category[c])))
        confidence = float(np.mean(by_category[category]))

    defect_flagged = any(
        r.get('is_defective') or r.get('stray_light_flag')
        or float(r.get('defect_probability', 0.0)) > 0.5
        for r in results)

    safe_print(f"\n  astrollm (session sample, {len(results)}/{len(idxs)} frame(s)): "
               f"category={category} conf={confidence:.0%}"
               + ("  [defect signal flagged]" if defect_flagged else ""))

    return {'category': category, 'category_confidence': confidence,
            'defect_flagged': defect_flagged}


def score_master_with_astrollm(master_image_path: str, args,
                               inferred_type: Optional[str] = None) -> None:
    """Score the final stacked master with astrollm, advisory-only (log only).

    Pass a rendered non-FITS image (the TIFF export when present, else the
    preview JPEG -- see src/pipeline.py's call site). A ``.fits`` path would
    be run through ``_render_light_for_onnx``'s single-frame Bayer debayer,
    wrong for an already-stacked (3, H, W) RGB master.

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
    _exp = result.get('predicted_exposure_s')
    _exp_str = f"  predicted_exposure={_exp:.0f}s" if _exp else ""
    safe_print(f"  astrollm (master): category={category} "
               f"conf={confidence:.0%}  "
               f"sky_brightness={result.get('sky_brightness', 0):.1f}  "
               f"stray_light_gradient={result.get('stray_light_gradient', 0):.1f}"
               f"{_exp_str}")

    if inferred_type and category and inferred_type != 'unknown':
        # astrollm's category head is a coarse 4-class taxonomy (galaxy,
        # nebula, star_cluster, comet) while inferred_type is fine-grained
        # (emission_nebula, reflection_nebula,
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
