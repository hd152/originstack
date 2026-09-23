"""Mine REAL astrophotography sessions for transient-triage training data,
as a companion to the fully-synthetic ``tools/gen_transient_triage_data.py``.

Real light frames of the same target on different nights give two things a
synthetic star field can't:

- **Real negatives (bogus class)**: stack each session with this project's
  own pipeline (a real ``originstack.py`` run, not a shortcut), then run the
  same ``--transient-detect`` comparison this codebase ships between
  consecutive sessions. Since no known real transient is expected in most
  amateur fields, every candidate that survives is a genuine artifact --
  cosmic ray, registration-slip dipole, hot pixel -- that got past Phase 1,
  the real production population rather than a synthetic guess at it.
- **Real positives (real-transient class)**: still can't get for free (no
  labelled real transients exist), but injecting a synthetic point source
  into a *copy* of one real stacked epoch before differencing rides on real
  noise, real PSF and real artifacts -- a meaningful upgrade over a fully
  synthetic star field for the "real" class too.

Usage:
    python tools/mine_real_transient_data.py \\
        --target-dir "G:\\astro\\Astrophotography\\Fireworks Galaxy" \\
        --work-dir transient_triage_real_work \\
        --out transient_triage_real_data.npz \\
        [--max-sessions N] [--n-inject-per-session 3] [--threshold 5.0]

Each session subfolder under ``--target-dir`` is stacked once and cached in
``--work-dir`` (an existing ``<session>.fits`` there is reused, not
re-stacked) -- a real multi-hundred-frame session can take minutes, so re-runs
while iterating on this script don't pay that cost twice. Never writes
anything back into ``--target-dir``.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.difference_imaging import _compare_epochs, estimate_background_sigma  # noqa: E402
from src.transient_triage import DEFAULT_STAMP_SIZE, build_stamps  # noqa: E402

_MATCH_RADIUS_PX = 4.0
_ORIGINSTACK_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'originstack.py'))


def _gaussian_psf(size: int, fwhm: float) -> np.ndarray:
    """Normalised Gaussian kernel -- same construction as
    tools/gen_transient_triage_data.py's copy, kept independent per this
    project's tools/ convention of freestanding scripts."""
    if size % 2 == 0:
        size += 1
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    c = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    g = np.exp(-(((yy - c) ** 2 + (xx - c) ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def _inject_point_source(rgb: np.ndarray, y: float, x: float, flux: float,
                         size: int = 21, fwhm: float = 3.0) -> np.ndarray:
    """Additively place a synthetic point source into a COPY of ``rgb``,
    equally across channels. ``flux`` is total (the kernel already sums to
    1), matching the convention tools/gen_transient_triage_data.py's
    ``_render_field`` uses (place flux at one pixel, then PSF-convolve)."""
    out = rgb.copy()
    patch = _gaussian_psf(size, fwhm) * flux
    half = size // 2
    h, w = rgb.shape[:2]
    y0, x0 = int(round(y)) - half, int(round(x)) - half
    ys, ye = max(0, y0), min(h, y0 + size)
    xs, xe = max(0, x0), min(w, x0 + size)
    if ys >= ye or xs >= xe:
        return out
    py0, px0 = ys - y0, xs - x0
    py1, px1 = py0 + (ye - ys), px0 + (xe - xs)
    sub_patch = patch[py0:py1, px0:px1]
    if out.ndim == 3:
        out[ys:ye, xs:xe, :] += sub_patch[:, :, None]
    else:
        out[ys:ye, xs:xe] += sub_patch
    return out


def _load_rgb(path: str) -> np.ndarray:
    from src.io_fits import load_fits
    arr, _ = load_fits(path)
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def discover_sessions(target_dir: str) -> list:
    """Immediate subdirectories of ``target_dir`` that contain FITS light
    frames, sorted by name -- session folder names are timestamped
    (``Target_YYYY-MM-DD_HH-MM-SS``), so name order is chronological order."""
    sessions = []
    for entry in sorted(os.listdir(target_dir)):
        d = os.path.join(target_dir, entry)
        if not os.path.isdir(d):
            continue
        if glob.glob(os.path.join(d, '*.fit*')):
            sessions.append(d)
    return sessions


def stack_session(session_dir: str, out_fits: str) -> bool:
    """Stack one session with the real pipeline, caching the result. Returns
    True on success (including a cache hit)."""
    if os.path.exists(out_fits):
        print(f'  (cached) {os.path.basename(out_fits)}')
        return True
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, _ORIGINSTACK_PY, '-d', session_dir, '-o', out_fits],
        capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0 or not os.path.exists(out_fits):
        print(f'  FAILED to stack {session_dir} ({elapsed:.0f}s):')
        print('    ' + (proc.stderr or proc.stdout)[-2000:].replace('\n', '\n    '))
        return False
    print(f'  stacked {os.path.basename(out_fits)} in {elapsed:.0f}s')
    return True


def mine_negatives(new_path: str, ref_path: str, size: int, threshold: float):
    """Every candidate from a real cross-session comparison is a hard
    negative -- no known real transient is expected between two ordinary
    nights of the same amateur target."""
    new_rgb, ref_rgb = _load_rgb(new_path), _load_rgb(ref_path)
    comparison = _compare_epochs(new_rgb, ref_rgb, threshold=threshold)
    if comparison is None or not comparison.transients:
        return np.zeros((0, 3, size, size), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    sigma_new = estimate_background_sigma(comparison.new_lum)
    sigma_ref = estimate_background_sigma(comparison.ref_lum)
    sigma_diff = estimate_background_sigma(comparison.difference)
    positions = [(t.y, t.x) for t in comparison.transients]
    stamps = build_stamps(comparison.new_lum.astype(np.float32),
                          comparison.ref_lum.astype(np.float32),
                          comparison.difference, positions,
                          sigma_new, sigma_ref, sigma_diff, size=size)
    labels = np.zeros(len(positions), dtype=np.float32)  # all bogus
    return stamps, labels


def mine_positives(stack_path: str, size: int, threshold: float,
                   n_inject: int, rng: np.random.Generator):
    """Inject synthetic point sources into a copy of a real stacked epoch,
    difference against the untouched original, and label the recovered
    injection sites real (everything else found is a hard negative)."""
    base_rgb = _load_rgb(stack_path)
    from src.difference_imaging import _to_luminance
    base_sigma = estimate_background_sigma(_to_luminance(base_rgb))
    h, w = base_rgb.shape[:2]

    all_stamps, all_labels = [], []
    margin = 25
    for _ in range(n_inject):
        y, x = rng.uniform(margin, h - margin), rng.uniform(margin, w - margin)
        flux = float(rng.uniform(20, 60)) * base_sigma
        injected = _inject_point_source(base_rgb, y, x, flux,
                                        fwhm=float(rng.uniform(2.5, 4.0)))
        comparison = _compare_epochs(injected, base_rgb, threshold=threshold)
        if comparison is None or not comparison.transients:
            continue
        sigma_new = estimate_background_sigma(comparison.new_lum)
        sigma_ref = estimate_background_sigma(comparison.ref_lum)
        sigma_diff = estimate_background_sigma(comparison.difference)
        positions = [(t.y, t.x) for t in comparison.transients]
        stamps = build_stamps(comparison.new_lum.astype(np.float32),
                              comparison.ref_lum.astype(np.float32),
                              comparison.difference, positions,
                              sigma_new, sigma_ref, sigma_diff, size=size)
        labels = np.array([1.0 if math.hypot(t.y - y, t.x - x) <= _MATCH_RADIUS_PX else 0.0
                          for t in comparison.transients], dtype=np.float32)
        all_stamps.append(stamps)
        all_labels.append(labels)

    if not all_stamps:
        return np.zeros((0, 3, size, size), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.concatenate(all_stamps, axis=0), np.concatenate(all_labels, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--target-dir', required=True,
                        help='Directory of session subfolders for ONE target, e.g. '
                             '"G:\\astro\\Astrophotography\\Fireworks Galaxy"')
    parser.add_argument('--work-dir', default=None,
                        help='Where stacked FITS + sidecars are cached (default: '
                             'tools/../transient_triage_real_work/<target name>)')
    parser.add_argument('--out', default=None,
                        help='Output .npz (default: tools/../transient_triage_real_data.npz)')
    parser.add_argument('--append-to', default=None,
                        help='Merge with an existing .npz (e.g. the synthetic one from '
                             'gen_transient_triage_data.py) instead of overwriting --out')
    parser.add_argument('--max-sessions', type=int, default=None,
                        help='Only stack/use the first N sessions, by chronological name '
                             'order (default: all)')
    parser.add_argument('--session-filter', default=None, metavar='SUBSTR',
                        help='Only use session folders whose name contains this substring '
                             '(applied before --max-sessions) -- handy for picking a small '
                             'session to validate against before committing to a big one')
    parser.add_argument('--n-inject-per-session', type=int, default=3)
    parser.add_argument('--threshold', type=float, default=5.0)
    parser.add_argument('--size', type=int, default=DEFAULT_STAMP_SIZE)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    target_name = os.path.basename(os.path.normpath(args.target_dir))
    work_dir = args.work_dir or os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'transient_triage_real_work', target_name))
    os.makedirs(work_dir, exist_ok=True)
    out_path = args.out or os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'transient_triage_real_data.npz'))

    sessions = discover_sessions(args.target_dir)
    if args.session_filter:
        sessions = [s for s in sessions if args.session_filter in os.path.basename(s)]
    if args.max_sessions:
        sessions = sessions[:args.max_sessions]
    if len(sessions) < 2:
        raise SystemExit(f"only {len(sessions)} session(s) with FITS lights found under "
                         f"{args.target_dir} -- need at least 2 to compare epochs")
    print(f'{target_name}: {len(sessions)} session(s)')

    stacked_paths = []
    for s in sessions:
        out_fits = os.path.join(work_dir, os.path.basename(s) + '.fits')
        if stack_session(s, out_fits):
            stacked_paths.append(out_fits)

    if len(stacked_paths) < 2:
        raise SystemExit(f"only {len(stacked_paths)} session(s) stacked successfully -- "
                         f"need at least 2")

    rng = np.random.default_rng(args.seed)
    all_stamps, all_labels = [], []

    print('Mining real negatives from consecutive session pairs...')
    for i in range(len(stacked_paths) - 1):
        stamps, labels = mine_negatives(stacked_paths[i + 1], stacked_paths[i],
                                        args.size, args.threshold)
        print(f'  {os.path.basename(stacked_paths[i + 1])} vs '
             f'{os.path.basename(stacked_paths[i])}: {len(labels)} candidate(s)')
        if len(labels):
            all_stamps.append(stamps)
            all_labels.append(labels)

    if args.n_inject_per_session > 0:
        print('Mining real-image-injection positives...')
        for p in stacked_paths:
            stamps, labels = mine_positives(p, args.size, args.threshold,
                                            args.n_inject_per_session, rng)
            n_pos = int(labels.sum())
            print(f'  {os.path.basename(p)}: {n_pos} real, {len(labels) - n_pos} bogus '
                 f'(of {args.n_inject_per_session} injected)')
            if len(labels):
                all_stamps.append(stamps)
                all_labels.append(labels)

    X = (np.concatenate(all_stamps, axis=0) if all_stamps
        else np.zeros((0, 3, args.size, args.size), dtype=np.float32))
    y = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,), dtype=np.float32)

    if args.append_to and os.path.exists(args.append_to):
        prev = np.load(args.append_to)
        if int(prev['size']) != args.size:
            raise SystemExit(f"--append-to size {int(prev['size'])} != --size {args.size}")
        X = np.concatenate([prev['X'], X], axis=0)
        y = np.concatenate([prev['y'], y], axis=0)
        out_path = args.append_to

    np.savez(out_path, X=X, y=y, size=args.size)
    n_pos = int(y.sum())
    print(f'Wrote {len(y)} labelled stamps ({n_pos} real, {len(y) - n_pos} bogus) to {out_path}')


if __name__ == '__main__':
    main()
