#!/usr/bin/env python3
"""Regression check on the twelve public sample sessions: stack each one and compare with a baseline.

    python tools/check_samples.py SAMPLES_DIR                 # stack all, compare, exit 1 on a difference
    python tools/check_samples.py SAMPLES_DIR --only 01,11    # some sessions (name prefixes)
    python tools/check_samples.py SAMPLES_DIR --update        # stack all and write a new baseline
    python tools/check_samples.py --adopt OUT_DIR             # baseline from existing OUT_DIR/NAME.log + NAME.fits

``SAMPLES_DIR`` holds the unzipped sessions (the folders of the Zenodo records, named like
``01_face-on_spiral_galaxy_M51_Whirlpool``). Each is stacked with ``originstack.py -v`` (the comet
session with ``--comet-mode``) and the run is compared with ``tools/data/samples_baseline.json``:

  * the target and its type as the pipeline identifies them (exact);
  * frames accepted (within 1) and the minutes of light stacked (within 2%);
  * the median per-frame star width from the quality table (within 15%);
  * on the stack itself: the median FWHM of its unsaturated stars and its pixel noise (within 15%);
    the noise only when the run's Gaia colour calibration agrees with the baseline's (its scales
    move the noise by ~15%: a run that could not reach Gaia gets a note, not a failure);
  * wall time is reported against the baseline and flagged only above 3x (machines differ).

A difference is not necessarily a bug (an intended change moves these numbers): look, then run
``--update`` to accept the new values. The baseline of 2026-11-02 is OriginStack 2.21.0 on an 8-core
laptop, CPU only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
BASELINE = os.path.join(ROOT, 'tools', 'data', 'samples_baseline.json')

TOL = {'accepted': 1, 'integration_min': 0.02, 'frame_fwhm': 0.15, 'stack_fwhm': 0.15, 'stack_noise': 0.15}
TIME_FLAG = 3.0


def parse_log(text: str) -> Dict[str, object]:
    """What a ``-v`` run printed that the check compares: acceptance, target, light time, widths, time."""
    out: Dict[str, object] = {}
    m = re.search(r'Accepted:\s*(\d+)/(\d+)', text)
    if m:
        out['accepted'], out['total'] = int(m.group(1)), int(m.group(2))
    m = re.search(r'Target:\s*(.*?)\s*\[([^\]]+)\]', text)
    if m:
        out['target'], out['target_type'] = m.group(1), m.group(2)
    m = re.search(r'Integration time:\s*([\d.]+) minutes', text)
    if m:
        out['integration_min'] = float(m.group(1))
    f = [float(x) for x in re.findall(r'FWHM=([\d.]+)', text)]
    if f:
        out['frame_fwhm'] = round(statistics.median(f), 2)
    m = re.search(r'Total time:\s*(?:(\d+)m\s*)?([\d.]+)s', text)
    if m:
        out['seconds'] = round(int(m.group(1) or 0) * 60 + float(m.group(2)), 1)
    out['warnings'] = len(re.findall(r'WARNING', text))
    # why Gaia colour calibration did not run, when the log says (see colour_mismatch)
    m = re.search(r'Colour calibration: (no stack WCS|not enough well-measured Gaia stars) -- skipped'
                  r'|colour calibration failed: (.*)|(Offline mode)', text)
    if m:
        out['colour_note'] = (m.group(1) or (f"failed: {m.group(2).strip()}" if m.group(2) else 'offline')).strip()
    return out


def tile_noise(lum: np.ndarray, tile: int = 64) -> float:
    """Per-pixel noise: the MAD of the plane minus its 4 px blur, per 64 px tile, at the 20th percentile.

    A bright star's blurred halo lifts the plain MAD of a crowded frame several-fold; the cleanest
    tiles are the ones with sky only. An absolute number is only as good as the field is empty, but
    it is stable for one session, which is what a regression check needs."""
    from scipy import ndimage
    hp = lum - ndimage.gaussian_filter(lum, 4.0)
    h, w = (hp.shape[0] // tile) * tile, (hp.shape[1] // tile) * tile
    blocks = hp[:h, :w].reshape(h // tile, tile, w // tile, tile).transpose(0, 2, 1, 3).reshape(-1, tile * tile)
    sig = 1.4826 * np.median(np.abs(blocks - np.median(blocks, axis=1, keepdims=True)), axis=1)
    return float(np.percentile(sig, 20))


def stack_metrics(path: str) -> Dict[str, object]:
    """Median FWHM of the stack's unsaturated stars and its per-pixel noise (luminance), ADU."""
    from astropy.io import fits

    from src.debayer import luminance
    from src.merge import load_merge_stack
    from src.psf_field import bright_unsaturated, star_fwhm
    from src.quality import detect_stars_auto
    stack, _meta = load_merge_stack(path)
    lum = luminance(stack).astype(np.float32)
    sigma = tile_noise(lum)
    out: Dict[str, object] = {'stack_noise': round(sigma, 2)}
    hdr = fits.getheader(path)
    out['colour_calibrated'] = bool(hdr.get('COLCAL', False))
    if out['colour_calibrated']:
        out['colour_scales'] = [round(float(hdr.get(f'COLCAL_{c}', 1.0)), 4) for c in 'RGB']
    src = detect_stars_auto(lum, sigma, background=float(np.median(lum)))
    if src is not None and len(src) >= 10:
        idx = bright_unsaturated(lum, src, 300, sigma)
        w = [star_fwhm(lum, float(src['ycentroid'][i]), float(src['xcentroid'][i])) for i in idx]
        w = [x for x in w if x]
        if len(w) >= 8:
            out['stack_fwhm'] = round(float(np.median(w)), 2)
    return out


def colour_mismatch(base: Dict[str, object], cur: Dict[str, object]) -> Optional[str]:
    """Why ``stack_noise`` cannot be compared, or None.

    The noise is in ADU of the saved linear stack, which is Gaia colour-calibrated: when the
    calibration does not run (Gaia unreachable or short of stars on that run), the same stack
    reads ~15% noisier (x1.147 on M51). The weekly check failed on exactly that, twice on one
    commit (M51 90.38 <-> 105.26, comet 105.72 <-> 120.50). Baselines made before this was
    recorded hold no ``colour_calibrated``; they were stacked online and are taken as calibrated
    here (a run that does calibrate is compared with them as before)."""
    b, c = base.get('colour_calibrated'), cur.get('colour_calibrated')
    if c is None or b == c or (b is None and c):
        return None
    why = f" ({cur['colour_note']})" if not c and cur.get('colour_note') else ''
    state = 'did not run' if not c else 'ran (the baseline is uncalibrated)'
    return (f"colour calibration {state}{why}: stack_noise not compared "
            f"({base.get('stack_noise')} -> {cur.get('stack_noise')}; calibration changes it ~15%)")


def compare(base: Dict[str, object], cur: Dict[str, object]) -> List[str]:
    """Differences between a baseline entry and a current run (empty = within tolerance).
    A colour-calibration mismatch is not one of them (``colour_mismatch``): it skips the
    noise comparison and is reported as a note."""
    problems: List[str] = []
    skip = {'stack_noise'} if colour_mismatch(base, cur) else set()
    for key in ('target', 'target_type'):
        if base.get(key) != cur.get(key):
            problems.append(f"{key}: {base.get(key)!r} -> {cur.get(key)!r}")
    if base.get('accepted') is not None:
        if cur.get('accepted') is None or abs(int(cur['accepted']) - int(base['accepted'])) > TOL['accepted']:
            problems.append(f"frames accepted: {base['accepted']}/{base.get('total')} -> {cur.get('accepted')}/{cur.get('total')}")
    for key in ('integration_min', 'frame_fwhm', 'stack_fwhm', 'stack_noise'):
        b, c = base.get(key), cur.get(key)
        if b is None or key in skip:
            continue
        if c is None:
            problems.append(f'{key}: {b} -> missing')
        elif abs(float(c) - float(b)) > TOL[key] * abs(float(b)):
            problems.append(f'{key}: {b} -> {c} ({100 * (float(c) / float(b) - 1):+.0f}%)')
    return problems


def run_session(folder: str, out_dir: str, name: str) -> Dict[str, object]:
    out_fits = os.path.join(out_dir, name + '.fits')
    cmd = [sys.executable, os.path.join(ROOT, 'originstack.py'), '-d', folder, '-o', out_fits, '-v']
    if name.startswith('11') or 'comet' in name.lower():
        cmd.append('--comet-mode')
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    log = r.stdout + r.stderr
    with open(os.path.join(out_dir, name + '.log'), 'w', encoding='utf-8') as fh:
        fh.write(log)
    res = parse_log(log)
    res.setdefault('seconds', round(time.time() - t, 1))
    res['returncode'] = r.returncode
    if os.path.exists(out_fits):
        res.update(stack_metrics(out_fits))
    return res


def adopt(out_dir: str) -> Dict[str, Dict[str, object]]:
    """Baseline entries from existing ``NAME.log`` + ``NAME.fits`` pairs in ``out_dir``."""
    base = {}
    for fn in sorted(os.listdir(out_dir)):
        if not fn.endswith('.log') or fn.startswith('_'):
            continue
        name = fn[:-4]
        with open(os.path.join(out_dir, fn), encoding='utf-8', errors='replace') as fh:
            entry = parse_log(fh.read())
        fits_path = os.path.join(out_dir, name + '.fits')
        if os.path.exists(fits_path):
            entry.update(stack_metrics(fits_path))
        base[name] = entry
    return base


def _write(path: str, base: dict, note: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump({'note': note, 'sessions': base}, fh, indent=1, sort_keys=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('samples', nargs='?', help='folder holding the unzipped sample sessions')
    ap.add_argument('--only', help='comma-separated name prefixes (e.g. 01,11)')
    ap.add_argument('--out', default=os.path.join(ROOT, 'sample_check_out'), help='where the stacks and logs go')
    ap.add_argument('--baseline', default=BASELINE)
    ap.add_argument('--update', action='store_true', help='write the current values as the new baseline')
    ap.add_argument('--adopt', metavar='DIR', help='build the baseline from existing NAME.log/NAME.fits in DIR')
    a = ap.parse_args(argv)
    note = f"OriginStack {open(os.path.join(ROOT, 'VERSION')).read().strip()}; stacked with -v on the unzipped Zenodo sessions"

    if a.adopt:
        base = adopt(a.adopt)
        _write(a.baseline, base, note)
        print(f'baseline of {len(base)} sessions written to {a.baseline}')
        return 0
    if not a.samples:
        ap.error('SAMPLES_DIR is required')
    names = sorted(d for d in os.listdir(a.samples) if os.path.isdir(os.path.join(a.samples, d)))
    if a.only:
        pre = [p.strip() for p in a.only.split(',')]
        names = [n for n in names if any(n.startswith(p) for p in pre)]
    if not names:
        sys.exit('no sample sessions found')
    os.makedirs(a.out, exist_ok=True)
    try:
        with open(a.baseline, encoding='utf-8') as fh:
            base = json.load(fh).get('sessions', {})
    except OSError:
        base = {}
    cur: Dict[str, Dict[str, object]] = {}
    failed = 0
    for n in names:
        res = run_session(os.path.join(a.samples, n), a.out, n)
        cur[n] = res
        problems = [] if a.update else compare(base.get(n, {}), res)
        if res.get('returncode') != 0:
            problems.insert(0, f"run failed (exit {res.get('returncode')}); see {os.path.join(a.out, n + '.log')}")
        if n not in base and not a.update:
            problems.append('not in the baseline')
        ratio = (res.get('seconds') or 0) / base[n]['seconds'] if n in base and base[n].get('seconds') else None
        if ratio and ratio > TIME_FLAG:
            problems.append(f"slow: {res.get('seconds')} s against {base[n]['seconds']} s ({ratio:.1f}x)")
        note = None if a.update else colour_mismatch(base.get(n, {}), res)
        cal = res.get('colour_calibrated')
        if cal and res.get('colour_scales'):
            cal_text = 'colour R/G/B x' + '/'.join(f'{v:.3f}' for v in res['colour_scales'])
        elif cal is False:
            cal_text = 'colour not calibrated' + (f" ({res['colour_note']})" if res.get('colour_note') else '')
        else:
            cal_text = 'colour ?'
        print(f"{n}: {'OK' if not problems else 'DIFFERS'}  accepted {res.get('accepted')}/{res.get('total')}, "
              f"stack FWHM {res.get('stack_fwhm')}, noise {res.get('stack_noise')}, {cal_text}, {res.get('seconds')} s"
              + ''.join('\n    ' + p for p in problems)
              + (f"\n    note: {note}" if note else ''))
        failed += bool(problems)
    if a.update:
        merged = {**base, **cur}
        _write(a.baseline, merged, note)
        print(f'baseline updated ({len(cur)} sessions) -> {a.baseline}')
        return 0
    print(f'\n{len(names) - failed} of {len(names)} sessions match the baseline')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
