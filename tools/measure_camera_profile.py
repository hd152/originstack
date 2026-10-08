#!/usr/bin/env python
"""Measure the per-camera constants src/camera_profile.py ships, from real sessions.

For each session directory (lights + the session's bias/dark/flat):

* **raw gain** -- src/camera_profile.py's two-point estimate on consecutive raw
  lights (the one a run verifies with), the pedestal from the session's bias; the
  sky level is recorded so the read + dark noise floor can be fitted across sessions
  (1/g_measured = 1/g + floor/sky; dev-notes/camera-profile.md).
* **CFA** -- what the session-constant CFA equalisation measures (G1/G2 gain and the
  2x2 green offsets, src/debayer.py ``cfa_frame_stats``) through the real Phase 1
  calibration, for both rcd and malvar, per frame, with the frame's sky level.

Writes one JSON object per session (``--out``, JSON lines), so the summary below can
be re-run without re-measuring. Not part of a stacking run.

Usage:
    python tools/measure_camera_profile.py --root C:/source/Astrophotography --out cam.jsonl
    python tools/measure_camera_profile.py --summary cam.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _session_dirs(root):
    out = []
    for dirpath, _dirs, files in os.walk(root):
        if any(f.startswith('Light') and f.endswith('.fits') for f in files):
            out.append(dirpath)
    return sorted(out)




def two_point_gain(lights, pattern, bias_path, n_pairs=6):
    """src/camera_profile.py's estimator (the one a run verifies with), the pedestal
    taken from the session's bias. Also returns the pedestal and the median sky above
    it -- a wrong pedestal shows up as gain correlating with sky level."""
    from src.camera_profile import measure_raw_gain
    from src.io_fits import load_frame
    ped = float(np.median(np.asarray(load_frame(bias_path)[0], np.float32)[::4, ::4]))
    g = measure_raw_gain(lights, pattern, ped, n_pairs=n_pairs)
    mid = np.asarray(load_frame(lights[len(lights) // 2])[0], np.float32)
    return ([float(v) for v in g] if g is not None else None), ped, float(np.median(mid[::4, ::4])) - ped



def cfa_samples(session, light_paths):
    """Per-frame CFA stats for rcd and malvar through the real Phase 1 calibration."""
    from src.cli import _build_masters, parse_args
    from src.frame_discovery import discover_frames
    from src.frame_processor import _process_single_frame
    from src.io_fits import load_frame
    args = parse_args(['-d', session, '-o', os.path.join(session, '_probe.fits')])
    frames = discover_frames(session)
    masters = _build_masters(frames, None, args)
    out = {}
    for method in ('rcd', 'malvar'):
        rows = []
        for p in light_paths:
            res = _process_single_frame(p, {}, masters, method, args.white_balance,
                                        session_bayer='RGGB', cfa_probe=True, allow_gpu=False)
            st = res.get('cfa_stats')
            if res.get('error') or not st:
                continue
            mosaic = np.asarray(load_frame(p)[0], np.float32)
            rows.append({'gain': st['gain'], 'grid': st['grid'],
                         'raw_median': float(np.median(mosaic[::4, ::4]))})
        out[method] = rows
    return out, masters


def measure_session(session, n_frames, with_cfa=True):
    from astropy.io import fits
    lights = sorted(glob.glob(os.path.join(session, 'Light*.fits')))
    h = fits.getheader(lights[0])
    pattern = str(h.get('BAYERPAT', 'RGGB')).strip().upper()
    pick = [lights[int((j + 0.5) * len(lights) / min(n_frames, len(lights)))]
            for j in range(min(n_frames, len(lights)))]
    rec = {'session': os.path.basename(session), 'path': session,
           'camera': h.get('CAMERA'), 'firmware': h.get('CREATOR'),
           'iso': h.get('ISOSPEED'), 'egain': h.get('EGAIN'), 'exptime': h.get('EXPTIME'),
           'temp': h.get('CCD-TEMP'), 'n_lights': len(lights), 'pattern': pattern}
    biases = sorted(glob.glob(os.path.join(session, 'bias*.fits')))
    if biases:
        g, ped, sky = two_point_gain(lights, pattern, biases[0])
        rec.update(gain=g, pedestal=ped, sky_above_pedestal=sky,
                   bias_iso=fits.getheader(biases[0]).get('ISOSPEED'))
    if not with_cfa:
        return rec
    try:
        rec['cfa'], _ = cfa_samples(session, pick)
    except Exception as e:      # a session without usable calibration still gives a gain
        rec['cfa_error'] = repr(e)
    return rec


def summarize(path):
    recs = [json.loads(line) for line in open(path, encoding='utf-8') if line.strip()]
    print(f"{len(recs)} sessions")
    print(f"{'session':42s} {'iso':>4s} {'fw':>9s} {'T':>5s} {'sky':>6s}  gain e-/ADU R/G/B")
    by_iso = {}
    for r in recs:
        g = r.get('gain') or [float('nan')] * 3
        fw = (r.get('firmware') or '').replace('Origin ', '')
        print(f"{r['session'][:42]:42s} {r['iso']:>4} {fw:>9s} {r['temp']:5.1f} "
              f"{r.get('sky_above_pedestal', float('nan')):6.0f}  " + " ".join(f"{v:.5f}" for v in g))
        by_iso.setdefault(r['iso'], []).append(r)
    for iso, rs in sorted(by_iso.items()):
        # a bias taken at another ISO is fine as a pedestal: ISO 200 / 500 pedestals
        # differ by 16 ADU against thousands of ADU of sky
        ok = [r for r in rs if r.get('gain') and np.all(np.isfinite(r['gain']))
              and r.get('sky_above_pedestal', 0) > 1500]
        print(f"\nISO {iso}: {len(rs)} sessions, {len(ok)} with a gain")
        if ok:
            g = np.array([r['gain'] for r in ok])
            gm = np.median(g, axis=1)
            sky = np.array([r['sky_above_pedestal'] for r in ok])
            print("  gain median " + " ".join(f"{v:.5f}" for v in np.median(g, 0))
                  + f"  all channels {np.median(g):.5f}; session scatter {gm.std() / gm.mean():.3f}, "
                  + f"channel spread {np.median(g.std(1) / g.mean(1)):.3f}")
            if len(ok) >= 3 and np.ptp(sky) > 0:
                # read + dark noise left in the variance: 1/g_meas = 1/g + floor / sky
                A = np.vstack([np.ones_like(sky), 1.0 / sky]).T
                fit = [np.linalg.lstsq(A, 1.0 / g[:, c], rcond=None)[0] for c in range(3)]
                print("  floor-corrected gain " + " ".join(f"{1 / a:.5f}" for a, _ in fit)
                      + "   floor ADU^2 " + " ".join(f"{v:.0f}" for _, v in fit))
                print(f"  corr(gain, sky above pedestal) {np.corrcoef(gm, sky)[0, 1]:+.2f}, "
                      f"(gain, temperature) {np.corrcoef(gm, [r['temp'] for r in ok])[0, 1]:+.2f}")
        for method in ('rcd', 'malvar'):
            gains, grids, sky = [], [], []
            for r in rs:
                for s in (r.get('cfa') or {}).get(method, []):
                    gains.append(s['gain'])
                    grids.append(s['grid'])
                    sky.append(s['raw_median'])
            if not gains:
                continue
            gains, grids, sky = np.array(gains), np.array(grids), np.array(sky)
            print(f"  CFA {method}: {len(gains)} frames, G2 gain median {np.median(gains):.5f} "
                  f"std {gains.std():.5f}; grid median "
                  + " ".join(f"{v:+.2f}" for v in np.median(grids, 0))
                  + "  std " + " ".join(f"{v:.2f}" for v in grids.std(0)))
            for c in range(4):
                if np.ptp(sky) > 0:
                    sl = np.polyfit(sky, grids[:, c], 1)[0]
                    print(f"    grid[{c}] vs raw sky level: slope {sl * 1000:+.3f} ADU per 1000 ADU, "
                          f"corr {np.corrcoef(sky, grids[:, c])[0, 1]:+.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root')
    ap.add_argument('--sessions', nargs='*')
    ap.add_argument('--out', default='camera_profile_measurements.jsonl')
    ap.add_argument('--frames', type=int, default=8)
    ap.add_argument('--no-cfa', action='store_true', help='gain only')
    ap.add_argument('--summary', help='summarise an existing JSON-lines file and exit')
    a = ap.parse_args()
    if a.summary:
        summarize(a.summary)
        return
    sessions = list(a.sessions or []) + (_session_dirs(a.root) if a.root else [])
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(line)['path'] for line in open(a.out, encoding='utf-8') if line.strip()}
    from src.utils import safe_print
    for s in sessions:
        if s in done:
            continue
        safe_print(f"measuring {s}")
        try:
            rec = measure_session(s, a.frames, with_cfa=not a.no_cfa)
        except Exception as e:
            safe_print(f"  failed: {e!r}")
            continue
        with open(a.out, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec) + '\n')
    summarize(a.out)


if __name__ == '__main__':
    main()
