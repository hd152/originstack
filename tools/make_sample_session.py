#!/usr/bin/env python3
"""Copy an Origin session folder into a shareable sample, with the observer's location removed.

    python tools/make_sample_session.py SESSION_DIR OUT_DIR [--max-lights 50] [--select even|best]

Copies the lights (``--select even``, the default: evenly spread when the session has more than
``--max-lights``; ``--select best``: the best-scoring light of each of that many time bins) and the
calibration frames, then scrubs the copies:

* FITS headers lose the site (LATITUDE, LONGITUD, ALTITUDE, SITE*, OBSGEO-*, GEO*) and the
  airmass (it follows from site, time and target, so it would give the site away);
* ``info.json`` loses ``gps`` and the session ``uuid``;
* the camera's serial (``CAMERA = Origin178-<unit>``, ``cameraID`` in the dark's json) is
  replaced by a fixed one -- the model stays, so the camera profile still applies;
* the Origin's own stack (``FinalStackedMaster.tiff``) and any other file are left out.

Pixel data is copied untouched. It then re-reads everything it wrote and fails if a scrubbed
field or the original coordinates remain, and checks the pixels are byte-identical.

What stays: the capture date and time, the ``TIMEZONE`` offset, the target's RA/Dec and the
mount orientation. Those say when and at what the telescope pointed, and the offset names a
broad region (-0700 is the US Pacific coast in summer), not a place.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from typing import Iterable, List, Optional

import numpy as np
from astropy.io import fits

SCRUB_EXACT = {'LATITUDE', 'LONGITUD', 'LONGITUDE', 'ALTITUDE', 'AIRMASS', 'ELEVATIO', 'ELEVATION',
               'LAT-OBS', 'LONG-OBS', 'ALT-OBS', 'GEOLAT', 'GEOLON', 'GEOELEV'}
SCRUB_PREFIX = ('SITE', 'OBSGEO', 'GPS')
FAKE_UNIT = '0' * 17
_CAMERA_UNIT = re.compile(r'(?<=-)[0-9a-fA-F]{8,}')
_FITS_EXT = ('.fits', '.fit', '.fts')


def is_site_key(key: str) -> bool:
    k = key.upper()
    return k in SCRUB_EXACT or k.startswith(SCRUB_PREFIX)


def scrub_header(header) -> List[str]:
    """Remove location cards and anonymise ``CAMERA`` in place; return the keys removed."""
    gone = sorted({k for k in header if k and is_site_key(k)})
    for k in gone:
        while k in header:
            del header[k]
    cam = header.get('CAMERA')
    if cam:
        header['CAMERA'] = _CAMERA_UNIT.sub(FAKE_UNIT, str(cam))
    return gone


def scrub_json(d: dict) -> dict:
    """Drop gps/uuid from a session ``info.json`` or a calibration json; anonymise camera ids."""
    d = dict(d)
    for inner in (d, d.get('StackedInfo') if isinstance(d.get('StackedInfo'), dict) else None):
        if inner is None:
            continue
        for k in ('gps', 'uuid', 'location', 'latitude', 'longitude', 'altitude'):
            inner.pop(k, None)
        if isinstance(inner.get('cameraID'), str):
            inner['cameraID'] = _CAMERA_UNIT.sub(FAKE_UNIT, inner['cameraID'])
    return d


def pick_evenly(names: List[str], n: int) -> List[str]:
    """``n`` names spread evenly over ``names`` (all of them when there are no more than ``n``)."""
    if len(names) <= n:
        return list(names)
    return [names[round(i * (len(names) - 1) / (n - 1))] for i in range(n)] if n > 1 else names[:1]


def pick_best_spread(names: List[str], scores: dict, n: int) -> List[str]:
    """``n`` names, one per consecutive time bin: the best-scoring light of each bin.

    ``names`` are in capture order. Splitting them into ``n`` bins and taking the best of each keeps
    the session's spread over the night (the field rotates, the sky changes) while leaving out the
    soft, cloudy or trailed frames an even spread would include. ``scores`` maps a name to its
    quality score (higher is better); a name without one (or NaN) loses to any scored name. All the
    names when there are no more than ``n``."""
    if len(names) <= n:
        return list(names)
    out: List[str] = []
    for chunk in np.array_split(np.arange(len(names)), n):
        def key(i):
            v = scores.get(names[i])
            return float('-inf') if v is None or v != v else float(v)
        out.append(names[max(chunk, key=key)])
    return out


def score_lights(src: str, names: List[str], workers: int = 6) -> dict:
    """Quality score of every named light, with the same metric ``--quality-sweep`` uses
    (``quality_sweep._score_one``: half-resolution luminance proxy, no calibration needed)."""
    from concurrent.futures import ProcessPoolExecutor

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.quality_sweep import _score_one
    paths = [os.path.join(src, n) for n in names]
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n, (_p, m, _err) in zip(names, ex.map(_score_one, paths, chunksize=4)):
            if m and m.get('score') is not None:
                out[n] = float(m['score'])
    return out


def _data_digest(path: str) -> str:
    with fits.open(path, do_not_scale_image_data=True, memmap=False) as hdul:
        return hashlib.sha256(hdul[0].data.tobytes()).hexdigest()


def _is_light(name: str) -> bool:
    return name.lower().startswith('light') and name.lower().endswith(_FITS_EXT)


def build(src: str, dst: str, max_lights: int, select: str = 'even') -> dict:
    os.makedirs(dst, exist_ok=True)
    names = sorted(os.listdir(src))
    all_lights = [n for n in names if _is_light(n)]
    if select == 'best' and len(all_lights) > max_lights:
        lights = pick_best_spread(all_lights, score_lights(src, all_lights), max_lights)
    else:
        lights = pick_evenly(all_lights, max_lights)
    others = [n for n in names if not _is_light(n)
              and (n.lower().endswith(_FITS_EXT) or n.lower().endswith('.json'))]
    report = {'lights': len(lights), 'fits': 0, 'removed': set(), 'digests': {}}
    for n in lights + others:
        s, d = os.path.join(src, n), os.path.join(dst, n)
        if n.lower().endswith('.json'):
            with open(s, encoding='utf-8') as f:
                data = scrub_json(json.load(f))
            with open(d, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            continue
        shutil.copyfile(s, d)
        report['digests'][n] = _data_digest(s)
        with fits.open(d, mode='update', do_not_scale_image_data=True) as hdul:
            report['removed'] |= set(scrub_header(hdul[0].header))
        report['fits'] += 1
    return report


def _original_coordinates(src: str) -> List[str]:
    """Latitude/longitude strings of the source session, to search the copy for."""
    out: List[str] = []
    for n in sorted(os.listdir(src)):
        if _is_light(n):
            h = fits.getheader(os.path.join(src, n))
            for k in ('LATITUDE', 'LONGITUD'):
                if h.get(k) is not None:
                    out.append(f"{abs(float(h[k])):.3f}")
            break
    return out


def verify(src: str, dst: str, report: dict) -> List[str]:
    """Problems found in the written sample (empty when it is clean)."""
    problems: List[str] = []
    coords = _original_coordinates(src)
    for n in sorted(os.listdir(dst)):
        p = os.path.join(dst, n)
        if n.lower().endswith(_FITS_EXT):
            h = fits.getheader(p)
            bad = [k for k in h if k and is_site_key(k)]
            if bad:
                problems.append(f'{n}: header still has {bad}')
            cam = h.get('CAMERA')
            if cam and _CAMERA_UNIT.search(str(cam)) and FAKE_UNIT not in str(cam):
                problems.append(f'{n}: CAMERA unit not anonymised ({cam})')
            if _data_digest(p) != report['digests'][n]:
                problems.append(f'{n}: pixel data changed')
        with open(p, 'rb') as f:
            blob = f.read()
        for c in coords:
            if c.encode() in blob:
                problems.append(f'{n}: contains {c}')
    return problems


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('session')
    ap.add_argument('out')
    ap.add_argument('--max-lights', type=int, default=50)
    ap.add_argument('--select', choices=('even', 'best'), default='even',
                    help="even: lights spread evenly over the session; best: the best-scoring light of each\n"
                         "of --max-lights consecutive time bins (same metric as --quality-sweep)")
    a = ap.parse_args(argv)
    if os.path.exists(a.out) and os.listdir(a.out):
        sys.exit(f'{a.out} is not empty')
    rep = build(a.session, a.out, a.max_lights, a.select)
    problems = verify(a.session, a.out, rep)
    sys.stdout.write(f"{rep['lights']} lights, {rep['fits']} FITS files; removed header cards: "
                     f"{', '.join(sorted(rep['removed'])) or 'none'}\n")
    if problems:
        sys.stdout.write('NOT CLEAN:\n' + '\n'.join('  ' + p for p in problems) + '\n')
        return 1
    sys.stdout.write('verified: no location left, camera unit anonymised, pixels identical\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
