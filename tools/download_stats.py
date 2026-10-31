#!/usr/bin/env python3
"""Record GitHub release download counts and show what changed since the last run.

GitHub keeps only a running total per release file, with no history. Run this now and then
(daily is plenty) and each run appends a snapshot to a log, then reports the change since the
previous one -- so a burst of downloads can be told apart from steady interest:

    python tools/download_stats.py                 # snapshot + report
    python tools/download_stats.py --no-save       # report against the last snapshot only
    python tools/download_stats.py --log my.jsonl  # keep the log somewhere else
    python tools/download_stats.py --repo owner/name

The log defaults to ``~/.originstack/download_stats.jsonl`` (outside the repo). Uses the ``gh``
CLI when it is installed (authenticated, no rate limit), else the public API.

Reading the report: real users download the newest release's installer, one platform each.
A sweep -- a scanner or mirror fetching every file of every release, checksums included --
shows up as a rise spread over many *old* releases and as ``.sha256`` downloads rising as
fast as the installers; the report flags both.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from typing import Dict, List, Optional

DEFAULT_REPO = 'hd152/originstack'
DEFAULT_LOG = os.path.join(os.path.expanduser('~'), '.originstack', 'download_stats.jsonl')
NEWEST = 3            # releases counted as "current"; a rise elsewhere is old-release traffic
SWEEP_RELEASES = 5    # this many old releases rising in one interval is flagged as a sweep


def fetch_releases(repo: str) -> List[dict]:
    """Every release of ``repo`` (newest first), with its assets' download counts."""
    path = f'repos/{repo}/releases'
    if shutil.which('gh'):
        out = subprocess.run(['gh', 'api', '--paginate', '--slurp', path],
                             capture_output=True, text=True, check=True).stdout
        pages = json.loads(out)
        return [r for page in pages for r in page]
    releases, page = [], 1
    while True:
        url = f'https://api.github.com/{path}?per_page=100&page={page}'
        req = urllib.request.Request(url, headers={'Accept': 'application/vnd.github+json',
                                                   'User-Agent': 'originstack-download-stats'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            chunk = json.load(resp)
        releases += chunk
        if len(chunk) < 100:
            return releases
        page += 1


def make_snapshot(releases: List[dict], now: Optional[float] = None) -> dict:
    """``{'time': epoch, 'order': [tags newest first], 'assets': {'tag/file': count}}``."""
    assets: Dict[str, int] = {}
    for rel in releases:
        for a in rel.get('assets') or []:
            assets[f"{rel['tag_name']}/{a['name']}"] = int(a.get('download_count', 0))
    return {'time': time.time() if now is None else now,
            'order': [r['tag_name'] for r in releases],
            'assets': assets}


def is_checksum(key: str) -> bool:
    return key.endswith('.sha256')


def per_release(snap: dict) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for key, n in snap['assets'].items():
        tag = key.split('/', 1)[0]
        out[tag] = out.get(tag, 0) + n
    return out


def diff(old: dict, new: dict) -> dict:
    """What changed from snapshot ``old`` to ``new``.

    Returns ``total``, ``checksums`` (rise in .sha256 files), ``by_release`` ({tag: rise},
    only those that rose), ``old_rising`` (how many releases outside the newest ``NEWEST``
    rose) and ``sweep`` (that count reached ``SWEEP_RELEASES``).
    """
    before, after = per_release(old), per_release(new)
    by_release = {t: n - before.get(t, 0) for t, n in after.items() if n - before.get(t, 0) > 0}
    total = sum(max(0, n - old['assets'].get(k, 0)) for k, n in new['assets'].items())
    checksums = sum(max(0, n - old['assets'].get(k, 0)) for k, n in new['assets'].items()
                    if is_checksum(k))
    current = set(new['order'][:NEWEST])
    old_rising = sum(1 for t in by_release if t not in current)
    return {'total': total, 'checksums': checksums, 'by_release': by_release,
            'old_rising': old_rising, 'sweep': old_rising >= SWEEP_RELEASES}


def load_last(log: str) -> Optional[dict]:
    try:
        with open(log, encoding='utf-8') as f:
            lines = [ln for ln in f if ln.strip()]
    except OSError:
        return None
    return json.loads(lines[-1]) if lines else None


def format_report(new: dict, old: Optional[dict]) -> str:
    rel = per_release(new)
    lines = [f"{sum(new['assets'].values())} downloads across {len(rel)} releases "
             f"({sum(n for k, n in new['assets'].items() if is_checksum(k))} are checksum files)"]
    lines.append('newest: ' + ', '.join(f'{t} {rel.get(t, 0)}' for t in new['order'][:NEWEST]))
    if old is None:
        lines.append('first snapshot saved; the next run reports the change')
        return '\n'.join(lines)
    d = diff(old, new)
    hours = (new['time'] - old['time']) / 3600
    lines.append(f"since {hours:.1f} h ago: +{d['total']} downloads "
                 f"(+{d['checksums']} checksum files)")
    for tag, n in sorted(d['by_release'].items(), key=lambda kv: -kv[1])[:12]:
        lines.append(f'  {tag}: +{n}')
    if d['sweep']:
        lines.append(f"looks like a sweep: {d['old_rising']} releases outside the newest "
                     f'{NEWEST} rose in this interval')
    elif d['total'] and not any(t in d['by_release'] for t in new['order'][:NEWEST]):
        lines.append('all of the rise is in old releases')
    return '\n'.join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--repo', default=DEFAULT_REPO)
    ap.add_argument('--log', default=DEFAULT_LOG, help='snapshot log (JSON lines)')
    ap.add_argument('--no-save', action='store_true', help='report only, do not append a snapshot')
    args = ap.parse_args(argv)

    snap = make_snapshot(fetch_releases(args.repo))
    old = load_last(args.log)
    sys.stdout.write(format_report(snap, old) + '\n')
    if not args.no_save:
        os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
        with open(args.log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(snap) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
