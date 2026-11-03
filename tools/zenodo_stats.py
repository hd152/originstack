"""Download and view counts of the published OriginStack sample sessions on Zenodo.

Reads the public record API (no token needed), prints a table with the change since the last
run and appends one row per record to a CSV history, so the trend can be followed.

    python tools/zenodo_stats.py                  # fetch, print, append to the history
    python tools/zenodo_stats.py --no-log         # just look
    python tools/zenodo_stats.py --log some.csv   # history file (default ~/originstack-zenodo-stats.csv)

Zenodo's counts are its own (unique_downloads de-duplicates repeat downloads by one visitor).
Record ids live in tools/data/zenodo_records.json.
"""
import argparse
import csv
import datetime
import json
import urllib.request
from pathlib import Path

RECORDS = Path(__file__).parent / 'data' / 'zenodo_records.json'
FIELDS = ('downloads', 'unique_downloads', 'views', 'unique_views')


def fetch(rec_id: int) -> dict:
    req = urllib.request.Request(f'https://zenodo.org/api/records/{rec_id}',
                                 headers={'User-Agent': 'originstack-zenodo-stats'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)['stats']


def last_rows(path: Path) -> dict:
    """Most recent logged row per record name."""
    if not path.exists():
        return {}
    with open(path, newline='', encoding='utf-8') as f:
        return {row['name']: row for row in csv.DictReader(f)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--log', type=Path, default=Path.home() / 'originstack-zenodo-stats.csv')
    ap.add_argument('--no-log', action='store_true', help='do not append to the history')
    a = ap.parse_args(argv)

    records = json.loads(RECORDS.read_text(encoding='utf-8'))
    prev = last_rows(a.log)
    now = datetime.datetime.now().isoformat(timespec='seconds')
    rows, totals = [], dict.fromkeys(FIELDS, 0)
    print(f"{'record':44s} {'downloads':>9s} {'unique':>7s} {'views':>7s}  since last run")
    for name, rec_id in records.items():
        s = fetch(rec_id)
        rows.append({'time': now, 'name': name, 'id': rec_id, **{k: s.get(k, 0) for k in FIELDS}})
        for k in FIELDS:
            totals[k] += s.get(k, 0)
        d = s.get('downloads', 0) - int(prev[name]['downloads']) if name in prev else None
        print(f"{name:44s} {s.get('downloads', 0):9d} {s.get('unique_downloads', 0):7d} "
              f"{s.get('views', 0):7d}  {'' if d is None else f'{d:+d}'}")
    print(f"{'total':44s} {totals['downloads']:9d} {totals['unique_downloads']:7d} {totals['views']:7d}")
    if not a.no_log:
        new = not a.log.exists()
        with open(a.log, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, ['time', 'name', 'id', *FIELDS])
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f'appended to {a.log}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
