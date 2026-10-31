"""tools/download_stats.py: snapshot, diff and sweep detection (no network)."""
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    'download_stats', Path(__file__).resolve().parents[1] / 'tools' / 'download_stats.py')
ds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ds)


def _rel(tag, **counts):
    return {'tag_name': tag, 'assets': [{'name': n, 'download_count': c} for n, c in counts.items()]}


def _releases(bump=None):
    bump = bump or {}
    tags = ['v9', 'v8', 'v7', 'v6', 'v5', 'v4', 'v3', 'v2', 'v1']
    return [_rel(t, **{'a.exe': 1 + bump.get(t, 0), 'a.exe.sha256': 1}) for t in tags]


def test_snapshot_keys_and_totals():
    snap = ds.make_snapshot(_releases(), now=100.0)
    assert snap['time'] == 100.0 and snap['order'][0] == 'v9'
    assert snap['assets']['v9/a.exe'] == 1
    assert ds.per_release(snap)['v3'] == 2


def test_diff_new_release_is_not_a_sweep():
    old = ds.make_snapshot(_releases(), now=0)
    new = ds.make_snapshot(_releases({'v9': 3}), now=3600)
    d = ds.diff(old, new)
    assert d['total'] == 3 and d['by_release'] == {'v9': 3}
    assert d['old_rising'] == 0 and not d['sweep']


def test_diff_flags_a_sweep_over_old_releases():
    old = ds.make_snapshot(_releases(), now=0)
    new = ds.make_snapshot(_releases({t: 1 for t in ('v6', 'v5', 'v4', 'v3', 'v2', 'v1')}), now=3600)
    d = ds.diff(old, new)
    assert d['old_rising'] == 6 and d['sweep']
    assert 'sweep' in ds.format_report(new, old)


def test_diff_counts_checksum_files():
    old = ds.make_snapshot(_releases(), now=0)
    rels = _releases()
    rels[0]['assets'][1]['download_count'] = 4
    d = ds.diff(old, ds.make_snapshot(rels, now=1))
    assert d['total'] == 3 and d['checksums'] == 3


def test_first_run_and_log_roundtrip(tmp_path):
    log = tmp_path / 'sub' / 'log.jsonl'
    assert ds.load_last(str(log)) is None
    snap = ds.make_snapshot(_releases(), now=5)
    assert 'first snapshot' in ds.format_report(snap, None)
    log.parent.mkdir()
    log.write_text(json.dumps(snap) + '\n', encoding='utf-8')
    assert ds.load_last(str(log))['time'] == 5
