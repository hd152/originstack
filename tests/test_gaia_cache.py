"""Local Gaia query cache (src/gaia_cache.py) against a fake TAP server.

The fake executes the ADQL ``gaia_cone_search`` builds (TOP n, CIRCLE, the
min-magnitude cut, ORDER BY G) on a synthetic catalogue, so every cached answer
can be compared with what the "server" would have returned.
"""
import math
import re

import numpy as np
import pytest
from astropy.table import Table

import src.gaia_cache as gc
import src.net_query as nq

COLS = ["source_id", "ra", "dec", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"]
NN = ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"]


@pytest.fixture
def server(monkeypatch, tmp_path):
    monkeypatch.setenv('ORIGINSTACK_STAR_INDEX', str(tmp_path))
    monkeypatch.delenv('ORIGINSTACK_NO_GAIA_CACHE', raising=False)
    rng = np.random.default_rng(0)
    n = 20000
    cat = dict(source_id=np.arange(n, dtype=float),
               ra=100.0 + rng.uniform(-1.5, 1.5, n), dec=20.0 + rng.uniform(-1.5, 1.5, n),
               phot_g_mean_mag=np.round(rng.uniform(8, 18, n), 6).astype(np.float32).astype(float))
    cat['phot_bp_mean_mag'] = cat['phot_g_mean_mag'] + 0.4
    cat['phot_rp_mean_mag'] = cat['phot_g_mean_mag'] - 0.5
    calls = []

    def tap(url, adql, timeout=60.0):
        calls.append(adql)
        top = int(re.search(r'TOP (\d+)', adql).group(1))
        cols = [c.strip() for c in re.search(r'TOP \d+ (.*) FROM', adql).group(1).split(',')]
        ra0, dec0, r = map(float, re.search(r"CIRCLE\('ICRS',([^,]+),([^,]+),([^)]+)\)", adql).groups())
        d = np.degrees(np.arccos(np.clip(
            np.sin(np.radians(cat['dec'])) * math.sin(math.radians(dec0))
            + np.cos(np.radians(cat['dec'])) * math.cos(math.radians(dec0))
            * np.cos(np.radians(cat['ra'] - ra0)), -1, 1)))
        sel = d <= r
        m = re.search(r'phot_g_mean_mag > ([0-9.eE+-]+)', adql)
        if m:
            sel &= cat['phot_g_mean_mag'].astype(np.float32).astype(float) > float(m.group(1))
        idx = np.flatnonzero(sel)
        idx = idx[np.argsort(cat['phot_g_mean_mag'][idx], kind='stable')][:top]
        return Table({c: cat[c][idx] for c in cols})
    monkeypatch.setattr(nq, 'tap_query', tap)
    return calls


def _q(ra, dec, r, n, mm=None, cols=COLS):
    return nq.gaia_cone_search(ra, dec, r, cols, max_rows=n, require_not_null=NN, min_mag=mm)


def _uncached(monkeypatch, *a, **k):
    monkeypatch.setenv('ORIGINSTACK_NO_GAIA_CACHE', '1')
    try:
        return _q(*a, **k)
    finally:
        monkeypatch.delenv('ORIGINSTACK_NO_GAIA_CACHE')


def _same(a, b):
    return a.colnames == b.colnames and all(np.array_equal(a[c], b[c]) for c in a.colnames)


def test_repeat_shifted_and_subcone_queries_are_served_locally_and_exactly(server, monkeypatch):
    first = _q(100.0, 20.0, 0.9, 1200)
    assert len(server) == 1
    assert _same(first, _uncached(monkeypatch, 100.0, 20.0, 0.9, 1200))
    n = len(server)
    for args in ((100.0, 20.0, 0.9, 1200), (100.02, 19.99, 0.9, 1200), (100.1, 20.1, 0.5, 300)):
        got = _q(*args)
        assert _same(got, _uncached(monkeypatch, *args))
    assert len(server) == n + 3              # only the three uncached reference queries


def test_a_cone_the_entry_does_not_cover_goes_to_the_server(server, monkeypatch):
    _q(100.0, 20.0, 0.9, 1200)
    before = len(server)
    got = _q(100.5, 20.0, 0.9, 1200)        # outside the padded cone
    assert len(server) == before + 1
    assert _same(got, _uncached(monkeypatch, 100.5, 20.0, 0.9, 1200))


def test_more_rows_than_a_truncated_entry_holds_go_to_the_server(server, monkeypatch):
    _q(100.0, 20.0, 0.9, 1200)
    before = len(server)
    got = _q(100.0, 20.0, 0.5, 3000)
    assert len(server) > before
    assert _same(got, _uncached(monkeypatch, 100.0, 20.0, 0.5, 3000))


def test_fainter_slice_boundary_matches_the_server(server, monkeypatch):
    first = _q(100.0, 20.0, 0.9, 1200)
    mm = float(np.max(first['phot_g_mean_mag']))
    got = _q(100.0, 20.0, 0.9, 3000, mm)
    assert _same(got, _uncached(monkeypatch, 100.0, 20.0, 0.9, 3000, mm))
    again = len(server)
    assert _same(_q(100.0, 20.0, 0.9, 3000, mm), got) and len(server) == again


def test_column_subset_and_other_orderings_bypass_correctly(server, monkeypatch):
    _q(100.0, 20.0, 0.9, 1200)
    n = len(server)
    sub = _q(100.0, 20.0, 0.5, 200, cols=["ra", "dec", "phot_bp_mean_mag"])
    assert len(server) == n and sub.colnames == ["ra", "dec", "phot_bp_mean_mag"]
    nq.gaia_cone_search(100.0, 20.0, 0.5, COLS, max_rows=10, require_not_null=NN, order_by=None)
    assert len(server) == n + 1             # unordered queries are never cached


def test_disabled_cache_writes_nothing(server, monkeypatch, tmp_path):
    monkeypatch.setenv('ORIGINSTACK_NO_GAIA_CACHE', '1')
    _q(100.0, 20.0, 0.9, 1200)
    assert not (tmp_path / 'gaia_cones').exists()
