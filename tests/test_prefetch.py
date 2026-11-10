import argparse
import math
import threading
import time
from types import SimpleNamespace

from src import net_query, prefetch


def _si(ra=202.47, dec=47.2, fov=(1.0, 0.7)):
    return SimpleNamespace(has_wcs=True, ra_rad=math.radians(ra), dec_rad=math.radians(dec),
                           fov_x_rad=math.radians(fov[0]), fov_y_rad=math.radians(fov[1]))


def _args(**kw):
    ns = argparse.Namespace(offline=False, wcs_refine=True, color_calibrate=True, _session_info=_si())
    ns.__dict__.update(kw)
    return ns


def test_field_from_session_info():
    ra, dec, radius = prefetch.field_from_session(_args(), [])
    assert abs(ra - 202.47) < 1e-9 and abs(dec - 47.2) < 1e-9
    assert abs(radius - 0.5 * math.hypot(1.0, 0.7)) < 1e-9


def test_field_unknown_without_session_or_pointing():
    class F:
        header = {}
    assert prefetch.field_from_session(_args(_session_info=None), [F()]) is None


def test_start_runs_in_the_background_and_join_waits_for_the_tiles(monkeypatch):
    seen = {}
    gate = threading.Event()

    def fake_run(field, tiles, gaia, cancel, tiles_done=None, method="solar"):
        gate.wait(5)
        seen.update(field=field, tiles=tiles, gaia=gaia)
        tiles_done.set()

    monkeypatch.setattr(prefetch, '_run', fake_run)
    monkeypatch.setattr(net_query, 'is_offline', lambda: False)
    args = _args()
    th = prefetch.start(args, [])
    assert th is not None and th.is_alive() and not seen        # main thread was not blocked
    gate.set()
    prefetch.join(args)
    assert seen['tiles'] and seen['gaia']
    assert args._catalog_prefetch is None and args._catalog_prefetch_tiles is None
    prefetch.join(args)                                         # a second join is harmless
    th.join(5)


def test_join_does_not_wait_for_the_cone_stage(monkeypatch):
    """The tile stage ends the wait; the cone query may still be running (Phase 4 needs it later)."""
    cone_gate = threading.Event()

    def fake_run(field, tiles, gaia, cancel, tiles_done=None, method="solar"):
        tiles_done.set()
        cone_gate.wait(10)

    monkeypatch.setattr(prefetch, '_run', fake_run)
    monkeypatch.setattr(net_query, 'is_offline', lambda: False)
    args = _args()
    th = prefetch.start(args, [])
    t0 = time.time()
    prefetch.join(args, timeout=5)
    assert time.time() - t0 < 2 and th.is_alive()
    cone_gate.set()
    th.join(5)


def test_join_gives_up_on_a_slow_tile_fetch(monkeypatch):
    """A slow archive must not hold the run up: after the timeout the main thread goes on."""
    stuck = threading.Event()

    def fake_run(field, tiles, gaia, cancel, tiles_done=None, method="solar"):
        stuck.wait(10)                                           # never sets tiles_done in time
        tiles_done.set()

    monkeypatch.setattr(prefetch, '_run', fake_run)
    monkeypatch.setattr(net_query, 'is_offline', lambda: False)
    args = _args()
    th = prefetch.start(args, [])
    t0 = time.time()
    prefetch.join(args, timeout=0.3)
    assert 0.2 < time.time() - t0 < 2 and th.is_alive()
    assert args._catalog_prefetch is None
    stuck.set()
    th.join(5)


def test_nothing_to_fetch_offline_or_without_catalogue_steps(monkeypatch):
    monkeypatch.setattr(net_query, 'is_offline', lambda: False)
    assert prefetch.start(_args(offline=True), []) is None
    assert prefetch.start(_args(wcs_refine=False, color_calibrate=False), []) is None


def test_prefetch_failure_is_swallowed(monkeypatch):
    def boom(*a):
        raise RuntimeError("network down")

    monkeypatch.setattr(prefetch, '_run', boom)
    monkeypatch.setattr(net_query, 'is_offline', lambda: False)
    args = _args()
    prefetch.start(args, [])
    prefetch.join(args)                                         # no exception: the tile event is set on failure too


def test_gaia_prefetch_asks_for_what_the_colour_calibration_asks_for(monkeypatch):
    """Same columns and not-null filter, and at least as many rows, or the cache could not answer.
    The default method (``solar``) goes through ``photometry.match_gaia_field``, the other two
    through ``color_calibrate.query_gaia_stars``."""
    import inspect

    from src import color_calibrate as cc
    from src import photometry
    calls = []
    monkeypatch.setattr(net_query, 'gaia_cone_search', lambda *a, **k: calls.append((a, k)) or None)
    monkeypatch.setattr(net_query, 'is_offline', lambda: False)

    prefetch._run((202.47, 47.2, 0.6), False, True, None)                 # solar (the default)
    (_, k), = calls
    assert k['columns'] == list(photometry.GAIA_FIELD_COLUMNS)
    assert k['require_not_null'] == list(photometry.GAIA_FIELD_NOT_NULL)
    assert k['max_rows'] >= photometry.GAIA_FIELD_ROWS
    # match_gaia_field really sends those (so the two cannot drift apart)
    src = inspect.getsource(photometry.match_gaia_field)
    assert 'list(GAIA_FIELD_COLUMNS)' in src and 'list(GAIA_FIELD_NOT_NULL)' in src
    assert photometry.match_gaia_field.__kwdefaults__['max_rows'] == photometry.GAIA_FIELD_ROWS

    calls.clear()
    prefetch._run((202.47, 47.2, 0.6), False, True, None, None, 'colorindex')
    (_, k1), = calls
    cc.query_gaia_stars({'CRVAL1': 202.47, 'CRVAL2': 47.2})
    (_, k2) = calls[-1]
    for key in ('columns', 'require_not_null'):
        assert k1[key] == k2[key], key
    assert k1['max_rows'] >= k2['max_rows']


def test_concurrent_tile_fetches_write_to_different_temp_files(monkeypatch, tmp_path):
    """The prefetch thread and the main thread can fetch the same tile at once."""
    import numpy as np

    from src import local_solve
    table = {'ra': np.array([1.0]), 'dec': np.array([2.0]), 'phot_g_mean_mag': np.array([9.0])}

    class Tab(dict):
        def __len__(self):
            return 1
    monkeypatch.setattr(net_query, '_require_online', lambda *a, **k: None)
    monkeypatch.setattr(net_query, 'tap_query', lambda *a, **k: Tab(table))
    seen = []
    real_save = np.save
    both_saving = threading.Barrier(2, timeout=10)     # the two fetches really overlap (a finished thread's id can be reused)

    def save(path, arr):
        seen.append(str(path))
        both_saving.wait()
        return real_save(path, arr)

    monkeypatch.setattr(np, 'save', save)
    real_fetch = local_solve.fetch_tile.__wrapped__          # tests/conftest.py stubs the module attribute
    errors = []

    def fetch():
        try:
            real_fetch(27, 27, root=str(tmp_path))
        except BaseException as exc:            # noqa: BLE001 -- shown by the assert below
            errors.append(repr(exc))

    ts = [threading.Thread(target=fetch) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    assert not errors, errors
    assert len(seen) == 2 and seen[0] != seen[1]
    assert local_solve.load_tile(27, 27, root=str(tmp_path)) is not None
