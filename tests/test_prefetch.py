import argparse
import math
import threading
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


def test_start_runs_in_the_background_and_join_waits(monkeypatch):
    seen = {}
    gate = threading.Event()

    def fake_run(field, tiles, gaia, cancel):
        gate.wait(5)
        seen.update(field=field, tiles=tiles, gaia=gaia)

    monkeypatch.setattr(prefetch, '_run', fake_run)
    monkeypatch.setattr(net_query, 'is_offline', lambda: False)
    args = _args()
    th = prefetch.start(args, [])
    assert th is not None and th.is_alive() and not seen        # main thread was not blocked
    gate.set()
    prefetch.join(args)
    assert seen['tiles'] and seen['gaia'] and not th.is_alive()
    assert args._catalog_prefetch is None
    prefetch.join(args)                                         # a second join is harmless


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
    prefetch.join(args)                                         # no exception


def test_gaia_prefetch_asks_for_what_colour_calibration_asks_for(monkeypatch):
    """Same columns, row cap and not-null filter, or the cache entry would not answer."""
    from src import color_calibrate as cc
    calls = []
    monkeypatch.setattr(net_query, 'gaia_cone_search', lambda *a, **k: calls.append((a, k)) or None)
    monkeypatch.setattr(net_query, 'is_offline', lambda: False)
    prefetch._run((202.47, 47.2, 0.6), False, True, None)
    cc.query_gaia_stars({'CRVAL1': 202.47, 'CRVAL2': 47.2})
    (_, k1), (_, k2) = calls
    for key in ('columns', 'require_not_null'):
        assert k1[key] == k2[key], key
    assert k1['max_rows'] >= k2['max_rows']          # more rows, so a smaller cone is still answered
