import os
import sys

# Ensure workspace package import works
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Never let a test reach for astropy's IERS tables over the network. Two tests
# touch IERS-dependent astropy (AltAz in test_sky_model, sidereal_time in
# test_observing_geometry); on a CI runner whose bundled table has gone stale
# astropy tries to refresh it and the suite hangs on a socket timeout rather
# than failing fast.
from src.utils import disable_astropy_network  # noqa: E402

disable_astropy_network()


import pytest  # noqa: E402


@pytest.fixture(autouse=True, scope='session')
def _no_network():
    """No test reaches a real server. Every pipeline test looked the target name
    up on SIMBAD; where the connection hangs that cost the TCP connect timeout
    (~21 s) per run, two-thirds of the whole suite's time. A request now fails at
    once with the error a dead network gives, which every caller already handles.
    Tests of the HTTP layer patch ``urllib.request.urlopen`` themselves, over this.
    Session-scoped: unittest classes stack in ``setUpClass``, before any
    function-scoped fixture."""
    import urllib.error
    import urllib.request

    def _refuse(req, *a, **k):
        url = getattr(req, 'full_url', req)
        raise urllib.error.URLError(f'network disabled in tests (would have contacted {url})')
    mp = pytest.MonkeyPatch()
    mp.setattr(urllib.request, 'urlopen', _refuse)
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _no_reachability_probe(monkeypatch):
    """net_query checks a server answers a TCP handshake before the first request;
    tests mock the HTTP layer and must not open real sockets. Tests of the check
    restore it from ``__wrapped__``."""
    from src import net_query
    real = getattr(net_query._require_reachable, '__wrapped__', net_query._require_reachable)

    def _skip(url):
        return None
    _skip.__wrapped__ = real
    monkeypatch.setattr(net_query, '_require_reachable', _skip)


@pytest.fixture(autouse=True)
def _no_sweep_of_the_real_temp_folder(monkeypatch):
    """cli.apply_temp_dir removes OriginStack leftovers from the temp folder once
    per process. Tests must never do that to the real one: a test run on a
    machine with an app run going deleted that run's per-session stacks."""
    import tempfile

    from src import cli
    monkeypatch.setattr(cli, '_SWEPT', set(cli._SWEPT) | {tempfile.gettempdir()})


@pytest.fixture(autouse=True)
def _no_star_index_downloads(tmp_path_factory, monkeypatch):
    """The built-in plate solver (and the default WCS refine after Phase 3) fetch
    Gaia index tiles over the network and cache them in the user's profile.
    Tests get an empty private index and no fetches; tests of the index itself
    (tests/test_local_solve.py) build their own tiles under a tmp root."""
    from src import local_solve
    monkeypatch.setenv('ORIGINSTACK_STAR_INDEX', str(tmp_path_factory.mktemp('star_index')))
    def _no_fetch(*a, **k):
        return None
    _no_fetch.__wrapped__ = getattr(local_solve.fetch_tile, '__wrapped__', local_solve.fetch_tile)
    monkeypatch.setattr(local_solve, 'fetch_tile', _no_fetch)


@pytest.fixture(autouse=True)
def _no_colour_calibration_queries(monkeypatch):
    """Colour calibration is on by default and queries Gaia DR3 for the stack's
    field. Pipeline tests get no calibration (and no network); tests of it
    restore the real function from ``__wrapped__``."""
    from src import color_calibrate
    real = getattr(color_calibrate.calibrate_linear_stack, '__wrapped__',
                   color_calibrate.calibrate_linear_stack)

    def _none(*a, **k):
        return None
    _none.__wrapped__ = real
    monkeypatch.setattr(color_calibrate, 'calibrate_linear_stack', _none)


@pytest.fixture(scope='module')
def tk_root():
    """A themed, withdrawn Tk root shared by a test module's desktop-app tests."""
    # One root for the module: creating and destroying a Tk root per test
    # intermittently crashed Tk inside tk.Tk() (Windows, Python 3.14).
    tk = pytest.importorskip('tkinter')
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('no display available for a real Tk root')
    root.withdraw()
    from src import desktop_app
    desktop_app._apply_theme(root)
    yield root
    root.destroy()
