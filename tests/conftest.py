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
