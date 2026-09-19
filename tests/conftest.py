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
