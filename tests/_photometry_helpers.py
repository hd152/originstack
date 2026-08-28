"""Shared synthetic-data helpers for the photometry test modules."""
from __future__ import annotations

import numpy as np
from astropy.io import fits


def make_wcs_header(H, W, scale_arcsec=2.0, crval=(180.0, 0.0)):
    """A minimal RA---TAN / DEC--TAN header centred on ``crval`` (deg)."""
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = W
    hdr["NAXIS2"] = H
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRPIX1"] = W / 2.0
    hdr["CRPIX2"] = H / 2.0
    hdr["CRVAL1"] = crval[0]
    hdr["CRVAL2"] = crval[1]
    s = scale_arcsec / 3600.0
    hdr["CD1_1"] = -s
    hdr["CD1_2"] = 0.0
    hdr["CD2_1"] = 0.0
    hdr["CD2_2"] = s
    return hdr


def add_gaussian(plane, x, y, total_flux, sigma=1.6):
    """Add a unit-integral Gaussian of the given total flux to ``plane``."""
    H, W = plane.shape
    r = int(np.ceil(4 * sigma))
    x0, x1 = max(0, int(x) - r), min(W, int(x) + r + 1)
    y0, y1 = max(0, int(y) - r), min(H, int(y) + r + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    g *= total_flux / (2 * np.pi * sigma ** 2)
    plane[y0:y1, x0:x1] += g


class FakeTable:
    """Minimal astropy-Table stand-in: ``t['col']`` -> ndarray, ``len(t)``."""

    def __init__(self, cols):
        self._cols = {k: np.asarray(v) for k, v in cols.items()}
        self.colnames = list(cols.keys())

    def __getitem__(self, key):
        return self._cols[key]

    def __len__(self):
        return len(next(iter(self._cols.values())))
