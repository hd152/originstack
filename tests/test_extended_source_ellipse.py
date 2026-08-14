"""Tests for src/registration.py's find_extended_source_ellipse (--galaxy-mode)."""
from __future__ import annotations

import numpy as np

from src.registration import find_extended_source_ellipse


def _elongated_blob(H, W, cy, cx, sigma_major, sigma_minor, theta_deg, amp=500.0, bg=1000.0):
    """An elongated 2D Gaussian, major axis rotated theta_deg from vertical (row axis)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dy, dx = yy - cy, xx - cx
    theta = np.radians(theta_deg)
    # Rotate into the blob's own frame.
    u = dy * np.cos(theta) + dx * np.sin(theta)      # along major axis
    v = -dy * np.sin(theta) + dx * np.cos(theta)     # along minor axis
    g = amp * np.exp(-(u ** 2) / (2 * sigma_major ** 2) - (v ** 2) / (2 * sigma_minor ** 2))
    return (bg + g).astype(np.float32)


def test_recovers_center_and_elongation():
    H, W = 400, 400
    cy, cx = 200.0, 220.0
    lum = _elongated_blob(H, W, cy, cx, sigma_major=40.0, sigma_minor=12.0, theta_deg=0.0)
    result = find_extended_source_ellipse(lum)
    assert result is not None
    fy, fx, a, b, axes = result
    assert abs(fy - cy) < 5.0
    assert abs(fx - cx) < 5.0
    # Elongated along the row (y) axis -> semi-major clearly bigger than minor.
    assert a > b * 1.5


def test_semi_axes_positive_and_ordered():
    H, W = 300, 300
    lum = _elongated_blob(H, W, 150, 150, sigma_major=25.0, sigma_minor=25.0, theta_deg=30.0)
    result = find_extended_source_ellipse(lum)
    assert result is not None
    _, _, a, b, _ = result
    assert a > 0 and b > 0
    assert a >= b


def test_max_axis_frac_caps_runaway_fit():
    H, W = 200, 200
    # A blob nearly filling the frame with a huge sigma.
    lum = _elongated_blob(H, W, 100, 100, sigma_major=300.0, sigma_minor=300.0, theta_deg=0.0)
    result = find_extended_source_ellipse(lum, max_axis_frac=0.3)
    assert result is not None
    _, _, a, _, _ = result
    assert a <= 0.3 * max(H, W) + 1e-6


def test_axes_are_orthonormal():
    H, W = 300, 300
    lum = _elongated_blob(H, W, 150, 150, sigma_major=35.0, sigma_minor=15.0, theta_deg=55.0)
    result = find_extended_source_ellipse(lum)
    assert result is not None
    _, _, _, _, axes = result
    # Columns are unit eigenvectors of a real symmetric matrix -> orthonormal.
    gram = axes.T @ axes
    assert np.allclose(gram, np.eye(2), atol=1e-6)
