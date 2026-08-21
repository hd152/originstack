"""Tests for src/registration.py's find_extended_source_ellipse (--galaxy-mode)."""
from __future__ import annotations

import numpy as np
import pytest

from src.registration import find_extended_source_ellipse, parse_galaxy_center_override


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


def test_prefers_larger_faint_blob_over_smaller_brighter_peak():
    """A bright, compact foreground star must not win over a larger but
    fainter extended object just because it has a higher peak. Reproduces a
    real failure: on a rich star field with a faint galaxy, the old
    peak-pixel-owning-blob rule locked the exclusion ellipse onto a bright
    star nowhere near the actual galaxy, so --galaxy-mode's protection was
    applied to the wrong object entirely."""
    H, W = 600, 600
    # A bright, small, compact "star" -- higher peak, small footprint.
    star = _elongated_blob(H, W, 150.0, 150.0, sigma_major=8.0, sigma_minor=8.0,
                           theta_deg=0.0, amp=2000.0, bg=0.0)
    # A fainter, much larger "galaxy" -- lower peak, big footprint.
    galaxy = _elongated_blob(H, W, 420.0, 420.0, sigma_major=70.0, sigma_minor=50.0,
                             theta_deg=20.0, amp=40.0, bg=1000.0)
    lum = (star + galaxy).astype(np.float32)

    result = find_extended_source_ellipse(lum)
    assert result is not None
    fy, fx, a, b, _ = result
    # Must center on the galaxy (420, 420), not the star (150, 150).
    assert abs(fy - 420.0) < 20.0
    assert abs(fx - 420.0) < 20.0
    assert a > 40.0  # a fit centered on the compact star would be far smaller


def test_border_touching_blob_is_excluded():
    """A vignetting/glow gradient anchored at the frame edge must not win
    even if it's the largest blob -- reproduces a real failure: on the same
    dense-star-field target, the single largest thresholded blob was a
    corner brightness falloff touching pixel (0,0), not the galaxy. A
    deliberately-framed target is essentially never clipped by the sensor
    edge, so border-touching is a general artifact tell."""
    H, W = 400, 400
    # Large "vignetting" blob anchored at the top-left corner (sigma=45 --
    # real vignetting/glow is typically concentrated tightly at the corner
    # itself, not spread a quarter of the frame wide; the real frame this
    # reproduces had its corner artifact's weighted centroid at ~6% of the
    # frame dimension from the edge, comfortably inside this function's 8%
    # exclusion margin).
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    corner = 60.0 * np.exp(-((yy / 45.0) ** 2 + (xx / 45.0) ** 2))
    # Smaller, fully-contained "galaxy" well inside the frame.
    galaxy = _elongated_blob(H, W, 280.0, 280.0, sigma_major=30.0, sigma_minor=25.0,
                             theta_deg=0.0, amp=40.0, bg=0.0)
    lum = (corner + galaxy).astype(np.float32)

    result = find_extended_source_ellipse(lum)
    assert result is not None
    fy, fx, _, _, _ = result
    assert abs(fy - 280.0) < 20.0
    assert abs(fx - 280.0) < 20.0


class TestGalaxyCenterOverride:
    """parse_galaxy_center_override (--galaxy-center): bypasses detection
    entirely with a user-supplied pixel coordinate, for fields where
    auto-detection keeps finding the wrong object."""

    def test_parses_x_y_order(self):
        cy, cx, a, b, axes = parse_galaxy_center_override("1421,914")
        assert cx == 1421.0
        assert cy == 914.0

    def test_default_radius_is_200(self):
        _, _, a, b, _ = parse_galaxy_center_override("100,100")
        assert a == 200.0
        assert b == 200.0

    def test_radius_override_applied(self):
        _, _, a, b, _ = parse_galaxy_center_override("100,100", radius=75.0)
        assert a == 75.0
        assert b == 75.0

    def test_axes_are_identity_circle(self):
        _, _, _, _, axes = parse_galaxy_center_override("100,100")
        assert np.array_equal(axes, np.eye(2))

    def test_malformed_string_raises(self):
        with pytest.raises(ValueError):
            parse_galaxy_center_override("not-a-valid-pair")
