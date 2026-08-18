"""Tests for NMF-based star/nebula source separation (--nmf-separate,
src/source_separation.py).
"""
from __future__ import annotations

import numpy as np

from src.source_separation import nmf_separate, separate_star_nebula


def _synthetic_mixture(h=48, w=48, seed=0):
    """Two non-negative sources with distinct spectral bases: a sparse,
    high-contrast 'star' activation (a few bright points) with a
    blue-white basis, and a smooth, extended 'nebula' activation (broad
    Gaussian) with a red-dominant basis -- mixed additively per-channel,
    the exact generative model NMF assumes.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    star_activation = np.zeros((h, w))
    for _ in range(5):
        cy, cx = rng.uniform(8, h - 8), rng.uniform(8, w - 8)
        star_activation += 50.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.5 ** 2))

    nebula_activation = 10.0 * np.exp(-((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * 15.0 ** 2))

    star_basis = np.array([0.9, 0.9, 1.0])    # blue-white
    nebula_basis = np.array([1.0, 0.3, 0.2])  # red-dominant

    img = (star_activation[..., None] * star_basis
           + nebula_activation[..., None] * nebula_basis)
    img += rng.normal(0, 0.05, img.shape)
    return np.clip(img, 0, None).astype(np.float32), star_activation, nebula_activation


class TestNmfSeparate:

    def test_output_shapes(self):
        img, _, _ = _synthetic_mixture()
        activations, basis, errors = nmf_separate(img, n_components=2, iterations=50)
        assert activations.shape == (2, 48, 48)
        assert basis.shape == (2, 3)
        assert len(errors) > 0

    def test_nonnegative_output(self):
        img, _, _ = _synthetic_mixture(seed=1)
        activations, basis, _ = nmf_separate(img, n_components=2, iterations=50)
        assert np.all(activations >= 0.0)
        assert np.all(basis >= 0.0)

    def test_basis_rows_sum_to_one(self):
        img, _, _ = _synthetic_mixture(seed=2)
        _, basis, _ = nmf_separate(img, n_components=2, iterations=50)
        np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1e-5)

    def test_reconstruction_error_is_nonincreasing(self):
        img, _, _ = _synthetic_mixture(seed=3)
        _, _, errors = nmf_separate(img, n_components=2, iterations=80)
        # Multiplicative-update NMF's standard guarantee.
        diffs = np.diff(errors)
        assert np.all(diffs <= 1e-6)  # small tolerance for float noise

    def test_error_decreases_substantially_from_random_init(self):
        img, _, _ = _synthetic_mixture(seed=4)
        _, _, errors = nmf_separate(img, n_components=2, iterations=100)
        assert errors[-1] < errors[0] * 0.5

    def test_recovers_components_up_to_permutation(self):
        # NMF's label ordering is arbitrary -- check that SOME assignment
        # of the two recovered activation maps correlates well with the
        # two true generative activation maps.
        img, true_star, true_nebula = _synthetic_mixture(seed=5)
        activations, _, _ = nmf_separate(img, n_components=2, iterations=150, seed=1)

        def corr(a, b):
            a, b = a.ravel(), b.ravel()
            return float(np.corrcoef(a, b)[0, 1])

        c00 = corr(activations[0], true_star)
        c01 = corr(activations[0], true_nebula)
        c10 = corr(activations[1], true_star)
        c11 = corr(activations[1], true_nebula)
        best_matching = max(c00 + c11, c01 + c10)
        assert best_matching > 1.4  # both matched components correlate strongly


class TestSeparateStarNebula:

    def test_labels_sparse_component_as_star(self):
        img, true_star, true_nebula = _synthetic_mixture(seed=6)
        result = separate_star_nebula(img, iterations=150, seed=1)

        def corr(a, b):
            a, b = a.ravel(), b.ravel()
            return float(np.corrcoef(a, b)[0, 1])

        assert corr(result['star'], true_star) > corr(result['star'], true_nebula)
        assert corr(result['nebula'], true_nebula) > corr(result['nebula'], true_star)

    def test_returns_expected_keys(self):
        img, _, _ = _synthetic_mixture(seed=7)
        result = separate_star_nebula(img, iterations=30)
        for key in ('star', 'nebula', 'star_basis', 'nebula_basis', 'ambiguous', 'errors'):
            assert key in result
        assert result['star'].shape == img.shape[:2]
        assert result['nebula'].shape == img.shape[:2]
