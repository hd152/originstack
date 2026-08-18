"""Non-negative matrix factorization for star/nebula source separation
(--nmf-separate).

Treats the stacked image's per-pixel channel vector (R, G, B) as a
non-negative mixture of a small number of "sources", each with its own
fixed spectral signature (basis) and a spatially-varying non-negative
activation map -- e.g. a stellar-continuum source (present wherever there's
a star) and a nebular-emission source (present in extended diffuse
structure). Standard multiplicative-update NMF (Lee & Seung 2001), applied
directly to the (n_pixels, n_channels) data matrix -- the same category of
technique as hyperspectral unmixing (e.g. SCARLET's constrained matrix
factorization for multi-band astronomical source separation).

Offered alongside, not replacing, ``src/star_removal.py``'s inpainting
approach: NMF separates signal *into components* rather than discarding a
masked region, which suits a downstream use like continuum-subtraction
better than inpainting does (the star component IS the thing you want,
not something to remove and fill in). Genuinely more failure-prone than
inpainting though -- NMF is a non-convex optimisation with no guaranteed
global optimum, sensitive to initialization and component count; the
convergence-error trace this returns is there so a caller can actually
check the fit, not just trust it blindly.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def nmf_separate(img: np.ndarray, n_components: int = 2,
                 iterations: int = 200, seed: int = 0,
                 tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, list]:
    """Multiplicative-update NMF: factor an ``(H, W, C)`` non-negative
    image into ``n_components`` non-negative ``(H, W)`` activation maps and
    their ``(n_components, C)`` channel-basis vectors.

    Returns ``(activations, basis, errors)``:
      - ``activations``: ``(n_components, H, W)`` -- each component's
        spatial map.
      - ``basis``: ``(n_components, C)`` -- each component's per-channel
        spectral signature (each row normalised to sum to 1).
      - ``errors``: per-iteration Frobenius reconstruction error, for
        convergence inspection -- non-increasing by the standard
        multiplicative-update NMF guarantee (Lee & Seung 2001); a caller
        that sees it plateau far from zero has a poor fit, not a bug.
    """
    h, w, c = img.shape
    v = np.clip(img.astype(np.float64), 0.0, None).reshape(h * w, c).T  # (C, N)
    n_pixels = h * w

    rng = np.random.default_rng(seed)
    basis = rng.uniform(0.1, 1.0, (c, n_components))                  # W: (C, K)
    activations = rng.uniform(0.1, 1.0, (n_components, n_pixels))     # H: (K, N)

    eps = 1e-10
    errors = []
    for _ in range(iterations):
        wtv = basis.T @ v
        wtwh = basis.T @ basis @ activations + eps
        activations *= wtv / wtwh

        vht = v @ activations.T
        whht = basis @ activations @ activations.T + eps
        basis *= vht / whht

        recon = basis @ activations
        err = float(np.linalg.norm(v - recon, 'fro'))
        errors.append(err)
        if len(errors) > 1 and abs(errors[-2] - errors[-1]) < tol * max(errors[-2], 1e-12):
            break

    # Normalise each component's basis vector to sum 1 (a spectral SHAPE,
    # not an arbitrary scale) -- rescale its activation map inversely so
    # the product (the reconstruction) is unchanged.
    row_sums = basis.sum(axis=0, keepdims=True)
    row_sums_safe = np.where(row_sums > eps, row_sums, 1.0)
    basis_norm = basis / row_sums_safe
    activations_norm = activations * row_sums_safe.T

    activations_out = activations_norm.reshape(n_components, h, w).astype(np.float32)
    basis_out = basis_norm.T.astype(np.float32)
    return activations_out, basis_out, errors


def separate_star_nebula(img: np.ndarray, iterations: int = 200, seed: int = 0) -> dict:
    """2-component NMF specialised for star/nebula separation.

    After fitting, labels the component with the higher peak-to-mean
    activation ratio as "stellar" (stars are sparse, bright, and
    high-contrast against their surroundings) and the other as "nebula"
    (smoother, more spatially extended) -- a simple, inspectable heuristic,
    not a guarantee: see the returned ``ambiguous`` flag.

    Returns ``{'star': (H,W) float32, 'nebula': (H,W) float32, 'star_basis':
    (C,), 'nebula_basis': (C,), 'ambiguous': bool, 'errors': list}``. The
    two maps are single-channel activation strength, not RGB.
    """
    activations, basis, errors = nmf_separate(
        img, n_components=2, iterations=iterations, seed=seed)

    ratios = []
    for k in range(2):
        a = activations[k]
        mean_a = float(a.mean())
        peak_a = float(np.percentile(a, 99.5))
        ratios.append(peak_a / max(mean_a, 1e-12))

    star_idx = int(np.argmax(ratios))
    nebula_idx = 1 - star_idx
    ambiguous = abs(ratios[0] - ratios[1]) < 0.15 * max(ratios)

    return {
        'star': activations[star_idx],
        'nebula': activations[nebula_idx],
        'star_basis': basis[star_idx],
        'nebula_basis': basis[nebula_idx],
        'ambiguous': bool(ambiguous),
        'errors': errors,
    }


def run_nmf_separation_report(img: np.ndarray, output_path: str,
                              iterations: int = 200) -> None:
    """Entry point wired to ``--nmf-separate``: runs the 2-component split
    and writes ``<output>_star_component.fits``/``_nebula_component.fits``.
    Fails soft (prints a message) rather than raising, matching this
    project's other opt-in diagnostic/utility outputs.
    """
    from astropy.io import fits
    from src.utils import safe_print

    try:
        result = separate_star_nebula(img, iterations=iterations)
        star_path = _sidecar_path(output_path, '_star_component.fits')
        nebula_path = _sidecar_path(output_path, '_nebula_component.fits')
        fits.PrimaryHDU(data=result['star']).writeto(star_path, overwrite=True)
        fits.PrimaryHDU(data=result['nebula']).writeto(nebula_path, overwrite=True)
        note = " (ambiguous split -- inspect before trusting the star/nebula labels)" \
            if result['ambiguous'] else ""
        safe_print(f"  NMF source separation: wrote star/nebula components{note}")
    except Exception as e:
        safe_print(f"  NMF source separation failed ({e}) -- skipping")


def _sidecar_path(output_path: str, suffix: str) -> str:
    import os
    return os.path.splitext(output_path)[0] + suffix
