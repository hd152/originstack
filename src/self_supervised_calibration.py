"""Self-supervised denoiser parameter calibration (--denoise-strength-calibrate).

Noise2Self (Batson & Royer 2019)'s actual contribution isn't a network
architecture -- it's a way to select a denoiser's *parameters* using only
the noisy data itself, no ground truth and no paired clean/noisy training
set. It applies to any denoiser, not just neural ones, which is what makes
this tractable here with no new dependency: this codebase already has 8
denoisers (wavelet, MMT, ACDNR, bilateral, NLM, aniso, curvelet, BM3D),
each with a hand-tuned default strength.

How it works: pick a small random subset J of pixels, replace each with an
interpolated value from its immediate neighbours (so the denoiser never
sees the true value there), run the denoiser on that masked image, and
compare its output *at J* against the true (un-masked) pixel values. This
self-supervised loss is, in expectation, an unbiased estimate of the
denoiser's true MSE against the (unknown) clean image, up to an additive
constant that doesn't depend on the denoiser's parameters -- so minimising
it over a parameter sweep approximately minimises true denoising error,
without ever seeing ground truth. Verified here against synthetic ground
truth (a known clean image + known noise, so the *actual* optimal
parameter can be computed directly for comparison) before being trusted
for the real wavelet-strength integration -- see
tests/test_self_supervised_calibration.py.
"""
from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np


def build_masked_image(img: np.ndarray, mask_frac: float = 0.02,
                       seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pick a random subset of interior pixels, replace each with the mean
    of its 4-connected neighbours, and return the masked image plus enough
    to score a denoiser's prediction there afterward.

    Returns ``(masked_img, ys, xs, true_vals)``: ``true_vals`` is
    ``img[ys, xs]`` (shape ``(n_mask, C)`` or ``(n_mask,)`` for a 2-D
    input) -- the values the masked positions actually held, which a
    denoiser run on ``masked_img`` never gets to see directly.
    """
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    n_mask = max(1, int(h * w * mask_frac))
    ys = rng.integers(1, h - 1, n_mask)
    xs = rng.integers(1, w - 1, n_mask)

    masked = img.copy()
    neighbor_avg = 0.25 * (img[ys - 1, xs] + img[ys + 1, xs]
                           + img[ys, xs - 1] + img[ys, xs + 1])
    true_vals = img[ys, xs].copy()
    masked[ys, xs] = neighbor_avg
    return masked, ys, xs, true_vals


def calibrate_denoiser_param(
    img: np.ndarray,
    denoise_fn: Callable[[np.ndarray, float], np.ndarray],
    param_grid: Sequence[float],
    mask_frac: float = 0.02,
    seed: int = 0,
) -> Tuple[float, np.ndarray]:
    """Self-supervised parameter sweep: for each candidate value in
    ``param_grid``, run ``denoise_fn(masked_img, value)`` and score its
    prediction at the masked-out pixels against their true values. Returns
    ``(best_value, losses)`` -- the parameter with the lowest self-
    supervised loss, and the full swept loss curve for inspection.

    ``denoise_fn`` must accept the same shape it's given (2-D or 3-D) and
    return an array of that same shape.
    """
    masked, ys, xs, true_vals = build_masked_image(img, mask_frac=mask_frac, seed=seed)

    losses = np.empty(len(param_grid))
    for i, value in enumerate(param_grid):
        denoised = denoise_fn(masked, value)
        pred_vals = denoised[ys, xs]
        losses[i] = float(np.mean((pred_vals.astype(np.float64)
                                   - true_vals.astype(np.float64)) ** 2))

    best_idx = int(np.argmin(losses))
    return float(param_grid[best_idx]), losses


def calibrate_wavelet_strength(
    img: np.ndarray,
    param_grid: Sequence[float] = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0),
    mask_frac: float = 0.02,
    seed: int = 0,
) -> Tuple[float, np.ndarray]:
    """Calibrate ``wavelet_denoise``'s ``threshold_factor`` (``--denoise-
    strength``) via ``calibrate_denoiser_param`` -- an alternative to
    ``estimate_denoise_strength``'s SNR-bucket heuristic that doesn't need
    an SNR estimate at all, just the stack itself.
    """
    from src.denoising import wavelet_denoise

    def _denoise_fn(x: np.ndarray, strength: float) -> np.ndarray:
        return wavelet_denoise(x, threshold_factor=float(strength))

    return calibrate_denoiser_param(img, _denoise_fn, param_grid,
                                    mask_frac=mask_frac, seed=seed)
