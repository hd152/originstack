"""Is Menon (2007) demosaicing worth using over Malvar for astro frames?

Compares this codebase's two native (Rust, numpy fallback) debayer kernels --
Malvar-He-Cutler (the default) and Menon (2007) DDFAPD directional filtering
(opt-in via --debayer-method menon2007) -- on two synthetic scenes.

Two scenes:
  1. "detail" -- fixed-frequency diagonal colour stripes (below Nyquist, not
     an aliased chirp), the classic demosaicing torture test for periodic
     chrominance. Surprisingly, Malvar wins here too on this diagonal
     pattern -- Menon's direction decision (horizontal vs vertical) has no
     clearly-better answer when gradient energy is split evenly between both
     axes, whereas Malvar's kernels aren't direction-selecting in the first
     place. Worth re-testing with pure horizontal/vertical stripes before
     trusting this as a general "Malvar beats Menon on detail" claim.
  2. "starfield" -- a synthetic astro frame: smooth sky background + Gaussian
     point-source stars + shot noise, mosaic-sampled the same way. This is
     the actual use case for this codebase, and where Menon does show a real
     (if modest) edge.

Metric: mean absolute error against the pre-mosaic RGB ground truth, interior
crop only (boundary handling differs between the two implementations and
isn't the point of this comparison).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.debayer import _HAS_NATIVE, debayer_malvar, debayer_menon2007

_OFFSETS = {  # pattern -> (r, g1, g2, b) offsets, matches src/debayer.py
    'RGGB': ((0, 0), (0, 1), (1, 0), (1, 1)),
}


def mosaic(rgb: np.ndarray, pattern: str = 'RGGB') -> np.ndarray:
    """Sample a full RGB image down to a single-channel Bayer mosaic."""
    (rr, rc), (g1r, g1c), (g2r, g2c), (br, bc) = _OFFSETS[pattern]
    raw = np.empty(rgb.shape[:2], dtype=rgb.dtype)
    raw[rr::2, rc::2] = rgb[rr::2, rc::2, 0]
    raw[g1r::2, g1c::2] = rgb[g1r::2, g1c::2, 1]
    raw[g2r::2, g2c::2] = rgb[g2r::2, g2c::2, 1]
    raw[br::2, bc::2] = rgb[br::2, bc::2, 2]
    return raw


def detail_scene(h=256, w=256, period_px=6, seed=0) -> np.ndarray:
    """Fixed-frequency diagonal colour stripes: the classic demosaicing
    torture test (periodic high-frequency chrominance), at a single frequency
    well below Nyquist (period_px=6 -> 3px/half-cycle, comfortably resolvable
    by a 2px-pitch Bayer sample) so results reflect algorithm quality rather
    than aliasing in an undefined regime. Diagonal orientation exercises both
    H and V directional interpolation, unlike pure horizontal/vertical stripes."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    phase = (yy + xx) * (2.0 * np.pi / period_px)
    rng = np.random.default_rng(seed)
    # Per-channel phase offset so colour (not just luma) carries fine detail.
    rgb = np.stack([
        500.0 + 400.0 * np.cos(phase),
        500.0 + 400.0 * np.cos(phase + 2.0),
        500.0 + 400.0 * np.cos(phase + 4.0),
    ], axis=-1)
    rgb += rng.normal(0, 5.0, rgb.shape)
    return np.clip(rgb, 0, None).astype(np.float32)


def starfield_scene(h=256, w=256, n_stars=40, seed=1) -> np.ndarray:
    """Synthetic astro frame: smooth sky + Gaussian stars + shot noise."""
    rng = np.random.default_rng(seed)
    sky = 1000.0 + rng.normal(0, 15.0, (h, w))
    rgb = np.stack([sky, sky * 1.02, sky * 0.98], axis=-1)  # mild sky colour cast
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(n_stars):
        cy = rng.uniform(10, h - 10)
        cx = rng.uniform(10, w - 10)
        sigma = rng.uniform(1.2, 2.5)
        amp = rng.uniform(500, 8000)
        # Slight per-channel colour (real stars aren't perfectly white).
        colour = rng.uniform(0.8, 1.2, 3)
        g = amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
        for c in range(3):
            rgb[:, :, c] += g * colour[c]
    return np.clip(rgb, 0, None).astype(np.float32)


def compare(name: str, rgb_truth: np.ndarray, margin: int = 12):
    raw = mosaic(rgb_truth, 'RGGB')

    t0 = time.perf_counter()
    out_malvar = debayer_malvar(raw, 'RGGB')
    t_malvar = time.perf_counter() - t0

    m = margin
    truth_c = rgb_truth[m:-m, m:-m, :].astype(np.float64)
    malvar_c = out_malvar[m:-m, m:-m, :].astype(np.float64)
    mae_malvar = np.abs(malvar_c - truth_c).mean()
    print(f"\n  Scene: {name}  ({rgb_truth.shape[1]}x{rgb_truth.shape[0]})")
    print(f"    malvar    MAE={mae_malvar:8.4f}  ({t_malvar*1000:6.2f} ms)")

    t0 = time.perf_counter()
    out_menon = debayer_menon2007(raw, 'RGGB')
    t_menon = time.perf_counter() - t0
    menon_c = out_menon[m:-m, m:-m, :].astype(np.float64)
    mae_menon = np.abs(menon_c - truth_c).mean()
    delta_pct = (mae_malvar - mae_menon) / mae_malvar * 100.0
    print(f"    menon2007 MAE={mae_menon:8.4f}  ({t_menon*1000:6.2f} ms)"
          f"   [{'menon better' if delta_pct > 0 else 'malvar better'} by {abs(delta_pct):.1f}%,"
          f" {t_menon / t_malvar:.1f}x slower]")


if __name__ == '__main__':
    _native_note = 'yes' if _HAS_NATIVE else 'no (numpy fallback -- pip install maturin && maturin develop --release in ext/astro_native)'
    print(f"Native kernels: {_native_note}\n")
    compare("detail (fixed-frequency colour stripes)", detail_scene())
    compare("starfield (astro-realistic)", starfield_scene())
