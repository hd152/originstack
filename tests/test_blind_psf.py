"""estimate_psf_blind must return the stars' own PSF. Its old default
'blind RL refinement' blended a patch of a whole-image correlation into the
estimate every iteration, flattening a sigma=1.5 px Gaussian PSF to a broad
plateau (peak 0.068 -> 0.011) -- and --auto enables the blind PSF for most
--deconvolve runs."""
import numpy as np
from astropy.table import Table

from src.psf_deconvolution import estimate_psf_blind


def test_blind_psf_matches_the_star_profile():
    rng = np.random.default_rng(7)
    h, w, sigma = 256, 256, 1.5
    yy, xx = np.mgrid[:h, :w]
    xs, ys = rng.uniform(20, w - 20, 60), rng.uniform(20, h - 20, 60)
    fluxes = rng.uniform(2000, 6000, 60)
    lum = np.full((h, w), 100.0)
    for x, y, f in zip(xs, ys, fluxes):
        lum += f / (2 * np.pi * sigma ** 2) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    lum += rng.normal(0, 1.0, lum.shape)
    img = np.repeat(lum[:, :, None], 3, axis=2).astype(np.float32)
    stars = Table({'xcentroid': xs, 'ycentroid': ys, 'flux': fluxes})

    psf, _ = estimate_psf_blind(img, stars)
    assert psf is not None
    c = psf.shape[0] // 2
    py, px = np.mgrid[:psf.shape[0], :psf.shape[1]]
    within3 = psf[(py - c) ** 2 + (px - c) ** 2 <= 9].sum() / psf.sum()
    assert within3 > 0.7          # ideal ~0.86 continuous, 0.81 sampled; was 0.22
    assert psf.max() > 0.8 / (2 * np.pi * sigma ** 2)      # ideal peak 0.071; was 0.011
