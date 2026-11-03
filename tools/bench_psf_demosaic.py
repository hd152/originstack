"""PSF-matched single-frame demosaic (APP's "Adaptive Airy Disc" idea) against RCD and Malvar.

Per channel, normalized convolution of the measured Bayer samples with a Gaussian matched to the
star width, on a synthetic star field with shot + read noise. Result (2026-11, not kept): RCD has the
lowest star and chroma error at every width; the Gaussian only lowers sky error because it blurs.
See dev-notes/siril-comparison.md. Run: python tools/bench_psf_demosaic.py
"""
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, r'C:\source\originstack'); sys.path.insert(0, r'C:\source\originstack\tools')
from bench_debayer_quality import mosaic  # noqa: E402

from src.debayer import debayer_malvar, debayer_rcd  # noqa: E402


def masks(h, w):
    m = np.zeros((3, h, w), np.float32)
    m[0, 0::2, 0::2] = 1; m[1, 0::2, 1::2] = 1; m[1, 1::2, 0::2] = 1; m[2, 1::2, 1::2] = 1
    return m

def nconv(raw, sk):
    h, w = raw.shape
    m = masks(h, w)
    out = np.empty((h, w, 3), np.float32)
    for c in range(3):
        num = gaussian_filter(raw * m[c], sk, mode='mirror')
        den = gaussian_filter(m[c], sk, mode='mirror')
        out[:, :, c] = num / np.maximum(den, 1e-6)
    return out

def scene(psf_sigma, h=256, w=256, n=60, seed=1):
    rng = np.random.default_rng(seed)
    sky = 800.0
    rgb = np.full((h, w, 3), sky, np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(n):
        cy, cx = rng.uniform(12, h - 12), rng.uniform(12, w - 12)
        s = psf_sigma * rng.uniform(0.9, 1.1)
        amp = 10 ** rng.uniform(2.3, 3.9)
        col = rng.uniform(0.6, 1.4, 3)
        g = amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
        for c in range(3):
            rgb[:, :, c] += g * col[c]
    return rgb

def run(psf_sigma, read=12.0, gain_e=1.0):
    truth = scene(psf_sigma)
    rng = np.random.default_rng(5)
    noisy_truth_mosaic = mosaic(truth)
    raw = rng.poisson(np.clip(noisy_truth_mosaic * gain_e, 0, None)) / gain_e + rng.normal(0, read, truth.shape[:2])
    raw = raw.astype(np.float32)
    m = 14
    t = truth[m:-m, m:-m].astype(np.float64)
    # sky-only region for noise, star region for error
    star = (truth[m:-m, m:-m, 1] - 800) > 40
    rows = {'rcd': debayer_rcd(raw), 'malvar': debayer_malvar(raw)}
    for k in (0.7, 0.9, 1.1, 1.4):
        rows[f'nconv{k}'] = nconv(raw, k * psf_sigma * 0.8 + 0.3)
    print(f'PSF sigma {psf_sigma} (FWHM {2.355*psf_sigma:.1f} px)')
    for k, v in rows.items():
        d = v[m:-m, m:-m].astype(np.float64) - t
        sky_noise = d[~star].std()
        ratio_err = np.abs(d[star]).mean()
        # colour error: per-pixel chroma (R-G, B-G) in star region
        ch = np.abs((d[..., 0] - d[..., 1])[star]).mean() + np.abs((d[..., 2] - d[..., 1])[star]).mean()
        print(f'  {k:10s} sky-err {sky_noise:6.2f}  star-err {ratio_err:7.2f}  chroma-err {ch:7.2f}')

if __name__ == '__main__':
    for s in (1.3, 1.8, 2.4):
        run(s)
