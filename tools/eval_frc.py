"""Split-half resolution and denoiser risk from two independent half-stacks.

  python tools/eval_frc.py A.fits B.fits [--denoised D.fits] [--json out.json]

A and B are two stacks of the same field built from disjoint halves of the frames (e.g. odd
and even; ``--noise-validate`` prints the same FRC line for its own halves). Both are reduced
to luminance. Reports (``src/frc.py``):

  * the Fourier ring correlation curve at a few spatial periods, the finest period it holds
    above 1/7, and the full-stack SNR per period;
  * with ``--denoised D.fits`` (the denoiser applied to A alone): the per-scale risk of D
    against the independent B, relative to doing nothing (a ratio above 1 at a scale means the
    denoiser destroys structure there; a bump at 16-64 px is over-smoothed faint extended
    light), and the method-noise check (is what it removed white, and uncorrelated with what
    it kept?).

Learned denoisers reproduce their inventions identically in A's and B's treatment, so read an
FRC of a network's output as an upper bound on its resolution.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_luminance(path: str) -> np.ndarray:
    from astropy.io import fits

    from src.debayer import luminance
    d = np.asarray(fits.getdata(path), dtype=np.float32)
    if d.ndim == 3:
        if d.shape[0] == 3 and d.shape[-1] != 3:
            d = np.transpose(d, (1, 2, 0))
        return luminance(d).astype(np.float64)
    return d.astype(np.float64)


def sky_mask(lum: np.ndarray) -> np.ndarray:
    """Pixels away from stars and bright structure: within 2.5 robust sigma of the smoothed median."""
    from scipy import ndimage
    sm = ndimage.gaussian_filter(lum, 3.0)
    med = float(np.median(sm))
    sig = 1.4826 * float(np.median(np.abs(sm - med)))
    return np.abs(sm - med) < 2.5 * max(sig, 1e-12)


def analyse(a: np.ndarray, b: np.ndarray, d: np.ndarray | None = None) -> dict:
    from src.frc import (
        fourier_ring_correlation,
        method_noise,
        resolution_period,
        scale_risk,
        ssnr_from_frc,
    )
    freq, frc = fourier_ring_correlation(a, b)
    snr = ssnr_from_frc(frc)['full']
    out = {'resolution_period_px': resolution_period(freq, frc), 'frc_at': {}, 'snr_full_at': {}}
    for period in (64, 32, 16, 8, 4, 3):
        j = int(np.argmin(np.abs(freq - 1.0 / period)))
        out['frc_at'][period] = float(frc[j])
        out['snr_full_at'][period] = float(snr[j])
    if d is not None:
        m = sky_mask(0.5 * (a + b))
        r = scale_risk(a, b, d, m)
        out['risk'] = {'sigma_px': r['sigma'].tolist(), 'ratio': r['ratio'].tolist(),
                       'noop_var': r['noop'].tolist()}
        out['method_noise'] = method_noise(a, d, m)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--denoised', help="denoiser output for A (same shape)")
    ap.add_argument('--json')
    args = ap.parse_args(argv)
    a, b = load_luminance(args.a), load_luminance(args.b)
    d = load_luminance(args.denoised) if args.denoised else None
    res = analyse(a, b, d)
    per = res['resolution_period_px']
    print("split-half resolution (FRC = 1/7): "
          + (f"{per:.1f} px period" if per else "halves agree out to Nyquist"))
    print("  period   FRC    full-stack SNR")
    for p in res['frc_at']:
        print(f"  {p:>4} px  {res['frc_at'][p]:5.2f}  {res['snr_full_at'][p]:8.2f}")
    if 'risk' in res:
        print("  denoiser risk vs doing nothing, by scale (ratio > 1 = structure destroyed):")
        for s, rr in zip(res['risk']['sigma_px'], res['risk']['ratio']):
            print(f"    {s:>4g}-{2 * s:<4g} px  {rr:6.2f}")
        mn = res['method_noise']
        ac = ", ".join(f"lag {k}: {v:+.2f}" for k, v in mn['autocorr'].items())
        print(f"  removed component: autocorrelation {ac}; correlation with output "
              f"{mn['corr_with_output']:+.2f}")
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as fh:
            json.dump(res, fh, indent=2, default=lambda o: None if o is None else float(o))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
