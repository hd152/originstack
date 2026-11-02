"""Injection-recovery test of Phase 4 on faint extended light, and blank-sky aperture curves.

Phase 4 (background extraction, denoising, local contrast) decides what happens to faint,
smooth, extended light -- integrated flux nebula, galactic cirrus, galaxy outskirts -- and the
other benches cannot tell how much of it survives: they have no ground truth for it. This
makes one. A Gaussian random field with the optical cirrus power spectrum (P(k) ~ k^-2.9,
Miville-Deschenes+ 2016) is added to a linear stack at a few fractions of the sky's per-pixel
noise, Phase 4 runs on it and on the untouched stack, and the difference of the two outputs
is compared with what was injected, ring by ring in spatial period:

  T(period) = sum Re[F(out_injected - out_control) conj(F(injected))] / sum |F(injected)|^2

1 means that scale passes through untouched, 0 that it is removed (a background step eating
it, a denoiser smoothing it). The difference with a control run cancels the stars and the
real structure, which Phase 4 changes in the same way in both.

The control output also gives blank-sky aperture curves: the scatter of the mean over
star-free circular apertures of radius r against the white-noise expectation
sigma_pix / sqrt(pi r^2). A ratio above 1 is the correlated noise (resampling, debayering,
denoising) that makes a faint-depth estimate extrapolated from the pixel sigma too optimistic.

Usage:
  python tools/bench_lsb_injection.py STACK.fits [--amplitudes 0.25 0.5 1 2] \\
      [--extra "--denoiser wavelet"] [--seed 1] [--json out.json]

``--extra`` flags are appended to the stack's own ``<stem>_config.toml`` settings, as in
``tools/bench_phase4.py``. The stack and its config are copied to a temporary directory so
the Phase 4 cache and any side files stay out of the original's folder. Delete nothing by hand:
the cache is keyed on the injected pixels. No network (``--offline``), no training, no GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
import tempfile
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

LUMA = np.array([0.299, 0.587, 0.114])
PERIOD_EDGES = (16, 32, 64, 128, 256, 512, 1024)       # px: ring edges of the transfer function
APERTURE_RADII = (4, 8, 16, 32, 64)


# ---------------------------------------------------------------------------
# The injected field
# ---------------------------------------------------------------------------

def cirrus_field(shape: Tuple[int, int], rng: np.random.Generator, beta: float = 2.9,
                 min_period: float = 16.0, max_period: float = 1000.0) -> np.ndarray:
    """Zero-mean, unit-variance Gaussian random field with power spectrum ``k**-beta``,
    limited to spatial periods between ``min_period`` and ``max_period`` pixels (raised-cosine
    edges, an octave wide)."""
    from scipy import fft
    h, w = shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    k = np.hypot(fy, fx)
    k[0, 0] = 1.0
    amp = k ** (-beta / 2.0)
    amp[0, 0] = 0.0
    lo, hi = 1.0 / max_period, 1.0 / min_period

    def edge(x, a, b):                      # 0 below a, 1 above b, cosine between
        t = np.clip((x - a) / (b - a), 0.0, 1.0)
        return 0.5 - 0.5 * np.cos(np.pi * t)

    amp *= edge(k, lo / 1.5, lo * 1.5) * (1.0 - edge(k, hi / 1.5, hi * 1.5))
    white = fft.fft2(rng.standard_normal(shape), workers=-1)
    field = fft.ifft2(white * amp, workers=-1).real
    return (field / field.std()).astype(np.float64)


def pixel_sigma(lum: np.ndarray) -> float:
    """Per-pixel noise of a luminance plane (MAD of the plane minus its 8 px blur)."""
    from src.background import _pixel_noise_sigma
    return float(_pixel_noise_sigma(np.ascontiguousarray(lum, dtype=np.float32)))


def inject(stack: np.ndarray, field: np.ndarray, amplitude: float,
           sigmas: Sequence[float]) -> np.ndarray:
    """``stack + amplitude * sigma_c * field`` in every channel (a grey signal)."""
    out = stack.astype(np.float64).copy()
    for c in range(stack.shape[2]):
        out[:, :, c] += amplitude * float(sigmas[c]) * field
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def transfer_function(diff: np.ndarray, injected: np.ndarray,
                      edges: Sequence[float] = PERIOD_EDGES) -> List[Dict[str, float]]:
    """Fraction of the injected field found in ``diff`` per band of spatial period (pixels).

    ``diff`` is (output with injection) - (control output), ``injected`` what was added, both
    in the same units and shape. Rings are in 1/period; a Hann window keeps the frame edge out."""
    from scipy import fft
    w = np.outer(np.hanning(diff.shape[0]), np.hanning(diff.shape[1]))
    fd = fft.fft2((diff - diff.mean()) * w, workers=-1)
    fi = fft.fft2((injected - injected.mean()) * w, workers=-1)
    k = np.hypot(np.fft.fftfreq(diff.shape[0])[:, None], np.fft.fftfreq(diff.shape[1])[None, :])
    num = (fd * np.conj(fi)).real
    den = (np.abs(fi) ** 2)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (k > 1.0 / hi) & (k <= 1.0 / lo)
        d = float(den[sel].sum())
        rows.append({'period_lo': float(lo), 'period_hi': float(hi),
                     'T': float(num[sel].sum() / d) if d > 0 else float('nan')})
    return rows


def sky_mask(lum: np.ndarray) -> np.ndarray:
    """Pixels within 2.5 robust sigma of the median of the 3 px smoothed plane (no stars)."""
    from scipy import ndimage
    sm = ndimage.gaussian_filter(lum, 3.0)
    med = float(np.median(sm))
    sig = 1.4826 * float(np.median(np.abs(sm - med)))
    return np.abs(sm - med) < 2.5 * max(sig, 1e-12)


def aperture_curves(lum: np.ndarray, mask: np.ndarray, radii: Sequence[int] = APERTURE_RADII,
                    rng: Optional[np.random.Generator] = None, max_apertures: int = 400
                    ) -> List[Dict[str, float]]:
    """Mean-flux scatter of blank-sky circular apertures against the white-noise expectation.

    Apertures sit where the whole disc is sky (``mask``) and no closer than their diameter, so
    the scatter is of independent regions. ``ratio`` = measured scatter / (pixel sigma /
    sqrt(area)); ``bias`` = mean of the aperture means minus the sky median, in pixel sigmas."""
    from scipy import signal
    rng = rng or np.random.default_rng(0)
    sig_pix = pixel_sigma(lum)
    sky_med = float(np.median(lum[mask]))
    rows = []
    for r in radii:
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        disc = (yy * yy + xx * xx <= r * r).astype(np.float64)
        area = disc.sum()
        full = signal.fftconvolve(mask.astype(np.float64), disc, mode='same') >= area - 0.5
        means = signal.fftconvolve(lum, disc / area, mode='same')
        ys, xs = np.nonzero(full[r:-r:2 * r, r:-r:2 * r])
        ys, xs = ys * 2 * r + r, xs * 2 * r + r
        if ys.size < 8:
            rows.append({'radius': float(r), 'n': int(ys.size), 'ratio': float('nan'),
                         'scatter': float('nan'), 'expected': float(sig_pix / np.sqrt(area)),
                         'bias_sigma': float('nan')})
            continue
        pick = rng.choice(ys.size, size=min(ys.size, max_apertures), replace=False)
        vals = means[ys[pick], xs[pick]]
        scatter = float(np.std(vals))
        expected = float(sig_pix / np.sqrt(area))
        rows.append({'radius': float(r), 'n': int(vals.size), 'scatter': scatter,
                     'expected': expected, 'ratio': scatter / expected,
                     'bias_sigma': float((vals.mean() - sky_med) / sig_pix)})
    return rows


# ---------------------------------------------------------------------------
# The benchmark
# ---------------------------------------------------------------------------

def luminance_of(img: np.ndarray) -> np.ndarray:
    return (np.asarray(img, dtype=np.float64) @ LUMA)


def bench(stack: np.ndarray, amplitudes: Sequence[float],
          run: Callable[[np.ndarray], np.ndarray], seed: int = 1) -> Dict[str, object]:
    """Transfer functions for each amplitude and the control's aperture curves.

    ``run(stack)`` is the pipeline under test: linear stack in, Phase 4 output out (same
    shape). It is called once for the control and once per amplitude."""
    rng = np.random.default_rng(seed)
    lum_in = luminance_of(stack)
    sigmas = [pixel_sigma(stack[:, :, c]) for c in range(stack.shape[2])]
    sig_lum = float(np.sqrt(np.sum((LUMA * np.asarray(sigmas)) ** 2)))     # independent channels
    field = cirrus_field(stack.shape[:2], rng)
    ctl = np.asarray(run(stack), dtype=np.float32)
    lum_ctl = luminance_of(ctl)
    out: Dict[str, object] = {'pixel_sigma_lum': sig_lum, 'seed': seed, 'amplitudes': {}}
    unit = field * float(LUMA @ np.asarray(sigmas))      # injected luminance per unit amplitude
    for amp in amplitudes:
        res = np.asarray(run(inject(stack, field, amp, sigmas)), dtype=np.float32)
        diff = luminance_of(res) - lum_ctl
        out['amplitudes'][str(amp)] = transfer_function(diff, amp * unit)
    out['apertures_control'] = aperture_curves(lum_ctl, sky_mask(lum_ctl), rng=rng)
    out['apertures_input'] = aperture_curves(lum_in, sky_mask(lum_in), rng=rng)
    return out


def format_report(res: Dict[str, object]) -> str:
    amps = list(res['amplitudes'].keys())
    lines = ["Injection-recovery transfer function T(period) -- fraction of injected faint extended "
             "light (P(k) ~ k^-2.9) that survives Phase 4",
             "  injected amplitude in units of the stack's per-pixel luminance noise "
             f"({res['pixel_sigma_lum']:.1f} ADU)",
             "  period (px)   " + "  ".join(f"{a:>7}" for a in amps)]
    first = res['amplitudes'][amps[0]] if amps else []
    for i, row in enumerate(first):
        cells = "  ".join(f"{res['amplitudes'][a][i]['T']:7.2f}" for a in amps)
        lines.append(f"  {row['period_lo']:>4g}-{row['period_hi']:<6g}  {cells}")
    lines.append("")
    lines.append("Blank-sky aperture scatter vs white-noise expectation (ratio > 1: correlated "
                 "noise; the depth a pixel-sigma extrapolation promises is that much too good)")
    lines.append("  radius   input ratio   output ratio   output bias (pixel sigma)")
    inp = {r['radius']: r for r in res['apertures_input']}
    for r in res['apertures_control']:
        i = inp.get(r['radius'], {})
        lines.append(f"  {r['radius']:>5g}   {i.get('ratio', float('nan')):11.2f}   "
                     f"{r['ratio']:12.2f}   {r['bias_sigma']:+10.3f}")
    return "\n".join(lines)


def _prepare_workdir(stack_path: str) -> Tuple[str, str]:
    """Copy the stack, its config and cached Gaia list into a temporary directory."""
    work = tempfile.mkdtemp(prefix="lsb_inject_")
    stem = os.path.splitext(stack_path)[0]
    for suffix in (".fits", "_config.toml", "_gaia.json"):
        if os.path.exists(stem + suffix):
            shutil.copy2(stem + suffix, os.path.join(work, os.path.basename(stem) + suffix))
    return work, os.path.join(work, os.path.basename(stem) + ".fits")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stack", help="linear stack FITS (RAWSTACK) of an earlier run")
    ap.add_argument("--amplitudes", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--extra", default="", help="extra command-line flags for Phase 4")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json")
    args = ap.parse_args(argv)

    from astropy.io import fits
    from bench_phase4 import run_phase4

    from src.merge import load_merge_stack

    stack, _meta = load_merge_stack(args.stack)
    header = fits.getheader(args.stack)
    work, work_stack = _prepare_workdir(args.stack)
    extra = ["--offline", *shlex.split(args.extra)]
    counter = [0]

    def run(arr: np.ndarray) -> np.ndarray:
        counter[0] += 1
        print(f"  Phase 4 run {counter[0]} of {1 + len(args.amplitudes)} ...", flush=True)
        P, _D, _a, _log = run_phase4(arr, extra, work_stack, work, f"run{counter[0]}",
                                     header=header)
        return np.asarray(P)

    try:
        res = bench(stack, args.amplitudes, run, seed=args.seed)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(format_report(res))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
