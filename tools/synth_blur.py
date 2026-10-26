"""Synthetic-blur pairs for the learned deconvolver, generated on the fly (torch).

Truth: a clean star-free background (tools/gen_synth_deconv.py ``bg``) plus point sources
drawn with a *narrow* PSF -- sharper than any real Origin stack -- so the network can learn
to sharpen past a session's best frames, which the seeing pairs cannot teach.

Input: the truth convolved with a random blur kernel K, plus real stack noise (``noise``,
spatially correlated like real stacks) scaled for Poisson noise, ``sqrt(1 + signal / B)``.
K is a Moffat (FWHM, beta, ellipticity, angle), optionally with a coma-like offset
component (centroid kept at zero, so input and truth stay registered), and slightly
different widths per channel. ``input = truth (x) K`` exactly on the pixel grid: the
network's task is to undo K, and since the truth PSF is narrow and fixed in shape, undoing
it means producing narrow stars whatever the input width.

All quantities are in the input's noise units, the space ``nn_denoise`` works in.
"""
from __future__ import annotations

import math

import numpy as np

PATCH = 128
BG = 192                 # generation canvas: PATCH plus a margin for K (circular conv wraps)
KR = 20                  # blur kernel radius (px)
SR = 15                  # truth star stamp radius (px)
SUB = 4                  # subpixel samples per axis when drawing the truth stars

# parameter ranges (feasibility defaults; see the runbook)
TRUTH_FWHM = (1.5, 2.3)
TRUTH_BETA = (3.0, 6.0)
K_FWHM = (1.5, 4.5)      # added blur; input stars ~ sqrt(truth^2 + K^2) wide
K_BETA = (2.5, 6.0)
K_AXIS = (0.75, 1.0)     # minor/major axis ratio
K_COMA = (0.0, 0.3)      # mass fraction in the offset component
K_CHAN = 0.06            # per-channel width spread (+-)
STARS = (1.0, 60.0)      # stars per BG x BG canvas, log-uniform mean
FLUX = (5.0, 5e5)        # total flux per star (noise units x px)
FLUX_SLOPE = 0.6         # dN / dlog F ~ F^-slope
PURE_SKY = 0.15          # fraction of canvases without extended background


def _moffat(r2, fwhm, beta):
    import torch
    alpha = fwhm / (2.0 * torch.sqrt(2.0 ** (1.0 / beta) - 1.0))
    return (1.0 + r2 / alpha ** 2) ** (-beta)


def _u(gen, lo, hi, n, dev):
    import torch
    return torch.tensor(gen.uniform(lo, hi, n), dtype=torch.float32, device=dev)


def blur_kernels(n: int, gen, dev):
    """(n, 3, 2KR+1, 2KR+1) unit-sum kernels, centroid at the centre."""
    import torch
    k = torch.arange(-KR, KR + 1, dtype=torch.float32, device=dev)
    yy, xx = torch.meshgrid(k, k, indexing="ij")
    fw = _u(gen, *K_FWHM, n, dev)[:, None, None, None]
    be = _u(gen, *K_BETA, n, dev)[:, None, None, None]
    q = _u(gen, *K_AXIS, n, dev)[:, None, None, None]
    th = _u(gen, 0, math.pi, n, dev)[:, None, None, None]
    chan = 1.0 + _u(gen, -K_CHAN, K_CHAN, 3 * n, dev).view(n, 3, 1, 1)
    coma = _u(gen, *K_COMA, n, dev) * torch.tensor(gen.random(n) < 0.5, device=dev, dtype=torch.float32)
    phi = _u(gen, 0, 2 * math.pi, n, dev)
    dist = _u(gen, 0.5, 2.0, n, dev)
    c = coma[:, None, None, None]
    # main component shifted back so the total centroid stays at zero
    dy, dx = (dist * torch.sin(phi))[:, None, None, None], (dist * torch.cos(phi))[:, None, None, None]

    def comp(oy, ox, scale):
        u = (xx - ox) * torch.cos(th) + (yy - oy) * torch.sin(th)
        v = -(xx - ox) * torch.sin(th) + (yy - oy) * torch.cos(th)
        return _moffat(u * u / q + v * v * q, fw * chan * scale, be)

    s = c / torch.clamp(1.0 - c, min=1e-6)
    main = comp(-dy * s, -dx * s, 1.0)
    side = comp(dy, dx, 1.5)
    main = main / main.sum((2, 3), keepdim=True)
    side = side / side.sum((2, 3), keepdim=True)
    ker = (1.0 - c) * main + c * side
    return ker / ker.sum((2, 3), keepdim=True), fw.view(n)


def draw_stars(n: int, gen, dev, size: int = BG, density=None, truth_fwhm=None):
    """(n, 3, size, size) point sources with the narrow truth PSF, and the per-canvas list
    of (y, x, flux) for evaluation."""
    import torch
    density, truth_fwhm = density or STARS, truth_fwhm or TRUTH_FWHM
    out = torch.zeros((n, 3, size + 2 * SR, size + 2 * SR), device=dev)
    lam = np.exp(gen.uniform(np.log(density[0]), np.log(density[1]), n)) * (size / BG) ** 2
    counts = gen.poisson(lam)
    tot = int(counts.sum())
    cat = []
    if tot == 0:
        return out[:, :, SR:-SR, SR:-SR], [np.zeros((0, 3))] * n
    idx = np.repeat(np.arange(n), counts)
    y = gen.uniform(0, size, tot)
    x = gen.uniform(0, size, tot)
    lo, hi = FLUX
    u = gen.random(tot)
    flux = lo * (1.0 - u * (1.0 - (hi / lo) ** -FLUX_SLOPE)) ** (-1.0 / FLUX_SLOPE)
    col = np.exp(0.25 * gen.standard_normal((tot, 3)))
    col /= col.mean(1, keepdims=True)
    fw = gen.uniform(*truth_fwhm, n)[idx]
    be = gen.uniform(*TRUTH_BETA, n)[idx]
    iy, ix = np.floor(y).astype(int), np.floor(x).astype(int)
    fy, fx = y - iy, x - ix
    # stamp pixel (j, i) covers [j - SR, j - SR + 1) relative to the star's integer pixel
    sub = (np.arange(SUB) + 0.5) / SUB
    k = np.arange(-SR, SR + 1)
    py = torch.tensor((k[:, None] + sub[None, :]).ravel(), dtype=torch.float32, device=dev)  # (S*SUB,)
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
    dyy = py[None, :] - t(fy)[:, None]                                # (tot, S*SUB)
    dxx = py[None, :] - t(fx)[:, None]
    r2 = dyy[:, :, None] ** 2 + dxx[:, None, :] ** 2
    prof = _moffat(r2, t(fw)[:, None, None], t(be)[:, None, None])
    m = 2 * SR + 1
    prof = prof.view(tot, m, SUB, m, SUB).mean((2, 4))
    prof = prof / prof.sum((1, 2), keepdim=True)
    stamps = prof[:, None] * (t(flux)[:, None] * t(col))[:, :, None, None]     # (tot, 3, m, m)
    hp = size + 2 * SR
    kk = torch.arange(m, device=dev)
    rows = torch.as_tensor(iy, device=dev)[:, None] + kk[None, :]             # (tot, m), padded coords
    cols = torch.as_tensor(ix, device=dev)[:, None] + kk[None, :]
    base = torch.as_tensor(idx, device=dev)[:, None, None, None] * 3 + torch.arange(3, device=dev)[None, :, None, None]
    lin = (base * hp + rows[:, None, :, None]) * hp + cols[:, None, None, :]
    out.view(-1).index_add_(0, lin.reshape(-1), stamps.reshape(-1))
    for i in range(n):
        sel = idx == i
        cat.append(np.c_[y[sel] - 0.5, x[sel] - 0.5, flux[sel]])      # pixel-centre coordinates
    return out[:, :, SR:-SR, SR:-SR], cat


def _rot_flip(t, gen):
    import torch
    t = torch.rot90(t, int(gen.integers(4)), (-2, -1))
    return torch.flip(t, (-1,)) if gen.random() < 0.5 else t


def convolve(img, ker):
    """Circular per-channel convolution of (n, 3, H, W) by (n, 3, k, k) kernels via FFT."""
    import torch
    h, w = img.shape[-2:]
    kp = torch.zeros_like(img)
    kk = ker.shape[-1]
    kp[..., :kk, :kk] = ker
    kp = torch.roll(kp, (-(kk // 2), -(kk // 2)), (-2, -1))
    return torch.fft.irfft2(torch.fft.rfft2(img) * torch.fft.rfft2(kp), s=(h, w))


class SynthSource:
    """Draws (input, truth, weight) batches from a gen_synth_deconv.py npz."""

    def __init__(self, path: str, dev: str):
        import torch
        f = np.load(path)
        self.bg = torch.from_numpy(f["bg"].astype(np.float32)).permute(0, 3, 1, 2)       # CPU, (N, 3, BG, BG)
        self.noise = torch.from_numpy(f["noise"].astype(np.float32)).permute(0, 3, 1, 2)
        self.noise_sky = torch.from_numpy(f["noise_sky"].astype(np.float32))
        self.dev = dev

    def batch(self, n: int, gen, size: int = PATCH, return_meta: bool = False):
        import torch
        dev = self.dev
        canvas = size + (BG - PATCH)
        if canvas == BG:
            bg = self.bg[gen.integers(len(self.bg), size=n)].to(dev)
        else:                                  # larger evaluation fields: tile the bg crop
            bg = self.bg[gen.integers(len(self.bg), size=n)].to(dev)
            reps = -(-canvas // BG)
            bg = bg.repeat(1, 1, reps, reps)[..., :canvas, :canvas]
        bg = torch.stack([_rot_flip(b, gen) for b in bg])
        gain = torch.tensor(gen.uniform(0.5, 1.5, n) * (gen.random(n) >= PURE_SKY), dtype=torch.float32, device=dev)
        stars, cat = draw_stars(n, gen, dev, size=canvas)
        truth = bg * gain[:, None, None, None] + stars
        ker, kfw = blur_kernels(n, gen, dev)
        blurred = convolve(truth, ker)
        m = (canvas - size) // 2
        truth, blurred = truth[..., m:m + size, m:m + size], blurred[..., m:m + size, m:m + size]
        reps = -(-size // PATCH)
        ni = gen.integers(len(self.noise), size=n)
        noise = torch.stack([_rot_flip(self.noise[i].to(dev).repeat(1, reps, reps)[..., :size, :size], gen)
                             for i in ni])
        sky = self.noise_sky[ni].to(dev)[:, :, None, None]
        var = 1.0 + torch.clamp(blurred, min=0.0) / sky
        zin = blurred + noise * torch.sqrt(var)
        if return_meta:
            cat = [c[(c[:, 0] >= m) & (c[:, 0] < m + size) & (c[:, 1] >= m) & (c[:, 1] < m + size)] - [m, m, 0]
                   for c in cat]
            return zin, truth, 1.0 / var, {"cat": cat, "k_fwhm": kfw.cpu().numpy(), "sky": sky}
        return zin, truth, 1.0 / var
