"""Train the learned star remover (``--star-removal nn``, src/nn_starless.py) and export its
weights to src/data/nn_starless.npz.

No real image is starless, so the targets are real linear stacks and the inputs are the
same stacks with stars added (``bank`` measures, per session, what those stars must look
like; ``train`` adds them on the fly):

* PSF: the median of the session's bright, isolated, unsaturated stars, per channel, out
  to ``PSF_R`` px -- the session's own seeing, its blue halo and its wings.
* colour: channel flux ratios of the session's detected stars, drawn at random.
* brightness: peaks from ~1.5 sky sigma to past the session's saturation level, mostly
  faint (a power law, as real star counts are), with a share spread evenly in log so bright
  and saturated stars are well represented; a saturated star is clipped at the session's
  own measured core level, as the stack clips it.
* shot noise: the stars' Poisson noise in sky-sigma units, ``var = z * sigma / sky`` per
  channel (sky-limited exposures), from the session's own sky level and noise.

The stacks' own stars stay in both input and target; a mask around each (out to where its
PSF falls below half a sky sigma) gives them zero loss weight, so the network is never
taught to keep or to remove them. Inputs are normalised with src.nn_denoise.normalise, as
at run time. The loss is L2 in noise units weighted by 1 / (1 + the added stars' shot-noise
variance), so a bright star's core does not dominate.

Usage:
    pip install torch            (into a separate venv: it is not a project dependency)
    python tools/train_nn_starless.py bank  --stacks DIR --bank starless_bank.npz
    python tools/train_nn_starless.py train --bank starless_bank.npz --steps 40000
    python tools/train_nn_starless.py eval  --stacks DIR [--holdout-only] --out-dir DIR
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import nn_denoise as nd  # noqa: E402
from src import nn_starless as ns  # noqa: E402

PATCH = 128            # training patch (px), a multiple of 16
PSF_R = 40             # empirical PSF radius (px): stamps are (2 * PSF_R + 1)^2
DEPTH = 4              # pooling levels
WIDTHS = (24, 48, 96, 96)
MID = 128
# sessions never used for training (scored by ``eval``): a nebula, a dense star cloud,
# a galaxy, and the 300-frame dithered night
HOLDOUT = ("Crab_Nebula", "Sagittarius_Star_Cloud", "Messier_101_2026-04-27",
           "Whirlpool_Galaxy_2026-05-10")


def is_holdout(name: str) -> bool:
    return any(h in name for h in HOLDOUT)


# ── per-session measurements ────────────────────────────────────────────────────

def load_stack(path: str) -> np.ndarray:
    from astropy.io import fits
    d = fits.getdata(path).astype(np.float32)
    if d.ndim == 3 and d.shape[0] == 3:
        d = np.ascontiguousarray(np.moveaxis(d, 0, -1))
    return d


def normalised(img: np.ndarray):
    """(z, bg, sigma, shot coefficient per channel) as src.nn_starless sees the image."""
    src = np.where(np.isfinite(img), img, 0.0).astype(np.float32)
    grid, sigma = nd.normalise(src)
    bg = nd.background_at(grid, 0, 0, *src.shape[:2])
    z = (src - bg) / sigma.astype(np.float32)
    sky = np.maximum(np.median(bg.reshape(-1, 3), axis=0), 1e-6)
    return z, bg, sigma, (sigma / sky).astype(np.float32)


def detect(z: np.ndarray):
    from src.quality import detect_stars_auto
    lum = (0.299 * z[..., 0] + 0.587 * z[..., 1] + 0.114 * z[..., 2]).astype(np.float32)
    s = detect_stars_auto(lum, 1.0, background=0.0)
    return s if s is not None else np.zeros(0)


def empirical_psf(z: np.ndarray, stars, sat_z: np.ndarray, max_stars: int = 60):
    """(3, 2R+1, 2R+1) unit-sum PSF per channel from bright isolated unsaturated stars."""
    h, w, _ = z.shape
    R = PSF_R
    if len(stars) == 0:
        return None
    xs, ys, pk = stars['xcentroid'], stars['ycentroid'], stars['peak']
    order = np.argsort(-pk)
    pts = np.column_stack([ys, xs])
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    stamps = []
    for i in order:
        y, x = ys[i], xs[i]
        if not (R + 2 <= y < h - R - 2 and R + 2 <= x < w - R - 2):
            continue
        core = z[int(round(y)) - 1:int(round(y)) + 2, int(round(x)) - 1:int(round(x)) + 2]
        if np.any(core.max(axis=(0, 1)) > 0.7 * sat_z) or core.min() < 10.0:
            continue                                   # saturated, or too faint for the wings
        near = tree.query_ball_point([y, x], R)
        if any(j != i and pk[j] > 0.05 * pk[i] for j in near):
            continue                                   # a neighbour would show in the median
        iy, ix = int(round(y)), int(round(x))
        st = z[iy - R - 1:iy + R + 2, ix - R - 1:ix + R + 2].astype(np.float64)
        st = ndi.shift(st, (iy - y, ix - x, 0), order=3, mode='nearest')[1:-1, 1:-1]
        st -= np.median(np.concatenate([st[:3].reshape(-1, 3), st[-3:].reshape(-1, 3),
                                        st[:, :3].reshape(-1, 3), st[:, -3:].reshape(-1, 3)]), axis=0)
        tot = st.sum(axis=(0, 1))
        if np.any(tot <= 0):
            continue
        stamps.append(st / tot)
        if len(stamps) >= max_stars:
            break
    if len(stamps) < 8:
        return None
    psf = np.median(np.stack(stamps), axis=0)
    psf = np.maximum(psf, 0.0)
    psf /= psf.sum(axis=(0, 1))
    return psf.transpose(2, 0, 1).astype(np.float32), len(stamps)


def star_colours(z: np.ndarray, stars, n: int = 400) -> np.ndarray:
    """(m, 3) channel flux ratios (G = 1) of detected stars, from 3x3 core sums."""
    out = []
    h, w, _ = z.shape
    for s in stars[np.argsort(-stars['peak'])][:n]:
        y, x = int(round(s['ycentroid'])), int(round(s['xcentroid']))
        if 2 <= y < h - 2 and 2 <= x < w - 2:
            f = z[y - 1:y + 2, x - 1:x + 2].sum(axis=(0, 1))
            if f[1] > 10 and np.all(f > 0):
                out.append(f / f[1])
    return np.array(out, np.float32) if out else np.ones((1, 3), np.float32)


def real_star_mask(shape, stars, psf: np.ndarray) -> np.ndarray:
    """Pixels where a real star's light exceeds half a sky sigma (from the session PSF's
    radial profile scaled to its peak), plus 2 px."""
    h, w = shape
    prof = psf.max(axis=0)
    cy = PSF_R
    radial = np.array([prof[cy, cy + r] if cy + r < prof.shape[1] else 0.0 for r in range(PSF_R + 1)])
    radial = radial / max(radial[0], 1e-12)
    m = np.zeros((h, w), bool)
    yy, xx = np.mgrid[-PSF_R - 8:PSF_R + 9, -PSF_R - 8:PSF_R + 9]
    rr = np.hypot(yy, xx)
    for s in stars:
        pk = max(float(s['peak']), 1.0)
        above = np.flatnonzero(radial * pk > 0.5)
        r = (above[-1] if above.size else 1) + 2
        if radial[-1] * pk > 0.5:
            r = PSF_R + 8                           # wings past the stamp: mask the whole stamp
        y, x = int(round(s['ycentroid'])), int(round(s['xcentroid']))
        y0, y1, x0, x1 = max(0, y - r), min(h, y + r + 1), max(0, x - r), min(w, x + r + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        sub = rr[PSF_R + 8 + (y0 - y):PSF_R + 8 + (y1 - y), PSF_R + 8 + (x0 - x):PSF_R + 8 + (x1 - x)] <= r
        m[y0:y1, x0:x1] |= sub
    return m


def session_bank(path: str, per_session: int, rng):
    img = load_stack(path)
    z, _bg, _sig, shot = normalised(img)
    stars = detect(z)
    sat_z = np.nanmax(z.reshape(-1, 3), axis=0).astype(np.float32)
    got = empirical_psf(z, stars, sat_z)
    if got is None:
        return None
    psf, n_psf = got
    mask = real_star_mask(z.shape[:2], stars, psf)
    h, w, _ = z.shape
    za, wa = [], []
    for _ in range(per_session):
        y = int(rng.integers(0, h - PATCH))
        x = int(rng.integers(0, w - PATCH))
        za.append(z[y:y + PATCH, x:x + PATCH].astype(np.float16))
        wa.append(~mask[y:y + PATCH, x:x + PATCH])
    return dict(z=np.stack(za), keep=np.stack(wa), psf=psf, col=star_colours(z, stars),
                sat=sat_z, shot=shot, n_stars=len(stars), n_psf=n_psf, masked=float(mask.mean()))


def build_bank(stacks: str, out: str, per_session: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    paths = sorted(p for p in glob.glob(os.path.join(stacks, '*.fits'))
                   if not is_holdout(os.path.basename(p)))
    Z, K, S, psfs, cols, sats, shots, names = [], [], [], [], [], [], [], []
    for p in paths:
        t = time.time()
        try:
            b = session_bank(p, per_session, rng)
        except Exception as exc:                      # an odd stack must not end the bank
            print(f"skip {os.path.basename(p)}: {exc}", flush=True)
            continue
        if b is None:
            print(f"skip {os.path.basename(p)}: too few PSF stars", flush=True)
            continue
        i = len(psfs)
        Z.append(b['z']); K.append(b['keep']); S.append(np.full(len(b['z']), i))
        psfs.append(b['psf']); cols.append(b['col']); sats.append(b['sat']); shots.append(b['shot'])
        names.append(os.path.basename(p))
        print(f"[{i}] {os.path.basename(p)}: {b['n_stars']} stars, PSF from {b['n_psf']}, "
              f"masked {b['masked']:.0%}, sat z {b['sat'].round(0)}, shot {b['shot'].round(4)} "
              f"({time.time() - t:.0f} s)", flush=True)
    ncol = max(len(c) for c in cols)
    colarr = np.stack([np.resize(c, (ncol, 3)) for c in cols])
    np.savez(out, z=np.concatenate(Z), keep=np.concatenate(K), sess=np.concatenate(S),
             psf=np.stack(psfs), col=colarr, sat=np.stack(sats), shot=np.stack(shots),
             names=np.array(names))
    print(f"bank: {sum(len(z) for z in Z)} patches from {len(psfs)} sessions -> {out}")


# ── network ─────────────────────────────────────────────────────────────────────

def make_net():
    import torch
    import torch.nn as nn

    def conv(i, o):
        return nn.Conv2d(i, o, 3, padding=1)

    class StarlessNet(nn.Module):
        """Mirrors src/nn_starless.forward layer for layer (names become the npz keys)."""

        def __init__(self, c=3):
            super().__init__()
            prev = c
            for k, f in enumerate(WIDTHS):
                setattr(self, f'e{k}a', conv(prev, f))
                setattr(self, f'e{k}b', conv(f, f))
                prev = f
            self.ma, self.mb = conv(prev, MID), conv(MID, MID)
            up = MID
            for k in reversed(range(DEPTH)):
                f = WIDTHS[k]
                setattr(self, f'd{k}a', conv(up + f, f))
                setattr(self, f'd{k}b', conv(f, f))
                up = f
            self.out = nn.Conv2d(WIDTHS[0], c, 1)
            self.act = nn.LeakyReLU(nd.LEAK)
            self.pool = nn.AvgPool2d(2)
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        def forward(self, x):
            a = self.act
            skips = []
            t = x
            for k in range(DEPTH):
                t = a(getattr(self, f'e{k}b')(a(getattr(self, f'e{k}a')(t))))
                skips.append(t)
                t = self.pool(t)
            t = a(self.mb(a(self.ma(t))))
            for k in reversed(range(DEPTH)):
                t = torch.cat([self.upsample(t), skips[k]], 1)
                t = a(getattr(self, f'd{k}b')(a(getattr(self, f'd{k}a')(t))))
            return self.out(t)

    return StarlessNet()


def export(net, path: str) -> None:
    np.savez(path, **{k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()})


def check_parity(net, path: str) -> float:
    """Max relative |torch - numpy| on a random input; the shipped path must match."""
    import torch
    x = np.random.default_rng(1).normal(0, 1, (3, 64, 96)).astype(np.float32)
    with torch.no_grad():
        ref = net.cpu().eval()(torch.from_numpy(x)[None])[0].numpy()
    nd._weights_cache.pop(path, None)
    got = ns.forward(x, nd.load_weights(path))
    return float(np.abs(ref - got).max() / (np.abs(ref).max() + 1e-12))


# ── star injection ──────────────────────────────────────────────────────────────

def inject(base, sess, bank, rng, dev, empty_frac: float = 0.1):
    """(input, star light, shot variance) for a batch of base patches (B, 3, P, P), with
    stars drawn from each patch's own session."""
    import torch
    import torch.nn.functional as F
    B, C, P, _ = base.shape
    R = PSF_R
    canvas = torch.zeros((B, C, P + 2 * R, P + 2 * R), device=dev)
    psf_all, col_all, sat_all, shot_all = bank
    for i in range(B):
        if rng.random() < empty_frac:
            continue
        s = int(sess[i])
        n = int(rng.poisson(np.exp(rng.uniform(np.log(2.0), np.log(40.0)))))
        if n == 0:
            continue
        # peak z: a power law (dN/dF ~ F^-1.6) from 1.5 sigma, plus a log-uniform share
        u = rng.random(n)
        pl = 1.5 * (1.0 - u) ** (-1.0 / 0.6)
        lu = np.exp(rng.uniform(np.log(1.5), np.log(4.0 * float(sat_all[s].max())), n))
        peak = np.where(rng.random(n) < 0.7, pl, lu).clip(1.5, 4.0 * float(sat_all[s].max()))
        cols = col_all[s][rng.integers(0, len(col_all[s]), n)]
        # PSF: the session's, rotated/flipped, scaled +-15% in width, shifted sub-pixel
        psf = psf_all[s]
        k = int(rng.integers(4))
        stamp = torch.rot90(psf, k, (1, 2))
        if rng.random() < 0.5:
            stamp = torch.flip(stamp, (2,))
        scale = rng.uniform(0.85, 1.15, n)
        dy, dx = rng.uniform(-0.5, 0.5, (2, n))
        theta = torch.zeros((n, 2, 3), device=dev)
        theta[:, 0, 0] = torch.as_tensor(1.0 / scale, device=dev, dtype=torch.float32)
        theta[:, 1, 1] = theta[:, 0, 0]
        theta[:, 0, 2] = torch.as_tensor(-2.0 * dx / (2 * R + 1), device=dev, dtype=torch.float32)
        theta[:, 1, 2] = torch.as_tensor(-2.0 * dy / (2 * R + 1), device=dev, dtype=torch.float32)
        grid = F.affine_grid(theta, (n, C, 2 * R + 1, 2 * R + 1), align_corners=False)
        st = F.grid_sample(stamp[None].expand(n, -1, -1, -1), grid, mode='bicubic',
                           padding_mode='zeros', align_corners=False).clamp_min(0.0)
        st = st / st.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        amp = torch.as_tensor(peak[:, None] * cols, device=dev, dtype=torch.float32)
        st = st * amp[:, :, None, None]
        ys = rng.integers(0, P, n)
        xs = rng.integers(0, P, n)
        for j in range(n):
            canvas[i, :, ys[j]:ys[j] + 2 * R + 1, xs[j]:xs[j] + 2 * R + 1] += st[j]
    stars = canvas[:, :, R:R + P, R:R + P]
    shot = torch.as_tensor(shot_all[sess], device=dev, dtype=torch.float32)[:, :, None, None]
    var = stars * shot
    noisy = base + stars + torch.randn_like(stars) * torch.sqrt(var)
    sat = torch.as_tensor(sat_all[sess], device=dev, dtype=torch.float32)[:, :, None, None]
    # a star clips where the stack clips; never clip the base's own pixels
    clipped = torch.where(stars > 0, torch.minimum(noisy, torch.maximum(sat, base)), noisy)
    return clipped, stars, var


def train(bank_path: str, steps: int, out: str, batch: int, lr: float, seed: int = 0,
          resume: bool = False) -> None:
    import torch
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    f = np.load(bank_path)
    Z, K, sess = f["z"], f["keep"], f["sess"]
    bank = (torch.as_tensor(f["psf"], device=dev), f["col"], f["sat"], f["shot"])
    n = len(Z)
    rng = np.random.default_rng(seed)
    sessions = np.unique(sess)
    val_sess = rng.choice(sessions, max(1, len(sessions) // 10), replace=False)
    val = np.isin(sess, val_sess)
    tr_idx, va_idx = np.flatnonzero(~val), np.flatnonzero(val)
    print(f"{n} patches from {len(sessions)} sessions, train {len(tr_idx)}, val {len(va_idx)} "
          f"(sessions {list(f['names'][val_sess])}), device {dev}", flush=True)

    net = make_net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)

    def batch_of(idx, gen, aug):
        base = torch.from_numpy(Z[idx].astype(np.float32)).permute(0, 3, 1, 2).to(dev)
        keep = torch.from_numpy(K[idx].astype(np.float32))[:, None].to(dev)
        if aug:
            k = int(gen.integers(4))
            base, keep = torch.rot90(base, k, (2, 3)), torch.rot90(keep, k, (2, 3))
            if gen.random() < 0.5:
                base, keep = torch.flip(base, (3,)), torch.flip(keep, (3,))
        x, stars, var = inject(base, sess[idx], bank, gen, dev)
        return x, base, keep, var

    def loss_of(x, base, keep, var):
        pred = net(torch.asinh(x / nd.ASINH_SOFT)) * torch.sqrt(1.0 + torch.clamp(x, min=0.0))
        w = keep / (1.0 + var)
        return (w * (x + pred - base) ** 2).mean(), (w * (x - base) ** 2).mean()

    vgen_seed = 12345

    def validate():
        net.eval()
        gen = np.random.default_rng(vgen_seed)          # the same injected stars every time
        tot = base_l = 0.0
        with torch.no_grad():
            for s in range(0, len(va_idx), 32):
                ids = va_idx[s:s + 32]
                lo, bl = loss_of(*batch_of(ids, gen, False))
                tot += float(lo) * len(ids)
                base_l += float(bl) * len(ids)
        net.train()
        return tot / len(va_idx), base_l / len(va_idx)

    ckpt = os.path.splitext(out)[0] + "_ckpt.pt"
    start = 1
    if resume and os.path.exists(ckpt):
        st = torch.load(ckpt, weights_only=False)
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        rng.bit_generator.state = st["rng"]
        torch.set_rng_state(st["torch_rng"])
        start = st["step"] + 1
        print(f"resumed from {ckpt} at step {st['step']}", flush=True)

    t0 = time.time()
    run = 0.0
    best = [float("inf"), None]
    for step in range(start, steps + 1):
        lo, _ = loss_of(*batch_of(rng.choice(tr_idx, batch), rng, True))
        opt.zero_grad()
        lo.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        run += lo.item()
        if step % 1000 == 0 or step == steps:
            v, vb = validate()
            if v < best[0]:
                best[0], best[1] = v, {k: t.detach().clone() for k, t in net.state_dict().items()}
            print(f"step {step:6d}  train {run / 1000:.4f}  val {v:.4f}  (identity {vb:.4f})  "
                  f"{time.time() - t0:.0f} s", flush=True)
            run = 0.0
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "step": step},
                       ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
    if best[1] is not None:
        net.load_state_dict(best[1])
        print(f"best validation {best[0]:.4f} kept", flush=True)
    final_val = validate()[0]
    export(net, out)
    err = check_parity(net, out)
    print(f"exported {out}; numpy vs torch max rel diff {err:.2e}")
    if err > 1e-4:
        raise SystemExit("numpy forward pass does not match torch")
    with open(os.path.splitext(out)[0] + "_train.json", "w") as fh:
        json.dump({"steps": steps, "batch": batch, "lr": lr, "patches": int(n), "val": final_val,
                   "sessions": [str(s) for s in f["names"]], "holdout": list(HOLDOUT)}, fh)


# ── evaluation on held-out sessions ─────────────────────────────────────────────

def evaluate(stacks: str, model: str, out_dir: str, holdout_only: bool, seed: int = 7) -> None:
    """Per held-out stack: (1) injected stars into the real stack, run the full runtime
    path (src.nn_starless.remove_stars_nn) and score against the stack itself -- fraction of
    the injected flux removed, and the error left under the injected stars relative to the
    sky noise; (2) away from every star (real or injected), how much the network changed the
    image (it should change nothing: structure kept); (3) a starless JPEG of the real stack
    for visual review."""
    import torch
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = sorted(glob.glob(os.path.join(stacks, '*.fits')))
    if holdout_only:
        paths = [p for p in paths if is_holdout(os.path.basename(p))]
    rows = []
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        img = load_stack(p)
        z, bg, sigma, shot = normalised(img)
        stars = detect(z)
        sat_z = np.nanmax(z.reshape(-1, 3), axis=0).astype(np.float32)
        got = empirical_psf(z, stars, sat_z)
        if got is None:
            print(f"{name}: no PSF", flush=True)
            continue
        psf, _ = got
        real = real_star_mask(z.shape[:2], stars, psf)
        # (1) injected stars on a grid of whole-image tiles
        bank = (torch.as_tensor(psf[None]), star_colours(z, stars)[None], sat_z[None], shot[None])
        h, w, _ = z.shape
        H16, W16 = h - h % PATCH, w - w % PATCH
        zt = torch.from_numpy(np.ascontiguousarray(z[:H16, :W16].transpose(2, 0, 1)))[None]
        tiles = zt.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)          # 1,3,ny,nx,P,P
        ny, nx = tiles.shape[2], tiles.shape[3]
        tb = tiles.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, PATCH, PATCH).contiguous()
        x, inj, _var = inject(tb, np.zeros(len(tb), int), bank, rng, 'cpu', empty_frac=0.0)
        zin = x.reshape(ny, nx, 3, PATCH, PATCH).permute(2, 0, 3, 1, 4).reshape(3, H16, W16).numpy().transpose(1, 2, 0)
        sinj = inj.reshape(ny, nx, 3, PATCH, PATCH).permute(2, 0, 3, 1, 4).reshape(3, H16, W16).numpy().transpose(1, 2, 0)
        img_in = img.copy()
        sig = sigma.astype(np.float32)
        img_in[:H16, :W16] = zin * sig + bg[:H16, :W16]
        starless, _ = ns.remove_stars_nn(img_in, model)
        zs = (starless[:H16, :W16] - bg[:H16, :W16]) / sig
        zb = z[:H16, :W16]
        under = (sinj.max(axis=2) > 3.0) & ~real[:H16, :W16]
        removed = float(np.sum((zin - zs)[under]) / max(np.sum(sinj[under]), 1e-9))
        resid = float(np.sqrt(np.mean((zs - zb)[under] ** 2)))
        resid_id = float(np.sqrt(np.mean((zin - zb)[under] ** 2)))
        clear = ~real[:H16, :W16] & (ndi.maximum_filter(sinj.max(axis=2), 2 * PSF_R + 1) < 0.05)
        changed = float(np.sqrt(np.mean((zs - zin)[clear] ** 2)))
        # (2b) structure: bright non-star signal away from all stars must survive
        struct = clear & (zb.mean(axis=2) > 3.0)
        kept = float(np.sum(zs[struct]) / max(np.sum(zb[struct]), 1e-9)) if struct.any() else float('nan')
        # (3) the real stack, starless
        sl_real, st_real = ns.remove_stars_nn(img, model)
        from src.io_fits import render_preview_float
        for tag, arr in (('input', img), ('starless', sl_real)):
            disp = render_preview_float(arr, stretch='ghs')
            Image.fromarray((np.clip(disp, 0, 1) * 255).astype(np.uint8)).save(
                os.path.join(out_dir, f"{name}_{tag}.jpg"), quality=90)
        row = dict(name=name, injected_flux_removed=removed, rms_under_injected=resid,
                   rms_under_injected_identity=resid_id, rms_change_clear_sky=changed,
                   structure_kept=kept, n_under=int(under.sum()), n_clear=int(clear.sum()))
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(os.path.join(out_dir, "eval.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bank")
    b.add_argument("--stacks", required=True)
    b.add_argument("--bank", required=True)
    b.add_argument("--per-session", type=int, default=200)
    t = sub.add_parser("train")
    t.add_argument("--bank", required=True)
    t.add_argument("--steps", type=int, default=40000)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--out", default=os.path.join(ROOT, "src", "data", "nn_starless.npz"))
    t.add_argument("--resume", action="store_true")
    e = sub.add_parser("eval")
    e.add_argument("--stacks", required=True)
    e.add_argument("--model", default=os.path.join(ROOT, "src", "data", "nn_starless.npz"))
    e.add_argument("--out-dir", required=True)
    e.add_argument("--holdout-only", action="store_true")
    a = ap.parse_args()
    if a.cmd == "bank":
        build_bank(a.stacks, a.bank, a.per_session)
    elif a.cmd == "train":
        train(a.bank, a.steps, a.out, a.batch, a.lr, resume=a.resume)
    else:
        evaluate(a.stacks, a.model, a.out_dir, a.holdout_only)


if __name__ == "__main__":
    main()
