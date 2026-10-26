"""Train the Noise2Noise denoiser (``--denoiser nn``, src/nn_denoise.py) on quarter-stacks
from tools/gen_n2n_pairs.py and export its weights to src/data/nn_denoise.npz.

``torch`` is a script-local dependency, not a runtime one: the shipped forward pass is
numpy (src/nn_denoise.forward), which this script checks against torch before exporting.

Pairs: any two quarter-stacks of a session (no frame in common), plus the two halves
(q0+q1 vs q2+q3), so the network sees two depths per session. Inputs are normalised with
src.nn_denoise.normalise exactly as at run time; the target is put through the *input's*
background and sigma. The loss is L2 weighted by 1 / local noise variance (a heavily
smoothed (A - B)^2), so bright stars, whose Poisson noise is large in sky-sigma units, do
not dominate it.

``bank --mode deconv`` builds the learned deconvolver's bank instead, from
``gen_n2n_pairs.py --mode seeing`` (input = stack of all but the sharpest frames, target =
stack of the sharpest): the same sky with a sharper PSF and independent noise. There the
(A - B)^2 weight would be large exactly on the stars, whose profiles differ by design, so
the weight is 1 / the target's own noise variance: sky sigma^2 plus a Poisson term linear
in the smoothed signal, fitted per session (``target_variance``). Half the patches are
centred near stars, which carry most of the PSF information.

``train --synth synth/train.npz`` mixes synthetic-blur pairs (tools/synth_blur.py, ingredients
from tools/gen_synth_deconv.py) into a deconvolution bank's batches: ``--synth-frac`` of each
batch is drawn on the fly with a truth sharper than any real stack, so the network can learn
to sharpen past the seeing pairs' sharp-30% target. The weights kept are the best on the sum
of real and synthetic validation loss, each relative to its identity loss. ``--init``
starts from exported weights (e.g. the shipped v3). ``--peak-penalty`` adds a one-sided
cost on point-like excess over the synthetic truth (``peak_penalty``).

Usage:
    pip install torch            (into a separate venv: it is not a project dependency)
    python tools/train_nn_denoise.py bank  --pairs G:/output/nn_denoise/pairs --bank bank.npz
    python tools/train_nn_denoise.py train --bank bank.npz --steps 30000
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Optional

import numpy as np
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import nn_denoise as nd  # noqa: E402

PATCH = 128
VAR_BLUR = 4.0
# Held out of training entirely, for validation and for the benchmark sessions.
HOLDOUT = ("Omega_Nebula", "Sunflower_Galaxy", "Sculptor_Galaxy", "Orion_Nebula", "Pleiades")


def is_holdout(name: str) -> bool:
    return any(name.startswith(h) for h in HOLDOUT)


def session_pairs(d: str):
    """(input, target) image pairs of one session, as (H, W, 3) float32."""
    q = [np.load(os.path.join(d, f"q{i}.npy")) for i in range(4)]
    h01, h23 = (q[0] + q[1]) / 2, (q[2] + q[3]) / 2
    return [(q[0], q[1]), (q[1], q[0]), (q[2], q[3]), (q[3], q[2]), (h01, h23), (h23, h01)]


def normalised_pair(a: np.ndarray, b: np.ndarray):
    """(zA, zB, weight) for a whole image pair, B through A's normalisation."""
    grid, sigma = nd.normalise(a)
    bg = nd.background_at(grid, 0, 0, *a.shape[:2])
    za = (a - bg) / sigma.astype(np.float32)
    zb = (b - bg) / sigma.astype(np.float32)
    var = np.stack([ndi.gaussian_filter((za[..., c] - zb[..., c]) ** 2, VAR_BLUR) for c in range(3)], -1)
    return za, zb, (2.0 / np.maximum(var, 1e-3)).astype(np.float32)


def target_variance(z: np.ndarray, sky_level: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-pixel noise variance of a stack in noise units, ``a * (1 + max(signal, 0) / B)``:
    sky-limited Poisson noise grows with the light, at the rate the sky level B sets.

    ``a`` is measured directly (MAD of lag-4 row differences on the faintest 30% of the
    smoothed image -- lag 4 skips the debayer/warp correlation). ``sky_level`` is B per
    channel in the same noise units. Two fits were tried and dropped: the intercept of a
    variance-vs-signal line fell to its 1e-3 floor on several sessions (1000x sky weight,
    loss in the thousands), and its slope, from lag-4 differences on stars and nebulae,
    measured their structure rather than their noise (normalised noise 0.14-12 on the
    brightest pixels). Without ``sky_level`` (or B <= 0) the variance is ``a`` alone."""
    out = np.empty_like(z)
    for ch in range(z.shape[2]):
        x = z[..., ch]
        sm = ndi.gaussian_filter(x, 2.0)
        d = ((x[:, 4:] - x[:, :-4]) / np.sqrt(2.0)).ravel()
        sv = sm[:, 2:-2].ravel()
        sky = d[sv < np.quantile(sv, 0.3)]
        a = max((1.4826 * np.median(np.abs(sky - np.median(sky)))) ** 2, 1e-6)
        bl = float(sky_level[ch]) if sky_level is not None else 0.0
        out[..., ch] = a * (1.0 + np.maximum(sm, 0.0) / bl) if bl > 0 else a
    return out


def sky_level_z(bg: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """The sky level B per channel in noise units (median background / sigma)."""
    return np.median(bg.reshape(-1, bg.shape[-1]), axis=0) / sigma


SKY_MATCH_SIGMA = 32.0     # px; far above a star's scale, so its sharpening survives


def deconv_pair(a: np.ndarray, b: np.ndarray):
    """(zA, zB, weight) for a seeing pair, B through A's normalisation, weight 1 / B's
    noise variance.

    B's large-scale difference from A (Gaussian, ``SKY_MATCH_SIGMA``) is removed first:
    the sharpest frames are a time-biased subset (seeing follows altitude), so their sky
    level differs, and v2 learned that shift -- held-out sky error 3-6x that of no
    processing until the low-frequency restore put the input's background back. A
    sharper star keeps its flux, so its difference smooths to ~0 at that scale."""
    b = b - np.stack([ndi.gaussian_filter(b[..., c] - a[..., c], SKY_MATCH_SIGMA) for c in range(3)], -1)
    grid, sigma = nd.normalise(a)
    bg = nd.background_at(grid, 0, 0, *a.shape[:2])
    za = (a - bg) / sigma.astype(np.float32)
    zb = (b - bg) / sigma.astype(np.float32)
    return za, zb, (1.0 / target_variance(zb, sky_level_z(bg, sigma))).astype(np.float32)


def build_bank(pairs_dirs, out: str, per_pair, seed: int = 0, mode: str = "denoise") -> None:
    """Patch bank from one pairs folder or several (e.g. seeing pairs and the cross-night
    pairs of tools/gen_crossnight_pairs.py, which are in the same format). ``per_pair`` is
    one count for every folder or one per folder."""
    rng = np.random.default_rng(seed)
    za_l, zb_l, w_l, sess = [], [], [], []
    if isinstance(pairs_dirs, str):
        pairs_dirs = [pairs_dirs]
    counts = [per_pair] * len(pairs_dirs) if isinstance(per_pair, int) else list(per_pair)
    if len(counts) == 1:
        counts *= len(pairs_dirs)
    if len(counts) != len(pairs_dirs):
        raise SystemExit("--per-pair: give one count, or one per --pairs folder")
    dirs, n_of = [], {}
    for p, c in zip(pairs_dirs, counts):
        for d in sorted(glob.glob(os.path.join(p, "*"))):
            if os.path.exists(os.path.join(d, "meta.json")):
                dirs.append(d)
                n_of[d] = c
    for i, d in enumerate(dirs):
        name = os.path.basename(d)
        if is_holdout(name):
            continue
        t = time.time()
        if mode == "deconv":
            with open(os.path.join(d, "meta.json")) as fh:
                meta = json.load(fh)
            if min(meta.get("n_target", 0), meta.get("n_input", 0)) < 3:
                print(f"[{i + 1}/{len(dirs)}] {name}: skipped, no seeing split "
                      f"({meta.get('n_target')} / {meta.get('n_input')} frames)", flush=True)
                continue
            pairs = [(np.load(os.path.join(d, "input.npy")), np.load(os.path.join(d, "target.npy")))]
        else:
            pairs = session_pairs(d)
        for a, b in pairs:
            za, zb, w = deconv_pair(a, b) if mode == "deconv" else normalised_pair(a, b)
            h, wd, _ = za.shape
            m = 16                      # skip the stack's ragged edge
            stars = None
            if mode == "deconv":
                from src.star_detect import detect_stars_matched_filter as detect
                lum = za.mean(-1)
                st = detect(lum - np.median(lum))
                stars = np.c_[np.asarray(st["ycentroid"]), np.asarray(st["xcentroid"])]
            for k in range(n_of[d]):
                if stars is not None and len(stars) and k % 2 == 0:
                    cy, cx = stars[rng.integers(len(stars))]
                    y = int(np.clip(cy - rng.integers(16, PATCH - 16), m, h - PATCH - m))
                    x = int(np.clip(cx - rng.integers(16, PATCH - 16), m, wd - PATCH - m))
                else:
                    y = rng.integers(m, h - PATCH - m)
                    x = rng.integers(m, wd - PATCH - m)
                za_l.append(za[y:y + PATCH, x:x + PATCH].astype(np.float16))
                zb_l.append(zb[y:y + PATCH, x:x + PATCH].astype(np.float16))
                w_l.append(w[y:y + PATCH, x:x + PATCH].astype(np.float16))
                sess.append(i)
        print(f"[{i + 1}/{len(dirs)}] {name}: {time.time() - t:.0f} s", flush=True)
    np.savez(out, za=np.stack(za_l), zb=np.stack(zb_l), w=np.stack(w_l), sess=np.array(sess))
    print(f"bank: {len(za_l)} patches -> {out}")


def make_net(filters: int = 24):
    import torch.nn as nn

    def conv(i, o):
        return nn.Conv2d(i, o, 3, padding=1)

    class NnDenoiseNet(nn.Module):
        """Mirrors src/nn_denoise.forward layer for layer (names become the npz keys)."""

        def __init__(self, c=3, f=filters):
            super().__init__()
            self.e0a, self.e0b = conv(c, f), conv(f, f)
            self.e1a, self.e1b = conv(f, 2 * f), conv(2 * f, 2 * f)
            self.ma, self.mb = conv(2 * f, 4 * f), conv(4 * f, 4 * f)
            self.d1a, self.d1b = conv(4 * f + 2 * f, 2 * f), conv(2 * f, 2 * f)
            self.d0a, self.d0b = conv(2 * f + f, f), conv(f, f)
            self.out = nn.Conv2d(f, c, 1)
            self.act = nn.LeakyReLU(nd.LEAK)
            self.pool = nn.AvgPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        def forward(self, x):
            import torch
            a = self.act
            s0 = a(self.e0b(a(self.e0a(x))))
            s1 = a(self.e1b(a(self.e1a(self.pool(s0)))))
            m = a(self.mb(a(self.ma(self.pool(s1)))))
            d1 = a(self.d1b(a(self.d1a(torch.cat([self.up(m), s1], 1)))))
            d0 = a(self.d0b(a(self.d0a(torch.cat([self.up(d1), s0], 1)))))
            return self.out(d0)

    return NnDenoiseNet()


def export(net, path: str) -> None:
    np.savez(path, **{k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()})


def check_parity(net, path: str) -> float:
    """Max |torch - numpy| on a random input; the shipped path must match the trained one."""
    import torch
    x = np.random.default_rng(1).normal(0, 1, (3, 64, 96)).astype(np.float32)
    with torch.no_grad():
        ref = net.cpu().eval()(torch.from_numpy(x)[None])[0].numpy()
    nd._weights_cache.pop(path, None)
    got = nd.forward(x, nd.load_weights(path))
    return float(np.abs(ref - got).max() / (np.abs(ref).max() + 1e-12))


def faint_sources(n: int, size: int, gen, dev):
    """(n, 3, size, size) random faint sources in noise units, the same ones to be added
    to input and target: elliptical Gaussians and exponential discs, 0.8-8 px scale,
    peak 0.3-10 sigma (log-uniform), colour varying +-30% per channel, ~3 per patch.

    Without them the network learns that faint isolated blobs are rare in real
    patches, and shrinking them is the minimum-MSE answer: injected blobs of 0.5-1.25
    sky sigma kept only 48-84% of their flux (v1). Added to both images, they leave the
    Noise2Noise target unbiased and only change the prior."""
    import torch
    out = torch.zeros((n, 3, size, size), device=dev)
    yy, xx = torch.meshgrid(torch.arange(size, device=dev, dtype=torch.float32),
                            torch.arange(size, device=dev, dtype=torch.float32), indexing="ij")
    counts = gen.poisson(3.0, n)
    for i in range(n):
        for _ in range(int(counts[i])):
            cy, cx = gen.uniform(0, size, 2)
            sa = gen.uniform(0.8, 8.0)
            sb = sa * gen.uniform(0.4, 1.0)
            th = gen.uniform(0, np.pi)
            dy, dx = yy - float(cy), xx - float(cx)
            u = (dx * np.cos(th) + dy * np.sin(th)) / sa
            v = (-dx * np.sin(th) + dy * np.cos(th)) / sb
            r2 = u * u + v * v
            prof = torch.exp(-0.5 * r2) if gen.random() < 0.5 else torch.exp(-torch.sqrt(r2) * 1.6783)
            peak = float(np.exp(gen.uniform(np.log(0.3), np.log(10.0))))
            col = np.clip(1.0 + 0.3 * gen.standard_normal(3), 0.3, 2.0)
            out[i] += torch.tensor(peak * col, device=dev, dtype=torch.float32)[:, None, None] * prof
    return out


PEAK_MASK_SIGMA = 0.3      # truth point-filter response (noise units) that marks a real source
PEAK_MASK_GROW = 3         # px, dilation of that mask (7 covered 63% of the canvas)


def peak_penalty(out, truth, w, masked: bool = True):
    """One-sided cost of point-like excess of ``out`` over a noiseless ``truth``: the
    luminance residual through a difference of Gaussians (sigma 1 px minus 3 px, zero sum),
    positive part squared, weighted by ``w``. L2 alone is symmetric; on synthetic pairs s1/s2
    turned ~2 sigma noise bumps into point sources (unconfirmed faint peaks about twice v3's).

    ``masked`` charges it only where the truth has no source (its own filter response below
    ``PEAK_MASK_SIGMA``, grown by ``PEAK_MASK_GROW``). Unmasked (s3a/s3b, weight 10 / 30) it
    also charged real stars' peaks and the network dimmed them: synthetic flux 0.87 / 0.80
    against 0.94, invented peaks 52 / 65 per Mpx against 62."""
    import torch
    import torch.nn.functional as F
    k = torch.arange(-6, 7, dtype=torch.float32, device=out.device)
    g = lambda s: torch.exp(-0.5 * (k / s) ** 2) / torch.exp(-0.5 * (k / s) ** 2).sum()  # noqa: E731
    dog = g(1.0)[:, None] * g(1.0)[None, :] - g(3.0)[:, None] * g(3.0)[None, :]
    r = (out - truth).mean(1, keepdim=True)
    mf = F.conv2d(r, dog[None, None], padding=6)
    cost = w.mean(1, keepdim=True) * torch.relu(mf) ** 2
    if masked:
        src = F.conv2d(truth.mean(1, keepdim=True), dog[None, None], padding=6) > PEAK_MASK_SIGMA
        g = PEAK_MASK_GROW
        near = F.max_pool2d(src.float(), 2 * g + 1, stride=1, padding=g) > 0
        cost = cost * (~near).float()
    return cost.mean()


def train(bank: str, steps: int, out: str, batch: int, lr: float, seed: int = 0, resume: bool = False,
          inject: bool = False, synth: Optional[str] = None, synth_frac: float = 0.5,
          init: Optional[str] = None, synth_set: Optional[dict] = None, peak_pen: float = 0.0,
          filters: int = 24) -> None:
    import torch
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    f = np.load(bank)
    za, zb, w, sess = f["za"], f["zb"], f["w"], f["sess"]
    n = len(za)
    # hold out ~5% of sessions' patches for a validation loss (holdout targets are not in the bank at all)
    rng = np.random.default_rng(seed)
    val_sess = rng.choice(np.unique(sess), max(1, len(np.unique(sess)) // 20), replace=False)
    val = np.isin(sess, val_sess)
    tr_idx, va_idx = np.flatnonzero(~val), np.flatnonzero(val)
    print(f"{n} patches, train {len(tr_idx)}, val {len(va_idx)}, device {dev}", flush=True)

    net = make_net(filters).to(dev)
    if init:
        with np.load(init) as fw:
            net.load_state_dict({k: torch.from_numpy(fw[k]) for k in fw.files})
        print(f"initialised from {init}", flush=True)
    syn = n_syn = syn_val = None
    if synth:
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import synth_blur
        from synth_blur import SynthSource
        for k, v in (synth_set or {}).items():
            if not hasattr(synth_blur, k):
                raise SystemExit(f"--synth-set: synth_blur has no {k}")
            setattr(synth_blur, k, v)
            print(f"synth_blur.{k} = {v}", flush=True)
        syn = SynthSource(synth, dev)
        n_syn = int(round(batch * synth_frac))
        vg = np.random.default_rng(seed + 123)
        syn_val = [tuple(t.cpu() for t in syn.batch(32, vg)) for _ in range(16)]
        print(f"synthetic: {len(syn.bg)} backgrounds, {len(syn.noise)} noise patches, "
              f"{n_syn} of {batch} per batch", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)

    def batch_of(idx, aug):
        a, b, ww = (torch.from_numpy(arr[idx].astype(np.float32)).permute(0, 3, 1, 2).to(dev) for arr in (za, zb, w))
        if aug:
            k = int(rng.integers(4))
            a, b, ww = (torch.rot90(t, k, (2, 3)) for t in (a, b, ww))
            if rng.random() < 0.5:
                a, b, ww = (torch.flip(t, (3,)) for t in (a, b, ww))
            if inject:
                src = faint_sources(a.shape[0], a.shape[2], rng, dev)
                a, b = a + src, b + src
        return a, b, ww

    def loss_of(a, b, ww, n_tail=0):
        """(loss, identity loss); with ``n_tail`` the last n samples are synthetic and also
        pay the peak penalty."""
        pred = net(torch.asinh(a / nd.ASINH_SOFT)) * torch.sqrt(1.0 + torch.clamp(a, min=0.0))
        lo = (ww * (a + pred - b) ** 2).mean()
        if peak_pen > 0 and n_tail:
            lo = lo + peak_pen * peak_penalty((a + pred)[-n_tail:], b[-n_tail:], ww[-n_tail:])
        return lo, (ww * (a - b) ** 2).mean()

    def validate():
        net.eval()
        tot = base = 0.0
        with torch.no_grad():
            for s in range(0, len(va_idx), 64):
                lo, bl = loss_of(*batch_of(va_idx[s:s + 64], False))
                tot += float(lo) * len(va_idx[s:s + 64])
                base += float(bl) * len(va_idx[s:s + 64])
        net.train()
        return tot / len(va_idx), base / len(va_idx)

    def validate_synth():
        net.eval()
        with torch.no_grad():
            r = [loss_of(*(t.to(dev) for t in b), n_tail=len(b[0])) for b in syn_val]
        net.train()
        return float(np.mean([float(a) for a, _ in r])), float(np.mean([float(b) for _, b in r]))

    # a checkpoint every 1000 steps: a run on this machine was lost at step 14000 to an I/O
    # error on the drive its log was on
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
        if syn is None:
            lo, _ = loss_of(*batch_of(rng.choice(tr_idx, batch), True))
        else:
            a, b, ww = batch_of(rng.choice(tr_idx, batch - n_syn), True)
            sa, sb, sw = syn.batch(n_syn, rng)
            lo, _ = loss_of(torch.cat([a, sa]), torch.cat([b, sb]), torch.cat([ww, sw]), n_tail=n_syn)
        opt.zero_grad()
        lo.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        run += lo.item()
        if step % 1000 == 0 or step == steps:
            v, vb = validate()
            sel, extra = v, ""
            if syn is not None:
                sv, svb = validate_synth()
                sel = v / vb + sv / svb
                extra = f"  synth val {sv:.4f} (identity {svb:.4f})"
            if sel < best[0]:
                # keep the best-validation weights: deconvolution v2 was best at step
                # 10000 and overfit after (val 1.996 -> 2.215 while train kept falling)
                best[0], best[1] = sel, {k: t.detach().clone() for k, t in net.state_dict().items()}
            print(f"step {step:6d}  train {run / 1000:.4f}  val {v:.4f}  (identity {vb:.4f}){extra}  "
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
                   "inject": inject, "synth": synth, "synth_frac": synth_frac if synth else None,
                   "init": init, "synth_set": synth_set, "peak_penalty": peak_pen, "filters": filters}, fh)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bank")
    b.add_argument("--pairs", required=True, nargs="+", help="one or more pairs folders")
    b.add_argument("--bank", required=True)
    b.add_argument("--per-pair", type=int, nargs="+", default=[40],
                   help="patches per pair: one count, or one per --pairs folder")
    b.add_argument("--mode", choices=["denoise", "deconv"], default="denoise")
    t = sub.add_parser("train")
    t.add_argument("--bank", required=True)
    t.add_argument("--steps", type=int, default=30000)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--out", default=os.path.join(ROOT, "src", "data", "nn_denoise.npz"))
    t.add_argument("--resume", action="store_true", help="continue from <out>_ckpt.pt")
    t.add_argument("--inject", action="store_true",
                   help="add random faint sources to training patches (faint_sources)")
    t.add_argument("--synth", help="gen_synth_deconv.py npz: mix synthetic-blur pairs into each batch")
    t.add_argument("--synth-frac", type=float, default=0.5, help="share of each batch that is synthetic")
    t.add_argument("--init", help="start from these exported weights (.npz)")
    t.add_argument("--filters", type=int, default=24,
                   help="U-Net base width (the numpy forward pass takes any; run time grows ~ its square)")
    t.add_argument("--peak-penalty", type=float, default=0.0,
                   help="weight of the synthetic-batch peak penalty (peak_penalty)")
    t.add_argument("--synth-set", action="append", default=[], metavar="NAME=VALUE",
                   help="override a synth_blur.py constant, e.g. FLUX=20,5e5 (repeatable)")
    args = ap.parse_args()
    if args.cmd == "bank":
        build_bank(args.pairs, args.bank, args.per_pair, mode=args.mode)
    else:
        train(args.bank, args.steps, args.out, args.batch, args.lr, resume=args.resume, inject=args.inject,
              synth=args.synth, synth_frac=args.synth_frac, init=args.init, peak_pen=args.peak_penalty,
              filters=args.filters,
              synth_set={k: tuple(float(x) for x in v.split(",")) if "," in v else float(v)
                         for k, v in (kv.split("=", 1) for kv in args.synth_set)})


if __name__ == "__main__":
    main()
