# Learned deconvolution (`--deconvolve nn`): runbook

Status (2026-10-14): **training data half-built, no model yet.** Everything below runs on
any Windows machine with the repo, Python, a Rust toolchain and ideally an NVIDIA GPU.
Branch: `feat/nn-denoiser`.

## The idea, in one paragraph

Train the same small U-Net as `--denoiser nn` (`src/nn_denoise.py`), but on *seeing
pairs*: within each session, the stack of the sharpest 30% of frames (target) against the
stack of the rest (input). Same sky, sharper PSF, disjoint frames so independent noise --
an unbiased Noise2Noise-style target with the session's own real PSF (seeing, tracking,
optics, debayer). No synthetic PSFs, nothing foreign to hallucinate from. Ceiling: it only
learns to sharpen as far as the session's best frames; blur shared by every frame is not
removed. Feasibility (2026-10-14, stars on the same positions, `common_star_fwhm`):
sharp-30% stack vs rest = Omega 0.89, Pinwheel 0.78, Sculptor 0.88, Sunflower 0.88 FWHM.
Existing RL deconvolution (`--deconvolve rl`) masks every star out to 5 FWHM (it rings),
so it never makes stars smaller; a ringing-free learned one would.

## 0. Machine setup

```powershell
git clone https://github.com/hd152/originstack.git
cd originstack
git checkout feat/nn-denoiser
pip install -r requirements.txt -r requirements-dev.txt
pip install maturin
cd ext/astro_native; python -m maturin build --release
pip install --force-reinstall --no-deps target/wheels/astro_native-0.57.0-cp310-abi3-win_amd64.whl
cd ../..
python -c "import astro_native as a; print(a.__version__, hasattr(a, 'nn_conv3x3'))"   # 0.57.0 True
# training only (not a project dependency); CUDA build for an NVIDIA GPU:
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available())"
pytest -q -n auto      # should pass
```

**Power settings first** -- on the first machine, Modern Standby powered the USB data drive
off mid-run twice (one training run lost at step 14000, the data run below aborted):

- Settings -> System -> Power -> sleep when plugged in: **Never**.
- Power Options -> advanced -> USB settings -> USB selective suspend: **Disabled**.
- Device Manager -> USB Root Hubs / the drive's enclosure -> Power Management: untick
  "Allow the computer to turn off this device".
- Keep the data on an **internal** drive if you can, and always write logs and the bank
  to an internal drive (a failed log write killed a run: exit code 120, no traceback).

## 1. Choose the sessions

Origin session folders (each has an `info.json`). The first machine used 83 sessions of
56 targets: >= 45 lights, at most the 2 largest sessions per target, no comets or single
bright stars. Regenerate the list on the new machine (edit `ROOT`):

```python
import os, glob, re, collections
ROOT = "D:/astro/Astrophotography"          # <- where the sessions are
OUT = "sessions.txt"
rows = []
for info in glob.glob(ROOT + "/**/info.json", recursive=True):
    d = os.path.dirname(info).replace("\\", "/"); name = os.path.basename(d)
    if re.search(r"^(C20|3I)|Arcturus|Vega|Mizar|Double_Double|Mosaic|Ptolemy", name):
        continue
    n = sum(1 for f in glob.glob(d + "/**/*.fits", recursive=True)
            if os.path.basename(f).lower().startswith("light"))
    if n >= 45:
        rows.append((re.sub(r"_(Tile\d+_)?\d{4}-\d\d-\d\d.*", "", name), n, d))
by = collections.defaultdict(list)
for t, n, d in rows: by[t].append((n, d))
sel = [d for t, v in sorted(by.items()) for n, d in sorted(v, reverse=True)[:2] if n <= 360]
open(OUT, "w").write("\n".join(sel) + "\n"); print(len(by), "targets,", len(sel), "sessions")
```

Held-out targets (never trained on, used for evaluation; `tools/train_nn_denoise.py`
`HOLDOUT`): Omega Nebula, Sunflower Galaxy, Sculptor Galaxy, Orion Nebula, Pleiades. Keep
them in the list -- the bank builder skips them, the evaluation uses them.

## 2. Build the seeing pairs (~2-4 min per session, ~4 h for 83)

```powershell
python tools/gen_n2n_pairs.py --mode seeing --out D:/nn_deconv/pairs --list sessions.txt > C:/nn_deconv_gen.log 2>&1
```

Per session it runs the normal stacker, measures every aligned frame's FWHM on the same
~60 bright isolated stars, and writes `input.npy` / `target.npy` / `fwhm.npy` /
`meta.json` (~150 MB per session), then aborts the run. **Restartable**: sessions with a
`meta.json` are skipped, so just rerun the same command after a failure. Check
`grep -c ": ok"` and `grep -A15 FAILED` in the log. The first machine finished 7 sessions
(on `G:\output\nn_deconv\pairs`, copy them over to skip them).

## 3. Build the training bank (~30 min, ~5 GB)

```powershell
python tools/train_nn_denoise.py bank --mode deconv --pairs D:/nn_deconv/pairs --bank C:/nn/bank_deconv.npz --per-pair 240
```

`--per-pair 240`: one pair per session (the denoiser had six), so 240 patches each gives
a bank the size of the denoiser's (~17k patches). Half are centred near stars. The loss
weight is 1 / the target's own noise variance (`target_variance`), not the denoiser's
`(A - B)^2`.

## 4. Train (~50-80 min on a GTX 1650 Ti at full power)

```powershell
python tools/train_nn_denoise.py train --bank C:/nn/bank_deconv.npz --steps 30000 --out C:/nn/model_deconv_v1.npz --resume > C:/nn/train_deconv.log 2>&1
```

- Checkpoint every 1000 steps (`model_deconv_v1_ckpt.pt`); after any crash rerun the
  **same command** (same `--steps`) and it resumes.
- Do **not** pass `--inject` (the faint-source augmentation adds the same sources to input
  and target, which is wrong when their PSFs differ).
- Healthy log: `val` well below `identity`, falling; ends with `numpy vs torch max rel diff`
  ~1e-6. The denoiser reached val 1.0145 against identity 1.95; deconvolution numbers are
  on a different scale (target-variance weight), compare runs with each other.
- Watch the GPU: `nvidia-smi --query-gpu=power.draw,clocks.sm,clocks_throttle_reasons.active --format=csv`.
  The 1650 Ti ran 3.5x slower at an 18-20 W cap (battery / quiet profile) than at 38-40 W.

## 5. Evaluate on the held-out pairs

```powershell
python tools/eval_nn_deconv.py --pairs D:/nn_deconv/pairs --model C:/nn/model_deconv_v1.npz --restore 0 4 8 --json C:/nn/eval_deconv_v1.json
```

The held-out pairs must exist: they are in the session list, so step 2 builds them.
Columns: `error` = squared error against the noise-free *sharp* sky in units of the
target's noise variance (lower is better; compare with `none` and `nn_denoise`);
`width/target` = star width vs the sharp stack on the same stars (1.0 = as sharp as the
best frames); `width/input` (< 1 = sharpened). `--restore` tries the low-frequency
restore of `nn_denoise` (it helped the denoiser keep faint light; for deconvolution it may
partly undo the sharpening at > 4 px -- that is what to find out).

**Go / no-go:** worth integrating if `width/input` is clearly below 1 (target 0.85-0.92 on
the full stack) with `error` below `nn_denoise`'s, and the crops (next step) show no
ringing or invented detail. Also run the faint-source injection test idea from the
denoiser work (inject faint blobs, check their flux survives) before shipping.

## 6. Then (not done yet): integrate as `--deconvolve nn`

- `src/cli.py`: add `nn` to `--deconvolve` choices; map to `args.deconvolve_nn`, and do
  not set `args.deconvolve` (that triggers the RL step). The network also denoises (its
  target is a clean sharp sky in expectation), so with `--deconvolve nn` the other
  denoisers and chroma NR should be off, like `--denoiser nn`.
- `src/postprocess.py`: run it at the nn-denoiser step (6.7c) with
  `nn_denoise(stacked, model_path=<src/data/nn_deconv.npz>, restore_sigma=...)`, and set
  `args._preview_noise_scale` the same way.
- Ship the weights as `src/data/nn_deconv.npz` and add them to `packaging/originstack.spec`.
- Changing `--deconvolve`'s help text orphans its translations: re-key `src/locales/*.json`
  (see commit 1e46ecc for how) and run `tests/test_i18n.py`.
- Full-pipeline check: `tools/bench_phase4.py <stacks> --crops --variant rl="--deconvolve rl" --variant nn="--deconvolve nn"`
  on the held-out linear stacks; compare `fwhm_ratio`, `ring`, `sky_noise`, `faint_cnr`.

## Open items from the denoiser (same branch)

- `--denoiser nn` widens stars 1-2.5% on the nebula/cluster fields in `bench_phase4`
  (Omega, Orion, Pleiades); not with `--config nn_chroma_nr=true`. Cause not found.
- A faint ring is still visible around Sunflower's brightest star with `--denoiser nn`.
