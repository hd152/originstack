# Learned deconvolution (`--deconvolve nn`): runbook

Status (2026-10-16): **v3 trained, passes every step-5 check, and is integrated as
`--deconvolve nn` (step 6) with `restore_sigma=4`.** See "v3 results" and step 6 below. Everything below runs on
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

## Local copy on the first machine (`C:\nn_work`, made 2026-10-14)

Everything the run needs, off the USB drive:

| Path | Contents |
|---|---|
| `C:\nn_work\sessions\<target>\<session>` | the 83 session folders (lights, calibration, `info.json`), 137 GB |
| `C:\nn_work\sessions.txt` | the session list with these local paths -- use it in step 2 (skip step 1) |
| `C:\nn_work\nn_deconv\pairs` | the seeing pairs finished so far (skipped on rerun) |
| `C:\nn_work\heldout_stacks` | linear stacks + `_config.toml` of the 5 held-out targets, for `bench_phase4.py` |
| `C:\nn_work\denoiser` | `--denoiser nn` v1/v2 weights, checkpoints, training logs, evaluations, its 5 GB bank |

| `C:\nn_work\venv` | training Python: base Python's torch (CUDA) + astro_native 0.57.0, nothing on G: |

Nothing below needs the USB drive. On that machine (repo at `C:\source\originstack`):

```powershell
python tools/gen_n2n_pairs.py --mode seeing --out C:/nn_work/nn_deconv/pairs --list C:/nn_work/sessions.txt
C:/nn_work/venv/Scripts/python tools/train_nn_denoise.py bank --mode deconv --pairs C:/nn_work/nn_deconv/pairs --bank C:/nn_work/bank_deconv.npz --per-pair 240
C:/nn_work/venv/Scripts/python tools/train_nn_denoise.py train --bank C:/nn_work/bank_deconv.npz --steps 30000 --out C:/nn_work/model_deconv_v1.npz --resume > C:/nn_work/train_deconv.log 2>&1
python tools/eval_nn_deconv.py --pairs C:/nn_work/nn_deconv/pairs --model C:/nn_work/model_deconv_v1.npz --restore 0 4 8 --json C:/nn_work/eval_deconv_v1.json
```
To use another machine, copy `C:\nn_work` across and regenerate `sessions.txt` with the
snippet below (`ROOT` = the copied `sessions` folder).

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

Then the checks the error score does not cover -- faint sources, ringing, invented sources --
and crops to look at:

```powershell
python tools/eval_nn_faint.py --pairs C:/nn_work/nn_deconv/pairs --model C:/nn_work/model_deconv_v3.npz --json C:/nn_work/faint_v3.json --crops C:/nn_work/crops_v3
```

## v3 results (2026-10-16, second machine: RTX 3050 Laptop 4 GB, `C:\nn_work`)

Trained after `57fbe10` (target sky matched to the input, best-validation weights): 83
sessions / 56 targets, pairs ~5 h, bank 17,760 patches in 7 min, 30,000 steps in 3.2 h.
Validation 1.443 at step 1000, best **1.2248 at step 23,000** (identity 1.8223), 1.22-1.26
from step 15,000 on -- no overfitting (v2: 1.996 -> 2.215). Exported numpy vs torch max rel
diff 5.3e-7. Weights: `C:\nn_work\model_deconv_v3.npz` (not shipped).

`eval_nn_deconv.py`, 9 held-out pairs (Omega x2, Orion x2, Pleiades x2, Sculptor, Sunflower x2):

| | width / input | width / target | error vs `nn_denoise` | sky error |
|---|---|---|---|---|
| r0 | 0.82-0.92 | 0.96-1.09 | lower on 8/9 (Pleiades 2025-10: 0.554 vs 0.481) | within 0.01 of nn_denoise's on 8/9 |
| r4 | 0.81-0.92 | 0.96-1.08 | lower on 9/9 | within 0.01 of nn_denoise's on 8/9 |

The exception for sky error is Orion 2025-12-11 (0.07-0.08 vs 0.06); every method is far
below no processing (0.32-0.49). v2's sky-level learning is gone (its held-out sky error was
3-6x no processing). `eval_nn_faint.py` on the same 9 pairs (1024x1536 centre crops):

- **Faint sources: use r4.** Neutral exponential discs (1.5 / 3 px) at 0.5-1 sky sigma keep
  35-83% of their flux at r0 (worse than `nn_denoise`'s 79-102%), **81-99% at r4**, matching
  the denoiser on every session; PSF-shaped point sources 79-107% at r4 (weakest: Pleiades
  2025-10 at 1-2 sigma). The restore costs no sharpness (width/input r4 = r0 within 0.007).
  Both Omega pairs have only 2 free sky positions (the nebula fills the frame): little weight.
- **Ringing: none.** Deepest dip of the bright, isolated, unsaturated stars' radial profile
  below the local sky, 1.5-6 FWHM: 0.00 to -0.02 sky sigma for every method (the input
  itself: 0 to -0.17).
- **No invented sources.** The crops show faint point sources on the sky that are hard to see
  in the input. Faint peaks of each output checked against the independent sharp stack (its
  PSF-matched filter > 2.5 sigma within 2 px): 94-100% confirmed for the deconvolver, against
  3-21% for random sky positions. It shows 20-60% more real faint peaks above 2 sigma than
  `nn_denoise` (Sunflower 2026-03: 97 vs 72 per Mpx; a matched filter on the input: 51).
- Crops (`C:\nn_work\crops_v3`): stars visibly tighter, no rings or blobs; nebula interiors
  (Omega) unchanged in structure; a faint irregular skirt around the Pleiades' brightest
  stars, not a dip.

## 6. Integrated as `--deconvolve nn` (2026-10-16)

Done as planned below: `deconvolve_nn` (with `nn_deconv_restore_sigma` 4), the nn step
(6.7c) runs `src/data/nn_deconv.npz`, chroma NR and every denoiser are off (marked explicit
so presets/`--auto` cannot re-enable them; `--denoiser nn` is replaced), the weights are in
the packaging spec, the help is re-keyed in all locales, `tests/test_nn_denoise.py` covers
CLI, config round trip, the Phase 4 call and the real model. On ideal Gaussian stars the
network narrows them only 2-5% (1.6 px sigma: 2%; 1.2 px: 5%), flux within 1%: it learned
the real seeing blur of Origin sessions, so do not assume the 8-18% on other optics.

Full pipeline, `bench_phase4.py` on one fresh stack of each held-out target (stacked with
this branch; `C:\nn_work\heldout_stacks`, results `C:\nn_work\bench_v3.json`, crops and
side-by-sides `C:\nn_work\bench_v3\cmp_*.jpg`). `fwhm_ratio` = Phase 4 output / linear,
same stars; `sky_noise` display 8-bit; `ring` in each variant's own sky-noise units, so
multiply by `sky_noise` to compare (positive = bright halo, negative = dark moat):

| target | fwhm base / rl / nn_denoise / **nn_deconv** | sky_noise base / nn_deconv | ring x sky_noise base / rl / nn_deconv |
|---|---|---|---|
| Omega 2026-08 | 1.064 / 1.038 / 1.065 / **0.924** | 0.41 / 0.12 | 0.29 / 0.40 / 0.15 |
| Orion 2025-12-27 | 1.061 / 1.074 / 1.072 / **0.955** | 0.07 / 0.02 | 0.08 / **-1.42** / 0.06 |
| Pleiades 2025-11 | 1.068 / 1.034 / 1.088 / **0.988** | 0.48 / 0.09 | 0.11 / 0.05 / 0.02 |
| Sculptor | 0.880 / 0.799 / 0.860 / **0.752** | 0.67 / 0.04 | 0.79 / 0.06 / 0.80 |
| Sunflower 2026-03 | 0.882 / 0.822 / 0.878 / **0.739** | 0.74 / 0.05 | 0.30 / 0.08 / 0.48 |

The narrowest stars on all five, 6-17% narrower than the default chain, at the nn denoiser's
sky noise (5-15x below the default's); RL's dark moats (Orion) and sky speckle (Sunflower)
do not appear. `faint_cnr` is 10-20% below `--denoiser nn`'s and far above the default's.
Like `--denoiser nn`, hue error on bright pixels is ~5 deg (default 2-3 deg) and Pleiades'
display star-colour agreement with Gaia is a little lower (rho 0.61 vs 0.70).

Open: the bright halo ring around Sunflower's brightest star (already listed for
`--denoiser nn`, also faintly in the default chain) is a little more distinct; the clean
sky shows it more. Not a dark moat.

The original plan:

- Use `restore_sigma=4` (above): r0 loses up to two thirds of 0.5-1 sigma faint sources.

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
