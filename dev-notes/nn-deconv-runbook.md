# Learned deconvolution (`--deconvolve nn`): runbook

Status (2026-10-25): **v4, trained against Pan-STARRS survey truth, replaces v3**, blended with
the nn denoiser. `--auto` uses it on galaxies (35% v4; stars ~15% narrower than v3 at equal or
better faint contrast) and now on nebulae, clusters and star fields too (75%; stars 10-35%
narrower than with `--denoiser nn`, faint contrast +10-20%). See "Survey truth: v4" at the end.

Status (2026-10-21): synthetic-blur training passed its feasibility gate, but its best model (s4)
is only 1-3% sharper than v3 in the full pipeline; v3 stays shipped (see
"Synthetic-blur training" below).

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

The narrowest stars on all five, 7-16% narrower than the default chain, at about the nn
denoiser's sky noise (3.5-19x below the default's); RL's dark moats (Orion) and sky speckle (Sunflower)
do not appear. `faint_cnr` is 1-20% below `--denoiser nn`'s and far above the default's.
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

## Synthetic-blur training: feasibility run (2026-10-21) -- passes the gate, not shipped

Branch `feat/nn-deconv-synthetic`. Model `s1` = v3 fine-tuned 10,000 steps, half of each batch
synthetic (GTX 1650 Ti, 19 min). Weights `C:\nn_work\model_deconv_s1.npz` (not shipped);
logs and evaluations in `C:\nn_work\synth`.

What was built (it differs from the plan below where noted):
- `tools/gen_synth_deconv.py`: per session, in the input's noise units -- clean starless
  backgrounds (sharp stack, bundled denoiser without restore, `star_removal` inpaint), real
  stack noise (input minus its denoised self, empty-sky 128 px patches; lag-1 correlation
  0.41, so the debayer/warp correlation is kept -- no synthetic noise model needed), sky level
  B. 83 sessions in ~50 min: 4440 / 2960 train, 540 / 360 held-out patches (1.4 GB).
- `tools/synth_blur.py`: pairs drawn on the GPU during training (0.05 s per 8). Truth = bg +
  stars with a Moffat PSF of FWHM 1.5-2.3 px; **random positions and power-law fluxes, not
  Gaia** (the seeing pairs have no WCS). Input = truth convolved with K exactly on the pixel
  grid (Moffat FWHM 1.5-4.5, beta 2.5-6, axis ratio 0.75-1, coma component in half the
  kernels with its centroid kept at zero, +-6% width per channel) + real noise x
  sqrt(1 + signal/B). Weight 1/(1 + signal/B).
- `train_nn_denoise.py train --synth ... --synth-frac 0.5 --init src/data/nn_deconv.npz`;
  best weights by real val/identity + synthetic val/identity.
- `tools/eval_synth_deconv.py` (exact truth) and `tools/eval_gaia_pairs.py` (Gaia close
  pairs and single-star splits on the held-out linear stacks; WCS refined first --
  Pleiades' header is off by (3.9, -7.9) px; only pairs whose fainter star is >= 15 sigma in
  the input, or noise alone makes the input "resolve" more than a denoised output).

```powershell
python tools/gen_synth_deconv.py --pairs C:/nn_work/nn_deconv/pairs --out C:/nn_work/synth
C:/nn_work/venv/Scripts/python tools/train_nn_denoise.py train --bank C:/nn_work/bank_deconv.npz --synth C:/nn_work/synth/train.npz --synth-frac 0.5 --init src/data/nn_deconv.npz --steps 10000 --lr 3e-4 --out C:/nn_work/model_deconv_s1.npz --resume
C:/nn_work/venv/Scripts/python tools/eval_synth_deconv.py --synth C:/nn_work/synth/holdout.npz --model v3=src/data/nn_deconv.npz --model s1=C:/nn_work/model_deconv_s1.npz
python tools/eval_gaia_pairs.py C:/nn_work/heldout_stacks/*.fits --model v3=src/data/nn_deconv.npz --model s1=C:/nn_work/model_deconv_s1.npz
```

Training: real val 1.2952 (identity 1.8722; v3's own best 1.2248), synthetic val 2.79
(identity 27.64).

Results (restore 4 px throughout):

| check | v3 | s1 |
|---|---|---|
| synthetic held-out: width / truth, / input | 1.80, 0.90 | **1.11, 0.56** |
| synthetic: flux, signal error, invented peaks / Mpx | 0.96, 1058, 35 | 0.94, **128**, 58 (input 122, nn_denoise 12) |
| real held-out pairs: width / input | 0.81-0.92 | **0.74-0.90** |
| real: width / sharp stack | 0.96-1.08 | **0.88-1.07** (< 1 on 8/9) |
| real: error vs `nn_denoise` | lower 9/9 | lower 8/9, equal on Pleiades 2025-10 (v3 was worse there) |
| Gaia: single stars split | 0 | **0** (53-149 tested per target) |
| Gaia: pairs 3.5-4.5 px resolved | Omega 20%, Orion 0/3 | Omega 30%, Orion 1/3 (separation 0.83x Gaia's) |
| ring (eval_nn_faint) | 0.00 to -0.01 | 0.00 to -0.01 |
| faint discs 1.5 px, 0.5-1 sigma | 0.81-0.96 | 0.84-0.99 (>= v3 on every session) |
| faint peaks > 2 sigma confirmed by the sharp stack | 94-100% | 87-98% |
| faint peaks > 3 sigma confirmed | 99-100% | 96-99% |

Gate (below) met: held-out stars narrow clearly past the sharp-stack target and nothing
splits in the Gaia check. Below 3.5 px no method separates a pair, so the pair half of the
Gaia check is weak (41 usable pairs, mostly Omega); the single-star split test carries it.

Open before a full version / shipping:
- **Unconfirmed faint peaks roughly double** (> 2 sigma: Pleiades 2025-11 87% confirmed vs
  98%, ~12 vs ~1.5 per Mpx; Omega 2026-08 ~36 vs ~17). Most of s1's extra peaks are real
  (random sky confirms 4-17%), but this is the invented-detail risk the plan named. Same
  signal on synthetic fields (58 vs 35 / Mpx). Candidates: lower the synthetic star density
  floor, add faint-source-free synthetic patches, a peak penalty, or a smaller K range.
- Synthetic flux 0.94 (v3 0.96); not seen on real data (`eval_nn_faint` point sources
  0.94-1.06).
- Real val 1.295 vs v3's 1.225: a little lost on the seeing-pair target itself.
- Not yet run: `bench_phase4.py` on the full pipeline, crops on all targets, ideal-Gaussian
  narrowing, CPU inference time (same network, so unchanged).
- Truth stars were drawn at random rather than at Gaia positions; real star statistics and
  saturated cores are not modelled.

### Rounds 2-4: the faint-peak problem (2026-10-21)

All from v3, 10,000 steps, lr 3e-4. s2: `--synth-frac 0.35 --synth-set FLUX=20,5e5 --synth-set
STARS=1,25` (fewer, brighter synthetic stars). s3a / s3b: s2 + `--peak-penalty 10 / 30`
unmasked. s4: s2 + `--peak-penalty 80` masked (`peak_penalty`: one-sided cost of point-like
excess over the synthetic truth, DoG 1 px - 3 px, only where the truth's own response is
< 0.3 sigma, grown 3 px; grown 7 px the mask covered 63% of the canvas). Weights
`C:\nn_work\model_deconv_s{2,3a,3b,4}.npz`.

Unconfirmed peaks = `eval_nn_faint` peaks > 2 sigma per Mpx x (1 - share confirmed by the
sharp stack), exact from the JSONs (mean over the 9 held-out pairs):

| | v3 | s1 | s2 | s3a | s3b | s4 |
|---|---|---|---|---|---|---|
| real val (identity 1.872) | 1.225 | 1.295 | 1.273 | 1.308 | 1.348 | 1.279 |
| synthetic: width / truth | 1.80 | 1.11 | 1.10 | 1.11 | 1.13 | 1.12 |
| synthetic: flux | 0.96 | 0.94 | 0.94 | 0.87 | 0.80 | **0.97** |
| synthetic: invented / Mpx | 35 | 58 | 62 | 52 | 65 | **39** |
| real: width / input | 0.81-0.92 | 0.74-0.90 | 0.75-0.90 | 0.74-0.90 | -- | 0.76-0.91 |
| real: width / sharp stack (< 1) | 0.96-1.08 | 0.88-1.07 (8/9) | 0.88-1.08 (7/9) | 0.87-1.07 (8/9) | -- | 0.89-1.08 (7/9) |
| real: error < `nn_denoise` | 9/9 | 8/9 + 1 equal | 8/9 | 9/9 | -- | 8/9 |
| real: unconfirmed peaks / Mpx | **4.8** | 12.6 | 10.3 | 8.3 | -- | 10.0 |
| real: faint discs 1.5 px, 0.5-1 sigma | 0.86 | 0.91 | 0.90 | 0.89 | -- | **0.92** |
| Gaia single-star splits | 0 | 0 | 0 | 1/50 (Orion) | -- | 0 |

(s3b was dropped after the synthetic check. The miss in "error < nn_denoise" is always
Pleiades 2025-10: s2 0.504, s4 0.494 vs 0.484.)

- Fewer, brighter synthetic stars (s2) barely moved the real unconfirmed peaks.
- An unmasked penalty dims real stars instead (s3a/s3b flux 0.87 / 0.80) and s3a split a star.
- The masked penalty (s4) fixes the *synthetic* problem completely -- invented peaks back at
  v3's level with the best flux of any model, sharpening kept, no splits, real val as s2 --
  but the *real* unconfirmed peaks stay at about twice v3's (10.0 vs 4.8 per Mpx; > 3 sigma
  all models 96-100% confirmed). So the real excess is not what the synthetic pairs model.
  Either real noise structure the synthetic noise (empty-sky patches) never shows, or the
  check itself: the sharp stack (30% of the frames) cannot confirm the faintest real
  sources, and a model that recovers more of them shows more "unconfirmed" peaks. v3 at 4.8
  says the check is not simply proportional to sharpening; not ruled out either way.

**Best candidate: s4.** Not shipped.

### Second-night check: the excess is mostly the shallow sharp stack (2026-10-21)

`tools/eval_peaks_second_night.py`: the same peaks (as `eval_nn_faint.false_sources`),
confirmed against the *other* session of the same target -- its full stack, independent
noise, mapped by star match + affine fit (0.14-0.21 px rms, no resampling). All frames of
the same session would not do: the input holds 70% of them, so its noise confirms itself.

A direction is valid only if the reference confirms the unprocessed input's own peaks at
least as well as the sharp stack does. Three are not, their reference night being shallower
despite its frames (Pleiades 2025-11 is 1.45x noisier per pixel than 2025-10): Omega
2026-08 -> 2025-11, Pleiades 2025-10 -> 2025-11, Sunflower 2026-03 -> 2026-04 (input
unconfirmed 14 -> 113, 36 -> 125, 8 -> 20 per Mpx). Sculptor has one session. The five valid
directions, unconfirmed peaks > 2 sigma per Mpx, sharp stack -> second night:

| session (reference frames vs sharp stack's) | input | nn_denoise | v3 | s2 | s4 |
|---|---|---|---|---|---|
| Omega 2025-11 (114 vs 21) | 17.2 -> 3.8 | 3.2 -> 2.5 | 10.8 -> 7.6 | 19.1 -> 8.9 | 19.7 -> 9.5 |
| Orion 2025-12-11 (130 vs 36) | 3.2 -> 0.0 | 1.3 -> 0.0 | 3.8 -> 1.9 | 7.0 -> 2.5 | 7.0 -> 3.2 |
| Orion 2025-12-27 (121 vs 39) | 3.2 -> 2.5 | 0.6 -> 0.6 | 2.5 -> 1.9 | 3.8 -> 3.2 | 3.8 -> 1.9 |
| Pleiades 2025-11 (277 vs 82) | 8.3 -> 1.9 | 0.0 -> 0.0 | 1.3 -> 0.0 | 7.0 -> 0.0 | 8.3 -> 0.0 |
| Sunflower 2026-04 (158 vs 14) | 6.4 -> 0.6 | 0.6 -> 0.0 | 1.3 -> 0.0 | 3.2 -> 0.0 | 2.5 -> 0.0 |
| mean | 7.7 -> 1.8 | 1.1 -> 0.6 | 3.9 -> **2.3** | 8.0 -> 2.9 | 8.3 -> **2.9** |

Against a deep independent reference s4's excess over v3 shrinks from 2.1x to 1.3x (2.9 vs
2.3 per Mpx), and on the two deepest references (Pleiades 2025-11, Sunflower 2026-04) every
s4 peak the sharp stack missed is confirmed. So most of the "unconfirmed" peaks were real
faint sources the sharp stack is too shallow to confirm. A small residual excess remains
on Omega 2025-11 (9.5 vs 7.6) and Orion 2025-12-11 (3.2 vs 1.9): five directions, four
targets, too few to call. `eval_nn_faint`'s sharp-stack confirmation understates a sharper
model; use the second-night check alongside it.

### Full pipeline: s4 against v3 (2026-10-21)

`bench_phase4.py` on the five held-out stacks, `--variant nn="--deconvolve nn"`, once with
v3 and once with s4 copied over `src/data/nn_deconv.npz` (worktree only, restored after;
`--deconvolve nn` has no model-path option). Results `C:\nn_work\synth\bench_{v3,s4}.json`,
display JPEGs `C:\nn_work\bench_s4\{v3,s4}`. `ring` x `sky_noise` as above (positive =
bright halo):

| target | fwhm_ratio base / v3 / s4 | sky_noise v3 / s4 | faint_cnr v3 / s4 | ring x sky_noise base / v3 / s4 | hue err v3 / s4 |
|---|---|---|---|---|---|
| Omega 2026-08 | 1.061 / 0.932 / **0.900** | 0.284 / 0.296 | 105 / 102 | 1.17 / 0.37 / 0.37 | 8.5 / 8.6 |
| Orion 2025-12-27 | 1.064 / 0.954 / 0.960 | 0.085 / 0.086 | 297 / 296 | 0.30 / 0.22 / 0.26 | 5.6 / 5.4 |
| Pleiades 2025-11 | 1.075 / 0.978 / **0.956** | 0.288 / 0.308 | 68 / 66 | 3.37 / 1.38 / 1.44 | 12.9 / 12.8 |
| Sculptor | 0.905 / 0.753 / **0.744** | 0.054 / 0.054 | 287 / 302 | 0.33 / 0.57 / **0.73** | 3.8 / 3.8 |
| Sunflower 2026-03 | 0.885 / 0.750 / **0.727** | 0.047 / 0.048 | 235 / 253 | 0.22 / 0.40 / 0.42 | 4.6 / 4.7 |

In the delivered image s4 is **only 1-3% narrower than v3** on four targets (Orion 0.6%
wider), with the same noise, faint contrast, colour (Gaia rho within 0.02) and hue. The
large synthetic gain (width/truth 1.80 -> 1.12) and the real-pair gain (width/input
0.81-0.92 -> 0.76-0.91) shrink to this through the rest of Phase 4 (restore, star
reduction, stretch). Sculptor's bright halo is a little stronger (0.57 -> 0.73).

**Verdict: s4 is a real but small improvement; not worth shipping on its own.** The
synthetic pairs teach sharpening the real images barely use: the network is the same
24-filter U-Net, and the synthetic-to-real gap (noise model, real PSF wings, saturated
cores, background statistics) is where further gains would have to come from -- the
weeks-long "full version" above, not more rounds of this one. Keep v3 shipped. Weights
and ingredients stay in `C:\nn_work` (`model_deconv_s{1,2,3a,3b,4}.npz`, `synth/`, 1.4 GB)
until that is decided.

### The original plan (2026-10-18)

Why: the seeing-pair target caps sharpening at the session's sharpest-30% stack -- blur every
frame shares (optics, focus, debayer) is never removed, and ideal Gaussian stars narrow only
2-5%. BlurXTerminator-style correction needs targets sharper than any real data.

Why not just re-blur the existing targets: an input made from the target shares its noise
(breaks the independent-noise property the seeing pairs rely on), and the truth would still be
the best-frame stack, so nothing teaches the network to go past it.

Plan:
- **Truth**: stars drawn at Gaia positions/fluxes (`gaia_cache`) with a narrow PSF, on a
  clean extended background (a denoised, star-removed real stack; the inpaint
  `star_removal` is good enough for training targets).
- **Input**: truth blurred by a PSF family -- Moffat over FWHM/beta, ellipticity, coma,
  slightly different width per channel -- plus noise matched to real stacks *including its
  spatial correlation* (debayer + warp; `noise_model`). Uncorrelated noise will not transfer.
- **Mix** synthetic pairs with the real seeing pairs so the network stays honest on real data;
  `train_nn_denoise.py bank --mode synthetic`, loss weighting adjusted for noiseless targets.
- **Evaluation** (the hard part): `width/target` and `eval_nn_faint`'s invented-source check
  both reference the sharp stack, so they cannot judge detail beyond it. Add (a) held-out
  synthetic fields with exact truth, (b) a Gaia close-pair check on real stacks: pairs below
  the input's resolution split at the right separation and flux ratio, and no split where Gaia
  has a single star.

Order: a 2-3 day feasibility run first (small generator, short training, held-out check);
go only if held-out stars narrow clearly past the sharp-stack target without failing the
Gaia check. Full version estimated at 2-3 weeks plus ~3-4 h per training run (RTX 3050),
3-6 runs expected.

Risks: invented detail (the real-pair network cannot invent sources; synthetic targets lose
that guarantee -- the Gaia check is the guard); the 24-filter U-Net may need more depth or
width for larger varying blur, which costs CPU inference time (never timed at full
resolution); a large synthetic-to-real gap could limit the gain over today's 7-16%.

## BXT-inspired open-source direction

This is a possible future feature, not a claim of compatibility with or parity with
BlurXTerminator. The practical target is a locally adaptive, artifact-aware astronomical
deconvolver that is trained and validated in this project. A true BXT-like result requires
neural-network training; end users would receive versioned weights and would not need to
train a model themselves.

### Two implementation paths

- **Classical path (no training):** extend `psf_deconvolution.py` to fit a spatial PSF field
  from stars, then apply locally varying RL/TV/sparse deconvolution with star masks, flux
  constraints, and halo/ringing protection. This is interpretable and useful, but is more
  sensitive to PSF-fitting errors and is not expected to match a learned system.
- **Learned path (recommended):** extend the existing `--deconvolve nn` model with local PSF
  conditioning and separate star/non-stellar behavior. Training would use a larger mixture of
  real seeing pairs and validated synthetic-blur pairs, with losses for star width,
  ellipticity, flux conservation, faint-structure preservation, and halo/ringing penalties.

The learned path should expose conservative controls for star sharpening, extended-object
sharpening, correction strength, and automatic mode. Existing star detection, masks, PSF
fitting, and held-out evaluation tools are reusable.

### Proposed order and effort

1. Improve the current seeing-pair model and expand the held-out validation set: 2-4 weeks.
2. Add spatial PSF features (local FWHM, ellipticity, orientation, field position, and
   confidence): 4-8 weeks.
3. Add star/non-stellar controls and artifact safeguards: 2-5 weeks.
4. Harden validation and CPU/GPU inference: 4-8 weeks.

A useful BXT-inspired v1 is therefore an approximately 8-12 week effort; a production-grade
feature is more likely 4-6 months. These estimates assume access to a GPU and a diverse
training corpus. The main risk is not the neural-network code: it is obtaining trustworthy
targets sharper than the input without teaching the model to invent detail.

Do not market this as "open-source BlurXTerminator" or promise feature parity. The appropriate
description is "locally adaptive, artifact-aware neural astronomical deconvolution."

## Cross-night pairs: x1 (2026-10-23) -- sharpest yet, noisier; not shipped

Branch `feat/nn-deconv-crossnight`. The seeing pair's target caps sharpening at the session's
sharpest-30% stack (median per-frame width ratio 0.89 against the rest). A *stricter* cut
barely helps: the sharpest 10% are only 3% narrower than the sharpest 30% (median per-frame
FWHM, 83 sessions; p10-p90 0.93-0.98) for 1.7x the target noise. Two nights of the same target
differ far more: the better night's sharp-30% against the softer night's rest is 0.61 (27
targets, p10 0.40, p90 0.81). `tools/gen_crossnight_pairs.py` turns the seeing pairs into
cross-night pairs (no restacking): the sharpest night's sharp stack aligned onto the softest
night's grid with `--merge`'s registration (once per target, on the soft night's sharp stack;
the two soft stacks share a grid, and registering each separately gave Flame -6.9 and 0.0 deg),
a <= 1 px star-offset check, `--merge`'s gain + quadratic sky surface, cut to the common
rectangle, width ratio measured on the same stars and kept in 0.65-0.85 (the soft night's
sharp stack is the input when its rest is too soft). 13 pairs of 27 targets (10 gaps too wide,
4 alignments refused); 10 outside `HOLDOUT`.

```
python tools/gen_crossnight_pairs.py --seeing C:/nn_work/nn_deconv/pairs --out C:/nn_work/nn_deconv/crossnight
C:/nn_work/venv/Scripts/python tools/train_nn_denoise.py bank --mode deconv --pairs C:/nn_work/nn_deconv/pairs C:/nn_work/nn_deconv/crossnight --per-pair 240 480 --bank C:/nn_work/bank_deconv_x1.npz
C:/nn_work/venv/Scripts/python tools/train_nn_denoise.py train --bank C:/nn_work/bank_deconv_x1.npz --steps 10000 --lr 3e-4 --init src/data/nn_deconv.npz --out C:/nn_work/model_deconv_x1.npz --resume
```

Bank 22,560 patches (21% cross-night), 940 s on the 1650 Ti; best val 1.382 at step 4000
(identity 2.518; val set differs from v3's, not comparable), parity 5.5e-7.

Held-out pairs (`eval_nn_deconv`, r4): width / input 0.81-0.91 vs v3's 0.82-0.92 (1-4%
narrower on 7 of 8 seeing pairs), but error against the sharp stack up on all 8 (Pleiades
2025-10 0.441 -> 0.509, above `nn_denoise`'s 0.484) -- partly by design, output narrower than
that target is scored as error. On the 3 held-out cross-night pairs, error against the better
night: Omega 4.276 -> 4.106, Sunflower 0.189 -> 0.181, Orion equal.

Invented peaks (`eval_peaks_second_night`, unconfirmed > 2 sigma per Mpx, second night): **x1
at or below v3 in every direction** (Omega 7.6 -> 5.1, Orion 1.9 -> 0.0 and 1.9 -> 0.6,
Pleiades 92.9 -> 81.5 on the shallow-reference direction, the rest equal) -- unlike s4.

Full pipeline (`bench_phase4`, x1 copied over `src/data/nn_deconv.npz` in the worktree only;
`C:\nn_work\synth\bench_x1.json`), v3 / x1:

| target | fwhm_ratio | sky_noise | faint_cnr | ring x sky_noise |
|---|---|---|---|---|
| Omega | 0.932 / **0.922** | 0.284 / 0.316 | 105 / 94 | 0.37 / 0.36 |
| Orion | 0.954 / **0.927** | 0.085 / 0.102 | 297 / 248 | 0.22 / 0.20 |
| Pleiades | 0.978 / **0.912** | 0.288 / 0.344 | 68 / 57 | 1.38 / 1.49 |
| Sculptor | 0.753 / **0.736** | 0.054 / 0.067 | 287 / 230 | 0.56 / 0.71 |
| Sunflower | 0.750 / **0.729** | 0.047 / 0.056 | 235 / 193 | 0.40 / 0.56 |

**Verdict: the sharpest model so far (1-7% narrower than v3 in the delivered image, mean 3.3%,
against s4's 1-3%), with no extra invented peaks, but 11-24% more sky noise, 10-20% less
faint contrast and a stronger bright halo on the galaxies (Sculptor, Sunflower).** Not shipped:
it trades noise for sharpness rather than improving both. Weights `C:\nn_work\model_deconv_x1.npz`.

**Matched noise (2026-10-23): x1 is not better than v3, only further along the same curve.**
On the 9 held-out seeing inputs (centre 2304x1536), star width / input (same stars) and sky
noise (robust sigma of the high-passed luminance on the faintest 40%, / the input's), medians:

| variant | width | noise |
|---|---|---|
| `nn_denoise` | 0.999 | 0.041 |
| v3 | 0.868 | 0.050 |
| x1 | 0.852 | 0.060 (1.14-1.31x v3's per session) |
| 0.7 x1 + 0.3 `nn_denoise` (v3's noise) | 0.892 | 0.050 |
| 0.5 x1 + 0.5 v3 | 0.852 | 0.052 |

Brought down to v3's noise by blending with the denoiser, x1 is *wider* than v3 (0.892 vs
0.868): the extra sharpness is bought with noise, not with information the cross-night pairs
added. Every trained variant so far (v3, s1-s4, x1) sits on one width-noise trade-off; moving
the curve needs a better prior (more capacity, or truth sharper and cleaner than any Origin
stack), not more real pairs. (`_sky_stats` sigma is not a pixel-noise measure here: it gave the
denoiser the input's noise.)

## Capacity sweep (2026-10-23): a wider network does not help -- stop here

Same seeing-pair bank (17,760 patches, 83 pairs, `--per-pair 240`), v3's recipe from scratch
(30,000 steps, lr 1e-3), U-Net base width 24 / 32 / 48 (`train_nn_denoise.py train --filters`;
the numpy forward pass takes any width). GTX 1650 Ti at 38 W: 55 min / 1.7 h / 2.0 h.
`tools/eval_nn_matched_noise.py` on the 9 held-out seeing inputs (centre 2304x1536), medians:

| model | params | best val (identity 1.872) | width | noise / v3 | width at v3's noise | s / 3.5 Mpx |
|---|---|---|---|---|---|---|
| v3 (shipped) | 0.27 M | 1.2248 | 0.868 | 1.00 | 0.868 | 6.4 |
| f24 (retrain) | 0.27 M | 1.2201 | 0.855 | 1.00 | **0.856** | 5.9 |
| f32 | 0.47 M | **1.2110** | 0.874 | 1.01 | 0.877 | 9.2 |
| f48 | 1.06 M | 1.2135 | 0.866 | 1.02 | 0.871 | 21.1 |

The wider networks reach a slightly lower validation loss but are **not sharper at equal noise**
(f32 and f48 are wider than f24), and cost 1.6x / 3.6x the run time. Invented peaks
(`eval_peaks_second_night`): all four within each other's spread on every direction (f48 a
little higher on Omega 2026-08 -> 2025-11, 44.8 vs 37.1 per Mpx; Pleiades 2025-10 within 3).

Two conclusions:
- **Capacity is not the limit; the training target is.** With v3 (seeing pairs), s1-s4
  (synthetic blur), x1 (cross-night pairs) and f32/f48 (wider), every variant lands on one
  width-noise curve. More sharpening needs a target sharper and cleaner than any Origin stack
  (e.g. survey imaging as truth with measured Origin PSFs and noise), which is weeks of work and
  not started.
- **Run-to-run spread is ~1.5%:** f24, the same recipe as v3 on the same pairs, is 1.4% narrower
  at the same noise. Differences of 1-3% between models trained once (s4 vs v3 above) are within
  it. Not a reason to replace v3.

Models `C:\nn_work\model_deconv_f{24,32,48}.npz`; results `C:\nn_work\capacity\`.

## Survey truth: v4 (2026-10-24) -- shipped, blended with the denoiser

The capacity sweep ended on "the training target is the limit". v4 replaces the target: a deep
survey image of the same sky instead of a slightly sharper Origin stack. Fine-tuned from v3, same
0.27 M-parameter U-Net; `src/data/nn_deconv.npz` is v4 from this date.

**Data.** All 83 sessions restacked with the default settings and `--offline` (so not colour
calibrated; the colour model below absorbs it). `tools/fetch_ps1_truth.py` cuts 3 crops of
600 px per stack (246 crops) and fetches Pan-STARRS DR1 g, r, i stack cutouts through STScI's
fitscut, which returns linear counts. The CDS HiPS r and i keep the per-skycell asinh
compression and are unusable. Every Origin pixel averages a 6x6 grid of bilinear PS1 samples.
Crops spanning skycells are stitched. The crop WCS is built by shifting CRPIX: `WCS.slice`
put the centre 0.26 deg off.

**Target** (`tools/gen_survey_pairs.py`):
- Colour: one non-negative 3x3 matrix with rows summing to 1, plus a per-crop, per-channel
  scale, fitted on isolated unsaturated stars' aperture fluxes. Unconstrained fits collapsed.
  Rows are Origin R, G, B against PS1 g, r, i: `[[.33 .51 .16] [.56 .43 .01] [.91 .09 .00]]`.
  Crops with colour scatter above 0.25 dex (nebula-dominated photometry) are left out.
- Sharpness: Gaussian FWHM 2.0 px.
- Large scales: everything above 16 px comes from the Origin input. PS1 is sky-subtracted on
  arcminute scales.
- Masks: no PS1 coverage, saturated Origin cores.
- Weights: 1 / max(input variance, 0.25) with a 5% systematic term. With weight 1, a few star
  cores carried the loss (train 1277 vs val 33). Without the floor and the systematic term,
  val was 20180.

**Two fixes after the first model (v1):**
- **Invented peaks.** PS1 is about 8x deeper than an Origin stack, so the raw target holds
  thousands of real faint stars the input cannot show. v1 learnt to put one on noise bumps:
  1.7x v3's invented peaks against the truth in the pilot. The S/N gate (`snr_gate`) uses the
  survey target only where the input's matched-filter S/N exceeds 3-6 sigma, and the plain
  `nn_denoise` output elsewhere.
- **Bright stars.** Stars saturated in PS1 have no truth. Masked out, the network never saw
  one, and v1 improvised: donut rings on Sunflower, flat bloated discs on Sculptor, soft blobs
  on Orion. v4 trains them toward v3's output (`--bright-model`, weight 1). They now render as
  v3's did; the side-by-side was checked on all five held-out targets.

**Training.** Bank of 3,887 patches. `train_nn_denoise.py train --steps 15000 --lr 3e-4 --init
<v3> --resume`, 23 min. Best val 0.3187 (identity 1.636). numpy vs torch
max rel diff 4.9e-7.

**Against the PS1 truth** (`tools/eval_ps1_truth.py`, 27 held-out crops, medians):

| model | error | signal error | width / truth | invented / Mpx | noise / input |
|---|---|---|---|---|---|
| v3 | 5.41 | 6.55 | 1.79 | 419 | 0.231 |
| v1 | 5.21 | 5.42 | **1.24** | 352 | 0.258 |
| **v4** | **5.18** | 6.24 | 1.41 | **321** | 0.256 |

Equal-noise width (`tools/eval_nn_matched_noise.py`, 9 held-out seeing inputs): v3 0.868,
v1 0.496, v4 0.581.

**Full pipeline** (`tools/bench_phase4.py`, five held-out targets). v4 alone left 30-36% more
sky noise than v3 on the galaxies and lost 24-31% faint contrast. Blending its output with the
nn denoiser's (`nn_deconv_blend`, share of v4 kept) trades that back. Galaxies, against v3
(the previous `--auto` choice):

| target | blend | width | sky noise | faint contrast | Gaia colour rho |
|---|---|---|---|---|---|
| Sunflower | v3 | 0.750 | 0.047 | 235 | 0.809 |
| | 0.50 | 0.557 | 0.048 | 218 | 0.829 |
| | **0.35** | **0.640** | 0.045 | 237 | 0.830 |
| Sculptor | v3 | 0.753 | 0.054 | 287 | 0.771 |
| | 0.50 | 0.573 | 0.054 | 283 | 0.753 |
| | **0.35** | **0.643** | 0.050 | 303 | 0.763 |

`--auto` uses 0.35 on galaxies (`auto_settings.NN_DECONV_GALAXY_BLEND`), so stars are about 15%
narrower than with v3, with faint contrast and colour equal or better. An explicit
`--deconvolve nn` gets 0.75 (`cli.py` default; `--config nn_deconv_blend=...` overrides,
including under `--auto`). The blend runs both networks: 26 s for the nn step on a full
Sunflower stack.

**Nebulae and star fields: switched too (2026-10-25), at 0.75.** Against what `--auto` used
there (`--denoiser nn` with chroma NR first):

| target | variant | width | sky noise | faint contrast | JPEG colour rho | linear colour rho / slope |
|---|---|---|---|---|---|---|
| Orion | denoiser | 1.067 | 0.093 | 273 | 0.776 | 0.959 / 0.68 |
| | v4 0.75 | 0.696 | 0.082 | 309 | 0.850 | 0.978 / 0.88 |
| Pleiades | denoiser | 1.078 | 0.326 | 62 | 0.478 | 0.968 / 0.71 |
| | v4 0.75 | 0.730 | 0.302 | 68 | 0.339 | 0.993 / 0.92 |
| Omega | denoiser | 1.061 | 0.326 | 91 | 0.497 | 0.987 / 0.82 |
| | v4 0.75 | 0.905 | 0.286 | 109 | 0.537 | 0.990 / 0.95 |

The first reading was that v4 costs colour on Pleiades and Omega: about 2 deg more hue error and
lower star-colour rank correlation, both read off the JPEG. That is the stretch, not the network.

- `bench_phase4`'s new `*_lin` metrics score the same stars on the Phase 4 output before the
  stretch, and there v4's colour is better on all three targets.
- The default per-channel stretch (`--stretch-color channel`, the user's preferred look) whitens
  star cores. v4's narrower, brighter cores clip more.
- With `--stretch-color preserve` the JPEG agrees. Colour rho: Orion 0.870 -> 0.912, Pleiades
  0.761 -> 0.947, Omega 0.774 -> 0.802. Faint contrast +18 / +27 / +24%.
- `hue_err_deg` compares the JPEG with each variant's own Phase 4 output, so it measures the
  stretch only.

Two things tried here did not help:
- **Chroma NR before the network:** gives most of the sharpening back (width 0.82 / 0.93 / 0.98).
- **v4's luminance with the denoiser's colour** (ratio transfer, additive near zero): even
  sharper (Orion 0.62), but JPEG colour worse still (Orion hue error 7.9, Pleiades colour rho
  0.28). It only made cores brighter.

One cost: chroma mottle on Pleiades' sky, 6.0 vs 5.4 (it gets no chroma NR now). Orion and Omega
are lower than before. `--auto` keeps `--denoiser nn` as the fallback when
`src/data/nn_deconv.npz` is missing.

### More data and a sky weight: v5 (2026-10-26) -- not shipped

Three times the data: 9 crops per stack (`fetch_ps1_truth.crop_centres`' extra positions), 738
crops, a bank of 11,854 patches. Trained like v4 (from v3, 15k steps, lr 3e-4). Two variants
added a sky term to the loss weight, `ww *= 1 + k * sky` with `sky` = pixels whose 5x5-averaged
input is under 1 sigma, to push the network toward the denoiser's quiet sky so galaxies could
drop the second pass. The trainer option was not kept.

Against the PS1 truth (79 held-out crops, medians) and equal-noise width (seeing inputs):

| model | width / truth | invented / Mpx | equal-noise width |
|---|---|---|---|
| v4 | 1.37 | **364** | 0.506 |
| v5 | 1.32 | 411 | 0.545 |
| v5, sky weight 3 | **1.31** | 402 | 0.504 |
| v5, sky weight 20 | 1.34 | 394 | **0.490** |

Galaxies in the full pipeline, sky weight 20 against shipped v4 at 0.35:

| target | variant | width | sky noise | faint contrast | linear colour rho |
|---|---|---|---|---|---|
| Sunflower | v4 0.35 | 0.640 | **0.045** | **237** | **0.889** |
| | s20 alone | **0.421** | 0.062 | 167 | 0.861 |
| | s20 0.60 | 0.519 | 0.051 | 204 | 0.882 |
| Sculptor | v4 0.35 | 0.643 | **0.050** | **303** | **0.851** |
| | s20 alone | **0.465** | 0.067 | 227 | 0.809 |
| | s20 0.60 | 0.560 | 0.056 | 270 | 0.840 |

Three times the data moves nothing outside noise, and the sky weight only slides along the same
width-for-noise line the blend already gives: alone it is a third narrower but 35-40% noisier,
with 25-30% less faint contrast and worse colour on Sculptor. v4 + blend stays. Same conclusion
as the capacity sweep, now for data volume too: with this network the target sets the limit.

### One network for the galaxy blend (2026-10-27) -- not shipped

Galaxies run both networks (`nn_deconv_blend` 0.35). Both work in noise units and the
low-frequency restore after them is linear, so one network can be trained on the blend of
their raw outputs: input = each training session's seeing-pair input stack, target =
0.35 x v4 + 0.65 x the nn denoiser on it (11,840 patches from 74 sessions, half centred on
the brightest 0.5% of pixels, weight 1 / the input's noise variance), fine-tuned from v4,
15k steps, lr 3e-4. Val 0.0309 (identity 1.347).

It reproduces the blend closely on the held-out galaxies: RMS difference 0.07-0.09 sigma
(0.027 on sky) against a 1.0 sigma change from the input. In the full pipeline:

| target | variant | width | sky noise | faint contrast | Gaia colour rho (linear) |
|---|---|---|---|---|---|
| Sunflower | v4 + blend 0.35 | 0.640 | **0.045** | **237** | 0.889 |
| | distilled, one pass | 0.636 | 0.049 | 220 | 0.886 |
| Sculptor | v4 + blend 0.35 | 0.643 | **0.050** | **303** | 0.851 |
| | distilled, one pass | 0.638 | 0.054 | 282 | 0.846 |

One pass is 7.8 s on a full Sunflower stack, so it would save about 8 s per galaxy run, for
8-9% more sky noise and 7% less faint contrast. Not worth it; the two passes stay.

### Chroma NR on the sky after the network (2026-10-27) -- not shipped

The nebula path (0.75) skips chroma NR. `reduce_chroma_noise` after the network, with its own
luminance heuristic (everything above sky + 3 sigma protected; the earlier attempt used the
star mask and so smoothed the nebula too):

| target | variant | blotch | sky noise | linear colour rho / slope | Gaia rho (linear) |
|---|---|---|---|---|---|
| Pleiades | v4 0.75 | 6.03 | 0.302 | 0.993 / 0.92 | |
| | + sky chroma NR | 5.96 | 0.264 | 0.991 / 0.91 | |
| Orion | v4 0.75 | 4.64 | 0.082 | 0.978 / 0.88 | 0.925 |
| | + sky chroma NR | 4.68 | 0.076 | 0.951 / 0.79 | 0.902 |
| Omega | v4 0.75 | 11.41 | 0.286 | 0.990 / 0.95 | 0.959 |
| | + sky chroma NR | 11.66 | 0.269 | 0.989 / 0.95 | 0.959 |

Pleiades' mottle drops 1%; Orion's colour loses more. With `chroma_nr_large_sigma` 12 Orion's
Gaia rho fell to 0.73. Note the plain wavelet path these configs were saved with scores 6.14 on
the Pleiades, so v4's mottle is no worse than that; the 6.0 vs 5.4 above compares against
`--denoiser nn` with chroma NR first.

Data `C:\nn_work\survey\`, `C:\nn_work\distill\`, `C:\nn_work\mottle\` (models, `train_*`, `eval_*`, `matched_*`, `bench_*`,
`bright_v3_v1_v2.jpg`; stacks, crops and banks deleted).

## Pixel-scale normalisation (2026-10-31)

The deconvolver only saw Origin stacks (star FWHM 2.9-5.1 px, Gaussian fit). Resampling the
Omega held-out stack (4.41 px) to mimic other pixel scales, the sharpening it applied depended
on the star width: FWHM out / in = 0.64 at 2.2 px (undersampled) ... 0.74 at 4.4 ... 0.91 at
8.4 px. `nn_denoise(..., fwhm_range=TRAIN_FWHM_RANGE)` (`--deconvolve nn`, config key
`nn_scale_normalise`, default on) measures the stars' Gaussian FWHM (`measure_star_fwhm`;
`quality.measure_fwhm` reads 7.4 px on a 4.4 px stack, so not that), and when it is outside
(2.3, 5.2) px resamples (cubic spline) so the stars land on the nearest bound, runs, and
resamples back. Inside the range -- every Origin session -- the output is bit-identical to
before. Result on the same rescaled Omega: 0.68 / 0.67 / 0.74 / 0.81 / 0.79 at 2.2 / 3.0 / 4.4 /
6.0 / 8.4 px (the round trip alone changes FWHM < 1%). Targeting the nearest bound rather than
the 4.4 px centre gives up some uniformity (0.74-0.82 centred) to leave Origin untouched.
Not skipped by `--deconvolve nn` blend: the blended denoiser is scale-tolerant and not resampled.
Caveats: a rescaled Origin stack keeps Origin's PSF shape and has resampling-correlated noise;
real other-camera stacks are still needed. Upsampling beyond 2.5x (FWHM < 1.1 px) is skipped.

**Seestar S50 check (2026-10-31).** Two S50 stacks (M13 73 subs, M101 207 subs; 2.4"/px, 50 mm,
star FWHM 2.6 px), the network alone on the linear stack, M101 against Pan-STARRS truth
(`tools/fetch_ps1_truth.py`, 2 crops, 92 stars, same stars in every image): FWHM input -> output
-> truth = 2.7 -> 1.5 -> 1.2 px (output/input 0.58; still 20-30% wider than truth, so not
over-sharpened); the 2.5-5 px wing halo falls from 3.3% of the peak to 1.0-1.5% (truth 0.07%),
no negative ring. Resampling by 1.08 to reach a 2.8 px lower bound was slightly worse than not
resampling (FWHM 1.49 vs 1.43 faint, halo 1.0 vs 0.9%), hence the 2.3 px lower bound. Flux in a
6 px aperture: faint stars (8-30 sigma) 1.01-1.02 of the input, mid/bright (30+ sigma) 0.91-0.94
-- the full network (blend 1.0) is not flux-preserving; check Origin before photometry runs
after `--deconvolve nn`. Caveats: PS1 truth is gri-averaged on the S50 grid (its ~1.2 px width
is partly pixel integration); M13 has no WCS, so no truth; no blend or Phase 4 in this run.

**Flux conservation vs star width (2026-10-31).** Aperture flux (3 x FWHM radius, annulus
background removed, same isolated unsaturated stars, luminance) of the network output / input,
full network (blend 1.0), median per brightness bin (8-30 / 30-200 / >200 sigma):

| stack | star FWHM px | 8-30 | 30-200 | >200 |
|---|---|---|---|---|
| Omega | 4.42 | 1.017 | 0.999 | 1.001 |
| Orion | 5.08 | 1.003 | 0.996 | 1.003 |
| Pleiades | 4.37 | 1.015 | 0.999 | 1.003 |
| Sunflower | 3.56 | 0.983 | 0.976 | - |
| Sculptor | 2.97 | 0.959 | 0.954 | - |
| S50 M101 | 2.72 | 0.961 | 0.942 | 0.948 |
| S50 M13 | 2.67 | 0.957 | 0.938 | 0.940 |

Flux is kept to ~0.5% at FWHM >= 4.4 px and falls as stars get narrower, on Origin's own
Sculptor (2.97 px) as much as on S50: it is a property of the network on near-undersampled
stars, not of the camera. Photometry is unaffected (it reads the linear stack, never Phase 4
output); only post-NN star brightness (e.g. colour of tight stars) is low by 2-6%.

**Star-aware restore (2026-10-31).** The loss above is redistribution, not removal: the network
alone keeps ~82% of a tight star's flux and the 4 px restore returns it, but into an 8-12 px
halo (M13, 3 FWHM aperture 0.94, 12 px 1.01). `nn_denoise(star_restore_frac=0.4)` (config
`nn_deconv_star_restore_frac`, `--deconvolve nn` only) restores near detected stars (full
weight within 1 FWHM, fading to 3 FWHM) with a Gaussian of 0.4 x the stars' FWHM instead of 4
px. Aperture flux out/in at 2 / 3 / 4.5 x FWHM on M13: 0.895 / 0.944 / 0.991 -> 1.00; Sculptor
0.898 / 0.958 / 0.992 -> 1.00; star width unchanged (M13 1.55 -> 1.58, Sculptor 0.515 -> 0.512).
Blur widths tried: 0.5 px (0.19 FWHM) gives the sharpening back (FWHM 2.17), a fixed 1 px did
that on 4.4 px stars (Omega width ratio 0.74 -> 0.90), 0.4 x FWHM still cost Omega 0.74 ->
0.82. Wide stars lose no flux, so the effect fades out between 3.6 and 4.4 px
(`STAR_RESTORE_FWHM`): Omega/Orion/Pleiades are unchanged, Sunflower, Sculptor and both S50
stacks are at 0.996-1.014 (was 0.94-0.98). Not run: the Phase 4 bench (ring, hue, faint_cnr)
with it on; the mask only touches pixels within 3 FWHM of detections.
