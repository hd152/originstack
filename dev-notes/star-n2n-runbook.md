# Learned star removal by Noisier2Noise: runbook (EXPERIMENTAL, started 2026-10-19)

Status (2026-10-19): tests pass; **v1 trained and evaluated: no-go** (see "v1 results").
No CLI flag; `src/star_n2n.py` is not wired into Phase 4.

## Idea

On linear data stars are additive: `y = x + s1`. Inject a second star layer `s2` drawn like
the real one, train the `nn_denoise` U-Net to predict `y` from `z = y + s2` (L2 -> E[y|z]).
If stars are a marked Poisson process independent of the sky and `s2` is the real process
thinned by `alpha`, each star in z is injected with probability alpha / (1 + alpha), so

    E[x|z] = z - (1 + alpha) / alpha * (z - E[y|z])      (alpha = 1: 2 E[y|z] - z)

(Noisier2Noise, Moran et al., CVPR 2020, with the star layer as the additive "noise"). No
starless ground truth: every session trains on itself, with its own PSF, debayer and noise.

## Files

- `src/star_n2n.py`: `StarField` (injection model from the image itself: per-channel stamps
  of bright isolated unsaturated stars, nearest stamp used so a position-dependent PSF carries
  over; donor fluxes/colours = the detected stars' luminosity function, capped at the brightest
  stamp; injection density = smoothed detection density; photon noise s / B), `starless_z`,
  `remove_stars_nn`.
- `tools/gen_starless_bank.py`: bank from the seeing pairs (`input.npy` + `target.npy` =
  full stack), output readable by `tools/train_nn_denoise.py train` unchanged.
- `tools/eval_star_n2n.py`: metrics needing no ground truth (real/injected flux left,
  faint-disc flux kept, sky shift, holes), crops.
- `tests/test_star_n2n.py`: injection model, oracle check of the formula, identity network,
  bank, eval metrics on an identity model.

## Steps

```powershell
pytest -q tests/test_star_n2n.py                      # first: never run yet
# pairs: the 83 seeing pairs already on the first machine (C:\nn_work\nn_deconv\pairs),
# or rebuild them: dev-notes/nn-deconv-runbook.md steps 1-2
python tools/gen_starless_bank.py --pairs C:/nn_work/nn_deconv/pairs --bank C:/nn/bank_starless.npz --limit 5 --per-session 40   # smoke
python tools/gen_starless_bank.py --pairs C:/nn_work/nn_deconv/pairs --bank C:/nn/bank_starless.npz --alpha 1.0
python tools/train_nn_denoise.py train --bank C:/nn/bank_starless.npz --steps 10000 --out C:/nn/model_starless_v1.npz --resume
python tools/eval_star_n2n.py C:/nn_work/heldout_stacks/*.fits --model C:/nn/model_starless_v1.npz --alpha 1.0 --json C:/nn/eval_starless_v1.json --crops C:/nn/crops_starless_v1
```

Do not pass `--inject` to training. The bank's `<bank>.json` records alpha; eval and
inference must use the same value. Feasibility first: ~10k steps (~1 h on the RTX 3050).

## Go / no-go

Go if, on the held-out stacks: `inject_left` and `real_left` well below 0.2 in every tertile,
`faint_kept` >= 0.8, `|sky_shift|` < 0.05 sigma, `hole` > -1 sigma, and the crops show no
blotches. The identity model scores ~1 / ~1 / 1 / 0 / 0 (see the test).

## Known gaps / assumptions to watch

- Saturated stars are not injected, so the network never learns to remove them (not
  additive anyway). Pair with an extended-PSF subtraction for those.
- Stars below detection are not injected: they stay in the output.
- Injected photon noise is white; real stack noise is correlated (debayer, warp). If the
  network keys on that, `inject_left` will look better than `real_left`.
- Run-time inputs have 1 / (1 + alpha) the training star density. Try alpha 0.5 vs 1.
- The U-Net's receptive field (~24 px radius) limits removal of large halos.
- Star positions are not independent of the sky (dark nebulae); the density map only
  approximates that.

## v1 results (2026-10-19): no-go -- the network tells injected stars from real ones

Bank: 74 sessions (holdouts excluded), 17,760 patches, alpha 1, ~6 s/session. 10k steps
in 15.5 min (RTX 3050, ~93 s / 1000 steps); val 9.04 at step 1000, best **2.53 at step
9000** (identity 26.80), still falling. numpy vs torch 6e-7. Files in `C:\nn_work\starless`
(`bank_starless.npz`, `model_starless_v1.npz`, `train_v1.log`, `eval_v1.json`, `crops_v1`).

| stack | real_left faint / mid / bright | inject_left faint / mid / bright | faint_kept | sky_shift | hole |
|---|---|---|---|---|---|
| Omega | 0.47 / 0.25 / -0.28 | -0.06 / -0.53 / -0.94 | 0.87 | -0.009 | 0.05 |
| Orion | 0.35 / 0.27 / 0.87 | 0.00 / -0.91 / -0.92 | 0.87 | -0.010 | 6.05 |
| Pleiades | 0.61 / 0.49 / 0.63 | -0.21 / -0.61 / -0.92 | 0.86 | -0.019 | 1.32 |
| Sculptor | 0.07 / 0.02 / 0.05 | -0.14 / -0.52 / -1.08 | 0.85 | -0.025 | -0.21 |
| Sunflower | 0.52 / 0.61 / 0.77 | -0.41 / -0.34 / -0.92 | 0.84 | -0.021 | 0.19 |

`faint_kept` and `sky_shift` pass; star removal fails. The raw network output f(z) keeps
(1 + left) / 2 of a star: ~0.04 of a bright *injected* star but 0.6-0.9 of a real one on
four of five stacks. Noisier2Noise needs f to be unable to tell them apart (then it keeps
1 / (1 + alpha) = 0.5 of both); this one recognises the injected stars, removes them fully,
and the 2x extrapolation turns that into an over-subtraction of the injected (-0.9) and an
under-removal of the real stars. Sculptor (sparse, clean) is the exception: real 0.02-0.07,
crops mostly cleared, a few dark pits. Orion's `hole` +6 is unremoved bright stars, not holes.

Likely tells, in the order to test them:
- **Integer-pixel placement of 60 fixed stamps per session**: every injected star is an
  exact copy of one of the 3 nearest stamps, same sub-pixel phase; real stars vary.
  Fix: sub-pixel shift (Lanczos) each stamp to a random phase.
- **Stamps carry their own sky noise** (cutouts of real stars, scaled by flux / stamp flux):
  a tapered 25 px disc of extra noise around each injected star, plus the stamp's own
  photon noise, on top of the white photon noise added afterwards. Fix: stamps from a
  stack of several stars or a fitted PSF model, noise added once.
- **White photon noise vs correlated real noise** (debayer, warp) -- see "Known gaps" above.
Diagnostic before retraining: score f on injected stars with each tell removed in turn.

## v2 / v3 (2026-10-19): clean PSFs; the network still told injected stars apart by shape

**Diagnostic** (scratch scripts, on v1): the raw network keep fraction of injected stars with
one tell removed at a time. Clean PSF (median of centroid-aligned stamps) lifted bright
injected keep 0.04 -> 0.74 (Sunflower), -0.01 -> 0.37 (Sculptor); photon noise off,
correlated photon noise and sub-pixel phase: no change. PSF radius 24 instead of 12: Orion
bright 0.21 -> 0.86 (wings). So the stamps' own sky noise was the main tell.

**v2** (one clean r=24 PSF per session, 4x4 sub-pixel phases, bank patches centred on real
and injected stars alike), 10k steps: gap narrowed (Sculptor bright real 0.31 / injected
0.06, was 0.05 / -1.08) but real stars mostly stayed (0.2-0.64 left). Injected stars were
rounder than real ones at the edges (ellipticity 0.11 vs 0.17) -> per-zone cores.

**v3** (core from the 15 nearest stars of each of 4x4 zones), 30k steps (46 min, val 2.75
vs identity 23.2): *worse* -- real 0.63-0.92 left, injected -0.8 on Omega/Orion/Pleiades.

**The validation loss measures discriminability.** If injected and real stars cannot be
told apart, f(z) = z - s/2 on every star and the loss cannot go below ~half of identity
(alpha 1, equal counts): ~11.6 here. v1 (2.53) and v3 (2.75) were far below: the network
was recognising the injected stars. Watch this number in training.

Which tell (on v3): photon noise white / none / correlated -- no change (faint injected
kept 0.45 vs real 0.87). **Pasted real cutouts** (copied as they are into empty sky) were
kept like real stars (0.78-0.94 vs real in place 0.81-0.96): the tell is the PSF shape.
The sigma-clipped stack makes bright stars narrower than faint ones, and every injected
star had the bright stamps' shape. Second-moment widths measured on white noise suggested
a 0.1-0.18 px gap at mid flux; measured on the real stack's (correlated) noise, after the
fix below, injected and real agree to +-0.1 px either way.

**v4**: a PSF per donor-flux quartile and zone (cores from that bin's isolated stars, scaled
by a template fit inside 4 px, wings from the bright stamps).
