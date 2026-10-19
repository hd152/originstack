# Learned star removal by Noisier2Noise: runbook (EXPERIMENTAL, started 2026-10-19)

Status: code written, **tests not yet run**, no bank built, no model trained. No CLI flag;
`src/star_n2n.py` is not wired into Phase 4.

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
