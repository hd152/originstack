# Learned deconvolution: per-session refit of the output layer (2026-10-22)

Experiment, not shipped. Tool: `tools/eval_session_refit.py`. Context: the learned
deconvolver (`--deconvolve nn`, [nn-deconv-runbook](nn-deconv-runbook.md)).

## Idea

The network's output layer is a 1x1 convolution, so its prediction is linear in the last
feature map d0 (24 channels). On a session's own seeing pair (input = all but the sharpest
30% of frames, target = the sharpest) the training loss `w (zA + pred s - zB)^2` is a
weighted least-squares problem in that layer's 3 x 25 weights: a closed-form solve, ridge-
regularised toward the shipped weights (`lam` relative to the mean diagonal of the normal
matrix). No training at run time, numpy only, seconds per session once d0 is computed.

## Test

Each of the 9 held-out pairs: fit on the left half, score on the right half (48 px apart,
more than the receptive field), so the fit never sees the scored pixels' noise. Score as
`eval_nn_deconv.py` (restore 4 px). With `lam` = 1e6 the refit reproduces the shipped
output exactly (sanity check).

| | error | sky error | width / sharp stack | width / input |
|---|---|---|---|---|
| v3 (shipped) | 0.495 | 0.079 | 0.998 | 0.863 |
| v3 + refit lam 1 | 0.472 (-4.5%) | 0.082 | 1.000 | 0.864 |
| v3 + refit lam 0.1 | 0.457 (-7.5%) | 0.082 | 0.998 | 0.862 |
| **v3 + refit lam 0.01** | **0.450 (-9.1%)** | 0.084 | 0.998 | 0.863 |
| s4 (synthetic-blur) | 0.521 | 0.080 | 0.970 | 0.839 |
| s4 + refit lam 0.01 | 0.464 (-10.6%) | 0.085 | 0.996 | 0.860 |

(means over the 9 pairs; error relative to each pair's own base, then averaged)

- Lower error on 9/9 pairs; largest where v3 was worst: Pleiades 2025-10 0.515 -> 0.302
  (-41%), Omega 2025-11 0.332 -> 0.290.
- Mean sharpness unchanged; per night it moves toward the session's own sharp stack from
  either side (v3 on Omega 2025-11 1.031 -> 1.010, on Sunflower 2026-03 0.949 -> 0.969).
  It calibrates *how much* to sharpen per night; it cannot sharpen past the sharp stack
  (that is its target), so it erases s4's extra sharpening (s4 + refit ~ v3 + refit).
- Sky error rises ~6% (0.079 -> 0.084).

## To ship it

Phase 3 would have to produce the seeing split during a normal run: per-frame FWHM is
already measured in Phase 1; the sharp stack is a second accumulation of 30% of the frames
(the rest follows from the full stack for a mean combine, approximately for sigma-clip),
plus one forward pass to d0 over the image (~15 s) and the 3 x (25 x 25) solve. Worth it
for the error (-9%, up to -41% on a bad night), not for sharpness. Not done; the sky-error
cost and the extra Phase 3 work are the open questions. Sessions with fewer than ~10
frames in the sharp subset would fall back to the shipped weights.

## Wired into the pipeline (`--nn-refit`), then measured: worse (2026-10-22)

`--nn-refit` (with `--deconvolve nn`): `stacking.session_nn_refit` builds the split while the
aligned frames exist (Phase 1 FWHM, sharpest 30% vs the rest, MAD sigma-clipped means read
per row band), `nn_denoise.refit_output_layer` solves the layer, `<stem>_nnrefit.npz`
keeps it for `--from-stack` (ignored if fitted on another model file). 27-43 s per session.

Full pipeline, `bench_phase4.py` on fresh stacks, shipped weights -> refitted (the side file
moved away and back; the stacks' config carries `nn_refit = true`, so a variant flag cannot
turn it off):

| | star width (P4 / linear) | sky noise | faint_cnr | ring (own noise units) |
|---|---|---|---|---|
| Sunflower 2026-03 (held out) | 0.740 -> 0.788 | 0.062 -> 0.092 | 179 -> 123 | 9.2 -> 4.3 |
| Sculptor 2025-10-11 (held out) | 0.754 -> 0.778 | 0.050 -> 0.073 | 552 -> 375 | 20.1 -> 12.0 |
| Black Eye 2026-03 (training) | 0.891 -> 0.891 | 0.034 -> 0.033 | 262 -> 266 | ~0 |

**Worse in the delivered image**: stars 3-6% wider, sky ~47% noisier, faint contrast a third
lower; only the bright-star ring is weaker. The held-out-half test above scored the layer on
the *rest-of-frames* stack it was fitted for; in the pipeline it runs on the full (proper
coadd) stack after Phase 4's early steps -- sharper input, and a layer calibrated on the
softer stack is too cautious there. Not for merging as it stands; a version would have to
fit on the stack it is applied to (e.g. the full stack against a sharper-subset target,
with independent noise), which the seeing split does not provide.
