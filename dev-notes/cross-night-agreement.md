# Cross-night agreement: two nights as independent half-maps (2026-10-22)

Experiment, not shipped. Tool: `tools/eval_cross_night.py`. Data: the held-out seeing pairs
of the four targets with two nights (Omega, Orion, Pleiades, Sunflower), both directions.

## Setup

For nights A and B of one target: inputs are A's *input* stack (all but its sharpest 30%)
and B's full stack, registered onto A (star match + affine, cubic) and flux/sky matched;
the independent reference is A's *sharp* stack (no frames in common with either). Faint
discs (1.5 / 3 px, 0.5-2 sigma) are injected into **both** nights (real sources: should be
kept), bright points (1 px, 10 / 30 / 100 sigma) into **one** night only (a satellite
glint, an unrejected hot-pixel cluster, a transient: should go). Methods act on the pair;
"mean" is the inverse-variance weighted mean (what `--merge` does).

## 1. Wiener gain from agreement (cryo-EM FSC style) -- dropped

Per starlet scale, signal power `smooth(a*b)` (noise cancels), noise power from
`smooth((a-b)^2)`, coefficient x S/(S+N) (`cross_night_filter`). Suppresses one-night
features (33% left vs the mean's 88%, Sunflower) but **keeps only 30-67% of faint sources
both nights show** (`nn_denoise`: 79-98%): the window average underestimates a small
source's power. Signal error no better than the mean.

## 2. Consistency selection -- promising

Weighted mean everywhere, except where a starlet coefficient differs between the nights by
more than `k * sqrt(noise_a^2 + noise_b^2 + (0.3 * |mean|)^2)`; there the smaller-magnitude
coefficient (the night that does not show the feature) is used
(`cross_night_consistency`). k = 3:

| | mean | consistency k3 |
|---|---|---|
| one-night 100 sigma point, flux left | 7-88% (that night's weight) | **~0%** in every direction it was measured |
| one-night 30 sigma | 7-88% | **0-12%** (Sunflower 2026-03 + 04: 50%) |
| one-night 10 sigma | 7-88% | 3-86% |
| faint sources in both nights (0.5-2 sigma) | 100% | **100%** |
| sky error where the nights' calibration differs | Omega 4.42; Pleiades 37.9 / 23.0 | **2.43; 20.7 / 12.2** |
| signal error vs the reference | | Pleiades 2025-11+10 166 -> 84, Orion 12-11+27 486 -> 431; but Pleiades 2025-10+11 701 -> 865, Orion 12-27+11 438 -> 497 |

k = 4 rejects less (30 sigma: up to 27% left). Followed by `nn_denoise` the faint-source and
artefact figures carry through unchanged.

- Bright one-night artefacts are removed and faint real sources untouched; sky offsets that
  differ between nights (Omega, Pleiades) are halved instead of averaged in.
- **Limit**: an artefact in the much deeper night cannot be contradicted by a shallow one
  (Omega 2026-08 against a night 4.5x noisier: 30 sigma kept 92%).
- **Open problem**: bright structure. Stars at different seeing disagree at fine scales by
  more than the 0.3 tolerance; taking the smaller coefficient biases them, which is the
  mixed signal error. Next: tolerance from the measured PSF difference (or only act where
  the mean is near the noise), then a version inside `merge.merge_previous_stacks` (the
  nights are already registered and flux-matched there), scored on the same pairs.
