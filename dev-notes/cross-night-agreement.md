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

## 3. In `--merge`: `--merge-consistency` (opt-in, 2026-10-22)

`merge.consistency_combine`, called from `merge_previous_stacks` (`--merge` and the
multi-session combine) when `--merge-consistency` is given. N nights, leave-one-out: night i
is left out of a starlet coefficient when it differs from the others' weighted mean beyond
`k` (3) x sqrt(noise_i^2 + noise_others^2 + (0.3 mean)^2), the others are within 3 sigma of
no signal, and night i is the stronger (with two nights: the "absent" rule of section 2).
Built on the way, each measured:

- **Noise per scale** from the difference to the other nights (shared structure cancels),
  split by the stacks' inverse-variance weights. A night's own MAD counted nebula structure
  as noise, 5-8x too high at coarse scales, so the test went lenient exactly there.
- **Carry**: a flag at >= 5 sigma (chance ~1e-6) also leaves the night out at that place
  (grown 2^(j+1) px) on every coarser *detail* scale; without it the unflagged coarse
  scales of a point artefact stayed behind as a ~3-sigma ring. Carrying the 3-sigma flags
  too dropped 12% of a pure-noise field's coefficients; rejecting whole image areas around
  flags made 32 px holes over 64% of a test field.
- **Coarse plane always the plain mean.** Carrying into it swapped in the other night's
  local sky/nebula level: a background step under every removed artefact (residual -11 to
  -17%) and signal-error swings of -51 to +14% (Orion).

Final, 8 directions vs the plain weighted mean (`merge_consistency` in
`tools/eval_cross_night.py`, results `C:\nn_work\synth\xnight6.json`):

| direction | sky error | signal error | one-night 10 / 30 / 100 sigma left |
|---|---|---|---|
| Omega 2025-11 + 2026-08 | -40% | +5% | -0.01 / -0.04 / 0.00 |
| Omega 2026-08 + 2025-11 | -38% | -11% | 0.95 / 0.91 / - (the other night 4.5x noisier) |
| Orion 12-11 + 12-27 | +15% | -22% | 0.07 / -0.09 / -0.04 |
| Orion 12-27 + 12-11 | -4% | +9% | 0.14 / -0.10 / -0.04 |
| Pleiades 10 + 11 | -43% | +9% | 0.44 / -0.13 / -0.04 |
| Pleiades 11 + 10 | -45% | -24% | 0.18 / -0.13 / -0.04 |
| Sunflower 03 + 04 | 0% | -18% | 0.87 / 0.48 / -0.04 |
| Sunflower 04 + 03 | +5% | +2% | -0.01 / -0.02 / 0.00 |

Mean sky -19%, signal -6% (median -4%), faint sources both nights show kept 100% in every
direction. Bright structure is the open problem: signal error worse in 4 of 8 (+2 to +9%),
from real differences between nights on bright nebulae that pass the "absent" test at some
scales. So it is opt-in, for a combine that shows an artefact, not a default. Synthetic
tests: `tests/test_merge_consistency.py`. Not yet tried on a real multi-night `--merge`
end to end (with the merge's own quadratic sky matching, which this evaluation did not use).

## 4. Real merges, and the fixes they forced (2026-10-22)

The held-out evaluation never put two *real* nights through `--merge`; doing that
(`--merge A.fits --merge-consistency` on Pleiades, Orion and Sunflower, against the same
merge without it) found three faults the synthetic injections could not show, each fixed
and re-measured:

1. **Star cores flattened** (Pleiades, nights 17 degrees apart): stars 40% wider at the same
   flux, the colour calibration lost 7 of 42 stars and moved blue by 8%. The softer night's
   core sat under 3 sigma at the finest scales, so the sharper night's core counted as a
   one-night feature. Fix: "absent" must hold at the next coarser scale too.
2. **Wavelet blocks on clipped cores** (Orion's M42): differences up to 500 sigma. The
   nights clip at different levels after flux scaling. Fix: the plain mean around any
   night's clipped plateau (`color_calibrate._clipped_weight`), not a brightness cut, so a
   bright glint stays rejectable.
3. **Blue taken from half the stars** (Pleiades): night A's blue was noisier and its stars'
   blue weaker after flux scaling; deciding per channel called real stars absent in that
   blue and swapped in the weak one (198 of 400 stars moved > 3% in B/G). Fix: decide on
   luminance, apply the decision to all three channels (13 of 400 left, by rotated
   diffraction spikes, which really are in one night only).

Final code, real merges against the plain merge: star widths 1.000 on all three; colour
calibration identical on Orion and Sunflower (same stars, scales within 0.1%), Pleiades
B x0.928 vs x0.904 with the same 42 stars; 0.13-0.60% of coefficients changed.

Final code, the 8 held-out directions vs the plain weighted mean
(`C:\nn_work\synth\xnight9.json`):

| direction | sky error | signal error | one-night 10 / 30 / 100 sigma left |
|---|---|---|---|
| Omega 2025-11 + 2026-08 | -39% | -0.1% | -0.02 / -0.04 / 0.00 |
| Omega 2026-08 + 2025-11 | -35% | -1.3% | 0.95 / 0.90 / - (other night 4.5x noisier) |
| Orion 12-11 + 12-27 | 0% | 0.0% | -0.09 / -0.09 / -0.02 |
| Orion 12-27 + 12-11 | 0% | 0.0% | -0.03 / -0.10 / -0.02 |
| Pleiades 10 + 11 | -42% | 0.0% | 0.44 / -0.14 / -0.01 |
| Pleiades 11 + 10 | -46% | -0.5% | 0.01 / -0.13 / -0.02 |
| Sunflower 03 + 04 | 0% | 0.0% | 0.84 / 0.20 / -0.17 |
| Sunflower 04 + 03 | +5% | 0.0% | -0.01 / -0.02 / 0.00 |

Signal error never worse (mean -0.3%), sky error -20% on average, faint sources both
nights show kept 100%, one-night 100-sigma artefacts removed in every direction (a slight
dip where they were, up to -17%). Still opt-in: a single real multi-night user merge
with a genuine artefact has not been seen yet, and on rotated nights it also trims
diffraction spikes, which moves the colour calibration slightly (Pleiades +2.7% in B).
