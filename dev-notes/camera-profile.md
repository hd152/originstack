# Camera profile (Origin178) — measurements and decisions

Measured 2026-10-07/08 on one Origin (unit `932f562381e97ce70`), every session on
disk: 32 at ISO 200, 10 at ISO 500, firmware 1.2.5282 / 1.3.5343 / 1.4.6084, sensor
20.9-35.2 °C. Tool: `tools/measure_camera_profile.py` (gain + CFA). The colour and
library numbers came from scratch scripts on linear (Phase 1-3) stacks of every
session with >= 20 lights and on 6 calibrated frames per session.

## 1. Gain — shipped

* **Estimator.** Two-point photon transfer on *consecutive raw lights*:
  `gain = (signal - pedestal) / (var(F1 - F2)/2 - floor)`, sky-end signal bins,
  sigma-clipped variance (`camera_profile.raw_pair_gain`).
  * The variance-vs-signal **slope** (noise_model.py's method) does not work on raw
    Origin data: values come in coarse steps, a MAD of stepped data takes only a few
    discrete values, and over one frame's narrow sky range the slope was mostly
    quantisation (session-to-session scatter 10-16%, R/G/B in one session up to 30%
    apart). Temporal differencing alone did not fix that; the two-point ratio did.
* **Result.** ISO 200: 0.0161 e-/ADU raw two-point, **±1.4% between 31 sessions**,
  no dependence on firmware or temperature (corr -0.03); the remaining correlation
  with sky level (+0.64) is the read + dark noise left in the variance. Fitting
  `1/g_meas = 1/g + floor/S` across sessions: **ISO 200 g = 0.0164-0.0166**
  (floor 7.5-20k ADU², residual 0.7-1.2% R/G, 2.9% B), **ISO 500 g = 0.0063-0.0066**
  (8 sessions). Shipped 0.0165 / 0.0064 with per-channel floors.
  Header `EGAIN`: 0.0742 / 0.0297 — 4.5x / 4.6x too high.
* **Read noise not shipped.** The Origin reuses library calibration frames (every
  session's `bias0009.fits` is the same file); their pair noise (31.45 ADU at ISO 200,
  identical to the digit in every ISO 200 bias = 0.5 e-) implies averaged masters.
  Nothing in the pipeline consumes read noise.
* **Check per run** (`verify_gain`, photometry runs only): the middle two lights;
  dim-sky sessions with the floor: Whirlpool 2026-04-06 0.0164, Sagittarius (2800 ADU
  sky) 0.0157, Needle (ISO 500) 0.0063 — all within the 15% tolerance.

## 2. CFA equalisation prior — not shipped

`cfa_frame_stats` through the real Phase 1 calibration, 8 frames per session:

| | within a session | between sessions |
|---|---|---|
| G2/G1 gain | ±0.00008 | ±0.0018 (ISO 200), ±0.003 (ISO 500) |
| 2x2 green offsets (rcd) | 0.35-0.6 ADU | 5-11 ADU std |
| 2x2 green offsets (malvar) | 0.4-0.8 ADU | 4-15 ADU std |

Sky level does not explain the between-session part (a linear sky model leaves the
std unchanged). A camera-wide value would be worse than the per-session
measurement, which `_measure_session_cfa` already does. Open lead: the session's
flat or sky colour.

## 3. Per-unit library — not built

* **Bad pixels.** Persistent outliers (> 6 sigma against the same-plane 3x3 median in
  >= 5 of 6 frames) over 33 sessions: every pixel hot in >= 16 sessions is already in
  the session dark's hot map, all recurring cold pixels are too; at most 8 recurring
  hot pixels would be added. (The dark map itself flags 214,576 pixels = 3.4% of the
  sensor — probably over-flagging; not investigated.)
* **Vignetting.** Fractional background shape after flat calibration (block median
  of 6 debayered frames, 30 fields without frame-filling nebulosity): pairwise
  correlation between sessions ~0.3, a common 0.5-0.7% rms pattern. Leave-one-out
  (map from the other sessions) after removing a quadratic — DBE removes at least
  that — improves the residual by **3% (R), 5% (G), 0% (B)**. Pinwheel's 1.5-2.8%
  R-G residual is not shared with other nights. `--vignette-map` stays for manual use.

## 4. Colour response — slopes shipped, curves not

* No usable published QE curve was available, and inventing one was not an option.
* Instrumental colour vs Gaia BP-RP on the linear stacks (the solar fit's star
  selection, SNR > 20): **B-R slope median 0.828, ±0.06 between nights** (bootstrap
  per session ±0.02-0.10); **G-R slope 0.27-0.77, median 0.40 — night-dependent**
  well beyond its bootstrap error (Crab 0.27±0.01, Needle 0.61±0.04; ISO 500 nights
  0.45-0.61; some April ISO 200 nights high). Unexplained.
* **Decision test.** Sessions with >= 40 stars, 6-14-star random subsamples, white
  point against the session's full free fit, prior from the *other* sessions:

  | stars | B-R free / prior (median, p90 mag) | G-R free / prior |
  |---|---|---|
  | 6 | 0.043 / 0.029, 0.113 / 0.073 | 0.028 / 0.024 |
  | 10 | 0.034 / 0.024, 0.091 / 0.059 | 0.021 / 0.020 |
  | 14 | 0.030 / 0.022, 0.077 / 0.054 | 0.018 / 0.019 |

  So `fit_channel_scales_solar(slope_prior=)` keeps the profile's slopes and fits
  only intercepts with 6-14 good stars (a free fit is unchanged at >= 15).
* **Saturation guard fallback.** On a sparse field the 99.99th percentile is the sky,
  so `0.6 x p99.99` sat below every star's sky-inclusive peak and rejected them all;
  when that leaves too few stars, saturation is judged at `sky + 0.7 (max - sky)`.
* **Effect on the 28 real stacks** (scales with/without the prior, COLCAL undone):
  25 unchanged to the last digit; Duck and NGC 2244 2026-03-12 now calibrate by a
  normal free fit (70 / 23 stars, rescued by the guard fallback; R 0.709, B 1.06,
  in line with the other nights); the 21-frame RA 12h27 field calibrates through the
  prior (10 stars) where it had none; M37 still declines (below).
* **Not fixed here:** rich clusters (M37, NGC 2244 2026-03-12, Duck) decline because
  `match_gaia_field` keeps the brightest few hundred Gaia stars (G 10.8-12.9), all
  saturated in 20 s subs. A fainter magnitude window for colour calibration would fix
  it.
