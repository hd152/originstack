# Changelog

All notable user-facing changes to OriginStack are documented here. Format
loosely follows [Keep a Changelog](https://keepachangelog.com/); versions
match the `VERSION` file and `v*` git tags.

## [Unreleased]

### Changed

- **Softening and noise gap investigated; README corrected.** The 2.2.2 note that OriginStack's stack is wider than its own
  single frames (so registration is to blame) was a measurement artifact: Siril's stack measures the same 13% wider than
  its own registered frames, because Gaussian fits to single low-signal frames are biased narrow by noise. Ruled out as
  causes of the extra width (Omega only) and the noise gap: registration accuracy (~0.1 px translation, 0.003 degrees of
  rotation, scale 4e-5 between frames), the warp kernel (Lanczos-3 against cubic spline: identical to two decimals),
  rejection method and strength, patch and inverse-variance weighting, local normalisation, and debayer method. Found:
  a single frame after calibration and debayering is ~9-10% noisier than Siril's in green and blue and ~11% quieter in
  red (Malvar-He-Cutler against RCD). The rest of the stack-level gap is still unexplained.

## [2.2.2] - 2026-09-21

### Added

- **Advanced features page** on the website (`docs/advanced.html`): what each advanced feature does and its flag, real
  session-report, aberration and dither figures from the Omega run, and an honest list of what was tried and did not work.
- **Linux build.** `packaging/build_linux.sh` produces `OriginStack-<version>-linux-x64.tar.gz` (+ `.sha256`): the desktop
  app as a PyInstaller bundle with `install.sh` (per-user install into `~/.local/share`, an application-menu entry, and
  `--uninstall`). `packaging/verify_build.sh` runs it under a display and checks that it starts, that the native Rust
  kernels loaded rather than the numpy fallback, and that a real multi-worker stack completes. The release workflow builds
  it on `ubuntu-22.04` (glibc 2.35) and attaches it to the release; it can also be run by hand from the Actions tab.

### Changed

- **The Siril comparison was wrong about sharpness, and is corrected.** It reported OriginStack's stars as tighter than
  Siril's, from comparing each stack's FWHM over its *own* detected star list; which stars were picked moved the number
  by more than the claimed difference. Measured on the same stars in both stacks, each on its own pixel grid
  (`tools/common_star_fwhm.py`, tested on synthetic fields of known width), OriginStack's stars are 13% wider on Omega and
  about equal on Sunflower and Sculptor. (An earlier version of this note said the stack was wider than its own single
  frames, so registration was to blame; that was a measurement artifact, withdrawn under Unreleased below. Registration checks out:
  ~0.1 px translation and 0.003 degrees of rotation error between frames.) Three
  sessions (Omega 114, Sunflower 158, Sculptor 532 frames) now also report memory and disk: OriginStack's peak memory is
  flat at ~17 GB from 114 to 532 frames while Siril's grows from 6.6 to 11.2 GB, Siril is as fast on Omega and 1.2-1.8x
  faster on the larger sessions, and OriginStack's stack is 1.0-1.6x noisier per pixel. README and website say so.
  `tools/bench_vs_siril.py` measures memory and disk and uses the shared-star metric.

### Fixed

- **Linux: Phase 1 hung forever with the native extension built.** Process pools used the platform default start method,
  which is `fork` on Linux before Python 3.14. The workers were copied after the Rust kernels' thread pool had started, so
  each inherited that pool's state without its threads and waited forever on its first parallel call (a packaged build sat
  for two hours with four idle workers). Every pool now uses `spawn`, as Windows always did (`src.utils.mp_context`), with a
  test that fails if a pool is added without it.
- **Desktop app off Windows:** logs went to the current directory; they now go to `$XDG_STATE_HOME/OriginStack/logs`
  (`~/Library/Logs` on macOS). Error and confirm-on-close dialogs did nothing on Linux; they now use Tk's message boxes. The
  window icon and fonts are set per platform.

## [2.2.1] - 2026-09-21

### Added

- **`--offline`: no network requests at all.** A normal run may look up a target name on SIMBAD when the built-in table
  does not know it -- the session name, the FITS `OBJECT` header, and the name derived from the folder -- and the optional
  features (plate solving, annotation, photometry, catalogue colour calibration, comet ephemerides) go online when enabled.
  `--offline` (also a checkbox in the desktop app's Core options) switches all of it off, with a note in the log for any
  enabled feature it skips. The guard sits in the low-level HTTP helpers, so every service is covered, and a test counts
  real connection attempts. The README has a new "Network use" table.
- **Project website** at https://hd152.github.io/originstack/ (GitHub Pages, from `docs/`).

### Fixed

- **`originstack.py --help` crashed** with `ValueError: unsupported format character` since 2026-09-19: the
  `--cfa-drizzle` help text contained a bare `%`. Escaped, and a test now formats every help string.
- **Desktop app:** the "Show expert options" checkbox was clipped in its sidebar; it now reads "Expert options".

### Changed

- **Release workflow:** each SignPath signing step waits up to 2 hours for the manual approval the Foundation requires
  (the default was 10 minutes). Signing stays off until the project is approved.

## [2.2.0] - 2026-09-21

### Changed

- **Debayer is ~2.8x faster; a 158-frame run goes from 135 s to 126 s.** The G1/G2 gain and the four 2x2 green
  offsets that the per-frame equalisation re-measured on every frame (six sigma-clipped medians over ~100 MB of
  strided reads, memory-bandwidth bound) are now measured once per session from eight frames spread through it and
  applied to every frame. Debayer drops from ~1.26 s to ~0.45 s per frame under 16 workers. The stack differs from
  the per-frame path by ~0.1 ADU against 95-164 ADU of noise; FWHM and noise are identical. Falls back to the
  per-frame path for short sessions (< 12 lights), GPU, non-Malvar debayer, or when the sampled frames disagree.
  `--no-session-cfa-eq` restores the old behaviour.
- **Temporary frame memmaps are no longer flushed to disk** just before being deleted (12+ GB of pointless serial
  writes on a 158-frame session). The final stack is bit-identical; checked on Windows and on Linux (WSL2).
- **Desktop app: expert options are hidden by default.** Diagnostics, fine-tuning and experimental options appear
  only after ticking "Show expert options". The CLI is unchanged.
- **README:** a "Compared with Siril and DeepSkyStacker" section with timings, star sharpness and matched-sharpness
  noise from two Origin sessions, including what the numbers do not show.

### Added

- **`tools/bench_vs_siril.py`** reproduces the Siril comparison on any folder of Bayer FITS lights.
- **Optional code signing in the release workflow** (SignPath Foundation): signs the exe and installer when the
  `SIGNPATH_ORGANIZATION_ID` variable is set; releases stay unsigned until then. See `packaging/README.md`.

## [2.1.1] - 2026-09-20

### Added

- **Windows installer.** Releases now include `OriginStack-<version>-setup.exe` (Inno Setup): a per-user install with a Start Menu entry and an uninstaller, so the app and its `_internal` folder always arrive together. Running `OriginStack.exe` from inside the zip preview failed with "Failed to load Python DLL" because only the exe was extracted. The release workflow installs the built setup silently and runs the packaging verification against the installed app before attaching it.

### Changed

- **README refresh.** New sample images (Whirlpool, Omega Nebula, Orion Nebula, Sagittarius Star Cloud), a rewritten Performance section with the current end-to-end benchmarks, and a shorter introduction.

## [2.1.0] - 2026-09-20

### Added

- **`-o` accepts a folder.** Give an existing folder, a path ending in a separator, or a name with no
  extension and the FITS (and its matching JPG) are named `<session>_stacked` inside it, creating the
  folder if needed and adding `_2`, `_3`... rather than overwriting an existing pair. Also applies to the
  desktop app's output field.
- **`--drizzle-method splat`**: the original area-overlap drizzle. Every input pixel is a square drop of
  side `--drizzle-pixfrac` x scale, deposited into the output by exact overlap. About 6x faster than the
  default resample path (22x with a small pixfrac), no Lanczos ringing, but softer -- on a synthetic
  dithered scene its error against truth was 40-44 against 35 for resample, so it is opt-in, and best with
  many dithered frames and a pixfrac below 1. Ignored (with a message) for `--drizzle-kernel psf/magic`
  and `--elastic-registration`.

### Removed

- **Denoisers that did not earn their place.** A ground-truth benchmark
  (`tools/bench_denoise_quality.py`: synthetic nebula/galaxy/star-field
  scenes with a known clean image, noise correlated like real stacks) found:
  - **NLM** cut star peaks to 50% of their true brightness and gave ~11x the
    input error inside star cores, at 17 s/MP. Removed.
  - **MMT** erased ~93% of fine structure at its default strength (a median
    cascade deletes thin filaments by construction) and had been the
    `--auto` primary for galaxy, globular-cluster and planetary-nebula
    targets. Removed; those targets now use the curvelet wavelet.
  - **BM3D** was the best quality but slow (14 s/MP) and needs a
    non-commercial-licensed package; its pure-scipy fallback was measurably
    worse than the package. Removed with its Rust DCT/block-matching kernels
    (`astro_native` 0.23.0) and the `bm3d` entry in `THIRD_PARTY_NOTICES.md`.
  - **Plain (non-adaptive) wavelet** and `adaptive_wavelet_denoise` are
    merged into the curvelet denoiser, which is numerically identical to
    plain BayesShrink at `directional_protect_strength=0`. (See *Changed*
    below: `--denoiser wavelet` is now the one canonical name and
    protection is `--wavelet-protect`.)
    `--denoise-strength`, `--denoise-strength-calibrate` and the Noise2Self
    calibration module (`self_supervised_calibration.py`) only served the
    removed non-adaptive path and are gone too.
  `--denoiser nlm|mmt|bm3d` are now rejected. Presets and config files that
  named them fall back to the curvelet default (unknown config keys are
  ignored).

### Changed

- **Multi-session (hierarchical) runs post-process once.** Each session's stack used to go through the whole of Phase 4 and then the combined stack went through it again, although the combine only reads the linear per-session FITS. Sessions that will be combined now skip Phase 4 (and their throwaway preview stays linear); it runs once on the combined stack. Single sessions, filter-split groups and the combined stack are unchanged.
- **Faster Phase 1, identical output** (`astro_native` 0.27.0). Calibration (bias, scaled dark, flat, finite
  check, clip) and both hot-pixel passes (Bayer mosaic and RGB) each run as one native call instead of
  ~6-20 full-frame numpy passes: per 2048x3056 frame calibrate 40 -> 9 ms, Bayer hot-pixel fix 715 -> 29 ms,
  hot-pixel map replacement 666 -> 12 ms, RGB hot-pixel fix 1174 -> 52 ms (single thread). These were about a
  third of Phase 1's worker time under 16 workers, where memory bandwidth is the limit. Bit-identical results.
  Per-frame pre-gradient removal (on in saved `--auto` configs) is fused too: 349 -> 8 ms per frame.
- **AVX2 median, one portable build** (`astro_native` 0.30.0). The 3x3 median (and through it the hot-pixel repair) picks an AVX2 version at run time on CPUs that have it: median 1.7x, hot-pixel repair ~15-20%. Same results, and the wheel still runs on any x86-64 CPU. Release symbols are stripped.
- **Faster alignment** (`astro_native` 0.29.0). Phase 3 now warps only the common crop of each frame instead of the whole frame (identical pixels), and the rotated Lanczos-3 warp takes its weights from an interpolated table (1.5x; 99.985% of output values identical, the rest one ulp off). Fireworks session (141 frames, 16 workers) alignment: 17.8 s -> 13.6 s (crop) -> 10.9 s (table).
- **Faster debayer stage** (`astro_native` 0.28.0). The sigma-clipped medians, G1/G2 equalisation and Bayer-grid equalisation no longer copy strided views or the 75 MB RGB frame, and hot-pixel removal, gray-world white balance and the luminance recompute work in place / in one pass. Real session (148 frames, 16 workers), ms per frame: debayer 2207 -> 1002, hot-pixel removal 489 -> 310, white balance 255 -> 187, luminance 259 -> 68. Same output as before.
- **Faster drizzle, identical output** (`astro_native` 0.26.0). The default resample path now warps,
  weights and accumulates each frame in one native pass instead of building a temporary image and adding
  it under a lock: 385 to 269 ms per 2x frame, and 1193 to 305 ms with `--drizzle-pixfrac < 1` (which
  built seven full-size temporaries per frame). Bit-identical to the previous result.

- **`--denoiser wavelet` and `curvelet` are one denoiser.** They already ran
  the same function; `wavelet` just forced structure protection to 0. `wavelet`
  is now the canonical name (`curvelet` stays as an alias), and protection is its
  own option, `--wavelet-protect 0-1` (default 0.6; 0 = plain BayesShrink).
  **Behaviour change:** an old command line using `--denoiser wavelet` now gets
  0.6 protection; `--denoiser wavelet --wavelet-protect 0` is the old behaviour.
  `--skip-step` accepts `wavelet` or `curvelet`.
- **`--merge` and hierarchical combines weight by measured noise, and match flux
  scale first.** Stacks from different exposure/ISO were averaged in raw ADU and
  weighted by frame count. Each previous stack is now mapped onto the current
  stack's flux scale (robust per-channel gain and sky offset) and weighted by
  inverse noise variance; 3 px of each warped footprint's rim is trimmed so it
  no longer draws an outline. On five real Fireworks Galaxy sessions (10 s ISO 200,
  25 s ISO 500, 30 s ISO 200) the fitted scales matched what exposure and ISO
  predict, and pixel noise fell 37% (38.0 to 24.1) with 376 frames.
- **Hierarchical (multi-session) runs post-process the combined stack.** It used
  to be only a linear stack with a linear preview. It now goes through Phase 4
  with the reference target's own effective settings, and the reference grid is
  the stack with the most integration time (was: most frames).
- **The galaxy preset's preview black point is 1.0, not 3.0.** At 3.0 (2.6 after
  the depth rule) a low-surface-brightness disk rendered black and only the core
  survived, though the linear data held the whole galaxy.
- **DBE re-admits gradient patches.** The sampler rejected any patch brighter
  than a luminance-based `sky + 2 sigma`, so in the strongest channel most patches
  (R: 165 accepted vs ~715 for G/B) and all the edge ones were dropped; the edge
  glow survived DBE (R +114 ADU) while G/B were removed completely. A polynomial
  fitted to the accepted patches now re-admits candidates it explains, growing
  outward. Every edge is within +-3 ADU afterwards and the galaxy disk is
  unchanged or slightly brighter over sky. This, not a narrow strip, was the
  coloured band along the bottom of the preview.

- **Faster default path, identical output** (`astro_native` 0.25.0). Two kernels that run on every
  stack got faster without changing a single output value: **white balance** is now one
  native pass instead of ~8 numpy temporaries (543 to 78 ms per frame single-thread), and the
  **Lanczos-3 warp** used for alignment and drizzle computes its tap weights with three trig
  calls per axis instead of twelve and reads each RGB tap row once (2187 to 625 ms for a rotated
  frame). Both are bit-identical to what they replace, checked on a real frame. Alignment,
  the largest single block of a real run, was compute-bound in that warp, not disk-bound as first
  suspected.

### Fixed

- **Gray-world white balance is more accurate.** Channel means were accumulated in float32 over
  ~6M pixels, which drifted up to ~1.5% on a real frame's red channel and skewed the gains; they
  are now accumulated in float64 (native and numpy paths identical). Colour balance shifts very
  slightly.
- **Frames a rotating field breaks are registered, not stacked at the wrong
  position.** The pyramid shift is translation-only, so on an alt-az mount it
  returns garbage for a rotated frame: 34 of 148 frames (40 of 117 in another
  session) were flagged as outliers and stacked at zero or a coarse shift. They are
  now registered by the blind rigid star match. Real session: pixel noise 39.7 to
  38.0, star sharpness +17%.
- **The registration residual gate could reject a whole session.** A saved config
  that enabled `pre_gradient_removal` raised the measured SNR from 1.7 to 5.0,
  which shrank the adaptive threshold to its 1.5 px floor against a true median
  residual of 1.83 px: 144 of 145 frames were rejected and one frame was stacked,
  with only a log line. When the absolute gate would fail most of a session, frames
  are now judged against the session's own median + 4 robust sigma (still capped).
- **Saved configs could not be loaded on Windows.** `save_effective_config` wrote
  `log_file = "C:\Users\..."` unescaped, which TOML rejects ("Invalid hex
  value"); the run then fell back to defaults with only a warning. Strings are
  now escaped and round-trip through `tomllib`. Note that a saved config also
  bakes in `--auto`'s derived choices (`pre_gradient_removal`, `galaxy_mode`, ...)
  as if set by hand, which changes Phase 1 metrics: it is not equivalent to
  re-running with `--auto`.
- **Fireworks Galaxy (NGC 6946) was not recognised**, so `--auto` never skipped
  `sky_residual` and the galaxy was fit away as background. Added, and any unknown
  target name containing "galaxy" now infers the galaxy type.
- `remove_stars`' FWHM fallback returned NaN when no frame had a measured FWHM
  (`np.median([]) or 4.0` never used its default).
- `tools/lint_conventions.py --git` crashed on Windows decoding git output as
  cp1252; it now decodes UTF-8.

- **`--variance-stabilize` (and `--auto`'s rule enabling it) had no effect on
  the default denoiser.** It was only wired into the wavelet paths;
  `--denoiser curvelet`, the default, silently ignored it. It is now applied
  by the curvelet denoiser, so `--auto` runs on curvelet targets now really
  use it (benchmark: faint-region error 0.87 → 0.85 at unchanged detail retention).

### Added

- **`--banding-removal`: row/column banding removal per calibrated Bayer frame.**
  Per colour plane against its own trend, highlight-protected against the *local*
  level, significance-gated so a clean frame is left essentially untouched. Noise
  is measured from neighbour differences (vignetting inflates a plane MAD: 786 vs a
  true 455 ADU on a real frame). Real Origin frames carry ~12 ADU of genuine
  per-frame row banding, about 3% of pixel noise -- negligible after stacking.
- **Per-frame transparency and `--transparency-min`.** The median flux of a fixed
  star ensemble against the reference, normalised to the session median, measured
  after registration (real session: 0.86 to 1.23). Thin cloud dims stars without
  changing FWHM or SNR much, so the quality gate can pass it.
- **`--session-report`**: `<output>_session.png` and `.csv` plotting FWHM,
  transparency, SNR, background, drift, field rotation, ellipticity, residual and
  temperature against time, with drift rate, periodic tracking error, rotation rate
  and focus-vs-temperature drift reported. Drift is referenced to the frame
  *centre*: a transform's raw translation is the displacement of the (0, 0) corner,
  which under 0.5 deg/min of field rotation swept an arc that read as 23 px/min of
  "drift" on the first real run. Frames the residual check skipped (it samples
  ~20% of a large session) plot as gaps, not as a residual of zero.
- **`--distortion-model`**: one radial distortion for the whole session, fitted
  from every frame's star matches and applied as per-frame displacement fields in
  the same resample pass elastic registration uses. Applied only when it reduces
  the residual on held-out frames. On the Fireworks session it found a1 = +0.0007
  (no measurable distortion in the Origin's optics) and correctly declined to apply
  anything.
- **`--noise-validate`**: odd/even half-stacks give the full stack's measured
  noise map (`<output>_noise.fits`) and a local-correlation map of which structure
  is repeatable (`<output>_consistency.fits`). Real session: stack noise R 43.2 /
  G 36.5 / B 36.9 ADU, 1.16x what per-frame noise / sqrt(N) predicts, with 61% of
  the frame showing repeatable structure.
- **`--moving-objects` / `--moving-objects-stack`**: asteroid-like movers found by
  linking per-frame residuals in velocity space, with an optional stack along each
  track. Real session: 1497 candidate detections, 0 linked tracks -- no false tracks
  from the clutter.
- **`--lightcurve-analysis`**: Lomb-Scargle period search (with FAP) and a
  box-least-squares + trapezoid transit fit on `--photometry-timeseries` output.
- **`--cfa-drizzle`** and the native `cfa_drizzle_frame` kernel (`astro_native`
  0.24.0): recombine each frame's measured Bayer samples. 407 s to 26 s on 148
  frames. On a well-sampled real stack (FWHM 4.9 px) it does not help: luma noise
  -13% but colour noise 2x and no sharpness gain, so it stays opt-in for
  undersampled data.
- **`--starless-process`** (denoisers, local contrast and deconvolution on a
  starless layer, stars added back) and **`--layered-stretch`** (preview black and
  white points from a starless copy). Measured on a real stack: the starless
  denoise gained ~7% lower noise; deconvolving the starless layer did *not* help at
  this SNR (RL fragmented the disk, sparse rang); the layered stretch reveals the
  full galaxy disk that the plain stretch leaves near-black.
- **Edge-band correction** (`--skip-step edge_bands`) before the final sky
  flattening, as a backstop for sky excess that rises toward the frame edges.
- **`--transient-detect REF.fits`: find what changed between two epochs.**
  Stacking software answers "what does my target look like?"; this answers
  "did anything *appear*?" — novae, dwarf-nova outbursts, supernovae,
  asteroids crossing the field, variable stars. Uses ZOGY proper image
  subtraction (Zackay, Ofek & Gal-Yam 2016), which cross-convolves each epoch
  with the *other's* PSF so stellar residuals cancel even when the two nights
  had different seeing. A plain `new - ref` leaves a bright dipole at every
  star, scaling with stellar brightness — the artefacts land exactly where
  the interesting objects are. The reference is registered with the same
  blind, rotation-agnostic star match `--merge` uses. Writes
  `<output>_difference.fits`, `<output>_scorr.fits` (significance in sigma)
  and `<output>_transients.csv`, with sky coordinates when a WCS is present.
  `--transient-threshold` sets the cut (default 5σ); the score is properly
  calibrated, so that is a real significance rather than an arbitrary number.
  Diagnostic only — it never alters the stack, and a failure can't cost you
  the output. The reference must be a **linear** stack (the main output FITS,
  `RAWSTACK=True`), exactly as for `--merge`: a post-processed one mismatches
  the flux scale and every star in the field would report as a transient, so
  it is refused with a message rather than compared. Where the warped
  reference has no coverage (a cross-night pair on an alt-az mount differs by
  field rotation, leaving empty corners) results are masked out and the log
  says what fraction of the frame the comparison covers — without that, stars
  in those corners came out as confident "brightenings" (6 false candidates
  on a synthetic 9° rotation, 0 with the mask). The registration residual fed
  to the astrometric term is now *measured* from the matched stars, with 0.3 px
  as a floor; it used to be a hard-coded 0.3 reported as a measurement.
- **Astrometric noise is propagated, not ignored.** `S_corr` includes source
  (Poisson) *and* astrometric noise. The latter is what makes this usable on
  real data: registration is never perfect, and a sub-pixel slip leaves a
  residual proportional to the local image gradient — largest at bright
  stars. Without that term every bright star in the frame reports as a
  high-significance transient.
- **`--bg-method physical`: an experimental sky background model from first
  principles.** Models the sky as a sum of components whose *spatial shapes
  are fixed by geometry* — scattered moonlight (Krisciunas & Schaefer 1991),
  van Rhijn airglow, zodiacal light, ground-source skyglow — leaving one
  **non-negative** amplitude per component free, so unlike a free-form
  surface it has nowhere to put a nebula. `--light-pollution-azimuth` aims
  the skyglow term. Ephemerides are closed-form rather than astropy, whose
  `AltAz` path needs the IERS tables the packaged app excludes; validated
  against astropy at **0.009° for the sun and 0.05° for the moon**.

  **Read this before using it.** On a real 1° deep-sky field the model does
  not work, and it now detects that and declines. Measured on a real Lagoon
  session: the zenith angle varies by only 0.94° across the whole frame and
  the azimuth by 1.5°, so every component map is essentially flat, the fit
  has nothing to grip, and subtracting it made the corner-to-corner gradient
  **worse** (67 → 111 ADU) where DBE removed 68% of it. Sweeping
  `--light-pollution-azimuth` through all 360° moved the residual by under
  0.01 ADU. This is structural, not a tuning problem: physical sky components
  vary on ten-degree scales, so a narrow field's gradient is dominated by
  *instrumental* effects — vignetting, amp glow, filter gradients — that a
  model of the sky cannot represent by construction. `remove_physical_sky`
  therefore measures whether it actually flattened the background and falls
  back to DBE when it did not. Expect it to earn its place only on wide
  fields; on typical deep-sky framing it will stand aside. The field-size
  gate runs *before* any geometry is built, so a telescope field is declined
  in effectively no time (it used to cost ~1 s and ~580 MB first), and the
  reason reported is the real one — too narrow a field, a failed fit, or a fit
  that did not help — rather than one sentence for all of them. Azimuth and
  helio-ecliptic longitude are interpolated on the circle, so a field
  straddling due north no longer gets a bogus 360° ramp; and a failed
  non-negative fit declines to DBE instead of silently substituting the
  unbounded fit described below as harmful.

  Two earlier claims are corrected by that testing. **DBE does not eat
  nebulosity** — it retained 99.5% of the Lagoon's signal; the session that
  motivated this work was damaged by the `sky_residual` *residual passes*, a
  different step, which `--auto` already skips for extended targets. And the
  "98% of a synthetic nebula preserved vs a blind surface that eats it" result
  holds only for a synthetic gradient *built from the model's own basis*; it
  does not generalise to real data.
- **Non-negativity is load-bearing.** Across a real field the component maps
  are nearly collinear, so an *unbounded* fit synthesises a bump from large
  cancelling coefficients and inverts a nebula (measured at −25% preservation,
  worse than doing nothing). Constraining every coefficient to ≥ 0 removes
  that freedom: a non-negative sum of monotonic ramps stays monotonic and
  cannot have an interior maximum.
- **Honest attribution.** The fitted coefficients would let the model report
  what a gradient was *made of*, but over a typical field the component maps
  differ by a few percent and the split between them is not identifiable
  (design-matrix condition number ~1e5–1e7) even though the removal is valid.
  Rather than print a confident-looking breakdown that means nothing, it says
  so and reports the condition number.
- **`--uncertainty-propagate`: error bars that survive post-processing.**
  Phase 3 already computes an exact per-pixel standard error for the linear
  stack (`--uncertainty-map`, the Gauss-Markov estimator's own
  `1/sqrt(sum 1/var)`), and Phase 4 then discarded it — every background
  extraction, denoise and deconvolution reshapes the noise field, so that map
  no longer describes the image you actually look at. This carries it through
  by pushing `K` noise realizations (`--uncertainty-realizations`, default 8)
  through the **unmodified** post-processing chain and measuring the per-pixel
  spread. Monte Carlo rather than analytic propagation deliberately: nine of
  the Phase 4 steps are nonlinear denoisers and four are iterative
  deconvolvers, several spatially adaptive, so no closed-form Jacobian exists
  for most of the chain — and this stays correct automatically when a new
  denoiser is added. Writes `<output>_sigma_final.fits` (propagated standard
  error) and `<output>_snr.fits` (per-pixel signal-to-noise above sky), and
  logs what fraction of the frame clears 3σ and 5σ. **Accuracy:** each
  realization carries √2× the real noise (the stack already holds ~σ; the
  realization adds another σ), so steps that estimate their parameters from
  the data denoise it slightly harder than the real image and the spread comes
  back a little low. Measured against this project's own `wavelet_denoise`
  across σ ∈ {1, 4, 12} and thresholds ∈ {2, 3, 5}: 0.93–1.03× the true
  output noise — a few percent, well inside the ~25% Monte Carlo error at the
  default K. A probe at half amplitude reports how noise-scale-dependent the
  chain actually was, and the log warns if it is strongly so. Holds a copy of
  the linear stack plus two float64 accumulators (~1.1 GB at 24 MP).
- **Clamped-pixel detection.** Pixels the chain pins to a constant (the sky
  pedestal lift and the non-negativity clips both do this) come back with
  exactly zero propagated sigma. Those are reported as *undefined* confidence
  (`NaN`), not near-infinite confidence, and the clamped fraction is logged
  separately — a large one means the chain is flattening the image rather
  than measuring it.
- **`--error-aware-stretch SIGMA`.** Sets the preview JPEG black point at the
  SIGMA-confidence contour, so anything the propagated error bars cannot
  separate from sky clips to black instead of being lifted into apparent
  structure — the classic over-stretch failure where amplified correlated
  noise reads as nebulosity. Requires `--uncertainty-propagate`.
- **`--auto` suggests optional features instead of silently skipping them.**
  After classifying a target it prints a "Suggested for this target (not
  enabled)" block for features that carry a real cost or need input it cannot
  invent: `--bg-method physical` (only for extended targets with a session
  solve + GPS **and** a field of at least 5°, since below that it declines),
  `--transient-detect` when `--merge` already supplies an earlier epoch, and
  `--uncertainty-propagate` when an uncertainty map exists. It only acts on
  things that are cheap and harmless: `--stack-method ivw` now also writes
  `<output>_sigma.fits` (cheap, not free — the sigma kernel measured ~0.135 s
  against 0.049 s for the plain combine — and it does add a file to the output
  directory, which the log line says).
- **A project icon and logo**, generated from code by `tools/make_icon.py`
  rather than committed as opaque binaries. `packaging/icon.ico` now carries
  six sizes (it lacked 64 and 128, which Windows uses for Alt-Tab), and the
  packaged app now bundles it so the *window* icon shows, not just the
  taskbar/Explorer one.

### Changed

- **Multi-night directories are no longer pooled into one stack by default.**
  Pooling every subfolder's lights was unconditional, on the reasoning that a
  multi-night session directory is the common case. It is — and that is
  exactly the case where pooling costs the most. On an alt-az mount the field
  rotates as the target tracks, so sessions at different hour angles are
  rotated relative to each other, and the common crop keeps only what *every*
  frame covers. Measured across five real Lagoon sessions: 26.8° of rotation,
  42% of the frame discarded. OriginStack now predicts that rotation from
  metadata alone (each session's `info.json` plus the first and last light
  header — free, and decided before any stacking starts) and stacks the
  sessions separately when it exceeds 3°, merging them with rotation
  awareness afterwards. The prediction reports *avoidable* rotation — the
  total spread minus the most any single session covers — because a session
  merely 15 minutes long already sweeps ~3.6° and would otherwise be advised
  to split away from itself. Explicit `--combine-sessions`, `--hierarchical`
  and `--mosaic` still win — including when they come from a saved `--config`
  file, which the first version silently overrode — and unreadable metadata
  keeps the old pooling behaviour rather than guessing, now *saying so*
  instead of pooling without a word. The prediction no longer depends on
  astropy's IERS tables (see Fixed), so a source checkout and the packaged
  app choose the same way. The `--combine-sessions` / `--hierarchical` help
  text, which the GUI shows, describes this instead of "the default".
- **Per-subfolder stacks are combined with rotation-aware registration.** The
  hierarchical combine registered each stack against the first with a pure
  (dy, dx) translation, which cannot represent rotation at all — so on the
  same alt-az data above it left stars smeared toward the corners, silently,
  since the shift search returns a plausible-looking number either way. It
  now routes through the same blind, rotation-agnostic star match `--merge`
  uses, weighting each session by its own `NFRAMES` so a deep session is not
  diluted by a shallow one, and taking the deepest stack as the reference
  grid rather than whichever subfolder sorted first. A failure now stops with
  the per-target stacks kept on disk and a pointer at `--merge`, instead of
  falling back to producing the smeared composite this replaces. The combined
  output now carries the *reference* stack's own integration time and dates
  plus `RAWSTACK=True`, so it can be fed into `--merge` or `--transient-detect`
  — previously it lost every aggregate and was not marked linear, silently.
  Choosing the reference grid reads header metadata rather than loading every
  stack's pixels to read one integer.
- **`--uncertainty-realizations` and `--transient-threshold` reject nonsense.**
  Fewer than 2 realizations has no spread to measure (it used to be clamped to
  2 without a word), and a threshold of 0 or below admits every pixel and
  reports the 500 noisiest as detections.
- **ZOGY is ~3.4× faster on the Origin's frame size.** The 1096-px axis is
  8 × 137, which drops pocketfft onto its slow Bluestein path; frames are now
  padded to an FFT-friendly size (which also moves the FFT's circular
  wraparound out into cropped padding) and intermediates are released as soon
  as they are consumed. `S_corr` is unchanged to 2e-5σ in the interior, and
  detection is a single pass rather than one full-frame scan per candidate.

### Fixed

- **`--auto` no longer carries one target's tuning into the next.** In a
  multi-target run every target shared one settings object. Most settings
  were unaffected, because the advisor recomputes each one per target — but
  `--skip-step` is *appended* to, guarded by "not already present", so once a
  nebula target added `sky_residual` every later target saw it there and left
  it alone. A globular cluster stacked from the same directory silently
  skipped its sky-residual correction because a different object wanted that,
  with no log line to say so. Each target is now advised from the settings
  you actually passed. The reset also removes attributes a target *created*
  (a measured photon-transfer gain, an originvision defect flag) — the same
  leak one layer further out, which would have handed a calibration-less
  target the previous target's gain.
- **The packaged app and a source checkout no longer stack the same directory
  differently.** `packaging/originstack.spec` strips astropy's IERS tables, but
  the new rotation prediction called `Time.sidereal_time("apparent")`, which
  needs them. In the exe it raised `FileNotFoundError`, a fail-soft `except`
  turned that into `None`, and the exe silently always pooled while a checkout
  split. Sidereal time is now closed-form (agrees with astropy to 0.0024°
  against a 3° threshold), and `altaz` falls back to it when astropy or its
  tables are missing — which is also why `--photometry`'s airmass term and
  `--fix-atmospheric-dispersion`'s zenith angle silently vanished in the exe.
  astropy is also told never to reach for the network mid-run.
- **`observing_geometry` handles Celestron Origin timestamps.** `astropy`'s
  `Time` rejects `2026-08-31T20:40:35-0700`, which is exactly what an Origin
  writes to `DATE-OBS` and `info.json`, so airmass, zenith angle and
  parallactic angle all returned `None`. `--photometry` silently lost its
  airmass extinction term (absorbed into the zero point — a real photometric
  error, not a missing nicety), so zero points may shift once it is applied.
  Offsets are converted to UTC, not stripped: discarding `-0700` is seven hours
  of Earth rotation.
- **Local contrast no longer carves a dark collar around bright stars.**
  Blurring at the enhancement scales smears a star's core outward, so just
  beyond the protected core the blurred luminance far exceeds the original and
  the enhancement subtracted real nebulosity: 6.1% mean darkening in an
  r = 8–22 px annulus on a real stack (14.2% worst). The detail source is now
  clipped at the 99th percentile — 0.5% / 1.8% — keeping ~81% of the contrast
  gain away from stars. **This changes every default output** (local contrast
  is on by default), so a re-stack will differ visibly around bright stars
  from an earlier one.
- **`--uncertainty-propagate` no longer overwrites real sidecars with noise.**
  The quieted realizations left `--aberration-report`, `--export-masks`,
  `--keep-intermediates` and the comet sidecars enabled, so each was rewritten
  once per realization under a swallowed stdout and the file left on disk was
  the *last noise realization*. `--denoise-strength-calibrate` is also now off
  during realizations (it was K × 9 extra full-image denoises).
- **`--transient-detect` catalogues get RA/Dec.** A bare `WCS(header)` on the
  `(3, H, W)` cube is 3-axis, still passes `has_celestial`, then fails on 2-D
  pixels — blanking every row of the one column that makes a candidate
  checkable against MPC/TNS/VSX. It is built with `naxis=2`, and a failed
  conversion is now announced instead of swallowed.

## [2.0.1] - 2026-09-15

### Changed

- **`--originvision` comet class is suppressed, not shape-gated.** The
  hand-tuned 8-connected-component blob shape gate (two val-set-tuned
  constants, maintained in both Rust and numpy) is gone; a top `comet`
  prediction is now simply demoted to the runner-up class, since the
  current checkpoint's comet head isn't trusted. `category_shape_gated`
  stays in the result dict (now meaning "comet was demoted").
- **`--originvision-timeout` / `--originvision-python` / `--originvision-script`
  removed.** They were inert no-ops left from the old subprocess scorer and
  guarded a flag spelling that only ever shipped in-process; passing them
  now errors instead of being silently ignored. `--originvision-model`,
  `--originvision-dir`, `--originvision-checkpoint` are unchanged.
- **`quality.compute_quality_metrics`** takes one `level={'full','quick','gate'}`
  argument instead of the `quick` / `gate_only` bool pair.
- **Smaller Windows bundle.** The PyInstaller spec now excludes two payloads
  nothing in the app reaches: `PIL.AvifImagePlugin` (the AVIF codec, ~7.5 MB
  `_avif.pyd` + libavif — previews are JPEG/PNG only) and `scipy.io`
  (MATLAB/WAV/NetCDF readers, ~2 MB). ~10 MB off the onedir, ~4 MiB off the
  distributed zip.

### Fixed

- **`--originvision` native inference releases the GIL** (`py.allow_threads`)
  around preprocessing + the `tract` forward pass, so `--originvision-workers`
  / `--originvision-score-all` actually parallelise on the native backend.
- **A malformed `--originvision-model` can no longer abort a run.** A panic
  inside `tract`'s parser is caught in the kernel and re-raised as a normal
  `RuntimeError` (previously a `pyo3_runtime.PanicException`, a
  `BaseException` that slipped past the advisory-path error handling), and a
  zero-dimension input array is rejected cleanly.
- **`--originvision` native and onnxruntime backends now agree** on a model
  whose graph-output count doesn't match its `head_order` metadata — both
  reject it (the native path previously truncated silently).
- **Collection quality sweep (`--quality-sweep`)** drops `quality_gate`'s
  absolute `hard_limit` stage for OSC frames scored via the half-resolution
  Bayer proxy (2×2 averaging shifts SNR/contrast/dynamic-range off the
  cutoff scale); the folder-relative stages still run, and unreadable frames
  are flagged directly.
- The native `astro_native` crate now commits its `Cargo.lock`, and CI's new
  `native` job builds the crate so the native-kernel parity suites
  (`tests/test_native.py`, native-vs-onnxruntime originvision parity) run on
  every push instead of only on a developer machine.

## [2.0.0] - 2026-09-07

### Breaking

- **`--astrollm*` → `--originvision*`.** The upstream model project renamed
  itself (`astrollm` → `originvision`), and OriginStack follows: every
  `--astrollm`, `--astrollm-score-all`, `--astrollm-model` etc. is now
  `--originvision*`, the `ASTROLLM_DIR` env var is `ORIGINVISION_DIR`, and
  `FrameInfo.metrics['astrollm']` is `['originvision']`. **No aliases** —
  a command line still passing `--astrollm` will error. `--astrollm-python`
  / `--astrollm-script` / `--astrollm-timeout` were already inert after the
  scorer moved in-process and are removed entirely.
- **Star removal is opt-in.** The `<output>_starless.fits` sidecar
  surprised users who never asked for it. It is **off by default** now;
  `--remove-stars` enables it. `--no-remove-stars` is kept as a hidden
  no-op so existing command lines don't error. `--auto` no longer turns it
  on for any target type. The main output (with stars) is unaffected
  either way.

### Added

- **`--originvision` inference runs fully in-process in Rust.** The new
  `astro_native.originvision_score` kernel does the whole path —
  preprocessing (percentile stretch, resize/centre-crop, comet shape-gate)
  plus the ONNX forward pass, via the pure-Rust `tract` runtime — with **no
  Python ONNX dependency and no extra DLL**. A source checkout without
  `ext/astro_native/` built falls back to a Python `onnxruntime` path;
  `onnxruntime` is no longer bundled in the packaged app. The exported
  model ships inside the package at `src/data/originvision.onnx`.
- **`--originvision-model PATH`** overrides the bundled model (e.g. to test
  a newer checkpoint). The bundled model is now "v4" — a from-scratch,
  no-SSL 7-task run (best-by-category checkpoint, epoch 15). vs the
  previous model it matches or beats on category, reject-recall and
  quality; it is ~0.07 lower on exposure-class accuracy.
- **`--photometry`: absolute aperture photometry on the linear stack.**
  Detects stars, cone-searches Gaia DR3 around the field (via the header
  WCS — a Celestron Origin `info.json` session solve or `--plate-solve`),
  cross-matches, and does per-channel **partial-pixel** circular-aperture
  photometry (supersampled aperture edge + robust sky annulus). Fits a
  robust per-channel photometric zero point
  `m_cal = m_inst - k·X + ZP + CT·(BP-RP - ref)` against Gaia G/BP/RP
  (RP→R, G→G, BP→B — a coarse OSC mapping, not a filter-matched
  transform), with an airmass term `X` from the site GPS + observation
  time in `info.json` when present (otherwise extinction folds into the
  zero point). Writes `<output>_photometry.csv` (per-star RA/Dec,
  flux/mag/mag-err/SNR per channel, saturation flag) and
  `MAGZP_R/G/B` / `MAGZPE_R/G/B` / `MAGCT_R/G/B` / `PHOTGAIN` header
  keywords. Flags: `--photometry-color-terms` (fit the per-channel colour
  term instead of a plain median), `--photometry-extinction-k` (override
  the nominal `k` = R 0.09 / G 0.15 / B 0.23 mag/airmass),
  `--photometry-gain` (electrons/ADU for the Poisson error term).
- **`--photometry-timeseries`: per-frame differential light curves.**
  Runs a separate pass over the registered subs (right after Phase 3),
  aperture-photometering a fixed Gaia star list on every frame and
  ensemble-differential-calibrating a per-frame zero point (removes
  transparency / airmass drift). Writes `<output>_lightcurves.csv` (one
  row per frame × star — MJD, airmass, per-channel mag/mag-err) and
  `<output>_lightcurve_stats.csv` (per star — mean, rms, MAD, reduced χ²
  vs a constant, and a `variable` flag). `--photometry-target "RA,DEC"` or
  `"px:X,Y"` marks and reports one star. Needs a session `info.json` WCS
  (`--plate-solve` runs too late for per-frame work); differential only,
  no absolute zero point.
- **Photon-transfer gain / read-noise from raw calibration frames.** When
  `--photometry`/`--photometry-timeseries` is set and no `--photometry-gain`
  is given, a Janesick two-frame difference over ≥2 bias + ≥2 flat frames
  estimates the sensor gain (e-/ADU) and read noise (e-), feeding a real
  Poisson term into the per-star photometric errors.
- **Quality-sweep score cache.** `--quality-sweep` writes a versioned
  `.sweepcache.json` at the sweep root, keyed by each frame's mtime+size,
  so re-sweeps of a mostly-unchanged collection only re-score new or
  changed frames. `--sweep-no-cache` ignores and does not write it.

### Changed

- **`--auto` now skips the sky-residual correction passes for emission and
  reflection nebulae too, not just galaxies.** `remove_sky_residual`'s
  mesh background fit reads a frame-filling nebula as elevated background
  and subtracts through it (confirmed on a real Lagoon Nebula session —
  the passes removed over half the nebulosity). DBE's own protected pass
  already does the main background-flattening job for these targets. Small
  nebulae on empty sky (low emission/reflection blend weight) keep the
  step. Override either way with an explicit `--skip-step`.
- **`--fix-atmospheric-dispersion` now auto-derives `--plate-scale` and
  `--zenith-angle`** (from the header WCS, and from the `info.json` GPS +
  observation time) when they are not given. `--parallactic-angle` stays
  required — a wrong value shifts colour channels the wrong way and
  mapping it onto the detector needs the image north angle.
- **Desktop app layout.** The GHS-slider **STRETCH** panel is removed
  (`ui_events.restretch` / the preview's `replace_pixels` stay as
  API). RECENT FRAMES moves next to the preview on the right, so the whole
  left column below the pipeline bar is the log. Form field labels are now
  human ("Light frames", "Stack method", …) instead of the raw `--flag`
  text, with the flag name in the hover tooltip. Previews and the
  thumbnail ring clear at the start of each new run.
- **`psutil` is an optional dependency now, not a core one.** Every use is
  already `try/except`-guarded with a fixed fallback; installing it lets
  the pipeline size workers and memmaps to real free RAM.

### Internal

- **`astro_native` 0.18.0 → 0.21.0.** New kernels: `originvision_score`
  (full `--originvision` inference via `tract`), `fit_psf_moffat2d_native`
  / `fit_psf_gauss2d_native` (2D star-profile PSF fits — removes scipy
  `curve_fit`'s per-iteration Python callback, ~4× on a 30-star PSF
  estimate), `aperture_photometry_batch` (partial-pixel aperture
  photometry for N centres at once, ~150× at time-series scale). Release
  profile `panic = "abort"` → `"unwind"` so a malformed
  `--originvision-model` raises a Python exception instead of aborting the
  process; `lto` `true` → `"thin"` (tract's ~120-crate dep tree made fat
  LTO builds punishing).
- **`vendor/astrollm/` → `vendor/originvision/`** — an upstream provenance
  snapshot only, not on the runtime path.
- **Refactor: shared photometry primitives.** New `src/photometry_core.py`
  (aperture-photometry kernel + dispatcher, WCS→pixel projection, field
  centre/radius) and `src/observing_geometry.py` (alt/az, airmass, zenith
  & parallactic angle from lat/long + RA/Dec + UTC). `color_calibrate` and
  `photometric_calibration`'s per-star aperture loops are now thin
  wrappers over the shared native kernel.
- **`postprocess_stack` hardening.** A failure in the star-removal /
  starless-sidecar block can no longer abort the output stage and leave an
  orphan `_starless.fits` with no main output — it is now a warning.
- **Dependency trim.** `onnxruntime` is no longer installed by
  `build_windows.ps1` or collected into the packaged app.

## [1.0.0] - 2026-08-21

### Changed

- **Desktop app rewritten as a native window.** `python desktop_app.py` (and
  the packaged `OriginStack.exe`) is now a genuine `tkinter` window (stdlib)
  instead of a `pywebview`-wrapped local HTTP dashboard. No more Microsoft
  Edge WebView2 Runtime dependency — one fewer thing that can be missing on
  an end user's machine. The Setup form, live progress/log, and preview
  panel (zoom/pan, live re-stretch, before/after wipe-slider compare,
  per-frame thumbnail ring) all carry over with the same functionality.
- The standalone `--web-view` CLI flag (browser-tab dashboard) is removed;
  the desktop app is now the only GUI surface. The CLI itself is unchanged.
- **`--originvision` no longer scores every accepted light frame by default.**
  It now samples 3 frames (fast, ~8s each) to feed the target-classification
  prior and a defect-nudge, same as before but bounded in time regardless of
  session size. Full per-frame scoring is still available, opt-in, via the
  new `--originvision-score-all` flag. `--originvision-score-all` alone (without
  `--originvision`) is a no-op and now warns at startup.

### Internal

- `src/webview.py`'s HTTP/SSE transport is gone. Its state/data logic (log
  buffer, phase/progress, named preview slots, per-frame thumbnail ring,
  on-demand re-stretch) moved to `src/ui_events.py`, polled in-process by
  the tkinter window instead of pushed over a socket.
- `src/webview_control.py` renamed to `src/desktop_control.py` (schema
  introspection + `RunManager` are unchanged; only the name, since
  "webview" no longer describes anything in this codebase).
- `packaging/originstack.spec` bundles `tkinter` instead of excluding it;
  drops the `pywebview`/`pythonnet`/`clr_loader` PyInstaller hooks.
  `packaging/verify_build.ps1` proves the packaged build works via a window
  appearing + a startup log line (native-kernel check) and a new
  `--verify-headless` desktop-app flag (multiprocessing regression check),
  replacing the old HTTP-endpoint polling.
- **Six new native (Rust) kernels** (`ext/astro_native` 0.17.0 → 0.18.0),
  closing out the remaining Python-loop hot paths identified by a full
  profiling pass: `fit_moffat_native` (saturated-star repair's Moffat wing
  fit — removes scipy `curve_fit`'s per-iteration Python callback, 39x on a
  single fit), `mesh_median_grid` (background-extraction mesh median, on
  the default pipeline, 34x), `local_normalize_grid`
  (`--local-normalize`'s per-frame coarse background grid, 8x),
  `stamp_star_disks` (star-removal mask, on by default, 50x on a
  4000-star field), `bresenham_line_native` (`--trail-reject`'s line
  rasterization, 163x), and `radial_bin_median` (`--comet-mode`'s radial
  profile, 39x). Every kernel keeps its original numpy/scipy path as an
  automatic fallback when the native extension isn't built; see
  `CLAUDE.md`'s "Native (Rust) acceleration" section for kernel-by-kernel
  detail and `tests/test_native.py` for parity tests.

## Earlier development

Versions prior to 1.0.0 (`v0.1.0` – `v0.9.0`) predate this changelog; see
the [GitHub Releases](https://github.com/hd152/originstack/releases) page
and `git log` for that history.

