# Roadmap

Working list of what could make OriginStack better, drawn from the open items in the other
dev-notes and from the 2.20 work on other cameras. Not a promise. Each item says what it needs
(**data** = a stack or session we do not have; **train** = needs torch and hours of GPU/CPU;
**code** = can start now) and a rough size (S: a day or less, M: a few days, L: a week or more).
"Tried and reverted" in CLAUDE.md lists what not to retry blind.

Last reviewed 2026-11-01.

## 1. Finish what 2.20 started: other cameras

The learned deconvolver now rescales to the trained star width and loads mono / `.fits.fz` /
cube FITS. It is verified on Origin, Seestar S50 and LCO mono only.

| Item | Needs | Size |
|---|---|---|
| Test on a DSLR/mirrorless stack (1-2"/px, different PSF), a ZWO/QHY mono set, narrowband Ha/OIII/SII | data | S each |
| Star removal on other cameras (same rescale idea; in progress) | code | M |
| Ellipticity guard (0.35) against a real trailed-star session | data | S |
| Stars within ~30 px of the frame edge miss the star-aware restore (detector border): reflect-pad before detection | code | S |
| Faint-source flux vs Pan-STARRS truth on S50 data (only Origin has it) | code | S |
| Narrowband workflow end to end: per-filter stacks -> `combine` SHO/HOO with the learned deconvolver on each mono channel and the same blend | data | M |
| Camera profiles for S50 / S30 / common DSLRs (gain per ISO, colour slopes) like Origin178 | data | M |
| Wider-PSF fine-tuning of the network (decided not warranted; revisit only if a camera shows the rescale failing) | train | L |

## 2. Star removal and star handling

- Brightest unsaturated stars keep a small core (real_left bright up to 0.20 on Orion): inference
  residual-core step in progress; real fix is a training change. **train**, L.
- Saturated stars are left in place by design; their diffraction spikes and reflection rings
  remain. A spike/ring model (symmetric halo is already subtracted) would let `--remove-stars`
  finish the job. **code**, L.
- Stars below detection are not injected in training, so they stay. **train**, M.
- Star recolouring / per-star colour correction on the stretched image (Gaia colour is known).
  **code**, M.

## 3. Calibration and robustness

- Warn (done: #238) and then optionally *rebin or crop* a mismatched master instead of dropping
  it. **code**, S.
- Auto-detect and handle a mixed-gain / mixed-binning folder: split into groups with a message,
  instead of mixing. **code**, M.
- Flat-less sessions: gradient + vignetting from a per-instrument map exists; extend the
  vignette-map library beyond Origin. **data**, M.
- Frame-level health report in the desktop app (why each frame was rejected, with a thumbnail).
  The data exists in the Phase 1 gates. **code**, M.
- Meridian-flip and field-rotation aware session splitting on non-Origin mounts (rotation
  prediction is Origin-header based). **code**, M.

## 4. Quality and speed

- Interleave two output pixels in the Lanczos-3 warp row (the AVX2 lane attempt gained nothing;
  this one is untried). **code**, M.
- Phase 3 combine for very large sessions on 16 GB machines: measured, but only on the 8-core
  laptop. A low-memory profile test on a smaller box. **code**, S.
- Re-bench the whole pipeline against current Siril and PixInsight WBPP on the same data once per
  release; keep `bench_vs_siril.py` results in the compare page. **code**, S per release.
- GPU stays opt-in; revisit only if VRAM-bound workers go away. Low priority.

## 5. Desktop app and ease of use

- Nn blend / "learned sharpening strength" as a field (config-only today; needs 8 language
  entries). **code**, S.
- Before/after slider per Phase 4 step inside the app (the website has it). **code**, M.
- A first-run "pick your camera" step that sets sensible defaults for non-Origin data and says
  which learned features apply. **code**, S.
- Drag-and-drop a folder; remember last folders; one-click "stack tonight + merge with previous".
  **code**, M.
- Crash reports: a "copy diagnostics" button (versions, native status, GPU, the config) to make
  bug reports reproducible. **code**, S.
- Linux and macOS desktop builds exist; verify them on real machines each release. **data**, S.

## 6. Science outputs

- Photometry is calibrated for Origin; extend gain/PTC to other cameras via the profile work
  above. **data**, M.
- AAVSO-style export of the time series (the folder `C:\nn_work\aavso` suggests a start).
  **code**, M.
- Transient detection against a catalogue (Gaia / Minor Planet Center) to cut false positives
  before the CNN triage. **code**, M.
- Astrometric report for asteroid / comet sessions (MPC 80-column). **code**, M.

## 7. Project and community

- Release 2.20 (everything since 2.19.0): changelog entry, tag, wheels.
- Pin the sample session and the website's "other cameras" page as the first thing a new user
  sees; track downloads (`tools/download_stats.py`) per release. 14-day traffic is ~230 unique
  visitors and ~14 downloads per release, so the funnel, not features, is the bottleneck.
- Ask on Cloudy Nights / Seestar groups for volunteer test stacks (DSLR, ZWO, trailed) -- this
  unblocks most of section 1 and is the highest-value, lowest-effort item.
- Reviewed translations: the 8 languages are machine-drafted. Ask native speakers to review the
  home page and desktop app strings. **data**, S each.
- Code-signing for the Windows installer (policy page exists) to remove the SmartScreen warning.

## Suggested order

1. Release 2.20, then collect test stacks (section 7 -> unblocks section 1).
2. Finish star removal for other cameras + bright cores (in progress).
3. Section 5 quick wins: camera step, diagnostics button, blend field.
4. Section 3 mixed-gain/binning handling.
5. Photometry on other cameras + AAVSO export.
