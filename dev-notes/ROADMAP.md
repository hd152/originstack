# Roadmap

Working list of what could make OriginStack better. Not a promise. Each item says what it needs
(**data** = a stack or session we do not have; **train** = needs torch and hours of GPU/CPU;
**code** = can start now) and a rough size (S: a day or less, M: a few days, L: a week or more).
"Tried and reverted" in CLAUDE.md lists what not to retry blind.

Last reviewed 2026-11-09. Done items are listed at the bottom so they are not proposed again.

## 1. Other cameras

The learned deconvolver rescales to the trained star width and loads mono / `.fits.fz` / cube FITS.
It is verified on Origin, Seestar S50 and LCO mono only.

| Item | Needs | Size |
|---|---|---|
| Test on a DSLR/mirrorless stack (1-2"/px, different PSF), a ZWO/QHY mono set, narrowband Ha/OIII/SII | data | S each |
| Star removal on other cameras (same rescale idea; in progress) | code | M |
| Ellipticity guard (0.35) against a real trailed-star session | data | S |
| Faint-source flux vs Pan-STARRS truth on S50 data (only Origin has it) | code | S |
| Narrowband workflow end to end: per-filter stacks -> `combine` SHO/HOO with the learned deconvolver on each mono channel and the same blend | data | M |
| Camera profiles for S50 / S30 / common DSLRs (gain per ISO, colour slopes) like Origin178 (`src/data/camera_profiles/`) | data | M |
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

- Flat-less sessions: gradient + vignetting from a per-instrument map exists; extend the
  vignette-map library beyond Origin. **data**, M.
- Meridian-flip and field-rotation aware session splitting on non-Origin mounts (rotation
  prediction is Origin-header based). **code**, M.

## 4. Quality and speed

- Phase 3 combine for very large sessions: measured on the 8-core laptop and, since 2026-11-09, on a
  16 GB laptop (300 lights: frame store compressed in Phase 1, aligned stack on disk, no out-of-memory;
  Phase 1 dropped to 2 workers when only 0.8 GB was free). Open: what dominates the 380-430 s there.
  **code**, S.
- `--gpu-coadd`: 2x on a GTX 1650 Ti with the stack in RAM, but 2.5-3x slower than the CPU on two laptop
  GPUs with a disk-backed stack. Reading each frame once (PR "GPU coadd: read a disk-backed frame
  once") took the RTX 3050's combine from 80.9 to 53.1 s, still above the CPU's 32.2 s, and did nothing for
  the Radeon iGPU; the stage profile (crate 0.67) puts the 3050's combine at 43.6 s with every stage 3-5x slower than the same stage
  on the GTX 1650 Ti: cause unknown: both laptops were plugged in, so look at GPU clocks/power limit, thermals and the hybrid-graphics path (see [gpu-coadd.md](gpu-coadd.md)). Do not promote
  it until a discrete card beats the CPU on a real session. **code**, M.
- Re-bench the whole pipeline against current Siril and PixInsight WBPP on the same data once per
  release; keep `bench_vs_siril.py` results in the compare page. **code**, S per release.
- GPU stays opt-in; revisit only if VRAM-bound workers go away. Low priority.

## 5. Desktop app and ease of use

- A first-run "pick your camera" step: today the folder scan only shows a note for an unprofiled camera.
  **code**, S.
- Linux and macOS desktop builds exist; verify them on real machines each release. **data**, S.

## 6. Science outputs

- Photometry is calibrated for Origin; extend gain/PTC to other cameras via the profile work above.
  **data**, M.
- Transient detection against a catalogue (Gaia / Minor Planet Center) to cut false positives before the
  CNN triage. **code**, M.
- Astrometric report for asteroid / comet sessions (MPC 80-column). **code**, M.

## 7. Project and community

- Pin the sample session and the website's "other cameras" page as the first thing a new user sees; track
  downloads (`tools/download_stats.py`) per release. 14-day traffic was ~230 unique visitors and ~14
  downloads per release, so the funnel, not features, is the bottleneck.
- Ask on Cloudy Nights / Seestar groups for volunteer test stacks (DSLR, ZWO, trailed) -- this unblocks most
  of section 1 and is the highest-value, lowest-effort item.
- Reviewed translations: the 8 languages are machine-drafted. Ask native speakers to review the home page
  and desktop app strings. **data**, S each.
- Code-signing for the Windows installer (policy page exists) to remove the SmartScreen warning; needs a
  certificate.
- Run `.github/workflows/samples.yml` once by hand after it lands (the Zenodo zip layout is a guess), and
  `--update` the baseline if a Linux runner needs it.

## Done (do not propose again)

- Release 2.20 and later (VERSION is 2.23.0); the changelog entry is kept per release.
- Desktop app: copy diagnostics (Help > Report a problem), drag-and-drop a folder, recent folders, a camera
  note for unprofiled cameras, per-frame rejection reasons in the frame graph (`--frame-report`),
  before/after of one Phase 4 step, the learned-sharpening strength field (`--nn-deconv-blend`), a one-click
  "add earlier stacks of this target" checkbox (`previous_stacks.py`).
- Mixed binning / frame shape / gain / camera in one folder: split into groups
  (`frame_discovery.differing_settings`); a mismatched master is cropped or dropped with a message
  (`cli._plan_master_crop`, `_drop_mismatched_masters`).
- Stars within ~30 px of the frame edge: the learned deconvolver reflect-pads before detection.
- AAVSO export of the photometry time series (`aavso_report.py`).
- Performance budget in CI (`tools/perf_budget.py`); weekly sample-session regression workflow.
- Catalogue prefetch during Phases 1-3 (`prefetch.py`).
- Tried and dropped: Lanczos warp two-pixel interleave, CPU/GPU frame split of the proper coadd.
