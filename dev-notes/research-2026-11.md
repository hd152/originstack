# Research round, 2026-11-10

Ten candidates from a literature pass (multi-frame restoration, reference-guided background,
self-supervised multi-exposure denoising, calibration noise, GPU inference, storage width), each
measured on the 16 GB laptop (Ryzen 8 cores / 16 threads, 13.9 GB RAM, RTX 3050 laptop GPU 4 GB)
with the sessions in `C:\source\Astrophotography`. Three changed the code (the 16-bit aligned stack
was decided and built afterwards, the same day); the rest are recorded so
they are not retried blind. Scratch scripts were not kept; each entry says how it was measured.

## Kept

### Full-field extension reads each frame once (speed, bit-identical)

Whirlpool 2026-05-10 (300 lights, 295 stacked; Phase 1 store compressed, aligned stack 21 GB on
disk). Before: 7 min 5.6 s, of which stacking 5 min 9 s:

| Stage | before | after |
|---|---|---|
| Frame alignment | 63.5 s | 64.1 s |
| Stack combine | 70.7 s | 82.2 s |
| Proper coadd | 48.7 s | 56.1 s |
| Full-field extension (+3% area) | **123.6 s** | **24.2 s** |
| Whole run | 425.6 s | 347.8 s |

(combine and coadd moved by this laptop's usual run-to-run noise; nothing in them changed.)
`full_field._combine_window` warped one window at a time, in row bands, and took every frame
from `mem_rgb` again for each: four windows, one or two bands each, plus `frame_sky_offsets`'
own pass. With frames in RAM or a memmap that is a view; with the compressed store every access
decompresses the whole 75 MB frame, so the session was decompressed six to eight times.
`_combine_windows` plans the windows' bands into passes that fit a memory budget (a quarter of
the available memory, 768 MB to 3 GB: `_band_budget`, `_plan_passes`), reads each frame once per
pass in four threads, warps it into every window of the pass and takes its sky median in the
first pass. Warp and combine are per pixel, so the stack and the coverage map are bit-identical
(`np.array_equal` on the real session's FITS, and against the old module on synthetic rotated
and shifted sessions at three budgets). The loop now reports progress and honours Cancel
(`pipeline._apply_full_field` re-raises `RunCancelled`). Tests: `tests/test_full_field.py`.

### Survey H-alpha reference for DBE (quality at the DBE stage, opt-in)

`src/halpha_reference.py`, `--config dbe_halpha_reference=true`. Results, gates and why the
finished picture barely changes: [phase4-quality.md](phase4-quality.md). In short: two-night
mismatch 0.351 -> 0.199 (NGC 2244) and 0.339 -> 0.221 (Eagle), target light kept 0.21 -> 0.42
and 0.36 -> 0.76; declined correctly on galaxies, clusters and one nebula whose map is a smooth
gradient; the delivered JPEG moves by a few levels because the sky floor and the black point
remove what DBE now keeps. Evaluated on two targets only: the weak-target set of 2026-10
(Horsehead, Lagoon, Pleiades, Flaming Star...) was not on this machine.

## Measured and dropped

**NN U-Net forward pass on the GPU (wgpu).** A compute-shader port (`gpu_nn.rs`, not kept):
activations as `(H, W, C/4)` arrays of `vec4`, weights as one `mat4x4` per (output block, tap,
input block), one invocation per pixel and four output channels, the tile's 13 dispatches in one
submission. Output matched the CPU pass to 1.0e-5 (noise units). Speed on the RTX 3050: 175 ms
per 432 px tile (641 ms at 816 px, so compute-bound) against 66 ms for `nn_forward` with FMA.
The same network through torch/cuDNN: 1.0 s for a 2048 x 3056 frame (0.66 s in half precision)
against ~3.2 s of CPU tiles; a first reading of "9x" compared it with a CPU pass that was
sharing the machine with a stacking run. So the ceiling is 3x per pass (~4 s of a run that
uses both networks) and a plain WGSL convolution is 8x short of it. gpu-coadd.md already lists
the U-Net among the GPU experiments that were not kept.

**More of the master dark's pixel pattern.** Hypothesis from the 2026-10 dithered Whirlpool
test (16 px noise 5-11% lower without the dark): the dark stamps its own noise into every frame.
It cannot: `cli._build_masters` blurs a single-file dark at sigma 20 px, and Origin's file is
already an average of 9 exposures (`dark0009.json`). The opposite question -- is real pattern
being thrown away by the blur? -- measured on Whirlpool 2026-04-06 (60 lights, per CFA plane,
high-pass at 6 px, hot pixels left out): the lights' fixed pattern is 62-96 ADU (split-half
estimate) against 825-1040 ADU of single-frame noise; the dark's pattern correlates with it at
0.15-0.23 with slope 0.32-0.34 (exposure ratio 0.67 x the dark being 9 C warmer), and fitting it
out leaves 58-93 ADU. That is 0.6-0.9% of a frame's variance before the fit and a tenth of that
gained, averaged down further by the drift. Not worth a calibration change.

**Multi-exposure denoiser (ASTERIS, Guo et al. 2026).** Its gain over a stack has to come from
structure the mean of the exposures does not hold. Sunflower 2026-03-12, the aligned frames
split into 8 interleaved sub-means (a hook on `validate_noise`, central 1000 px, sky pixels):
kurtosis of the residuals +0.006 / -0.004 / -0.001 (R/G/B), spread of the per-pixel sample
variance 1.004 / 0.994 / 1.000 of the chi-square value, sub-stack noise equal to 2.7%, an oracle
inverse-variance weighting of the sub-stacks gains 0.04% in variance. Gaussian, even, equal:
the mean is sufficient and there is nothing for a spatiotemporal network to use. No training
run was made. One session; a session with clouds or a bright moon passing could differ, but
those frames are already weighted or rejected.

**`--stream` on a low-memory machine.** Whirlpool 300: 9 min 19.5 s against 5 min 48 s, with
all 300 frames (no statistical gates) and no proper coadd.

**3 x 2^k planes for the GPU coadd.** The FFT pads each axis to a power of two. With the 32 px
pad a common crop under 2016 rows already fits 2048, and the width (under 3040 px) would fit
3072 instead of 4096: 25% fewer pixels at best, on device stages that are half of a combine the
host already bounds (14.8 of 28.6 s). Under 1% of a run for a mixed-radix shader; not started.

**Joint multi-frame restoration (ImageMM, Sukurdeep et al. 2025).** Not run. For stationary
background-limited noise the proper coadd is a sufficient statistic (Zackay & Ofek 2017), so a
multiplicative multi-frame deconvolution of the frames is a deconvolution of the proper image
with its effective PSF, which `--deconvolve rl` does (and the learned deconvolver beats); what
it could add is a PSF that varies over the field, where the tile-aware coadd measured 0-3.7%
excess width (psf-census.md) and `nn_tilt_equalise` already addresses the horizontal gradient.
This is an argument, not a measurement: its reference implementation (TensorFlow) was not run.

**Synthetic-blur training for the deconvolver.** Already done (2026-10-21, model s4: 1-3%
narrower than v3 in the delivered image) and superseded by v4 with Pan-STARRS truth:
nn-deconv-runbook.md.

## Decided afterwards: the 16-bit aligned stack (built the same day)

**A 16-bit aligned stack.** After the full-field fix the low-memory run is bounded by the
aligned stack on disk: 21 GB written once (alignment 64 s) and read twice (combine 71-82 s,
proper coadd 49-56 s). Storing it as uint16 with a per-frame scale (step far below the noise:
frame noise 566 ADU against a step of 2-4 ADU) would halve that. It is not a contained change:
about fifteen call sites take the stack as a float32 array (the native combine, proper coadd,
local normalisation, noise validation), and the rule that RAM and disk placement give the same
bits means RAM-placed stacks must be quantised too, so every stack's values change slightly
and the sample baseline moves. That is a release decision, not an optimisation.

**Built, as the default on the default combine path** (`frame_store.HalfFloatStack`,
`--aligned-stack half|full`). float16 rather than uint16: a frame is stored relative to its own
median with a power-of-two scale (largest sample in [8192, 16384)), so the step is relative --
1/4096 of a sample's distance from the sky -- and stays below the shot noise on bright stars,
where a fixed uint16 step would not. Reads decode into fresh float32 arrays (the fused native
combine reads row bands decoded in threads, `read_band`; proper coadd decodes a frame per
combine thread and only the star stamps in its measuring pass, `lazy_frame`), so RAM and disk
placement stay bit-identical and no kernel changed. Consumers that take the stack as one array
(the numpy combines, the other stack methods, local normalisation, the moving-object search,
the GPU coadd, elastic registration) keep float32: `stacking._half_stack_wanted`.

Rosette (NGC 2244 2026-03-12, 56 frames, `--no-color-calibrate` on both so Gaia's answer cannot
differ): difference to the float32 stack 1.10 / 0.58 / 0.88 ADU rms (0.69 / 0.53 / 0.64% of the
stack noise), mean +0.24 to +0.32 ADU; 1,187 pixels in 464 star cores differ by more than 20
ADU (up to 868 on a 57,000 ADU core): one frame's clip decision flipping. Aperture fluxes of the
400 brightest stars: half / full 1.00002 median, MAD 2e-5, worst 0.2%. (A first pair of runs
with colour calibration on differed by whole channel gains: Gaia's deeper query answered one run
and not the other; on the saved stacks `calibrate_linear_stack` declines both.)

Whirlpool 300 frames on the 16 GB laptop (before the coadd floor below): aligned stack 21.0 -> 10.5 GB on disk; alignment 64.1 -> 53.2 s, combine 82.2 -> 39.5 s, proper coadd 56.1 -> 47.1 s, stacking phase 3 min 49 s -> 2 min 46 s, whole run 5 min 48 s -> 5 min 0 s (one run each; this laptop moves 15-40% between identical runs, but the combine halved, as the halved read predicts).

**The proper coadd was sensitive to sub-noise input changes (found here, not caused here; fixed).**
On Whirlpool the half stack's result first differed from the float32 one by 25-30% of the stack
noise (rms, pure pixel-scale noise), against 0.7% with `--no-proper-coadd`. The half path itself
was exact (coadd on the half stack and on its float32 decode: bit-identical), and adding 0.2 ADU
rms of random noise to the float32 frames moved the float32 coadd by 27.7 ADU rms -- as much.
Cause: the difference was zero below 0.35 cycles/px and 18 / 43 / 130% of the stack's own power
in the 0.40-0.45 / 0.45-0.50 / corner bands; there a 5.8 px FWHM PSF's spectrum is below float32
FFT rounding (true 8.5e-9 at 0.45 cycles/px against 1.0e-8 from a float32 transform, 7.8e-10
against 6.9e-9 at Nyquist; a 2.4 px PSF, Rosette, stays above 9e-3, which is why it was stable).
`R = num / sqrt(den)` then weighted the frames by rounding noise. Fix: `proper_coadd._PSF_FLOOR`
(1e-5) added at every frame's PSF centre (`psf_kernel`, all three paths), so where the PSF has no
information the frames combine as a whitened inverse-variance mean. 60-frame Whirlpool: the
0.2 ADU perturbation now moves the stack by 0.087 ADU rms (was 27.7); float32 vs half 0.10-0.17%
of the noise (was 20-24%). Against the old coadd: power identical to four digits below 0.3
cycles/px, 5-15% lower in the top bands (pixel noise -0.4 to -0.8%), shared-star FWHM 5.795 vs
5.805 px (259 stars, median ratio 1.000), bright-star aperture fluxes within 0.01%, peaks +-5%
(single pixels in the band that changed). `test_proper_coadd_is_stable_to_sub_noise_input_changes`:
0.06% of the noise with the floor, 14% without.
