# GPU proper coadd (`--gpu-coadd`)

The combine pass of the proper coadd ([src/proper_coadd.py](../src/proper_coadd.py)) on the GPU, through
wgpu (Vulkan, DX12 or Metal: any vendor, no CUDA). Opt-in, and **not in the packaged app or the PyPI
wheels**: it lives behind the `gpu` Cargo feature of `astro_native`.

```bash
cd ext/astro_native
maturin build --release --features gpu            # adds astro_native.gpu_* (+2.2 MB wheel, +5.6 MB .pyd)
pip install --force-reinstall target/wheels/astro_native-*.whl   # then confirm astro_native.__version__
python originstack.py -d lights/ -o stacked.fits --gpu-coadd
python tools/ab_gpu_coadd.py --synthetic 96       # CPU vs GPU on one stack: time and difference
```

`--gpu-coadd` falls back to the CPU pass, with a note, when the build has no `gpu` feature, no adapter
exists, the frames do not fit (3 channels only, planes of at most 4096 px = frames up to 4064 px), the
device runs out of memory, or anything in the driver fails. `RunCancelled` propagates and the device
session is released. `ORIGINSTACK_GPU_DISABLE=1` forces "no adapter" (tests; read once per process),
`ORIGINSTACK_GPU_ADAPTER=<substring>` picks an adapter by name, `ORIGINSTACK_GPU_ALLOW_SOFTWARE=1` admits a software rasteriser (lavapipe; CI's `gpu-test` job), `ORIGINSTACK_GPU_MAX_MB` (default 3072)
caps the device memory a run may ask for. The option is hidden from the desktop form
(`desktop_control._UNSUPPORTED_DESTS`): the packaged app cannot have it.

## Design

[ext/astro_native/src/gpu.rs](../ext/astro_native/src/gpu.rs) is the device context (one wgpu device per
process, adapter choice, error plumbing, the pipeline/buffer/dispatch/read-back helpers) and
[gpu_coadd.rs](../ext/astro_native/src/gpu_coadd.rs) the kernels and the session
(`gpu_coadd_begin` / `_add` / `_finish` / `_abort`; `gpu_fft2` is a test hook).

Per frame, on the device: reject outliers against the reference stack on the frame's flux scale and
subtract its sky into zero-padded planes (the same f32 arithmetic as `proper_coadd_prep`), render the
PSF plane from a 49 x 49 stamp (corners at the FFT origin), 2-D FFT, split, and accumulate
`num += wn F conj(P)`, `den += wf |P|^2` over the half plane. At the end `R = num / sqrt(den)`,
completed by Hermitian symmetry, is inverse-transformed and cropped.

- **FFT:** power-of-two rows of at most 4096 in workgroup memory, two radix-2 stages per barrier
  (radix-4 style), a tiled transpose, rows again; the spectrum stays transposed (only element-wise work
  follows). The PSF's row pass transforms only its 2r+1 nonzero rows.
- **Packing:** channels 0 and 1 share one complex plane (`ch0 + i ch1`) and are separated by symmetry;
  channel 2 and the PSF each have a plane. The final inverse packs two channels the same way.
- **Host overlap:** per-frame sky and noise (CPU, thread pool) run ahead of the device, chunks of 8
  frames; `gpu_coadd_add` releases the GIL. The session mutex is taken *inside* the detached closure
  (taking it first and then waiting for the GIL deadlocked in a related prototype).
- **Padding:** the caller pads each axis to the next power of two >= size + 32 (`_gpu_plane`); a
  1990 x 2945 frame uses 2048 x 4096 planes (CPU path: about 2058 x 3024), which costs about 36% extra
  FFT work on the width.
- **Memory:** `session_bytes` = 7 complex planes + half-plane numerator and denominator + the reference +
  two frame upload slots (about 0.83 GB for 1990 x 2945 frames on 2048 x 4096 planes, 1.8 GB at the
  4064 px maximum). Checked against `ORIGINSTACK_GPU_MAX_MB` and each buffer against the adapter's
  limits before anything is allocated; wgpu errors (out of memory, device lost) are recorded by an
  uncaptured-error handler and raised as `RuntimeError` at the next call, instead of wgpu's default panic.

## Numerics (the output is not bit-identical to the CPU path)

- f32 throughout (WGSL has no f64). Twiddles are built in f64 on the host: GPU `sin`/`cos` are only
  accurate to about 2^-11 absolute, and with a sine in the shader tiny fractional coordinates produced
  zero or garbage weights in a related warp prototype.
- **The PSF must not share an FFT plane with data.** FFT error scales with the plane's largest
  component. Packing `ch2 + i PSF` left the (sum 1) PSF's spectrum with phase noise at high frequency,
  and `R = F conj(P) / |P|` amplifies it: 25-64% noise-like error against a float64 reference, with
  perfect agreement for a delta-function PSF (which is why it passed the first check).
  `tests/test_gpu_coadd.py::test_coadd_matches_float64_reference` runs both.
- Accuracy against a float64 reference matches the CPU path's own float32 arithmetic (relative rms
  about 6e-6 on 24 random frames).

## Measurements (GTX 1650 Ti, 4 GB, Vulkan; i9-10885H; timings move up to ~40% between runs)

Real session, Whirlpool, 206 stacked frames of 1990 x 2945 x 3 (prototype, A/B inside one run):

| | CPU | GPU |
|---|---|---|
| whole `proper_coadd` call (incl. PSF measuring) | 20.2 s | 10.9 s |
| combine pass | 16.9 s | 9.0 s |

Difference to the CPU result: rms 0.00023 ADU against an output noise of 45 ADU, max 0.023 ADU,
p99.99 0.002 ADU, mean 1.5e-6 ADU; brightest 4355 pixels: max relative 3.4e-7; no NaNs.

Synthetic 96 frames: CPU combine 7.7 s, GPU 3.5 s (difference rms 4e-5 ADU against 3-5 ADU of noise).
Per frame the device takes about 34 ms (staged: row FFTs 6.0, column FFTs 8.1, transposes 3.0, prep 2.4,
accumulate 2.7, the rest upload) against about 80-100 ms for the CPU path; the FFT passes run at about
36 GB/s effective, i.e. they are compute-bound, not bandwidth-bound.

The productised code, full 248-frame Whirlpool session (206 stacked), three runs back to back
(`--gpu-coadd` against the default, and `--gpu-coadd` with `ORIGINSTACK_GPU_DISABLE=1`):

| | CPU | `--gpu-coadd` | `--gpu-coadd`, GPU forced off |
|---|---|---|---|
| combine pass | 17.8 s | 8.6 s | 18.1 s (fell back, with a note) |
| "Proper coadd" step | 20.7 s | 11.7 s | 21.0 s |
| Stacking phase | 56.6 s | 48.7 s | 62.4 s |
| whole run | 2 m 06 s | 2 m 02 s | 2 m 16 s |

The whole-run difference is small and noisy (these runs were serial on a machine with nothing else
running, and the CPU run's other phases varied by several seconds); the step itself is what moved.
Final linear stacks (3 x 2025 x 3001): the forced-off run is **bit-identical** to the CPU run; the GPU
run differs by rms 0.0011 ADU (2.3e-5 of the 45 ADU green-channel noise), max 0.39 ADU (one pixel),
p99.99 0.006 ADU, mean -8e-4 ADU; the 4547 brightest pixels differ by at most 8e-6 relative; no NaNs;
`NFRAMES` and `INTGTIME` equal.

## Not done

- Mixed-radix sizes (a 3000 px width pads to 4096), more than 3 channels, frames above 4064 px.
- Packing two PSFs per plane: **tried, no gain, reverted.** Frames go through in pairs, the two (sum 1,
  same magnitude) PSFs share one complex plane (`psf_a + i psf_b`, split by Hermitian symmetry in the
  accumulate kernel), which removes one of the six column FFTs per pair. Accuracy was unchanged
  (difference to the CPU path 1.3e-5 of noise, bright pixels 3.5e-7, all 23 GPU tests incl. the float64
  PSF-precision test passed), so same-magnitude PSFs do pack safely. Speed: combine pass of 48 synthetic
  1990 x 2945 frames, 10 alternating runs each, median 1.9 s before and 1.9-2.0 s after (49 frames,
  odd tail: 2.0 s vs 2.1 s), within noise, for +2 planes and +2 upload slots (about +0.3 GB). The pass is
  not limited by the FFT work saved (the row pass of the PSF is already only 2r+1 rows).
- Sky and noise per frame are still CPU work (about 3.6 s over 206 frames, hidden behind the device).
- The experiment also tried a GPU Lanczos warp (about 2x, 7 s of a run), a per-pixel sigma-clip combine
  (6x *slower* than the CPU: per-thread quickselect in local memory), the NN U-Net and the Phase 1 pixel
  work; none of them is part of this feature.
