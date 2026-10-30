# astro_native

Native (Rust) hot-path kernels for [OriginStack](https://github.com/hd152/originstack), a stacker
for Celestron Origin images: sigma-clip and other stack combines, Lanczos warp and drizzle,
debayering (RCD, Malvar, Menon), hot-pixel and spike rejection, filters, background fitting,
star detection and a small convolution kernel for the bundled denoising network.

This is a support library, not a general astronomy toolkit. Every kernel mirrors a numpy/scipy
reference that lives in OriginStack (which falls back to it when this module is absent), the
function signatures follow OriginStack's needs, and they may change between releases. Per-kernel
design and timings: [dev-notes/native-kernels.md](https://github.com/hd152/originstack/blob/main/dev-notes/native-kernels.md).

- Wheels are built for baseline x86-64 (Windows, Linux, macOS) and Apple silicon; SIMD (AVX2,
  SSSE3) is chosen at run time, so one wheel runs on any CPU of its architecture.
- Python 3.10+ (abi3), numpy 2.4.6+.
- Licensed GPL-3.0-or-later.
