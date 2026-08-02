# Third-Party Notices

OriginStack itself is licensed under the MIT License (see [LICENSE](LICENSE)). This
file lists the third-party software it depends on, so users and redistributors know
what else they're pulling in and under what terms. It's generated from the packages
declared in [requirements.txt](requirements.txt), [requirements-gpu.txt](requirements-gpu.txt),
and [ext/astro_native/Cargo.toml](ext/astro_native/Cargo.toml) as of 2026-08 — consult
each project's own distribution for the authoritative license text; this is a summary,
not a substitute for it.

## Python — required runtime dependencies

| Package | License |
|---|---|
| [NumPy](https://numpy.org/) | BSD-3-Clause |
| [Astropy](https://www.astropy.org/) | BSD-3-Clause |
| [SciPy](https://scipy.org/) | BSD-3-Clause |
| [tqdm](https://tqdm.github.io/) | MPL-2.0 AND MIT (dual; MIT portions are the tqdm-authored code) |
| [Pillow](https://python-pillow.github.io/) | MIT-CMU (the historical "PIL Software License") |
| [psutil](https://github.com/giampaolo/psutil) | BSD-3-Clause |

## Python — optional runtime dependencies

Each is only imported if installed; every feature that uses one degrades gracefully
without it (see `CLAUDE.md`'s "Optional dependencies" section).

| Package | Used for | License |
|---|---|---|
| [rawpy](https://github.com/letmaik/rawpy) | Camera RAW input (`src/io_raw.py`) | MIT (wraps [LibRaw](https://www.libraw.org/), dual LGPL-2.1 / CDDL-1.0 — rawpy's own docs note its GPL2/GPL3 demosaic packs are deliberately excluded as GPL-incompatible with MIT) |
| [tifffile](https://www.cgohlke.com/) | TIFF input/output (`src/io_tiff.py`, `--export tiff`) | BSD-3-Clause |
| [CuPy](https://cupy.dev/) | GPU acceleration (`--use-gpu`) | MIT |
| [Cython](https://cython.org/) | Build dependency for some CuPy wheels (`requirements-gpu.txt`) | Apache-2.0 |
| **[bm3d](https://webpages.tuni.fi/foi/GCF-BM3D/)** | `--denoiser bm3d` collaborative-filter denoising | **Free for non-commercial use only** (Tampere University) — this is *not* a permissive open-source license like the others on this page. It is optional and not installed by default; if you enable `--denoiser bm3d` in a commercial context, you are responsible for obtaining your own license from the rights holder. Every other denoiser in this pipeline (wavelet, MMT, ACDNR, NLM, bilateral, aniso) has no such restriction. |

## Rust — native extension build dependencies

[ext/astro_native/](ext/astro_native/) (the optional `astro_native` PyO3 module — see
CLAUDE.md's "Native (Rust) acceleration") is built from these crates. All are
permissive (MIT and/or Apache-2.0, plus two narrower cases noted below); none are
copyleft. Full transitive closure, via `cargo metadata`:

| Crate | License |
|---|---|
| pyo3, pyo3-ffi, pyo3-macros, pyo3-macros-backend, pyo3-build-config | MIT OR Apache-2.0 |
| numpy (Rust crate, PyO3's ndarray/numpy bridge) | BSD-2-Clause |
| ndarray | MIT OR Apache-2.0 |
| rayon, rayon-core | MIT OR Apache-2.0 |
| crossbeam-deque, crossbeam-epoch, crossbeam-utils | MIT OR Apache-2.0 |
| autocfg, cfg-if, either, heck, indoc, libc, once_cell, portable-atomic, portable-atomic-util, proc-macro2, quote, rustc-hash, rustversion, syn, unindent | MIT OR Apache-2.0 |
| matrixmultiply, rawpointer | MIT/Apache-2.0 |
| memoffset | MIT |
| num-complex, num-integer, num-traits | MIT OR Apache-2.0 |
| target-lexicon | Apache-2.0 WITH LLVM-exception |
| unicode-ident | (MIT OR Apache-2.0) AND Unicode-3.0 |

Neither pyo3 nor any of its dependencies are linked into or distributed with the
Python package unless you build `astro_native` yourself (`maturin develop --release`
or `maturin build --release`) — see CLAUDE.md for build instructions. The numpy
fallback path has no Rust dependency at all.

## Validation-only (not distributed, not a runtime dependency)

| Package | Used for | License |
|---|---|---|
| [colour-demosaicing](https://github.com/colour-science/colour-demosaicing) | Reference implementation cross-checked in `tests/test_debayer_malvar.py` and `tests/test_debayer_menon2007.py` (bit-exact parity tests, skipped if not installed) | BSD-3-Clause |

This package is never imported by any code path a user actually runs — it exists
only so the test suite can validate this codebase's own from-scratch Malvar-He-Cutler
and Menon (2007) implementations against an independent reference. It is not required
to install or run OriginStack.
