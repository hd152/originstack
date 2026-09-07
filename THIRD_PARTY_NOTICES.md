# Third-Party Notices

OriginStack itself is licensed under the MIT License (see [LICENSE](LICENSE)). This
file lists the third-party software it depends on, so users and redistributors know
what else they're pulling in and under what terms. It's generated from the packages
declared in [requirements.txt](requirements.txt), [requirements-gpu.txt](requirements-gpu.txt),
and [ext/astro_native/Cargo.toml](ext/astro_native/Cargo.toml) as of 2026-09 — consult
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
| [onnxruntime](https://onnxruntime.ai/) | `--originvision` inference **fallback** for a source checkout without `astro_native` built (the packaged app uses the native Rust `tract` kernel instead — see the Rust section) | MIT |
| [CuPy](https://cupy.dev/) | GPU acceleration (`--use-gpu`) | MIT |
| [Cython](https://cython.org/) | Build dependency for some CuPy wheels (`requirements-gpu.txt`) | Apache-2.0 |
| **[bm3d](https://webpages.tuni.fi/foi/GCF-BM3D/)** | `--denoiser bm3d` collaborative-filter denoising | **Free for non-commercial use only** (Tampere University) — this is *not* a permissive open-source license like the others on this page. It is optional and not installed by default; if you enable `--denoiser bm3d` in a commercial context, you are responsible for obtaining your own license from the rights holder. Every other denoiser in this pipeline (wavelet, MMT, ACDNR, NLM, bilateral, aniso) has no such restriction. |

## Rust — native extension build dependencies

[ext/astro_native/](ext/astro_native/) (the optional `astro_native` PyO3 module — see
CLAUDE.md's "Native (Rust) acceleration") is built from a transitive closure of
**~147 crates** (`cargo metadata`, 2026-09). Every one is permissive — no copyleft,
nothing non-commercial. License breakdown:

| License class | Count (approx.) |
|---|---|
| `MIT OR Apache-2.0` (and spelling variants like `MIT/Apache-2.0`) | ~115 |
| `MIT` only | ~11 |
| `Unlicense OR MIT` / `Unlicense/MIT` | ~6 |
| `Apache-2.0` only, or `Apache-2.0 WITH LLVM-exception` | ~4 |
| `BSD-2-Clause` (incl. the Rust `numpy`/ndarray-numpy bridge) | 2 |
| triple-licensed with `Zlib` / `0BSD` / `BlueOak-1.0.0` as one option (always also MIT or Apache-2.0) | ~6 |
| `(MIT OR Apache-2.0) AND Unicode-3.0` (`unicode-ident`) | 1 |
| `Zlib` (`adler2` / miniz) | 1 |

Notable families in that closure:

- **pyo3 / pyo3-ffi / pyo3-macros / numpy / ndarray / rayon / crossbeam** — the
  Python-binding and array/parallelism layer. MIT OR Apache-2.0 (numpy: BSD-2-Clause).
- **tract-onnx and its `tract-*` siblings** (`tract-core`, `tract-hir`,
  `tract-linalg`, `tract-data`, `tract-nnef`, `tract-onnx-opl`) — the pure-Rust ONNX
  inference runtime used by the `--originvision` native kernel (`mod originvision` in
  `lib.rs`). MIT OR Apache-2.0 (Sonos). Pulls the bulk of the ~147: `prost`/protobuf
  codegen, `smallvec`, `num-*`, `anyhow`, `nom`, `flate2`/`miniz_oxide`, `getrandom`,
  `half`, `scan_fmt`, etc. — all in the permissive set above.

Neither pyo3, tract, nor any of their dependencies are linked into or distributed with
the Python package unless you build `astro_native` yourself (`maturin develop --release`
or `maturin build --release`) — see CLAUDE.md for build instructions. The numpy
fallback path has no Rust dependency at all, and `--originvision` then falls back to
the Python `onnxruntime` package (MIT, listed under optional dependencies above).

## Validation-only (not distributed, not a runtime dependency)

| Package | Used for | License |
|---|---|---|
| [colour-demosaicing](https://github.com/colour-science/colour-demosaicing) | Reference implementation cross-checked in `tests/test_debayer_malvar.py` and `tests/test_debayer_menon2007.py` (bit-exact parity tests, skipped if not installed) | BSD-3-Clause |

This package is never imported by any code path a user actually runs — it exists
only so the test suite can validate this codebase's own from-scratch Malvar-He-Cutler
and Menon (2007) implementations against an independent reference. It is not required
to install or run OriginStack.
