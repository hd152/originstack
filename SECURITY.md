# Security Policy

## Scope

OriginStack is a local image-processing pipeline. It reads FITS files from disk and writes output files. It does not run as a server, does not accept network connections, and does not expose a public API surface. The main security-relevant concerns are:

- **External subprocess execution**: `--star-remove` (Starnet++), `--bg-method graxpert` (GraXpert), and `--plate-solver astap` (ASTAP) invoke external binaries. Only binaries discovered on your `PATH` or specified via explicit path flags are used.
- **Network access**: `--plate-solve` (astrometry.net mode) sends your stacked FITS image to [nova.astrometry.net](https://nova.astrometry.net) for solving. Do not use this flag if your data is sensitive.
- **API keys**: `ASTROMETRY_API_KEY` is read from environment variables. Do not commit this to version control or include it in config TOML files that are checked in.
- **FITS file parsing**: FITS files from untrusted sources are parsed by `astropy`. Malformed FITS files could trigger issues in the parser. Only process files from trusted telescopes and cameras.
- **JSON sidecar parsing**: per-frame `*.json` sidecars and session `info.json` files co-located with the data are read with the standard-library `json` parser to backfill missing metadata (ISO, gain, temperature, WCS). No code is executed from them; only known keys are used. Still, only process metadata from trusted capture apps.
- **Native (Rust) extension**: the optional `astro_native` module (`ext/astro_native/`) is compiled from the source in this repository via PyO3/maturin. It performs numeric work on in-memory float32 arrays only — no file, network, or subprocess access — and every kernel has a pure-Python fallback. Build it yourself from source; do not install prebuilt `astro_native` wheels from untrusted third parties.

## Reporting a Vulnerability

If you discover a security issue in OriginStack, please report it by opening a [GitHub Issue](../../issues) with the label `security`. For sensitive disclosures, email the repository owner directly before filing a public issue.

Please include:
- A clear description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fixes if you have them

We aim to respond to security reports within 5 business days.
