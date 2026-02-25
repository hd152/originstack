# Release Notes

All notable changes to this project will be documented in this file.

## v0.1.0 - Initial Release (2026-02-24)

Highlights:

- Implemented streaming FITS stacker with three-phase processing (validation, registration, stacking).
- Automatic frame detection and per-folder calibration (bias/dark/flat) with master frame creation.
- Bayer demosaicing (bilinear) and Malvar option (`--debayer-method malvar`).
- Quality analysis (brightness, contrast, star counts) and percentile-based filtering.
- Registration with sub-pixel shifts, automatic cropping to valid region.
- Optional drizzle combining (`--drizzle-scale`), white-balance methods, hot-pixel removal, background subtraction.
- Hierarchical folder processing and preview JPEG generation.
- CLI with `-d/--directory`, `-o/--output`, `--no-registration`, `--quality-filter`, and other options.
- Unit tests, synthetic smoke-run generator, and GitHub Actions CI workflow included.

Files of interest:
- `astro_stack.py` — main CLI script and implementation.
- `PROJECT_SPEC.md` — full design/specification.
- `tests/` — unit tests (pytest).
- `.github/workflows/ci.yml` — CI pipeline for tests and synthetic smoke-run.

Upgrade notes:

- This is the first public release; no prior versions exist. Expect iterative improvements to registration and debayer algorithms.

How to install:

```bash
pip install -r requirements.txt
```

How to run basic example:

```bash
python astro_stack.py -d lights/ -o stacked.fits --quality-filter
```

Contributors:
- Initial implementation by repository owner and automated assistant tooling.

## Unreleased

- CI trigger: minor documentation update to re-run workflows.
