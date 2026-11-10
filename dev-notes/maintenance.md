# Maintenance runbook

State when the project went into maintenance (2026-11-02): version 2.21.0, CI, CodeQL and the Pages
build green on `main`, 2481 tests, release assets for Windows, macOS (Apple silicon and Intel) and
Linux. This note is what to do to keep it that way, and where the knowledge lives.

## Routine

| Look at | Where | Why |
|---|---|---|
| CI, CodeQL, Pages deploy on `main` | GitHub Actions | a red `main` blocks everything else |
| Issues | https://github.com/hd152/originstack/issues | first route for users (the sample page and README point there) |
| The contact mailbox | `originstack@originstack.site` | listed on the sample-sessions page, README and `SECURITY.md`; a security report arrives here |
| Dependency updates | `requirements*.txt`, `ext/astro_native/Cargo.toml` | Python 3.12+ is the supported version; the installers ship 3.12 |
| The Zenodo records | the twelve DOIs on `docs/sample-sessions.html` | they must keep resolving; metadata (not files) can still be edited after publishing |

## Cutting a release

The 2.21.0 release followed these steps; `6395855` (2.20.0) has the same shape.

1. On a branch from `origin/main`: set `VERSION`; turn `## [Unreleased]` of `CHANGELOG.md` into
   `## [X.Y.Z] - date` (leave a fresh empty `[Unreleased]`); add a block to `docs/whats-new.html`
   (plain language, one `<section id="vX-Y">`); run `python tools/build_cli_reference.py` (the
   command-line page prints the version); run `python tools/build_site_i18n.py --check`.
2. Full suite (`pytest -q -n auto`, ~2 minutes) and `python -m ruff check .`, then a PR "Release
   X.Y.Z" and merge it.
3. `gh release create vX.Y.Z --target <merge commit> --title "OriginStack X.Y.Z" --notes-file notes.md`.
   The notes are the "Which download?" table (see an earlier release) plus the changelog entry.
   Creating the release pushes the tag, which starts `.github/workflows/release.yml`: it checks
   `VERSION` equals the tag, builds Windows (installer, portable exe, zip), macOS arm64 and x86_64
   and Linux, and uploads nine files to the release. Check that all nine arrive.
4. A native-kernel change also needs `ext/astro_native/Cargo.toml` bumped (lint rule OS005). There is no
   native-wheel tag and no PyPI upload: the crate is not published (decided 2026-11-09), the downloads bundle
   their own build, and `.github/workflows/native-wheels.yml` only builds and smoke-tests the wheels on PRs.

Do not tag a release without deciding to: the tag triggers builds and public downloads.

## The sample sessions (Zenodo)

Twelve sessions, one per target type, 42-45 frames each, CC BY 4.0, author "OriginStack": see
`docs/sample-sessions.html` for the list, DOIs and checksums. Procedure, so it can be repeated or
extended (a new version of a record, a thirteenth session):

1. **Choose.** Score candidate sessions with the `--quality-sweep` metric (`quality_sweep._score_one`,
   8 evenly spaced lights per session) and rank within a target type by median FWHM, its spread,
   the share rejected and the star count; settle close calls by looking at the Origin's own
   `FinalStackedMaster.tiff`.
2. **Build.** `python tools/make_sample_session.py SESSION OUT --max-lights 45 --select best`
   copies the best-scoring light of each of 45 time bins, with the session's dark, flat, bias and
   `info.json`, and removes the location, airmass, session id and camera serial (it re-reads its
   output and fails if anything is left).
3. **Check.** `python tools/check_samples.py SAMPLES_DIR` stacks every session and compares it with
   `tools/data/samples_baseline.json` (frames accepted, target type, light time, star width, noise).
   A difference needs a look, not necessarily a fix; `--update` accepts new values.
4. **Upload.** Zip each session (a README inside), write a manifest (`records`: name, Zenodo metadata,
   files), then `python tools/upload_zenodo.py MANIFEST.json`. It creates private drafts, uploads,
   verifies size and MD5 and stops. `--drafts-only` creates drafts and reserved DOIs without files;
   `--status` shows them; `--check` validates the manifest offline. The token is the user
   environment variable `ZENODO_TOKEN` (create it on zenodo.org under Account > Applications with the
   scopes `deposit:write` and `deposit:actions`); never put it in a file or a commit. An upload of
   6 GB took about 4 hours on a 0.4 MB/s link and resumes after an interruption.
5. **Publish** only after reading each draft: `upload_zenodo.py MANIFEST.json --publish`
   (all) or `--publish NAME`. It refuses a record whose draft differs from the local files.
   **A published record cannot be deleted and its files cannot change**; metadata can be edited on
   the Zenodo site, and a new version is the way to change files. The tool does not edit published
   records.
6. **Page.** `docs/sample-sessions.html` is plain HTML (cards, schema.org Dataset markup for Google
   Dataset Search, thumbnails in `docs/img/samples/`). Edit it by hand for a new record; it is listed
   in `tools/build_site_i18n.py` (`ENGLISH_ONLY`) and in the sitemap, so `build_site_i18n.py` must
   be re-run after a change to either.

## What to check after a change

- Any change to Phase 1-3 settings: the checkpoint fingerprint (`_P1_EXTRA`/`_P23_EXTRA`, see
  `CLAUDE.md`).
- Any change to a Phase 4 step: `tests/test_phase4_snapshots.py`, and the early-step cache key.
- Any change to option help in `cli.py`: `tools/build_cli_reference.py` and the translations
  (`tools/check_translations.py`); to a website page: `tools/build_site_i18n.py`.
- Output-affecting changes: `tools/bench_phase4.py` on the held-out stacks, and
  `tools/check_samples.py` on the sample sessions.

## Known limits (so they are not mistaken for bugs)

- Everything was measured on one Origin camera unit and a few dozen sessions; other units, and other
  cameras, are less tested. Gains of the 2.21 changes: `CHANGELOG.md` and the notes below.
- Opt-in, checked on five stacks only, no comparison with truth data: `halo_removal_method=psf`
  ([halo-model](halo-model.md)), `nn_tilt_equalise` ([psf-tilt-equalise](psf-tilt-equalise.md)),
  `deconvolve_target_ratio`. The noise-map flattening of `--full-field` rims was tested on synthetic
  data only.
- Origin stacks have a clean star-width gradient along one sensor axis (12-55%, see
  [stack-optics](stack-optics.md)); `--deconvolve nn` flattens most of it.
- The research that led to 2.21, with 66 cited sources and what was tried and dropped:
  `reports/Astrophotography image quality techniques.md` and `research_notes/`. Ideas not built:
  a sensor-frame ghost and ring diagnostic, guided-filter tone mapping of bright cores, XP-based
  colour calibration, Ha/OIII extraction from the Bayer data (needs a real dual-band session).

## Where things are

`CLAUDE.md` (architecture, rules that must hold, what was tried and reverted), `dev-notes/` (per-topic
measurements), `tools/` (benchmarks, sample tools, the Zenodo uploader), `tests/` (2481).
