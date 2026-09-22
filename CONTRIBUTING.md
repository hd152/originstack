# Contributing to OriginStack

Thanks for taking a look. This is a small solo-maintained project, so keep it simple: open an issue
before a large PR so effort isn't wasted, and small fixes/PRs are welcome directly.

## Before you start

- **Bug?** Use the [bug report template](https://github.com/hd152/originstack/issues/new/choose) —
  it asks for the version, OS, and a log, which is usually what's needed to reproduce a real-session
  issue.
- **Feature idea or "how do I...":** [Discussions](https://github.com/hd152/originstack/discussions)
  is a better fit than an issue for anything that isn't a concrete bug.
- **Larger change:** open an issue first describing what you want to do and why, so the approach can
  be discussed before you write code.

## Setup

```bash
git clone https://github.com/hd152/originstack
cd originstack
python -m venv .venv && .venv\Scripts\activate    # or: source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt    # ruff + test tooling
pip install pytest
```

The native (Rust) extension is optional for development — everything has a NumPy fallback. If you're
touching `ext/astro_native/`, see "Native (Rust) acceleration" in the README for the build steps.

## Before opening a PR

```bash
python -m pytest -q                            # tests must pass
python -m ruff check .                         # style
python tools/lint_conventions.py               # project-specific conventions
python tools/lint_conventions.py --git origin/main   # diff-scoped checks (print noise, Cargo bump)
```

All four are gated in CI. `tools/lint_conventions.py` explains each rule (OS001–OS005) when it fires —
read the message rather than working around it.

## Conventions

- **Branch:** work happens on `develop`; it's merged into `main` for releases. A small fix can go
  straight to `develop`.
- **Commit messages:** a one-line summary, then (for anything non-trivial) a paragraph explaining
  *why*, not just what changed — see recent commits in `git log` for the house style. Include
  measurements for anything performance-related (this repo cares a lot about "measured, not assumed").
- **New native (Rust) kernel:** bump `ext/astro_native/Cargo.toml`'s version, add a parity test against
  the NumPy fallback, and reference the new `#[pyfunction]` from a test (`tools/lint_conventions.py`'s
  OS004/OS005 check for both).
- **Logging:** use `logging.getLogger("originstack")`, never a bare `__name__` or the root logger
  (OS001).
- **Optional dependencies:** any module-level import of an optional third-party package in `src/` must
  be `try/except`-guarded (OS002) — the pipeline should degrade gracefully, not crash, when e.g.
  `rawpy` isn't installed.
- **No bare `print()`** in library code under `src/` — use `safe_print` (OS003, warning-level).
- **Claims need a measurement.** If a change is described as faster, more accurate, or fixing a real
  bug, say what was measured and how (synthetic test, real session, before/after numbers). See
  `CLAUDE.md` for the level of detail expected — it's the project's own memory of what's been tried,
  measured, and sometimes reverted.

## What CLAUDE.md is

`CLAUDE.md` is a working log of the codebase for AI coding assistance, but it doubles as the most
detailed source of "why is it built this way" for a human contributor too — worth reading before a
non-trivial change, especially in `src/background.py`, `src/registration.py`, or anything in
"Native (Rust) acceleration."

## License

By contributing, you agree your contribution is licensed under this project's [MIT license](LICENSE).
