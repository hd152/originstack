# OriginStack Project Guidelines

## Repository guidance

- Read [CLAUDE.md](../CLAUDE.md) before making non-trivial changes. It is the source of truth for architecture, pipeline invariants, native-extension rules, packaging, networking, and translation workflows.
- Keep `originstack.py` and `desktop_app.py` as thin compatibility shims; implementation belongs under `src/`.
- Read the relevant file in `dev-notes/` before changing behavior that depends on performance measurements, pipeline phases, checkpointing, memory placement, or native kernels.

## Validation

- Run the smallest relevant test selection while iterating; run `pytest -q` when changes span multiple pipeline components.
- For lint-related changes, run:
  - `python -m ruff check .`
  - `python tools/lint_conventions.py`
- Use the repository's documented commands for synthetic-stack smoke tests and packaging validation when those areas change.
- Do not add or require optional dependencies without preserving the guarded fallback behavior described in `CLAUDE.md`.

## Implementation conventions

- Preserve bit-identical NumPy/native behavior where the repository requires it. Any native kernel needs a NumPy fallback and a parity test in `tests/test_native.py`; bump `ext/astro_native/Cargo.toml` when native source changes.
- Use `logging.getLogger("originstack")` and `safe_print` in library code. Guard optional imports in `src/`.
- Route HTTP through `src/net_query.py` so `--offline` remains effective.
- Keep Phase 1–3 settings in checkpoint fingerprints and preserve `--from-stack` cache/config semantics when adding options or changing pipeline boundaries.
- Check cancellation and progress behavior in long-running loops, and preserve explicit `RunCancelled` handling.
- Keep user-visible desktop strings translatable with `_()`/`N_()`; update the English source and regenerate derived website translations rather than editing generated language pages directly.

## Scope discipline

- Make surgical changes, follow existing helpers and naming, and avoid unrelated cleanup.
- Add or update tests for changed behavior and update directly related documentation.
- Never commit secrets or generated artifacts unless the repository workflow explicitly requires them.
