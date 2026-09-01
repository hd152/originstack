#!/usr/bin/env python3
"""OriginStack project-convention linter.

Enforces repo-specific rules that a generic style linter (ruff) cannot express.
Run standalone or via pytest (`tests/test_lint_conventions.py`); also wired into CI.

    python tools/lint_conventions.py             # check, human-readable
    python tools/lint_conventions.py --github     # GitHub Actions annotations
    python tools/lint_conventions.py --verbose    # list every OS003 site, not a summary
    python tools/lint_conventions.py --git BASE   # also run diff-aware rules vs BASE
    python tools/lint_conventions.py --strict     # warnings fail too

Rules
-----
OS001  logging.getLogger() must use the shared name "originstack", not __name__
       or the root logger, so records reach the configured handler. (error)
OS002  Known-optional third-party imports at module scope in src/ must sit inside
       a try/except so the pipeline degrades gracefully when absent. (error)
OS003  Library code under src/ should not call bare print(); use safe_print /
       print_phase so output tees into the desktop UI event sink. Summarised per
       file unless --verbose; per-line and diff-scoped under --git. (warning)
OS004  Every #[pyfunction] exported from the native crate should be referenced by
       name somewhere under tests/ (parity / smoke coverage). (warning)
OS005  --git only: if ext/astro_native/src/ changed vs BASE, the crate version in
       ext/astro_native/Cargo.toml must have changed too. (warning)
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

# Third-party packages this codebase treats as optional (import guarded by
# try/except, feature degrades if missing). Stdlib and hard deps (numpy, scipy)
# are intentionally excluded.
OPTIONAL_PACKAGES = {
    "rawpy", "tifffile", "cupy", "cupyx", "astro_native", "tqdm",
    "PIL", "psutil", "bm3d", "colour_demosaicing", "cv2", "matplotlib",
}

# src/ modules where bare print() is legitimate (they define/own the console path).
PRINT_OK = {"utils.py", "desktop_app.py", "native_dialog.py"}


class Finding:
    __slots__ = ("path", "line", "code", "msg", "level")

    def __init__(self, path, line, code, msg, level="error"):
        self.path = Path(path)
        self.line = line
        self.code = code
        self.msg = msg
        self.level = level

    def _rel(self) -> str:
        return os.path.relpath(self.path, ROOT)

    def format_plain(self) -> str:
        return f"{self._rel()}:{self.line}: {self.level.upper()} {self.code}: {self.msg}"

    def format_github(self) -> str:
        rel = self._rel().replace(os.sep, "/")
        kind = "error" if self.level == "error" else "warning"
        return f"::{kind} file={rel},line={self.line}::{self.code} {self.msg}"


def _parse(path: Path):
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as e:
        return e


def _module_level_imports(tree: ast.Module):
    """Yield (node, guarded) for imports at module scope, descending only through
    module-level `if` / `try` blocks (not into functions or classes)."""
    def walk(body, in_try):
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                yield node, in_try
            elif isinstance(node, ast.Try):
                yield from walk(node.body, True)
                for h in node.handlers:
                    yield from walk(h.body, True)
                yield from walk(node.orelse, in_try)
                yield from walk(node.finalbody, in_try)
            elif isinstance(node, ast.If):
                yield from walk(node.body, in_try)
                yield from walk(node.orelse, in_try)
    yield from walk(tree.body, False)


def check_logger_name(path, tree, out):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "getLogger"):
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id == "logging"):
            continue
        if not node.args:
            out.append(Finding(path, node.lineno, "OS001",
                               'logging.getLogger() uses the root logger; pass "originstack"'))
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and arg.value == "originstack":
            continue
        if isinstance(arg, ast.Name) and arg.id == "__name__":
            out.append(Finding(path, node.lineno, "OS001",
                               'logging.getLogger(__name__) -- use the shared name '
                               '"originstack" so records reach the configured handler'))
        elif isinstance(arg, (ast.Constant, ast.Name)):
            shown = getattr(arg, "value", getattr(arg, "id", "?"))
            out.append(Finding(path, node.lineno, "OS001",
                               f'logging.getLogger({shown!r}) -- expected "originstack"'))


def check_optional_imports(path, tree, out):
    for node, guarded in _module_level_imports(tree):
        if guarded:
            continue
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        else:
            names = [node.module or ""]
        for name in names:
            if name.split(".")[0] in OPTIONAL_PACKAGES:
                top = name.split(".")[0]
                out.append(Finding(path, node.lineno, "OS002",
                                   f"module-level import of optional package {top!r} is not "
                                   "inside a try/except -- src/ must degrade when it is absent"))


def _main_guard_lines(tree: ast.Module) -> set[int]:
    lines: set[int] = set()
    for node in tree.body:
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"):
            for child in ast.walk(node):
                if hasattr(child, "lineno"):
                    lines.add(child.lineno)
    return lines


def check_bare_print(path, tree, out):
    if path.name in PRINT_OK:
        return
    skip = _main_guard_lines(tree)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "print" and node.lineno not in skip):
            out.append(Finding(path, node.lineno, "OS003",
                               "bare print() in library code -- use safe_print / print_phase "
                               "so output reaches the desktop UI event sink", level="warning"))


def check_native_parity(out):
    lib = ROOT / "ext" / "astro_native" / "src" / "lib.rs"
    tests_dir = ROOT / "tests"
    if not lib.exists() or not tests_dir.exists():
        return
    lines = lib.read_text(encoding="utf-8", errors="replace").splitlines()
    names: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        if "#[pyfunction]" in line:
            for j in range(i + 1, min(i + 6, len(lines))):
                m = re.search(r"fn\s+([a-zA-Z_]\w*)\s*[<(]", lines[j])
                if m:
                    names.append((j + 1, m.group(1)))
                    break
    if not names:
        return
    test_blob = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                          for p in tests_dir.rglob("*.py"))
    for lineno, name in names:
        if name.startswith("_") or re.search(rf"\b{re.escape(name)}\b", test_blob):
            continue
        out.append(Finding(lib, lineno, "OS004",
                           f"native pyfunction {name!r} is not referenced under tests/ -- "
                           "add a parity or smoke test", level="warning"))


def _git(args: list[str]):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout


def check_cargo_version_bump(base, out):
    try:
        changed = _git(["diff", "--name-only", f"{base}...HEAD"]).split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return
    if not any(f.startswith("ext/astro_native/src/") for f in changed):
        return
    cargo_diff = ""
    if "ext/astro_native/Cargo.toml" in changed:
        try:
            cargo_diff = _git(["diff", f"{base}...HEAD", "--", "ext/astro_native/Cargo.toml"])
        except subprocess.CalledProcessError:
            pass
    if not re.search(r"^[+-]\s*version\s*=", cargo_diff, re.M):
        cargo = ROOT / "ext" / "astro_native" / "Cargo.toml"
        lineno = next((i for i, ln in enumerate(
            cargo.read_text(encoding="utf-8").splitlines(), 1)
            if ln.strip().startswith("version")), 1) if cargo.exists() else 1
        out.append(Finding(cargo, lineno, "OS005",
                           f"native crate source changed vs {base} but Cargo.toml version "
                           "was not bumped", level="warning"))


def _changed_lines(base: str) -> dict[str, set[int]]:
    """Map repo-relative path -> set of added line numbers in `base...HEAD`."""
    try:
        diff = _git(["diff", "--unified=0", f"{base}...HEAD"])
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}
    out: dict[str, set[int]] = {}
    cur = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            cur = line[6:]
            out.setdefault(cur, set())
        elif line.startswith("@@") and cur is not None:
            m = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if m:
                start = int(m.group(1))
                count = int(m.group(2) or "1")
                out[cur].update(range(start, start + count))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--github", action="store_true", help="emit GitHub Actions annotations")
    ap.add_argument("--verbose", action="store_true", help="list every OS003 site")
    ap.add_argument("--git", metavar="BASE", help="also run diff-aware rules against BASE ref")
    ap.add_argument("--strict", action="store_true", help="treat warnings as errors")
    args = ap.parse_args(argv)

    findings: list[Finding] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = _parse(path)
        if isinstance(tree, SyntaxError):
            findings.append(Finding(path, tree.lineno or 1, "OS000",
                                    f"syntax error: {tree.msg}"))
            continue
        check_logger_name(path, tree, findings)
        check_optional_imports(path, tree, findings)
        check_bare_print(path, tree, findings)

    check_native_parity(findings)
    if args.git:
        check_cargo_version_bump(args.git, findings)

    # OS003 handling: diff-scope under --git, summarise otherwise (unless --verbose).
    if args.git:
        touched = _changed_lines(args.git)
        findings = [f for f in findings if f.code != "OS003"
                    or f.line in touched.get(f._rel().replace(os.sep, "/"), set())]
    elif not args.verbose:
        by_file: dict[str, list[Finding]] = {}
        kept: list[Finding] = []
        for f in findings:
            if f.code == "OS003":
                by_file.setdefault(f._rel(), []).append(f)
            else:
                kept.append(f)
        for rel, fs in by_file.items():
            kept.append(Finding(ROOT / rel, min(x.line for x in fs), "OS003",
                                f"{len(fs)} bare print() call(s) in library code "
                                "(run --verbose to list, --git to scope to your diff)",
                                level="warning"))
        findings = kept

    findings.sort(key=lambda f: (f._rel(), f.line, f.code))
    for f in findings:
        print(f.format_github() if args.github else f.format_plain())

    errors = [f for f in findings if f.level == "error"
              or (args.strict and f.level == "warning")]
    warns = [f for f in findings if f.level == "warning"]
    print(f"\n{len(errors)} error(s), {len(warns)} warning(s)", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
