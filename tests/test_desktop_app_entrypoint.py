"""Regression test for the root desktop_app.py entry-point shim.

Real-world bug: a packaged (PyInstaller) build on Windows opened one full
GUI window per CPU core the moment Phase 1 started. Root cause:
ProcessPoolExecutor's 'spawn' start method re-executes the frozen exe for
every worker, and without multiprocessing.freeze_support() as the first
statement in `if __name__ == '__main__':`, each spawned worker fell through
to this file's own main() instead of the multiprocessing worker bootstrap --
re-launching the whole app once per worker.

Can't exercise this at runtime in a unit test (it only manifests via a real
spawned child process re-executing a frozen exe), so this checks the source
directly: freeze_support() must be called, and must come before main().
"""
from __future__ import annotations

import ast
from pathlib import Path


def _main_guard_body() -> list:
    src = (Path(__file__).resolve().parent.parent / "desktop_app.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"):
            return node.body
    raise AssertionError("desktop_app.py has no `if __name__ == '__main__':` block")


def _call_names(body: list) -> list:
    """Top-level call expressions in source order, as dotted-name strings."""
    names = []
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call):
                func = node.func
                parts = []
                while isinstance(func, ast.Attribute):
                    parts.append(func.attr)
                    func = func.value
                if isinstance(func, ast.Name):
                    parts.append(func.id)
                names.append(".".join(reversed(parts)))
    return names


def test_freeze_support_called_before_main():
    calls = _call_names(_main_guard_body())
    assert "multiprocessing.freeze_support" in calls, (
        "desktop_app.py's __main__ guard must call multiprocessing.freeze_support() "
        "-- without it, a frozen build's spawned ProcessPoolExecutor workers "
        "re-launch the whole GUI instead of running as worker processes."
    )
    freeze_idx = calls.index("multiprocessing.freeze_support")
    main_idx = next(i for i, c in enumerate(calls) if c.endswith("main"))
    assert freeze_idx < main_idx, "freeze_support() must be called before main()"
