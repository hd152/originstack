"""Guards that keep the two linters green.

- ruff (style) must report no violations under the repo's pyproject.toml config.
- tools/lint_conventions.py (project conventions) must report no *errors*.

Plus a few unit checks of the convention rules against synthetic snippets so a
broken rule is caught here rather than by a silent pass in CI.
"""
from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_HAS_RUFF = importlib.util.find_spec("ruff") is not None

ROOT = Path(__file__).resolve().parent.parent
LINTER = ROOT / "tools" / "lint_conventions.py"

_spec = importlib.util.spec_from_file_location("lint_conventions", LINTER)
lc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lc)


@pytest.mark.skipif(not _HAS_RUFF, reason="ruff not installed")
def test_ruff_clean():
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--output-format=concise", "."],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_conventions_no_errors():
    rc = lc.main(["--github"])
    assert rc == 0, "tools/lint_conventions.py reported errors (see stdout above)"


# --- rule unit checks -------------------------------------------------------

def _run(check, src: str):
    out: list = []
    check(Path("mem.py"), ast.parse(src), out)
    return [f.code for f in out]


def test_os001_flags_dunder_and_root():
    assert _run(lc.check_logger_name, "import logging\nx = logging.getLogger(__name__)\n") == ["OS001"]
    assert _run(lc.check_logger_name, "import logging\nx = logging.getLogger()\n") == ["OS001"]
    assert _run(lc.check_logger_name, 'import logging\nx = logging.getLogger("originstack")\n') == []


def test_os002_module_scope_only():
    assert _run(lc.check_optional_imports, "import rawpy\n") == ["OS002"]
    assert _run(lc.check_optional_imports, "try:\n    import rawpy\nexcept ImportError:\n    rawpy = None\n") == []
    # function-local lazy import is allowed
    assert _run(lc.check_optional_imports, "def f():\n    import rawpy\n    return rawpy\n") == []
    # hard deps are not in the optional set
    assert _run(lc.check_optional_imports, "import numpy\n") == []


def test_os003_skips_main_guard():
    body = "def f():\n    print('x')\n"
    assert _run(lc.check_bare_print, body) == ["OS003"]
    guarded = "if __name__ == '__main__':\n    print('x')\n"
    assert _run(lc.check_bare_print, guarded) == []
