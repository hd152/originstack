"""``--help`` must render: argparse %-formats every help string, so a bare '%' in one
(``--cfa-drizzle``'s "luma noise -13% but ...") made ``originstack.py --help`` raise
ValueError for every user."""
import subprocess
import sys
import tempfile

import pytest

from src.cli import build_parser


def test_parser_renders_full_help():
    text = build_parser().format_help()
    assert '--offline' in text and '--no-auto' in text


def test_every_help_string_formats_with_argparse_params():
    for action in build_parser()._actions:
        if isinstance(action.help, str):
            params = dict(vars(action), prog='originstack')
            try:
                action.help % params
            except (ValueError, KeyError, TypeError) as exc:      # pragma: no cover - failure path
                pytest.fail(f"{action.option_strings}: help does not %-format ({exc})")


def test_help_flag_exits_cleanly_from_the_command_line():
    with tempfile.TemporaryDirectory() as cwd:
        proc = subprocess.run([sys.executable, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'originstack.py'), '--help'],
                              capture_output=True, text=True, cwd=cwd, timeout=120)
    assert proc.returncode == 0, proc.stderr[-400:]
    assert '--offline' in proc.stdout
