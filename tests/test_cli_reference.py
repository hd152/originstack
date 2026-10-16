"""The website's command-line reference is generated from the CLI's parsers and must be current."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))


def test_cli_reference_is_current():
    import build_cli_reference
    text = build_cli_reference.render()
    assert build_cli_reference.PAGE.read_text(encoding='utf-8') == text, \
        'run python tools/build_cli_reference.py'


def test_cli_reference_lists_every_visible_option():
    import argparse

    import build_cli_reference

    from src.cli import build_parser
    text = build_cli_reference.PAGE.read_text(encoding='utf-8')
    for action in build_parser()._actions:
        if action.help == argparse.SUPPRESS or isinstance(action, argparse._HelpAction):
            continue
        for opt in action.option_strings:
            assert f'<code>{opt}</code>' in text, opt


def test_cli_reference_leaves_out_removed_flags():
    import build_cli_reference

    from src.cli import _RemovedFlag, build_parser
    text = build_cli_reference.PAGE.read_text(encoding='utf-8')
    removed = [o for a in build_parser()._actions if isinstance(a, _RemovedFlag) for o in a.option_strings]
    assert removed
    assert not any(f'<code>{o}</code>' in text for o in removed)
