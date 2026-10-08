"""List and check the desktop app's translatable strings.

The keys are the English texts the app shows (``src/i18n.py``): every literal
passed to ``_()`` or ``N_()`` in the desktop modules, plus what the Setup form
builds from ``cli.build_parser()`` -- group titles, field labels, the one-line
summaries and the full help shown in tooltips -- and the target-type names.

    python tools/i18n_strings.py                 # coverage per language
    python tools/i18n_strings.py --missing de    # JSON of untranslated keys for one language
    python tools/i18n_strings.py --prune         # drop keys the app no longer shows

Exit status is 1 if a catalog has a translation whose {placeholders} differ
from its English key (that would raise at display time).
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOCALE_DIR = ROOT / 'src' / 'locales'
SOURCES = [ROOT / 'src' / 'desktop_app.py']
_PLACEHOLDER = re.compile(r'\{(\w*)\}')


def _literal_calls(path: Path) -> Iterable[str]:
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ('_', 'N_') and node.args
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            yield node.args[0].value


def app_keys() -> List[str]:
    keys: Set[str] = set()
    for path in SOURCES:
        keys.update(_literal_calls(path))
    from src.auto_settings import TARGET_LABELS
    from src.desktop_app import _field_label
    from src.desktop_control import get_form_schema
    keys.update(TARGET_LABELS.values())
    for group, fields in get_form_schema().items():
        keys.add(group)
        for f in fields:
            keys.add(_field_label(f))
            if f.get('summary'):
                keys.add(f['summary'])
            if f.get('help'):
                keys.add(f['help'])
    return sorted(k for k in keys if k.strip())


def placeholders(text: str) -> Set[str]:
    return set(_PLACEHOLDER.findall(text))


def load(code: str) -> Dict[str, str]:
    path = LOCALE_DIR / f'{code}.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def save(code: str, data: Dict[str, str]) -> None:
    meta = {k: v for k, v in data.items() if k.startswith('_')}
    body = {k: data[k] for k in sorted(k for k in data if not k.startswith('_'))}
    (LOCALE_DIR / f'{code}.json').write_text(
        json.dumps({**meta, **body}, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


def placeholder_errors(code: str, data: Dict[str, str]) -> List[str]:
    return [k for k, v in data.items()
            if not k.startswith('_') and v and placeholders(k) != placeholders(v)]


def main(argv=None) -> int:
    from src.i18n import LANGUAGES
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--missing', metavar='LANG', help='print untranslated keys for LANG as JSON')
    ap.add_argument('--prune', action='store_true', help='remove keys the app no longer uses')
    args = ap.parse_args(argv)

    keys = app_keys()
    if args.missing:
        data = load(args.missing)
        print(json.dumps({k: '' for k in keys if not data.get(k)}, ensure_ascii=False, indent=1))
        return 0

    bad = False
    key_set = set(keys)
    for code in LANGUAGES:
        if code == 'en':
            continue
        data = load(code)
        if args.prune:
            data = {k: v for k, v in data.items() if k.startswith('_') or k in key_set}
            save(code, data)
        done = sum(1 for k in keys if data.get(k))
        stale = [k for k in data if not k.startswith('_') and k not in key_set]
        errors = placeholder_errors(code, data)
        bad |= bool(errors)
        print(f'{code:6} {done}/{len(keys)} translated, {len(stale)} stale, '
              f'{len(errors)} placeholder errors')
        for k in errors[:5]:
            print(f'   placeholder mismatch: {k!r}')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
