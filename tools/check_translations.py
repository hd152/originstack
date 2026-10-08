"""Check translation catalogs against the English keys they translate.

For the desktop app (``src/locales/<code>.json``) and the website
(``i18n/site/<code>.json``): every key must still exist in the English, every
translation must be non-empty, keep the same ``{placeholders}``, and, on the
website, keep the same HTML tags (each tag, attributes and all, the same number of times --
word order may move them). Coverage is reported, not required: an
untranslated string falls back to English.

    python tools/check_translations.py           # all languages
    python tools/check_translations.py de ja     # some
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

_TAG = re.compile(r'</?[a-zA-Z][^<>]*>')
_PH = re.compile(r'\{(\w*)\}')


def problems(keys: List[str], catalog: Dict[str, str], html: bool = False) -> List[str]:
    key_set = set(keys)
    out = []
    for k, v in catalog.items():
        if k.startswith('_'):
            continue
        if k not in key_set:
            out.append(f'unknown key: {k[:70]!r}')
            continue
        if not isinstance(v, str) or not v.strip():
            out.append(f'empty: {k[:70]!r}')
            continue
        if set(_PH.findall(k)) != set(_PH.findall(v)):
            out.append(f'placeholders differ: {k[:70]!r}')
        if html and Counter(_TAG.findall(k)) != Counter(_TAG.findall(v)):
            out.append(f'HTML tags differ: {k[:70]!r}')
    return out


def main(argv=None) -> int:
    import build_site_i18n
    import i18n_strings

    from src.i18n import LANGUAGES
    codes = (argv if argv is not None else sys.argv[1:]) or [c for c in LANGUAGES if c != 'en']
    sets = [('app', i18n_strings.app_keys(), ROOT / 'src' / 'locales'),
            ('site', build_site_i18n.all_keys(), ROOT / 'i18n' / 'site')]
    failed = False
    for code in codes:
        for name, keys, folder in sets:
            path = folder / f'{code}.json'
            catalog = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
            found = problems(keys, catalog, html=(name == 'site'))
            done = sum(1 for k in keys if catalog.get(k))
            print(f'{code:6} {name:4} {done}/{len(keys)} translated, {len(found)} problems')
            for p in found[:10]:
                print('    ' + p)
            failed |= bool(found)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
