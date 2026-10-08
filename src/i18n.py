"""User-interface translations for the desktop app.

The English text is the key: ``_('Start')`` returns the translation for the
current language, or the English unchanged when there is none, so a missing
string never breaks the app. Templates keep ``{name}`` placeholders and are
filled by ``_(text, **values)``. ``N_()`` marks a string for translation where
it is defined (a module-level table) and is translated where it is shown.

Catalogs are JSON files in ``src/locales/<code>.json`` mapping English to the
translation; ``tools/i18n_strings.py`` lists the strings to translate and
checks the catalogs. Only the desktop app's own text is translated: the log,
progress messages and the command line stay English, so a log a user sends
with a problem report is readable and matches the source.

The language is, in order: ``$ORIGINSTACK_LANG``, the choice saved from the
app's language menu, then the operating system's display language, then
English.
"""
from __future__ import annotations

import json
import locale
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

_LOCALE_DIR = Path(__file__).resolve().parent / 'locales'

# code -> the language's own name, as shown in the language menu.
LANGUAGES: Dict[str, str] = {
    'en': 'English',
    'de': 'Deutsch',
    'es': 'Español',
    'fr': 'Français',
    'it': 'Italiano',
    'pt_BR': 'Português (Brasil)',
    'ja': '日本語',
    'zh_CN': '简体中文',
}

_catalog: Dict[str, str] = {}
_current = 'en'


def N_(text: str) -> str:
    """Mark *text* for translation without translating it yet."""
    return text


def _(text: str, **values) -> str:
    """*text* in the current language, with ``{name}`` placeholders filled."""
    out = _catalog.get(text) or text
    if values:
        try:
            return out.format(**values)
        except (KeyError, IndexError, ValueError):
            # A broken translation must not stop the app: fall back to English.
            return text.format(**values)
    return out


def current_language() -> str:
    return _current


def normalize(tag: Optional[str]) -> Optional[str]:
    """Map a locale tag ('de_DE.UTF-8', 'pt-BR', 'zh-Hans-CN', 'fr') to a
    supported code, or None. Portuguese goes to Brazilian Portuguese; Chinese
    only in its Simplified forms (a Traditional-Chinese reader gets English
    rather than a script they did not ask for)."""
    if not tag:
        return None
    t = tag.split('.')[0].split('@')[0].replace('-', '_')
    parts = t.split('_')
    lang = parts[0].lower()
    rest = [p.upper() for p in parts[1:]]
    if lang == 'zh':
        if 'HANT' in rest or any(r in ('TW', 'HK', 'MO') for r in rest):
            return None
        return 'zh_CN'
    if lang == 'pt':
        return 'pt_BR'
    return lang if lang in LANGUAGES else None


def system_language() -> str:
    """The operating system's display language as a supported code, else 'en'."""
    candidates = []
    try:
        if sys.platform == 'win32':
            import ctypes
            langid = ctypes.windll.kernel32.GetUserDefaultUILanguage()
            candidates.append(locale.windows_locale.get(langid))
        elif sys.platform == 'darwin':
            # GUI apps on macOS usually start without LANG; the preferred
            # languages list is the user's actual choice.
            out = subprocess.run(['defaults', 'read', '-g', 'AppleLanguages'],
                                 capture_output=True, text=True, timeout=3).stdout
            candidates += [s.strip().strip('",') for s in out.splitlines()
                           if s.strip() not in ('(', ')', '')][:3]
    except Exception:
        pass
    for var in ('LANGUAGE', 'LC_ALL', 'LC_MESSAGES', 'LANG'):
        value = os.environ.get(var)
        if value:
            candidates += value.split(':')
    for c in candidates:
        code = normalize(c)
        if code:
            return code
    return 'en'


def _settings_path() -> Path:
    if sys.platform == 'win32':
        base = Path(os.environ.get('APPDATA') or Path.home())
    elif sys.platform == 'darwin':
        base = Path.home() / 'Library' / 'Application Support'
    else:
        base = Path(os.environ.get('XDG_CONFIG_HOME') or Path.home() / '.config')
    return base / 'OriginStack' / 'settings.json'


def saved_choice() -> str:
    """'auto' or a language code saved from the app's language menu."""
    try:
        value = json.loads(_settings_path().read_text(encoding='utf-8')).get('language', 'auto')
        return value if value == 'auto' or value in LANGUAGES else 'auto'
    except (OSError, ValueError, AttributeError):
        return 'auto'


def save_choice(choice: str) -> None:
    path = _settings_path()
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    data['language'] = choice
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    except OSError:
        pass


def resolve_language() -> str:
    """The language to use: $ORIGINSTACK_LANG, then the saved choice, then the system."""
    env = normalize(os.environ.get('ORIGINSTACK_LANG')) if os.environ.get('ORIGINSTACK_LANG') else None
    if os.environ.get('ORIGINSTACK_LANG', '').lower() == 'en':
        return 'en'
    if env:
        return env
    choice = saved_choice()
    if choice != 'auto':
        return choice
    return system_language()


def load_catalog(code: str) -> Dict[str, str]:
    if code == 'en':
        return {}
    try:
        data = json.loads((_LOCALE_DIR / f'{code}.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items()
            if isinstance(k, str) and isinstance(v, str) and v and not k.startswith('_')}


def set_language(code: str) -> str:
    """Switch the process to *code* (unknown codes fall back to English)."""
    global _catalog, _current
    code = code if code in LANGUAGES else 'en'
    _catalog = load_catalog(code)
    _current = code
    return code
