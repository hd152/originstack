"""Translations: the lookup and language choice in src/i18n.py, the app and
website catalogs, and that the generated website pages are up to date."""
import json
import sys
from pathlib import Path

import pytest

from src import i18n

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))


@pytest.fixture
def english():
    yield
    i18n.set_language('en')


def test_missing_string_falls_back_to_english(english):
    i18n.set_language('de')
    assert i18n._('A string no catalog has') == 'A string no catalog has'
    assert i18n._('{n} untranslated', n=3) == '3 untranslated'


def test_broken_translation_falls_back_instead_of_raising(english, monkeypatch):
    monkeypatch.setattr(i18n, '_catalog', {'{n} lights': '{count} Lights'})
    assert i18n._('{n} lights', n=4) == '4 lights'


@pytest.mark.parametrize('code', [c for c in i18n.LANGUAGES if c != 'en'])
def test_translation_is_used(code, english):
    i18n.set_language(code)
    assert i18n._('Cancel') != 'Cancel'
    assert i18n.current_language() == code


def test_unknown_language_is_english(english):
    assert i18n.set_language('xx') == 'en'
    assert i18n._('Start') == 'Start'


@pytest.mark.parametrize('tag, code', [
    ('de_DE.UTF-8', 'de'), ('fr-CA', 'fr'), ('pt-BR', 'pt_BR'), ('pt_PT', 'pt_BR'),
    ('zh-Hans-CN', 'zh_CN'), ('zh_CN', 'zh_CN'), ('zh-TW', None), ('zh-Hant', None),
    ('ja-JP', 'ja'), ('en_GB', 'en'), ('nl_NL', None), ('', None), (None, None)])
def test_normalize(tag, code):
    assert i18n.normalize(tag) == code


def test_resolve_order(monkeypatch, tmp_path):
    monkeypatch.setattr(i18n, '_settings_path', lambda: tmp_path / 'settings.json')
    monkeypatch.setattr(i18n, 'system_language', lambda: 'fr')
    monkeypatch.delenv('ORIGINSTACK_LANG', raising=False)
    assert i18n.resolve_language() == 'fr'
    i18n.save_choice('ja')
    assert i18n.saved_choice() == 'ja'
    assert i18n.resolve_language() == 'ja'
    monkeypatch.setenv('ORIGINSTACK_LANG', 'de')
    assert i18n.resolve_language() == 'de'
    monkeypatch.setenv('ORIGINSTACK_LANG', 'en')
    assert i18n.resolve_language() == 'en'


def test_saved_choice_survives_a_corrupt_settings_file(monkeypatch, tmp_path):
    path = tmp_path / 'settings.json'
    path.write_text('not json', encoding='utf-8')
    monkeypatch.setattr(i18n, '_settings_path', lambda: path)
    assert i18n.saved_choice() == 'auto'
    i18n.save_choice('de')
    assert json.loads(path.read_text(encoding='utf-8')) == {'language': 'de'}


def test_every_language_has_both_catalogs():
    for code in i18n.LANGUAGES:
        if code == 'en':
            continue
        assert (ROOT / 'src' / 'locales' / f'{code}.json').exists(), code
        assert (ROOT / 'i18n' / 'site' / f'{code}.json').exists(), code


def test_catalogs_keep_placeholders_and_markup():
    """A dropped {placeholder} raises at display time and a dropped tag breaks
    a page; coverage itself is not required (missing text stays English)."""
    import check_translations
    assert check_translations.main([]) == 0


def test_generated_site_pages_are_up_to_date():
    import build_site_i18n
    files, _missing = build_site_i18n.outputs()
    stale = [str(p.relative_to(ROOT)) for p, text in files.items()
             if not p.exists() or p.read_text(encoding='utf-8') != text]
    assert not stale, f'run python tools/build_site_i18n.py: {stale}'


def test_translated_page_links_resolve():
    """Every relative link in a translated page points at a file that exists."""
    import re

    import build_site_i18n
    for _code, folder, *_rest in build_site_i18n.LANGS:
        for page in build_site_i18n.PAGES:
            path = ROOT / 'docs' / folder / page
            text = path.read_text(encoding='utf-8')
            urls = re.findall(r'\b(?:href|src)="([^"]+)"', text)
            for s in re.findall(r'\b(?:srcset|imagesrcset)="([^"]+)"', text):
                urls += [item.split()[0] for item in s.split(',') if item.strip()]
            for url in urls:
                if re.match(r'^([a-z]+:|//|#)', url):
                    continue
                target = (path.parent / url.split('#')[0].split('?')[0]).resolve()
                if url.split('#')[0] in ('', './', '../') or url.endswith('/'):
                    target = target / 'index.html'
                assert target.exists(), f'{folder}/{page}: {url}'


@pytest.fixture(scope='module')
def tk_root():
    # One root for the module, as in test_desktop_setup_form.py: a root per
    # test intermittently crashed Tk on Windows.
    tk = pytest.importorskip('tkinter')
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('no display available for a real Tk root')
    root.withdraw()
    from src import desktop_app
    desktop_app._apply_theme(root)
    yield root
    root.destroy()


@pytest.mark.parametrize('code', [c for c in i18n.LANGUAGES if c != 'en'])
def test_setup_form_builds_in_every_language(code, tk_root, english):
    from src import desktop_app
    i18n.set_language(code)
    form = desktop_app.SetupForm(tk_root)
    try:
        form.pack()
        tk_root.update_idletasks()
        rows = form.describe_run()
        assert rows and all(isinstance(h, str) and isinstance(t, str) for h, t in rows)
        assert rows[0][0] == i18n._('Frames')
        # Translated text never reaches what Start submits.
        assert form.read_form() == {}
    finally:
        form.destroy()
