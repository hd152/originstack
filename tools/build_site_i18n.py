"""Build the translated website pages from the English ones.

The English pages in ``docs/`` are the source. Each language's catalog,
``i18n/site/<code>.json``, maps an English text segment to its translation;
this script writes ``docs/<folder>/<page>`` for every language, with the
links, ``lang``, canonical URL and structured data adjusted, and keeps the
language switcher, the ``hreflang`` alternates and the sitemap in step on the
English pages too.

A segment is a run of text and inline markup between block-level tags (a
paragraph, a heading, a list item, a table cell...), with its inline tags
kept, plus the translatable attributes (alt, aria-label, title, the meta
descriptions) and the text fields of the JSON-LD and ``#i18n`` blocks. Code
blocks (``<pre>``), scripts and styles are left alone. A segment with no
translation stays English.

    python tools/build_site_i18n.py              # write the pages
    python tools/build_site_i18n.py --check      # exit 1 if any page is out of date
    python tools/build_site_i18n.py --missing de # JSON of untranslated segments
"""
from __future__ import annotations

import argparse
import html
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / 'docs'
CATALOGS = ROOT / 'i18n' / 'site'
SITE = 'https://originstack.site/'

PAGES = ['index.html', 'stack-celestron-origin.html', 'privacy.html']
# Pages in the sitemap that are not translated.
ENGLISH_ONLY = ['advanced.html']

# catalog code, folder, hreflang / html lang, og:locale, own name
LANGS = [
    ('de', 'de', 'de', 'de_DE', 'Deutsch'),
    ('es', 'es', 'es', 'es_ES', 'Español'),
    ('fr', 'fr', 'fr', 'fr_FR', 'Français'),
    ('it', 'it', 'it', 'it_IT', 'Italiano'),
    ('nl', 'nl', 'nl', 'nl_NL', 'Nederlands'),
    ('pt_BR', 'pt-br', 'pt-BR', 'pt_BR', 'Português'),
    ('ja', 'ja', 'ja', 'ja_JP', '日本語'),
    ('zh_CN', 'zh-cn', 'zh-CN', 'zh_CN', '简体中文'),
]

INLINE = {'a', 'abbr', 'b', 'br', 'code', 'em', 'i', 'kbd', 'mark', 's', 'small',
          'strong', 'sub', 'sup', 'time', 'u', 'wbr'}
RAW_TEXT = {'script', 'style', 'pre'}
TEXT_ATTRS = ('alt', 'aria-label', 'title', 'placeholder')
META_NAMES = {'description', 'twitter:title', 'twitter:description'}
META_PROPS = {'og:title', 'og:description', 'og:image:alt'}
JSONLD_FIELDS = ('description', 'headline')
_LETTER = re.compile(r'[^\W\d_]', re.UNICODE)
_URL_ATTR = re.compile(r'\b(href|src)="([^"]*)"')
_SRCSET_ATTR = re.compile(r'\b(srcset|imagesrcset)="([^"]*)"')


def norm(text: str) -> str:
    return ' '.join(text.split())


# ── tokenizing ───────────────────────────────────────────────────────────

class _Tokens(HTMLParser):
    """The document as raw-text tokens, so unchanged parts are written back
    byte for byte."""

    def __init__(self, text: str):
        super().__init__(convert_charrefs=False)
        self.toks: List[Tuple[str, str, Optional[str]]] = []  # (kind, raw, tag)
        self.feed(text)
        self.close()

    def handle_starttag(self, tag, attrs):
        self.toks.append(('start', self.get_starttag_text(), tag))

    def handle_startendtag(self, tag, attrs):
        self.toks.append(('start', self.get_starttag_text(), tag))

    def handle_endtag(self, tag):
        self.toks.append(('end', f'</{tag}>', tag))

    def handle_data(self, data):
        self.toks.append(('data', data, None))

    def handle_entityref(self, name):
        self.toks.append(('data', f'&{name};', None))

    def handle_charref(self, name):
        self.toks.append(('data', f'&#{name};', None))

    def handle_comment(self, data):
        self.toks.append(('comment', f'<!--{data}-->', None))

    def handle_decl(self, decl):
        self.toks.append(('decl', f'<!{decl}>', None))

    def handle_pi(self, data):
        self.toks.append(('decl', f'<?{data}>', None))


def _attr(raw: str, name: str) -> Optional[str]:
    m = re.search(r'\s' + re.escape(name) + r'="([^"]*)"', raw)
    return html.unescape(m.group(1)) if m else None


def _set_attr(raw: str, name: str, value: str) -> str:
    esc = html.escape(value, quote=True)
    return re.sub(r'(\s' + re.escape(name) + r'=")[^"]*(")', lambda m: m.group(1) + esc + m.group(2),
                  raw, count=1)


# ── walking a page ───────────────────────────────────────────────────────

class Page:
    """One English page, walked once; ``render`` produces any language."""

    def __init__(self, name: str):
        self.name = name
        self.text = (DOCS / name).read_text(encoding='utf-8')
        self.toks = _Tokens(self.text).toks

    def _walk(self, tr):
        """Yield output pieces, calling ``tr(english) -> translated`` for each
        segment and attribute; ``tr`` collects keys when extracting."""
        out: List[str] = []
        buf: List[str] = []
        raw_tag: Optional[str] = None      # inside script/style/pre
        raw_attrs = ''
        skip = False                       # inside a generated i18n block

        def flush():
            if not buf:
                return
            raw = ''.join(buf)
            buf.clear()
            if skip or not _LETTER.search(html.unescape(re.sub(r'<[^>]+>', '', raw))):
                out.append(raw)
                return
            lead = raw[:len(raw) - len(raw.lstrip())]
            trail = raw[len(raw.rstrip()):]
            out.append(lead + tr(norm(raw)) + trail)

        for kind, raw, tag in self.toks:
            if raw_tag is not None:
                if kind == 'end' and tag == raw_tag:
                    raw_tag = None
                    out.append(raw)
                elif kind == 'data' and raw_tag == 'script':
                    out.append(self._script(raw, raw_attrs, tr))
                else:
                    out.append(raw)
                continue
            if kind == 'comment':
                flush()
                if raw.startswith('<!-- i18n:'):
                    skip = True
                elif raw.startswith('<!-- /i18n:'):
                    skip = False
                out.append(raw)
                continue
            if kind == 'data' or (kind in ('start', 'end') and tag in INLINE):
                buf.append(raw if kind != 'start' or skip else self._attrs(raw, tag, tr))
                continue
            flush()
            if kind == 'start':
                if tag in RAW_TEXT:
                    raw_tag, raw_attrs = tag, raw
                out.append(raw if skip else self._attrs(raw, tag, tr))
            else:
                out.append(raw)
        flush()
        return ''.join(out)

    @staticmethod
    def _attrs(raw: str, tag: str, tr) -> str:
        for name in TEXT_ATTRS:
            v = _attr(raw, name)
            if v and _LETTER.search(v):
                raw = _set_attr(raw, name, tr(norm(v)))
        if tag == 'meta':
            if _attr(raw, 'name') in META_NAMES or _attr(raw, 'property') in META_PROPS:
                v = _attr(raw, 'content')
                if v:
                    raw = _set_attr(raw, 'content', tr(norm(v)))
        return raw

    @staticmethod
    def _script(raw: str, start_tag: str, tr) -> str:
        kind = _attr(start_tag, 'type') or ''
        if kind == 'application/ld+json':
            data = json.loads(raw)
            for f in JSONLD_FIELDS:
                if isinstance(data.get(f), str):
                    data[f] = tr(norm(data[f]))
            return '\n' + json.dumps(data, ensure_ascii=False, indent=2) + '\n'
        if kind == 'application/json' and _attr(start_tag, 'id') == 'i18n':
            data = json.loads(raw)
            return json.dumps({k: tr(norm(v)) for k, v in data.items()}, ensure_ascii=False)
        return raw

    def keys(self) -> List[str]:
        found: List[str] = []

        def collect(s):
            found.append(s)
            return s
        self._walk(collect)
        return found

    def render(self, catalog: Dict[str, str], missing: List[str]):
        def tr(s):
            t = catalog.get(s)
            if not t:
                missing.append(s)
                return s
            return t
        return self._walk(tr)


# ── per-language adjustments ─────────────────────────────────────────────

def page_url(folder: Optional[str], page: str) -> str:
    path = '' if page == 'index.html' else page
    return SITE + (f'{folder}/' if folder else '') + path


def alternates_block(page: str) -> str:
    lines = [f'<link rel="alternate" hreflang="en" href="{page_url(None, page)}">']
    lines += [f'<link rel="alternate" hreflang="{hl}" href="{page_url(folder, page)}">'
              for _code, folder, hl, _loc, _name in LANGS]
    lines.append(f'<link rel="alternate" hreflang="x-default" href="{page_url(None, page)}">')
    return '<!-- i18n:alternates -->\n' + '\n'.join(lines) + '\n<!-- /i18n:alternates -->\n'


def langs_block(page: str, current: Optional[str]) -> str:
    """Footer language switcher; links are relative to the current folder."""
    target = '' if page == 'index.html' else page
    up = '../' if current else ''
    links = []
    for folder, name, hl in [(None, 'English', 'en')] + [(f, n, h) for _c, f, h, _l, n in LANGS]:
        href = up + (f'{folder}/' if folder else '') + target or './'
        if href == '':
            href = './'
        mark = ' aria-current="true"' if folder == current else ''
        links.append(f'<a href="{href}" hreflang="{hl}" lang="{hl}"{mark}>{name}</a>')
    return ('<!-- i18n:langs --><span class="langs">' + ' '.join(links)
            + '</span><!-- /i18n:langs -->')


def with_generated_blocks(text: str, page: str, current: Optional[str]) -> str:
    """Insert or refresh the alternates in <head> and the footer switcher."""
    alt = alternates_block(page)
    if '<!-- i18n:alternates -->' in text:
        text = re.sub(r'<!-- i18n:alternates -->.*?<!-- /i18n:alternates -->\n', lambda m: alt,
                      text, flags=re.S)
    else:
        text = text.replace('</head>', alt + '</head>', 1)
    langs = langs_block(page, current)
    if '<!-- i18n:langs -->' in text:
        text = re.sub(r'<!-- i18n:langs -->.*?<!-- /i18n:langs -->', lambda m: langs, text,
                      flags=re.S)
    else:
        text = re.sub(r'(<footer>\s*<div class="wrap">.*?)(\n\s*</div>\s*</footer>)',
                      lambda m: m.group(1) + '\n    ' + langs + m.group(2), text, count=1,
                      flags=re.S)
    return text


def _local_url(url: str) -> str:
    """A link written for docs/, as seen from docs/<folder>/."""
    if not url or re.match(r'^([a-z]+:|//|/|#|\.\./)', url):
        return url
    path = re.split(r'[#?]', url, maxsplit=1)[0]
    if path in ('', './') or path in PAGES:   # translated pages: stay in the language folder
        return url
    return '../' + url


def localize(text: str, page: str, folder: str, hreflang: str, og_locale: str) -> str:
    text = re.sub(r'<html lang="[^"]*">', f'<html lang="{hreflang}">', text, count=1)
    if hreflang in ('ja', 'zh-CN'):
        # site.css switches these pages to CJK system fonts; the Latin web
        # fonts would be downloaded for nothing.
        text = re.sub(r'<link rel="preload" href="fonts/[^"]*" as="font"[^>]*>\n', '', text)
    url = page_url(folder, page)
    text = re.sub(r'(<link rel="canonical" href=")[^"]*(")', lambda m: m.group(1) + url + m.group(2), text)
    text = re.sub(r'(<meta property="og:url" content=")[^"]*(")', lambda m: m.group(1) + url + m.group(2), text)
    if 'og:locale' not in text:
        text = text.replace('<meta property="og:type"', f'<meta property="og:locale" content="{og_locale}">\n'
                            '<meta property="og:type"', 1)
    text = text.replace('"@type": "TechArticle",', f'"@type": "TechArticle",\n  "inLanguage": "{hreflang}",', 1)

    def fix(m):
        return f'{m.group(1)}="{_local_url(m.group(2))}"'

    def fix_set(m):
        parts = []
        for item in m.group(2).split(','):
            bits = item.strip().split()
            if bits:
                bits[0] = _local_url(bits[0])
                parts.append(' '.join(bits))
        return f'{m.group(1)}="{", ".join(parts)}"'
    text = _URL_ATTR.sub(fix, text)
    text = _SRCSET_ATTR.sub(fix_set, text)
    return text


# ── sitemap ──────────────────────────────────────────────────────────────

def sitemap(lastmods: Dict[str, str]) -> str:
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" '
           'xmlns:xhtml="http://www.w3.org/1999/xhtml">']
    for page in PAGES + ENGLISH_ONLY:
        variants = [(None, 'en')] + ([(f, h) for _c, f, h, _l, _n in LANGS] if page in PAGES else [])
        for folder, _hl in variants:
            out.append('  <url>')
            out.append(f'    <loc>{page_url(folder, page)}</loc>')
            if page in lastmods:
                out.append(f'    <lastmod>{lastmods[page]}</lastmod>')
            if page in PAGES:
                for f2, h2 in variants:
                    out.append(f'    <xhtml:link rel="alternate" hreflang="{h2}" href="{page_url(f2, page)}"/>')
                out.append(f'    <xhtml:link rel="alternate" hreflang="x-default" href="{page_url(None, page)}"/>')
            out.append('  </url>')
    out.append('</urlset>')
    return '\n'.join(out) + '\n'


def _lastmods() -> Dict[str, str]:
    text = (DOCS / 'sitemap.xml').read_text(encoding='utf-8')
    found = {}
    for loc, mod in re.findall(r'<loc>([^<]+)</loc>\s*<lastmod>([^<]+)</lastmod>', text):
        if loc.startswith(SITE) and loc.count('/') == 3:
            found[loc[len(SITE):] or 'index.html'] = mod
    return found


# ── main ─────────────────────────────────────────────────────────────────

def load_catalog(code: str) -> Dict[str, str]:
    path = CATALOGS / f'{code}.json'
    if not path.exists():
        return {}
    return {k: v for k, v in json.loads(path.read_text(encoding='utf-8')).items()
            if not k.startswith('_') and isinstance(v, str) and v}


def outputs() -> Tuple[Dict[Path, str], Dict[str, int]]:
    files: Dict[Path, str] = {}
    missing_counts: Dict[str, int] = {}
    for page in PAGES:
        src = Page(page)
        files[DOCS / page] = with_generated_blocks(src.text, page, None)
    for code, folder, hreflang, og_locale, _name in LANGS:
        catalog = load_catalog(code)
        missing: List[str] = []
        for page in PAGES:
            # Translate from the English page with fresh generated blocks, so
            # the source's own switcher/alternates never reach the catalog.
            src = Page(page)
            src.text = files[DOCS / page]
            src.toks = _Tokens(src.text).toks
            body = src.render(catalog, missing)
            body = with_generated_blocks(body, page, folder)
            files[DOCS / folder / page] = localize(body, page, folder, hreflang, og_locale)
        missing_counts[code] = len(set(missing))
    files[DOCS / 'sitemap.xml'] = sitemap(_lastmods())
    return files, missing_counts


def all_keys() -> List[str]:
    seen, keys = set(), []
    for page in PAGES:
        p = Page(page)
        p.text = with_generated_blocks(p.text, page, None)
        p.toks = _Tokens(p.text).toks
        for k in p.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--check', action='store_true', help='exit 1 if any generated file is stale')
    ap.add_argument('--missing', metavar='LANG', help='print untranslated segments for LANG as JSON')
    args = ap.parse_args(argv)
    if args.missing:
        catalog = load_catalog(args.missing)
        print(json.dumps({k: '' for k in all_keys() if k not in catalog}, ensure_ascii=False, indent=1))
        return 0
    files, missing = outputs()
    stale = [p for p, text in files.items()
             if not p.exists() or p.read_text(encoding='utf-8') != text]
    if args.check:
        for p in stale:
            print(f'stale: {p.relative_to(ROOT)}')
        return 1 if stale else 0
    for p in stale:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(files[p], encoding='utf-8', newline='\n')
    for code, n in missing.items():
        print(f'{code:6} {n} untranslated segments')
    print(f'{len(stale)} files written')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
