"""Build the website's command-line reference from the CLI's own parsers.

``docs/command-line.html`` lists every option of ``originstack.py`` (from
``cli.build_parser()``) and of the ``combine`` subcommand (from
``channel_combine.build_combine_parser()``): flags, value, choices, default and
the help text, grouped as ``--help`` groups them. Hidden options (removed flags
kept for old command lines) are left out. Because the page is generated, changing
an option's help in the code changes the page; ``tests/test_cli_reference.py``
fails until it is rebuilt.

    python tools/build_cli_reference.py           # write the page
    python tools/build_cli_reference.py --check   # exit 1 if it is out of date
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / 'docs' / 'command-line.html'
sys.path.insert(0, str(ROOT))


def _visible_actions(group) -> List[argparse.Action]:
    return [a for a in group._group_actions
            if a.help != argparse.SUPPRESS and not isinstance(a, argparse._HelpAction)]


def _anchor(prefix: str, action: argparse.Action) -> str:
    longest = max(action.option_strings, key=len) if action.option_strings else action.dest
    return prefix + re.sub(r'[^a-z0-9]+', '-', longest.lower()).strip('-')


def _value(action: argparse.Action) -> str:
    """What the option takes: nothing for a switch, else its metavar or type."""
    if action.nargs == 0:
        return ''
    if action.metavar:
        mv = action.metavar if isinstance(action.metavar, str) else ' '.join(action.metavar)
    elif action.choices:
        mv = ''
    else:
        mv = action.dest.upper()
    if action.nargs in ('*', '+'):
        mv = f'{mv} ...'
    elif action.nargs == '?':
        mv = f'[{mv}]'
    return mv


def _default(action: argparse.Action) -> str:
    d = action.default
    if action.nargs == 0 or d is None or d is argparse.SUPPRESS or d == '' or d == []:
        return ''
    return str(d)


def _help(parser: argparse.ArgumentParser, action: argparse.Action) -> str:
    text = parser._get_formatter()._expand_help(action) if action.help else ''
    return ' '.join(text.split())


def _entry(parser, action, prefix: str) -> str:
    flags = ', '.join(f'<code>{html.escape(o)}</code>' for o in action.option_strings)
    value = _value(action)
    if value:
        flags += f' <var>{html.escape(value)}</var>'
    meta = []
    if action.choices:
        meta.append('Choices: ' + ', '.join(f'<code>{html.escape(str(c))}</code>' for c in action.choices))
    default = _default(action)
    if default:
        meta.append(f'Default: <code>{html.escape(default)}</code>')
    if action.required:
        meta.append('Required')
    meta_html = f'\n          <p class="meta">{" &middot; ".join(meta)}</p>' if meta else ''
    return (f'      <div id="{_anchor(prefix, action)}"><dt>{flags}</dt>\n'
            f'        <dd><p>{html.escape(_help(parser, action))}</p>{meta_html}</dd></div>')


def _sections(parser, prefix: str, skip_titles: Tuple[str, ...] = ()) -> List[Tuple[str, str, str]]:
    out = []
    for group in parser._action_groups:
        actions = _visible_actions(group)
        if not actions or group.title in skip_titles:
            continue
        title = 'Other options' if group.title in ('options', 'optional arguments') else group.title
        gid = prefix + re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
        body = '\n'.join(_entry(parser, a, prefix) for a in actions)
        out.append((gid, title, body))
    return out


TEMPLATE = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OriginStack command-line reference</title>
<meta name="description" content="Every option of the OriginStack command line, with its choices, default and description, generated from the program itself (version {version}).">
<meta name="keywords" content="OriginStack command line, originstack.py options, astrophotography stacking command line, stack-method, stretch, drizzle, merge, from-stack">
<meta name="robots" content="index, follow">
<meta name="theme-color" content="#0b1220">
<link rel="canonical" href="https://originstack.site/command-line.html">
<meta property="og:type" content="website">
<meta property="og:site_name" content="OriginStack">
<meta property="og:url" content="https://originstack.site/command-line.html">
<meta property="og:title" content="OriginStack command-line reference">
<meta property="og:description" content="Every command-line option, with choices, defaults and descriptions.">
<link rel="icon" href="img/favicon.png">
<link rel="preload" href="fonts/bricolage-grotesque-normal-latin.woff2" as="font" type="font/woff2" crossorigin>
<link rel="preload" href="fonts/source-serif-4-normal-latin.woff2" as="font" type="font/woff2" crossorigin>
<link rel="stylesheet" href="site.css">

<script data-goatcounter="https://hd152.goatcounter.com/count" async src="https://gc.zgo.at/count.js"></script>
</head>
<body>
<!-- Generated by tools/build_cli_reference.py from the program's own option definitions. Do not edit by hand. -->

<header class="hero slim">
  <div class="wrap">
    <div class="bar">
      <a href="./"><picture><source type="image/webp" srcset="img/logo-dark-200.webp 200w, img/logo-dark-400.webp 400w" sizes="188px"><img src="img/logo-dark.png" alt="OriginStack" width="920" height="256"></picture></a>
      <nav aria-label="Pages">
        <a href="./">Home</a>
        <a href="docs.html">Documentation</a>
        <a href="stack-celestron-origin.html">Guide</a>
        <a href="./#get">Install</a>
      </nav>
    </div>
    <h1>Command-line reference</h1>
    <p class="lede">Every option of <code>originstack.py</code>, version {version}, generated from the program's own definitions, so it matches what <code>--help</code> prints. The desktop app shows the same options under <em>Additional options</em>, with shorter labels.</p>
    <p class="toc">
{toc}
    </p>
  </div>
</header>

<main>
<section id="usage">
  <div class="wrap">
    <h2>Usage</h2>
    <pre>python originstack.py -d LIGHTS_FOLDER -o OUTPUT.fits [options]
python originstack.py --from-stack STACK.fits -o NEW.fits [options]
python originstack.py combine --mode sho --ha Ha.fits --oiii OIII.fits --sii SII.fits -o sho.fits</pre>
    <p class="plain measure">Most runs need only <code>-d</code>: the target is detected and the settings tuned for it (<code>--auto</code>, on by default). An option you give always wins over the tuning. Every run saves its effective settings as <code>OUTPUT_config.toml</code>; <code>--config</code> applies a saved file again.</p>
    <p class="filter"><label for="opt-filter">Filter options</label> <input id="opt-filter" type="search" placeholder="e.g. drizzle, stretch, offline" autocomplete="off"> <span id="opt-count" aria-live="polite"></span></p>
  </div>
</section>
{sections}
</main>

<footer>
  <div class="wrap">
    <span>OriginStack is free software under the GNU GPL v3.</span>
    <span><a href="./">Home</a> &nbsp; <a href="docs.html">Documentation</a> &nbsp; <a href="stack-celestron-origin.html">How to stack Origin images</a> &nbsp; <a href="https://github.com/hd152/originstack">Source</a> &nbsp; <a href="https://github.com/hd152/originstack/issues">Report a problem</a> &nbsp; <a href="privacy.html">Privacy</a></span>
  </div>
</footer>

<script>
(function () {{
  var input = document.getElementById('opt-filter'), count = document.getElementById('opt-count');
  var items = Array.prototype.slice.call(document.querySelectorAll('.opts > div'));
  var groups = Array.prototype.slice.call(document.querySelectorAll('section.optgroup'));
  input.addEventListener('input', function () {{
    var q = input.value.trim().toLowerCase(), shown = 0;
    items.forEach(function (el) {{
      var hit = !q || el.textContent.toLowerCase().indexOf(q) >= 0;
      el.hidden = !hit; if (hit) shown++;
    }});
    groups.forEach(function (g) {{ g.hidden = !g.querySelector('.opts > div:not([hidden])'); }});
    count.textContent = q ? shown + ' of ' + items.length : '';
  }});
}})();
</script>
</body>
</html>
'''


def render() -> str:
    from src.channel_combine import build_combine_parser
    from src.cli import build_parser
    from src.utils import read_version
    main_sections = _sections(build_parser(), 'opt-')
    combine_sections = _sections(build_combine_parser(), 'combine-')
    toc = [f'      <a href="#{gid}">{html.escape(title)}</a>' for gid, title, _ in main_sections]
    toc.append('      <a href="#combine">combine</a>')
    blocks = []
    for gid, title, body in main_sections:
        blocks.append(f'<section id="{gid}" class="optgroup">\n  <div class="wrap">\n    <h2>{html.escape(title)}</h2>\n'
                      f'    <dl class="opts">\n{body}\n    </dl>\n  </div>\n</section>')
    combine_body = []
    for gid, title, body in combine_sections:
        heading = 'Inputs and output' if title == 'Other options' else title
        combine_body.append(f'    <h3 id="{gid}">{html.escape(heading)}</h3>\n    <dl class="opts">\n{body}\n    </dl>')
    blocks.append('<section id="combine" class="optgroup">\n  <div class="wrap">\n    <h2>The <code>combine</code> subcommand</h2>\n'
                  '    <p class="intro measure">Combines separately stacked channel images (LRGB or narrowband) into one colour image: '
                  '<code>python originstack.py combine [options]</code>.</p>\n'
                  + '\n'.join(combine_body) + '\n  </div>\n</section>')
    return TEMPLATE.format(version=html.escape(read_version()), toc='\n'.join(toc),
                           sections='\n'.join(blocks))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--check', action='store_true', help='exit 1 if the page is out of date')
    a = ap.parse_args()
    text = render()
    current = PAGE.read_text(encoding='utf-8') if PAGE.exists() else None
    if a.check:
        if current != text:
            print(f'{PAGE.relative_to(ROOT)} is out of date: run python tools/build_cli_reference.py')
            return 1
        return 0
    if current != text:
        with open(PAGE, 'w', encoding='utf-8', newline='\n') as f:
            f.write(text)
        print(f'wrote {PAGE.relative_to(ROOT)}')
    else:
        print(f'{PAGE.relative_to(ROOT)} is up to date')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
