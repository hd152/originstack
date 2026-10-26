""""Support OriginStack": one Ko-fi address in the app, GitHub's Sponsor button, the README
and the website."""
from pathlib import Path

from src.desktop_control import SUPPORT_URL

ROOT = Path(__file__).resolve().parent.parent


def test_support_url_is_the_ko_fi_page_named_in_funding_yml():
    funding = (ROOT / '.github' / 'FUNDING.yml').read_text(encoding='utf-8')
    name = next(line.split(':', 1)[1].strip() for line in funding.splitlines()
                if line.startswith('ko_fi:'))
    assert SUPPORT_URL == f'https://ko-fi.com/{name}'


def test_readme_and_every_home_page_link_to_it():
    assert SUPPORT_URL in (ROOT / 'README.md').read_text(encoding='utf-8')
    pages = [ROOT / 'docs' / 'index.html'] + sorted((ROOT / 'docs').glob('*/index.html'))
    assert len(pages) == 9
    for page in pages:
        # the tip line under the download notes and the footer link
        assert page.read_text(encoding='utf-8').count(f'href="{SUPPORT_URL}"') == 2, page
