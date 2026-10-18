"""Desktop app "Report a problem": the pre-filled GitHub issue URL."""
import os
from urllib.parse import parse_qs, urlparse

from src.desktop_control import _REPORT_URL_MAX, ISSUES_URL, problem_report_url
from src.utils import read_version


def _fields(url):
    assert url.startswith(ISSUES_URL + '?')
    q = parse_qs(urlparse(url).query)
    return q['title'][0], q['body'][0]


def test_system_details_and_review_note():
    title, body = _fields(problem_report_url())
    assert title == f'[{read_version()}] '
    assert f'OriginStack {read_version()}' in body
    assert 'Python' in body
    assert 'this issue is public' in body
    assert '**Log**' not in body                 # no log, no empty log section


def test_error_goes_in_title_and_body():
    title, body = _fields(problem_report_url('', 'FrameStoreSpaceError: temp disk full\nmore'))
    assert title == f'[{read_version()}] FrameStoreSpaceError: temp disk full'
    assert '**Error**' in body and 'more' in body


def test_log_tail_keeps_the_newest_lines_within_the_url_limit():
    log = '\n'.join(f'line {i:05d} ' + 'x' * 60 for i in range(2000))
    url = problem_report_url(log)
    _title, body = _fields(url)
    assert len(url) <= _REPORT_URL_MAX
    assert 'line 01999' in body                  # the end of the log survives
    assert 'line 00000' not in body
    assert '**Log** (last ' in body


def test_short_log_is_kept_whole_and_blank_lines_dropped():
    _title, body = _fields(problem_report_url('Discovering frames...\n\n  Found 133 FITS files\n'))
    assert '**Log** (last 2 lines)' in body
    assert 'Found 133 FITS files' in body


def test_progress_runs_keep_their_last_line():
    log = '\n'.join(['Stacking'] + [f'    Combining: {r}/1964 rows' for r in (323, 969, 1964)]
                    + ['  Frame 1/3 ok', '  Frame 2/3 rejected', 'Done'])
    _title, body = _fields(problem_report_url(log))
    assert 'Combining: 1964/1964' in body and 'Combining: 969/' not in body
    assert '**Log** (last 4 lines)' in body
    assert 'Frame 2/3 rejected' in body and 'Frame 1/3' not in body


def test_home_folder_is_redacted():
    home = os.path.expanduser('~')
    log = f'Input:  {home}{os.sep}astro{os.sep}M101\nOutput: {home.replace(os.sep, "/")}/out.fits'
    _title, body = _fields(problem_report_url(log, f'cannot write {home}{os.sep}out.fits'))
    assert home not in body and home.replace(os.sep, '/') not in body
    assert f'~{os.sep}astro{os.sep}M101' in body and '~/out.fits' in body
