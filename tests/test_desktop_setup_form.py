"""Setup form helpers: field descriptions, changed markers, search, and the
name lists the desktop windows key on (src/desktop_common.py) (a wrong dest there fails silently -- the
main form's 'deconvolve' entry matched nothing, so Deconvolution was missing
from it)."""
import pytest

from src.desktop_control import _FIELD_SUMMARIES, get_form_schema, summarize_help


def _all_dests():
    return {f['dest'] for fields in get_form_schema().values() for f in fields}


class TestSummarizeHelp:
    def test_first_sentence(self):
        assert summarize_help('Remove stars. Writes a sidecar.', 'bool_true') == 'Remove stars.'

    def test_does_not_stop_at_eg_or_inside_parentheses(self):
        text = 'Upscale (e.g. 2. or 3.) the output, e.g. for drizzle. More text.'
        assert summarize_help(text, 'number') == \
            'Upscale (e.g. 2. or 3.) the output, e.g. for drizzle.'

    def test_short_heading_colon_is_not_the_end(self):
        assert summarize_help('Bayer-aware drizzle: recombine samples. Slow.', 'bool_true') == \
            'Bayer-aware drizzle: recombine samples.'

    def test_drops_dangling_default(self):
        assert summarize_help('Master combine method (default: median). Other.', 'select') == \
            'Master combine method.'

    def test_negated_checkbox_reads_as_enabled_state(self):
        assert summarize_help('Disable chroma noise reduction', 'bool_false') == \
            'Chroma noise reduction.'
        assert summarize_help('Do not refine the WCS; slower.', 'bool_false') == 'Refine the WCS.'

    def test_other_negated_wording_is_dropped_not_shown_backwards(self):
        assert summarize_help('Re-measure the gain on every frame.', 'bool_false') == ''

    def test_length_cap(self):
        out = summarize_help('word ' * 100, 'text', limit=40)
        assert len(out) <= 41 and out.endswith('…')

    def test_empty(self):
        assert summarize_help('', 'text') == ''


def test_every_field_has_a_summary_key():
    for fields in get_form_schema().values():
        for f in fields:
            assert 'summary' in f


def test_name_lists_refer_to_real_dests():
    from src import desktop_common
    dests = _all_dests()
    for name, keys in (('_FIELD_SUMMARIES', _FIELD_SUMMARIES),
                       ('_COMMON_DESTS', desktop_common._COMMON_DESTS),
                       ('_FIELD_LABELS', desktop_common._FIELD_LABELS),
                       ('_EXPERT_DESTS', desktop_common._EXPERT_DESTS),
                       ('_NO_CHANGE_MARK', desktop_common._NO_CHANGE_MARK)):
        missing = set(keys) - dests
        assert not missing, f'{name} names unknown dests: {sorted(missing)}'


def test_example_images_exist():
    from src.desktop_common import _EXAMPLES_DIR, _TARGET_CARDS
    for card in _TARGET_CARDS:
        if card[5]:
            assert (_EXAMPLES_DIR / card[5]).is_file()


def test_hidden_aliases_do_not_replace_the_visible_option():
    from src.desktop_control import build_argv_from_form
    assert build_argv_from_form({'comet_mode': True}) == ['--comet-mode']
    argv = build_argv_from_form({'full_field': 0.7})
    assert argv == ['--full-field', '0.7']
    from src.cli import parse_args
    assert parse_args(['-d', '.'] + argv).full_field == 0.7
