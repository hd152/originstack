"""Setup form helpers: field descriptions, changed markers, search, and the
name lists desktop_app.py keys on (a wrong dest there fails silently -- the
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
    from src import desktop_app
    dests = _all_dests()
    for name, keys in (('_FIELD_SUMMARIES', _FIELD_SUMMARIES),
                       ('_COMMON_DESTS', desktop_app._COMMON_DESTS),
                       ('_FIELD_LABELS', desktop_app._FIELD_LABELS),
                       ('_EXPERT_DESTS', desktop_app._EXPERT_DESTS),
                       ('_NO_CHANGE_MARK', desktop_app._NO_CHANGE_MARK)):
        missing = set(keys) - dests
        assert not missing, f'{name} names unknown dests: {sorted(missing)}'


@pytest.fixture(scope='module')
def tk_root():
    # One root for the module: creating and destroying a Tk root per test
    # intermittently crashed Tk inside tk.Tk() (Windows, Python 3.14).
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


@pytest.fixture
def form(tk_root):
    from src import desktop_app
    f = desktop_app.SetupForm(tk_root)
    f.pack()
    tk_root.update_idletasks()
    yield f
    f.destroy()


def _visible(form, dest):
    return all(w.winfo_manager() == 'grid' for w in form._fields[dest]['widgets'])


def test_deconvolution_is_on_the_main_form(form):
    assert form._fields['deconvolve_mode']['group'] is None


def test_shared_dest_gets_the_widget_build_argv_uses(form):
    # --no-cosmic-ray-rejection and --cosmic-ray-rejection share a dest; the
    # checkbox must be the store_true one, or ticking it emits nothing.
    from src.desktop_control import build_argv_from_form
    form.vars['cosmic_ray_rejection'].set(True)
    assert build_argv_from_form(form.read_form()) == ['--cosmic-ray-rejection']


def test_changed_marker_count_and_reset(form):
    info = form._fields['banding_removal']
    assert info['dot'].cget('text') == ''
    form.vars['banding_removal'].set(True)
    assert info['dot'].cget('text') == '●'
    assert info['reset'].cget('text') == 'reset'
    assert '1 changed' in form._toggle_btn.cget('text')
    assert form.read_form() == {'banding_removal': True}
    form._reset('banding_removal')
    assert info['dot'].cget('text') == ''
    assert 'changed' not in form._toggle_btn.cget('text')
    assert form.read_form() == {}


def test_path_fields_are_not_marked(form):
    form.vars['directory'].set('C:/nowhere')
    assert form._fields['directory']['dot'].cget('text') == ''


def test_search_filters_across_groups_and_includes_expert(form):
    assert not _visible(form, 'fix_atmospheric_dispersion')  # expert, hidden by default
    form.search_var.set('atmospheric dispersion')
    assert form._advanced_shown
    assert _visible(form, 'fix_atmospheric_dispersion')
    assert not _visible(form, 'banding_removal')
    shown = [g for g, lbl in form._nav_labels.items() if lbl.winfo_manager()]
    assert shown and all(form._nav_labels[g].cget('text').endswith(')') for g in shown)

    form.search_var.set('zzqx no such setting')
    assert form._current_group is None
    assert form._no_match.winfo_manager() == 'pack'

    form.search_var.set('')
    assert not _visible(form, 'fix_atmospheric_dispersion')
    assert _visible(form, 'banding_removal')
    assert form._no_match.winfo_manager() == ''
