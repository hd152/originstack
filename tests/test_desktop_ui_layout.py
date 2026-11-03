"""Desktop-app layout polish: the one-line preflight and the Compare box.

Run against the real methods on stand-in objects holding real widgets: a whole ``App`` per module
left Tk in a state where the next module's root intermittently failed to open."""
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace

import src.desktop_app as da

App_COLS = da.App._FRAME_COLS


def test_preflight_collapses_to_one_line_when_all_pass_and_expands_on_a_problem(tk_root):
    frame = ttk.Frame(tk_root)
    me = SimpleNamespace(
        preflight_vars={k: tk.StringVar(value='✓ ' + k) for k in ('a', 'b', 'c')},
        _preflight_ok_var=tk.StringVar(), form=SimpleNamespace(_data_summary='35 lights'))
    me._preflight_ok = ttk.Label(frame, textvariable=me._preflight_ok_var)
    me._preflight_grid = ttk.Frame(frame)
    da.App._show_preflight_summary(me)
    assert me._preflight_ok.winfo_manager() == 'pack'
    assert me._preflight_grid.winfo_manager() == ''
    assert '35 lights' in me._preflight_ok_var.get()
    me.preflight_vars['b'].set('⚠ Low disk space')
    da.App._show_preflight_summary(me)
    assert me._preflight_grid.winfo_manager() == 'pack'
    assert me._preflight_ok.winfo_manager() == ''


def test_compare_box_follows_the_compare_checkbox_even_when_set_from_code(tk_root):
    me = SimpleNamespace(compare_var=tk.BooleanVar(tk_root, value=False))
    me.compare_combo = ttk.Combobox(tk_root, state='disabled')
    me.compare_var.trace_add('write', lambda *_a: da.App._sync_compare_box(me))
    me.compare_var.set(True)
    assert str(me.compare_combo.cget('state')) == 'readonly'
    me.compare_var.set(False)           # how the code switches compare off
    assert str(me.compare_combo.cget('state')) == 'disabled'


def test_frame_rows_sort_by_the_clicked_column_and_cycle_back_to_arrival_order(tk_root):
    rows = [('b.fits', '60.0', '1.5', 40, '4.0', True), ('a.fits', '70.0', '1.6', 50, '5.5', False),
            ('c.fits', '--', '1.4', 45, '4.5', True)]
    tree = ttk.Treeview(tk_root, columns=App_COLS, show='headings')
    me = SimpleNamespace(_frame_sort=None, _FRAME_COLS=App_COLS, _frame_headings={c: c for c in App_COLS},
                         frames_tree=tree, _frame_rows_shown=[])
    me._sorted_frame_rows = lambda r: da.App._sorted_frame_rows(me, r)
    assert me._sorted_frame_rows(rows) == rows
    for c in App_COLS:
        tree.heading(c, text=c)
    da.App._sort_frames(me, 'score')
    assert [r[0] for r in me._sorted_frame_rows(rows)] == ['a.fits', 'b.fits', 'c.fits']   # high first, '--' last
    assert tree.heading('score')['text'].endswith('▼')
    da.App._sort_frames(me, 'score')
    assert [r[0] for r in me._sorted_frame_rows(rows)] == ['c.fits', 'b.fits', 'a.fits']
    da.App._sort_frames(me, 'score')
    assert me._frame_sort is None and me._sorted_frame_rows(rows) == rows
    da.App._sort_frames(me, 'frame')
    assert [r[0] for r in me._sorted_frame_rows(rows)] == ['a.fits', 'b.fits', 'c.fits']
