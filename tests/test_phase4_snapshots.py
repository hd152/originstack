"""Step snapshots (postprocess.StepSnapshots): a Phase 4 re-run that resumes at a toggled
step is bit-identical to a full run with the same skip list."""
import copy

import numpy as np
import pytest

import src.postprocess as pp
from src import cli
from src.models import ProcessingStats


def _image(H=192, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :H]
    img = np.full((H, H, 3), 500.0) + 0.4 * xx[..., None] + 0.2 * yy[..., None]
    for (cy, cx), amp in (((40, 50), 5000.0), ((120, 60), 9000.0), ((90, 150), 3000.0)):
        prof = amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.0 ** 2))
        img += prof[..., None] * np.array([1.0, 0.8, 0.6])
    return (img + rng.normal(0, 12, img.shape)).astype(np.float32)


def _args(skip=()):
    args = cli.parse_args(['-d', '.'])
    args._diagnostic_dir = None
    args.denoise_bilateral = True
    args.denoise_curvelet = True
    args.scnr = True
    args.skip_step = list(skip)
    return args


def _run(img, args):
    return pp.postprocess_stack(img.copy(), args, [], ProcessingStats())


def test_resume_matches_a_full_run(monkeypatch):
    img = _image()
    args = _args()
    snaps = pp.StepSnapshots()
    args._pp_snaps = snaps
    base = _run(img, args)
    assert np.array_equal(base, _run(img, _args()))          # recording changes nothing
    assert {'start', 'bilateral', 'wavelet', 'scnr', 'star_reduce', 'local_contrast',
            'sky_neutralize'} <= set(snaps.snaps)
    early = []
    real = pp._postprocess_early
    monkeypatch.setattr(pp, '_postprocess_early', lambda *a, **k: early.append(1) or real(*a, **k))

    for step, resume in (('bilateral', 'bilateral'), ('wavelet', 'wavelet'), ('scnr', 'scnr'),
                         ('local_contrast', 'local_contrast'), ('star_reduce', 'star_reduce')):
        args.skip_step = [step]
        snaps.resume = resume
        got = _run(img, args)
        assert early == [], 'resuming redid the early block'
        assert np.array_equal(got, _run(img, _args([step]))), step
        early.clear()
        # and back on again
        args.skip_step = []
        snaps.resume = resume
        assert np.array_equal(_run(img, args), base), step


def test_resume_from_start_and_missing_snapshot():
    img = _image()
    args = _args()
    snaps = pp.StepSnapshots()
    args._pp_snaps = snaps
    base = _run(img, args)
    args.skip_step = ['sky_neutralize']
    snaps.resume = 'start'
    assert np.array_equal(_run(img, args), _run(img, _args(['sky_neutralize'])))
    # a step whose snapshot was not kept falls back to the nearest earlier one
    del snaps.snaps['scnr']
    assert snaps.start_at('scnr') == 'wavelet'
    args.skip_step = ['scnr']
    snaps.resume = 'scnr'
    assert np.array_equal(_run(img, args), _run(img, _args(['scnr'])))
    assert base is not None


def test_budget_keeps_start_only():
    snaps = pp.StepSnapshots(budget_bytes=1)
    img = np.zeros((8, 8, 3), np.float32)
    snaps.record('start', img, None, None, force=True)
    snaps.record('bilateral', img, None, None)
    assert set(snaps.snaps) == {'start'}
    assert snaps.start_at('local_contrast') == 'start'
    assert snaps.start_at('start') == 'start'


def test_loaded_snapshot_is_a_copy():
    snaps = pp.StepSnapshots()
    img = np.ones((4, 4, 3), np.float32)
    snaps.record('start', img, None, None, force=True)
    out, _, _ = snaps.load('start')
    out[:] = 5
    assert snaps.load('start')[0][0, 0, 0] == 1.0
    img[:] = 7
    assert snaps.load('start')[0][0, 0, 0] == 1.0


def _ui_with_run(tmp_path):
    from src.io_fits import save_preview_rgb
    from src.ui_events import UIEvents
    img = _image()
    args = _args()
    snaps = pp.StepSnapshots()
    args._pp_snaps = snaps
    result = _run(img, args)
    import copy as _copy
    args_pp = _copy.copy(args)
    del args._pp_snaps
    ui = UIEvents()
    ui.attach()
    path = str(tmp_path / "stack.jpg")
    save_preview_rgb(result, path, stretch='ghs')
    ui.remember_final_preview(result, path, args_pp)
    ui.remember_postprocess(img.copy(), args_pp, [], ProcessingStats(), snaps)
    return ui, img, path


def test_ui_rerun_toggles_steps_and_matches_a_full_run(tmp_path):
    ui, img, path = _ui_with_run(tmp_path)
    steps = {s['id']: s['on'] for s in ui.postprocess_steps()}
    assert steps['local_contrast'] and steps['bilateral'] and 'nn' not in steps
    assert ui.snapshot()['postprocess'] is True

    assert ui.rerun_postprocess([]) is None                       # nothing changed
    data = ui.rerun_postprocess(['local_contrast', 'bilateral'])
    assert data.startswith(b'\xff\xd8')
    ref = _run(img, _args(['local_contrast', 'bilateral']))
    assert np.array_equal(ui._final_src['rgb'], ref)
    assert {s['id']: s['on'] for s in ui.postprocess_steps()}['local_contrast'] is False
    assert ui.named_jpeg('final') == data

    # an early step: redoes the background block, still matches
    ui.rerun_postprocess(['local_contrast', 'bilateral', 'background'])
    assert np.array_equal(ui._final_src['rgb'],
                          _run(img, _args(['local_contrast', 'bilateral', 'background'])))
    # and everything back on
    ui.rerun_postprocess([])
    assert np.array_equal(ui._final_src['rgb'], _run(img, _args()))


def test_ui_rerun_without_a_run_and_after_a_new_run(tmp_path):
    import pytest
    ui, _, _ = _ui_with_run(tmp_path)
    ui.run_started()
    assert ui.postprocess_steps() is None
    with pytest.raises(RuntimeError):
        ui.rerun_postprocess(['scnr'])


def test_desktop_panel_lists_steps_and_reruns_on_toggle(tmp_path, monkeypatch):
    """The Post-processing panel: one box per step, a change re-runs Phase 4 off the GUI thread."""
    import pytest
    pytest.importorskip('tkinter')
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('no display available for a real Tk root')
    try:
        from src import desktop_app
        monkeypatch.setattr('src.utils.should_check_for_update', lambda: False)
        app = desktop_app.App(root)
        ui, img, _ = _ui_with_run(tmp_path)
        app.ui = ui
        snap = ui.snapshot()
        snap['run_status'] = 'ok'
        app._sync_pp_bar(snap)
        assert 'local_contrast' in app._pp_vars and app.pp_bar.winfo_manager() == 'pack'
        app._pp_vars['local_contrast'].set(False)
        assert app._pp_timer is not None
        root.after_cancel(app._pp_timer)
        app._run_pp()                       # what the debounce timer calls
        for _ in range(600):
            root.update()
            if not app._pp_busy:
                break
            root.after(50)
            root.update()
        assert not app._pp_busy
        assert 'Updated' in app.pp_status_var.get() or 'Aktual' in app.pp_status_var.get()
        assert np.array_equal(ui._final_src['rgb'], _run(img, _args(['local_contrast'])))
        # the starless box: shown without stars, a sidecar next to the preview, and back
        app.pp_starless_var.set(True)
        app._run_pp()
        for _ in range(600):
            root.update()
            if not app._pp_busy:
                break
            root.after(50)
            root.update()
        assert app._pp_starless_shown is True and (tmp_path / 'stack_starless.jpg').exists()
        app.pp_starless_var.set(False)
        app._run_pp()
        for _ in range(600):
            root.update()
            if not app._pp_busy:
                break
            root.after(50)
            root.update()
        assert app._pp_starless_shown is False
        # a finished-run snapshot without a kept session hides the panel
        ui.run_started()
        app._sync_pp_bar(ui.snapshot())
        assert app.pp_bar.winfo_manager() == ''
    finally:
        root.destroy()


def test_starless_view_writes_a_sidecar_and_leaves_the_stars_in_files(tmp_path):
    import os
    ui, _, path = _ui_with_run(tmp_path)
    before = open(path, 'rb').read()
    on = ui.starless_view(True)
    sidecar = str(tmp_path / "stack_starless.jpg")
    assert os.path.exists(sidecar) and open(sidecar, 'rb').read() != before
    assert open(path, 'rb').read() == before                  # stars-in JPEG untouched
    assert ui.named_jpeg('final') == on
    cached = ui._final_src['starless_img']
    ui.starless_view(True)
    assert ui._final_src['starless_img'] is cached            # not detected again
    off = ui.starless_view(False)
    assert off != on and ui.named_jpeg('final') == off
    # a re-run replaces the image, and with it the cached starless one
    ui.rerun_postprocess(['local_contrast'])
    assert 'starless_img' not in ui._final_src


def test_starless_view_needs_a_finished_image_and_stars():
    import pytest

    from src.ui_events import UIEvents
    ui = UIEvents()
    ui.attach()
    with pytest.raises(RuntimeError):
        ui.starless_view(True)
    ui.remember_final_preview(np.full((64, 64, 3), 500.0, np.float32), 'x.jpg')
    with pytest.raises(RuntimeError, match='no usable star'):
        ui.starless_view(True)
