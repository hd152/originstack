# Qt desktop window and the stretch editor

Why, how it is laid out, and what was measured while building it (2026-11).

## Why Qt

The tkinter window had no smooth image viewer (every zoom re-made a `PhotoImage`), no real
slider or drag-handle widgets, and its ~3000 lines mixed form logic with widget calls. The move
to Qt (PySide6) was done as a rewrite of the window only: `desktop_control.RunManager`,
`ui_events.UIEvents`, the pipeline and the translation catalogs are the ones the tkinter
window used. The tkinter window was removed (it had been kept as `--tk` for one release); `src/desktop_app.py` is now only the launcher.

## Layout (`src/qt_app/`)

| File | Role |
|------|------|
| `form_model.py` | `FormModel(QObject)`: one value per argparse `dest`, the defaults, `read_form`, target cards / run mode as views over `target_type` / `comet_mode` / `preset`, folder scan, presets, "This run" text. No widgets: testable headless. |
| `setup_form.py` | `SetupForm`, `GoalPicker`/`Card`, `FieldView` (one field = dot, label, input, Browse, reset, hint). Additional options: sidebar of groups, search, only-changed, expert toggle. Pages are shown/hidden, not stacked (a `QStackedWidget` is as tall as its tallest page). |
| `viewer.py` | `ImageView` (`QGraphicsView`: wheel zoom about the pointer, drag pan, wipe painted in `paintEvent` over the second image), `HistogramView` (the displayed picture's 8-bit histogram), `PreviewPane`. |
| `stretch_panel.py`, `levels.py` | The stretch editor (below). |
| `post_panel.py` | Phase 4 step boxes (debounced re-run, or *Update as I tick* off + *Apply*), starless view, before/after of one step in a dialog. |
| `main_window.py` | Header, left column (form / run controls / pipeline / log), right column (picture over tabbed tools, then the run summary), poll timer, drag and drop, update download, report menu. |

Worker threads (`threading.Thread`) emit Qt signals; widgets are only touched on the GUI thread.

Gotchas met:

- A window-wide `QShortcut('F')` swallows the letter in every line edit. The picture shortcuts
  (F, 1, C) are scoped to the picture widget.
- A scroll area with sliders and combo boxes: the wheel must scroll, not change a setting
  (`NoWheelComboBox`).
- Card thumbnails set the minimum width of the Setup column unless they are a width-flexible
  widget (`PixmapLabel`).
- The finished snapshot can arrive while the run thread still counts as running (the same
  latent bug was in the tkinter window): apply the last snapshot's status on every poll.
- `QPushButton` text `&&`: avoid ampersands in labels, write "and".

## Stretch editor

`UIEvents.live_stretch(params, stretch, color)` renders the remembered final image (`fs['rgb']`)
with the editor's settings and returns a JPEG, the linear histogram, the tone curve and the
marker positions. Nothing is written; `restretch(stretch, color, params)` is *Apply and save*.
`fs['kw']` holds the current settings (`fs['kw0']` the run's, for *Reset*), so a Phase 4 re-run
keeps them.

Settings: `ghs_b` (strength), `ghs_sp` (symmetry point), `ghs_hp` (highlight protection),
`black_sigma`, `white_pct` (new: the percentile that becomes white, default 99.5, was hard-coded
in four places in `render_preview_float`), `nebula_auto`. `stretch_editor.controls_for(stretch)`
lists what each stretch reads (checked by a test that renders each setting changed).

### Fidelity of the live preview

A 3033 x 2017 stack renders at 1400 px in 0.36 s. To make that preview equal what *Apply* writes:

1. **Point-sample, do not average.** The stretch takes its black point from the sky sigma and its
   white point from percentiles. Averaging f x f pixels shrinks sigma (by f for white noise) and
   dims the bright tail. Every f-th pixel keeps both. Measured on the Eagle stack: float render of
   the sampled image vs the full render sampled at the same pixels, mean abs difference 0.00004
   (99th percentile 0.0002). A JPEG of a very dark image differs from the float by ~0.3 of an
   8-bit level, which looked like a statistics mismatch until the float comparison.
2. **Scale the pixel-count rules.** `_extended_highlight_top` (roll-off) needs the largest region
   above white to be >= max(0.015% of the frame, `PREVIEW_ROLLOFF_MIN_PIXELS` = 1000 px), the
   star-core feather has a 2000 px floor, and the feather itself is a few pixels wide. On the
   reduction these counts are 1/f^2 too large. `pixel_scale` (area ratio) is passed as a context
   variable; a synthetic compact nebula (M57-like, ~2.6k px above white) came out 4.7x further
   from the full render without it (mean error 0.0135 vs 0.0028).
3. **Auto-lift.** On a frame-filling bright nebula with `black_sigma <= 0`,
   `_extended_nebula_ghs` raises b (and sets SP to 0) to whatever lifts the faint wings, never
   below the given values. Moving the strength slider below that level changes nothing, so the
   editor exposes it as *Auto-lift faint nebula*; unticking it copies the effective b and SP into
   the sliders so nothing jumps.
4. **White floor.** White is never closer than `PREVIEW_WHITE_MIN_SIGMA` (50) sky sigmas to the
   sky: on an empty field `white_pct` does nothing (the marker simply does not move).

Local stretches (`masked`, `multiscale`, `clahe`) work on pixel neighbourhoods, so the live
preview approximates them; *Apply and save* is exact.

### Polish batch (same week)

- **Theme**: `theme.py` holds two palettes; `apply_mode` rebinds the module-level colour names
  before the window is built, so widgets read `theme.ACCENT` at construction. The combo arrow and
  the check mark are drawn with `QPainter` into PNGs in the temp folder at start-up (a style sheet
  can only point at image files; this avoids shipping data files). The faint-text colour was
  2.4:1 on the panel; both palettes now pass 4.5:1 for every text colour
  (`TestTheme`).
- **Remembered state** (`window_state.py`): geometry, the three splitters, the open tool tab,
  collapsed tools, Additional options and the target cards, under `qt_window` in the app
  settings; recent folders under `recent_folders`.
- **Editor**: `live_stretch(max_dim, persist)`; the per-size cache is keyed by `max_dim` (a
  small drag copy and the full one). While a slider or marker is held (`_set_dragging`) the
  render is 700 px, on release 1400 px. Undo steps are committed after 450 ms of quiet, so a
  drag is one step. The saved/failed message is sticky until the next edit: a render that
  finishes after *Apply* otherwise overwrote it.
- **Viewer**: `ImageView` handles the wipe drag, the minimap click, pinch (touch gesture and
  macOS native zoom) and the hold-B key itself; the pane re-emits `hold_original` and
  `wipe_dragged`. The symmetry marker is placed from the effective `ghs_sp` between the black
  and white markers (an approximation: the real curve rescales SP by up to 3x on faint
  targets).
- **Phase times**: `UIEvents.run_finished` closes the phase the run ended in so the last
  phase has an `elapsed`.

### 1:1 viewing and the frame graph

- **Real-pixel scene.** `ImageView` sets the scene rect to the *real* image size and scales the
  pixmap item by `logical / shown` (`set_image(img, logical_size)`), so a 1024 px preview, a
  1400 px live render and the full-size array are interchangeable: the zoom and position stay,
  `one_to_one()` is 1 screen pixel per real pixel, the zoom label is native percent, and the
  inspector reports real coordinates. `needs_detail()` is `scale * item_scale > 1.05`: the shown
  pixels are being magnified. The main window waits 250 ms after a zoom, then asks
  `UIEvents.full_view()` (a full-size `render_preview_rgb_float`, 2.5 s on a 6 Mpx stack; over
  60 Mpx a point-sampled reduction with `pixel_scale`, like the live stretch) on a worker thread
  and swaps it in with `load_array`. A generation counter drops a result when another picture
  was shown meanwhile. A logical size is only used when the aspect matches (2%).
- **Linear values.** `UIEvents.sample_linear(x, y)` indexes the linear stack; the sky is measured
  once on an 800 px sample and kept.
- **Excluding frames.** `--exclude-frames` (Phase 1 group, so it is in the fingerprint) is applied
  where the lights are discovered (`cli`, one target): a new list is a full Phase 1 again, there
  is no per-frame cache to reuse. The *review* is the cheap path: `args._review_hook` is called
  right after `quality_gate` (`frame_processor.apply_frame_review`), before registration or any
  file is touched; the app's hook (`RunManager._review_hook`) publishes every light's metrics
  and the rejection reasons through `UIEvents.review_request`, waits on an event (polling the
  cancel flag every 0.2 s) and returns the dropped names, which become rejected with the reason
  "excluded by you (frame review)". Hook and `--exclude-frames` names match on file name only.
- **Graph.** `FrameGraph` paints (a single scatter, median of the kept frames dashed; rejected =
  red ring, excluded = grey cross), hit-tests only the frames near the pointer, and reports a click
  or a box selection. The excluded set lives in the Setup form's `exclude_frames` setting, so the
  graph, the Additional options field and a CLI command always agree. `UIEvents` keeps up to 1500
  frame rows (`_MAX_FRAME_ROWS`, was 60). A QWidget attribute called `metric` shadows
  `QPaintDevice.metric()` and breaks painting: the graph's is `measure`.

### Not done

- Per-step parameters for Phase 4 (denoise strength, star-reduction amount) with live feedback:
  `StepSnapshots` is keyed by step name, so parameters would have to join the key, and the
  expensive steps (wavelet, NN) take seconds to minutes. The step boxes re-run from the earliest
  changed step's snapshot, which is what keeps cheap late steps fast.
- Curves / per-channel levels after the stretch.
- Taskbar/dock progress: Qt 6 dropped `QWinTaskbarButton`; it would need `ITaskbarList3` through ctypes (Windows) and a dock badge on macOS.
- The starless view is not edited live (the editor stretches the stars-in image, or the layered
  stretch when the run has a starless layer).
