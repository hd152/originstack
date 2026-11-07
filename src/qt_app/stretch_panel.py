"""The stretch editor: pick the stretch, move its strength, symmetry, highlight protection,
black and white points with sliders or by dragging the markers on the histogram, and watch the
picture follow. *Apply and save* renders at full size and rewrites the preview JPEG (and the
PNG exports); until then nothing is written.

Rendering goes through ``UIEvents.live_stretch`` (a point-sampled reduction, see
``src/stretch_editor.py``) on a worker thread; only the newest request is shown, and a smaller
copy is rendered while a slider or marker is being dragged. Edits are kept in an undo history,
double-clicking a slider (or its label) puts it back to the run's value, and holding **B** in
the picture shows the stretch the run used.
"""
from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src import stretch_editor as se
from src.i18n import N_, _
from src.qt_app.levels import LevelsView
from src.qt_app.widgets import ClickableLabel, NoWheelComboBox, set_role

LIVE_DEBOUNCE_MS = 25
COMMIT_MS = 450            # an edit burst this long after the last change is one undo step
FULL_DIM, DRAG_DIM = 1400, 700
HISTORY_MAX = 100

# label, tooltip (translated where shown)
_PARAM_TEXT = {
    'ghs_b': (N_('Strength'),
              N_('How hard the faint signal is lifted. Higher shows fainter detail and more noise.')),
    'ghs_sp': (N_('Symmetry point'),
               N_('Where the curve puts its contrast, as a fraction of the range between black and '
                  'white. Lower favours the faintest signal.')),
    'ghs_hp': (N_('Highlight protection'),
               N_('Keeps the brightest tones from clipping: closer to 1 protects less.')),
    'black_sigma': (N_('Black point'),
                    N_('Where black sits, in sky noise units (σ) above the sky level. Above 0 clips '
                       'the noise floor to black; below 0 keeps the sky noise visible.')),
    'white_pct': (N_('White point'),
                  N_('The brightness percentile that becomes white. Lower brightens the picture and '
                     'clips more stars. It never sits closer than 50σ to the sky.')),
}


class _Results(QObject):
    live = Signal(int, str, object, bool)        # request number, status, value, show the picture
    peek = Signal(str, object)
    applied = Signal(str, object)


class _ResetSlider(QSlider):
    double_clicked = Signal()

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit()
        event.accept()


class _ResetLabel(ClickableLabel):
    double_clicked = Signal()

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit()
        event.accept()


class ParamRow(QWidget):
    """A label, a slider and a number box for one setting. ``edited`` fires only when the user
    changes it; ``reset_requested`` on a double-click (or the reset mark, shown while the value
    differs from the run's)."""

    edited = Signal(str, float)
    reset_requested = Signal(str)
    drag_started = Signal()
    drag_finished = Signal()

    def __init__(self, key: str, parent=None):
        super().__init__(parent)
        self.key = key
        lo, hi, step, decimals = se.PARAM_RANGES[key]
        self._lo, self._step = lo, step
        self._base: Optional[float] = None
        label, tip = _PARAM_TEXT[key]
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.label = _ResetLabel(_(label))
        self.label.setCursor(Qt.CursorShape.ArrowCursor)
        self.label.setMinimumWidth(120)
        self.label.setToolTip(_(tip) + '\n' + _('Double-click to go back to the run\'s value.'))
        self.label.double_clicked.connect(lambda: self.reset_requested.emit(self.key))
        lay.addWidget(self.label)
        self.slider = _ResetSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, int(round((hi - lo) / step)))
        self.slider.setToolTip(_(tip))
        self.slider.setAccessibleName(_(label))
        self.slider.double_clicked.connect(lambda: self.reset_requested.emit(self.key))
        self.slider.sliderPressed.connect(self.drag_started.emit)
        self.slider.sliderReleased.connect(self.drag_finished.emit)
        lay.addWidget(self.slider, 1)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setSingleStep(step)
        self.spin.setDecimals(decimals)
        self.spin.setFixedWidth(76)
        self.spin.setToolTip(_(tip))
        self.spin.setAccessibleName(_(label))
        lay.addWidget(self.spin)
        self.mark = set_role(ClickableLabel('↺'), 'reset')
        self.mark.setFixedWidth(16)
        self.mark.setToolTip(_('Back to the run\'s value'))
        self.mark.clicked.connect(lambda: self.reset_requested.emit(self.key))
        self.mark.setVisible(False)
        lay.addWidget(self.mark)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        self._silent = False

    def value(self) -> float:
        return float(self.spin.value())

    def set_base(self, v: Optional[float]) -> None:
        self._base = v
        self._update_mark()

    def _update_mark(self) -> None:
        differs = self._base is not None and abs(self.value() - self._base) > self._step / 2
        self.mark.setVisible(bool(differs) and self.spin.isEnabled())

    def set_value(self, v: float) -> None:
        """Show ``v`` without reporting an edit."""
        self._silent = True
        try:
            self.spin.setValue(v)
            self.slider.setValue(int(round((self.spin.value() - self._lo) / self._step)))
        finally:
            self._silent = False
        self._update_mark()

    def _from_slider(self, i: int) -> None:
        if self._silent:
            return
        self._silent = True
        self.spin.setValue(self._lo + i * self._step)
        self._silent = False
        self._update_mark()
        self.edited.emit(self.key, self.value())

    def _from_spin(self, v: float) -> None:
        if self._silent:
            return
        self._silent = True
        self.slider.setValue(int(round((v - self._lo) / self._step)))
        self._silent = False
        self._update_mark()
        self.edited.emit(self.key, self.value())

    def set_active(self, on: bool) -> None:
        for w in (self.label, self.slider, self.spin):
            w.setEnabled(on)
        self._update_mark()


class StretchEditor(QWidget):
    """Controls and levels for the finished run's picture."""

    #: a live picture to show (jpeg bytes, caption)
    live_image = Signal(bytes, str)
    #: the saved picture (jpeg bytes, slug, caption)
    image_ready = Signal(bytes, str, str)
    #: the viewer should leave compare mode: the picture is about to change
    leave_compare = Signal()
    #: a finished run left something to edit / no longer does
    availability_changed = Signal(bool)

    def __init__(self, ui, run_manager, stretch_choices: List[str], color_choices: List[str],
                 parent=None):
        super().__init__(parent)
        self.ui = ui
        self.rm = run_manager
        self._res = _Results(self)
        self._res.live.connect(self._on_live)
        self._res.peek.connect(self._on_peek)
        self._res.applied.connect(self._on_applied)
        self._busy = False
        self._pending = False
        self._want_show = False
        self._seq = 0
        self._dragging = 0
        self._peeking = False
        self._sticky = False                       # a Saved / failed message stays until the next edit
        self._last: Optional[dict] = None          # the newest live result
        self._base: Optional[dict] = None          # the run's settings (Reset, double-click, peek)
        self._shown_path: Optional[str] = None
        self._history: List[dict] = []
        self._hist_i = -1
        self.available = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        top.addWidget(QLabel(_('Stretch')))
        self.stretch_combo = NoWheelComboBox()
        self.stretch_combo.addItems(stretch_choices)
        self.stretch_combo.setAccessibleName(_('Stretch'))
        top.addWidget(self.stretch_combo)
        top.addWidget(QLabel(_('Colour')))
        self.color_combo = NoWheelComboBox()
        self.color_combo.addItems(color_choices)
        self.color_combo.setAccessibleName(_('Colour'))
        top.addWidget(self.color_combo)
        self.auto_box = QCheckBox(_('Auto-lift faint nebula'))
        self.auto_box.setChecked(True)
        self.auto_box.setToolTip(_(
            'On a frame-filling bright nebula the stretch raises its strength (and sets the symmetry '
            'point to 0) until the faint outer parts show. Turn it off to set both yourself.'))
        top.addWidget(self.auto_box)
        top.addStretch(1)
        self.log_box = QCheckBox(_('Log scale'))
        self.log_box.setChecked(True)
        self.log_box.setToolTip(_('Counts on a logarithmic scale, so the faint tail shows.'))
        top.addWidget(self.log_box)
        lay.addLayout(top)

        presets = QHBoxLayout()
        presets.addWidget(set_role(QLabel(_('Saved stretches')), 'dim'))
        self.preset_combo = NoWheelComboBox()
        self.preset_combo.setMinimumWidth(160)
        self.preset_combo.setAccessibleName(_('Saved stretches'))
        presets.addWidget(self.preset_combo)
        self.preset_save = QPushButton(_('Save as…'))
        self.preset_delete = QPushButton(_('Delete'))
        presets.addWidget(self.preset_save)
        presets.addWidget(self.preset_delete)
        presets.addStretch(1)
        lay.addLayout(presets)

        self.levels = LevelsView()
        lay.addWidget(self.levels)

        grid = QGridLayout()
        self.rows: Dict[str, ParamRow] = {}
        for i, key in enumerate(se.PARAM_KEYS):
            row = ParamRow(key)
            row.edited.connect(self._on_param)
            row.reset_requested.connect(self._reset_param)
            row.drag_started.connect(lambda: self._set_dragging(True))
            row.drag_finished.connect(lambda: self._set_dragging(False))
            self.rows[key] = row
            grid.addWidget(row, i // 2, i % 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        lay.addLayout(grid)

        bottom = QHBoxLayout()
        self.undo_btn = QToolButton()
        self.undo_btn.setText('↶')
        self.undo_btn.setToolTip(_('Undo the last stretch change') + ' (Ctrl+Z)')
        self.redo_btn = QToolButton()
        self.redo_btn.setText('↷')
        self.redo_btn.setToolTip(_('Redo it') + ' (Ctrl+Y)')
        self.undo_btn.clicked.connect(self.undo)
        self.redo_btn.clicked.connect(self.redo)
        self.reset_btn = QPushButton(_('Reset'))
        self.reset_btn.setToolTip(_('Back to the stretch this run used.'))
        self.reset_btn.clicked.connect(self._on_reset)
        self.apply_btn = QPushButton(_('Apply and save'))
        set_role(self.apply_btn, 'accent')
        self.apply_btn.setToolTip(_('Render at full size and rewrite the preview JPEG (and the '
                                    'PNG exports).'))
        self.apply_btn.clicked.connect(self._on_apply)
        self.status = set_role(QLabel(''), 'dim')
        self.status.setWordWrap(True)
        for w in (self.undo_btn, self.redo_btn, self.reset_btn, self.apply_btn):
            bottom.addWidget(w)
        bottom.addWidget(self.status, 1)
        lay.addLayout(bottom)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(lambda: self._request(show=True))
        self._commit_timer = QTimer(self)
        self._commit_timer.setSingleShot(True)
        self._commit_timer.timeout.connect(self._commit)
        self.stretch_combo.activated.connect(lambda _i: self._on_stretch_changed())
        self.color_combo.activated.connect(lambda _i: self._on_edit())
        self.auto_box.toggled.connect(self._on_auto_toggled)
        self.log_box.toggled.connect(self.levels.set_log)
        self.preset_combo.activated.connect(lambda _i: self._load_preset())
        self.preset_save.clicked.connect(self._save_preset)
        self.preset_delete.clicked.connect(self._delete_preset)
        self.levels.black_moved.connect(self._on_black_dragged)
        self.levels.sym_moved.connect(self._on_sym_dragged)
        self.levels.white_moved.connect(self._on_white_dragged)
        self.levels.drag_started.connect(lambda: self._set_dragging(True))
        self.levels.drag_finished.connect(lambda: self._set_dragging(False))
        ctx = Qt.ShortcutContext.WidgetWithChildrenShortcut
        QShortcut(QKeySequence('Ctrl+Z'), self, activated=self.undo, context=ctx)
        QShortcut(QKeySequence('Ctrl+Y'), self, activated=self.redo, context=ctx)
        QShortcut(QKeySequence('Ctrl+Shift+Z'), self, activated=self.redo, context=ctx)
        self._refresh_presets()
        self._update_active()
        self._update_history_buttons()

    # ── state ───────────────────────────────────────────────────────────

    def params(self) -> dict:
        out = {k: r.value() for k, r in self.rows.items()}
        out['nebula_auto'] = self.auto_box.isChecked()
        return out

    def state_now(self) -> dict:
        """Everything the editor sets: the stretch, the colour mode and the parameters."""
        return {'stretch': self.stretch_combo.currentText(), 'color': self.color_combo.currentText(),
                **self.params()}

    def load_state(self, state: dict) -> None:
        """Show ``state`` (``UIEvents.stretch_state``) in the controls without rendering."""
        self.stretch_combo.setCurrentText(state.get('stretch', 'ghs'))
        self.color_combo.setCurrentText(state.get('color', 'preserve'))
        for k, row in self.rows.items():
            row.set_value(float(state.get(k, se.PARAM_DEFAULTS[k])))
        self.auto_box.blockSignals(True)
        self.auto_box.setChecked(bool(state.get('nebula_auto', True)))
        self.auto_box.blockSignals(False)
        self._update_active()
        self._update_labels()

    def _update_active(self) -> None:
        stretch = self.stretch_combo.currentText()
        used = se.controls_for(stretch)
        for k, row in self.rows.items():
            row.set_active(k in used)
        self.auto_box.setEnabled(stretch in se.GHS_FAMILY)
        self.levels.set_enabled_markers('black_sigma' in used, 'white_pct' in used,
                                        sym='ghs_sp' in used)

    def _update_labels(self) -> None:
        r = self.rows
        self.levels.set_marker_labels(black=f"{r['black_sigma'].value():.1f}σ",
                                      sym=f"{r['ghs_sp'].value():.2f}",
                                      white=f"{r['white_pct'].value():.2f}%")

    def sync(self, snap: dict) -> None:
        """Follow the UI snapshot: available once a run has finished with its picture kept."""
        info = snap.get('restretch')
        ok = info is not None and snap['run_status'] == 'ok'
        if not ok:
            if self.available:
                self.available = False
                self._shown_path = None
                self._last = None
                self._base = None
                self.levels.set_data(None, None, None, None)
                self.availability_changed.emit(False)
            return
        if self._shown_path != info['path']:
            self._shown_path = info['path']
            state = self.ui.stretch_state()
            if state is None:
                return
            self._base = dict(state)
            self.load_state(state)
            for k, row in self.rows.items():
                row.set_base(float(state.get(k, se.PARAM_DEFAULTS[k])))
            self._history = [self.state_now()]
            self._hist_i = 0
            self._update_history_buttons()
            self.status.setText('')
            self._last = None
            self.available = True
            self.availability_changed.emit(True)
            self._request(show=False)

    def refresh(self) -> None:
        """The picture changed under the editor (a Phase 4 step was switched): redo the levels."""
        if self.available:
            self._request(show=False)

    # ── edits ───────────────────────────────────────────────────────────

    def _schedule(self) -> None:
        self._timer.start(LIVE_DEBOUNCE_MS)

    def _on_edit(self) -> None:
        """Something the user changed: redraw soon, and make it an undo step after a pause."""
        self._sticky = False
        self._schedule()
        self._commit_timer.start(COMMIT_MS)

    def _on_param(self, key: str, _value: float) -> None:
        self._update_labels()
        self._on_edit()

    def _on_stretch_changed(self) -> None:
        self._update_active()
        self._on_edit()

    def _on_auto_toggled(self, on: bool) -> None:
        if not on and self._last is not None:
            # hand over the strength and symmetry point the lift chose, so nothing jumps
            eff = self._last.get('effective') or {}
            for k in ('ghs_b', 'ghs_sp'):
                if k in eff:
                    self.rows[k].set_value(float(eff[k]))
            self._update_labels()
        self._on_edit()

    def _reset_param(self, key: str) -> None:
        if self._base is None:
            return
        self.rows[key].set_value(float(self._base.get(key, se.PARAM_DEFAULTS[key])))
        self._update_labels()
        self._on_edit()

    def _axis_xs(self):
        """Black and white as sigmas above the sky, or None before the first render."""
        lv = self.levels
        if lv.black is None or lv.white is None:
            return None
        return float(se.from_axis(lv.black)), float(se.from_axis(lv.white))

    def _on_black_dragged(self, u: float) -> None:
        self.rows['black_sigma'].set_value(se.black_sigma_at(u))
        self._update_labels()
        self._on_edit()

    def _on_white_dragged(self, u: float) -> None:
        if self._last is None:
            return
        self.rows['white_pct'].set_value(
            se.white_pct_at(u, self._last['lum'], self._last['med'], self._last['sigma']))
        self._update_labels()
        self._on_edit()

    def _on_sym_dragged(self, u: float) -> None:
        xs = self._axis_xs()
        if xs is None or xs[1] <= xs[0]:
            return
        sp = (float(se.from_axis(u)) - xs[0]) / (xs[1] - xs[0])
        self.rows['ghs_sp'].set_value(sp)
        self._update_labels()
        self._on_edit()

    def _set_dragging(self, on: bool) -> None:
        """While a slider or marker is held, render a smaller copy; let go for the full one."""
        self._dragging = max(0, self._dragging + (1 if on else -1))
        if not on and self._dragging == 0:
            self._timer.stop()
            self._request(show=True)

    # ── undo / redo ─────────────────────────────────────────────────────

    def _commit(self) -> None:
        """Make the current settings an undo step (if they differ from the last one)."""
        if not self._history:
            return
        now = self.state_now()
        if self._history[self._hist_i] == now:
            return
        del self._history[self._hist_i + 1:]
        self._history.append(now)
        del self._history[:-HISTORY_MAX]
        self._hist_i = len(self._history) - 1
        self._update_history_buttons()

    def _update_history_buttons(self) -> None:
        self.undo_btn.setEnabled(self._hist_i > 0)
        self.redo_btn.setEnabled(0 <= self._hist_i < len(self._history) - 1)

    def _go_to(self, i: int) -> None:
        self._hist_i = max(0, min(i, len(self._history) - 1))
        self.load_state(self._history[self._hist_i])
        self._update_history_buttons()
        self._request(show=True)

    def undo(self) -> None:
        if not self.available:
            return
        self._commit_timer.stop()
        self._commit()                               # a change still waiting becomes a step first
        if self._hist_i > 0:
            self._go_to(self._hist_i - 1)

    def redo(self) -> None:
        if self.available and self._hist_i < len(self._history) - 1:
            self._go_to(self._hist_i + 1)

    # ── saved stretches ─────────────────────────────────────────────────

    @staticmethod
    def _presets() -> dict:
        from src.desktop_control import load_app_setting
        data = load_app_setting('stretch_presets', {})
        return data if isinstance(data, dict) else {}

    def _refresh_presets(self, select: str = '') -> None:
        self.preset_combo.clear()
        self.preset_combo.addItems(sorted(self._presets()))
        self.preset_combo.setCurrentIndex(self.preset_combo.findText(select) if select else -1)

    def _save_preset(self) -> None:
        from src.desktop_control import save_app_setting
        name, ok = QInputDialog.getText(self, _('Save stretch'), _('Name for this stretch:'),
                                        text=self.preset_combo.currentText())
        name = name.strip()
        if ok and name:
            presets = self._presets()
            presets[name] = self.state_now()
            save_app_setting('stretch_presets', presets)
            self._refresh_presets(select=name)

    def _load_preset(self) -> None:
        state = self._presets().get(self.preset_combo.currentText())
        if state is not None and self.available:
            self.load_state(state)
            self._on_edit()

    def _delete_preset(self) -> None:
        from src.desktop_control import save_app_setting
        presets = self._presets()
        if presets.pop(self.preset_combo.currentText(), None) is not None:
            save_app_setting('stretch_presets', presets)
            self._refresh_presets()

    def _on_reset(self) -> None:
        state = self.ui.reset_stretch()
        if state is not None:
            self.load_state(state)
            self.leave_compare.emit()
            self._commit()
            self._request(show=True)

    # ── live rendering ──────────────────────────────────────────────────

    def _request(self, show: bool) -> None:
        if self.rm.is_running() or not self.available:
            return
        self._want_show = self._want_show or show
        if self._busy:
            self._pending = True
            return
        self._busy = True
        self._seq += 1
        seq, show = self._seq, self._want_show
        self._want_show = False
        params, stretch, color = self.params(), self.stretch_combo.currentText(), \
            self.color_combo.currentText()
        max_dim = DRAG_DIM if self._dragging else FULL_DIM
        if show and not self._sticky:
            self.status.setText(_('Rendering…'))

        def bg() -> None:
            try:
                self._res.live.emit(seq, 'ok', self.ui.live_stretch(params, stretch, color, max_dim),
                                    show)
            except Exception as e:
                self._res.live.emit(seq, 'error', str(e), show)
        threading.Thread(target=bg, daemon=True).start()

    def _on_live(self, seq: int, status: str, value, show: bool) -> None:
        self._busy = False
        if not self.available:                 # a new run began while this rendered: drop it
            self._pending = False
            return
        if status == 'ok':
            self._last = value
            m = value['markers']
            eff = value.get('effective') or {}
            sym = None
            if 'ghs_sp' in eff:
                xb, xw = float(se.from_axis(m['black'])), float(se.from_axis(m['white']))
                sym = float(se.to_axis(xb + float(eff['ghs_sp']) * (xw - xb)))
            self.levels.set_data(value['hist'], value['curve'], m['black'], m['white'], sym)
            self.levels.set_lum(value.get('lum'), value.get('med', 0.0), value.get('sigma', 1.0))
            lifted = self.auto_box.isChecked() and eff.get('lifted')
            used = se.controls_for(self.stretch_combo.currentText())
            self.levels.set_enabled_markers('black_sigma' in used, 'white_pct' in used,
                                            sym=('ghs_sp' in used) and not lifted)
            if self._sticky:
                pass
            elif lifted:
                self.status.setText(_('Auto-lift set strength {b:.1f} and symmetry {sp:.2f} for this '
                                      'nebula. Turn it off to set them yourself.',
                                      b=eff['ghs_b'], sp=eff['ghs_sp']))
            elif not self._pending:
                self.status.setText(_('Live preview is a reduced copy; Apply and save renders at '
                                      'full size.'))
            if show and not self._pending and not self._peeking:
                self.leave_compare.emit()
                self.live_image.emit(value['jpeg'], _('Final (live preview)'))
        else:
            self.status.setText(_('Preview failed: {error}', error=value))
        if self._pending:
            self._pending = False
            self._request(show=False)

    # ── hold to see the stretch the run used ────────────────────────────

    def peek(self, on: bool) -> None:
        """Show the run's own stretch while held (nothing is changed or kept); on release the
        current settings come back."""
        if not self.available or self._base is None:
            return
        self._peeking = on
        if not on:
            self._request(show=True)
            return
        base = dict(self._base)

        def bg() -> None:
            try:
                r = self.ui.live_stretch({k: base[k] for k in se.PARAM_KEYS + ('nebula_auto',)
                                          if k in base}, base.get('stretch'), base.get('color'),
                                         FULL_DIM, persist=False)
                self._res.peek.emit('ok', r['jpeg'])
            except Exception as e:
                self._res.peek.emit('error', str(e))
        threading.Thread(target=bg, daemon=True).start()

    def _on_peek(self, status: str, value) -> None:
        if status == 'ok' and self._peeking:
            self.live_image.emit(value, _('Final (as the run made it)'))

    # ── apply ───────────────────────────────────────────────────────────

    def _on_apply(self) -> None:
        if self.rm.is_running():
            return
        stretch, color, params = (self.stretch_combo.currentText(), self.color_combo.currentText(),
                                  self.params())
        self._commit_timer.stop()
        self._commit()
        self.apply_btn.setEnabled(False)
        self.status.setText(_('Rendering at full size…'))

        def bg() -> None:
            try:
                self._res.applied.emit('ok', self.ui.restretch(stretch, color, params))
            except Exception as e:
                self._res.applied.emit('error', str(e))
        threading.Thread(target=bg, daemon=True).start()

    def _on_applied(self, status: str, value) -> None:
        self.apply_btn.setEnabled(True)
        if not self.available:
            return
        if status == 'ok':
            self._sticky = True
            self.status.setText(_('Saved {name}', name=os.path.basename(self._shown_path or '')))
            self.leave_compare.emit()
            self.image_ready.emit(value, 'final',
                                  _('Final ({stretch})', stretch=self.stretch_combo.currentText()))
        else:
            self._sticky = True
            self.status.setText(_('Re-stretch failed: {error}', error=value))
