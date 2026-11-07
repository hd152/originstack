"""After a run: switch Phase 4 steps off and on and compare one step before and after.
Everything runs through ``UIEvents`` (``rerun_postprocess``, ``starless_view``,
``compare_postprocess_step``); this module is only the controls. The stretch is the
stretch editor's (``stretch_panel``)."""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.desktop_common import _PP_STEP_LABELS
from src.i18n import _
from src.qt_app.viewer import PreviewPane
from src.qt_app.widgets import HSlider, NoWheelComboBox, set_role

PP_DEBOUNCE_MS = 600


class _Results(QObject):
    """Carries a worker thread's result back to the GUI thread (queued across threads)."""
    rerun = Signal(str, object, float)        # status, value, seconds
    compare = Signal(object, str, object)     # cancel event, status, value


class CompareStepDialog(QDialog):
    """The final image with one Phase 4 step off (left) and on (right), under a wipe slider."""

    def __init__(self, step_label: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(_('Before / after: {step}', step=step_label))
        self.resize(1100, 760)
        lay = QVBoxLayout(self)
        self.view = PreviewPane()
        self.view.set_caption(_('Comparing…'))
        lay.addWidget(self.view, 1)
        self.wipe = HSlider()
        self.wipe.moved.connect(self.view.set_wipe)
        lay.addWidget(self.wipe)
        row = QHBoxLayout()
        row.addWidget(set_role(QLabel(_('Left: {step} off. Right: on. Drag the slider; wheel zooms, '
                                        'drag pans.', step=step_label)), 'dim'))
        row.addStretch(1)
        flip = QPushButton(_('Hold to show on'))
        flip.pressed.connect(lambda: self.view.set_wipe(0.0))
        flip.released.connect(lambda: self.view.set_wipe(self.wipe.fraction()))
        row.addWidget(flip)
        close = QPushButton(_('Close'))
        close.clicked.connect(self.close)
        row.addWidget(close)
        lay.addLayout(row)

    def show_result(self, off: bytes, on: bytes, step_label: str) -> None:
        self.view.load_slot(off, 'compare_off', _('{step}: off | on', step=step_label))
        self.view.set_compare_slot(on, 'compare_on')
        self.view.set_compare(True)
        self.view.set_wipe(self.wipe.fraction())


class PostProcessPanel(QWidget):
    """The re-stretch bar and the Post-processing steps group of the preview column."""

    #: a new final image to show: (jpeg bytes, slug, caption)
    image_ready = Signal(bytes, str, str)
    #: the viewer should leave compare mode (a new final image replaces the wipe)
    leave_compare = Signal()

    def __init__(self, ui, run_manager, parent=None):
        super().__init__(parent)
        self.ui = ui
        self.rm = run_manager
        self._res = _Results(self)
        self._res.rerun.connect(self._on_rerun_done)
        self._res.compare.connect(self._on_compare_done)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        # ── Phase 4 steps ──
        self.pp_group = QWidget()
        pl = QVBoxLayout(self.pp_group)
        pl.setContentsMargins(0, 0, 0, 0)
        self.boxes_grid = QGridLayout()
        pl.addLayout(self.boxes_grid)
        self.starless_box = QCheckBox(_('Show without stars (also saves a _starless JPEG)'))
        self.starless_box.toggled.connect(lambda _on: self._run_pp())
        pl.addWidget(self.starless_box)
        lrow = QHBoxLayout()
        self.live_box = QCheckBox(_('Update as I tick'))
        self.live_box.setChecked(True)
        self.live_box.setToolTip(_('Redo Phase 4 shortly after the last box changed. Turn it off to '
                                   'change several steps first, then press Apply.'))
        lrow.addWidget(self.live_box)
        self.apply_btn = QPushButton(_('Apply'))
        self.apply_btn.clicked.connect(lambda: self._run_pp())
        lrow.addWidget(self.apply_btn)
        lrow.addStretch(1)
        pl.addLayout(lrow)
        crow = QHBoxLayout()
        crow.addWidget(QLabel(_('Compare a step')))
        self.cmp_combo = NoWheelComboBox()
        self.cmp_combo.setMinimumWidth(220)
        crow.addWidget(self.cmp_combo)
        self.cmp_btn = QPushButton(_('Before / after'))
        self.cmp_btn.clicked.connect(self._on_compare_step)
        crow.addWidget(self.cmp_btn)
        crow.addStretch(1)
        pl.addLayout(crow)
        self.pp_status = set_role(QLabel(''), 'dim')
        pl.addWidget(self.pp_status)
        lay.addWidget(self.pp_group)
        self.pp_group.hide()

        self._pp_boxes: Dict[str, QCheckBox] = {}
        self._cmp_ids: Dict[str, str] = {}
        self._pp_shown: Optional[str] = None
        self._pp_busy = False
        self._pp_pending = False
        self._starless_shown = False
        self._cmp_dialog: Optional[CompareStepDialog] = None
        self._cmp_cancel: Optional[threading.Event] = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._run_pp)

    # ── show / hide from the UI snapshot ────────────────────────────────

    def sync(self, snap: Dict[str, Any]) -> None:
        self._sync_steps(snap)

    def _sync_steps(self, snap: Dict[str, Any]) -> None:
        info = snap.get('restretch')
        if not snap.get('postprocess') or info is None or snap['run_status'] != 'ok':
            if self._pp_shown is not None:
                self.pp_group.hide()
                self._pp_shown = None
            return
        if self._pp_shown == info['path'] or self._pp_busy:
            return
        steps = self.ui.postprocess_steps()
        if not steps:
            return
        self._pp_shown = info['path']
        self.starless_box.blockSignals(True)
        self.starless_box.setChecked(False)
        self.starless_box.blockSignals(False)
        self._starless_shown = False
        while self.boxes_grid.count():
            w = self.boxes_grid.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        self._pp_boxes = {}
        for n, st in enumerate(steps):
            box = QCheckBox(_(_PP_STEP_LABELS.get(st['id'], st['label'])))
            box.setChecked(st['on'])
            box.toggled.connect(lambda _on: self._on_step_changed())
            self._pp_boxes[st['id']] = box
            self.boxes_grid.addWidget(box, n // 3, n % 3)
        self._cmp_ids = {_(_PP_STEP_LABELS.get(st['id'], st['label'])): st['id'] for st in steps}
        self.cmp_combo.clear()
        self.cmp_combo.addItems(list(self._cmp_ids))
        self.pp_status.setText('')
        self.pp_group.show()

    # ── steps on and off ────────────────────────────────────────────────

    def _on_step_changed(self) -> None:
        """A box changed: re-run Phase 4 shortly after the last change (or wait for Apply)."""
        if self.live_box.isChecked():
            self._timer.start(PP_DEBOUNCE_MS)
        else:
            self.pp_status.setText(_('Changes not applied yet.'))

    def _run_pp(self) -> None:
        """Bring the final image in line with the boxes: redo Phase 4 if a step changed, then
        show it with or without its stars. One at a time; a change made meanwhile runs after."""
        if self.rm.is_running():
            return
        if self._pp_busy:
            self._pp_pending = True
            return
        off = [i for i, b in self._pp_boxes.items() if not b.isChecked()]
        starless = self.starless_box.isChecked()
        shown_before = self._starless_shown
        self._pp_busy = True
        self.pp_status.setText(_('Updating…'))
        started = time.time()

        def _bg() -> None:
            try:
                data = self.ui.rerun_postprocess(off)        # shows the stars-in image again
                shown = False if data is not None else shown_before
                if starless or starless != shown:
                    data = self.ui.starless_view(starless)
                    shown = starless
                self._res.rerun.emit('ok', (data, shown), time.time() - started)
            except Exception as e:
                self._res.rerun.emit('error', str(e), 0.0)
        threading.Thread(target=_bg, daemon=True).start()

    def _on_rerun_done(self, status: str, value, secs: float) -> None:
        self._pp_busy = False
        if status == 'ok':
            data, self._starless_shown = value
            if data is not None:
                self.leave_compare.emit()
                self.image_ready.emit(bytes(data), 'final', _('Final (starless)')
                                      if self._starless_shown else _('Final (post-processed)'))
                self.pp_status.setText(_('Updated in {secs} s', secs=f'{secs:.0f}'))
            else:
                self.pp_status.setText('')
        else:
            # what the viewer shows is unknown after a failure: the box follows the last good state
            self.starless_box.blockSignals(True)
            self.starless_box.setChecked(self._starless_shown)
            self.starless_box.blockSignals(False)
            self.pp_status.setText(_('Update failed: {error}', error=value))
        if self._pp_pending:
            self._pp_pending = False
            self._run_pp()

    # ── before / after of one step ──────────────────────────────────────

    def close_compare(self) -> None:
        if self._cmp_cancel is not None:
            self._cmp_cancel.set()
        dlg, self._cmp_dialog = self._cmp_dialog, None
        if dlg is not None:
            dlg.close()

    def _on_compare_step(self) -> None:
        step_id = self._cmp_ids.get(self.cmp_combo.currentText())
        if step_id is None or self._pp_busy or self.rm.is_running():
            return
        step_label = self.cmp_combo.currentText()
        self.close_compare()
        cancel = threading.Event()
        self._cmp_cancel = cancel
        dlg = CompareStepDialog(step_label, self)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.finished.connect(lambda _r: cancel.set())
        self._cmp_dialog = dlg
        dlg.show()
        self._pp_busy = True
        self.cmp_btn.setEnabled(False)
        self.pp_status.setText(_('Comparing…'))

        def _bg() -> None:
            try:
                off, on = self.ui.compare_postprocess_step(step_id, cancel=cancel)
                self._res.compare.emit(cancel, 'ok', (off, on, step_label))
            except Exception as e:
                self._res.compare.emit(cancel, 'error', str(e))
        threading.Thread(target=_bg, daemon=True).start()

    def _on_compare_done(self, cancel, status: str, value) -> None:
        self._pp_busy = False
        self.cmp_btn.setEnabled(True)
        self.pp_status.setText('')
        dlg = self._cmp_dialog
        if not cancel.is_set() and dlg is not None:
            if status == 'ok':
                off, on, step_label = value
                dlg.show_result(off, on, step_label)
            else:
                self.close_compare()
                self.pp_status.setText(_('Compare failed: {error}', error=value))
        if self._pp_pending:
            self._pp_pending = False
            self._run_pp()

    def is_busy(self) -> bool:
        return self._pp_busy

