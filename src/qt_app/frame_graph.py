"""The frame-quality graph: one metric per frame (score, FWHM, SNR, stars) in capture order, to
spot a bad stretch of the night, with frames picked by click or by dragging a box and then left
out of the stack (or put back).

``FrameGraph`` draws and reports clicks and selections; ``FramesPanel`` adds the metric choice,
the Exclude / Include buttons and the banner shown while a run waits for the user's review.
The excluded names live in the Setup form's ``exclude_frames`` setting, so they are part of the
run exactly like any other setting.
"""
from __future__ import annotations

from typing import List, Optional, Set

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.i18n import N_, _
from src.qt_app import theme
from src.qt_app.widgets import NoWheelComboBox, set_role

# key, label, the better direction
METRICS = (
    ('score', N_('Quality score'), 'high'),
    ('fwhm', N_('Star width (FWHM, px)'), 'low'),
    ('snr', N_('Signal to noise'), 'high'),
    ('stars', N_('Stars found'), 'high'),
)
_HIT = 9          # px
REVIEW_COUNTDOWN_S = 5     # a frame review continues by itself after this, unless you touch something


class FrameGraph(QWidget):
    """Frames in capture order along x, the chosen metric up. Frames the pipeline rejected are
    red rings, frames you left out are grey crosses, the picked ones have a bright ring. Click
    a frame to look at it; drag a box to pick several."""

    frame_clicked = Signal(str)
    selection_changed = Signal(object)          # set of names
    pressed = Signal()                          # the user pressed the mouse on the graph

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setMouseTracking(True)
        self.setAccessibleName(_('Graph of one quality measure for every frame'))
        self.rows: List[dict] = []
        self.measure = 'score'
        self.excluded: Set[str] = set()
        self.selected: Set[str] = set()
        self.focus: Optional[str] = None
        self._hover: Optional[int] = None
        self._band: Optional[QRectF] = None
        self._press: Optional[QPointF] = None
        self._add = False

    # ── data ────────────────────────────────────────────────────────────

    def set_rows(self, rows: List[dict]) -> None:
        self.rows = list(rows)
        names = {r['name'] for r in self.rows}
        self.selected &= names
        self.update()

    def set_measure(self, key: str) -> None:
        self.measure = key
        self.update()

    def set_excluded(self, names) -> None:
        self.excluded = set(names)
        self.update()

    def set_selected(self, names) -> None:
        self.selected = set(names)
        self.update()
        self.selection_changed.emit(set(self.selected))

    def set_focus(self, name: Optional[str]) -> None:
        self.focus = name
        self.update()

    # ── geometry ────────────────────────────────────────────────────────

    def _values(self) -> np.ndarray:
        return np.array([float(r.get(self.measure, 0.0) or 0.0) for r in self.rows])

    def _plot(self) -> QRectF:
        return QRectF(44, 8, max(10, self.width() - 54), max(10, self.height() - 26))

    def _range(self):
        v = self._values()
        if not len(v):
            return 0.0, 1.0
        lo, hi = float(v.min()), float(v.max())
        pad = (hi - lo) * 0.08 or max(abs(hi) * 0.1, 1.0)
        return lo - pad, hi + pad

    def _point(self, i: int) -> QPointF:
        r = self._plot()
        lo, hi = self._range()
        n = len(self.rows)
        x = r.left() + (i + 0.5) / max(1, n) * r.width()
        y = r.bottom() - (float(self.rows[i].get(self.measure, 0.0) or 0.0) - lo) / (hi - lo) * r.height()
        return QPointF(x, y)

    def _nearest(self, pos: QPointF) -> Optional[int]:
        if not self.rows:
            return None
        best, which = _HIT + 1.0, None
        # frames are in x order: only look at the ones near the pointer
        n = len(self.rows)
        r = self._plot()
        centre = int((pos.x() - r.left()) / max(1.0, r.width()) * n)
        span = max(2, int(_HIT / max(1.0, r.width() / n)) + 2)
        for i in range(max(0, centre - span), min(n, centre + span + 1)):
            d = float(np.hypot(*(np.array([self._point(i).x(), self._point(i).y()]) - [pos.x(), pos.y()])))
            if d < best:
                best, which = d, i
        return which

    def _inside(self, rect: QRectF) -> Set[str]:
        return {self.rows[i]['name'] for i in range(len(self.rows)) if rect.contains(self._point(i))}

    # ── mouse ───────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.pressed.emit()
            self._press = event.position()
            self._add = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self._band = None

    def mouseMoveEvent(self, event):
        pos = event.position()
        if self._press is not None and (pos - self._press).manhattanLength() > 5:
            self._band = QRectF(self._press, pos).normalized()
            self.update()
            return
        i = self._nearest(pos)
        if i != self._hover:
            self._hover = i
            self.setToolTip(self.describe(i) if i is not None else '')
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or self._press is None:
            return
        if self._band is not None:
            picked = self._inside(self._band)
            self.set_selected((self.selected | picked) if self._add else picked)
        else:
            i = self._nearest(event.position())
            if i is None:
                if not self._add:
                    self.set_selected(set())
            else:
                name = self.rows[i]['name']
                self.set_selected(({*self.selected, name}) if self._add else {name})
                self.focus = name
                self.frame_clicked.emit(name)
        self._press = self._band = None
        self.update()

    def leaveEvent(self, event):
        self._hover = None
        self.update()

    def describe(self, i: int) -> str:
        r = self.rows[i]
        text = (f"{r['name']}\n" + _('score {score}, FWHM {fwhm} px, SNR {snr}, {stars} stars',
                                      score=r.get('score', 0), fwhm=r.get('fwhm', 0),
                                      snr=r.get('snr', 0), stars=r.get('stars', 0)))
        if r['name'] in self.excluded:
            text += '\n' + _('Left out by you')
        elif not r.get('ok', True):
            text += '\n' + (r.get('reason') or _('Rejected by the quality checks'))
        return text

    # ── paint ───────────────────────────────────────────────────────────

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.WELL))
        r = self._plot()
        p.setPen(QColor(theme.LINE))
        p.drawRect(r)
        small = QFont(self.font())
        small.setPointSizeF(7.5)
        p.setFont(small)
        if not self.rows:
            p.setPen(QColor(theme.TEXT_FAINT))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, _('Frames appear here as they are measured.'))
            p.end()
            return
        lo, hi = self._range()
        p.setPen(QColor(theme.TEXT_FAINT))
        for k in range(5):                                    # y grid and labels
            v = lo + (hi - lo) * k / 4
            y = r.bottom() - (v - lo) / (hi - lo) * r.height()
            p.setPen(QColor(theme.LINE_SOFT))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setPen(QColor(theme.TEXT_FAINT))
            p.drawText(QRectF(0, y - 7, 40, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f'{v:.3g}')
        p.drawText(QPointF(r.left(), self.height() - 4), '1')
        p.drawText(QRectF(r.right() - 40, self.height() - 16, 40, 12),
                   Qt.AlignmentFlag.AlignRight, str(len(self.rows)))
        kept = [float(x.get(self.measure, 0.0) or 0.0) for x in self.rows
                if x.get('ok', True) and x['name'] not in self.excluded]
        if kept:                                              # median of the frames that stay
            med = float(np.median(kept))
            y = r.bottom() - (med - lo) / (hi - lo) * r.height()
            p.setPen(QPen(QColor(theme.TEXT_FAINT), 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
        for i, row in enumerate(self.rows):
            pt = self._point(i)
            name = row['name']
            if name in self.excluded:
                p.setPen(QPen(QColor(theme.TEXT_FAINT), 1.6))
                p.drawLine(pt + QPointF(-3.5, -3.5), pt + QPointF(3.5, 3.5))
                p.drawLine(pt + QPointF(-3.5, 3.5), pt + QPointF(3.5, -3.5))
            elif not row.get('ok', True):
                p.setPen(QPen(QColor(theme.BAD), 1.6))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(pt, 3.5, 3.5)
            else:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(theme.ACCENT2))
                p.drawEllipse(pt, 3.0, 3.0)
            if name in self.selected or name == self.focus or i == self._hover:
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(QColor(theme.ACCENT if name in self.selected else theme.TEXT), 1.8))
                p.drawEllipse(pt, 6.5, 6.5)
        if self._band is not None:
            c = QColor(theme.ACCENT)
            c.setAlpha(40)
            p.fillRect(self._band, c)
            p.setPen(QPen(QColor(theme.ACCENT), 1, Qt.PenStyle.DashLine))
            p.drawRect(self._band)
        p.end()


class FramesPanel(QWidget):
    """The graph with its controls, and the banner of a run waiting for the review."""

    frame_clicked = Signal(str)
    excluded_changed = Signal(object)            # set of names left out
    continue_clicked = Signal()
    restack_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._left = 0                         # seconds until the review continues by itself
        self._count_timer = QTimer(self)
        self._count_timer.setInterval(1000)
        self._count_timer.timeout.connect(self.tick)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 0)

        self.banner = QFrame()
        self.banner.setObjectName('reviewBanner')            # styled in theme.stylesheet
        bl = QHBoxLayout(self.banner)
        bl.setContentsMargins(8, 6, 8, 6)
        self._paused_text = _('Paused for your review. Pick the frames you do not want (click, or drag '
                              'a box; Shift adds), choose Exclude, then continue.')
        self.banner_text = QLabel(self._paused_text)
        self.banner_text.setWordWrap(True)
        bl.addWidget(self.banner_text, 1)
        self.continue_btn = QPushButton(_('Continue stacking'))
        set_role(self.continue_btn, 'accent')
        self.continue_btn.clicked.connect(self._continue_now)
        self.stay_btn = QPushButton(_('Stay paused'))
        self.stay_btn.clicked.connect(self.stop_countdown)
        bl.addWidget(self.stay_btn)
        bl.addWidget(self.continue_btn)
        lay.addWidget(self.banner)
        self.banner.hide()

        top = QHBoxLayout()
        top.addWidget(set_role(QLabel(_('Measure')), 'dim'))
        self.metric_combo = NoWheelComboBox()
        for key, label, _better in METRICS:
            self.metric_combo.addItem(_(label), key)
        self.metric_combo.setAccessibleName(_('Measure'))
        top.addWidget(self.metric_combo)
        top.addStretch(1)
        self.count_label = set_role(QLabel(''), 'dim')
        top.addWidget(self.count_label)
        lay.addLayout(top)

        self.graph = FrameGraph()
        lay.addWidget(self.graph)

        bar = QHBoxLayout()
        self.exclude_btn = QPushButton(_('Exclude picked'))
        self.exclude_btn.setToolTip(_('Leave the picked frames out of the stack.'))
        self.include_btn = QPushButton(_('Include picked'))
        self.include_btn.setToolTip(_('Put the picked frames back.'))
        self.clear_btn = QPushButton(_('Clear exclusions'))
        self.restack_btn = QPushButton(_('Stack again without them'))
        self.restack_btn.setToolTip(_('Start a new run that leaves the excluded frames out.'))
        for b in (self.exclude_btn, self.include_btn, self.clear_btn, self.restack_btn):
            bar.addWidget(b)
        bar.addStretch(1)
        lay.addLayout(bar)

        self.metric_combo.activated.connect(lambda _i: self.graph.set_measure(self.metric_combo.currentData()))
        self.graph.frame_clicked.connect(self.frame_clicked.emit)
        self.graph.selection_changed.connect(lambda _s: self._update_buttons())
        self.graph.pressed.connect(self.stop_countdown)
        self.metric_combo.activated.connect(lambda _i: self.stop_countdown())
        self.exclude_btn.clicked.connect(lambda: (self.stop_countdown(), self._change_excluded(add=True)))
        self.include_btn.clicked.connect(lambda: (self.stop_countdown(), self._change_excluded(add=False)))
        self.clear_btn.clicked.connect(lambda: (self.stop_countdown(), self._set_excluded(set())))
        self.restack_btn.clicked.connect(self.restack_clicked.emit)
        self._update_buttons()

    # ── data ────────────────────────────────────────────────────────────

    def set_rows(self, rows: List[dict]) -> None:
        self.graph.set_rows(rows)
        self._update_buttons()

    def set_excluded(self, names) -> None:
        """Show ``names`` as left out (from the form setting); does not announce a change."""
        self.graph.set_excluded(names)
        self._update_buttons()

    def excluded(self) -> Set[str]:
        return set(self.graph.excluded)

    def set_review_mode(self, on: bool, countdown: int = REVIEW_COUNTDOWN_S) -> None:
        """Show the review banner. With ``countdown`` seconds the review continues by itself
        unless the user touches the graph, the buttons or the table (``stop_countdown``)."""
        self.banner.setVisible(on)
        self.restack_btn.setVisible(not on)
        if on and countdown > 0:
            self.start_countdown(countdown)
        else:
            self.stop_countdown()

    # ── the automatic continue ──────────────────────────────────────────

    def start_countdown(self, seconds: int) -> None:
        self._left = int(seconds)
        self.stay_btn.show()
        self._show_countdown()
        self._count_timer.start()

    def stop_countdown(self) -> None:
        """Stay paused until Continue is pressed (any interaction calls this)."""
        was = self._count_timer.isActive()
        self._count_timer.stop()
        self._left = 0
        self.stay_btn.hide()
        if was or self.banner_text.text() != self._paused_text:
            self.banner_text.setText(self._paused_text)

    def counting(self) -> bool:
        return self._count_timer.isActive()

    def tick(self) -> None:
        if not self._count_timer.isActive():
            return                           # stopped: a late tick must not continue the run
        self._left -= 1
        if self._left <= 0:
            self._count_timer.stop()
            self.stay_btn.hide()
            self.continue_clicked.emit()
        else:
            self._show_countdown()

    def _show_countdown(self) -> None:
        self.banner_text.setText(_('Paused for your review. Continuing in {n} s unless you pick or change '
                                   'something.', n=self._left))

    def _continue_now(self) -> None:
        self.stop_countdown()
        self.continue_clicked.emit()

    # ── edits ───────────────────────────────────────────────────────────

    def _change_excluded(self, add: bool) -> None:
        picked = set(self.graph.selected)
        self._set_excluded((self.graph.excluded | picked) if add else (self.graph.excluded - picked))

    def _set_excluded(self, names: Set[str]) -> None:
        self.graph.set_excluded(names)
        self._update_buttons()
        self.excluded_changed.emit(set(names))

    def _update_buttons(self) -> None:
        picked, n = self.graph.selected, len(self.graph.excluded)
        self.exclude_btn.setEnabled(bool(picked - self.graph.excluded))
        self.include_btn.setEnabled(bool(picked & self.graph.excluded))
        self.clear_btn.setEnabled(bool(n))
        self.restack_btn.setEnabled(bool(n))
        total = len(self.graph.rows)
        self.count_label.setText(_('{n} left out of {total}', n=n, total=total) if n else
                                 (_('{total} frames', total=total) if total else ''))


def names_to_setting(names) -> str:
    """The ``--exclude-frames`` text for ``names``."""
    return ','.join(sorted(names))


def setting_to_names(text) -> Set[str]:
    from src.frame_processor import parse_frame_names
    return parse_frame_names(text)

