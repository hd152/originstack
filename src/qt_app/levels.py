"""The levels view of the stretch editor: the linear image's histogram in sky sigmas, the tone
curve the stretch applies, and draggable black, symmetry and white point markers."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from src.i18n import _
from src.qt_app import theme
from src.stretch_editor import AXIS_MIN_SIGMA, from_axis, to_axis

_CHANNELS = ('#ff5a5a', '#5adf7a', '#5a9aff')
_TICKS = (0, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000)
_HANDLE = 6          # grab distance in px
_MARKERS = ('black', 'sym', 'white')


class LevelsView(QWidget):
    """Histogram (red, green, blue, luminance) with the stretch's tone curve over it.

    The x axis is in sky sigmas above the median on an ``asinh`` scale, so the sky peak and the
    long bright tail are both readable; y is the log of the counts (or linear, ``log_y``).
    The black, symmetry and white markers can be dragged: they emit ``black_moved`` /
    ``sym_moved`` / ``white_moved`` with the axis position (``stretch_editor.black_sigma_at`` /
    ``white_pct_at`` turn that into a setting). Hovering reads out the level under the pointer.
    """

    black_moved = Signal(float)
    sym_moved = Signal(float)
    white_moved = Signal(float)
    drag_finished = Signal()
    drag_started = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(130)
        self.setMouseTracking(True)
        self.setAccessibleName(_('Histogram of the linear image with the stretch curve'))
        self.hist: Optional[dict] = None
        self.curve: Optional[tuple] = None
        self.black: Optional[float] = None
        self.white: Optional[float] = None
        self.sym: Optional[float] = None
        self.enabled = {'black': True, 'sym': False, 'white': True}
        self.labels = {'black': '', 'sym': '', 'white': ''}
        self.log_y = True
        self._lum: Optional[np.ndarray] = None
        self._med, self._sigma = 0.0, 1.0
        self._hover_x: Optional[float] = None
        self._drag: Optional[str] = None
        self._lo = float(to_axis(AXIS_MIN_SIGMA))
        self._hi = 1.0

    # ── data ────────────────────────────────────────────────────────────

    def set_data(self, hist: Optional[dict], curve: Optional[tuple], black: Optional[float],
                 white: Optional[float], sym: Optional[float] = None) -> None:
        self.hist, self.curve = hist, curve
        if hist is not None:
            self._lo, self._hi = float(hist['edges'][0]), float(hist['edges'][-1])
        if self._drag is None:               # a marker being dragged follows the pointer
            self.black, self.white, self.sym = black, white, sym
        self.update()

    def set_lum(self, lum: Optional[np.ndarray], med: float, sigma: float) -> None:
        """The luminance behind the histogram (for the hover readout)."""
        if lum is not None and lum.size > 200_000:
            lum = lum.ravel()[::max(1, lum.size // 200_000)]
        self._lum, self._med, self._sigma = lum, med, sigma if sigma > 0 else 1.0

    def set_enabled_markers(self, black: bool, white: bool, sym: bool = False) -> None:
        self.enabled = {'black': black, 'sym': sym, 'white': white}
        self.update()

    def set_marker_labels(self, **labels: str) -> None:
        """Text shown beside a marker (its setting's value), e.g. ``black='1.5σ'``."""
        self.labels.update(labels)
        self.update()

    def set_log(self, on: bool) -> None:
        self.log_y = bool(on)
        self.update()

    def _pos(self, name: str) -> Optional[float]:
        return {'black': self.black, 'sym': self.sym, 'white': self.white}[name]

    # ── geometry ────────────────────────────────────────────────────────

    def _plot(self) -> QRectF:
        return QRectF(8, 6, max(10, self.width() - 16), max(10, self.height() - 24))

    def _x(self, u: float) -> float:
        r = self._plot()
        return r.left() + (u - self._lo) / max(1e-9, self._hi - self._lo) * r.width()

    def _u(self, x: float) -> float:
        r = self._plot()
        return self._lo + (x - r.left()) / max(1e-9, r.width()) * (self._hi - self._lo)

    def _marker_at(self, x: float) -> Optional[str]:
        best, which = _HANDLE + 1.0, None
        for name in _MARKERS:
            u = self._pos(name)
            if u is not None and self.enabled[name]:
                d = abs(self._x(u) - x)
                if d < best:
                    best, which = d, name
        return which

    # ── mouse ───────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = self._marker_at(event.position().x())
            if self._drag:
                self.drag_started.emit()
                self._move_to(event.position().x())

    def mouseMoveEvent(self, event):
        self._hover_x = event.position().x()
        if self._drag:
            self._move_to(event.position().x())
        else:
            self.setCursor(Qt.CursorShape.SizeHorCursor if self._marker_at(event.position().x())
                           else Qt.CursorShape.ArrowCursor)
            self.update()

    def leaveEvent(self, event):
        self._hover_x = None
        self.update()

    def mouseReleaseEvent(self, event):
        if self._drag:
            self._drag = None
            self.drag_finished.emit()

    def _move_to(self, x: float) -> None:
        u = min(self._hi, max(self._lo, self._u(x)))
        if self._drag == 'black':
            if self.white is not None:
                u = min(u, self.white)
            self.black = u
            self.black_moved.emit(u)
        elif self._drag == 'sym':
            lo = self.black if self.black is not None else self._lo
            hi = self.white if self.white is not None else self._hi
            u = min(max(u, lo), hi)
            self.sym = u
            self.sym_moved.emit(u)
        else:
            if self.black is not None:
                u = max(u, self.black)
            self.white = u
            self.white_moved.emit(u)
        self.update()

    def readout(self) -> str:
        """The level under the pointer: sigmas above the sky and the share of pixels darker."""
        if self._hover_x is None or self._lum is None or self.hist is None:
            return ''
        u = min(self._hi, max(self._lo, self._u(self._hover_x)))
        sig = float(from_axis(u))
        value = self._med + sig * self._sigma
        pct = 100.0 * float(np.mean(self._lum < value))
        return _('{sigma:.1f}σ above the sky · {pct:.2f}% of pixels darker', sigma=sig, pct=pct)

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
        for t in _TICKS:                                   # sigma ticks
            u = float(to_axis(t))
            if self._lo < u < self._hi:
                x = self._x(u)
                p.setPen(QColor(theme.LINE_SOFT))
                p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
                p.setPen(QColor(theme.TEXT_FAINT))
                p.drawText(QPointF(x + 2, r.bottom() + 12), f'{t}σ')
        if self.hist is None:
            p.setPen(QColor(theme.TEXT_FAINT))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, _('Waiting for the stack…'))
            p.end()
            return
        edges = self.hist['edges']
        counts = self.hist['counts'] if self.log_y or 'linear' not in self.hist else self.hist['linear']
        centres = 0.5 * (edges[:-1] + edges[1:])
        xs = np.array([self._x(u) for u in centres])
        for ch in (0, 1, 2, 3):
            colour = QColor(_CHANNELS[ch]) if ch < 3 else QColor(theme.TEXT)
            pts = [QPointF(x, r.bottom() - c * (r.height() - 2)) for x, c in zip(xs, counts[ch])]
            if ch < 3:
                path = QPainterPath(QPointF(xs[0], r.bottom()))
                for pt in pts:
                    path.lineTo(pt)
                path.lineTo(QPointF(xs[-1], r.bottom()))
                fill = QColor(colour)
                fill.setAlpha(40)
                p.fillPath(path, fill)
            pen = QPen(colour, 1.0 if ch < 3 else 1.4)
            if ch == 3:
                colour.setAlpha(170)
                pen.setColor(colour)
            p.setPen(pen)
            p.drawPolyline(QPolygonF(pts))
        if self.curve is not None and len(self.curve[0]):      # tone curve, output 0..1
            p.setPen(QPen(QColor(theme.ACCENT3), 2))
            p.drawPolyline(QPolygonF([QPointF(self._x(u), r.bottom() - float(y) * r.height())
                                      for u, y in zip(*self.curve)]))
        colours = {'black': theme.MARK_BLACK, 'sym': theme.ACCENT2, 'white': theme.TEXT}
        for name in _MARKERS:
            u = self._pos(name)
            if u is None or (name == 'sym' and not self.enabled['sym']):
                continue
            on = self.enabled[name]
            x = self._x(u)
            c = QColor(colours[name] if on else theme.TEXT_FAINT)
            p.setPen(QPen(c, 1.5, Qt.PenStyle.SolidLine if on else Qt.PenStyle.DashLine))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            if on:
                p.setBrush(c)
                p.drawPolygon(QPolygonF([QPointF(x - 5, r.bottom() + 1), QPointF(x + 5, r.bottom() + 1),
                                         QPointF(x, r.bottom() - 7)]))
                p.setBrush(Qt.BrushStyle.NoBrush)
            text = self.labels.get(name, '')
            if text and on:
                w = p.fontMetrics().horizontalAdvance(text) + 6
                bx = min(max(x + 4, r.left()), r.right() - w)
                box = QRectF(bx, r.top() + 2 + (14 if name == 'sym' else 0), w, 14)
                bg = QColor(theme.PANEL)
                bg.setAlpha(220)
                p.fillRect(box, bg)
                p.setPen(c)
                p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
        text = self.readout()
        if text:
            p.setPen(QColor(theme.TEXT_DIM))
            p.drawText(QRectF(r.left() + 4, r.bottom() - 16, r.width() - 8, 14),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
        p.end()
