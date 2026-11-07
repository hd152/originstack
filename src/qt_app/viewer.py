"""The image viewer: smooth zoom and pan (wheel, pinch), 1:1, a draggable before/after wipe, a
minimap while zoomed in, a pixel inspector, and a histogram."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QPolygonF, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.i18n import _
from src.qt_app import theme
from src.qt_app.widgets import qimage_from_bytes, set_role

_MINIMAP_W = 150
_WIPE_GRAB = 8


def image_to_array(img: QImage) -> np.ndarray:
    """(h, w, 3) uint8 view of an RGB888 image (copied, so the QImage may go away)."""
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    buf = np.frombuffer(img.constBits(), dtype=np.uint8, count=bpl * h).reshape(h, bpl)
    return buf[:, :w * 3].reshape(h, w, 3).copy()


def histogram_bins(arr: np.ndarray, bins: int = 64) -> np.ndarray:
    """Per-channel counts, (3, bins), of an 8-bit picture (stride-sampled to ~250 px)."""
    step = max(1, max(arr.shape[0], arr.shape[1]) // 250)
    a = arr[::step, ::step, :3]
    return np.stack([np.histogram(a[..., c], bins=bins, range=(0, 256))[0] for c in range(3)])


class ImageView(QGraphicsView):
    """Zoom with the wheel or a pinch (about the pointer), pan by dragging, an optional wipe
    between two images (left of the line shows A, right shows B; drag the line), a minimap of
    where the view sits once zoomed in, and a pixel read-out under the pointer."""

    zoom_changed = Signal(float)         # percent of the fit scale
    view_changed = Signal()
    wipe_dragged = Signal(float)         # the wipe line was dragged on the picture (0..1)
    hover_pixel = Signal(int, int, object)   # x, y, (r, g, b); x < 0 outside the picture
    key_hold = Signal(str, bool)         # a hold-key (B) went down / up
    detail_wanted = Signal()             # zoomed past the pixels shown: a larger copy would help

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item: Optional[QGraphicsPixmapItem] = None
        self._pix_b: Optional[QPixmap] = None
        self._qimg: Optional[QImage] = None
        self._logical = (0, 0)           # the real image's size; the scene is measured in its pixels
        self._thumb: Optional[QPixmap] = None
        self.compare_on = False
        self.wipe_frac = 0.5
        self.inspect = False
        self.show_minimap = True
        self.empty_hint = ''
        self._fit_scale = 1.0
        self._fitted = True
        self._wipe_drag = False
        self._map_drag = False
        self.setBackgroundBrush(QColor(theme.WELL))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.viewport().grabGesture(Qt.GestureType.PinchGesture)
        self.grabGesture(Qt.GestureType.PinchGesture)
        self.setAccessibleName(_('Picture'))

    # ── images ──────────────────────────────────────────────────────────

    def has_image(self) -> bool:
        return self._item is not None

    def image_size(self):
        """The real image's size in pixels (the picture shown may be a reduced copy of it)."""
        return self._logical

    def item_scale(self) -> float:
        """Scene units (real-image pixels) per pixel of the picture shown: 1 for the full-size
        picture, more for a reduced copy."""
        return 1.0 if self._item is None else float(self._item.scale())

    def set_image(self, img: QImage, logical_size=None) -> None:
        """Show ``img`` as a picture of ``logical_size`` real pixels (default: its own size).
        A reduced copy and the full-size picture of the same image are interchangeable: the
        zoom and position stay. A different image size refits."""
        pm = QPixmap.fromImage(img)
        logical = tuple(logical_size) if logical_size else (pm.width(), pm.height())
        refit = self._item is None or logical != self._logical
        self._qimg = img
        self._logical = logical
        self._thumb = pm.scaledToWidth(_MINIMAP_W, Qt.TransformationMode.SmoothTransformation)
        if self._item is None:
            self._item = self._scene.addPixmap(pm)
        else:
            self._item.setPixmap(pm)
        self._item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._item.setScale(logical[0] / max(1, pm.width()))
        self._scene.setSceneRect(QRectF(0, 0, logical[0], logical[1]))
        if refit:
            self.fit()
        else:
            self.viewport().update()
            self._check_detail()

    def needs_detail(self) -> bool:
        """True when the picture shown is being magnified: each of its pixels covers more than
        one screen pixel, so a larger copy would show more."""
        return self._item is not None and self.scale_factor() * self.item_scale() > 1.05

    def _check_detail(self) -> None:
        if self.needs_detail():
            self.detail_wanted.emit()

    def set_compare_image(self, img: Optional[QImage]) -> None:
        self._pix_b = None if img is None else QPixmap.fromImage(img)
        self.viewport().update()

    def clear(self) -> None:
        self._scene.clear()
        self._item = None
        self._pix_b = None
        self._qimg = self._thumb = None
        self._logical = (0, 0)
        self.compare_on = False
        self.resetTransform()
        self._fit_scale = 1.0
        self.zoom_changed.emit(100.0)
        self.viewport().update()

    # ── view transform ──────────────────────────────────────────────────

    def scale_factor(self) -> float:
        return self.transform().m11()

    def fit(self) -> None:
        if self._item is None:
            return
        self._fitted = True
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._fit_scale = self.scale_factor()
        self._emit_zoom()

    def one_to_one(self) -> None:
        if self._item is None:
            return
        self._fitted = False
        c = self.mapToScene(self.viewport().rect().center())
        self.resetTransform()
        self.centerOn(c)
        self._emit_zoom()

    def _emit_zoom(self) -> None:
        """Report the zoom as a percentage of the real image's pixels (100 = 1:1)."""
        self.zoom_changed.emit(self.scale_factor() * 100)
        self.view_changed.emit()
        self.viewport().update()
        self._check_detail()

    def _zoom_by(self, factor: float) -> None:
        if self._item is None:
            return
        new = max(self._fit_scale * 0.5, min(self.scale_factor() * factor, 40.0))
        factor = new / self.scale_factor()
        self._fitted = False
        self.scale(factor, factor)
        self._emit_zoom()

    def wheelEvent(self, event):
        self._zoom_by(1.15 if event.angleDelta().y() > 0 else 1 / 1.15)

    def event(self, event):
        # Trackpad pinch: Qt delivers it as a gesture (touch screens) or as a native zoom
        # gesture (macOS trackpads); both give a relative change.
        if event.type() == QEvent.Type.Gesture:
            pinch = event.gesture(Qt.GestureType.PinchGesture)
            if pinch is not None:
                last = pinch.lastScaleFactor() or 1.0
                if last:
                    self._zoom_by(pinch.scaleFactor() / last)
                event.accept()
                return True
        elif event.type() == QEvent.Type.NativeGesture:
            if event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                self._zoom_by(1.0 + float(event.value()))
                return True
        return super().event(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._item is not None and self._fitted:
            self.fit()

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        self.view_changed.emit()

    # ── compare ─────────────────────────────────────────────────────────

    def set_compare(self, on: bool) -> None:
        self.compare_on = on
        self.viewport().update()

    def set_wipe(self, frac: float) -> None:
        self.wipe_frac = max(0.0, min(1.0, frac))
        self.viewport().update()

    def _wipe_x(self) -> float:
        return self.wipe_frac * self.viewport().width()

    # ── minimap ─────────────────────────────────────────────────────────

    def _minimap_rect(self) -> Optional[QRectF]:
        """Where the minimap sits (bottom right) while the view is zoomed in past the fit."""
        if (not self.show_minimap or self._thumb is None or self._fit_scale <= 0
                or self.scale_factor() < self._fit_scale * 1.25):
            return None
        w, h = self._thumb.width(), self._thumb.height()
        vp = self.viewport()
        return QRectF(vp.width() - w - 10, vp.height() - h - 10, w, h)

    def _minimap_view(self, mm: QRectF) -> QRectF:
        sr = self._scene.sceneRect()
        vis = self.mapToScene(self.viewport().rect()).boundingRect().intersected(sr)
        sx, sy = mm.width() / sr.width(), mm.height() / sr.height()
        return QRectF(mm.left() + vis.left() * sx, mm.top() + vis.top() * sy,
                      vis.width() * sx, vis.height() * sy)

    def _centre_from_minimap(self, mm: QRectF, pos: QPointF) -> None:
        sr = self._scene.sceneRect()
        x = (pos.x() - mm.left()) / mm.width() * sr.width()
        y = (pos.y() - mm.top()) / mm.height() * sr.height()
        self.centerOn(QPointF(min(max(x, 0), sr.width()), min(max(y, 0), sr.height())))

    # ── mouse and keys ──────────────────────────────────────────────────

    def mousePressEvent(self, event):
        pos = event.position()
        if event.button() == Qt.MouseButton.LeftButton and self._item is not None:
            mm = self._minimap_rect()
            if mm is not None and mm.contains(pos):
                self._map_drag = True
                self._centre_from_minimap(mm, pos)
                return
            if self.compare_on and self._pix_b is not None and abs(pos.x() - self._wipe_x()) <= _WIPE_GRAB:
                self._wipe_drag = True
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = event.position()
        if self._wipe_drag:
            self.set_wipe(pos.x() / max(1, self.viewport().width()))
            self.wipe_dragged.emit(self.wipe_frac)
            return
        if self._map_drag:
            mm = self._minimap_rect()
            if mm is not None:
                self._centre_from_minimap(mm, pos)
            return
        if self.compare_on and self._pix_b is not None and abs(pos.x() - self._wipe_x()) <= _WIPE_GRAB:
            self.viewport().setCursor(Qt.CursorShape.SplitHCursor)
        elif self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag and not event.buttons():
            self.viewport().unsetCursor()
        super().mouseMoveEvent(event)
        if self.inspect:
            self._emit_pixel(pos)

    def mouseReleaseEvent(self, event):
        if self._wipe_drag or self._map_drag:
            self._wipe_drag = self._map_drag = False
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        if self.inspect:
            self.hover_pixel.emit(-1, -1, ())
        super().leaveEvent(event)

    def _emit_pixel(self, pos: QPointF) -> None:
        if self._qimg is None:
            return
        pt = self.mapToScene(pos.toPoint())
        x, y = int(pt.x()), int(pt.y())                      # real-image pixels
        k = self.item_scale()
        sx, sy = int(pt.x() / k), int(pt.y() / k)            # the same spot in the picture shown
        if 0 <= x < self._logical[0] and 0 <= y < self._logical[1] \
                and 0 <= sx < self._qimg.width() and 0 <= sy < self._qimg.height():
            c = self._qimg.pixelColor(sx, sy)
            self.hover_pixel.emit(x, y, (c.red(), c.green(), c.blue()))
        else:
            self.hover_pixel.emit(-1, -1, ())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_B and not event.isAutoRepeat():
            self.key_hold.emit('B', True)
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_B and not event.isAutoRepeat():
            self.key_hold.emit('B', False)
            return
        super().keyReleaseEvent(event)

    # ── paint ───────────────────────────────────────────────────────────

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self.viewport())
        vp = self.viewport().rect()
        if self._item is None:
            p.setPen(QColor(theme.TEXT_FAINT))
            hint = self.empty_hint or _('Waiting for the first stack preview…')
            p.drawText(vp, Qt.AlignmentFlag.AlignCenter, '◆ ORIGINSTACK\n\n' + hint)
            p.end()
            return
        if self.compare_on and self._pix_b is not None:
            x = int(self._wipe_x())
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            p.save()
            p.setClipRect(x, 0, vp.width() - x, vp.height())
            p.setTransform(self.viewportTransform())
            p.drawPixmap(QRectF(0, 0, self._logical[0], self._logical[1]), self._pix_b,
                         QRectF(self._pix_b.rect()))
            p.restore()
            p.setPen(QPen(QColor(theme.ACCENT2), 2))
            p.drawLine(x, 0, x, vp.height())
            p.setBrush(QColor(theme.ACCENT2))                # a grip to show it can be dragged
            p.drawRoundedRect(QRectF(x - 5, vp.height() / 2 - 14, 10, 28), 3, 3)
        mm = self._minimap_rect()
        if mm is not None:
            p.setOpacity(0.9)
            p.fillRect(mm.adjusted(-2, -2, 2, 2), QColor(theme.PANEL))
            p.drawPixmap(mm.toRect(), self._thumb)
            p.setOpacity(1.0)
            p.setPen(QPen(QColor(theme.ACCENT), 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(self._minimap_view(mm))
            p.setPen(QPen(QColor(theme.LINE), 1))
            p.drawRect(mm)
        p.end()


class HistogramView(QWidget):
    """Red/green/blue outlines of the picture shown: where the sky sits and whether anything
    is clipped at either end (the square root keeps the sky peak from flattening the rest)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(64)
        self.setAccessibleName(_('Histogram of the picture shown'))
        self._counts: Optional[np.ndarray] = None

    def set_array(self, arr: Optional[np.ndarray]) -> None:
        self._counts = None if arr is None else np.sqrt(histogram_bins(arr).astype(float))
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(theme.WELL))
        p.setPen(QColor(theme.LINE))
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        if self._counts is not None:
            peak = float(self._counts.max()) or 1.0
            w, h = self.width(), self.height()
            n = self._counts.shape[1]
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            for ch, colour in enumerate(('#ff5a5a', '#5adf7a', '#5a9aff')):
                pts = [QPointF(i * (w - 2) / (n - 1) + 1, h - 2 - self._counts[ch, i] / peak * (h - 6))
                       for i in range(n)]
                p.setPen(QPen(QColor(colour), 1))
                p.drawPolyline(QPolygonF(pts))
        p.end()


class PreviewPane(QWidget):
    """The viewer with its toolbar (Fit, 1:1, zoom, Inspect, Histogram) and caption: the Qt
    counterpart of the tkinter ``PreviewCanvas``, with the same loading API."""

    caption_changed = Signal(str)
    hold_original = Signal(bool)         # B held / released
    wipe_dragged = Signal(float)
    detail_wanted = Signal()            # zoomed past the pixels shown (see ImageView)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.linear_sampler = None      # (x, y) -> {'rgb', 'sigma'} of the linear data, or None
        self.current_slug = ''
        self.compare_slug = ''
        self._source_bytes: Optional[bytes] = None
        self._img_a: Optional[QImage] = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        tools = QHBoxLayout()
        fit = QPushButton(_('Fit'))
        fit.clicked.connect(self.fit)
        one = QPushButton('1:1')
        one.clicked.connect(self.one_to_one)
        self.zoom_label = set_role(QLabel('—'), 'dim')
        self.zoom_label.setMinimumWidth(48)
        self.inspect_box = QCheckBox(_('Pixel'))
        self.inspect_box.setToolTip(_('Show the position and colour of the pixel under the pointer (I)'))
        self.inspect_box.toggled.connect(self._toggle_inspect)
        self.hist_box = QCheckBox(_('Histogram'))
        self.hist_box.toggled.connect(self._toggle_histogram)
        tools.addWidget(fit)
        tools.addWidget(one)
        tools.addWidget(self.zoom_label)
        self.pixel_label = set_role(QLabel(''), 'mono')
        tools.addWidget(self.pixel_label, 1)
        tools.addWidget(self.inspect_box)
        tools.addWidget(self.hist_box)
        lay.addLayout(tools)
        self.view = ImageView()
        self.view.setMinimumHeight(220)
        self.view.zoom_changed.connect(lambda _pct: self.zoom_label.setText(self._zoom_text))
        self.view.hover_pixel.connect(self._on_pixel)
        self.view.key_hold.connect(lambda _k, down: self.hold_original.emit(down))
        self.view.wipe_dragged.connect(self.wipe_dragged.emit)
        self.view.detail_wanted.connect(self.detail_wanted.emit)
        lay.addWidget(self.view, 1)
        self.hist = HistogramView()
        self.hist.hide()
        lay.addWidget(self.hist)
        self.caption_label = set_role(QLabel(_('Waiting for the first stack…')), 'dim')
        lay.addWidget(self.caption_label)
        QShortcut(QKeySequence('I'), self, activated=lambda: self.inspect_box.toggle(),
                  context=Qt.ShortcutContext.WidgetWithChildrenShortcut)

    # ── same surface as the tk PreviewCanvas ────────────────────────────

    @property
    def compare_on(self) -> bool:
        return self.view.compare_on

    def has_image(self) -> bool:
        return self.view.has_image()

    def set_caption(self, text: str) -> None:
        self.caption_label.setText(text)
        self.caption_changed.emit(text)

    def caption(self) -> str:
        return self.caption_label.text()

    def source_bytes(self) -> bytes:
        return self._source_bytes or b''

    def set_empty_hint(self, text: str) -> None:
        self.view.empty_hint = text
        self.view.viewport().update()

    def load_slot(self, jpeg_bytes: bytes, slug: str, caption: str, logical_size=None) -> None:
        """Show a JPEG. ``logical_size``: the real image's pixels when the JPEG is a reduced
        copy (so 1:1 and the zoom percentage mean real pixels)."""
        img = qimage_from_bytes(jpeg_bytes)
        if img is None:
            return
        if logical_size and abs(img.width() / img.height() - logical_size[0] / logical_size[1]) > 0.02:
            logical_size = None                  # not a copy of that image after all
        self._source_bytes = jpeg_bytes
        self._img_a = img
        self.current_slug = slug
        self.view.set_image(img, logical_size)
        self.set_caption(caption)
        self._update_histogram()

    def load_array(self, rgb8: np.ndarray, size) -> None:
        """Swap in a larger copy (uint8 h x w x 3) of the picture shown, of a ``size`` image;
        the caption, zoom and position stay."""
        h, w = rgb8.shape[:2]
        arr = np.ascontiguousarray(rgb8)
        img = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._img_a = img
        self.view.set_image(img, size)
        self.set_detail_status('')

    def set_detail_status(self, text: str) -> None:
        self.zoom_label.setText(self._zoom_text + (f'  {text}' if text else ''))

    @property
    def _zoom_text(self) -> str:
        return f'{round(self.view.scale_factor() * 100)}%' if self.view.has_image() else '—'

    def set_compare_slot(self, jpeg_bytes: Optional[bytes], slug: str) -> None:
        self.compare_slug = slug
        self.view.set_compare_image(qimage_from_bytes(jpeg_bytes) if jpeg_bytes else None)

    def clear(self) -> None:
        self._img_a = None
        self._source_bytes = None
        self.current_slug = self.compare_slug = ''
        self.view.clear()
        self.set_caption(_('Waiting for the first stack…'))
        self.hist.set_array(None)
        self.pixel_label.setText('')

    def fit(self) -> None:
        self.view.fit()

    def one_to_one(self) -> None:
        self.view.one_to_one()

    def set_compare(self, on: bool) -> None:
        self.view.set_compare(on)

    def set_wipe(self, frac: float) -> None:
        self.view.set_wipe(frac)

    def _toggle_inspect(self, on: bool) -> None:
        self.view.inspect = on
        if not on:
            self.pixel_label.setText('')

    def _on_pixel(self, x: int, y: int, rgb) -> None:
        if x < 0:
            self.pixel_label.setText('')
            return
        text = f'x {x}  y {y}   R {rgb[0]:3d}  G {rgb[1]:3d}  B {rgb[2]:3d}'
        sample = self.linear_sampler(x, y) if self.linear_sampler is not None else None
        if sample:
            r, g, b = sample['rgb']
            text += f'   {_("linear")} {r:.4g} {g:.4g} {b:.4g} ({sample["sigma"]:+.1f}σ)'
        self.pixel_label.setText(text)

    def _toggle_histogram(self, on: bool) -> None:
        self.hist.setVisible(on)
        self._update_histogram()

    def _update_histogram(self) -> None:
        if self.hist.isVisible() or self.hist_box.isChecked():
            self.hist.set_array(None if self._img_a is None else image_to_array(self._img_a))
