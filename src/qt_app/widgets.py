"""Small shared widgets and image helpers for the Qt window."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QComboBox, QFrame, QLabel, QSlider, QSpinBox, QWidget


def set_role(widget: QWidget, role: str, **props: str) -> QWidget:
    """Tag ``widget`` for the stylesheet (``QLabel[role="dim"]``) and return it."""
    widget.setProperty('role', role)
    for k, v in props.items():
        widget.setProperty(k, v)
    return widget


def repolish(widget: QWidget) -> None:
    """Re-apply the stylesheet after a dynamic property changed."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def label(text: str = '', role: str = '', wrap: bool = False) -> QLabel:
    lbl = QLabel(text)
    if role:
        set_role(lbl, role)
    if wrap:
        lbl.setWordWrap(True)
    return lbl


def qimage_from_bytes(data: bytes) -> Optional[QImage]:
    """Decode a JPEG/PNG byte string; None when it is not an image."""
    img = QImage()
    if not data or not img.loadFromData(data):
        return None
    return img.convertToFormat(QImage.Format.Format_RGB888)


class NoWheelComboBox(QComboBox):
    """A combo box that ignores the wheel unless focused: inside a scrolling form the wheel is
    meant to scroll, not to change a setting under the pointer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class NoWheelSpinBox(QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class ClickableLabel(QLabel):
    """A label that emits ``clicked`` (header links, the reset link)."""
    clicked = Signal()

    def __init__(self, text: str = '', parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class ClickableFrame(QFrame):
    """A frame that emits ``clicked`` and can be activated from the keyboard (the target cards)."""
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.clicked.emit()
        else:
            super().keyPressEvent(event)


class PixmapLabel(QWidget):
    """A picture scaled down (never up) to the width it is given, aspect kept: the target-card
    thumbnails must not set the minimum width of the Setup column."""

    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self._pm = pixmap
        self.setMinimumSize(48, 30)

    def sizeHint(self) -> QSize:
        return self._pm.size()

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, w: int) -> int:
        return int(min(w, self._pm.width()) * self._pm.height() / max(1, self._pm.width()))

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        w = min(self.width(), self._pm.width())
        h = int(w * self._pm.height() / max(1, self._pm.width()))
        if h > self.height():
            h = self.height()
            w = int(h * self._pm.width() / max(1, self._pm.height()))
        p.drawPixmap((self.width() - w) // 2, 0, w, h, self._pm)
        p.end()


class HSlider(QSlider):
    """Horizontal slider reporting a float in [0, 1] (the compare wipe)."""
    moved = Signal(float)

    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setRange(0, 1000)
        self.setValue(500)
        self.valueChanged.connect(lambda v: self.moved.emit(v / 1000.0))

    def fraction(self) -> float:
        return self.value() / 1000.0

    def set_fraction(self, f: float) -> None:
        self.setValue(int(round(max(0.0, min(1.0, f)) * 1000)))


def card_pixmap(example_path, key: str, size=(132, 84)) -> QPixmap:
    """Thumbnail of a target card: its example result centre-cropped and scaled, or a small
    drawing for the cards without one."""
    w, h = size
    if example_path is not None and example_path.exists():
        src = QPixmap(str(example_path))
        if not src.isNull():
            cw = int(src.width() * 0.7)
            ch = int(cw * h / w)
            crop = src.copy((src.width() - cw) // 2, (src.height() - ch) // 2, cw, ch)
            return crop.scaled(w, h, Qt.AspectRatioMode.IgnoreAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
    import random
    pm = QPixmap(w, h)
    pm.fill(QColor(6, 7, 10))
    p = QPainter(pm)
    rnd = random.Random(7)
    for _i in range(50):
        v = rnd.randrange(60, 190)
        p.setPen(QColor(v, v, v))
        p.drawPoint(rnd.randrange(w), rnd.randrange(h))
    if key == 'cluster':
        p.setPen(Qt.PenStyle.NoPen)
        for r, a in ((16, 20), (9, 40), (4, 70)):
            p.setBrush(QColor(255, 240, 210, a))
            p.drawEllipse(QRectF(w / 2 - r, h / 2 - r, 2 * r, 2 * r))
        for _i in range(260):
            x, y = rnd.gauss(w / 2, w / 9), rnd.gauss(h / 2, h / 7)
            v = rnd.randrange(150, 255)
            p.setPen(QColor(v, v, int(v * 0.9)))
            p.drawPoint(int(x), int(y))
    else:
        p.setPen(QColor(232, 230, 223))
        p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, 'AUTO')
    p.end()
    return pm
