"""Colours, fonts and the application stylesheet of the Qt window.

Two palettes (dark sky and light). ``apply_mode`` rebinds the module-level colour names, so
code that reads ``theme.ACCENT`` while building widgets gets the active palette; the mode is
chosen before the window is built (a change applies at the next start, like the language).
Every text colour clears WCAG AA (4.5:1) on its background in both palettes
(``tests/test_qt_app.py::TestTheme``).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

DARK = dict(
    BG='#08090c', PANEL='#0f1117', WELL='#030304', LINE='#23262f', LINE_SOFT='#1a1c22',
    TEXT='#e8e6df', TEXT_DIM='#9298aa', TEXT_FAINT='#7f8598',
    ACCENT='#ff6a3d', ACCENT_TEXT='#1a0a04', ACCENT2='#3dd6e0', ACCENT3='#ffb454',
    LOG_FG='#ffb454', SUCCESS='#5cd98f', BAD='#ff4757', MARK_BLACK='#9aa0b0',
)
LIGHT = dict(
    BG='#f4f5f8', PANEL='#ffffff', WELL='#fbfbfd', LINE='#c9ccd6', LINE_SOFT='#e3e5ec',
    TEXT='#1c1e26', TEXT_DIM='#4f5566', TEXT_FAINT='#666c7e',
    ACCENT='#c2410c', ACCENT_TEXT='#ffffff', ACCENT2='#0b6b7a', ACCENT3='#9a5b00',
    LOG_FG='#7a4300', SUCCESS='#18794e', BAD='#c4202f', MARK_BLACK='#4f5566',
)
PALETTES = {'dark': DARK, 'light': LIGHT}

# Active palette (rebound by apply_mode; dark until then).
BG = PANEL = WELL = LINE = LINE_SOFT = TEXT = TEXT_DIM = TEXT_FAINT = ''
ACCENT = ACCENT_TEXT = ACCENT2 = ACCENT3 = LOG_FG = SUCCESS = BAD = MARK_BLACK = ''
MODE = 'dark'

SANS = 'Segoe UI' if sys.platform == 'win32' else 'DejaVu Sans'
MONO = 'Consolas' if sys.platform == 'win32' else 'DejaVu Sans Mono'
_CJK_SANS = {'ja': 'Yu Gothic UI', 'zh_CN': 'Microsoft YaHei UI'}


def apply_mode(mode: str) -> str:
    """Make ``mode`` ('dark' or 'light') the active palette; returns the mode in force."""
    global MODE
    MODE = mode if mode in PALETTES else 'dark'
    globals().update(PALETTES[MODE])
    return MODE


apply_mode('dark')


def resolve_mode(choice: str) -> str:
    """'dark' / 'light' as given; 'auto' follows the system colour scheme."""
    if choice in PALETTES:
        return choice
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QGuiApplication
        scheme = QGuiApplication.styleHints().colorScheme()
        return 'light' if scheme == Qt.ColorScheme.Light else 'dark'
    except Exception:
        return 'dark'


def sans_for(language: str) -> str:
    """The UI font family for ``language`` (CJK languages need a face that has the glyphs)."""
    return _CJK_SANS.get(language, SANS)


def contrast(fg: str, bg: str) -> float:
    """WCAG contrast ratio of two ``#rrggbb`` colours."""
    def lum(h: str) -> float:
        h = h.lstrip('#')
        c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    hi, lo = sorted((lum(fg), lum(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def make_icons() -> dict:
    """Draw the small glyphs the stylesheet refers to (combo arrow, check mark) in the active
    colours and return their folder-relative URLs. Qt style sheets can only point at image
    files; drawing them here avoids shipping data files."""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
    folder = Path(tempfile.gettempdir()) / f'originstack_qt_{MODE}'
    folder.mkdir(parents=True, exist_ok=True)

    def canvas(size=16):
        img = QImage(size, size, QImage.Format.Format_ARGB32)
        img.fill(Qt.GlobalColor.transparent)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        return img, p

    img, p = canvas()
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(TEXT_DIM))
    p.drawPolygon(QPolygonF([QPointF(4, 6), QPointF(12, 6), QPointF(8, 11)]))
    p.end()
    img.save(str(folder / 'down.png'))
    img, p = canvas()
    p.setPen(QPen(QColor(ACCENT_TEXT), 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                  Qt.PenJoinStyle.RoundJoin))
    p.drawPolyline(QPolygonF([QPointF(3.5, 8.5), QPointF(6.5, 11.5), QPointF(12.5, 4.5)]))
    p.end()
    img.save(str(folder / 'check.png'))
    return {'down': (folder / 'down.png').as_posix(), 'check': (folder / 'check.png').as_posix()}


def stylesheet(sans: str = SANS) -> str:
    ic = make_icons()
    return f'''
* {{ font-family: "{sans}"; font-size: 9pt; }}
QWidget {{ background: {BG}; color: {TEXT}; }}
QMainWindow, QDialog {{ background: {BG}; }}
QLabel {{ background: transparent; }}
QLabel[role="dim"] {{ color: {TEXT_DIM}; }}
QLabel[role="hint"] {{ color: {TEXT_FAINT}; font-size: 8pt; }}
QLabel[role="changed"] {{ color: {ACCENT}; }}
QLabel[role="reset"] {{ color: {ACCENT2}; text-decoration: underline; }}
QLabel[role="section"] {{ color: {TEXT_DIM}; font-weight: bold; letter-spacing: 1px; }}
QLabel[role="link"] {{ color: {TEXT_DIM}; }}
QLabel[role="link"]:hover {{ color: {ACCENT}; }}
QLabel[role="update"] {{ color: {ACCENT}; }}
QLabel[role="mono"] {{ font-family: "{MONO}"; }}
QFrame#header {{ background: {PANEL}; border-bottom: 1px solid {LINE}; }}
QFrame#header QLabel {{ background: transparent; }}
QMenuBar {{ background: {PANEL}; color: {TEXT}; border-bottom: 1px solid {LINE}; }}
QMenuBar::item {{ background: transparent; padding: 4px 10px; }}
QMenuBar::item:selected {{ background: {LINE_SOFT}; }}
QMenu {{ background: {PANEL}; border: 1px solid {LINE}; padding: 4px 0; }}
QMenu::item {{ padding: 5px 26px 5px 22px; }}
QMenu::item:selected {{ background: {LINE_SOFT}; }}
QMenu::separator {{ height: 1px; background: {LINE}; margin: 4px 8px; }}
QGroupBox {{ border: 1px solid {LINE}; border-radius: 3px; margin-top: 14px; padding: 8px 6px 6px 6px;
             background: {PANEL}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {TEXT_DIM};
                    font-weight: bold; }}
QGroupBox QLabel, QGroupBox QCheckBox {{ background: transparent; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ background: {WELL}; border: 1px solid {LINE};
    border-radius: 2px; padding: 3px 5px; selection-background-color: {ACCENT};
    selection-color: {ACCENT_TEXT}; }}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {TEXT_FAINT}; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {ACCENT2}; }}
QComboBox QAbstractItemView {{ background: {PANEL}; selection-background-color: {LINE_SOFT};
    selection-color: {TEXT}; border: 1px solid {LINE}; outline: 0; }}
QComboBox::drop-down {{ border: none; width: 22px; subcontrol-origin: padding;
    subcontrol-position: center right; }}
QComboBox::down-arrow {{ image: url({ic['down']}); width: 16px; height: 16px; }}
QComboBox:disabled {{ color: {TEXT_FAINT}; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 0; border: none; }}
QPushButton {{ background: {LINE_SOFT}; border: 1px solid {LINE}; border-radius: 2px;
    padding: 4px 12px; }}
QPushButton:hover {{ border-color: {TEXT_FAINT}; }}
QPushButton:pressed {{ background: {LINE}; }}
QPushButton:focus {{ border: 1px solid {ACCENT2}; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; }}
QPushButton[role="accent"] {{ background: {ACCENT}; color: {ACCENT_TEXT}; border-color: {ACCENT};
    font-weight: bold; }}
QPushButton[role="accent"]:focus {{ border: 2px solid {ACCENT2}; }}
QPushButton[role="accent"]:disabled {{ background: {LINE_SOFT}; color: {TEXT_FAINT};
    border-color: {LINE}; }}
QPushButton[role="flat"] {{ background: transparent; border: none; color: {TEXT_DIM}; }}
QPushButton[role="flat"]:hover {{ color: {TEXT}; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 2px;
    padding: 2px 6px; color: {TEXT_DIM}; }}
QToolButton:hover {{ border-color: {LINE}; color: {TEXT}; }}
QToolButton:checked {{ background: {LINE_SOFT}; color: {TEXT}; border-color: {LINE}; }}
QCheckBox {{ spacing: 6px; background: transparent; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {TEXT_FAINT};
    background: {WELL}; border-radius: 2px; }}
QCheckBox::indicator:hover {{ border-color: {TEXT_DIM}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT};
    image: url({ic['check']}); }}
QCheckBox::indicator:disabled {{ border-color: {LINE}; }}
QCheckBox:focus {{ color: {TEXT}; }}
QCheckBox:disabled {{ color: {TEXT_FAINT}; }}
QPlainTextEdit#log {{ background: {WELL}; color: {LOG_FG}; border: 1px solid {LINE};
    font-family: "{MONO}"; }}
QTableWidget {{ background: {WELL}; gridline-color: {LINE_SOFT}; border: 1px solid {LINE};
    alternate-background-color: {PANEL}; }}
QHeaderView::section {{ background: {PANEL}; color: {TEXT_DIM}; border: none;
    border-right: 1px solid {LINE}; padding: 3px 6px; }}
QTableWidget::item:selected {{ background: {LINE}; color: {TEXT}; }}
QListWidget {{ background: {PANEL}; border: none; outline: 0; }}
QListWidget::item {{ padding: 8px 12px; color: {TEXT_DIM}; }}
QListWidget::item:hover {{ color: {TEXT}; }}
QListWidget::item:selected {{ background: {LINE_SOFT}; color: {TEXT}; border-left: 2px solid {ACCENT}; }}
QTabWidget::pane {{ border: 1px solid {LINE}; border-top: none; background: {PANEL}; }}
QTabBar::tab {{ background: {BG}; color: {TEXT_DIM}; padding: 6px 14px; border: 1px solid {LINE};
    border-bottom: none; margin-right: 2px; border-top-left-radius: 3px; border-top-right-radius: 3px; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{ background: {PANEL}; color: {TEXT}; border-top: 2px solid {ACCENT}; }}
QProgressBar {{ background: {WELL}; border: 1px solid {LINE}; border-radius: 2px; height: 8px;
    text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {ACCENT}; }}
QScrollArea {{ border: none; }}
QScrollBar:vertical {{ background: {BG}; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {LINE}; min-height: 24px; border-radius: 4px; }}
QScrollBar::handle:vertical:hover {{ background: {TEXT_FAINT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ background: {BG}; height: 10px; }}
QScrollBar::handle:horizontal {{ background: {LINE}; min-width: 24px; border-radius: 4px; }}
QSplitter::handle {{ background: {LINE_SOFT}; }}
QSplitter::handle:hover {{ background: {ACCENT2}; }}
QSplitter::handle:vertical {{ height: 5px; }}
QSplitter::handle:horizontal {{ width: 5px; }}
QSlider::groove:horizontal {{ background: {LINE}; height: 4px; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {TEXT}; border: 2px solid {ACCENT}; width: 10px;
    margin: -6px 0; border-radius: 7px; }}
QSlider::handle:horizontal:hover {{ background: {ACCENT}; }}
QSlider::groove:horizontal:disabled {{ background: {LINE_SOFT}; }}
QSlider::sub-page:horizontal:disabled {{ background: {TEXT_FAINT}; }}
QSlider::handle:horizontal:disabled {{ background: {LINE}; border-color: {TEXT_FAINT}; }}
QToolTip {{ background: {PANEL}; color: {TEXT}; border: 1px solid {LINE}; padding: 4px; }}
QFrame[role="card"] {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 3px; }}
QFrame[role="card"]:hover {{ border-color: {TEXT_FAINT}; }}
QFrame[role="card"]:focus {{ border: 2px solid {ACCENT2}; }}
QFrame[role="card"][selected="true"] {{ border: 2px solid {ACCENT}; }}
QFrame[role="card"] QLabel {{ background: transparent; }}
QLabel[role="phase"] {{ background: {PANEL}; color: {TEXT_FAINT}; border: 1px solid {LINE};
    padding: 4px; }}
QLabel[role="phase"][state="active"] {{ color: {TEXT}; border-color: {ACCENT}; }}
QLabel[role="phase"][state="done"] {{ color: {SUCCESS}; border-color: {SUCCESS}; }}
QFrame#reviewBanner {{ background: {LINE_SOFT}; border: 1px solid {ACCENT}; border-radius: 3px; }}
QFrame#reviewBanner QLabel {{ border: none; background: transparent; }}
QLabel[role="badge"] {{ background: {ACCENT2}; color: {ACCENT_TEXT if MODE == 'light' else '#04161a'};
    font-size: 7pt; font-weight: bold; padding: 0 4px; }}
'''
