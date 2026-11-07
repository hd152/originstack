"""The keyboard shortcuts, as one list the Help dialog and the menus share, and the dialog."""
from __future__ import annotations

from PySide6.QtWidgets import QDialog, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from src.i18n import N_, _
from src.qt_app.widgets import set_role

# (keys, what it does, where it works); the text is translated where shown.
SHORTCUTS = (
    ('Ctrl+O', N_('Choose the light-frames folder'), N_('Anywhere')),
    ('Ctrl+Return', N_('Start the run'), N_('Anywhere')),
    ('Esc', N_('Cancel the run'), N_('While running')),
    ('Ctrl+T', N_('Show or hide the tools under the picture'), N_('Anywhere')),
    ('Ctrl+M', N_('Maximise the picture (hide the tools)'), N_('Anywhere')),
    ('F', N_('Fit the picture to the window'), N_('Picture')),
    ('1', N_('Show the picture at 100%'), N_('Picture')),
    ('C', N_('Compare with another milestone'), N_('Picture')),
    ('I', N_('Show the pixel under the pointer'), N_('Picture')),
    ('B', N_('Hold to see the stretch the run used'), N_('Picture')),
    ('Ctrl+Z', N_('Undo the last stretch change'), N_('Stretch editor')),
    ('Ctrl+Y', N_('Redo it'), N_('Stretch editor')),
    ('Double-click', N_('Reset a stretch slider to the run\'s value'), N_('Stretch editor')),
)


class ShortcutsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(_('Keyboard shortcuts'))
        lay = QVBoxLayout(self)
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        for i, (keys, what, where) in enumerate(SHORTCUTS):
            k = QLabel(keys)
            k.setStyleSheet('font-weight: bold;')
            grid.addWidget(k, i, 0)
            grid.addWidget(QLabel(_(what)), i, 1)
            grid.addWidget(set_role(QLabel(_(where)), 'dim'), i, 2)
        lay.addLayout(grid)
        row = QHBoxLayout()
        row.addStretch(1)
        close = QPushButton(_('Close'))
        close.clicked.connect(self.accept)
        row.addWidget(close)
        lay.addLayout(row)
