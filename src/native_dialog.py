"""Native Windows message boxes via ctypes -- no bundled GUI toolkit needed.

desktop_app.py previously used tkinter for its two dialogs (fatal-error
popup, close-confirmation), which pulls the entire Tcl/Tk runtime into a
packaged build for a couple of message boxes (~7-8MB of the exe's payload).
user32.dll's MessageBoxW is already on every Windows install and does the
same job in one call -- same "small ctypes wrapper over dependency weight"
choice src/notify.py already made for OS notifications.

Only meaningful on Windows; every function no-ops (returns False) elsewhere.
"""
from __future__ import annotations

import ctypes
import sys

MB_OK = 0x00000000
MB_YESNO = 0x00000004
MB_ICONERROR = 0x00000010
MB_ICONQUESTION = 0x00000020
MB_TOPMOST = 0x00040000
IDYES = 6


def show_error(title: str, message: str) -> bool:
    """Blocking native error dialog. Returns True if it was shown."""
    if sys.platform != "win32":
        return False
    try:
        ctypes.windll.user32.MessageBoxW(
            None, message, title, MB_OK | MB_ICONERROR | MB_TOPMOST)
        return True
    except Exception:
        return False


def ask_yes_no(title: str, message: str, default: bool = True) -> bool:
    """Blocking native Yes/No dialog. Returns the user's choice, or
    *default* if the dialog itself could not be shown (never trap the user
    behind a broken confirmation)."""
    if sys.platform != "win32":
        return default
    try:
        result = ctypes.windll.user32.MessageBoxW(
            None, message, title, MB_YESNO | MB_ICONQUESTION | MB_TOPMOST)
        return result == IDYES
    except Exception:
        return default
