"""Best-effort native OS notification for the desktop app.

A ~20-line ctypes wrapper around Shell_NotifyIcon's balloon-tip API rather
than a dependency like plyer: shell32.dll is present on every Windows
install (zero new pip dependency, zero new PyInstaller-collection risk),
matching this project's existing preference for small native/stdlib
implementations over dependency weight (see requirements.txt's notes on
dropping cv2/skimage/astroquery/PyWavelets). Legacy balloon-tip style, not a
modern Action Center toast -- an accepted v1 tradeoff.

Only meaningful on Windows; a no-op (silently False) everywhere else, and
any failure (no desktop session, shell32 unavailable, etc.) is swallowed --
this is cosmetic and must never affect run state.
"""
from __future__ import annotations

import sys
import ctypes
from ctypes import wintypes

NIM_ADD = 0x00000000
NIM_DELETE = 0x00000002
NIF_INFO = 0x00000010
NIIF_INFO = 0x00000001


class _NOTIFYICONDATA(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("hWnd", wintypes.HWND),
        ("uID", wintypes.UINT),
        ("uFlags", wintypes.UINT),
        ("uCallbackMessage", wintypes.UINT),
        ("hIcon", wintypes.HANDLE),
        ("szTip", ctypes.c_wchar * 128),
        ("dwState", wintypes.DWORD),
        ("dwStateMask", wintypes.DWORD),
        ("szInfo", ctypes.c_wchar * 256),
        ("uTimeoutOrVersion", wintypes.UINT),
        ("szInfoTitle", ctypes.c_wchar * 64),
        ("dwInfoFlags", wintypes.DWORD),
    ]


def notify_windows(title: str, message: str) -> bool:
    """Show a Windows balloon-tip notification. Returns True on apparent
    success, False otherwise (including on any non-Windows platform)."""
    if sys.platform != "win32":
        return False
    try:
        shell32 = ctypes.windll.shell32
        nid = _NOTIFYICONDATA()
        nid.cbSize = ctypes.sizeof(_NOTIFYICONDATA)
        nid.hWnd = None
        nid.uID = 0
        nid.uFlags = NIF_INFO
        nid.szInfo = message[:255]
        nid.szInfoTitle = title[:63]
        nid.dwInfoFlags = NIIF_INFO
        nid.uTimeoutOrVersion = 10000  # ms, ignored on modern Windows (OS-controlled)
        added = shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(nid))
        if not added:
            return False
        # Immediately request removal -- the balloon itself has already been
        # queued for display by NIM_ADD; there is no persistent tray icon to
        # manage since this app has no long-lived tray presence.
        shell32.Shell_NotifyIconW(NIM_DELETE, ctypes.byref(nid))
        return True
    except Exception:
        return False
