"""Dropping a folder on the window (src/drop_files.py)."""
import sys

import pytest

from src.drop_files import enable_file_drop, folder_from_drop


def test_a_dropped_folder_wins_over_files(tmp_path):
    (tmp_path / 'lights').mkdir()
    (tmp_path / 'a.fits').write_bytes(b'')
    assert folder_from_drop([str(tmp_path / 'a.fits'), str(tmp_path / 'lights')]) == str(tmp_path / 'lights')


def test_a_dropped_file_means_the_folder_holding_it(tmp_path):
    (tmp_path / 'Light0001.fits').write_bytes(b'')
    assert folder_from_drop([str(tmp_path / 'Light0001.fits')]) == str(tmp_path)


def test_nothing_usable_means_no_folder(tmp_path):
    assert folder_from_drop([]) is None
    assert folder_from_drop([str(tmp_path / 'gone.fits')]) is None


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows shell drop')
def test_a_real_wm_dropfiles_reaches_the_callback(tk_root, tmp_path):
    import ctypes
    from ctypes import wintypes
    got = []
    assert enable_file_drop(tk_root, got.append)
    hwnd = ctypes.windll.user32.GetParent(tk_root.winfo_id()) or tk_root.winfo_id()

    # An HDROP: a DROPFILES header (20 bytes, wide chars) then the paths, each NUL-ended, and a
    # final NUL.
    names = [str(tmp_path), str(tmp_path / 'b.fits')]
    body = ''.join(n + '\0' for n in names) + '\0'
    data = body.encode('utf-16-le')
    k32 = ctypes.windll.kernel32
    k32.GlobalAlloc.restype = wintypes.HGLOBAL
    k32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    k32.GlobalLock.restype = ctypes.c_void_p
    k32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    k32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    h = k32.GlobalAlloc(0x0042, 20 + len(data))            # GMEM_MOVEABLE | GMEM_ZEROINIT
    ptr = k32.GlobalLock(h)
    ctypes.memmove(ptr, (20).to_bytes(4, 'little') + bytes(12) + (1).to_bytes(4, 'little'), 20)
    ctypes.memmove(ptr + 20, data, len(data))
    k32.GlobalUnlock(h)
    user32 = ctypes.windll.user32
    user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.SendMessageW(hwnd, 0x0233, h, 0)
    tk_root.update()                                       # runs the after_idle hand-over
    assert got == [names]


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows shell drop')
def test_hooking_a_window_twice_keeps_the_first_procedure_alive(tk_root):
    """The second hook chains to the first one's thunk; losing that thunk crashed the next message
    (an access violation in the desktop-app tests, which build several apps on one root)."""
    import ctypes
    import gc
    from ctypes import wintypes

    from src import drop_files
    assert enable_file_drop(tk_root, lambda paths: None)
    assert enable_file_drop(tk_root, lambda paths: None)
    hwnd = ctypes.windll.user32.GetParent(tk_root.winfo_id()) or tk_root.winfo_id()
    assert len(drop_files._keep[hwnd]) >= 2
    gc.collect()
    user32 = ctypes.windll.user32
    user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.SendMessageW(hwnd, 0x0000, 0, 0)    # WM_NULL goes down the whole chain
    tk_root.update()
