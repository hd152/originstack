"""Global temporary-file registry: auto-remove all registered paths on exit or interrupt."""
from __future__ import annotations

import atexit
import gc
import os
import shutil
import threading
from typing import List

_lock: threading.Lock = threading.Lock()
_paths: List[str] = []


def register(path: str) -> None:
    """Register *path* for deletion when the process exits (including Ctrl+C)."""
    with _lock:
        if path not in _paths:
            _paths.append(path)


def deregister(path: str) -> None:
    """Remove *path* from the cleanup registry after a successful manual deletion."""
    with _lock:
        try:
            _paths.remove(path)
        except ValueError:
            pass


def _do_cleanup() -> None:
    """atexit handler: delete every registered path still on disk (closing any open
    frame store first: Windows cannot delete a file that is still memory-mapped)."""
    try:
        from src.frame_store import cleanup_all_stores
        cleanup_all_stores()
    except Exception:
        pass
    gc.collect()
    with _lock:
        remaining = list(_paths)
        _paths.clear()
    for p in remaining:
        try:
            if os.path.isfile(p):
                os.remove(p)
            elif os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass


atexit.register(_do_cleanup)


def cleanup_now() -> None:
    """Delete every registered path now and close any frame store still open.

    For the end of each run, success or not. The desktop app keeps one process
    for many runs, so the at-exit handler alone left a failed run's temp files
    (60 GB of aligned frames on an 850-light session) until the app was closed --
    and never, if it was killed. Paths that cannot be removed yet stay
    registered for the at-exit pass."""
    try:
        from src.frame_store import cleanup_all_stores
        cleanup_all_stores()
    except Exception:
        pass
    gc.collect()
    with _lock:
        remaining = list(_paths)
    for p in remaining:
        try:
            if os.path.isfile(p):
                os.remove(p)
            elif os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            if not os.path.exists(p):
                deregister(p)
        except Exception:
            pass


# Temp files only OriginStack writes (frame arrays, rejection masks, per-session
# stacks of a hierarchical run and their sidecars).
_ORPHAN_PATTERNS = ('stack_aligned_*.dat', 'stack_rgb_*.dat', 'stack_lum_*.dat',
                    'stream_rgb_*.dat', 'stack_rejmap_*.dat', 'wavelet_combine_*.dat',
                    'master_*.dat', '*_stack.fits', '*_stack.jpg', '*_stack_config.toml')


def sweep_orphans(folders=None, min_age_hours: float = 6.0):
    """Remove OriginStack temp files left by a run that crashed or was killed.

    Only names OriginStack itself creates, and only files untouched for
    ``min_age_hours`` -- an older file cannot belong to a run that is still going
    (they are written throughout a run), so another instance is safe. Returns
    (files removed, bytes freed)."""
    import glob
    import tempfile
    import time
    if folders is None:
        folders = [tempfile.gettempdir()]
    cutoff = time.time() - min_age_hours * 3600.0
    n = freed = 0
    for folder in {os.path.abspath(f) for f in folders if f}:
        for pat in _ORPHAN_PATTERNS:
            for p in glob.glob(os.path.join(folder, pat)):
                try:
                    st = os.stat(p)
                    if st.st_mtime > cutoff or not os.path.isfile(p):
                        continue
                    os.remove(p)
                    n += 1
                    freed += st.st_size
                except OSError:
                    pass          # in use, or no permission: leave it
    return n, freed
