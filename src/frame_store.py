"""Where the big per-session frame arrays live: RAM when it comfortably fits,
a temp file (np.memmap) otherwise. ``--frame-store auto|ram|disk``.

The aligned-frame and per-frame RGB/luminance arrays are tens of GB on a long
session. As temp-file memmaps they cost real time even when the machine has
the RAM to spare: every page is a file-backed fault on first touch, dirty pages
are written back to disk, and alignment ran at ~10 frames/s against ~20 on the
same frames held in memory. ``auto`` puts an array in RAM only
if, when it is created, it fits inside the currently *available* memory minus a
reserve (``RESERVE_FRAC`` of total RAM plus ``reserve_mb`` the caller adds for
its own workers), so it never pushes the machine into paging. Measured on a
158-frame session (run-to-run noise on that machine was +-20 s): the aligned
stack in RAM made Phase 3 faster every time (18.7-23 s vs ~25 s), but Phase 1's
arrays in shared memory made Phase 1 erratic (37 s in some runs, 50 s in others
-- fresh pages zero-filled while 16 worker processes write). So under ``auto``
the aligned stack prefers RAM, and Phase 1's arrays prefer the temp file unless
the temp disk would be left short, in which case they go to RAM too (if it
fits). ``ram`` puts everything in RAM (no temp files), ``disk`` everything on
disk.

Arrays written by Phase 1's worker *processes* need cross-process memory, so
they use ``multiprocessing.shared_memory``; arrays only the main process
touches are plain numpy. Both are ``_RamArray`` (an ndarray with a no-op
``flush``, which the memmap code paths call). A worker attaches by the
returned spec: a file path, or ``"shm:<name>"``.
"""
from __future__ import annotations

import logging
import os
import tempfile
from multiprocessing import shared_memory
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils import safe_print

_log = logging.getLogger('originstack')

# Of total RAM, always left free. 0.30 kept a 532-frame Sculptor session's 14.7 GB
# aligned stack on disk (Windows counted ~38 of 64 GB as available after Phase 1's
# 53 GB of dirty temp-file pages): alignment 32.5 -> 16.2 s and the whole stacking
# phase 85.8 -> 63.0 s with it in RAM, while 0.20 still leaves ~13 GB for everything else.
RESERVE_FRAC = 0.20
SHM_PREFIX = 'shm:'


class _RamArray(np.ndarray):
    """ndarray that accepts the memmap API the pipeline uses (``flush``)."""

    def flush(self) -> None:
        pass


class _ScratchMemmap(np.memmap):
    """Temp-file memmap whose ``flush`` is a no-op. The file is scratch: it is
    deleted at cleanup, never reopened after a crash (checkpoints do not point at
    it), and every process maps the same file, which the OS keeps coherent across
    views. Flushing only forced a synchronous writeback: on a 532-frame session the
    aligned stack's flush took ~29 s of a 58 s alignment, writing 14.7 GB that was
    read straight back from the page cache and then deleted."""

    def flush(self) -> None:
        pass


def _available_mb() -> Tuple[Optional[float], Optional[float]]:
    try:
        import psutil
        v = psutil.virtual_memory()
        return v.available / 1e6, v.total / 1e6
    except Exception:
        return None, None


class FrameStore:
    """Creates and cleans up session frame arrays; see the module docstring."""

    def __init__(self, mode: str = 'auto'):
        self.mode = mode if mode in ('auto', 'ram', 'disk') else 'auto'
        self._files: List[str] = []
        self._memmaps: List[np.memmap] = []
        self._shms: List[shared_memory.SharedMemory] = []
        self.placement: Dict[str, str] = {}

    @staticmethod
    def _fits_ram(nbytes: int, reserve_mb: float) -> bool:
        avail, total = _available_mb()
        if avail is None:
            return False
        return nbytes / 1e6 <= avail - RESERVE_FRAC * total - reserve_mb

    @staticmethod
    def _disk_ok(nbytes: int) -> bool:
        """The temp disk keeps max(10%, 5 GB) free after this file."""
        try:
            import shutil
            du = shutil.disk_usage(tempfile.gettempdir())
            return du.free - nbytes >= max(0.10 * du.total, 5e9)
        except Exception:
            return True

    def _want_ram(self, nbytes: int, reserve_mb: float, prefer: str) -> bool:
        if self.mode == 'disk':
            return False
        if self.mode == 'ram':
            return True
        if prefer == 'disk' and self._disk_ok(nbytes):
            return False
        return self._fits_ram(nbytes, reserve_mb)

    def create(self, prefix: str, dtype, shape: tuple, shared: bool = False,
               reserve_mb: float = 0.0, prefer: str = 'ram') -> Tuple[np.ndarray, str]:
        """(array, spec). ``shared``: other processes will attach by spec.
        ``reserve_mb``: memory the caller still needs on top (e.g. its workers).
        ``prefer`` (auto mode): 'ram' -- RAM whenever it fits; 'disk' -- a temp file
        unless that would leave the temp disk short (< max(10%, 5 GB) free), then
        RAM if it fits. Disk is the last resort either way."""
        dt = np.dtype(dtype)
        nbytes = int(np.prod(shape)) * dt.itemsize
        if nbytes > 0 and self._want_ram(nbytes, reserve_mb, prefer):
            try:
                if shared:
                    shm = shared_memory.SharedMemory(create=True, size=nbytes)
                    arr = np.ndarray(shape, dtype=dt, buffer=shm.buf).view(_RamArray)
                    self._shms.append(shm)
                    spec = SHM_PREFIX + shm.name
                else:
                    arr = np.empty(shape, dtype=dt).view(_RamArray)
                    spec = ''
                self.placement[prefix] = f'RAM ({nbytes / 1e9:.1f} GB)'
                return arr, spec
            except (MemoryError, OSError) as e:
                _log.debug("frame store: RAM allocation of %.1f GB failed (%s); using disk",
                           nbytes / 1e9, e)
        fd, path = tempfile.mkstemp(suffix='.dat', prefix=prefix)
        os.close(fd)
        try:
            from src.cleanup import register as _cleanup_register
            _cleanup_register(path)
        except Exception:
            pass
        mm = _ScratchMemmap(path, dtype=dt, mode='w+', shape=shape)
        self._files.append(path)
        self._memmaps.append(mm)
        self.placement[prefix] = f'disk ({nbytes / 1e9:.1f} GB)'
        return mm, path

    def report(self) -> str:
        return ', '.join(f"{k.rstrip('_')}: {v}" for k, v in self.placement.items())

    def cleanup(self) -> None:
        import gc
        for mm in self._memmaps:
            try:
                if getattr(mm, '_mmap', None) is not None:
                    mm._mmap.close()
            except Exception:
                pass
        self._memmaps.clear()
        gc.collect()
        for p in self._files:
            try:
                os.remove(p)
            except Exception:
                pass
            try:
                from src.cleanup import deregister as _cleanup_deregister
                _cleanup_deregister(p)
            except Exception:
                pass
        self._files.clear()
        for shm in self._shms:
            _close_shm(shm)
        self._shms.clear()


def _close_shm(shm: shared_memory.SharedMemory) -> None:
    try:
        shm.close()
    except BufferError:
        # a numpy view still exports the buffer; unlink anyway (freed when it dies)
        pass
    except Exception:
        pass
    try:
        shm.unlink()
    except Exception:
        pass


# --- worker side ------------------------------------------------------------

_ATTACHED: Dict[str, shared_memory.SharedMemory] = {}


def open_frame_array(spec: str, dtype, shape: tuple) -> np.ndarray:
    """A worker's writable view of an array created by ``FrameStore.create``."""
    if spec.startswith(SHM_PREFIX):
        name = spec[len(SHM_PREFIX):]
        shm = _ATTACHED.get(name)
        if shm is None:
            try:      # 3.13+: do not let this process's resource tracker unlink it
                shm = shared_memory.SharedMemory(name=name, create=False, track=False)
            except TypeError:
                shm = shared_memory.SharedMemory(name=name, create=False)
            _ATTACHED[name] = shm        # kept for the worker's lifetime
        return np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)
    return np.memmap(spec, dtype=dtype, mode='r+', shape=shape)


def announce(store: FrameStore) -> None:
    if store.placement:
        safe_print(f"  Frame store ({store.mode}): {store.report()}")
