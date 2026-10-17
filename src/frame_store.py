"""Where the big per-session frame arrays live: RAM when it comfortably fits,
a temp file (np.memmap) otherwise. ``--frame-store auto|ram|disk``.

Phase 1 RGB and luminance frames can additionally use independent, losslessly
compressed Rust LZ4 files when ``--compress-phase1`` is selected or when the
projected Phase 1 array and worker memory exceed the safe available-memory budget.

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
import sys
import tempfile
import uuid
import weakref
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils import safe_print

_log = logging.getLogger('originstack')

# Of total RAM, always left free. 0.30 kept a 532-frame Sculptor session's 14.7 GB
# aligned stack on disk (Windows counted ~38 of 64 GB as available after Phase 1's
# 53 GB of dirty temp-file pages): alignment 32.5 -> 16.2 s and the whole stacking
# phase 85.8 -> 63.0 s with it in RAM, while 0.20 still leaves ~13 GB for everything else.
RESERVE_FRAC = 0.20
SHM_PREFIX = 'shm:'
COMPRESSED_PREFIX = 'lz4:'
# Compressed / raw size assumed by the up-front disk check. Byte-shuffled LZ4 gave
# 1.33x on a noisy float32 frame (only the exponent byte compresses), so 0.8 errs safe.
COMPRESSED_SIZE_ESTIMATE = 0.8
_NATIVE_COMPRESSOR = None


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


class _CompressedFrameArray:
    """Frame-indexed, lossless disk store with the subset of ndarray operations
    used by the pipeline. Frame compression and decompression run in Rust."""

    def __init__(self, prefix: str, dtype, shape: tuple):
        self.prefix = prefix
        self.dtype = np.dtype(dtype)
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self._frame_shape = self.shape[1:]

    def _path(self, index: int) -> str:
        return f'{self.prefix}{index:08d}.lz4'

    def _normalise_index(self, index: int) -> int:
        index = int(index)
        if index < 0:
            index += self.shape[0]
        if not 0 <= index < self.shape[0]:
            raise IndexError("frame index out of range")
        return index

    def _read_frame(self, index: int) -> np.ndarray:
        native = _require_native_compressor()
        index = self._normalise_index(index)
        if hasattr(native, 'decompress_frame_file_into'):
            # >= 0.60: read into a pooled buffer, not a fresh frame-sized bytes object
            frame = np.empty(self._frame_shape, self.dtype)
            native.decompress_frame_file_into(self._path(index), frame.view(np.uint8).reshape(-1),
                                              self.dtype.itemsize)
            return frame
        with open(self._path(index), 'rb') as stream:
            compressed = stream.read()
        if hasattr(native, 'decompress_frame_into'):
            # astro_native >= 0.58: chunks decoded in parallel straight into an
            # uninitialised frame (no zero fill, no frame-sized scratch); reads both formats
            frame = np.empty(self._frame_shape, self.dtype)
            native.decompress_frame_into(compressed, frame.view(np.uint8).reshape(-1),
                                         self.dtype.itemsize)
            return frame
        raw = native.decompress_frame(
            compressed,
            int(np.prod(self._frame_shape)) * self.dtype.itemsize,
            self.dtype.itemsize)
        frame = np.frombuffer(raw, dtype=self.dtype).reshape(self._frame_shape)
        # astro_native >= 0.56 returns a fresh writable array; older wheels return bytes
        return frame if frame.flags.writeable else frame.copy()

    def _write_frame(self, index: int, frame: np.ndarray) -> None:
        native = _require_native_compressor()
        index = self._normalise_index(index)
        try:
            contiguous = np.ascontiguousarray(
                np.broadcast_to(np.asarray(frame, dtype=self.dtype), self._frame_shape))
        except ValueError as exc:
            raise ValueError(
                f"Expected a value broadcastable to frame shape {self._frame_shape}"
            ) from exc
        path = self._path(index)
        temporary = f'{path}.{os.getpid()}.tmp'
        flat = contiguous.view(np.uint8).reshape(-1)
        try:
            if hasattr(native, 'compress_frame_chunked_to_file'):
                # >= 0.59: the same bytes, coded into pooled buffers and written from
                # there (no concatenated copy, no Python bytes, no write-call copy)
                native.compress_frame_chunked_to_file(flat, self.dtype.itemsize, temporary)
            else:
                # chunked (>= 0.58): shuffled in cache, chunks compressed in parallel
                encode = getattr(native, 'compress_frame_chunked', native.compress_frame)
                compressed = encode(flat, self.dtype.itemsize)
                with open(temporary, 'wb') as stream:
                    stream.write(compressed)
            os.replace(temporary, path)
        finally:
            try:
                os.remove(temporary)
            except FileNotFoundError:
                pass

    def __getitem__(self, key):
        if isinstance(key, tuple):
            frame_key, *rest = key
            subkey = tuple(rest)
        else:
            frame_key, subkey = key, ()
        if isinstance(frame_key, (int, np.integer)):
            frame = self._read_frame(frame_key)
            return frame[subkey] if subkey else frame
        indices = np.arange(self.shape[0])[frame_key]
        if np.ndim(indices) == 0:
            frame = self._read_frame(int(indices))
            return frame[subkey] if subkey else frame
        frames = np.stack([self._read_frame(int(i)) for i in np.asarray(indices).flat])
        frames = frames.reshape((*np.shape(indices), *self._frame_shape))
        return frames[(slice(None), *subkey)] if subkey else frames

    def __setitem__(self, key, value) -> None:
        if isinstance(key, tuple):
            frame_key, *rest = key
            subkey = tuple(rest)
        else:
            frame_key, subkey = key, ()
        if isinstance(frame_key, (int, np.integer)):
            index = self._normalise_index(frame_key)
            frame = self._read_frame(index) if subkey else None
            if subkey:
                frame[subkey] = value
            else:
                frame = value
            self._write_frame(index, frame)
            return
        indices = np.arange(self.shape[0])[frame_key]
        values = np.asarray(value)
        if np.ndim(indices) == 0:
            index = int(indices)
            frame = self._read_frame(index) if subkey else None
            if subkey:
                frame[subkey] = value
            else:
                frame = value
            self._write_frame(index, frame)
            return
        for offset, index in enumerate(np.asarray(indices).flat):
            frame = self._read_frame(int(index)) if subkey else None
            if subkey:
                frame[subkey] = values[offset]
            else:
                frame = values[offset]
            self._write_frame(int(index), frame)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        result = np.stack([self._read_frame(i) for i in range(self.shape[0])])
        if dtype is not None:
            result = result.astype(dtype, copy=False)
        if copy:
            result = result.copy()
        return result

    def flush(self) -> None:
        pass


def _require_native_compressor():
    global _NATIVE_COMPRESSOR
    if _NATIVE_COMPRESSOR is not None:
        return _NATIVE_COMPRESSOR
    try:
        import astro_native
    except ImportError as exc:
        raise RuntimeError(
            "Lossless Phase 1 compression requires the Rust astro_native extension. "
            "Build it with 'maturin develop --release' from ext/astro_native, or use "
            "the default uncompressed frame store."
        ) from exc
    if not (hasattr(astro_native, 'compress_frame')
            and hasattr(astro_native, 'decompress_frame')):
        raise RuntimeError(
            "The installed astro_native extension lacks Rust frame compression. "
            "Rebuild it with 'maturin develop --release' from ext/astro_native."
        )
    _NATIVE_COMPRESSOR = astro_native
    return _NATIVE_COMPRESSOR


def _available_mb() -> Tuple[Optional[float], Optional[float]]:
    try:
        import psutil
        v = psutil.virtual_memory()
        return v.available / 1e6, v.total / 1e6
    except Exception:
        if sys.platform == 'win32':
            return _windows_memory_mb()
        return None, None


def _windows_memory_mb() -> Tuple[Optional[float], Optional[float]]:
    """Return physical memory figures when psutil is unavailable on Windows."""
    try:
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ('dwLength', ctypes.c_ulong),
                ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        get_memory_status = ctypes.WinDLL('kernel32', use_last_error=True).GlobalMemoryStatusEx
        get_memory_status.argtypes = [ctypes.POINTER(_MemoryStatusEx)]
        get_memory_status.restype = ctypes.c_bool
        if not get_memory_status(ctypes.byref(status)):
            return None, None
        return status.ullAvailPhys / 1e6, status.ullTotalPhys / 1e6
    except Exception:
        return None, None


_LIVE_STORES: 'weakref.WeakSet[FrameStore]' = weakref.WeakSet()


def cleanup_all_stores() -> None:
    """Close and delete every frame store still open (end of a run, any outcome).
    A store whose owner raised before its own ``cleanup()`` -- the aligned stack on
    a disk-full or failed combine -- otherwise kept its temp file for the life of
    the process."""
    for store in list(_LIVE_STORES):
        try:
            store.cleanup()
        except Exception:
            pass


class FrameStore:
    """Creates and cleans up session frame arrays; see the module docstring."""

    def __init__(self, mode: str = 'auto'):
        self.mode = mode if mode in ('auto', 'ram', 'disk') else 'auto'
        self._files: List[str] = []
        self._memmaps: List[np.memmap] = []
        self._shms: List[shared_memory.SharedMemory] = []
        self.placement: Dict[str, str] = {}
        _LIVE_STORES.add(self)

    @staticmethod
    def _fits_ram(nbytes: int, reserve_mb: float) -> bool:
        avail, total = _available_mb()
        if avail is None:
            return False
        return nbytes / 1e6 <= avail - RESERVE_FRAC * total - reserve_mb

    @staticmethod
    def phase1_compression_needed(nbytes: int, reserve_mb: float) -> Optional[bool]:
        """Whether Phase 1's uncompressed arrays exceed the safe RAM budget.

        ``None`` means memory telemetry is unavailable, so automatic mode
        cannot make a reliable prediction and leaves compression disabled.
        """
        avail, total = _available_mb()
        if avail is None or total is None:
            return None
        budget_mb = avail - RESERVE_FRAC * total - reserve_mb
        return nbytes / 1e6 > max(0.0, budget_mb)

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
               reserve_mb: float = 0.0, prefer: str = 'ram',
               compressed: bool = False) -> Tuple[Any, str]:
        """(array, spec). ``shared``: other processes will attach by spec.
        ``reserve_mb``: memory the caller still needs on top (e.g. its workers).
        ``prefer`` (auto mode): 'ram' -- RAM whenever it fits; 'disk' -- a temp file
        unless that would leave the temp disk short (< max(10%, 5 GB) free), then
        RAM if it fits. Disk is the last resort either way."""
        dt = np.dtype(dtype)
        nbytes = int(np.prod(shape)) * dt.itemsize
        if compressed:
            _require_native_compressor()
            _require_disk(int(nbytes * COMPRESSED_SIZE_ESTIMATE), prefix)
            token = uuid.uuid4().hex
            compressed_prefix = os.path.join(
                tempfile.gettempdir(), f'{prefix}compressed_{token}_')
            arr = _CompressedFrameArray(compressed_prefix, dt, shape)
            paths = [arr._path(i) for i in range(shape[0])]
            self._files.extend(paths)
            try:
                from src.cleanup import register as _cleanup_register
                for path in paths:
                    _cleanup_register(path)
            except Exception:
                pass
            self.placement[prefix] = f'compressed ({nbytes / 1e9:.1f} GB uncompressed)'
            return arr, COMPRESSED_PREFIX + compressed_prefix
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
        _require_disk(nbytes, prefix)
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


class FrameStoreSpaceError(OSError):
    """Neither RAM nor the temp disk can hold a frame array."""


def _require_disk(nbytes: int, prefix: str) -> None:
    """Refuse up front when the temp disk cannot hold the array at all.

    The file is created sparse, so running out of space only shows when the
    pages are written -- deep into Phase 1 or alignment, as a crash or a stall,
    after many minutes. An 849-frame session needed ~85 GB for Phase 1's arrays
    and 60 GB more for the aligned stack."""
    try:
        import shutil
        free = shutil.disk_usage(tempfile.gettempdir()).free
    except Exception:
        return
    if nbytes > free - 1e9:
        what = prefix.rstrip('_').replace('_', ' ')
        raise FrameStoreSpaceError(
            f"Not enough space for the {what} frames: {nbytes / 1e9:.1f} GB needed, "
            f"{free / 1e9:.1f} GB free in the temp folder ({tempfile.gettempdir()}). "
            f"Choose a temp folder on a drive with more room (Temp folder in the app, "
            f"--temp-dir on the command line), free some disk space, or stack fewer "
            f"frames at a time (stack each night separately and combine them with --merge).")


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
    if spec.startswith(COMPRESSED_PREFIX):
        prefix = spec[len(COMPRESSED_PREFIX):]
        return _CompressedFrameArray(prefix, dtype, shape)
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
