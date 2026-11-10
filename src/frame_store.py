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
# Phase 1 compression (auto) pays off once the frames are well beyond what the page cache can
# hold: on a 14 GB laptop (frames ~2.6x available RAM) uncompressed disk frames thrashed in
# alignment and combine; on a 64 GB machine (532 frames, ~1.1x) compression cost 35 s of 194.
COMPRESS_OVER_AVAILABLE = 1.5
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


class HalfFloatStack:
    """The aligned stack (N, H, W, C) held as float16: half the memory, and half the disk
    traffic when it does not fit in RAM (``--aligned-stack half``, the default where the
    combine supports it; ``run_stacking_phase``).

    Each frame is stored as ``float16((v - offset) / scale)`` with its own ``offset`` (the
    frame's median, from a strided sample) and ``scale`` (a power of two that keeps the
    largest sample inside float16's range), and read back as
    ``float32(h) * scale + offset``. A float16 keeps 11 bits of mantissa, so a sample is
    rounded by at most 1/4096 of its distance from the frame's median: around the sky that
    is thousands of times smaller than the pixel noise (0.10 ADU rms on noise of 500), and
    on a bright star it stays far below the star's own shot noise. NaN and infinity survive.

    Reading always decodes, into a fresh C-contiguous float32 array, so what a consumer
    sees does not depend on whether ``raw`` is in RAM or a temp-file memmap: the stack is
    bit-identical across placements, as it is for the float32 store. It is not bit-identical
    to the float32 store itself.

    Supports what the combine, the proper coadd and the noise validation use: ``shape``,
    ``dtype`` (float32, what reads return), ``stack[j]``, ``stack[j, rows, cols]``,
    ``stack[a:b, ...]``, ``lazy_frame(j)`` (decode only what is indexed) and
    ``read_band(r0, r1)``. Anything else gets a float32 copy from ``to_float32``."""

    decodes = True          # a read is already a private copy (no need to copy it again)

    def __init__(self, raw):
        if raw.ndim != 4 or raw.dtype != np.float16:
            raise ValueError("HalfFloatStack needs an (N, H, W, C) float16 array")
        self.raw = raw
        self.shape = tuple(raw.shape)
        self.ndim = 4
        self.dtype = np.dtype(np.float32)
        self.offset = np.zeros(raw.shape[0], np.float32)
        self.scale = np.ones(raw.shape[0], np.float32)

    def __len__(self) -> int:
        return self.shape[0]

    def flush(self) -> None:
        self.raw.flush()

    def write_frame(self, j: int, frame: np.ndarray) -> None:
        """Store ``frame`` (H, W, C) float32 as frame ``j``. ``frame`` is scratch: it is
        overwritten with the scaled values."""
        if frame.dtype != np.float32:
            raise ValueError("write_frame needs a float32 frame")
        sample = frame[::16, ::16]
        fin = sample[np.isfinite(sample)]
        off = np.float32(np.median(fin)) if fin.size else np.float32(0.0)
        np.subtract(frame, off, out=frame)
        with np.errstate(invalid='ignore'):
            lo, hi = float(np.nanmin(frame)), float(np.nanmax(frame))
        if not (np.isfinite(lo) and np.isfinite(hi)):
            # an infinite sample (rare): the scale comes from the finite ones, or the rest of
            # the frame would overflow float16 too
            fin = frame[np.isfinite(frame)]
            lo, hi = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 0.0)
        peak = max(abs(lo), abs(hi))
        # the largest sample lands in [8192, 16384): float16 reaches 65504
        exp = int(np.ceil(np.log2(peak / 16384.0))) if peak > 0 else 0
        exp = max(-100, min(100, exp))
        if exp:
            np.multiply(frame, np.float32(2.0 ** -exp), out=frame)
        with np.errstate(over='ignore'):        # an infinite sample stays infinite
            self.raw[j] = frame
        self.offset[j] = off
        self.scale[j] = np.float32(2.0 ** exp)

    def _decode(self, part: np.ndarray, j: int) -> np.ndarray:
        out = np.asarray(part).astype(np.float32)
        if self.scale[j] != 1.0:
            np.multiply(out, self.scale[j], out=out)
        np.add(out, self.offset[j], out=out)
        return out

    def lazy_frame(self, j: int) -> "_LazyHalfFrame":
        return _LazyHalfFrame(self, int(j))

    def __getitem__(self, key):
        first, rest = (key[0], key[1:]) if isinstance(key, tuple) else (key, ())
        if isinstance(first, (int, np.integer)):
            j = int(first)
            if j < 0:
                j += self.shape[0]
            return self._decode(self.raw[(j,) + tuple(rest)], j)
        if isinstance(first, slice):
            idx = range(*first.indices(self.shape[0]))
            return np.stack([self._decode(self.raw[(j,) + tuple(rest)], j) for j in idx]) if len(idx) else \
                np.empty((0,) + np.asarray(self.raw[(slice(0, 0),) + tuple(rest)]).shape[1:], np.float32)
        raise TypeError("HalfFloatStack is indexed by frame (an integer or a slice) first; "
                        "use to_float32() for anything else")

    def read_band(self, r0: int, r1: int, workers: int = 8) -> np.ndarray:
        """Rows ``r0:r1`` of every frame, decoded: (N, r1 - r0, W, C) float32. The frames
        are read and converted in threads (numpy releases the GIL for both)."""
        N, _H, W, C = self.shape
        out = np.empty((N, r1 - r0, W, C), np.float32)

        def one(j):
            np.copyto(out[j], self.raw[j, r0:r1], casting='same_kind')
            if self.scale[j] != 1.0:
                np.multiply(out[j], self.scale[j], out=out[j])
            np.add(out[j], self.offset[j], out=out[j])

        n = max(1, min(int(workers), N, os.cpu_count() or 4))
        if n == 1:
            for j in range(N):
                one(j)
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=n, thread_name_prefix='half-read') as pool:
                for _ in pool.map(one, range(N)):
                    pass
        return out

    def to_float32(self, store: "FrameStore", prefix: str = 'stack_aligned_f32_') -> np.ndarray:
        """The whole stack decoded into a float32 array from ``store`` (RAM or a temp
        file): for a consumer that needs a real ndarray."""
        arr, _ = store.create(prefix, 'float32', self.shape, reserve_mb=2000.0)
        for j in range(self.shape[0]):
            arr[j] = self[j]
        return arr


class _LazyHalfFrame:
    """One frame of a ``HalfFloatStack`` that decodes only what is indexed (the proper
    coadd's measuring pass reads a few star stamps per frame, not the frame)."""

    def __init__(self, stack: HalfFloatStack, j: int):
        self._stack, self._j = stack, j
        self.shape = stack.shape[1:]
        self.ndim = 3
        self.dtype = stack.dtype

    def __getitem__(self, key):
        return self._stack._decode(self._stack.raw[self._j][key], self._j)

    def __array__(self, dtype=None, copy=None):
        a = self._stack[self._j]
        return a if dtype is None else a.astype(dtype, copy=False)


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
            _log.debug("cleanup_all_stores: ignored error", exc_info=True)


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
        """Whether automatic mode compresses Phase 1's frame arrays; see
        ``phase1_compression_decision``."""
        return FrameStore.phase1_compression_decision(nbytes, reserve_mb)[0]

    @staticmethod
    def phase1_compression_decision(nbytes: int, reserve_mb: float) -> Tuple[Optional[bool], str]:
        """``(compress, reason)`` for Phase 1's frame arrays in automatic mode.

        Arrays that fit the safe RAM budget are not compressed. Beyond it they go to a temp
        file, and compression only pays when that file will thrash the page cache (more
        than ``COMPRESS_OVER_AVAILABLE`` x available RAM) or the temp disk cannot hold it
        uncompressed. Compressing whenever the budget was exceeded cost a 64 GB machine
        35 s of 194 on 532 frames (1.1x available RAM), and the decision moved with a
        snapshot of free memory: the same 344-frame session compressed in one run and not
        the next. ``None`` means memory telemetry is unavailable; compression stays off.
        """
        avail, total = _available_mb()
        if avail is None or total is None:
            return None, "memory availability could not be measured"
        budget_mb = avail - RESERVE_FRAC * total - reserve_mb
        if nbytes / 1e6 <= max(0.0, budget_mb):
            return False, "frame arrays fit within the safe available-memory budget"
        if not FrameStore._disk_ok(nbytes):
            return True, "the temp disk cannot hold the uncompressed frame arrays"
        if nbytes / 1e6 > COMPRESS_OVER_AVAILABLE * avail:
            return True, (f"frame arrays exceed {COMPRESS_OVER_AVAILABLE:g}x the available "
                          "memory, so uncompressed temp files would thrash the page cache")
        return False, ("frame arrays go to temp files, within "
                       f"{COMPRESS_OVER_AVAILABLE:g}x the available memory")

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
                _log.debug("create: ignored error", exc_info=True)
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
            _log.debug("create: ignored error", exc_info=True)
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
                _log.debug("cleanup: ignored error", exc_info=True)
        self._memmaps.clear()
        gc.collect()
        for p in self._files:
            try:
                os.remove(p)
            except Exception:
                _log.debug("cleanup: ignored error", exc_info=True)
            try:
                from src.cleanup import deregister as _cleanup_deregister
                _cleanup_deregister(p)
            except Exception:
                _log.debug("cleanup: ignored error", exc_info=True)
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
        _log.debug("_close_shm: ignored error", exc_info=True)
    try:
        shm.unlink()
    except Exception:
        _log.debug("_close_shm: ignored error", exc_info=True)


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
