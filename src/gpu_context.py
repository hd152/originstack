"""GPU / CPU abstraction layer."""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import numpy as np
from scipy import ndimage

from src.utils import safe_print

_log = logging.getLogger('originstack')

try:
    import cupy as cp
    HAS_CUPY = True
except Exception:
    cp = None
    HAS_CUPY = False


def _looks_like_cuda_oom(exc: BaseException) -> bool:
    """Return True if *exc* looks like a GPU out-of-memory error.

    Standalone so both ``GpuContext.is_oom`` and the unraisablehook below
    (installed on module objects, not GpuContext instances) can share it.
    """
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    return ('out of memory' in msg
            or 'cudaerrormemoryal' in msg
            or 'outofmemory' in name
            or 'cudaruntimeerror' in name
            or 'cudadrivererror' in name)


_unraisable_hook_installed = False


def _install_cuda_unraisablehook():
    """Silence the CUDA-OOM cleanup-noise flood from ``__dealloc__``.

    Once a CUDA context is fatally corrupted by an OOM, every later
    garbage-collection of a leftover cupy Memory/Module/Stream object also
    fails -- and exceptions raised inside a destructor can't propagate to
    application code, so no try/except anywhere can catch them. Python's
    only hook for this is sys.unraisablehook, which fires once per such
    finalizer failure regardless of which call site or object it came from
    (observed in the wild as thousands of repeated tracebacks for the rest
    of the process's life, including after a fully successful run). We
    still forward anything that doesn't look like a CUDA-OOM to the
    previous hook so genuinely unexpected unraisable errors stay visible.
    """
    global _unraisable_hook_installed
    if _unraisable_hook_installed:
        return
    _unraisable_hook_installed = True
    _prev_hook = sys.unraisablehook

    def _hook(unraisable):
        exc = unraisable.exc_value
        if exc is not None and _looks_like_cuda_oom(exc):
            return
        _prev_hook(unraisable)

    sys.unraisablehook = _hook


class GpuContext:
    """Array-agnostic computation context.

    Provides ``xp`` (numpy or cupy), ``xndimage`` and ``xsignal`` so that
    every compute function can be written once and dispatched to GPU or CPU.
    """

    def __init__(self, use_gpu: bool = False):
        self.active = False
        self.xp = np
        self.xndimage = ndimage
        self.xsignal = None
        self.device_name = "CPU"
        self.vram_total_mb = 0.0
        self.vram_free_mb = 0.0

        if use_gpu and HAS_CUPY:
            _install_cuda_unraisablehook()
            try:
                cp.cuda.Device(0).compute_capability
                self.xp = cp
                import cupyx.scipy.ndimage as _cp_ndimage
                self.xndimage = _cp_ndimage
                try:
                    import cupyx.scipy.signal as _cp_signal
                    self.xsignal = _cp_signal
                except ImportError:
                    from scipy import signal as _signal
                    self.xsignal = _signal
                self.active = True
                dev = cp.cuda.Device(0)
                self.device_name = str(dev)
                mem = dev.mem_info
                self.vram_free_mb = mem[0] / 1024 ** 2
                self.vram_total_mb = mem[1] / 1024 ** 2
            except Exception as exc:
                _log.warning("GPU init failed (%s), falling back to CPU.", exc)
                self.xp = np
                self.xndimage = ndimage
                self.active = False

        if self.xsignal is None:
            from scipy import signal as _signal
            self.xsignal = _signal

    # --- transfer helpers ---------------------------------------------------
    def to_device(self, arr: np.ndarray):
        """Move *arr* to GPU.  No-op when running on CPU."""
        if self.active:
            return cp.asarray(arr)
        return arr

    def to_host(self, arr) -> np.ndarray:
        """Move *arr* to CPU numpy.  No-op when already numpy."""
        if self.active and hasattr(arr, 'get'):
            return arr.get()
        return np.asarray(arr)

    def free_pool(self):
        """Release CuPy's cached GPU memory.

        Swallows any CUDA errors — when the GPU is completely exhausted even
        cudaFree can fail, and we must not let cleanup raise while we're
        already handling an OOM exception.
        """
        if self.active:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
            try:
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass

    def disable(self):
        """Permanently fall back to CPU for this session after an unrecoverable GPU error."""
        self.free_pool()
        self.active = False
        self.xp = np
        self.xndimage = ndimage

    def available_vram_mb(self) -> float:
        if self.active:
            return cp.cuda.Device(0).mem_info[0] / 1024 ** 2
        return 0.0

    def max_gpu_workers(self, per_worker_mb: float, reserve_mb: float = 512.0) -> int:
        """Return max thread count that fits in VRAM, minimum 1."""
        if not self.active:
            return max(1, os.cpu_count() or 4)
        avail = self.available_vram_mb() - reserve_mb
        if avail <= 0 or per_worker_mb <= 0:
            return 1
        return max(1, int(avail / per_worker_mb))

    def stream_context(self):
        """Return a context manager that creates a per-thread CUDA stream."""
        if self.active:
            return _CudaStreamContext(self)
        return _NullContext()

    def is_oom(self, exc: Exception) -> bool:
        """Return True if *exc* looks like a GPU out-of-memory error."""
        return _looks_like_cuda_oom(exc)

    def print_status(self):
        if self.active:
            safe_print(f"  GPU: {self.device_name}")
            safe_print(f"  VRAM: {self.vram_free_mb:.0f}/{self.vram_total_mb:.0f} MB free")
        else:
            safe_print(f"  Compute: CPU")


class _CudaStreamContext:
    """Context manager that creates a per-thread CUDA stream.

    Every GPU call site in this codebase enters this via
    ``with gpu.stream_context():`` before doing any real GPU work -- so a
    failure here is not hypothetical: even a "lightweight" CUDA stream needs
    a small driver-level allocation, which can fail under severe VRAM
    pressure before the caller's actual work has even started. Without this
    guard, that OOM propagated uncaught straight through every one of those
    call sites (observed in the wild via reload_accepted_frames's per-frame
    reload after a checkpoint resume) -- none of them wrapped the `with`
    statement itself in a try/except, only the work *inside* it.
    """
    def __init__(self, gpu: 'GpuContext'):
        self._gpu = gpu
        self._stream = None

    def __enter__(self):
        try:
            self._stream = cp.cuda.Stream(non_blocking=True)
            self._stream.__enter__()
            return self._stream
        except Exception as exc:
            self._stream = None
            if self._gpu.is_oom(exc):
                _log.warning(
                    "GPU stream creation failed (%s) -- disabling GPU for the rest of this run.", exc)
                self._gpu.disable()
                return None  # caller sees gpu.active=False from here on, same as _NullContext
            raise

    def __exit__(self, *exc):
        if self._stream is None:
            return False
        # Swallow secondary cleanup errors -- a stream left partially
        # initialised by an OOM during the `with` block's own body can fail
        # here too (observed in the wild as a masking
        # "'Stream' object has no attribute '_foreign_stream_ref'"
        # AttributeError), and must not replace the real exception already
        # propagating through this context manager.
        try:
            self._stream.synchronize()
        except Exception:
            pass
        try:
            self._stream.__exit__(*exc)
        except Exception:
            pass
        return False


class _NullContext:
    """No-op context manager for CPU fallback."""
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


_gpu: Optional[GpuContext] = None


def get_gpu() -> GpuContext:
    global _gpu
    if _gpu is None:
        _gpu = GpuContext(use_gpu=False)
    return _gpu
