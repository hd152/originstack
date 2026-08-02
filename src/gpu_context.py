"""GPU / CPU abstraction layer."""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
from scipy import ndimage

from src.utils import safe_print

try:
    import cupy as cp
    HAS_CUPY = True
except Exception:
    cp = None
    HAS_CUPY = False


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
                logging.warning("GPU init failed (%s), falling back to CPU.", exc)
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
            return _CudaStreamContext()
        return _NullContext()

    def is_oom(self, exc: Exception) -> bool:
        """Return True if *exc* looks like a GPU out-of-memory error."""
        msg  = str(exc).lower()
        name = type(exc).__name__.lower()
        return ('out of memory' in msg
                or 'cudaerrormemoryal' in msg
                or 'outofmemory' in name
                or 'cudaruntimeerror' in name)

    def print_status(self):
        if self.active:
            safe_print(f"  GPU: {self.device_name}")
            safe_print(f"  VRAM: {self.vram_free_mb:.0f}/{self.vram_total_mb:.0f} MB free")
        else:
            safe_print(f"  Compute: CPU")


class _CudaStreamContext:
    """Context manager that creates a per-thread CUDA stream."""
    def __enter__(self):
        self._stream = cp.cuda.Stream(non_blocking=True)
        self._stream.__enter__()
        return self._stream

    def __exit__(self, *exc):
        self._stream.synchronize()
        self._stream.__exit__(*exc)
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
