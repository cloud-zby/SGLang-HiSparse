"""
ssd_offload/backends/gds_backend.py
====================================

方式二: GPUDirect Storage (GDS) via the official cuFile API.

CPU still issues the command, but the DATA moves GPU <-> storage *directly*,
without a CPU bounce buffer, by using NVIDIA's cuFile library.

Implementation notes (based on the official GDS cuFile API Reference):
  * Driver:  cuFileDriverOpen() / cuFileDriverClose()
  * Handle:  cuFileHandleRegister(&fh, &descr) / cuFileHandleDeregister(fh)
             descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD (Linux); descr.handle.fd
  * Buffer:  cuFileBufRegister(devPtr, size, 0) / cuFileBufDeregister(devPtr)
  * IO:      cuFileRead(fh, bufPtr, size, file_off, buf_off) -> ssize_t
             cuFileWrite(...) similarly.
             Return >= 0 => bytes transferred; -1 => POSIX errno; otherwise a
             negative CUfileOpError value (from cufileop_status_error).

Only the synchronous cuFileRead/cuFileWrite are used here (matches the
top-level interface's synchronous submit).  cuFileBufRegister makes the GPU
buffer pinned and usable for P2P; when P2P is unsupported on a given fs/host,
cuFile transparently falls back to an internal compat path if
``allow_compat_mode`` is true (the default), so the call is still correct.

ctypes details
--------------
* The cuFile library is ``libcufile.so`` (loaded via ctypes.CDLL).
* We bind the exact C prototypes from cufile.h.

On hosts where the driver or P2P is unavailable, cuFile still works (compat
mode), so GDS backend is generally loadable; we detect driver-open failures
and report them clearly rather than crashing.
"""

from __future__ import annotations

import ctypes
import os
from typing import List, Optional

from ..core import (
    AccessMethod,
    BackendCapabilities,
    MemoryKind,
    OffloadRequest,
    ReadResult,
    SsdOffload,
)

# ---------------------------------------------------------------------------
# cuFile constants (from cufile.h) — subset used here
# ---------------------------------------------------------------------------
CU_FILE_SUCCESS = 0
CUFILEOP_BASE_ERR = 5000

# CUfileFileHandleType
CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1
CU_FILE_HANDLE_TYPE_OPAQUE_WIN32 = 2
CU_FILE_HANDLE_TYPE_USERSPACE_FS = 3

# CUfileError codes we care about for diagnostics
CU_FILE_DRIVER_NOT_INITIALIZED = 5001
CU_FILE_HANDLE_ALREADY_REGISTERED = 5028
CU_FILE_MEMORY_ALREADY_REGISTERED = 5023
CU_FILE_IO_NOT_SUPPORTED = 5008
CU_FILE_DEVICE_NOT_SUPPORTED = 5009
CU_FILE_PLATFORM_NOT_SUPPORTED = 5007


class CUfileError_t(ctypes.Structure):
    _fields_ = [
        ("err", ctypes.c_int),      # CUfileOpError
        ("cu_err", ctypes.c_int),   # CUresult
    ]


class CUfileHandle_t(ctypes.c_void_p):
    pass


class _HandleUnion(ctypes.Union):
    _fields_ = [
        ("fd", ctypes.c_int),
        ("handle", ctypes.c_void_p),
    ]


class CUfileDescr_t(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("handle", _HandleUnion),
        ("fs_ops", ctypes.c_void_p),
    ]


# ---------------------------------------------------------------------------
# Library loading + function binding
# ---------------------------------------------------------------------------
def _load_cufile():
    """Load libcufile and bind the needed C functions."""
    try:
        lib = ctypes.CDLL("libcufile.so")
    except OSError as exc:
        # Try the CUDA-toolkit path as a fallback.
        cands = [
            "/usr/local/cuda-13.2/targets/x86_64-linux/lib/libcufile.so",
            "/usr/local/cuda/targets/x86_64-linux/lib/libcufile.so",
        ]
        lib = None
        for c in cands:
            if os.path.exists(c):
                lib = ctypes.CDLL(c)
                break
        if lib is None:
            raise RuntimeError("libcufile.so not found; GDS backend unavailable") from exc

    # Bind functions
    fn = {}

    fn["driver_open"] = lib.cuFileDriverOpen
    fn["driver_open"].restype = CUfileError_t
    fn["driver_open"].argtypes = []

    fn["driver_close"] = lib.cuFileDriverClose
    fn["driver_close"].restype = CUfileError_t
    fn["driver_close"].argtypes = []

    fn["handle_register"] = lib.cuFileHandleRegister
    fn["handle_register"].restype = CUfileError_t
    fn["handle_register"].argtypes = [ctypes.POINTER(CUfileHandle_t),
                                      ctypes.POINTER(CUfileDescr_t)]

    fn["handle_deregister"] = lib.cuFileHandleDeregister
    fn["handle_deregister"].restype = None
    fn["handle_deregister"].argtypes = [CUfileHandle_t]

    fn["buf_register"] = lib.cuFileBufRegister
    fn["buf_register"].restype = CUfileError_t
    fn["buf_register"].argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

    fn["buf_deregister"] = lib.cuFileBufDeregister
    fn["buf_deregister"].restype = CUfileError_t
    fn["buf_deregister"].argtypes = [ctypes.c_void_p]

    fn["read"] = lib.cuFileRead
    fn["read"].restype = ctypes.c_ssize_t
    fn["read"].argtypes = [CUfileHandle_t, ctypes.c_void_p, ctypes.c_size_t,
                           ctypes.c_long, ctypes.c_long]

    fn["write"] = lib.cuFileWrite
    fn["write"].restype = ctypes.c_ssize_t
    fn["write"].argtypes = [CUfileHandle_t, ctypes.c_void_p, ctypes.c_size_t,
                            ctypes.c_long, ctypes.c_long]

    return lib, fn


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
class GdsSsdOffload(SsdOffload):
    backend_name = "gds"
    capabilities = BackendCapabilities(
        method=AccessMethod.GDS,
        supports_device_direct=True,   # when P2P available; else compat
        supports_host_bounce=True,     # compat mode inside cuFile
        requires_pinned_memory=True,   # cuFileBufRegister used for device buffers
        compat_mode=True,
    )

    def __init__(self, allow_compat: bool = True):
        super().__init__()
        self._lib, self._fn = _load_cufile()
        # Open the cuFile driver session.
        err = self._fn["driver_open"]()
        if err.err != CU_FILE_SUCCESS:
            raise RuntimeError(
                f"cuFileDriverOpen failed: code={err.err} (CUDA {err.cu_err}); "
                f"is nvidia-fs.ko loaded?")
        self._driver_open = True
        # Track registered file handles (fd -> CUfileHandle_t) and buffers.
        self._handles: dict = {}
        self._bufs: dict = {}   # address -> True

    def _open_impl(self, path: str, logical_size: int):
        # cuFile requires the fd; keep it open so cuFile can drive it.
        fd = os.open(path, os.O_RDWR | os.O_DIRECT)
        descr = CUfileDescr_t()
        descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD
        descr.handle.fd = fd
        fh = CUfileHandle_t()
        status = self._fn["handle_register"](ctypes.byref(fh), ctypes.byref(descr))
        if status.err != CU_FILE_SUCCESS:
            os.close(fd)
            raise RuntimeError(
                f"cuFileHandleRegister failed: code={status.err} "
                f"(errstr={status.err if abs(status.err) <= 5000 else 'cufile code'})")
        self._handles[fd] = fh
        return (fd, fh)  # backend-private

    def _close_impl(self, priv):
        if priv is None:
            return
        fd, fh = priv
        self._fn["handle_deregister"](fh)
        if fd in self._handles:
            del self._handles[fd]
        try:
            os.close(fd)
        except OSError:
            pass

    def _register_memory_impl(self, address, size, kind):
        if kind in (MemoryKind.DEVICE, MemoryKind.PINNED_HOST):
            # Pin/register for direct DMA. Device pointers benefit most.
            status = self._fn["buf_register"](address, size, 0)
            if status.err != CU_FILE_SUCCESS and status.err != CU_FILE_MEMORY_ALREADY_REGISTERED:
                # Not fatal: cuFile can fall back to an internal buffer (compat).
                return {"address": address, "registered": False}
            self._bufs[address] = True
            return {"address": address, "registered": True}
        return {"address": address, "registered": False}

    def _unregister_memory_impl(self, priv):
        if priv is None:
            return
        if priv.get("registered"):
            self._fn["buf_deregister"](priv["address"])
            self._bufs.pop(priv["address"], None)

    def _submit_impl(self, reqs: List[OffloadRequest], stream: Optional[int]) -> ReadResult:
        # Synchronous cuFileRead/cuFileWrite.
        for req in reqs:
            fd, fh = req.target._priv
            buf = req.memory.address
            file_off = req.target_offset
            buf_off = req.memory_offset
            size = req.length
            if req.direction == 1:  # WRITE
                ret = self._fn["write"](fh, buf, size, file_off, buf_off)
            else:                   # READ
                ret = self._fn["read"](fh, buf, size, file_off, buf_off)

            if ret < 0 or ret != size:
                return ReadResult(
                    False,
                    f"cuFile {'write' if req.direction == 1 else 'read'} returned "
                    f"{ret} (expected {size})",
                    0, ["accepted"] * len(reqs))
        return ReadResult(True, "", sum(r.length for r in reqs),
                          ["accepted"] * len(reqs))

    def _shutdown_impl(self):
        if getattr(self, "_driver_open", False):
            self._fn["driver_close"]()
            self._driver_open = False
