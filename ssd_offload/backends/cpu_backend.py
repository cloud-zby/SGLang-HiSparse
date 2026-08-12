"""
ssd_offload/backends/cpu_backend.py
====================================

方式三: CPU bounce-buffer path (POSIX pread/pwrite)

Both the command and the data pass through the CPU:
  * command: host issues os.pread/os.pwrite
  * data   : copied through a CPU-side aligned staging buffer, then moved to/
             from the caller's memory address.

This is the always-available reference backend and the fallback when GDS is
unavailable. It is the correct, safe choice on hosts where the SSD is the
system disk / not P2P-capable.

Alignment
---------
O_DIRECT requires buffer/offset/length aligned to the block size. We open the
file with O_DIRECT and transparently align any request by staging through a
4 KiB-aligned buffer (a small extra copy — acceptable for a reference path).
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

try:
    O_DIRECT = os.O_DIRECT
except AttributeError:  # non-Linux
    O_DIRECT = 0o40000

ALIGNMENT = 4096


def _align_up(v: int, a: int = ALIGNMENT) -> int:
    return (v + a - 1) & ~(a - 1)


def _align_down(v: int, a: int = ALIGNMENT) -> int:
    return v & ~(a - 1)


class _AlignedBuffer:
    """An ALIGNMENT-aligned heap buffer usable as an O_DIRECT staging area."""

    def __init__(self, size: int):
        if size <= 0:
            raise ValueError("size must be > 0")
        self.size = size
        total = size + ALIGNMENT
        self._raw = ctypes.create_string_buffer(total)
        raw = ctypes.addressof(self._raw)
        self._off = _align_up(raw, ALIGNMENT) - raw
        self.address = raw + self._off
        assert self.address % ALIGNMENT == 0


class CpuSsdOffload(SsdOffload):
    backend_name = "cpu"
    capabilities = BackendCapabilities(
        method=AccessMethod.CPU,
        supports_device_direct=False,
        supports_host_bounce=True,
        requires_pinned_memory=False,
    )

    def __init__(self, block_size: int = ALIGNMENT, use_o_direct: bool = True):
        super().__init__()
        self.block_size = block_size if (block_size % 512 == 0) else ALIGNMENT
        self.use_o_direct = use_o_direct
        # Each opened target -> open fd (stored in target._priv).
        # MemoryRegistration._priv unused for CPU backend.

    # ---- memory: CPU path can operate directly on any caller address ----
    def _register_memory_impl(self, address, size, kind):
        # Nothing to pin on the CPU bounce path; address is used as-is.
        return None

    def _unregister_memory_impl(self, priv):
        pass

    # ---- target ----
    def _open_impl(self, path: str, logical_size: int):
        flags = os.O_RDWR | (O_DIRECT if self.use_o_direct else 0)
        fd = os.open(path, flags)
        return fd

    def _close_impl(self, priv):
        if priv is not None:
            try:
                os.close(priv)
            except OSError:
                pass

    # ---- execute ----
    def _submit_impl(self, reqs: List[OffloadRequest], stream: Optional[int]) -> ReadResult:
        for req in reqs:
            err = self._odoio(fd=req.target._priv,
                              buf_addr=req.memory.address,
                              buf_off=req.memory_offset,
                              file_off=req.target_offset,
                              length=req.length,
                              write=(req.direction == 1))
            if err is not None:
                return ReadResult(False, err, 0,
                                  [s for s in ["accepted"] * len(reqs)])
        return ReadResult(True, "", sum(r.length for r in reqs),
                          ["accepted"] * len(reqs))

    def _odoio(self, fd, buf_addr, buf_off, file_off, length, write):
        bs = self.block_size
        start = _align_down(file_off, bs)
        end = _align_up(file_off + length, bs)
        seg = file_off - start
        total = end - start
        try:
            staging = _AlignedBuffer(total)
            if write:
                ctypes.memmove(staging.address + seg, buf_addr + buf_off, length)
                self._pwrite_all(fd, staging.address, start, total)
                os.fsync(fd)
            else:
                self._pread_all(fd, staging.address, start, total)
                ctypes.memmove(buf_addr + buf_off, staging.address + seg, length)
        except OSError as exc:
            return f"{'write' if write else 'read'} off={file_off} len={length}: {exc}"
        return None

    @staticmethod
    def _pread_all(fd, addr, offset, length):
        done = 0
        while done < length:
            data = os.pread(fd, length - done, offset + done)
            if not data:
                raise OSError("short read (EOF)")
            ctypes.memmove(addr + done, data, len(data))
            done += len(data)

    @staticmethod
    def _pwrite_all(fd, addr, offset, length):
        done = 0
        while done < length:
            chunk = ctypes.string_at(addr + done, min(length - done, 1 << 22))
            n = os.pwrite(fd, chunk, offset + done)
            if n <= 0:
                raise OSError("short write")
            done += n

    def _shutdown_impl(self):
        # All targets closed in SsdOffload.shutdown(); nothing else to do.
        pass
