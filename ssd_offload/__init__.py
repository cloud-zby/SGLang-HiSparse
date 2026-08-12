"""
ssd_offload/
============

SSD offload abstraction for GPU KV-cache-style data movement.

Provides a transport-agnostic top-level interface for reading/writing byte
ranges between a memory region (GPU or host) and a file on SSD, with **three
pluggable backends**:

  1. GDS backend   (GPUDirect Storage, cuFile)   -- 方式二
     CPU issues the I/O command, but data moves GPU <-> storage *directly*
     without a CPU bounce buffer.
  2. SCADA backend (Scaled Accelerated Data Access / Tutti-style) -- 方式一
     GPU threads themselves drive the NVMe (CPU appears O(1) per batch).
     This backend is a PLACEHOLDER on this machine and raises NotImplemented
     until real dedicated-NVMe hardware is present.
  3. CPU backend   (POSIX pread/pwrite via host buffer)          -- 方式三
     Both the command and the data pass through the CPU (bounce buffer).
     Used here as a correct, always-available reference/fallback.

The goal: the upper layer only sees the *top-level interface* (in this file).
Each backend implements that interface behind a common base class, and the
factory (``backends/factory.py``) selects one at runtime.

Design goals
------------
- **Transport-agnostic top-level API**: the app calls ``open/read/write/...``
  without caring whether GDS or CPU is underneath.
- **Pluggable backends**: same interface, different ``_execute``.
- **GDS backend is real** (uses the official cuFile API via ctypes against
  libcufile.so); graceful fallback to compat mode when P2P is unavailable.
- **SCADA placeholder**: reserved extension point, NotImplemented on this host.
"""

from .core import (
    AccessMethod,
    BackendCapabilities,
    MemoryRegistration,
    OffloadHandle,
    OffloadRequest,
    ReadResult,
    SsdOffload,
    SsdTarget,
)
from .backends import create_ssd_offload, list_backends

__all__ = [
    # top-level
    "SsdOffload",
    "SsdTarget",
    "OffloadHandle",
    "OffloadRequest",
    "MemoryRegistration",
    "ReadResult",
    "AccessMethod",
    "BackendCapabilities",
    # factory
    "create_ssd_offload",
    "list_backends",
]

__version__ = "0.1.0"
