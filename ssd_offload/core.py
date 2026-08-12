"""
ssd_offload/core.py
====================

The **transport-agnostic top-level interface** for SSD offload.

This module defines the public application-facing abstraction that is shared
by every backend (GDS / SCADA / CPU). The app code only ever talks to these
types; the backends plug in underneath.

Concepts
--------
* ``SsdTarget``        -- an opened file/region on SSD that can be read/written.
* ``MemoryRegistration`` -- a memory region (device or pinned host) known to the
  backend, usable as the source/destination of an offload.
* ``OffloadRequest``   -- a byte-range transfer intent (READ or WRITE) with
  explicit memory offset, target offset and length.
* ``SsdOffload``       -- the top-level engine: open/close targets, register/
  unregister memory, submit transfers, wait, release.

The interface deliberately does NOT expose backend-specific details (e.g. a
cuFile handle, a cudaStream, a NVMe queue).  Each backend keeps its own private
state inside ``SsdTarget`` / ``MemoryRegistration`` / ``SsdOffload``.

Method/filename naming keeps tests and callers identical across backends, so
switching from the CPU backend to the GDS backend (or later to SCADA) requires
no changes in the calling code -- only the factory selection changes.

Enums / capability model
------------------------
``AccessMethod`` enumerates the three transport families:
  GDS   : GPUDirect Storage (cuFile) -- direct data path, CPU issues commands.
  SCADA : GPU-driven NVMe (Tutti-style) -- CPU appears O(1) per batch.
  CPU   : POSIX bounce-buffer path.

``BackendCapabilities`` reports at runtime what a given backend actually
supports (some may overlap or degrade gracefully, e.g. GDS compat mode).
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Access method family (the three ways to reach storage from GPU memory)
# ---------------------------------------------------------------------------
class AccessMethod(IntEnum):
    CPU = 0    # 方式三: command + data both through CPU bounce buffer
    GDS = 1    # 方式二: CPU issues command, data GPU<->storage direct
    SCADA = 2  # 方式一: GPU threads drive NVMe; CPU O(1) per batch (placeholder)


# ---------------------------------------------------------------------------
# Memory kinds
# ---------------------------------------------------------------------------
class MemoryKind(IntEnum):
    HOST = 0
    PINNED_HOST = 1
    DEVICE = 2
    MANAGED = 3


# ---------------------------------------------------------------------------
# Capabilities / capability negotiation
# ---------------------------------------------------------------------------
@dataclass
class BackendCapabilities:
    """What a backend can actually do on the current host."""
    method: AccessMethod
    supports_device_direct: bool      # True if GDS P2P / SCADA GPU-drive
    supports_host_bounce: bool        # True if it can fall back to CPU path
    requires_pinned_memory: bool      # device buffers must be pinned/registered
    max_io_size: int = 0              # 0 = backend default / unlimited
    compat_mode: bool = False         # GDS-compat (POSIX fallback inside cuFile)

    def describe(self) -> str:
        return (
            f"method={self.method.name} device_direct={self.supports_device_direct} "
            f"host_bounce={self.supports_host_bounce} "
            f"requires_pinned={self.requires_pinned_memory} "
            f"compat={self.compat_mode}"
        )


# ---------------------------------------------------------------------------
# Public value types
# ---------------------------------------------------------------------------
@dataclass
class MemoryRegistration:
    """Public handle describing a registered memory region.

    ``address`` is an integer base address (as used by the backend). Suffices
    for both host (ctypes/managed) and device (GPU pointer) memory.
    """
    id: int
    address: int
    size: int
    kind: MemoryKind
    # Backend-private storage (never inspected by callers).
    _priv: object = field(default=None, repr=False, compare=False)


class SsdTarget:
    """An opened read/write region on SSD.

    Created via ``SsdOffload.open``.  The public ``uri`` and ``logical_size``
    are visible; backend-private state lives in ``_priv``.
    """

    def __init__(self, uri: str, logical_size: int, handle_id: int, priv: object = None):
        self.uri = uri
        self.logical_size = logical_size
        self.handle_id = handle_id
        self._priv = priv

    def valid(self) -> bool:
        return self.handle_id >= 0

    def __repr__(self) -> str:
        return f"SsdTarget(uri={self.uri!r}, size={self.logical_size}, id={self.handle_id})"


class OffloadHandle:
    """Opaque handle for an in-flight (or just-submitted) offload batch."""

    def __init__(self, handle_id: int, priv: object = None):
        self.handle_id = handle_id
        self._priv = priv

    def valid(self) -> bool:
        return self.handle_id >= 0


class OffloadRequest:
    """A single byte-range transfer intent (mirrors Tutti's IoRequest)."""

    __slots__ = ("direction", "memory", "memory_offset", "target",
                 "target_offset", "length")

    def __init__(self, direction: int, memory: MemoryRegistration,
                 memory_offset: int, target: SsdTarget,
                 target_offset: int, length: int):
        self.direction = direction          # 0 = READ (target->memory), 1 = WRITE
        self.memory = memory
        self.memory_offset = memory_offset
        self.target = target
        self.target_offset = target_offset
        self.length = length


class ReadResult:
    """Result of a completed offload batch: overall status + per-request state."""
    __slots__ = ("ok", "error", "accepted_bytes", "initial_states")

    def __init__(self, ok: bool, error: str = "", accepted_bytes: int = 0,
                 initial_states: Optional[List[str]] = None):
        self.ok = ok
        self.error = error
        self.accepted_bytes = accepted_bytes
        self.initial_states = initial_states or []


# ---------------------------------------------------------------------------
# Top-level engine (abstract base)
# ---------------------------------------------------------------------------
class SsdOffload:
    """Transport-agnostic SSD offload engine.

    Subclasses (in ``backends/``) implement the private ``_*`` hooks; all the
    public methods here are shared and backend-independent.

    Lifecycle / usage:
        engine = SsdOffload.create(factory_args)
        reg = engine.register_memory(address, size, kind)
        tgt = engine.open("file:///path/to/blob")
        io  = engine.submit([OffloadRequest(...)], stream=None)
        res = engine.wait(io)
        engine.release_io(io)
        engine.unregister_memory(reg)
        engine.close(tgt)
        engine.shutdown()
    """

    # --- to be implemented by backends ---
    backend_name: str = "base"
    capabilities: BackendCapabilities = BackendCapabilities(AccessMethod.CPU, False, True, False)

    def _open_impl(self, path: str, logical_size: int) -> object:
        raise NotImplementedError

    def _close_impl(self, priv: object) -> None:
        raise NotImplementedError

    def _register_memory_impl(self, address: int, size: int, kind: MemoryKind) -> object:
        # Return backend-private registration state, or None if not needed.
        return None

    def _unregister_memory_impl(self, priv: object) -> None:
        raise NotImplementedError

    def _submit_impl(self, reqs: List[OffloadRequest], stream: Optional[int]) -> ReadResult:
        # Execute the batch; return a ReadResult with ok/error/accepted_bytes.
        raise NotImplementedError

    def _shutdown_impl(self) -> None:
        raise NotImplementedError

    # --- shared public API ---
    def __init__(self):
        self._next_target_id = 0
        self._next_mem_id = 0
        self._targets: dict = {}     # id -> (public SsdTarget)
        self._mems: dict = {}        # id -> MemoryRegistration
        self._next_io_id = 0
        self._pending_results: dict = {}  # io_id -> ReadResult
        self._running = True

    # ---- memory ----
    def register_memory(self, address: int, size: int,
                        kind: MemoryKind = MemoryKind.DEVICE) -> MemoryRegistration:
        if not self._running:
            raise RuntimeError("engine not running")
        if not address or size <= 0:
            raise ValueError("address and size must be non-zero")
        priv = self._register_memory_impl(address, size, kind)
        self._next_mem_id += 1
        reg = MemoryRegistration(self._next_mem_id, address, size, kind, priv)
        self._mems[reg.id] = reg
        return reg

    def unregister_memory(self, reg: MemoryRegistration) -> None:
        if reg.id not in self._mems:
            raise KeyError("unknown memory registration")
        self._unregister_memory_impl(reg._priv)
        del self._mems[reg.id]

    # ---- target ----
    def open(self, uri: str, scheme: str = "file") -> SsdTarget:
        if not self._running:
            raise RuntimeError("engine not running")
        if uri.startswith("file://"):
            path = uri[len("file://"):]
        else:
            path = uri
        size = self._probe_file_size(path)
        priv = self._open_impl(path, size)
        self._next_target_id += 1
        tgt = SsdTarget(uri, size, self._next_target_id, priv)
        self._targets[tgt.handle_id] = tgt
        return tgt

    def close(self, tgt: SsdTarget) -> None:
        if tgt.handle_id not in self._targets:
            raise KeyError("unknown target")
        self._close_impl(tgt._priv)
        del self._targets[tgt.handle_id]

    @staticmethod
    def _probe_file_size(path: str) -> int:
        import os
        try:
            return os.path.getsize(path)
        except OSError:
            return 0

    # ---- io ----
    def submit(self, requests: List[OffloadRequest],
               stream: Optional[int] = None) -> OffloadHandle:
        if not self._running:
            raise RuntimeError("engine not running")
        self._next_io_id += 1
        # Delegate execution (synchronous in this design; GDS sync APIs block).
        res = self._submit_impl(requests, stream)
        # Store the result keyed by the handle for wait() to return.
        self._pending_results[self._next_io_id] = res
        return OffloadHandle(self._next_io_id, res)

    def wait(self, io: OffloadHandle, timeout_ms: int = 60000) -> ReadResult:
        res = self._pending_results.get(io.handle_id)
        if res is None:
            return ReadResult(False, "unknown io handle")
        return res

    def release_io(self, io: OffloadHandle) -> None:
        self._pending_results.pop(io.handle_id, None)

    # ---- lifecycle ----
    def shutdown(self) -> None:
        if not self._running:
            return
        for tgt_id, tgt in list(self._targets.items()):
            try:
                self._close_impl(tgt._priv)
            except Exception:
                pass
            del self._targets[tgt_id]
        for reg_id, reg in list(self._mems.items()):
            try:
                self._unregister_memory_impl(reg._priv)
            except Exception:
                pass
            del self._mems[reg_id]
        self._shutdown_impl()
        self._running = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
        return False
