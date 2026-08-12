"""
ssd_offload/backends/scada_backend.py
======================================

方式一: SCADA (Scaled Accelerated Data Access) — GPU-driven NVMe.

In this family (Tutti is an example), the GPU threads themselves drive the
NVMe controller: they parse LBAs, write SQEs, ring the doorbell, and poll the
completion queue — the CPU appears O(1) per batch, not per I/O.

This backend is a PLACEHOLDER on the current host:
  * Requires a DEDICATED (non-system) NVMe controller exposed as /dev/ssnvme*
    and the snvme kernel module + daemon (Tutti-style deployment).
  * The current machine's only NVMe is the system disk under an Intel VMD
    bridge, bound to the standard `nvme` driver, so it cannot be safely
    handed to a GPU-direct driver.

Whenever real hardware is available, implement ``_open_impl`` /
``_register_memory_impl`` / ``_submit_impl`` etc. behind the SAME top-level
interface — no caller change needed. Until then every operation raises
NotImplementedError.

The extension points mirror the other backends exactly, so filling them in is
mechanical.
"""

from __future__ import annotations

from typing import List, Optional

from ..core import (
    AccessMethod,
    BackendCapabilities,
    MemoryKind,
    OffloadRequest,
    ReadResult,
    SsdOffload,
)


class ScadaSsdOffload(SsdOffload):
    backend_name = "scada"
    capabilities = BackendCapabilities(
        method=AccessMethod.SCADA,
        supports_device_direct=True,   # GPU threads drive NVMe
        supports_host_bounce=False,
        requires_pinned_memory=True,
    )

    def __init__(self, device: str = "/dev/ssnvme0", **kwargs):
        super().__init__()
        self.device = device
        # TODO(SCADA): on a dedicated-NVMe host, load the Tutti snvme runtime,
        # register GPU buffers for peer-DMA, and open raw NVMe targets here.
        raise NotImplementedError(
            "SCADA backend requires a dedicated (non-system) NVMe controller "
            "exposed as /dev/ssnvme* (Tutti/snvme deployment). "
            "This host's only NVMe is the system disk under an Intel VMD "
            "bridge and cannot be safely handed to a GPU-direct driver."
        )

    def _open_impl(self, path: str, logical_size: int):
        raise NotImplementedError("SCADA backend unavailable on this host")

    def _close_impl(self, priv):
        raise NotImplementedError("SCADA backend unavailable on this host")

    def _register_memory_impl(self, address, size, kind):
        raise NotImplementedError("SCADA backend unavailable on this host")

    def _unregister_memory_impl(self, priv):
        raise NotImplementedError("SCADA backend unavailable on this host")

    def _submit_impl(self, reqs: List[OffloadRequest], stream: Optional[int]) -> ReadResult:
        raise NotImplementedError("SCADA backend unavailable on this host")

    def _shutdown_impl(self):
        raise NotImplementedError("SCADA backend unavailable on this host")
