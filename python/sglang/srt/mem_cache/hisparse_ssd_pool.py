"""
hisparse_ssd_pool.py
=====================

A drop-in replacement for the HiSparse **host-side KV pool** that persists the
full KV cache to NVMe SSD (instead of keeping it only in CPU pinned DRAM),
while retaining a device-addressable DRAM shadow for the swap-in CUDA kernel.

Why a DRAM shadow is needed
---------------------------
The HiSparse decode swap-in kernel
(``load_cache_to_device_buffer_mla`` / ``_dsv4_mla``) reads the "host cache"
from ``mem_pool_host.kv_buffer[layer_id]``, which must be a **device-addressable
host (pinned) buffer**. On hosts without GPU↔NVMe PCI P2P (this machine — a
system disk under an Intel VMD bridge, IOMMU on/pt, cuFile reports
``CU_FILE_IO_NOT_SUPPORTED`` for direct IO), an NVMe file cannot be read
directly by the kernel. Therefore the SSD-backed pool:

  * keeps a pinned DRAM ``kv_buffer`` as the kernel-visible working copy, and
  * persists the authoritative full KV to an NVMe file via ``ssd_offload``.

Data flow:

    backup ("eager backup" / staging):
        device KV -> DRAM shadow (kernel)  ->  NVMe file  (ssd_offload WRITE)
    load  ("swap-in" / preload):
        NVMe file -> DRAM shadow (ssd_offload READ)  ->  device buffer

API parity
----------
Mirrors the subset of the host-pool interface used by ``HiSparseCoordinator``:
  * ``alloc_paged_token_slots`` / ``allocated_host_indices`` / ``free``
  * ``backup_from_device_all_layer`` / ``load_to_device_per_layer``
  * ``destroy``
  * attributes: ``kv_buffer``, ``size``, ``available_size()``, ``page_size``,
    ``layer_num``, ``token_stride_size``, ... (delegated to the DRAM pool)

All other attributes/methods are delegated to the wrapped DRAM host pool via
``__getattr__``.

Backend selection (``offload_backend``)
---------------------------------------
  * ``cpudram`` : NOT routed here (the caller keeps the original DRAM pool).
  * ``cpu_ssd`` / ``cpu-bypass`` : persist via ``ssd_offload`` AccessMethod.CPU
    (O_DIRECT write-through SSD, bypassing the kernel page cache; always works).
  * ``gds``     : persist via ``ssd_offload`` AccessMethod.GDS (cuFile; falls
    back to compat mode on hosts without PCI P2P).
  * ``scada``   : persist via ``ssd_offload`` AccessMethod.SCADA (requires a
    dedicated NVMe controller; raises NotImplemented on this host).
"""

from __future__ import annotations

import logging
import os
import threading

from ssd_offload.core import MemoryKind, OffloadRequest

# OffloadRequest.direction: 0=READ(target->memory), 1=WRITE(memory->target)
_DIR_READ = 0
_DIR_WRITE = 1

logger = logging.getLogger(__name__)


class SsdHostPoolProxy:
    """A host-pool proxy adding NVMe persistence on top of a DRAM host pool.

    The wrapped ``dram_pool`` provides the kernel-addressable ``kv_buffer`` and
    slot allocation; this proxy intercepts the backup/load paths to additionally
    write/read the authoritative full KV to an NVMe file via ``ssd_offload``.

    The NVMe file is a flat byte array indexed by
    ``(layer_id * host_capacity + host_slot) * item_size_bytes``.
    """

    def __init__(
        self,
        dram_pool,
        offload_backend: str,
        offload_path: str,
        item_size_bytes: int,
        layer_num: int,
    ):
        self._dram_pool = dram_pool
        self._offload_backend = offload_backend
        self._offload_path = offload_path
        self._item_size_bytes = int(item_size_bytes)
        self._layer_num = int(layer_num)
        self._lock = threading.RLock()
        self._closed = False
        self._layer_regs = {}  # layer_id -> (buffer, MemoryRegistration)

        self._host_capacity = int(getattr(dram_pool, "size", 0))
        # Persist each layer as one OPAQUE byte blob. This is layout-agnostic:
        # it works for the flat MLA pool (kv_buffer[lid] = per-layer view) AND
        # for the paged/padded DeepSeek-V4 C4 pool (kv_buffer[lid] =
        # (num_host_pages, bytes_per_page_padded), where tokens are PACKED and
        # NOT contiguous). The swap-in kernel interprets kv_buffer[layer_id]
        # using the pool's own layout, so verbatim restore is always correct.
        self._layer_buffers = list(dram_pool.kv_buffer)
        self._bytes_per_layer = int(self._layer_buffers[0].nbytes)
        self._total_bytes = self._bytes_per_layer * self._layer_num

        self._engine = self._build_engine()

        os.makedirs(offload_path, exist_ok=True)
        self._nvmefile = os.path.join(offload_path, "hisparse_kv.bin")
        self._create_backing_file(self._total_bytes)
        self._target = self._engine.open(self._nvmefile)
        logger.info(
            "SsdHostPoolProxy: backend=%s path=%s total_bytes=%d "
            "(%d layers x %d B/layer, host_capacity=%d)",
            offload_backend, self._nvmefile, self._total_bytes,
            self._layer_num, self._bytes_per_layer, self._host_capacity,
        )

    # ------------------------------------------------------------ engine/file
    def _build_engine(self):
        from ssd_offload import AccessMethod, create_ssd_offload

        method = {
            "cpu_ssd": AccessMethod.CPU,
            "cpu-bypass": AccessMethod.CPU,
            "gds": AccessMethod.GDS,
            "scada": AccessMethod.SCADA,
        }[self._offload_backend]
        return create_ssd_offload(method)

    def _create_backing_file(self, total_bytes: int) -> None:
        if (
            not os.path.exists(self._nvmefile)
            or os.path.getsize(self._nvmefile) < total_bytes
        ):
            with open(self._nvmefile, "wb") as f:
                f.truncate(total_bytes)

    # ------------------------------------------------------------ delegation
    def __getattr__(self, name):
        return getattr(self._dram_pool, name)

    # ------------------------------------------------------------ allocation
    def alloc_paged_token_slots(self, *args, **kwargs):
        return self._dram_pool.alloc_paged_token_slots(*args, **kwargs)

    def allocated_host_indices(self, *args, **kwargs):
        return self._dram_pool.allocated_host_indices(*args, **kwargs)

    def free(self, indices):
        return self._dram_pool.free(indices)

    # ------------------------------------------------------------ backup
    def backup_from_device_all_layer(self, device_pool, host_indices, device_indices, io_backend):
        # 1) Normal DRAM shadow backup (kernel fills kv_buffer).
        self._dram_pool.backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )
        # 2) Persist the backed-up host slots (all layers) to NVMe.
        # The DRAM shadow is filled by an async CUDA kernel (io_backend="kernel");
        # ensure those writes are visible to the CPU before we read the shadow
        # and store it on NVMe (ssd_offload performs host-side O_DIRECT I/O).
        self._sync_device()
        self._bulk_io(host_indices, layer_id=None, direction=_DIR_WRITE)

    def _sync_device(self):
        """Bring any pending device->host shadow writes into coherence before
        the CPU persists the KV to NVMe. No-op when CUDA is unavailable."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.current_stream().synchronize()
        except Exception:
            pass

    # ------------------------------------------------------------ load
    def load_to_device_per_layer(self, device_pool, host_indices, device_locs, layer_id, io_backend):
        # Ensure the DRAM shadow for this layer is up-to-date from NVMe.
        self._bulk_io(host_indices, layer_id=layer_id, direction=_DIR_READ)
        self._dram_pool.load_to_device_per_layer(
            device_pool, host_indices, device_locs, layer_id, io_backend
        )

    # ------------------------------------------------------------ core IO
    def _bulk_io(self, host_indices, layer_id, direction):
        """Persist / restore one or more WHOLE layers to/from NVMe.

        ``layer_id is None`` -> operate on every layer (backup path).
        ``layer_id`` given  -> operate only on that layer (load path).

        Each layer is transferred as one opaque byte blob
        (``kv_buffer[layer_id].nbytes`` bytes) to the NVMe offset
        ``layer_id * bytes_per_layer``. This is intentionally layout-agnostic:
        the DRAM host pool may be flat-per-token (MLA/DSA) or paged/padded with
        packed, non-contiguous tokens (DeepSeek-V4 C4). Verbatim whole-buffer
        transfer preserves whichever layout the swap-in kernel expects, so both
        ``cpu-bypass`` and ``gds`` offloads are correct in every case.

        Backing up the whole layer (rather than only the freshly written host
        slots) is the safe generic choice: it keeps the NVMe copy of each layer
        an exact mirror of the working DRAM shadow, independent of paging/
        padding/compression details.
        """
        if self._closed:
            return
        reqs = []
        with self._lock:
            layers = (
                range(self._layer_num)
                if layer_id is None
                else [layer_id]
            )
            for lid in layers:
                buf = self._dram_pool.kv_buffer[lid]
                reg = self._layer_register(lid, buf)
                nbytes = self._bytes_per_layer
                file_off = lid * nbytes
                reqs.append(
                    OffloadRequest(
                        direction,
                        reg,
                        0,  # memory_offset: start of the whole layer
                        self._target,
                        file_off,
                        nbytes,
                    )
                )
        if reqs:
            self._exec(reqs)

    def _exec(self, reqs):
        io = self._engine.submit(reqs)
        res = self._engine.wait(io)
        self._engine.release_io(io)
        if not res.ok:
            raise RuntimeError(f"ssd_offload failed: {res.error}")

    # ------------------------------------------------------------ registration
    def _layer_register(self, layer_id, buf):
        if layer_id in self._layer_regs and self._layer_regs[layer_id][0] is buf:
            return self._layer_regs[layer_id][1]
        self._unregister_layer(layer_id)
        reg = self._engine.register_memory(
            buf.data_ptr(), buf.numel() * buf.element_size(),
            MemoryKind.PINNED_HOST,
        )
        self._layer_regs[layer_id] = (buf, reg)
        return reg

    def _unregister_layer(self, layer_id):
        entry = self._layer_regs.pop(layer_id, None)
        if entry is not None:
            try:
                self._engine.unregister_memory(entry[1])
            except Exception:
                pass

    # ------------------------------------------------------------ lifecycle
    def destroy(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for layer_id in list(self._layer_regs.keys()):
                self._unregister_layer(layer_id)
            try:
                self._engine.close(self._target)
            except Exception:
                pass
            try:
                self._engine.shutdown()
            except Exception:
                pass
        self._dram_pool.destroy()
