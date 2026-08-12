"""
ssd_offload/backends
====================

Pluggable backends implementing the top-level ``SsdOffload`` interface.

* ``cpu_backend.py``  -- 方式三: POSIX bounce-buffer path (always available,
  reference/fallback).
* ``gds_backend.py``  -- 方式二: GPUDirect Storage via the official cuFile API
  (real). Falls back to compat mode when P2P is unavailable.
* ``scada_backend.py``-- 方式一: GPU-driven NVMe (Tutti-style). PLACEHOLDER on
  this host (No dedicated NVMe); raises NotImplemented until real hardware.
"""

from .factory import create_ssd_offload, list_backends

__all__ = ["create_ssd_offload", "list_backends"]

