"""
ssd_offload/backends/factory.py
================================

Backend selection factory.

``create_ssd_offload(method, **kwargs)`` returns the appropriate top-level
``SsdOffload`` engine for the requested ``AccessMethod``.

Selection rules
---------------
* CPU  -> ``CpuSsdOffload`` (always available)
* GDS  -> ``GdsSsdOffload`` (requires libcufile; if unavailable at import, a
          clear error is raised; the caller may catch and fall back to CPU)
* SCADA-> ``ScadaSsdOffload`` (placeholder; raises NotImplemented on this host)
"""

from __future__ import annotations

from ..core import AccessMethod


def list_backends() -> dict:
    """Return {name: AccessMethod} describable backends for the current host."""
    return {
        "cpu": AccessMethod.CPU,
        "gds": AccessMethod.GDS,
        "scada": AccessMethod.SCADA,
    }


def create_ssd_offload(method: AccessMethod, **kwargs) -> "SSDOffloadT":
    """Instantiate the engine for the requested access method.

    Returns an instance of the top-level ``SsdOffload`` subclass.
    """
    if method == AccessMethod.CPU:
        from .cpu_backend import CpuSsdOffload
        return CpuSsdOffload(**kwargs)

    if method == AccessMethod.GDS:
        from .gds_backend import GdsSsdOffload
        return GdsSsdOffload(**kwargs)

    if method == AccessMethod.SCADA:
        from .scada_backend import ScadaSsdOffload
        return ScadaSsdOffload(**kwargs)

    raise ValueError(f"unknown access method: {method}")


# Type alias for readability (avoid circular import at module top).
SSDOffloadT = "object"
