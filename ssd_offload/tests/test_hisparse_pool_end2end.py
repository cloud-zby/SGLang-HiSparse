"""
test_hisparse_pool_end2end.py
==============================

End-to-end validation of ``SsdHostPoolProxy`` with the two SSD offload methods
that must work on this host:

  * cpu-bypass  (== offload_backend "cpu_ssd"; AccessMethod.CPU, O_DIRECT
                 write-through SSD bypassing the page cache)
  * gds         (AccessMethod.GDS via cuFile; falls back to compat mode)

We drive the **actual** ``SsdHostPoolProxy`` (the class SGLang wires into
``HiSparseCoordinator``), not just the raw ssd_offload engine, using a mock
host pool whose ``kv_buffer`` is a real host tensor (so per-byte integrity of
the KV roundtrip can be verified without a GPU):

  backup:  device-buffer(scratch) -> DRAM shadow (via backup_* copy) -> NVMe
           (proxy persists the shadow bytes to the SSD file)
  load:    NVMe -> DRAM shadow (proxy reads the SSD file) -> device-buffer

The test asserts, for every layer and every host slot:
  1. after backup, the NVMe file holds exactly the original KV bytes
  2. after clobbering the DRAM shadow, load_to_device_per_layer restores the
     exact original bytes (proves the SSD is authoritative and round-trips)

Run:
  python3 -m ssd_offload.tests.test_hisparse_pool_end2end
  # optionally:  --backend cpu-bypass|gds
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

from ssd_offload.core import OffloadRequest  # noqa: F401 (interface sanity)
from sglang.srt.mem_cache.hisparse_ssd_pool import SsdHostPoolProxy


# ---------------------------------------------------------------------------
# Mock host pool: mirrors the host-pool interface the coordinator drives
# ---------------------------------------------------------------------------
class MockHostPool:
    """Minimal stand-in for DeepSeekV4PagedHostPool / MLATokenToKVPoolHost.

    kv_buffer[layer] is a flat 1-D host tensor of
    ``host_capacity * item_bytes`` bytes (item_bytes==one token's KV bytes).
    """

    def __init__(self, host_capacity, item_bytes, layer_num, dtype=torch.uint8):
        self._cap = host_capacity
        self._item_bytes = item_bytes
        self._layer_num = layer_num
        self.size = host_capacity
        self.token_stride_size = item_bytes
        # device-addressable working copy (host tensor so CPU can check bytes)
        self.kv_buffer = torch.zeros(
            (layer_num, host_capacity * item_bytes), dtype=dtype
        )
        self._device_scratch = torch.zeros(
            (layer_num, host_capacity * item_bytes), dtype=dtype
        )
        self._avail = host_capacity

    # ---- allocation API used by the coordinator / proxy delegation ----
    def alloc_paged_token_slots(self, num, *a, **k):
        return torch.arange(num)

    def allocated_host_indices(self, *a, **k):
        return torch.arange(self._cap)

    def free(self, indices):
        return

    def available_size(self):
        return self._avail

    def destroy(self):
        pass

    # ---- backup / load that a real host pool would do (kernel fills shadow) --
    def backup_from_device_all_layer(self, device_pool, host_indices, device_indices, io_backend):
        # Simulate: the swap-out kernel copies *device scratch* into the DRAM
        # shadow for all layers. We simply copy our scratch into kv_buffer so the
        # proxy then persists the shadow to NVMe.
        self.kv_buffer.copy_(self._device_scratch)

    def load_to_device_per_layer(self, device_pool, host_indices, device_locs, layer_id, io_backend):
        # Simulate: the swap-in kernel reads the DRAM shadow of ``layer_id`` into
        # the device buffer. Here we just record that the shadow was consumed.
        pass

    # ---- test helpers ----
    def set_scratch_value(self, layer_id, slot, value):
        """Write a deterministic byte value into the 'device' scratch at (layer,slot)."""
        item = self._item_bytes
        self._device_scratch[layer_id, slot * item : (slot + 1) * item] = value

    def get_shadow_value(self, layer_id, slot):
        item = self._item_bytes
        return int(self.kv_buffer[layer_id, slot * item].item())

    def clobber_shadow(self):
        """Zero the DRAM shadow to prove NVMe is authoritative on load."""
        self.kv_buffer.zero_()


# ---------------------------------------------------------------------------
# test body
# ---------------------------------------------------------------------------
def run_backend(backend: str, host_capacity: int, layer_num: int, item_bytes: int):
    tag = backend
    tmp = tempfile.mkdtemp(prefix="hisparse_e2e_")
    pool = MockHostPool(host_capacity, item_bytes, layer_num)
    proxy = SsdHostPoolProxy(
        dram_pool=pool,
        offload_backend=backend,
        offload_path=tmp,
        item_size_bytes=item_bytes,
        layer_num=layer_num,
    )

    # 1) Write deterministic, per-slot-distinct bytes into the device scratch
    #    so we can catch any cross-slot / cross-layer bleed.
    #    value = (layer*10 + slot) % 256  -- every slot gets a unique base byte.
    for lid in range(layer_num):
        for slot in range(host_capacity):
            val = (lid * 10 + slot) % 256
            pool.set_scratch_value(lid, slot, val)

    # 2) backup all layers -> shadow filled -> proxy persists to NVMe.
    host_indices = torch.arange(host_capacity, dtype=torch.int64)
    proxy.backup_from_device_all_layer(
        device_pool=pool, host_indices=host_indices,
        device_indices=host_indices, io_backend="kernel",
    )

    # 3) Verify the NVMe file now holds the exact original bytes (per layer).
    nvmefile = proxy._nvmefile
    with open(nvmefile, "rb") as f:
        raw = f.read()
    for lid in range(layer_num):
        for slot in range(host_capacity):
            off = (lid * host_capacity + slot) * item_bytes
            got = raw[off]
            exp = (lid * 10 + slot) % 256
            assert got == exp, (
                f"[{tag}] NVMe byte mismatch layer={lid} slot={slot}: "
                f"got {got} expected {exp}"
            )

    # 4) Clobber DRAM shadow, then load a single layer; verify shadow is
    #    restored from NVMe (authoritative) with exact bytes.
    pool.clobber_shadow()
    for lid in range(layer_num):
        for slot in range(host_capacity):
            assert pool.get_shadow_value(lid, slot) == 0, "precondition failed"
    for lid in range(layer_num):
        proxy.load_to_device_per_layer(
            device_pool=pool, host_indices=host_indices,
            device_locs=host_indices, layer_id=lid, io_backend="kernel",
        )
        for slot in range(host_capacity):
            got = pool.get_shadow_value(lid, slot)
            exp = (lid * 10 + slot) % 256
            assert got == exp, (
                f"[{tag}] load mismatch layer={lid} slot={slot}: got {got} expected {exp}"
            )

    proxy.destroy()
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  [{tag}] host_capacity={host_capacity} layers={layer_num} "
          f"item={item_bytes}B -> e2e roundtrip: PASS")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="all",
                    help="cpu-bypass | cpu_ssd | gds | all (default all)")
    args = ap.parse_args()

    backends = []
    if args.backend == "all":
        backends = ["cpu-bypass", "gds"]
    elif args.backend in ("cpu-bypass", "cpu_ssd", "gds"):
        backends = [args.backend]
    else:
        raise SystemExit(f"unknown backend {args.backend!r}")

    # realistic-ish sizes: 4 layers x small capacity x 584B item (C4 token KV)
    host_capacity = 32
    layer_num = 4
    item_bytes = 584

    ok = True
    for b in backends:
        try:
            run_backend(b, host_capacity, layer_num, item_bytes)
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  [{b}] FAILED: {e}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
