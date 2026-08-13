"""
test_proxy_vs_dsv4_paged_layout.py
==================================

Decisive check: does ``SsdHostPoolProxy``'s *flat slot* indexing
(``host_slot * item_size_bytes``) match the REAL ``DeepSeekV4PagedHostPool``
memory layout used by DeepSeek-V4 / DeepSeek-V4-Flash HiSparse?

Real layout (see deepseek_v4_memory_pool.create_buffer and
DeepSeekV4PagedHostPool):
  * one KV "page" holds ``page_size`` tokens
  * a page row in host memory is ``bytes_per_page_padded`` bytes, where
      bytes_per_page_non_padded = page_size * 584
      bytes_per_page_padded   = ceil_div(bytes_per_page_non_padded, 576) * 576
  * tokens are PACKED inside the row ([value0..value63][scale0..scale63] style),
    NOT stored as ``page_size`` contiguous 584-byte cells.

The proxy assumes each host slot is a contiguous ``item_size_bytes``=584 range at
byte ``slot*584``. We replicate the real (paged+padded) host pool and check, for
both SSD backends, whether a backup->load roundtrip preserves the KV.

Run:  python3 -m ssd_offload.tests.test_proxy_vs_dsv4_paged_layout
      # optional --backend cpu-bypass|gds|all
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

from sglang.srt.mem_cache.hisparse_ssd_pool import SsdHostPoolProxy


def ceil_div(a, b):
    return (a + b - 1) // b


class Dsv4PagedMockPool:
    """Mirrors DeepSeekV4PagedHostPool (layer_first) faithfully."""

    def __init__(self, num_host_pages, page_size, layer_num, item_bytes=584):
        self.item_bytes = item_bytes
        self.page_size = page_size
        self.layer_num = layer_num
        # layout constants from deepseek_v4_memory_pool.create_buffer
        bytes_per_page_non_padded = page_size * item_bytes
        self.bytes_per_page_padded = ceil_div(bytes_per_page_non_padded, 576) * 576
        self.slot_page_size = page_size
        self.size = num_host_pages * page_size  # token-granular host capacity
        self.num_host_pages = num_host_pages
        self.token_stride_size = item_bytes
        # stride*index  (proxy reads .nbytes / data_ptr via torch)
        self.kv_buffer = [
            torch.zeros((num_host_pages, self.bytes_per_page_padded), dtype=torch.uint8)
            for _ in range(layer_num)
        ]
        self._scratch_pages = [
            torch.zeros((num_host_pages, self.bytes_per_page_padded), dtype=torch.uint8)
            for _ in range(layer_num)
        ]
        self._avail = self.size

    # ---- pool interface used by coordinator / proxy ----
    def alloc_paged_token_slots(self, *a, **k):
        return torch.arange(self.size, dtype=torch.int64)

    def allocated_host_indices(self, *a, **k):
        return torch.arange(self.size, dtype=torch.int64)

    def free(self, indices):
        pass

    def available_size(self):
        return self._avail

    def destroy(self):
        pass

    # ---- backup/load as the real kernel would (fill paged host rows) ----
    def backup_from_device_all_layer(self, device_pool, host_indices, device_indices, io_backend):
        # Real kernel copies device paged rows into host paged rows verbatim.
        for lid in range(self.layer_num):
            self.kv_buffer[lid].copy_(self._scratch_pages[lid])

    def load_to_device_per_layer(self, device_pool, host_indices, device_locs, layer_id, io_backend):
        pass

    # ---- test helpers: write a distinctive byte per host token cell ----
    def fill_scratch_distinct(self):
        """Put a unique byte at the START of each token cell inside each paged row.

        For DSV4 each row is packed; we place byte value (page*7 + within) at a
        known location inside the row to detect mis-placement after roundtrip.
        """
        for lid in range(self.layer_num):
            buf = self._scratch_pages[lid]
            for pg in range(self.num_host_pages):
                for t in range(self.page_size):
                    # token t of page pg starts somewhere in the row; use offset t*item
                    # (proxy would read flat slot (pg*page_size+t)*item)
                    buf[pg, t * self.item_bytes] = (
                        (lid * 101 + pg * 7 + t) % 256
                    )

    def check_restored(self):
        for lid in range(self.layer_num):
            buf = self.kv_buffer[lid]
            for pg in range(self.num_host_pages):
                for t in range(self.page_size):
                    exp = (lid * 101 + pg * 7 + t) % 256
                    got = int(buf[pg, t * self.item_bytes])
                    if got != exp:
                        return lid, pg, t, got, exp
        return None


def run(backend, num_pages, page_size, layers):
    tmp = tempfile.mkdtemp(prefix="dsv4l_")
    pool = Dsv4PagedMockPool(num_pages, page_size, layers)
    proxy = SsdHostPoolProxy(
        dram_pool=pool,
        offload_backend=backend,
        offload_path=tmp,
        item_size_bytes=pool.item_bytes,
        layer_num=layers,
    )
    idx = torch.arange(pool.size, dtype=torch.int64)
    pool.fill_scratch_distinct()
    proxy.backup_from_device_all_layer(pool, idx, idx, "kernel")
    # clobber host rows, then reload per layer -> simulated swap-in
    for lid in range(layers):
        pool.kv_buffer[lid].zero_()
    for lid in range(layers):
        proxy.load_to_device_per_layer(pool, idx, idx, lid, "kernel")
    bad = pool.check_restored()
    proxy.destroy()
    shutil.rmtree(tmp, ignore_errors=True)

    if bad is None:
        print(f"  [{backend}] DSV4 paged layout roundtrip: PASS")
        return True
    lid, pg, t, got, exp = bad
    print(
        f"  [{backend}] DSV4 paged layout MISMATCH layer={lid} page={pg} "
        f"token={t}: got {got} expected {exp}  -> data CORRUPTED"
    )
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="all")
    ap.add_argument("--num-pages", type=int, default=8)
    ap.add_argument("--page-size", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    args = ap.parse_args()

    backends = ["cpu-bypass", "gds"] if args.backend == "all" else [args.backend]
    ok = True
    for b in backends:
        ok &= run(b, args.num_pages, args.page_size, args.layers)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
