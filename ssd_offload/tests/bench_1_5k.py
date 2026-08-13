"""
ssd_offload/tests/bench_granularity.py  (renamed intent; kept filename bench_1_5k)
===================================================================================

Measure per-I/O throughput at a configurable granularity (default 1.5 KB) for
the two SSD offload methods that must work in Docker on this host:

  * cpu-bypass  (AccessMethod.CPU / O_DIRECT, 4KiB-aligned blocks)
  * gds         (AccessMethod.GDS  / cuFile compat mode)

Motivation: the HiSparse swap path issues many small transfers (KV subsets) that
are far smaller than the storage block. On the CPU/O_DIRECT path each small
transfer is expanded by ``_odoio`` to one or more full 4 KiB aligned blocks
(read-modify-write), so we report BOTH:

  * logical throughput  = size / elapsed  (bytes the caller asked for)
  * physical throughput = aligned_io / elapsed  (bytes actually moved to storage)

and the read/write amplification factor ``phy_bytes / logical_bytes``.

Layout: ``num_ops`` consecutive ``io_bytes``-wide segments at file offsets
``i * io_bytes``. For the CPU backend this makes each segment straddle 4 KiB
blocks (amplification depends on io_bytes vs 4096).

Usage (inside Docker, repo mounted at /workspace/sglang):
    python3 -m ssd_offload.tests.bench_1_5k --backend all --ops 100000 --io-bytes 1536
    python3 -m ssd_offload.tests.bench_1_5k --backend all --ops 100000 --io-bytes 1024
    python3 -m ssd_offload.tests.bench_1_5k --backend all --ops 100000 --io-bytes 10240
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ssd_offload.core import AccessMethod, MemoryKind, OffloadRequest
from ssd_offload.backends import create_ssd_offload

ALIGN = 4096


class _AlignedBuf:
    def __init__(self, size: int, align: int = ALIGN):
        self.size = size
        total = size + align
        self._raw = ctypes.create_string_buffer(total)
        raw = ctypes.addressof(self._raw)
        self._off = (align - raw % align) % align
        self.address = raw + self._off


def align_up(v, a=ALIGN):
    return (v + a - 1) // a * a


def run_bench(backend_name: str, ops: int, io_bytes: int, data_dir: str) -> None:
    method = {
        "cpu-bypass": AccessMethod.CPU,
        "cpu_ssd": AccessMethod.CPU,
        "gds": AccessMethod.GDS,
    }[backend_name]

    engine = create_ssd_offload(method)
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"bench_{method.name}.bin")
    total_bytes = ops * io_bytes
    with open(path, "wb") as f:
        f.truncate(total_bytes)

    tgt = engine.open(path)
    buf = _AlignedBuf(io_bytes)
    ctypes.memset(buf.address, 0xAB, io_bytes)
    mem = engine.register_memory(buf.address, io_bytes, MemoryKind.DEVICE)

    # --- physical I/O accounting (read/write amplification) ---
    # CPU/O_DIRECT path: _odoio expands each request to 4 KiB block(s).
    # For a request at file_off=f, len=L: it reads/writes [aligndown(f),
    # alignup(f+L)). Sum those expanded byte counts across all requests.
    def phy_bytes_for(f, ln):
        return align_up(f + ln) - (f // ALIGN) * ALIGN

    logical_total = total_bytes
    phy_logical = sum(phy_bytes_for(i * io_bytes, io_bytes) for i in range(ops))

    # Build the request list once (contiguous layout i*io_bytes).
    reqs = [OffloadRequest(1, mem, 0, tgt, i * io_bytes, io_bytes) for i in range(ops)]

    def run(direction, req_list):
        reqs2 = [
            OffloadRequest(direction, r.memory, r.memory_offset, r.target,
                           r.target_offset, r.length)
            for r in req_list
        ]
        t0 = time.perf_counter()
        io = engine.submit(reqs2)
        res = engine.wait(io)
        engine.release_io(io)
        dt = time.perf_counter() - t0
        if not res.ok:
            raise RuntimeError(f"{direction}: {res.error}")
        return dt

    def report(direction, dt):
        lgb = logical_total / 1e9
        pgb = phy_logical / 1e9
        print(
            f"      {direction:5s} @ {logical_total/(dt*1e6):8.1f} MB/s (logical "
            f"{lgb:.3f}GB) | phy~{phy_logical/(dt*1e6):8.1f} MB/s ({pgb:.3f}GB) | "
            f"amp~{phy_logical/logical_total:.2f}x"
        )

    print(f"  [{backend_name}] io={io_bytes}B x {ops} ops, "
          f"logical {logical_total/1e6:.1f} MB")
    wt = run(1, reqs)   # WRITE
    print(f"     WRITE:"); report("write", wt)
    rt = run(0, reqs)   # READ
    print(f"     READ :"); report("read ", rt)

    engine.unregister_memory(mem)
    engine.close(tgt)
    engine.shutdown()
    if os.path.exists(path):
        os.remove(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="all",
                    help="cpu-bypass | cpu_ssd | gds | all")
    ap.add_argument("--ops", type=int, default=100000)
    ap.add_argument("--io-bytes", type=int, default=1536,
                    help="per-I/O size in bytes (default 1536 = 1.5KB)")
    ap.add_argument("--data-dir", default="/tmp/ssd_offload_bench")
    args = ap.parse_args()

    backends = {
        "all": ["cpu-bypass", "gds"],
        "cpu-bypass": ["cpu-bypass"],
        "cpu_ssd": ["cpu_ssd"],
        "gds": ["gds"],
    }[args.backend]

    for b in backends:
        try:
            run_bench(b, args.ops, args.io_bytes, args.data_dir)
        except Exception as e:  # noqa: BLE001
            print(f"  [{b}] FAILED: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
# 1KB
docker exec -w /workspace/sglang sglang_zby python3 -m ssd_offload.tests.bench_1_5k --backend all --ops 20000 --io-bytes 1024
# 10KB
docker exec -w /workspace/sglang sglang_zby python3 -m ssd_offload.tests.bench_1_5k --backend all --ops 20000 --io-bytes 10240
"""