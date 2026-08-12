"""
ssd_offload/tests/test_ssd_offload.py
======================================

Tests for the transport-agnostic SSD offload interface.

The same test runner is used for any backend by parameterizing over
``AccessMethod``. This proves the top-level interface is identical regardless
of whether CPU / GDS / SCADA is underneath.

Run:
    python -m ssd_offload.tests.test_ssd_offload --method cpu
    python -m ssd_offload.tests.test_ssd_offload --method gds
    python -m ssd_offload.tests.test_ssd_offload --method scada   # placeholder
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
import sys
import time

# Ensure package importable from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ssd_offload.core import (
    AccessMethod,
    MemoryKind,
    OffloadRequest,
    SsdOffload,
)
from ssd_offload.backends import create_ssd_offload, list_backends


class _AlignedBuf:
    """An aligned buffer whose base address can be registered as memory."""

    def __init__(self, size: int, align: int = 4096):
        self.size = size
        total = size + align
        self._raw = ctypes.create_string_buffer(total)
        raw = ctypes.addressof(self._raw)
        self._off = (align - raw % align) % align
        self.address = raw + self._off

    def read(self, off=0, length=None):
        if length is None:
            length = self.size - off
        return ctypes.string_at(self.address + off, length)

    def write(self, data: bytes, off=0):
        ctypes.memmove(self.address + off, data, len(data))

    def fill(self, v: int, length=None):
        if length is None:
            length = self.size
        ctypes.memset(self.address, v, min(length, self.size))


def _deterministic(n: int, seed: int) -> bytes:
    out = bytearray()
    i = 0
    while len(out) < n:
        out.extend(hashlib.sha256(f"{seed}:{i}".encode()).digest())
        i += 1
    return bytes(out[:n])


def _make_file(path: str, size: int) -> None:
    with open(path, "wb") as f:
        f.truncate(size)


def run_correctness(engine: SsdOffload, path: str) -> None:
    # 1) round-trip write + read, byte-identical
    size = 4 * 1024 * 1024
    _make_file(path, size)
    tgt = engine.open(path)

    buf = _AlignedBuf(size)
    expected = _deterministic(size, seed=42)
    buf.write(expected)
    mem = engine.register_memory(buf.address, size, MemoryKind.DEVICE)

    # WRITE
    wr = OffloadRequest(1, mem, 0, tgt, 0, size)
    io = engine.submit([wr])
    res = engine.wait(io)
    engine.release_io(io)
    assert res.ok, f"write failed: {res.error}"
    assert res.accepted_bytes == size

    # READ
    out = _AlignedBuf(size)
    outmem = engine.register_memory(out.address, size, MemoryKind.DEVICE)
    rd = OffloadRequest(0, outmem, 0, tgt, 0, size)
    io = engine.submit([rd])
    res = engine.wait(io)
    engine.release_io(io)
    assert res.ok, f"read failed: {res.error}"
    assert out.read(0, size) == expected, "round-trip mismatch!"

    engine.unregister_memory(mem)
    engine.unregister_memory(outmem)
    engine.close(tgt)
    print(f"  [PASS] roundtrip {size>>20} MiB (backend={engine.backend_name})")


def run_offset(engine: SsdOffload, path: str) -> None:
    size = 1 * 1024 * 1024
    _make_file(path, 4 * 1024 * 1024)
    tgt = engine.open(path)
    buf = _AlignedBuf(size)
    data = _deterministic(size, seed=7)
    buf.write(data)
    mem = engine.register_memory(buf.address, size, MemoryKind.DEVICE)

    tgt_off, mem_off, length = 512 * 1024, 128 * 1024, 300 * 1024 + 13
    wr = OffloadRequest(1, mem, mem_off, tgt, tgt_off, length)
    io = engine.submit([wr]); res = engine.wait(io); engine.release_io(io)
    assert res.ok, f"offset write failed: {res.error}"

    out = _AlignedBuf(size)
    outmem = engine.register_memory(out.address, size, MemoryKind.DEVICE)
    rd = OffloadRequest(0, outmem, mem_off, tgt, tgt_off, length)
    io = engine.submit([rd]); res = engine.wait(io); engine.release_io(io)
    assert res.ok, f"offset read failed: {res.error}"
    assert out.read(mem_off, length) == data[mem_off:mem_off + length], "offset mismatch"

    engine.unregister_memory(mem); engine.unregister_memory(outmem)
    engine.close(tgt)
    print(f"  [PASS] offset+unaligned length (backend={engine.backend_name})")


def run_bandwidth(engine: SsdOffload, path: str, size_gb: float, chunk_kb: int) -> None:
    size = int(size_gb * 1024 * 1024 * 1024)
    chunks = chunk_kb * 1024
    _make_file(path, size)
    tgt = engine.open(path)

    buf = _AlignedBuf(chunks)
    buf.fill(0xAB)
    mem = engine.register_memory(buf.address, chunks, MemoryKind.DEVICE)

    n = size // chunks
    # WRITE bandwidth
    reqs = [OffloadRequest(1, mem, 0, tgt, i * chunks, chunks) for i in range(n)]
    t0 = time.perf_counter()
    io = engine.submit(reqs); res = engine.wait(io); engine.release_io(io)
    wt = time.perf_counter() - t0
    assert res.ok, f"bw write failed: {res.error}"

    # READ bandwidth
    reqs = [OffloadRequest(0, mem, 0, tgt, i * chunks, chunks) for i in range(n)]
    t0 = time.perf_counter()
    io = engine.submit(reqs); res = engine.wait(io); engine.release_io(io)
    rt = time.perf_counter() - t0
    assert res.ok, f"bw read failed: {res.error}"

    engine.unregister_memory(mem); engine.close(tgt)
    print(f"  [BENCH] backend={engine.backend_name} {size_gb} GiB chunk={chunk_kb} KiB")
    print(f"          write {size/1e9:.2f} GB .. {size/(wt*1e6):.1f} MB/s")
    print(f"          read  {size/1e9:.2f} GB .. {size/(rt*1e6):.1f} MB/s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["cpu", "gds", "scada"], default="cpu")
    ap.add_argument("--data-dir", default="/tmp/ssd_offload_test")
    ap.add_argument("--size-gb", type=float, default=0.5)
    ap.add_argument("--chunk-kb", type=int, default=4096)
    ap.add_argument("--bandwidth", action="store_true", help="run bandwidth test")
    args = ap.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    method = {"cpu": AccessMethod.CPU, "gds": AccessMethod.GDS,
              "scada": AccessMethod.SCADA}[args.method]

    print(f"Selected backend: {args.method} ({method.name})")

    # SCADA is a placeholder on this host -> expects NotImplementedError.
    if args.method == "scada":
        try:
            create_ssd_offload(method)
            print("  [FAIL] SCADA backend unexpectedly instantiated")
            return 1
        except NotImplementedError as e:
            print(f"  [PASS] SCADA placeholder correctly raises NotImplementedError")
            print(f"         reason: {e}")
            return 0

    try:
        engine = create_ssd_offload(method)
    except RuntimeError as e:
        print(f"  [SKIP] backend '{args.method}' unavailable: {e}")
        print("         (fall back to --method cpu if you only need semantics)")
        return 0

    print(f"Backend capabilities: {engine.capabilities.describe()}")
    path = os.path.join(args.data_dir, f"offload_{args.method}.bin")

    try:
        run_correctness(engine, path)
        run_offset(engine, path)
        if args.bandwidth:
            run_bandwidth(engine, path, args.size_gb, args.chunk_kb)
    finally:
        engine.shutdown()

    for p in (path,):
        if os.path.exists(p):
            os.remove(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
