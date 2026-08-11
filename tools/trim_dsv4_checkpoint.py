#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Trim a DeepSeek-V4-Flash checkpoint down to a small subset of layers.

Why
---
DeepSeek-V4-Flash is a 43-layer (2 dense + 21 HCA + 20 CSA) MoE model whose
checkpoint weighs ~160 GB on disk (fp4/fp8 compressed MoE experts are the vast
majority). When inference *speed* matters more than output fidelity, you can
drop most transformer layers and get a small checkpoint that fits an 80 GB GPU.

What it does
------------
* keeps only the requested layers (default: the 2 leading dense layers + a
  contiguous block of `--num-hca` HCA + `--num-csa` CSA layers),
* renumbers kept layers consecutively 0..K-1 so SGLang loads them verbatim,
* copies top-level weights (embed / lm_head / norm / hc_head_*),
* drops the MTP (`mtp.*`) next-token-prediction block by default (the main
  model never loads it — it only adds ~3.6 GB and is unneeded for one-step
  decoding),
* rewrites config.json (`num_hidden_layers` + `compress_ratios`) to match.

Dims stay correct
-----------------
* every layer is residual with hidden dim 4096, so removing layers never
  changes the hidden width;
* cross-layer HC state (prev_residual/prev_post/prev_comb) is Optional and
  re-initialized inside each layer's forward, so a contiguous kept block
  reloads cleanly (kept layer 0 restarts the chain);
* embed -> layers -> norm/hc_head -> lm_head are all model-level, so the final
  logits shape [B, vocab] is preserved.

Usage
-----
    python tools/trim_dsv4_checkpoint.py \
        --src  /mnt/sda/DeepSeek-V4-Flash \
        --dst  /mnt/sda/DeepSeek-V4-Flash-trim8 \
        --num-hca 3 --num-csa 3 \
        --dry-run

Run --dry-run first to inspect the layer plan and estimated size. This needs a
python env with `torch` and `safetensors` (the one used to launch sglang).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from typing import Dict, List, Optional, Tuple


def _need_deps():
    for mod in ("safetensors", "torch"):
        try:
            __import__(mod)
        except ImportError:
            sys.exit(f"Need `{mod}` in this python env. Aborting.")


def discover_shards(src: str) -> List[str]:
    shards = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
    assert shards, f"no .safetensors files under {src}"
    return [os.path.join(src, s) for s in shards]


LAYER_RE = re.compile(r"^(?:model\.)?layers\.(\d+)\.")


def layer_id_of(name: str):
    m = LAYER_RE.match(name)
    return int(m.group(1)) if m else None


_ELEM_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "F4": 1,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1,
    "U8": 1, "U16": 2, "U32": 4, "U64": 8, "BOOL": 1,
}


def _read_safetensors_header(path: str) -> dict:
    """Read a safetensors header with ONLY the stdlib (no torch/numpy)."""
    import struct

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def analyze(src: str):
    """Pure-stdlib scan: {shard_abs: {name: {"dtype","shape"}}} + {layer_id: bytes}."""
    headers: Dict[str, dict] = {}
    layer_bytes: Dict[int, int] = {}
    for sh in discover_shards(src):
        hdr = _read_safetensors_header(sh)
        meta = {}
        for k, v in hdr.items():
            el = _ELEM_BYTES.get(v["dtype"], 1)
            for s in v["shape"]:
                el *= s
            meta[k] = {"dtype": v["dtype"], "shape": tuple(v["shape"])}
            lid = layer_id_of(k)
            if lid is not None:
                layer_bytes[lid] = layer_bytes.get(lid, 0) + el
        headers[sh] = meta
    return headers, layer_bytes


def classify(layer_ids, compress_ratios):
    return {
        lid: {0: "dense", 4: "hca", 128: "csa"}.get(
            (compress_ratios[lid] if lid < len(compress_ratios) else 0),
            "dense",
        )
        for lid in layer_ids
    }


def build_plan(layer_ids, compress_ratios, num_hca, num_csa):
    cls = classify(layer_ids, compress_ratios)
    keep: List[int] = []

    # 1) leading dense layers
    i = 0
    while i < len(layer_ids) and cls[layer_ids[i]] == "dense":
        keep.append(layer_ids[i])
        i += 1

    # 2) a contiguous window after the leading dense block with exactly
    #    num_hca HCA + num_csa CSA layers
    seq = layer_ids[i:]
    block: Optional[List[int]] = None
    for start in range(len(seq)):
        hca = csa = 0
        window: List[int] = []
        for j in range(start, len(seq)):
            c = cls[seq[j]]
            if c == "hca":
                hca += 1
            elif c == "csa":
                csa += 1
            else:
                break
            window.append(seq[j])
            if hca == num_hca and csa == num_csa:
                block = window
                break
        if block is not None:
            break
    if block is None:
        raise SystemExit(
            f"no contiguous block with {num_hca} HCA + {num_csa} CSA. "
            f"classes={cls}"
        )
    keep.extend(block)

    # 3) trailing dense layers (generic; none in DeepSeek-V4)
    for lid in layer_ids:
        if lid > block[-1] and cls[lid] == "dense":
            keep.append(lid)

    return sorted(keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=None)
    ap.add_argument("--dst", default=None)
    ap.add_argument("--num-hca", type=int, default=3)
    ap.add_argument("--num-csa", type=int, default=3)
    ap.add_argument("--keep-mtp", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--fix-index",
        metavar="DST",
        help=(
            "Only rebuild model.safetensors.index.json in an already-trimmed "
            "checkpoint directory (mirrors the shard layout this tool writes). "
            "Used to repair a checkpoint whose index still references the "
            "original 46 shards. Still needs --src-index (the original "
            "checkpoint's index) and --num_hca/--num_csa to recompute the plan."
        ),
    )
    ap.add_argument(
        "--src-index",
        help="Original checkpoint's model.safetensors.index.json (for --fix-index).",
    )
    args = ap.parse_args()

    if args.fix_index:
        # ---- repair mode: only rebuild the index of an existing trimmed dir ----
        src_index = args.src_index
        if not src_index or not os.path.isfile(src_index):
            sys.exit(
                "--fix-index requires --src-index pointing at the ORIGINAL "
                "checkpoint's model.safetensors.index.json"
            )
        dst_dir = args.fix_index
        dst_cfg = os.path.join(dst_dir, "config.json")
        if not os.path.isfile(dst_cfg):
            sys.exit(f"--fix-index needs an existing checkpoint dir with config.json: {dst_dir}")
        with open(dst_cfg) as f:
            dcfg = json.load(f)
        n_hidden = int(dcfg.get("num_hidden_layers", 0))
        keep = list(range(n_hidden))  # default trim plan numbers layers 0..K-1 consecutively
        new_idx = {o: n for n, o in enumerate(keep)}
        rebuild_index(
            src_index=src_index,
            dst_index=os.path.join(dst_dir, "model.safetensors.index.json"),
            keep=keep,
            new_idx=new_idx,
            drop_mtp=not args.keep_mtp,
        )
        print(f"[fix-index] rebuilt {os.path.join(dst_dir, 'model.safetensors.index.json')}")
        return

    if not args.dry_run:
        _need_deps()

    if not args.src or not args.dst:
        sys.exit("--src and --dst are required (unless using --fix-index).")
    src, dst = args.src, args.dst
    headers, layer_bytes = analyze(src)
    layer_ids = sorted(layer_bytes.keys())
    cfg = json.load(open(os.path.join(src, "config.json")))
    compress_ratios = cfg.get("compress_ratios") or [0] * len(layer_ids)

    keep = build_plan(layer_ids, compress_ratios, args.num_hca, args.num_csa)
    new_idx = {orig: n for n, orig in enumerate(keep)}
    keep_set = set(keep)
    kept_bytes = sum(layer_bytes[i] for i in keep)

    print("=== DeepSeek-V4 trim plan ===")
    print(f"layer_ids         : {layer_ids}")
    print(f"layer classes     : {classify(layer_ids, compress_ratios)}")
    print(f"keeping {len(keep)} : {keep}")
    print(f"renumbered to     : {[new_idx[i] for i in keep]}")
    print(f"kept layer GB     : {kept_bytes/1e9:.2f}  (of {sum(layer_bytes.values())/1e9:.2f})")
    print(f"est. total (w/top): {(kept_bytes + 2.1e9)/1e9:.2f} GB (mtp: {3.6 if args.keep_mtp else 0} GB)")

    if args.dry_run:
        print("\n[dry-run] nothing written.")
        return

    from safetensors import safe_open
    import safetensors.torch

    # ---- collect entries: (dst_name, src_name, src_shard_abs) ----
    top_entries: List[Tuple[str, str, str]] = []
    layer_entries: Dict[int, List[Tuple[str, str, str]]] = {
        n: [] for n in range(len(keep))
    }
    for sh, tensors in headers.items():
        for src_name in tensors:
            lid = layer_id_of(src_name)
            if lid is not None:
                if lid not in keep_set:
                    continue
                new = new_idx[lid]
                dst_name = LAYER_RE.sub(f"layers.{new}.", src_name, count=1)
                layer_entries[new].append((dst_name, src_name, sh))
            elif src_name.startswith("mtp"):
                if args.keep_mtp:
                    top_entries.append((src_name, src_name, sh))
            else:
                top_entries.append((src_name, src_name, sh))

    dst_tmp = dst + ".tmp"
    if os.path.exists(dst_tmp):
        shutil.rmtree(dst_tmp)
    os.makedirs(dst_tmp, exist_ok=True)

    total_shards = len(keep) + 1

    def save(entries, shard_no):
        payload = {}
        for dst_name, src_name, src_shard in entries:
            with safe_open(src_shard, framework="pt") as f:
                payload[dst_name] = f.get_tensor(src_name)
        fname = f"model-{shard_no:05d}-of-{total_shards:05d}.safetensors"
        safetensors.torch.save_file(payload, os.path.join(dst_tmp, fname))
        del payload
        print(f"  wrote {fname} ({len(entries)} tensors)")

    print("\nWriting shards...")
    save(top_entries, 1)
    for n in range(len(keep)):
        save(layer_entries[n], n + 2)

    # copy non-shard files + rewrite config.json
    for f in os.listdir(src):
        p = os.path.join(src, f)
        if f.endswith(".safetensors") or os.path.isdir(p):
            continue
        if f == "model.safetensors.index.json":
            continue  # rebuilt below from the source index
        shutil.copy2(p, os.path.join(dst_tmp, f))

    cfg["num_hidden_layers"] = len(keep)
    cfg["compress_ratios"] = [compress_ratios[i] for i in keep]
    cfg.pop("num_nextn_predict_layers", None)
    with open(os.path.join(dst_tmp, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    # Rebuild model.safetensors.index.json from the source index + keep map.
    # This is a pure name remap (no shard bytes needed), so it always matches
    # the freshly written shards: top keys -> shard 1, kept layer n -> shard n+2.
    rebuild_index(
        src_index=os.path.join(src, "model.safetensors.index.json"),
        dst_index=os.path.join(dst_tmp, "model.safetensors.index.json"),
        keep=keep,
        new_idx=new_idx,
        drop_mtp=not args.keep_mtp,
    )

    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.rename(dst_tmp, dst)
    print(f"\nDone. Wrote trimmed checkpoint to {dst}")


def rebuild_index(src_index, dst_index, keep, new_idx, drop_mtp=True):
    """Rebuild the safetensors index for the trimmed checkpoint from the
    *source* index (pure renaming — never touches shard bytes)."""
    keep_set = set(keep)
    n_total = len(keep) + 1
    with open(src_index) as f:
        src = json.load(f)
    src_wm = src.get("weight_map", {})
    new_wm: Dict[str, str] = {}
    n_layer_keys = 0
    for name, _shard in src_wm.items():
        m = LAYER_RE.match(name)  # layers.N. or model.layers.N.
        if m:
            orig = int(m.group(1))
            if orig not in keep_set:
                continue
            new = new_idx[orig]
            new_name = LAYER_RE.sub(f"layers.{new}.", name, count=1)
            new_wm[new_name] = f"model-{new + 2:05d}-of-{n_total:05d}.safetensors"
            n_layer_keys += 1
        elif name.startswith("mtp"):
            if drop_mtp:
                continue
            new_wm[name] = f"model-00001-of-{n_total:05d}.safetensors"
        else:
            new_wm[name] = f"model-00001-of-{n_total:05d}.safetensors"
    # total_size is cosmetic metadata only; try real file sizes for accuracy.
    total = 0
    for sh in sorted(_shards_in(dst_index)):
        try:
            total += os.path.getsize(sh)
        except OSError:
            total = None
            break
    if total is None:
        src_total = src.get("metadata", {}).get("total_size", 0)
        total = int(src_total * (n_layer_keys / max(len(src_wm), 1)))
    new_index = {"metadata": {"total_size": total}, "weight_map": new_wm}
    with open(dst_index, "w") as f:
        json.dump(new_index, f, indent=2)
    print(f"  rebuilt model.safetensors.index.json: {len(new_wm)} tensors, {total/1e9:.2f} GB")


def _shards_in(index_path):
    """Shard files in the same directory as index_path."""
    d = os.path.dirname(index_path)
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".safetensors"))


if __name__ == "__main__":
    main()
