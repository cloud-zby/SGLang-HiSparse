# HiSparse: Hierarchical Sparse Attention 实现分析

> 本文档基于以下资料整理：
> - [HiSparse 官方文档（SGLang）](https://docs.sglang.io/docs/advanced_features/hisparse_guide)
> - [HiSparse 官方博客（LMSYS Org）](https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/)
> - SGLang 源码实际实现（`python/sglang/srt/` 下 `hisparse`、`sparsity`、相关 attention 后端）

---

## 1. 背景与动机

### 1.1 稀疏注意力为何仍受制于显存容量

长上下文 LLM 推理中，自注意力是计算与访存瓶颈。稀疏注意力（如 top-k 选择）能显著降低计算和 IO 开销，但**仍无法消除显存容量瓶颈**：即使每一步只有少数 KV 条目被激活，全量上下文 KV cache 仍须常驻 GPU HBM 以支持高速访问。结果，稀疏注意力往往**受显存容量约束**而非计算约束，限制了可并发的请求数（batch size）与整体吞吐。

### 1.2 HiSparse 的解决思路

HiSparse 借鉴了 [HiCache](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/) 的层次化内存思想：**主动把"不活跃"的 KV cache 卸载到 CPU 主机内存**，同时**在 GPU HBM 上保留一块小的"热"设备缓冲区**，用于缓存高频访问的 KV 区域。这样：

- **大幅降低每请求 GPU 显存占用**：每个请求只需一个固定大小的设备缓冲区（例如 4KB token 槽位），而非整条序列长度。
- **按需换入（swap-in）**：一个专用 CUDA kernel 根据 attention score 动态地把 top-k 最相关的 KV 从主机内存加载进设备缓冲区。
- **对 prefill 透明**：HiSparse 完全是 decode 侧优化，prefill 实例无需任何改动。

结合 **PD 分离（prefill-decode disaggregation）**，HiSparse 能显著提升 decode 并发度，官方基准显示在 256 并发时可达基线吞吐的 **3 倍以上**（长上下文场景最高约 5 倍）。

---

## 2. 前置条件

HiSparse 适用于**原生支持稀疏注意力**的模型，因为这类模型本身会为注意力选择一部分 token，从而可以在 GPU 上只保留 top-k KV、而把完整 KV 存到主机内存——**且不损失精度**。

| 条件 | 说明 |
|------|------|
| **模型架构** | DeepSeek Sparse Attention (DSA) 架构（如 DeepSeek-V3.2、GLM-5.1）或 **DeepSeek V4** |
| **运行模式** | 当前要求 **PD 分离模式**，且只在 **decode 实例**上启用 |
| **配套开关** | decode 实例需 `--disable-radix-cache`（HiSparse 与 decode 侧 radix cache 不兼容） |

---

## 3. 整体架构

HiSparse 采用**层次化内存**：

```
                    ┌─────────────────────────────┐
                    │   GPU HBM (decode 实例)      │
                    │                             │
                    │  每请求「热」设备缓冲区       │
                    │  (device_buffer_size 槽位)   │
                    │        ▲                    │
                    │        │ swap-in (top-k)     │
                    │        │ CUDA kernel         │
                    └────────┼────────────────────┘
                             │
                    ┌────────┼────────────────────┐
                    │        ▼                    │
                    │  CPU Pinned Memory (主机池)  │
                    │  每请求「完整」KV cache       │
                    └─────────────────────────────┘
```

### 3.1 两条数据通道（admit 路径）

| 路径 | 触发场景 | 说明 |
|------|---------|------|
| **Staging 路径** `admit_request_into_staging` | 非 PD（colocate）或普通 prefill→decode | prefill 结束后，将全量 KV 从 GPU 异步 DMA 到主机池（`write_staging_stream`）；等 DMA 完成后再为请求分配设备缓冲区 |
| **Direct-to-host 路径** `admit_request_direct` | PD 分离（disaggregated decode） | prefill 实例通过 RDMA **直接把完整 KV 写入 decode 实例的主机池**，decode 侧完全不经过 GPU 暂存，消除了 KV 传输期间的瞬时显存尖峰，也省掉 staging DMA 一步 |

对 **DeepSeek V4**，direct-to-host 路径只把 **C4 KV** 写入 decode 主机池；`c4_indexer` 与 **C128 KV** 仍是 device-to-device 传输。

### 3.2 主要代码模块

| 模块（源码路径） | 职责 |
|------------------|------|
| `managers/hisparse_coordinator.py` | **核心协调器**：管理设备/主机缓冲区、staging/direct 接入、swap-in、eager backup、请求生命周期 |
| `jit_kernel/hisparse.py` + `jit_kernel/csrc/hisparse.cuh` | **swap-in JIT CUDA kernel**（哈希表 + LRU + miss 拷贝），以及 DSv4 C4 传输内核 |
| `mem_cache/hisparse_memory_pool.py` | 设备端 KV 池：`HiSparseDSATokenToKVPool` |
| `mem_cache/deepseek_v4_memory_pool.py` | DSv4 设备池：`HiSparseC4DevicePool`（C4 压缩） |
| `mem_cache/allocator/hisparse.py` | 双层分配器：`HiSparseTokenToKVPoolAllocator` / `DeepSeekV4HiSparseTokenToKVPoolAllocator` |
| `mem_cache/pool_host/hisparse.py` | 主机池混入：`HiSparseHostPoolMixin`（按页分配主机槽位） |
| `mem_cache/memory_pool_host.py` | 主机池实现：`MLATokenToKVPoolHost` / `DeepSeekV4PagedHostPool` |
| `mem_cache/sparsity/` | 通用稀疏注意力框架：`SparseCoordinator`、`DeepSeekDSAAlgorithm`、`parse_hisparse_config` |
| `layers/attention/dsa_backend.py` | DSA attention 后端中逐层调用 swap-in |
| `disaggregation/decode.py` | PD 分离集成：主机池作为传输目标、准入控制 |
| `managers/scheduler.py` / `schedule_batch.py` | 调度集成：初始化 coordinator、每步 decode 触发 backup |
| `arg_groups/hisparse_hook.py` | CLI 参数校验（后端策略、KV dtype） |

---

## 4. 核心数据结构（HiSparseCoordinator）

`HiSparseCoordinator.__init__` 分配以下关键张量（`max_num_req_slots` 来自 `req_to_token_pool`，`compress_ratio` 对 MLS 为 1、对 DSV4 为 4）：

| 张量 | 形状 | 作用 |
|------|------|------|
| `req_to_device_buffer` | `(reqs, padded_buffer_size)` int64 | 每请求「设备缓冲区」物理槽位数组 |
| `req_device_buffer_size` | `(reqs,)` int64 (CPU) | 每请求当前缓冲容量 |
| `req_to_host_pool` | `(reqs, max_ctx_len+page)` int64 | 请求的 (压缩) token 位置 → 主机池槽位映射 |
| `req_to_host_pool_allocated_len` | `(reqs,)` int64 (CPU) | 每请求已分配的主机槽位数 |
| `req_device_buffer_tokens` | `(layer, reqs, padded_buffer_size)` int32 | 每层每槽缓存的是哪个 **token 位置** |
| `req_device_buffer_token_locs` | `(layer, reqs, padded_buffer_size)` int32 | 每层每槽对应的 **物理 KV 位置** |
| `lru_slots` | `(layer, reqs, device_buffer_size)` int16 | LRU 槽位顺序（最近使用的在前/后） |
| `top_k_device_locs_buffer` | `(reqs, top_k)` int32 | CUDA-graph 安全的 swap-in 输出缓冲 |
| `raw_indices_buffer` | `(reqs, top_k)` int32 | top-k 原始索引暂存区 |
| `num_real_reqs` | `(1,)` int32 | 当前真实请求数（供 CUDA-graph 填充块早退） |

其中 `padded_buffer_size = device_buffer_size + page_size`（多留 `page_size` 个槽给**最新 token**）。

**双流 + 事件**：`write_staging_stream`（staging DMA）、`decode_backup_stream`（eager backup）、`decode_producer_stream`（= 调度器 forward stream），配合 CUDA event 实现异步流水线，避免传输阻塞解码关键路径。

---

## 5. Decode 工作流

官方文档定义的每步 decode 流程（与源码 `map_last_loc_to_buffer` + `swap_in_selected_pages` 对应）：

1. **Forward decode** — 生成下一个 token
2. **Top-k 选择** — DSA Indexer 依据 attention score 选出最相关的 token 位置
3. **Swap-in** — CUDA kernel 把 top-k KV 从主机加载进设备缓冲区
4. **Decode attention** — 用 top-k 设备位置计算注意力
5. **Eager backup** — 异步把上一步 token 的 KV 从设备拷贝回主机

### 5.1 每步 decode 的调用链

```
Scheduler.prepare_for_decode
  → HiSparseCoordinator.map_last_loc_to_buffer(...)      # ① 增长缓冲区 + 异步 eager backup
ModelRunner._forward_raw (decode)
  → hisparse_coordinator.wait_for_pending_backup()       # ② 等待上一步 backup 完成
  → hisparse_coordinator.num_real_reqs.fill_(bs)
DSABackend.forward_decode (逐层)
  → hisparse_coordinator.swap_in_selected_pages(...)     # ③ 每层调用 swap-in kernel
  → page_table_1 = 返回的 top_k_device_locs
  → flashmla_sparse / flashmla_kv / tilelang DSA attention  # ④ 稀疏注意力
```

### 5.2 Swap-in 内核算法（`hisparse.cuh`）

`load_cache_to_device_buffer_kernel`，每个 thread block 处理一个请求：

1. **早退**：`bid >= num_real_reqs[0]` 直接返回（CUDA-graph 填充块）。
2. **快速路径**：若 `seq_len <= HOT_BUFFER_SIZE`，所有 KV 已在设备缓冲区内，直接返回索引，**不访问主机**。
3. 否则在共享内存构建**开放寻址哈希表**（Knuth 乘法哈希，`2654435761`），键为 top-k **token 位置**。若 top-k 含最新 token，绑定到 `newest_slot`（缓冲区末尾 `HOT_BUFFER_SIZE`），不计入 LRU 跟踪。
4. **命中/未命中检测**：扫描每个 LRU 缓冲区槽，查哈希表判 `is_hit`；用 `__ballot` + 前缀和统计 hits/evictables。Hits 从索引 0 向前排列，evictables 从 `HOT_BUFFER_SIZE-1` 向后排列。
5. **miss 处理**：对 top-k 中未命中的 token，分配一个被逐出的 LRU 槽位，把输出 `top_k_device_locs` 指向该槽，并更新该槽的缓存 token。
6. **LRU 重排写回**：`[evictable 前部] ++ [刚加载的 miss] ++ [hit 的 MRU 后部]`。
7. **miss 拷贝**：一个 warp 搬运一个 miss；MLS 走线性字节步长，DSv4 走页填充布局（仅 K）。

### 5.3 命中率优化

- **更大的热缓冲区**与 **LRU 逐出策略**能显著降低 miss 数（官方：4096 vs 2048 槽位时 miss 明显下降），进而降低关键路径上的 swap-in 延迟。

---

## 6. 内存池与分配器

### 6.1 双层分配器（`HiSparseTokenToKVPoolAllocator`）

一个请求的 KV 需要两类索引：

| 分配器 | 含义 |
|--------|------|
| `logical_attn_allocator` | 完整逻辑池（`size_full = size * host_to_device_ratio`），管理全量 KV 的逻辑分配 |
| `hisparse_attn_allocator` | 设备缓冲区物理池（`size_hisparse = size`），管理热点槽位 |
| `full_to_hisparse_device_index_mapping` | 逻辑索引 → hisparse 设备索引的映射表 |

`alloc_device_buffer` 从 hisparse 分配器为每个请求分配一个固定大小的缓冲区；`alloc_logical_only` 仅分配逻辑索引（direct-to-host 路径用）。

### 6.2 主机池

`HiSparseHostPoolMixin.alloc_paged_token_slots` 按**页对齐**分配主机槽位，返回 token 粒度槽位。MLS 用 `MLATokenToKVPoolHost`，DSV4 用 `DeepSeekV4PagedHostPool`（页填充 C4 布局，`item_bytes = bytes_per_page_padded`）。

---

## 7. PD 分离集成（Direct-to-Host）

在 `disaggregation/decode.py` 中：

1. **传输目标即主机池**：`transfer_kv_pool = hisparse_coordinator.mem_pool_host`（HiSparse 时），KV 数据标记为 `DRAM` 而非 `VRAM`。DSV4 下只有 C4 层去主机池，`c4_indexer`/C128 仍在 GPU。
2. **准入控制 `hisparse_req_budget`**：物理约束为设备缓冲区容量 —— `hisparse_attn_allocator.available_size() // padded_buffer_size - len(transfer_queue)`；每个被承认的请求先从 transfer_queue 减去预算。
3. **预分配**：`_pre_alloc` 用 `alloc_logical_only`（不预分配 hisparse 设备页），并分配主机索引作为 **RDMA 写入目标**；发送给 prefill 的是 host/C4 行。
4. **接入**：传输完成后调用 `admit_request_direct(req)`（跳过 staging），随后在 decode 中按需 swap-in。

```
Prefill GPU ──RDMA──▶ Decode Host Pool (CPU pinned memory)
                          │
                          ▼
                 alloc device buffer (4KB)
                          │
                          ▼
              swap-in kernel (on-demand top-k)
```

---

## 8. 配置参数

### 8.1 开关

| 参数 | 类型/默认 | 说明 |
|------|-----------|------|
| `--enable-hisparse` | flag; 默认关闭 | 在 decode 实例上启用 HiSparse |
| `--hisparse-config` | JSON 字符串 | HiSparse 详细配置 |

### 8.2 `--hisparse-config` 参数（源码默认值）

| 参数 | 类型 / 默认 | 说明 |
|------|-------------|------|
| `top_k` | int / `2048` | top-k 条目数（会被模型 `index_topk` 覆盖，若存在） |
| `device_buffer_size` | int / `2 * top_k = 4096` | 每请求 GPU 设备缓冲区 token 槽位数（校验 `>= top_k`） |
| `host_to_device_ratio` | int / `2` | 逻辑池与设备池大小比，决定主机内存容量 |
| `swap_in_block_size` | int / `960` | swap-in 内核 CUDA 线程块大小（校验 `1..1024` 整数） |
| `algorithm` | str / `None` | 稀疏算法（`"quest"` / `"deepseek_dsa"`；HiSparse 走 `deepseek_dsa`） |
| `backend` | str / `None` | 注意力后端 |
| `page_size` | int / `None` | 页大小 |
| `min_sparse_prompt_len` | int / `None` | 触发稀疏的最小序列长度 |
| 其余字段 | — | 进入 `sparse_extra_config`（算法特定） |

**示例**：
`--hisparse-config='{"top_k": 2048, "device_buffer_size": 6144, "host_to_device_ratio": 10, "swap_in_block_size": 960}'`

### 8.3 后端/Dtype 校验（`hisparse_hook.py`）

- **CUDA**：`bfloat16` → `flashmla_sparse`；`fp8_e4m3` → `flashmla_kv`。
- **ROCm**：`tilelang` / `aiter`。
- 支持的 KV cache dtype：`("bfloat16", "fp8_e4m3")`。
- SM100+（Blackwell）上 `--kv-cache-dtype` 默认 `fp8_e4m3`，旧架构默认 `bfloat16`。
- GLM DSA + SM120/SM121 + `fp8_e4m3` → `flashinfer_sparse_mla`。
- DeepSeek V4 用自有 `dsv4` attention 后端。

---

## 9. 部署

> HiSparse 目前要求 **PD 分离**，仅在 **decode 实例**启用。

### 9.1 Prefill 实例

```bash
python3 -m sglang.launch_server \
    --model-path /path/to/model --trust-remote-code \
    --port 8000 --host 0.0.0.0 \
    --context-length 81920 --chunked-prefill-size 65536 \
    --tp-size 8 --dp-size 8 --enable-dp-attention \
    --mem-fraction-static 0.85 \
    --disaggregation-mode prefill \
    --disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3 \
    --nnodes 1 --node-rank 0
```

（prefill 实例**不需要** `--enable-hisparse`，对 HiSparse 透明。）

### 9.2 Decode 实例（启用 HiSparse）

```bash
python3 -m sglang.launch_server \
    --model-path /path/to/model --trust-remote-code \
    --port 8000 --host 0.0.0.0 \
    --context-length 81920 --tp-size 8 --dp-size 8 --enable-dp-attention \
    --mem-fraction-static 0.85 --disable-radix-cache \
    --disaggregation-mode decode \
    --disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3 \
    --dist-init-addr 127.0.0.1:5757 --nnodes 1 --node-rank 0 \
    --enable-hisparse \
    --hisparse-config='{"top_k": 2048, "device_buffer_size": 6144, "host_to_device_ratio": 10, "swap_in_block_size": 960}'
```

**`host_to_device_ratio` 指导**：
- ~1 TB 主机内存 → `host_to_device_ratio: 5`
- ~2 TB 主机内存 → `host_to_device_ratio: 10`

---

## 10. 共享索引预取（Shared-index prefetch）

官方文档提及：当一个模型复用一个 anchor 层的 top-k 选择到后续若干 "skip" 层（DSA `index_topk_freq` / `index_topk_pattern`，GLM-5.2 原生为 IndexShare）时，所有 skip 层的工作集在 anchor 索引算出那一刻即已知。HiSparse 自动利用这一点：

- anchor 的 swap-in 内核记录其 **miss 计划**（哪些主机槽 → 哪些设备缓冲槽）；
- 每个 skip 层用**只拷贝内核**在旁路 stream 上提前重放该计划；
- 从而 skip 层的 host→device IO 与中间层的计算**重叠**，不落在 decode 关键路径上。

该预取对符合条件的模型自动开启（无流水线并行、无投机解码），可用环境变量 **`SGLANG_DISABLE_HISPARSE_PREFETCH=1`** 关闭（用于 A/B 对比）。

> 注：在本仓库当前检查的 coordinator 代码中，该预取逻辑尚未落入 `hisparse_coordinator.py`（属较新/进行中特性）；文档以官方文档描述为准，源码实现可能随版本演进。

---

## 11. 生命周期与请求管理

| 阶段 | 方法 | 说明 |
|------|------|------|
| 请求开始 | `SparseCoordinator.on_request_begin` | 注册请求状态（`prompt_len` 等） |
| Staging 接入 | `admit_request_into_staging` | 异步把 prefill KV 搬往主机池，进 `ack_staging_queue` |
| 就绪收集 | `collect_ready_reqs` | 轮询 DMA 完成事件，跨 TP MIN-reduce 后分配设备缓冲区 |
| Direct 接入 | `admit_request_direct` | PD 下 KV 已在主机池，直接分配设备缓冲区（短序列预加载 / 长序列置空） |
| 每步 decode | `map_last_loc_to_buffer` → `_eager_backup_previous_token` | 异步备份上一步 token KV 到主机 |
| swap-in | `swap_in_selected_pages` | 每层调用 JIT 内核加载 top-k |
| 结束 | `request_finished` / `abort_staging_request` | 释放设备缓冲区、主机索引，复位张量、LRU |

`request_finished` 还会输出调试日志：请求 ID、推理 token 总数、主机存储占用（GB）等。

---

## 12. 性能与结论

- 官方基准（GLM-5.1-FP8，32k 输入/8k 输出，PD-colocated 8×H200）：256 并发时 **>3× 基线吞吐**；长上下文 sweep 最高约 **5×**。
- 低并发时 HiSparse 有轻微开销（稀疏 KV 加载的额外 IO 超过显存节省）；高并发、显存压力占主导时增益显著。
- 通过更大的热缓冲区 + LRU 逐出可降低 miss 率与关键路径延迟。

### 局限与展望

- 当前支持 DSA 家族（DeepSeek-V3.2、GLM-5.1 等）与 DeepSeek V4，属实验特性。
- 设计面向高并发以最大化吞吐；可通过更好重叠降低开销（如共享索引预取），并在 Grace Blackwell 更高 CPU–GPU 带宽平台上进一步缓解。
- 计划扩展到 hybrid 模型等更广架构（延续 HiCache 方向）。

---