# SSD Offload文档（`ssd_offload`）

> 一份**与后端无关**的 SSD offload（GPU 显存 ⇄ SSD 数据搬移）Python 模块文档。
> 本模块让你用**同一套顶层接口**，通过三种底层技术之一读写 SSD：
>
> | 方式 | 名称 | 说明 | 本机状态 |
> |------|------|------|---------|
> | **方式一** | `SCADA`（[Tutti](https://github.com/xPU-IO/Tutti.git) 风格） | GPU 线程直接驱动 NVMe，CPU 每批只出现 O(1) 次 | 占位（本机抛 `NotImplementedError`） |
> | **方式二** | `GDS`（[GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage)） | CPU 发指令，数据 GPU⇄SSD 直连（不经 CPU 中转） | ✅ 可用（cuFile） |
> | **方式三** | `CPU` | 指令和数据都经 CPU 中转 | ✅ 可用（始终） |

---

## 目录

- [1. 快速上手](#1-快速上手)
- [2. 核心概念](#2-核心概念)
- [3. 顶层 API 详解](#3-顶层-api-详解)
- [4. 能力模型与后端选择](#4-能力模型与后端选择)
- [5. 典型调用流程](#5-典型调用流程)
- [6. 高级用法与注意事项](#6-高级用法与注意事项)
- [7. 性能参考](#7-性能参考)
- [8. 常见问题（FAQ）](#8-常见问题faq)
- [9. 项目结构](#9-项目结构)

---

## 1. 快速上手

```python
import os, sys
sys.path.insert(0, "/home/zby/sglang")          # 确保可 import ssd_offload
from ssd_offload import create_ssd_offload, AccessMethod
from ssd_offload.core import MemoryKind, OffloadRequest

# 0) 准备一个已存在的文件（作为 SSD 上的"目标区域"）
path = "/tmp/offload_blob.bin"
with open(path, "wb") as f:
    f.truncate(8 * 1024 * 1024)                  # 8 MiB

# 1) 创建引擎（只改这一行即可切换后端）
engine = create_ssd_offload(AccessMethod.GDS)    # GDS | CPU | SCADA

# 2) 注册一块"内存"（device 或 host 均可；这里用一块 ctypes 对齐 buffer 演示）
size = 4 * 1024 * 1024
buf  = (ctypes.c_ubyte * size)()                 # 宿主内存（也可用 cudaMalloc 指针）
mem  = engine.register_memory(ctypes.addressof(buf), size, MemoryKind.DEVICE)

# 3) 打开 SSD 目标
tgt  = engine.open(path)

# 4) 往 SSD 写 4 MiB
req  = OffloadRequest(direction=1,              # 1=WRITE(mem->ssd), 0=READ
                      memory=mem, memory_offset=0,
                      target=tgt, target_offset=0, length=size)
io   = engine.submit([req])                     # 提交
res  = engine.wait(io)                          # 等结果
assert res.ok, res.error
print("写入了", res.accepted_bytes, "字节")
engine.release_io(io)

# 5) 从 SSD 读回
req = OffloadRequest(direction=0, memory=mem, memory_offset=0,
                     target=tgt, target_offset=0, length=size)
io  = engine.submit([req]); res = engine.wait(io); engine.release_io(io)

# 6) 清理
engine.unregister_memory(mem)
engine.close(tgt)
engine.shutdown()
```

---

## 2. 核心概念

SSD offload 的本质是：把**一段连续的字节区间**在 **GPU/宿主内存** 与 **SSD 上的文件** 之间搬移。本模块用四个概念描述这件事：

| 概念 | 类型 | 描述 |
|------|------|------|
| **内存区** | `MemoryRegistration` | 一段被后端"认识"的连续内存（device 或 host），可作为读写源/目的 |
| **SSD 目标** | `SsdTarget` | 一个已打开的文件/区域，可被读写 |
| **搬移请求** | `OffloadRequest` | 一次字节区间搬运意图（READ 或 WRITE），含显式偏移与长度 |
| **引擎** | `SsdOffload` | 顶层门面：open/close 目标、register/unregister 内存、submit/wait/release 读写 |

**关键设计原则**：引擎的公开接口**不暴露**任何后端私有的细节（如 cuFile 句柄、cudaStream、NVMe 队列）。每个后端把这些私有状态藏在 `SsdTarget._priv`、`MemoryRegistration._priv`、`OffloadHandle._priv` 里。因此**调用方代码在三种后端间完全一致**，切换只需改工厂参数。

---

## 3. 顶层 API 详解

> 以下所有方法都定义在 `SsdOffload` 类上（`ssd_offload/core.py`）。
> 引擎实例由 `create_ssd_offload(method)` 返回。

### 3.1 创建引擎

```python
from ssd_offload import create_ssd_offload, AccessMethod
engine = create_ssd_offload(AccessMethod.GDS, **backend_kwargs)
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `method` | `AccessMethod` | 是 | 选择后端：`AccessMethod.CPU / GDS / SCADA` |
| `**backend_kwargs` | 任意 | 否 | 传给具体后端的构造参数（见下表） |

**各后端额外构造参数：**

| 后端 | 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|------|
| `CPU` | `block_size` | `int` | `4096` | O_DIRECT 对齐粒度 |
| `CPU` | `use_o_direct` | `bool` | `True` | 是否用 O_DIRECT（否则走 page cache） |
| `GDS` | `allow_compat` | `bool` | `True` | 是否允许 cuFile 在无 P2P 时回退兼容模式 |
| `SCADA` | `device` | `str` | `"/dev/ssnvme0"` | 预留：专用 NVMe 控制器路径 |

**返回值**：实现了 `SsdOffload` 接口的引擎实例。

**异常**：
- 未知 `method` → `ValueError`
- GDS 库不可用 → `RuntimeError`
- SCADA 本机不可用 → `NotImplementedError`

---

### 3.2 `register_memory(address, size, kind) -> MemoryRegistration`

把一段内存注册给后端，供后续读写使用。

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `address` | `int` | 是 | — | 内存区起始地址（整数指针值） |
| `size` | `int` | 是 | — | 内存区大小（字节），必须 `> 0` |
| `kind` | `MemoryKind` | 否 | `MemoryKind.DEVICE` | 内存种类 |

`MemoryKind` 枚举：`HOST=0`、`PINNED_HOST=1`、`DEVICE=2`、`MANAGED=3`。

**返回值** `MemoryRegistration`，字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `int` | 注册 ID（引擎内唯一） |
| `address` | `int` | 起始地址 |
| `size` | `int` | 大小（字节） |
| `kind` | `MemoryKind` | 内存种类 |
| `_priv` | `object` | 后端私有状态（勿触碰） |

**异常**：`address` 为 0 或 `size<=0` → `ValueError`；引擎已关闭 → `RuntimeError`。

> GDS 后端会对 `DEVICE`/`PINNED_HOST` 内存调用 `cuFileBufRegister`（固定内存，用于 P2P 直连）。

---

### 3.3 `unregister_memory(reg) -> None`

注销一段已注册的内存，释放后端资源。

| 参数 | 类型 | 说明 |
|------|------|------|
| `reg` | `MemoryRegistration` | 由 `register_memory` 返回的对象 |

**异常**：`reg` 未注册过 → `KeyError`。

---

### 3.4 `open(uri, scheme="file") -> SsdTarget`

打开 SSD 上的一块可读写的文件/区域。

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `uri` | `str` | 是 | — | 目标位置。`file:///abs/path` 或裸路径 `/abs/path` |
| `scheme` | `str` | 否 | `"file"` | 协议 scheme（当前固定 `file`） |

**返回值** `SsdTarget`，字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `uri` | `str` | 原始 URI |
| `logical_size` | `int` | 文件字节大小（由客户端探测） |
| `handle_id` | `int` | 目标 ID |
| `_priv` | `object` | 后端私有状态（勿触碰） |

**方法**：`target.valid() -> bool` 判断该目标是否有效。

**注意**：文件**必须已经存在**（O_DIRECT / cuFile 不能对不存在文件 create-open）。

---

### 3.5 `close(tgt) -> None`

关闭一个 SSD 目标，释放后端资源。

| 参数 | 类型 | 说明 |
|------|------|------|
| `tgt` | `SsdTarget` | 由 `open` 返回的目标 |

**异常**：目标未打开 → `KeyError`。

---

### 3.6 `submit(requests, stream=None) -> OffloadHandle`

提交一个批量读写请求（可多个请求合并为一次提交）。

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `requests` | `List[OffloadRequest]` | 是 | — | 待执行的搬移请求数组 |
| `stream` | `int`/`None` | 否 | `None` | 预留的 CUDA stream 句柄（当前为 `None`，同步执行） |

**返回值** `OffloadHandle`，用于后续 `wait`/`release_io`。方法 `handle.valid() -> bool`。

> 当前设计是**同步执行**：`submit` 内部已把 I/O 执行完并保存结果，`wait` 直接返回该结果。

---

### 3.7 `wait(io, timeout_ms=60000) -> ReadResult`

获取一次提交的完成结果。

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `io` | `OffloadHandle` | 是 | — | 由 `submit` 返回的句柄 |
| `timeout_ms` | `int` | 否 | `60000` | 超时毫秒（当前为同步设计，实际立即返回） |

**返回值** `ReadResult`，字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `ok` | `bool` | 整批是否成功 |
| `error` | `str` | 失败时的错误信息（成功时为空） |
| `accepted_bytes` | `int` | 成功搬运的字节数 |
| `initial_states` | `List[str]` | 每个请求的接受状态（`"accepted"`） |

---

### 3.8 `release_io(io) -> None`

释放一次提交所占用的句柄资源。

| 参数 | 类型 | 说明 |
|------|------|------|
| `io` | `OffloadHandle` | 由 `submit` 返回的句柄 |

---

### 3.9 `shutdown() -> None`

优雅关闭引擎：关闭所有已打开的目标、注销所有已注册的内存、并调用后端 `_shutdown_impl` 释放驱动/库资源（如 cuFile driver session）。

**支持上下文管理器**：

```python
with create_ssd_offload(AccessMethod.GDS) as engine:
    ...   # 退出 with 块时自动 shutdown()
```

---

### 3.10 `OffloadRequest` 构造

```python
OffloadRequest(direction, memory, memory_offset, target, target_offset, length)
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `direction` | `int` | **0 = READ**（target→memory）/**1 = WRITE**（memory→target） |
| `memory` | `MemoryRegistration` | 内存区 |
| `memory_offset` | `int` | 内存区内的起始偏移（字节） |
| `target` | `SsdTarget` | SSD 目标 |
| `target_offset` | `int` | 文件内的起始偏移（字节） |
| `length` | `int` | 搬运字节数 |

---

## 4. 能力模型与后端选择

### 4.1 `BackendCapabilities`（能力报告）

每个引擎暴露 `engine.capabilities`，属 `BackendCapabilities`，字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `method` | `AccessMethod` | 后端方式 |
| `supports_device_direct` | `bool` | 是否支持 GPU 直连（GDS P2P / SCADA GPU 驱动） |
| `supports_host_bounce` | `bool` | 是否能回退到 CPU 中转路径 |
| `requires_pinned_memory` | `bool` | device 内存是否需要固定/注册 |
| `max_io_size` | `int` | 最大单次 IO（0=后端默认） |
| `compat_mode` | `bool` | GDS 是否处于兼容模式（内部 POSIX 回退） |

方法 `capabilities.describe() -> str` 打印可读摘要。

### 4.2 如何选择

| 你的需求 | 推荐 `AccessMethod` |
|---------|--------------------|
| 想要最高 GPU⇄SSD 直连性能，且已装 `nvidia-fs`/cufile | `GDS` |
| 任何情况下都稳定可用，做参考/对比/降级 | `CPU` |
| 有专用 NVMe + Tutti/snvme 部署，想 GPU 线程直接驱动 | `SCADA`（占位） |

**降级指南**：`create_ssd_offload(AccessMethod.GDS)` 失败时，可捕获异常并回退 `CPU`：

```python
try:
    engine = create_ssd_offload(AccessMethod.GDS)
except RuntimeError:
    engine = create_ssd_offload(AccessMethod.CPU)
```

---

## 5. 典型调用流程

```
创建引擎 ──▶ 注册内存 ──▶ 打开目标 ──▶ 提交读写 ──▶ 等待结果 ──▶ 释放IO
   ◆                                                        ◆
   └──────────── shutdown：关目标/注销内存/释放驱动 ◀─────────┘
```

完整生命周期代码：

```python
from ssd_offload import create_ssd_offload, AccessMethod
from ssd_offload.core import MemoryKind, OffloadRequest

engine = create_ssd_offload(AccessMethod.GDS)      # 1. 创建
mem    = engine.register_memory(addr, size, MemoryKind.DEVICE)   # 2. 注册内存
tgt    = engine.open("file:///tmp/blob.bin")       # 3. 打开目标

# 4. 读写
for off in range(0, file_size, chunk):
    req = OffloadRequest(1, mem, 0, tgt, off, chunk)   # WRITE
    io  = engine.submit([req])
    res = engine.wait(io)
    assert res.ok
    engine.release_io(io)

# 5. 清理
engine.unregister_memory(mem)
engine.close(tgt)
engine.shutdown()
```

---

## 6. 高级用法与注意事项

1. **文件必须预先存在**：`open`/GDS 的 `cuFileHandleRegister` 不支持自动创建文件；请先用普通方式创建并 `truncate` 到目标大小。

2. **对齐**：底层多走 O_DIRECT，`length` 与 `offset` 建议对齐到 4 KiB，性能更佳。非对齐也能工作（后端内部自动对齐，代价是多一次复制）。

3. **GDS 的内存种类**：`DEVICE`/`PINNED_HOST` 会被 `cuFileBufRegister` 注册，是 P2P 直连的前提；`HOST`/`MANAGED` 走兼容路径。

4. **同步语义**：当前 `submit` 同步执行（GDS 用同步 `cuFileRead/Write`；CPU 用 `pread/pwrite`）。`stream` 参数预留，将来接异步时可传 CUDA stream。

5. **跨后端一致**：切换后端**只改 `create_ssd_offload` 那一行**，其余调用代码保持不变。

6. **SCADA 占位**：`ScadaSsdOffload` 构造即抛 `NotImplementedError`（本机无专用非系统 NVMe）。将来接真实 Tutti/snvme 时，只需填充其 `_open/_register/_submit/...` 扩展点，上层不变。

7. **支持上下文管理器**：`with create_ssd_offload(...) as engine:` 自动 `shutdown()`（见 3.9）。

---

## 7. 性能参考（本机实测）

> 机器：RTX 6000D 单卡；NVMe 为系统盘（ext4，VMD 桥下）。`/tmp`，chunk=4096 KiB，1 GiB。

| 后端 | 写带宽 | 读带宽 |
|------|--------|--------|
| `CPU`（方式三） | ~1001 MB/s | ~2828 MB/s |
| `GDS`（方式二，cuFile） | ~4213 MB/s | ~6005 MB/s |

GDS 相比 CPU：写 ~4.2 倍、读 ~2.1 倍。GDS 即便本机无专门 P2P 优化（走 cuFile 内部兼容/优化路径）也显著更快；一旦有 P2P 优化的 NVMe，`cuFileBufRegister` 的 device 直连路径会自动生效，无代码改动。

复现：

```bash
cd /home/zby/sglang
/usr/bin/python3 -m ssd_offload.tests.test_ssd_offload --method gds --bandwidth --size-gb 1
```

---

## 8. 常见问题（FAQ）

**Q1. `create_ssd_offload(AccessMethod.GDS)` 报错？**
A. 检查 `libcufile.so` 是否在 `LD_LIBRARY_PATH`、`nvidia-fs.ko` 是否加载（`lsmod | grep nvidia_fs`）。失败可回退 `CPU`。

**Q2. `open` 失败？**
A. 确保文件已存在。`file://` 前缀或绝对路径均可；相对路径需可访问。

**Q3. 为什么 READ 的 `direction=0`，WRITE 是 `direction=1`？**
A. 与 Tutti 的 `IoDirection` 对齐：`READ`=target→memory，`WRITE`=memory→target。查阅本模块 `OffloadRequest` 语义时以此为准。

**Q4. `submit` 会阻塞吗？**
A. 当前是同步设计，`submit` 内部执行完才返回，`wait` 立即取回结果。预留的 `stream` 参数为将来异步化。

**Q5. 我只有宿主内存（HOST），能用 GDS 吗？**
A. 可以。GDS 的 `cuFileRead/Write` 支持 host 指针（走兼容/非直连路径）。要用直连 P2P 才需要 `DEVICE`/`PINNED_HOST`。

---

## 9. 项目结构

```
ssd_offload/
├── __init__.py              # 顶层导出：create_ssd_offload / list_backends / 核心类型
├── core.py                  # ★ 顶层抽象：SsdOffload 接口 + 值类型 + 能力模型
├── backends/
│   ├── __init__.py          # 重新导出工厂函数
│   ├── factory.py           # create_ssd_offload(method) / list_backends()
│   ├── cpu_backend.py       # 方式三：CPU bounce（O_DIRECT）
│   ├── gds_backend.py       # 方式二：GDS（cuFile）
│   └── scada_backend.py     # 方式一：SCADA/Tutti（占位）
└── tests/
    └── test_ssd_offload.py  # --method {cpu,gds,scada} 正确性 + 带宽测试
```

| 文件 | 职责 |
|------|------|
| `core.py` | 定义 `SsdOffload` 抽象基类及其公共 API；`AccessMethod`/`MemoryKind` 枚举；`MemoryRegistration`/`SsdTarget`/`OffloadHandle`/`OffloadRequest`/`ReadResult`；`BackendCapabilities` |
| `factory.py` | 后端路由：`create_ssd_offload` 按 `AccessMethod` 返回对应引擎；`list_backends` 枚举可用后端 |
| `cpu_backend.py` | 实现 `_open/_close/_register/_unregister/_submit/_shutdown`（CPU bounce 路径） |
| `gds_backend.py` | 实现上述扩展点（GDS cuFile 路径：driver/handle/buffer/io） |
| `scada_backend.py` | 占位实现（构造即抛 `NotImplementedError`） |
