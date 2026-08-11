# Setup
- 添加当前用户到docker组中
```bash
sudo groupadd docker
sudo gpasswd -a $USER docker
sudo service docker restart
```
- 配置网络代理，参考文档[docs-1](https://clash.guide/clash-for-linux.html) [docs-2](https://dash.pqjc.site/dashboard)
```bash
sudo apt update && sudo apt install -y curl tar unzip xz-utils procps   # 安装依赖
git clone --branch master --depth 1 https://gh-proxy.org/https://github.com/nelvko/clash-for-linux-install.git && cd clash-for-linux-install && bash install.sh # 安装
clashsub add "你的订阅地址" # 订阅地址通过docs-2获取
clashsub ls # 查看订阅列表
clashsub use 1  # 启用第 1 个订阅
clashsub update # 更新当前订阅
clashon # 启动内核 + 设置系统代理
clashoff    # 关闭内核 + 取消系统代理
```
- 进入容器
```bash
sudo docker pull lmsysorg/sglang:v0.5.15
sudo docker run -itd --shm-size 64g --gpus all -v $HOME/.cache/huggingface/:/root/.cache/huggingface -v /mnt/sda:/mnt/sda -v /home/zby/sglang:/workspace/sglang -w /workspace/sglang --ipc=host --network=host --privileged --name sglang_zby lmsysorg/sglang:v0.5.15 /bin/zsh
# 进入容器
sudo docker exec -it sglang_zby /bin/zsh
# 停止和删除容器
sudo docker stop sglang_zby
sudo docker rm sglang_zby
```
- 安装SGLang（代码基于SGLang的release/v0.5.15分支）
```bash
git config --global --add safe.directory /workspace/sglang
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
pip config set install.trusted-host mirrors.ustc.edu.cn
pip install --upgrade pip
pip install -e "python"
# 验证安装
python3 -c "import sglang; print(sglang)"
```
- 安装SGLang Router
```bash
# 安装RUST
mkdir -p ~/.cargo
cat > ~/.cargo/config.toml << 'EOF'
[source.crates-io]
replace-with = "ustc-sparse"
[source.ustc-sparse]
registry = "sparse+https://mirrors.ustc.edu.cn/crates.io-index/"
EOF
cat ~/.cargo/config.toml
export RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static
export RUSTUP_UPDATE_ROOT=https://mirrors.ustc.edu.cn/rust-static/rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -v | sh -s -- -y  # nohup bash -c "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -v | sh -s -- -y" > rust.log 2>&1 &
source $HOME/.cargo/env
rustup update stable
# 验证Rust是否成功
rustc --version
RUSTUP_DIST_SERVER=https://rsproxy.cn RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup RUSTUP_TOOLCHAIN=1.97.1 pip install -e "sgl-model-gateway/bindings/python" -v
# 验证安装
python3 -c "import sglang_router; print(sglang_router)"
```
- 其他安装
```bash
pip install nvitop
```
- 安装Nvidia SCADA
```bash
# TODO
```
- 下载模型
```bash
pip install modelscope
nohup modelscope download --model deepseek-ai/DeepSeek-V4-Flash --local_dir /mnt/sda/DeepSeek-V4-Flash > download.log 2>&1 &
```
- 裁剪模型
```bash
# 第1步：先 dry-run，确认计划和体积
python tools/trim_dsv4_checkpoint.py \
  --src /mnt/sda/DeepSeek-V4-Flash \
  --dst /mnt/sda/zby/DeepSeek-V4-Flash-trim8 \
  --num-hca 3 --num-csa 3 \
  --dry-run
# 第2步：真正生成（需要 torch+safetensors 的 sglang 环境，容器里没问题）
python tools/trim_dsv4_checkpoint.py \
  --src /mnt/sda/DeepSeek-V4-Flash \
  --dst /mnt/sda/zby/DeepSeek-V4-Flash-trim8 \
  --num-hca 3 --num-csa 3
```
- 下载数据集
```bash
modelscope download --dataset gliang1001/ShareGPT_V3_unfiltered_cleaned_split --local_dir /mnt/sda/zby/ShareGPT_V3_unfiltered_cleaned_split
```
# Run
- 验证运行
```bash
OMP_NUM_THREADS=8 SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16 SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0 SGLANG_SM120_FLASHMLA_BACKEND=torch sglang serve \
    --model-path "/mnt/sda/zby/DeepSeek-V4-Flash-trim8" \
    --host 0.0.0.0 \
    --port 30000 \
    --tp 1 \
    --context-length 16384 \
    --chunked-prefill-size 4096 \
    --disable-flashinfer-autotune \
    --disable-cuda-graph
# 流式输出
curl -N -X POST "http://localhost:30000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/mnt/sda/zby/DeepSeek-V4-Flash-trim8",
        "messages": [
        {"role": "user", "content": "What is the capital of France?"}
        ],
        "max_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": true
    }'
# 非流式输出
curl -N -X POST "http://localhost:30000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/mnt/sda/zby/DeepSeek-V4-Flash-trim8",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "top_p": 0.9,
        "temperature": 0.7
    }'
pkill -f sglang
```
- 运行（跨机PD分离：节点一dev1跑Prefill、节点二dev2跑Decode）
  > ⚠️ 关键：Prefill 与 Decode 是**两个独立的 torch distributed 集群**，跨机互找靠的是
  > **Router + Prefill 上的 bootstrap server(mooncake) + 各自 `local_ip`**，而**不是** torch TCPStore。
  > 因此两个进程的 `--dist-init-addr` 必须各自指向**本机**（IP+端口不同），**绝不能指向对方**
  > （否则 Decode 会去连 Prefill 已占用的 store 端口而 600s 超时，正是你 02 节点卡死的原因）。
  > 另外两台机器之间**只有 IB 网段 10.0.0.x 互通**（dev1=10.0.0.1、dev2=10.0.0.2），
  > dev2 没有可用以太网，所以 Router 的地址、`SGLANG_HOST_IP`、`--dist-init-addr` 一律用 IB 地址。
  > IB 设备务必指向**各自节点上 Active/连通的那块 HCA**：
  > dev1 用 `mlx5_0`（=ibp3s0f0，Active，连 dev2 的 mlx5_1）、dev2 用 `mlx5_1`（=ibp3s0f1，Active）。
  > 不要像原来那样两个节点都写 `mlx5_0,mlx5_1`——dev2 的 mlx5_0 是 DOWN 的，RDMA 会失败。

    - 节点一（dev1）：启动Prefill实例
    ```bash
    MASTER_ADDR=10.0.0.1 MASTER_PORT=5757 \
    SGLANG_HOST_IP=10.0.0.1 \
    SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0 \
    SGLANG_SM120_FLASHMLA_BACKEND=torch \
    python3 -m sglang.launch_server \
        --model-path /mnt/sda/zby/DeepSeek-V4-Flash-trim8 \
        --served-model-name DeepSeek-V4-Flash-trim8 \
        --trust-remote-code \
        --host 0.0.0.0 --port 30001 \
        --tp 1 \
        --context-length 16384 \
        --chunked-prefill-size 4096 \
        --mem-fraction-static 0.85 \
        --disaggregation-mode prefill \
        --disaggregation-transfer-backend mooncake_tcp \
        --dist-init-addr 10.0.0.1:5757 \
        --disaggregation-bootstrap-port 8998 \
        --nnodes 1 \
        --node-rank 0 \
        --disable-flashinfer-autotune
        --load-balance-method round_robin
    ```
    - 节点二（dev2）：启动Decode实例
    首先登录节点二，并按照前面的步骤安装好SGLang
    ```bash
    ssh zby@n02 # 登录节点二
    sudo rsync -avz --progress ~/sglang/ zby@n02:~/sglang/    # 传输代码库
    sudo rsync -avz --progress /mnt/sda/zby/DeepSeek-V4-Flash-trim8 zby@n02:/mnt/sda/zbyzby/DeepSeek-V4-Flash-trim8 # 传输模型
    # 传输并启动docker
    docker save -o /mnt/sda/zby/my_image.tar lmsysorg/sglang:v0.5.15
    scp /mnt/sda/zby/my_image.tar zby@n02:~/
    docker load -i ~/my_image.tar
    sudo docker run -itd --shm-size 64g --gpus all -v $HOME/.cache/huggingface/:/root/.cache/huggingface -v /mnt/sda:/mnt/sda -v /home/zby/sglang:/workspace/sglang -w /workspace/sglang --ipc=host --network=host --privileged --name sglang_zby lmsysorg/sglang:v0.5.15 /bin/zsh
    sudo docker exec -it sglang_zby /bin/zsh
    # 然后安装SGLang
    git config --global --add safe.directory /workspace/sglang
    pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
    pip config set install.trusted-host mirrors.ustc.edu.cn
    pip install --upgrade pip
    pip install -e "python"
    ```
    启动Decode实例（注意：--dist-init-addr指向dev2自己10.0.0.2，而非dev1）
    ```bash
    # dev1节点的模型放在/mnt/sda/zby/DeepSeek-V4-Flash-trim8，dev2节点的模型放在/mnt/sda/zbyzby/DeepSeek-V4-Flash-trim8
    MASTER_ADDR=10.0.0.2 MASTER_PORT=5757 \
    SGLANG_HOST_IP=10.0.0.2 \
    SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0 \
    SGLANG_SM120_FLASHMLA_BACKEND=torch \
    OMP_NUM_THREADS=8 \
    python3 -m sglang.launch_server \
        --model-path /mnt/sda/zbyzby/DeepSeek-V4-Flash-trim8 \
        --served-model-name DeepSeek-V4-Flash-trim8 \
        --host 0.0.0.0 --port 30002 \
        --tp 1 \
        --context-length 16384 \
        --chunked-prefill-size 4096 \
        --mem-fraction-static 0.85 \
        --disable-radix-cache \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend mooncake_tcp \
        --dist-init-addr 10.0.0.2:5757 \
        --nnodes 1 \
        --node-rank 0 \
        --disable-flashinfer-autotune \
        --enable-hisparse \
        --hisparse-config='{"top_k": 2048, "device_buffer_size": 6144, "host_to_device_ratio": 2, "swap_in_block_size": 960}'
    ```
    - 启动Router（放在dev1节点，用IB地址指向prefill/decode）
```bash
python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://10.0.0.1:30001 \
    --decode http://10.0.0.2:30002 \
    --host 0.0.0.0 \
    --port 30000 \
    --model-path /mnt/sda/zby/DeepSeek-V4-Flash-trim8
```
- 启动Benchmark测试（在dev1节点上）
```bash
# 先CURL一个请求验证正确性
curl -N -X POST "http://127.0.0.1:30000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "DeepSeek-V4-Flash-trim8",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 100,
        "temperature": 0.7
    }'
# 启动测试
python3 -m sglang.bench_serving \
    --backend sglang \
    --dataset-path /mnt/sda/zby/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json \
    --dataset-name random \
    --random-input 4096 \
    --random-output 2048 \
    --num-prompts 50 \
    --max-concurrency 25 \
    --request-rate 1 \
    --random-range-ratio 0.15 \
    --host 127.0.0.1 \
    --port 30000 \
    --model /mnt/sda/zby/DeepSeek-V4-Flash-trim8 \
    --flush-cache
```

## 给LLM的提示词信息
```bash
HiSparse (Hierarchical Sparse Attention) reduces per-request GPU memory consumption during the decode phase by maintaining only a small “hot” KV buffer on GPU while keeping complete KV data in CPU pinned memory. Combined with PD disaggregation, it enables significantly higher decode concurrency.
Prerequisites: HiSparse works with models that use DeepSeek Sparse Attention (DSA) architectures (e.g., DeepSeek-V3.2, GLM-5.1) and DeepSeek V4. These models natively select a subset of tokens for attention, making it possible to keep only the top-k KV on GPU while storing the full KV in host memory — without accuracy loss. Additionally, HiSparse currently requires PD disaggregation mode and is enabled on the decode instance only. Why HiSparse? In long-context LLM inference, each decoding request holds a full-length KV cache on GPU, limiting the number of concurrent requests a decode instance can serve. HiSparse addresses this by: Reducing GPU memory per request: Each request occupies only a fixed-size device buffer (e.g., 4KB tokens) instead of the full sequence length. On-demand swap-in: A CUDA kernel dynamically loads the top-k most relevant KV entries from host memory based on attention scores. Transparent to prefill: HiSparse is entirely a decode-side optimization; the prefill instance requires no changes. Design Overview Decode Workflow Each decode step follows this flow: Forward decode — generate the next token; Top-k selection — select the most relevant token positions via attention scores; Swap-in — the CUDA kernel loads top-k KV entries from host to device buffer:; Short sequences (seq_len ≤ device_buffer_size): fast path, all KV already in buffer; Long sequences: hit detection → LRU reordering → miss handling (host → device copy); Decode attention — compute attention using the top-k device locations; Eager backup — asynchronously copy the previous token’s KV from device to host. PD Disaggregation Integration (Direct-to-Host): In PD disaggregation mode, the prefill instance transfers KV cache directly into the decode instance’s host pool via RDMA, bypassing the GPU entirely on the decode side. This eliminates the transient GPU memory spike during KV transfer and removes the staging DMA step. For DeepSeek V4, the direct-to-host path writes only C4 KV into the decode host pool. The c4_indexer and C128 KV remain device-to-device transfers. 
```