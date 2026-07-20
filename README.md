# Setup
- 添加当前用户到docker组中
```bash
sudo groupadd docker
sudo gpasswd -a $USER docker
sudo service docker restart
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
- 安装（代码基于SGLang的release/v0.5.15分支）
```bash
git config --global --add safe.directory /workspace/sglang
pip install --upgrade pip
pip install -e "python"
pip install nvitop
```
- 给LLM的提示词信息
```bash
HiSparse (Hierarchical Sparse Attention) reduces per-request GPU memory consumption during the decode phase by maintaining only a small “hot” KV buffer on GPU while keeping complete KV data in CPU pinned memory. Combined with PD disaggregation, it enables significantly higher decode concurrency.
Prerequisites: HiSparse works with models that use DeepSeek Sparse Attention (DSA) architectures (e.g., DeepSeek-V3.2, GLM-5.1) and DeepSeek V4. These models natively select a subset of tokens for attention, making it possible to keep only the top-k KV on GPU while storing the full KV in host memory — without accuracy loss. Additionally, HiSparse currently requires PD disaggregation mode and is enabled on the decode instance only. Why HiSparse? In long-context LLM inference, each decoding request holds a full-length KV cache on GPU, limiting the number of concurrent requests a decode instance can serve. HiSparse addresses this by: Reducing GPU memory per request: Each request occupies only a fixed-size device buffer (e.g., 4KB tokens) instead of the full sequence length. On-demand swap-in: A CUDA kernel dynamically loads the top-k most relevant KV entries from host memory based on attention scores. Transparent to prefill: HiSparse is entirely a decode-side optimization; the prefill instance requires no changes. Design Overview Decode Workflow Each decode step follows this flow: Forward decode — generate the next token; Top-k selection — select the most relevant token positions via attention scores; Swap-in — the CUDA kernel loads top-k KV entries from host to device buffer:; Short sequences (seq_len ≤ device_buffer_size): fast path, all KV already in buffer; Long sequences: hit detection → LRU reordering → miss handling (host → device copy); Decode attention — compute attention using the top-k device locations; Eager backup — asynchronously copy the previous token’s KV from device to host. PD Disaggregation Integration (Direct-to-Host): In PD disaggregation mode, the prefill instance transfers KV cache directly into the decode instance’s host pool via RDMA, bypassing the GPU entirely on the decode side. This eliminates the transient GPU memory spike during KV transfer and removes the staging DMA step. For DeepSeek V4, the direct-to-host path writes only C4 KV into the decode host pool. The c4_indexer and C128 KV remain device-to-device transfers. 
```
# Run
- 下载模型
```bash
pip install modelscope
nohup modelscope download --model deepseek-ai/DeepSeek-V4-Flash --local_dir /mnt/sda/DeepSeek-V4-Flash > download.log 2>&1 &
```
- 下载数据集
```bash
modelscope download --dataset gliang1001/ShareGPT_V3_unfiltered_cleaned_split --local_dir /workspace/sglang/dataset/ShareGPT_V3_unfiltered_cleaned_split
```
- 运行
```bash
# 启动Prefill实例
python3 -m sglang.launch_server \
    --model-path /mnt/sda/DeepSeek-V4-Flash \
    --trust-remote-code \
    --port 8000 \
    --host 0.0.0.0 \
    --context-length 81920 \
    --chunked-prefill-size 65536 \
    --tp-size 1 \
    --mem-fraction-static 0.85 \
    --disaggregation-mode prefill \
    --disaggregation-ib-device mlx5_0,mlx5_1 \
    --disaggregation-bootstrap-port 8998 \
    --nnodes 1 --node-rank 0 \
    --cpu-offload-gb 650 \
    --offload-group-size 4 \
    --offload-num-in-group 2 \
    --offload-mode cpu
# 启动Decode实例
python3 -m sglang.launch_server \
    --model-path /mnt/sda/DeepSeek-V4-Flash \
    --trust-remote-code \
    --port 8000 \
    --host 0.0.0.0 \
    --context-length 81920 \
    --tp-size 1 \
    --mem-fraction-static 0.85 \
    --disable-radix-cache \
    --disaggregation-mode decode \
    --disaggregation-ib-device mlx5_0,mlx5_1 \
    --prefill-addr 10.0.0.1:8998 \
    --nnodes 1 --node-rank 0 \
    --enable-hisparse \
    --hisparse-config='{"top_k": 2048, "device_buffer_size": 6144, "host_to_device_ratio": 10, "swap_in_block_size": 960}' \
    --cpu-offload-gb 650 \
    --offload-group-size 4 \
    --offload-num-in-group 2 \
    --offload-mode cpu
# 启动Benchmark测试
python3 -m sglang.bench_serving \
    --backend sglang \
    --dataset-path /workspace/sglang/dataset/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json \
    --dataset-name random \
    --random-input 40000 \
    --random-output 20000 \
    --num-prompts 200 \
    --max-concurrency 200 \
    --request-rate 40 \
    --random-range-ratio 1.0 \
    --host 127.0.0.1 \
    --port 20000 \
    --model /mnt/sda/DeepSeek-V4-Flash \
    --flush-cache
```

