# Apple Silicon GPU vs NVIDIA CUDA 架构对比实验

计算机体系结构课程演讲配套实验。在三台设备上运行相同基准测试，对比统一内存与离散显存的架构差异。

## 硬件

| 设备 | 架构 | 内存 | GPU |
|---|---|---|---|
| M5 Pro | 统一内存 (UMA) | 24GB LPDDR5X | Apple TBDR GPU |
| RTX 4070 Ti | 离散显存 | 12GB GDDR6X + 系统 DDR | NVIDIA IMR GPU |

## 实验列表

### 实验一：SLAM GPU 算子基准

跑两个 SLAM 中真正吃 GPU 的计算，测量 GPU 计算中数据传输的占比。全用 PyTorch，两边都是真 GPU（MPS / CUDA），不依赖 Open3D（Open3D 在 macOS 和 Windows 下都是 CPU-only，无法体现 GPU 架构差异）。

**算子 1 — Pairwise Distance + Argmin（模拟特征匹配 / ICP 最近邻）**:
10K × 10K 距离矩阵在 GPU 上算，取 argmin 找最近邻，结果传回 CPU

**算子 2 — Cholesky 分解（模拟 BA 优化的 Hessian 求解）**:
4000×4000 正定矩阵在 GPU 上做 Cholesky，结果传回 CPU

```bash
pip install torch numpy
python slam_ops_benchmark.py
```

**对比维度**：每个算子中传输耗时占总耗时的比例（UMA ≈ 0% vs PCIe 传输占比）

### 实验二：数据传输开销

直接测量不同数据量下 CPU ↔ GPU 往返延迟。

```bash
pip install torch
python transfer_benchmark.py
```

**对比维度**：1MB ~ 1GB 范围内传输耗时 vs 数据量的关系曲线

### 实验三：大模型推理

同一个 7B 模型，M5 Pro 直接加载 FP16 权重（24GB 够用），RTX 4070 Ti 必须 4-bit 量化（12GB 装不下 FP16 14GB 的模型）。

**M5 Pro**:
```bash
pip install mlx mlx-lm
python llm_benchmark.py --backend mlx
```

**RTX 4070 Ti**:
```bash
pip install llama-cpp-python
# 先下载 GGUF 模型:
# wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf
python llm_benchmark.py --backend llama.cpp
```

**对比维度**：模型是否能完整加载、token/s、首 token 延迟

### 实验四：Overdraw 可视化

无法自动化，需使用 GPU 调试工具手动截图。

**M5 Pro**:
1. Xcode → Open Developer Tool → Metal Frame Debugger
2. 运行任意 Metal 渲染 App，抓取一帧
3. Performance 面板 → Overdraw 热力图

**RTX 4070 Ti**:
1. NVIDIA Nsight Graphics → Frame Debugger
2. 运行任意渲染场景
3. 查看 Overdraw / Fragment Density 热力图

**对比维度**：TBDR 热力图大面积蓝色（低 Overdraw）vs IMR 热力图红黄区域（高 Overdraw）

## 核心测量原则

不比 "谁跑得更快"，比 **"时间花在哪了"**：
- M5 Pro 的时间几乎全在计算上
- RTX 4070 Ti 的时间被 PCIe 数据传输吃掉一大块

两台机器跑出各自的分布图，并排放到幻灯片上，结论一目了然。
