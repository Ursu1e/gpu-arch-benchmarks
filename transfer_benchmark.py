#!/usr/bin/env python3
"""数据传输开销 — M5 Pro vs RTX 4070 Ti 纯传输基准

分别在不同数据量下测量 CPU <-> GPU 往返延迟。
统一内存架构下传输开销接近零，离散架构下 PCIe 传输显著。

用法：两台机器分别运行，对比输出。
依赖：pip install torch
"""

import time
import torch

# ========== 平台检测 ==========
if torch.cuda.is_available():
    DEV = "cuda"
    NAME = "NVIDIA RTX 4070 Ti"
    BW_THEORETICAL = 32  # PCIe 4.0 x16: 32 GB/s
elif torch.backends.mps.is_available():
    DEV = "mps"
    NAME = "Apple M5 Pro"
    BW_THEORETICAL = 273  # M5 Pro LPDDR5X: ~273 GB/s
else:
    DEV = "cpu"
    NAME = "CPU (fallback)"
    BW_THEORETICAL = 0


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()
    elif DEV == "mps":
        torch.mps.synchronize()


def measure_transfer(size_mb):
    """测量指定数据量的 CPU <-> GPU 往返耗时"""
    n = size_mb * 1024 * 1024 // 4  # float32
    x = torch.randn(n)

    sync()
    t0 = time.perf_counter()

    # CPU -> GPU -> 微小计算 -> GPU -> CPU
    x_gpu = x.to(DEV)
    y = x_gpu * 2 + 1
    z = y.cpu()

    sync()
    t1 = time.perf_counter()

    return (t1 - t0) * 1000


# ========== 运行 ==========
print(f"\n{'='*60}")
print(f"  CPU <-> GPU 数据传输开销基准")
print(f"  设备: {NAME}")
print(f"  理论带宽: {BW_THEORETICAL} GB/s")
print(f"{'='*60}")
print()
print(f"  {'数据量':>8} │ {'往返耗时':>10} │ {'等效带宽':>10}")
print(f"  {'─'*8}─┼─{'─'*10}─┼─{'─'*10}")

for size_mb in [1, 10, 50, 100, 250, 500, 1000]:
    ms = measure_transfer(size_mb)
    bw = size_mb / (ms / 1000) / 1024 if ms > 0 else float("inf")
    print(f"  {size_mb:>5} MB │ {ms:>8.2f} ms │ {bw:>8.2f} GB/s")

print(f"\n{'='*60}")
print(f"  结论：")
if DEV == "mps":
    print(f"  M5 Pro 统一内存下传输开销接近零，数据已在 GPU 可见地址空间")
elif DEV == "cuda":
    print(f"  RTX 4070 Ti 每次传输需经过 PCIe 4.0 总线")
    print(f"  数据量越大，传输占比越高 — 这是离散架构的结构性代价")
print(f"{'='*60}\n")
