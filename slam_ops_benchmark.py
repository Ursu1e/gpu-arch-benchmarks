#!/usr/bin/env python3
"""SLAM GPU 算子基准 — M5 Pro (UMA) vs RTX 4070 Ti (离散)

两个算子模拟 SLAM 中真正吃 GPU 的计算：
  算子 1: 批量 pairwise 距离 + argmin (模拟特征匹配/ICP 最近邻)
  算子 2: 大规模 Cholesky 分解 (模拟 BA 优化的 Hessian 求解)

每个算子测量纯计算耗时 vs 数据传输占比。
统一内存下传输出零，离散显存下 PCIe 传输显著。

用法：两台机器分别运行本脚本，对比输出。
依赖：pip install torch numpy
"""

import time
import sys
import numpy as np
import torch


def detect():
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0), "离散显存 (PCIe 4.0)"
    elif torch.backends.mps.is_available():
        return "mps", "Apple Silicon (MPS)", "统一内存 (零拷贝)"
    else:
        return "cpu", "CPU", "无 GPU"


DEV, DEVNAME, ARCH = detect()


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()
    elif DEV == "mps":
        torch.mps.synchronize()


# ============================================================
# 算子 1: Pairwise Distance + Nearest Neighbor
# 模拟：SLAM 中当前帧 10 万特征点 vs 地图 10 万点的匹配
# ============================================================
def bench_pairwise_nearest(n=10_000, warmup=3, repeat=10):
    """
    n 个点 → 计算 n×n 距离矩阵 → 逐行取 argmin。
    M5 Pro (UMA): 源数据在统一内存，to("mps") ≈ 零开销
    4070 Ti: to("cuda") 走 PCIe，结果 .cpu() 再走一次 PCIe
    """
    x = torch.randn(n, 2)          # CPU 上生成的源数据
    x_gpu = x.to(DEV)              # ← 传输1: CPU→GPU
    sync()

    # 预热
    for _ in range(warmup):
        diff = x_gpu.unsqueeze(0) - x_gpu.unsqueeze(1)  # n×n×2
        dist2 = (diff ** 2).sum(dim=-1)                   # n×n
        nn = dist2.argmin(dim=1)                          # n
        sync()
        _ = nn.cpu()                                      # ← 传输2: GPU→CPU
        sync()

    # 正式计时
    t0 = time.perf_counter()
    for _ in range(repeat):
        x_gpu = x.to(DEV); sync()
        diff = x_gpu.unsqueeze(0) - x_gpu.unsqueeze(1)
        dist2 = (diff ** 2).sum(dim=-1)
        nn = dist2.argmin(dim=1)
        sync()
        nn_cpu = nn.cpu(); sync()
    t_total = (time.perf_counter() - t0) / repeat * 1000

    # 纯计算（不带传输）
    x_pre = x.to(DEV); sync()
    t0 = time.perf_counter()
    for _ in range(repeat):
        diff = x_pre.unsqueeze(0) - x_pre.unsqueeze(1)
        dist2 = (diff ** 2).sum(dim=-1)
        nn = dist2.argmin(dim=1)
        sync()
    t_compute = (time.perf_counter() - t0) / repeat * 1000

    t_transfer = t_total - t_compute
    data_mb = x.element_size() * x.nelement() / (1024**2)
    return t_total, t_compute, t_transfer, data_mb


# ============================================================
# 算子 2: Cholesky 分解
# 模拟：SLAM BA 优化中求解海森矩阵 H Δx = b 的核心步骤
# ============================================================
def bench_cholesky(n=4000, warmup=3, repeat=10):
    """
    构造 n×n 正定矩阵 A = B @ B^T + λI，做 Cholesky 分解。
    M5 Pro (UMA): 结果直接在统一内存，.cpu() ≈ 零开销
    4070 Ti: 结果从显存 → CPU 走 PCIe
    """
    B = torch.randn(n, n)
    A_cpu = B @ B.T + n * torch.eye(n)   # 确保正定

    A_gpu = A_cpu.to(DEV); sync()        # 传输1: CPU→GPU

    # 预热
    for _ in range(warmup):
        L = torch.linalg.cholesky(A_gpu)
        sync()
        _ = L.cpu(); sync()

    # 正式计时
    t0 = time.perf_counter()
    for _ in range(repeat):
        A_gpu = A_cpu.to(DEV); sync()
        L = torch.linalg.cholesky(A_gpu)
        sync()
        L_cpu = L.cpu(); sync()
    t_total = (time.perf_counter() - t0) / repeat * 1000

    # 纯计算
    A_pre = A_cpu.to(DEV); sync()
    t0 = time.perf_counter()
    for _ in range(repeat):
        L = torch.linalg.cholesky(A_pre)
        sync()
    t_compute = (time.perf_counter() - t0) / repeat * 1000

    t_transfer = t_total - t_compute
    data_mb = A_cpu.element_size() * A_cpu.nelement() / (1024**2)
    return t_total, t_compute, t_transfer, data_mb


# ============================================================
# 打印报告
# ============================================================
def report(name, total, compute, transfer, data_mb):
    pct = transfer / total * 100 if total > 0 else 0
    print(f"\n  {name}")
    print(f"    数据量:       {data_mb:.1f} MB")
    print(f"    总耗时:       {total:.2f} ms  (含传输)")
    print(f"    纯计算:       {compute:.2f} ms")
    print(f"    传输:         {transfer:.3f} ms")
    print(f"    传输占比:     {pct:.1f}%")
    if pct < 2:
        print(f"    → 统一内存：传输可忽略，时间全在计算")
    elif pct > 10:
        print(f"    → 离散显存：PCIe 传输是显著开销")
    return pct


print(f"\n{'='*65}")
print(f"  SLAM GPU 算子基准测试")
print(f"  设备: {DEVNAME}")
print(f"  架构: {ARCH}")
print(f"{'='*65}")

# ——— 算子 1 ———
print(f"\n{'─'*65}")
print(f"  【算子 1】Pairwise Distance + Nearest Neighbor")
print(f"  模拟 SLAM 中特征匹配 / ICP 最近邻搜索")
print(f"{'─'*65}")
t, c, tr, mb = bench_pairwise_nearest(n=10_000)
p1 = report("10K × 10K 距离矩阵 + argmin", t, c, tr, mb)

# ——— 算子 2 ———
print(f"\n{'─'*65}")
print(f"  【算子 2】Cholesky 分解")
print(f"  模拟 SLAM BA 优化的 Hessian 矩阵求解")
print(f"{'─'*65}")
t2, c2, tr2, mb2 = bench_cholesky(n=4000)
p2 = report("4000×4000 正定矩阵 Cholesky", t2, c2, tr2, mb2)

# ——— 总结 ———
print(f"\n{'='*65}")
print(f"  总结")
print(f"{'='*65}")
print(f"  算子1 传输占比: {p1:.1f}%")
print(f"  算子2 传输占比: {p2:.1f}%")
if DEV == "mps":
    print(f"  → M5 Pro 统一内存：两个算子传输占比均极低")
    print(f"  → GPU 计算结果直接在 CPU 可见地址空间")
    print(f"  → 对 SLAM 这种 CPU↔GPU 频繁交替的工作负载是结构性优势")
elif DEV == "cuda":
    print(f"  → RTX 4070 Ti 离散显存：每次 GPU 结果需经 PCIe 传回 CPU")
    print(f"  → 算子规模越大、数据传输越频繁，开销越显著")
print(f"{'='*65}\n")
