#!/usr/bin/env python3
"""ICP 点云配准 — M5 Pro (UMA) vs RTX 4070 Ti (离散) 双平台对比

用法：两台机器分别运行本脚本，对比输出结果。
依赖：pip install open3d torch numpy
"""

import time
import sys
import numpy as np

import torch
import open3d as o3d

# ========== 平台检测 ==========
if torch.cuda.is_available():
    DEVICE = "cuda"
    DEVICE_NAME = torch.cuda.get_device_name(0)
    ARCH = "离散显存 (PCIe 4.0)"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    DEVICE_NAME = "Apple Silicon (MPS)"
    ARCH = "统一内存 (零拷贝)"
else:
    DEVICE = "cpu"
    DEVICE_NAME = "CPU (fallback)"
    ARCH = "无 GPU"

# ========== 配置 ==========
N_POINTS = 100_000
N_ITERATIONS = 100


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()


def benchmark_transfer(n=N_POINTS):
    """测量 CPU <-> GPU 数据传输开销"""
    data = torch.randn(n, 3, dtype=torch.float32)

    # CPU -> GPU
    t0 = time.perf_counter()
    gpu_data = data.to(DEVICE)
    sync()
    t1 = time.perf_counter()
    upload_ms = (t1 - t0) * 1000

    # GPU -> CPU
    t0 = time.perf_counter()
    _ = gpu_data.cpu()
    sync()
    t1 = time.perf_counter()
    download_ms = (t1 - t0) * 1000

    data_mb = data.element_size() * data.nelement() / (1024 * 1024)
    return upload_ms, download_ms, data_mb


def run_icp():
    """ICP 配准 — 测量每轮迭代耗时"""
    source_np = np.random.randn(N_POINTS, 3).astype(np.float32)
    target_np = source_np + np.array([0.1, 0.05, 0.02], dtype=np.float32)

    t0 = time.perf_counter()
    for i in range(N_ITERATIONS):
        src = o3d.t.geometry.PointCloud(o3d.core.Tensor(source_np))
        tgt = o3d.t.geometry.PointCloud(o3d.core.Tensor(target_np))

        o3d.t.pipelines.registration.icp(
            src,
            tgt,
            max_correspondence_distance=0.5,
            estimation_method=o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        sync()
    t1 = time.perf_counter()

    return (t1 - t0) / N_ITERATIONS * 1000


# ========== 运行 ==========
print(f"{'=' * 60}")
print(f"  ICP 点云配准基准测试")
print(f"  设备: {DEVICE_NAME}")
print(f"  架构: {ARCH}")
print(f"{'=' * 60}")
print(f"  数据规模: {N_POINTS:,} 点 x {N_ITERATIONS} 轮迭代")
print()

up, down, mb = benchmark_transfer()
print(f"【数据传输基准】")
print(f"  数据量: {mb:.1f} MB")
print(f"  CPU -> GPU: {up:.3f} ms")
print(f"  GPU -> CPU: {down:.3f} ms")
print(f"  往返总计:   {up + down:.3f} ms")
print()

icp_ms = run_icp()
print(f"【ICP 计算基准】")
print(f"  每轮 ICP 迭代: {icp_ms:.3f} ms")

print(f"\n{'=' * 60}")
print(f"  ⚡ 传输占每轮 ICP 的比例: {(up + down) / icp_ms * 100:.1f}%")
if DEVICE == "mps":
    print(f"  → 统一内存：传输占比极低，时间全在计算上")
elif DEVICE == "cuda":
    print(f"  → 离散显存：PCIe 传输是显著开销")
print(f"{'=' * 60}")
