#!/usr/bin/env python3
"""大模型推理基准 — M5 Pro vs RTX 4070 Ti

测试不同规模模型的加载能力与推理速度。
核心对比点：统一内存的容量优势 vs 离散显存的容量天花板。

用法：
  # M5 Pro (24GB):
  python llm_benchmark.py --backend mlx

  # RTX 4070 Ti (12GB):
  python llm_benchmark.py --backend llama.cpp

依赖：
  M5 Pro: pip install mlx mlx-lm
  RTX 4070 Ti / 通用: pip install llama-cpp-python
  通用: pip install torch
"""

import argparse
import time
import subprocess
import sys
import json

# ========== 平台检测 ==========
import torch

if torch.cuda.is_available():
    DEV = "cuda"
    DEVICE_NAME = torch.cuda.get_device_name(0)
    VRAM_GB = torch.cuda.get_device_properties(0).total_mem / (1024**3)
elif torch.backends.mps.is_available():
    DEV = "mps"
    DEVICE_NAME = "Apple Silicon (MPS)"

    # 估算统一内存总量
    try:
        result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        VRAM_GB = int(result.stdout.strip()) / (1024**3)
    except:
        VRAM_GB = 24  # fallback
else:
    DEV = "cpu"
    DEVICE_NAME = "CPU"
    VRAM_GB = 0


def print_header():
    print(f"\n{'='*60}")
    print(f"  大模型推理基准测试")
    print(f"  设备: {DEVICE_NAME}")
    print(f"  可用内存: {VRAM_GB:.1f} GB")
    print(f"{'='*60}\n")


def check_model_fit(model_name, size_gb_fp16):
    """检查模型能否装入可用内存"""
    can_fp16 = VRAM_GB > size_gb_fp16 * 1.1
    need_quantize = size_gb_fp16 / 2  # 4-bit 约 1/4
    can_4bit = VRAM_GB > need_quantize * 1.1

    print(f"  模型: {model_name}")
    print(f"  FP16 权重: {size_gb_fp16:.1f} GB")
    print(f"  4-bit 量化: {need_quantize:.1f} GB")
    print(f"  FP16 能跑: {'是' if can_fp16 else '否 ⚠ 需量化'}")
    print(f"  4-bit 能跑: {'是' if can_4bit else '否 ⚠ 内存不足'}")
    return can_fp16, can_4bit


def test_mlx(model_id="mlx-community/Qwen2.5-7B-Instruct-4bit"):
    """使用 MLX 测试推理速度（仅 Apple Silicon）"""
    try:
        from mlx_lm import load, generate
    except ImportError:
        print("  请先安装: pip install mlx mlx-lm")
        print("  然后运行: mlx_lm.download --model mlx-community/Qwen2.5-7B-Instruct-4bit")
        return

    print(f"  加载模型: {model_id}")
    t0 = time.perf_counter()
    model, tokenizer = load(model_id)
    t1 = time.perf_counter()
    print(f"  加载耗时: {t1 - t0:.1f}s")

    prompt = "解释 GPU 的统一内存架构和离散显存架构的区别"
    print(f"  测试提示词: {prompt}")

    t0 = time.perf_counter()
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=128,
        verbose=False,
    )
    t1 = time.perf_counter()
    elapsed = t1 - t0
    tokens = len(tokenizer.encode(response))
    print(f"  生成 {tokens} tokens, 耗时 {elapsed:.1f}s")
    print(f"  速度: {tokens / elapsed:.1f} tokens/s")


def test_llama_cpp(model_path):
    """使用 llama.cpp Python 绑定测试（通用，支持 GPU offload）"""
    try:
        from llama_cpp import Llama
    except ImportError:
        print("  请先安装: pip install llama-cpp-python")
        return

    n_gpu_layers = 999 if DEV == "cuda" else -1
    print(f"  加载模型: {model_path}")
    print(f"  GPU offload layers: {n_gpu_layers}")

    t0 = time.perf_counter()
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=2048,
        verbose=False,
    )
    t1 = time.perf_counter()
    print(f"  加载耗时: {t1 - t0:.1f}s")

    prompt = "解释 GPU 的统一内存架构和离散显存架构的区别"
    print(f"  测试提示词: {prompt}")

    t0 = time.perf_counter()
    output = llm(
        prompt,
        max_tokens=128,
        echo=False,
    )
    t1 = time.perf_counter()
    elapsed = t1 - t0
    tokens = output["usage"]["completion_tokens"]
    print(f"  生成 {tokens} tokens, 耗时 {elapsed:.1f}s")
    print(f"  速度: {tokens / elapsed:.1f} tokens/s")


# ========== 主流程 ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["mlx", "llama.cpp", "auto"],
        default="auto",
        help="推理后端 (默认: M 系列自动选 MLX, 其他选 llama.cpp)",
    )
    args = parser.parse_args()

    print_header()

    # 自动选后端
    if args.backend == "auto":
        if DEV == "mps":
            args.backend = "mlx"
        else:
            args.backend = "llama.cpp"

    # 模型列表及 FP16 大小
    models = [
        ("Qwen2.5-7B", 14),
        ("Qwen2.5-14B", 28),
        ("Llama-3.1-8B", 15),
        ("Llama-3.1-70B", 140),
    ]

    print("【模型容量检查】")
    for name, size_gb in models:
        can_fp16, can_4bit = check_model_fit(name, size_gb)
        print()

    print("【推理速度测试】")
    if args.backend == "mlx":
        test_mlx()
    elif args.backend == "llama.cpp":
        # 需要用户先下载 GGUF 模型
        print(
            "  请先下载 GGUF 格式模型，例如:"
            "\n  wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"
            "\n  然后运行:"
            "\n  python llm_benchmark.py --backend llama.cpp"
        )

    print(f"\n{'='*60}")
    print(f"  说明：")
    print(f"  - M5 Pro 24GB: 7B 模型 FP16 直接加载，70B 需量化")
    print(f"  - RTX 4070 Ti 12GB: 7B 模型必须量化才能跑")
    print(f"  - 这不是算力差距，是显存容量天花板不同")
    print(f"{'='*60}\n")
