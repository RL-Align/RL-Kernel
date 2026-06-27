<p align="center">
  <img src="docs/assets/logo.png" width="220" alt="RL-Kernel Logo">
</p>

<h1 align="center">RL-Kernel</h1>

<p align="center">
  <strong>Extreme Infrastructure for GRPO & Large-Scale Reinforcement Learning.</strong>
</p>

<p align="center">
  <a href="https://rl-align.github.io/RL-Kernel/"><img src="https://img.shields.io/badge/Documentation-Docs-2ea44f" alt="Documentation"></a>
  <a href="https://discord.gg/RGUQrr74z"><img src="https://img.shields.io/badge/Discord-Join%20Us-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://deepwiki.com/RL-Align/RL-Kernel"><img src="https://img.shields.io/badge/Ask-DeepWiki-7B3FE4" alt="Ask DeepWiki"></a>
  <a href="https://github.com/RL-Align/RL-Kernel"><img src="https://img.shields.io/badge/Hardware-NVIDIA%20CUDA%20%7C%20AMD%20ROCm-orange" alt="Hardware"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
</p>

**RL-Kernel** is a high-performance, memory-efficient infrastructure for Reinforcement Learning post-training. It eliminates the memory and latency bottlenecks in Large Language Model alignment, This project targets AI infrastructure engineers, algorithm researchers, and enterprise-level large model alignment scenarios, providing specialized kernels for algorithms like **GRPO**, **PPO**, and **DPO**.


---

# Our Core Philosophy

**1. Operator-Level Train-Inference Consistency**
The biggest hidden barrier in large-scale RL is the subtle numerical divergence between rollout engines (e.g., vLLM) and training engines (e.g., Megatron/DeepSpeed). RL-Kernel provides mathematically rigorous, fused operators that lock down the computational graph. By guaranteeing absolute numerical consistency and deterministic reduction orders across the entire RL loop, we prevent reward hacking and distribution drift at the operator level.

**2. Extreme Memory & Compute Efficiency**
We replace naive PyTorch paths—which suffer from $O(G \cdot L \cdot V)$ memory explosion—with specialized industrial-grade kernels (like `prefix_shared_attention` and `fused_logp`). This reduces VRAM consumption by up to 10x, unlocking massive batch sizes for GRPO workloads without triggering Out-Of-Memory (OOM) errors.

---

# Global Architecture

RL-Kernel sits strictly at the operator layer, acting as a non-intrusive bridge between high-level alignment orchestration (e.g., vime, slime) and foundational execution engines. We ensure maximum throughput and rigorous numerical parity without modifying upstream framework source code.

<p align="center">
  <img src="docs/assets/RL-Kernel underlying operator library technical architecture.png" alt="RL-Kernel Global Architecture" width="800">
</p>

*Note: RL-Kernel integrates natively into Rollout Engines (vLLM, sglang, LMDeploy) and Training Engines (Megatron, DeepSpeed) via non-intrusive custom operator hooks, powered by underlying CUDA, Triton, and ROCm backends.*

---

# Performance Benchmarks: Breaking the Memory Wall

RL-Kernel is designed to solve the $O(G \cdot L \cdot V)$ memory explosion in DeepSeek-style **GRPO** training. A typical scenario is as follows:

### 1. Logprob Computation (Training Stability)
By implementing **Pre-allocated Chunking**, RL-Kernel maintains constant additional VRAM overhead regardless of the group size ($G$).

**Testbed**: NVIDIA A100 80GB | **Model**: Llama-3-8B | **Vocab**: 128,256 | **SeqLen**: 512

| Group Size ($G$) | TRL (Standard) | PyTorch Native | **RL-Kernel (Ours)** | Status |
| :--- | :--- | :--- | :--- | :--- |
| **G = 64** | OOM | 15.66 GB | **16.15 GB** | Success |
| **G = 128** | OOM | 31.31 GB | **31.80 GB** | Success |
| **G = 256** | **FAILED (OOM)** | 62.63 GB | **63.12 GB** | **Optimized** |

*Note: RL-Kernel is the only solution that successfully scales G=256 on a single A100 by keeping extra VRAM usage to a constant ~0.5GB.*


<p align="center">
  <img src="docs/assets/1. VRAM Saving.png" alt="Logprob VRAM Benchmark">
</p>

### 2. Sampling Latency (Rollout Speed)
Integrating **FlashInfer** fused kernels to accelerate the bottleneck of RL training: the sampling phase.

| Batch Size ($G$) | Native PyTorch | **RL-Kernel (Fused)** | **Speedup** |
| :--- | :--- | :--- | :--- |
| 32 | 176.79 ms | **1.08 ms** | **163x** |
| 64 | 10.54 ms | **1.31 ms** | **8x** |
| 128 | 18.89 ms | **1.86 ms** | **10x** |
| 256 | 36.23 ms | **2.94 ms** | **12x** |


<p align="center">
  <img src="docs/assets/2. sampling latency.png" alt="Sampling Latency Benchmark">
</p>

### 3. Real Model Validation (Qwen3-30B-A3B MoE)

**Testbed**: NVIDIA A100 80GB | **Model**: Qwen3-30B-A3B | **Vocab**: 151,936 | **dtype**: fp16

Model weights consume 56.9 GB — only 23 GB headroom remaining for training computation.


<p align="center">
  <img src="docs/assets/3. moe .png" alt="Real Model MoE Benchmark">
</p>

# Key Features

- **Zero-Growth Memory Pool**: Uses pre-allocated buffers and micro-chunking to prevent VRAM spikes during advantage calculation.
- **Fused Sampling Pipeline**: Direct integration with **FlashInfer** and **vLLM** backends for sub-2ms sampling latency.
- **Universal Backend Abstraction**: Unified API supporting both **NVIDIA (CUDA/FlashInfer)** and **AMD (ROCm/AITER)**.
- **Post-Training Ready**: Drop-in replacement for standard sampling and logprob operators in TRL or DeepSpeed-Chat.

---

# Architecture

RL-Kernel sits between high-level alignment libraries and low-level GPU kernels, ensuring maximum throughput without sacrificing flexibility.

---

# Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/RL-Align/RL-Kernel.git
cd RL-Kernel

# Install core dependencies (CUDA 12.4+ recommended)
pip install -e .
```

### Contributions
Inspired by the kernel designs of vLLM and DeepSpeed. As an active contributor to the AI Infrastructure ecosystem, RL-Kernel aims to push the boundaries of RL efficiency.

Target: Building the most efficient RLHF toolchain for the open-source community.

# Support
Don’t hesitate to ask!

Contact the developers and community on the [Discord](https://discord.gg/RGUQrr74z) if you need any help.

[Open an issue](https://github.com/RL-Align/RL-Kernel/issues) if you find a bug in **RL-Kernel**.

# Documentation

The documentation of **RL-Kernel** is located on the website: [https://rl-align.github.io/RL-Kernel](https://rl-align.github.io/RL-Kernel)
or in the [docs](./docs) directory of the source code.

# Acknowledgments

RL-Kernel builds on the shoulders of excellent open-source projects:

- **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** — We integrate FlashInfer's fused sampling kernels as the NVIDIA backend for our sampling pipeline. The sub-2ms sampling latency results are enabled by FlashInfer's highly optimized CUDA operators.
- **[vLLM](https://github.com/vllm-project/vllm)** — Inspired by vLLM's kernel design philosophy and hardware-aware scheduling approach.
- **[DeepSpeed](https://github.com/microsoft/DeepSpeed)** — Inspired by DeepSpeed's approach to memory-efficient training infrastructure.

We are grateful to these teams for their contributions to the open-source AI infrastructure ecosystem.
