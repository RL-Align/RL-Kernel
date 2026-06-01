<p align="center">
  <img src="./docs/assets/rl-engine-log-display.png" width="220" alt="RL-Kernel Logo">
</p>

<h1 align="center">RL-Kernel</h1>

<p align="center">
  <strong>Extreme Infrastructure for GRPO & Large-Scale Reinforcement Learning.</strong>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://github.com/Flink-ddd/RL-Kernel"><img src="https://img.shields.io/badge/Hardware-NVIDIA%20CUDA%20%7C%20AMD%20ROCm-orange" alt="Hardware"></a>
</p>

**RL-Kernel** is a high-performance, memory-efficient infrastructure for Reinforcement Learning (RL) post-training. It eliminates the memory and latency bottlenecks in Large Language Model (LLM) alignment, This project targets AI infrastructure engineers, algorithm researchers, and enterprise-level large model alignment scenarios, providing specialized kernels for algorithms like **GRPO**, **PPO**, and **DPO**.

---

## Performance Benchmarks: Breaking the Memory Wall

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

### 2. Sampling Latency (Rollout Speed)
Integrating **FlashInfer** fused kernels to accelerate the bottleneck of RL training: the sampling phase.

| Batch Size ($G$) | Native PyTorch | **RL-Kernel (Fused)** | **Speedup** |
| :--- | :--- | :--- | :--- |
| 64 | 219.4 ms | **0.55 ms** | **399x** |
| 128 | 14.08 ms | **0.67 ms** | **21x** |
| 256 | 25.49 ms | **1.15 ms** | **22x** |

---

## Key Features

- **Zero-Growth Memory Pool**: Uses pre-allocated buffers and micro-chunking to prevent VRAM spikes during advantage calculation.
- **Fused Sampling Pipeline**: Direct integration with **FlashInfer** and **vLLM** backends for sub-1ms sampling latency.
- **Universal Backend Abstraction**: Unified API supporting both **NVIDIA (CUDA/FlashInfer)** and **AMD (ROCm/AITER)**.
- **Post-Training Ready**: Drop-in replacement for standard sampling and logprob operators in TRL or DeepSpeed-Chat.

---

## Architecture

RL-Kernel sits between high-level alignment libraries and low-level GPU kernels, ensuring maximum throughput without sacrificing flexibility.



---

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/Flink-ddd/RL-Kernel.git
cd RL-Kernel

# Install core dependencies (CUDA 12.4+ recommended)
pip install -e .
```


### Contributions
Inspired by the kernel designs of vLLM and DeepSpeed. As an active contributor to the AI Infrastructure ecosystem, RL-Kernel aims to push the boundaries of RL efficiency.

Target: Building the most efficient RLHF toolchain for the open-source community.
