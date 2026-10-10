<p align="center">
  <img src="docs/assets/logo.png" width="220" alt="RL-Kernel logo">
</p>

<h1 align="center">RL-Kernel</h1>

<p align="center">
  <strong>Building cross-hardware and multi-model RL post-training infrastructure for kernel-level train–inference consistency.</strong>
</p>

<p align="center">
  <a href="https://rl-align.github.io/RL-Kernel/"><img src="https://img.shields.io/badge/Documentation-Docs-2ea44f" alt="Documentation"></a>
  <a href="https://rlalign.ai"><img src="https://img.shields.io/badge/Website-rlalign.ai-FF844B?logo=googlechrome&logoColor=white" alt="RL-Align website"></a>
  <a href="https://rl-align.slack.com/join/shared_invite/zt-46bxj7uyt-gEK3xzwSJr_lppJsZolR~g#/shared-invite/email"><img src="https://img.shields.io/badge/Slack-Join%20Us-4A154B" alt="Slack"></a>
  <a href="https://www.linkedin.com/company/rl-align"><img src="https://img.shields.io/badge/LinkedIn-Follow-0A66C2?logo=linkedin&logoColor=white" alt="Follow RL-Align on LinkedIn"></a>
  <a href="https://x.com/RLKernel"><img src="https://img.shields.io/badge/X-Follow-000000?logo=x&logoColor=white" alt="Follow RL-Kernel on X"></a>
  <a href="docs/community/wechat.md"><img src="https://img.shields.io/badge/WeChat-Join%20Group-07C160?logo=wechat&logoColor=white" alt="WeChat"></a>
  <a href="docs/assets/whatsapp-group.png"><img src="https://img.shields.io/badge/WhatsApp-Join%20Group-25D366?logo=whatsapp&logoColor=white" alt="WhatsApp"></a>
  <a href="https://deepwiki.com/RL-Align/RL-Kernel"><img src="https://img.shields.io/badge/Ask-DeepWiki-7B3FE4" alt="Ask DeepWiki"></a>
  <a href="#hardware-support"><img src="https://img.shields.io/badge/Supported-CUDA%20%7C%20ROCm-2ea44f" alt="CUDA and ROCm supported"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache 2.0 license"></a>
</p>

<p align="center">
  <a href="#architecture">Architecture</a> ·
  <a href="#current-scope-and-roadmap">Current scope</a> ·
  <a href="#benchmark-highlights">Results</a> ·
  <a href="#hardware-support">Hardware support</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="https://rl-align.github.io/RL-Kernel/">Documentation</a>
</p>

**RL-Kernel** is high-performance infrastructure for RL post-training. It provides
deterministic operators for consistent numerical computation across rollout and training
engines, together with hardware-specific kernels for faster execution and lower memory
use in GRPO, PPO, and related workloads.

Today, the end-to-end path covers Qwen3-8B Dense with vime, vLLM, and Megatron-LM.
Work on DeepSeek-V4 Flash MoE, Miles, and AReaL is ongoing.

## Why RL-Kernel?

Rollout and training engines can produce different log probabilities for the same tokens
and model weights because their kernels, batching, and reduction orders differ. Those
differences enter the policy ratios and KL terms used by RL algorithms.

- **Exact train–inference consistency:** deterministic operators keep rollout and training
  computations aligned. The published experiment records exact runtime LogP agreement
  across all 200 training steps.
- **RL operators:** deterministic attention, dense FFN, LogP, GRPO and PPO objectives,
  and collectives cover the numerical boundaries in RL post-training.
- **Performance:** fused computation and hardware-specific kernels reduce rollout time,
  memory use, and synchronization costs.
- **vime integration:** vime orchestrates vLLM rollout and Megatron-LM training, with
  RL-Kernel supplying the operators used by both engines.
- **Hardware:** NVIDIA SM90 and AMD gfx942 are supported. Ascend dav_c220 has partial
  operator coverage. Support for other hardware is in progress.

## Repository layout

The [directory and ownership guide](docs/architecture/repository-layout.md) describes
models, contracts, shared operators, hardware backends, rollout/train engines, tests,
CI and benchmark locations. See the [Dense refactor acceptance checklist](docs/validation/refactor-acceptance.md)
for CUDA/ROCm validation. Legacy import and launcher paths remain supported.

## Architecture

RL-Kernel sits between execution engines and accelerator backends. Its runtime adapters
select the operator implementation for each backend while keeping the same numerical
contract across rollout and training.

The architecture below shows how orchestration frameworks, execution engines, RL-Kernel
operators, and hardware backends fit together.

<p align="center">
  <img src="docs/assets/RL-Kernel underlying operator library technical architecture.png" alt="RL-Kernel global architecture" width="800">
</p>

## Current Scope and Roadmap

The current end-to-end path uses Qwen3-8B Dense with vime.

| Area | Current | Next |
| :--- | :--- | :--- |
| **Model** | Qwen3-8B Dense | [DeepSeek-V4-Flash-0731 MoE](docs/blog/2026-08-09-dsv4-flash-moe-consistency-roadmap.md) |
| **Orchestration** | vime | Miles and AReaL |
| **Engines** | vLLM rollout and Megatron-LM training | More rollout and training engines |

## Benchmark Highlights

### CUDA H100

<img src="docs/assets/blog/rl-kernel-v0.1.0/cuda-training-consistency.png" alt="Qwen3-8B H100 train/rollout mismatch count and maximum absolute LogP difference over 200 steps">

<img src="docs/assets/blog/rl-kernel-v0.1.0/cuda-mean-logprob-difference.png" alt="Qwen3-8B H100 mean absolute train/rollout LogP difference over 200 steps">

<img src="docs/assets/blog/rl-kernel-v0.1.0/cuda-performance.png" alt="Qwen3-8B H100 200-step performance matrix">

### ROCm MI300X

<img src="docs/assets/blog/rl-kernel-v0.1.0/rocm-training-consistency.png" alt="Qwen3-8B ROCm train/rollout mismatch count and maximum absolute LogP difference over 200 steps">

<img src="docs/assets/blog/rl-kernel-v0.1.0/rocm-mean-logprob-difference.png" alt="Qwen3-8B ROCm mean absolute train/rollout LogP difference over 200 steps">

<img src="docs/assets/blog/rl-kernel-v0.1.0/rocm-performance.png" alt="Qwen3-8B ROCm MI300X 200-step performance matrix">

## Hardware Support

RL-Kernel currently supports the following hardware targets.

| Hardware | Architecture | Software | Status |
| :--- | :--- | :--- | :--- |
| NVIDIA H100, H200, GH200 | SM90 | CUDA | **Supported** |
| AMD Instinct MI300A, MI300X, MI325X | gfx942 | ROCm | **Supported** |
| Huawei Ascend dav_c220 | dav-2201 | CANN 9.1.0 and Ascend C | **Partial** |
| Moore Threads | In development | MUSA | **In progress** |

The published end-to-end benchmark was run on H100. The ROCm extension and backend
checks were verified on MI300X. Ascend support is limited to dav_c220. Support for other
hardware models is in progress.

## Quick Start

Use a compatible vime environment on an eight-GPU H100 or MI300X node.
Clone the project and follow the [installation guide](docs/getting_started/installation.md)
for your CUDA or ROCm build:

git clone https://github.com/RL-Align/RL-Kernel.git<br>
cd RL-Kernel

Complete the one-time [CUDA setup](docs/usage/qwen3-vime-consistency.md#cuda-quick-path)
or [ROCm setup](docs/usage/qwen3-vime-consistency.md#rocm-mi300x-and-gfx942).
Save the backend, Python, framework, model and data paths in .rlk-profile.json
or select a profile with RLK_REPRO_PROFILE. No launcher edits are needed.
Both backends then use the same command:

```bash
./rlk run --tp 4 --rollout-tp 4 --temperature 0.7 --top-p 0.95 \
  --lr 5e-7 --kl-coef 0.01 --max-response-len 6912 --max-tokens-per-gpu 4096 --steps 200
```

Training TP and rollout TP are independent: choose 1, 2, 4 or 8. Training CP
defaults to 8 / TP; set --cp explicitly if needed. run waits, validates
train/rollout LogP, and defaults to consistency mode without rollout-logprob reuse.
Add --mode native for a native comparison, or replace run with plan to
inspect the command without launching a job.

On ROCm, this command selects Triton chunked Attention and sparse top-p
logprob/monitoring-entropy scoring. Training logprobs are independently
recomputed and validation requires bitwise agreement with rollout. Apply the
updated ROCm companion patches and rebuild the extension when updating.
With the supplied ROCm profile, a three-step check measured 58.29 s/step
versus native's 57.40 s (+1.55%), with zero raw-bit logprob mismatches.
Use `--steps 3` for that short check; timing depends on generated lengths and
the environment, and the native execution-record limitation is documented below.
See [ROCm performance reproduction](docs/usage/rocm-sparse-performance.md)
for the full comparison command, measurements and validation limits.

On CUDA, rollout CP and top-k are configurable too; this short check performs
two real updates and validates their artifacts:

```bash
./rlk verify --tp 1 --rollout-tp 1 --rollout-cp 8 --temperature 0.7 --top-p 0.95 --top-k -1
```

Rollout TP × CP must divide eight. CUDA accepts top-k -1 (disabled) or a
positive integer, and temperature 0 for greedy sampling. ROCm currently
requires top-k -1 and positive temperature. ROCm rollout CP > 1 remains
unvalidated, and verify is CUDA-only. See the
[CUDA/ROCm support table](docs/usage/cuda-rocm-consistency-audit.md#common-command)
and [measured H100 results](docs/usage/h100-matrix-validation.md) for exact coverage.

## Community and Contributions

Join us on [Slack](https://rl-align.slack.com/join/shared_invite/zt-46bxj7uyt-gEK3xzwSJr_lppJsZolR~g#/shared-invite/email)
or [WeChat](docs/community/wechat.md), and
[open an issue](https://github.com/RL-Align/RL-Kernel/issues) for bugs and feature requests.
Contributions to kernels, framework integrations, hardware adaptation, and benchmarks
are welcome. See the [contributing guide](docs/contributing/README.md).

## Acknowledgments

RL-Kernel builds on the work of the open-source AI infrastructure community, including
[vime](https://github.com/vllm-project/vime), [vLLM](https://github.com/vllm-project/vllm),
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM), and
[FlashInfer](https://github.com/flashinfer-ai/flashinfer).
We thank their contributors and everyone helping bring RL-Kernel to new accelerators.

Licensed under the [Apache License 2.0](LICENSE).
