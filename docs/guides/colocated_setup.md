# Colocated Dual-Engine Setup Guide

> **Status**: Experimental — validates the colocated architecture on 8×A100-SXM4-80GB.

## Overview

In standard RLHF/GRPO training, inference (rollout) and training run on **separate GPU pools**. This wastes GPU time — training GPUs idle during rollout, and inference GPUs idle during training.

The **colocated** architecture eliminates this waste by sharing ALL GPUs between both phases:

```
Disaggregated (standard):
  GPU 0-5: Training only    ← idle during rollout
  GPU 6-7: vLLM only        ← idle during training

Colocated (this guide):
  GPU 0-7: Both phases       ← always active
  Phase 1: vLLM wake  → 8-GPU inference → vLLM sleep
  Phase 2: DeepSpeed  → 8-GPU training  → weight sync
```

## How It Works

The key enabler is **vLLM's sleep mode** (v0.8+):

1. `llm.sleep(level=2)` — offloads model weights to CPU, discards KV cache, frees ~90% GPU memory
2. `llm.wake_up()` — reloads weights onto GPU, reallocates KV cache (~2-8 seconds)

This allows DeepSpeed to claim GPU memory for training while vLLM is sleeping, and vice versa.

### Weight Synchronization

After each training step, updated weights must be transferred to vLLM. Three methods:

| Method | Latency | Use Case |
|---|---|---|
| Sleep/wake cycle | 2-8s | Simple, always works (vLLM reloads from checkpoint) |
| `reload_weights` API | <1s | In-process vLLM, avoids full reload |
| CUDA VMM (WeightBridge) | ~0ms | Zero-copy, same-GPU (RL-Kernel native) |

The example uses `reload_weights` with sleep/wake fallback.

## Quick Start

```bash
cd RL-Kernel

# Smoke test (tiny model, 3 steps)
python examples/colocated_dual_engine.py \
  --model Qwen/Qwen3-0.6B --num-gpus 1 --steps 3

# Full run (8 GPUs)
python examples/colocated_dual_engine.py \
  --model Qwen/Qwen3-0.6B --num-gpus 8 --steps 20

# Benchmark: colocated vs disaggregated
python benchmarks/benchmark_colocated.py \
  --model Qwen/Qwen3-0.6B --steps 10

# Run tests
pytest tests/test_colocated_dual_engine.py -v
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `--model` | Qwen/Qwen3-0.6B | HuggingFace model path |
| `--num-gpus` | 8 | Number of GPUs (shared for both phases) |
| `--steps` | 5 | Training iterations |
| `--vllm-sleep-level` | 2 | 1=offload weights, 2=discard everything |
| `--vllm-gpu-memory-utilization` | 0.45 | vLLM memory fraction (leave room for training) |
| `--ds-zero-stage` | 2 | DeepSpeed ZeRO stage |
| `--lora-rank` | 16 | LoRA rank for parameter-efficient training |

### Memory Budget

With `gpu_memory_utilization=0.45` on A100-80GB:

- vLLM claims ~36GB per GPU (weights + KV cache)
- After sleep, ~4GB remains (metadata)
- DeepSpeed uses ~32GB (LoRA params + optimizer + activations)
- Headroom: ~44GB for gradient accumulation and spikes

## Architecture

```
┌──────────────────────────────────────────────────┐
│              ColocatedDualEngine                  │
├──────────────────────────────────────────────────┤
│  rollout_phase()                                  │
│    llm.wake_up() → generate() → llm.sleep()      │
│                                                   │
│  training_phase()                                 │
│    engine.forward() → backward() → step()         │
│                                                   │
│  sync_weights_phase()                             │
│    wake(tags=["weights"]) → reload → sleep()      │
├──────────────────────────────────────────────────┤
│  vLLM LLM          │  DeepSpeed Engine            │
│  (sleep/wake)       │  (ZeRO-2 + LoRA)            │
├──────────────────────────────────────────────────┤
│              Shared GPU Pool (N GPUs)             │
└──────────────────────────────────────────────────┘
```

## Relationship to RL-Kernel Components

This example reuses RL-Kernel's existing infrastructure:

- **WeightBridge** (`executors/bridge.py`): CUDA VMM / SharedMemory / IPC transport
- **DeepSpeedTrainingWorker** (`executors/deepspeed_trainer.py`): ZeRO training + weight publishing
- **VLLMSharedPrefixSampler** (`executors/vllm_sampler.py`): Rollout with prefix caching
- **RayActorManager** (`executors/ray_actor_manager.py`): Multi-process orchestration

The colocated example adds the **lifecycle orchestration** layer — the sleep/wake timing and phase sequencing that these components need to share GPUs.

## Known Limitations

1. **Single-node only** — multi-node colocated requires NCCL weight transport (not yet implemented in WeightBridge)
2. **No async overlap** — rollout and training are strictly sequential; async prefetch would further improve throughput
3. **Weight sync overhead** — sleep/wake cycle adds 2-8 seconds per step; CUDA VMM zero-copy would eliminate this
4. **Memory tuning** — `vllm_gpu_memory_utilization` must be tuned per model to avoid OOM

## References

- [vLLM Sleep Mode](https://docs.vllm.ai/en/latest/features/sleep_mode/)
- [veRL HybridFlow (EuroSys 2025)](https://arxiv.org/abs/2409.19256)
- [OpenRLHF Hybrid Engine](https://openrlhf.readthedocs.io/en/latest/hybrid_engine.html)
- [TRL vLLM Colocate](https://huggingface.co/blog/vllm-colocate)
- RL-Kernel Issues: #127, #129, #130
