# Hardware Benchmark Dashboard

This page defines the reporting format for reproducible RL-Kernel hardware benchmarks.
Entries marked `pending` have not yet been measured and are not performance claims.

## Reporting Rules

Every published result must record:

- hardware and software environment;
- RL-Kernel commit;
- exact reproduction command;
- workload shape and dtype;
- selected backend;
- latency, throughput, and peak VRAM;
- status and any limitation.

A fallback backend result must not be presented as a fused-kernel result.

## Status Definitions

| Status | Meaning |
| --- | --- |
| `pass` | The workload completed. Verify the selected backend separately before reporting an optimized result. |
| `blocked` | The workload could not run because of unavailable hardware, dependencies, or compiled extensions. |
| `oom` | The workload exceeded available GPU memory. |
| `pending` | No measurement has been collected yet. |

## Environment Matrix

| Environment ID | GPU | Architecture | Driver | Runtime | PyTorch | RL-Kernel Commit | Date |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `h100-sxm5` | H100 SXM5 80GB HBM3 | Hopper (SM90) | 535.309.01 | CUDA 12.4 | 2.6.0+cu124 | `6df029a` | 2026-07-19 |
| `mi300-template` | MI300X | CDNA 3 | pending | ROCm pending | pending | pending | pending |

## Selected LogP Results

| Environment | Backend | Batch | Sequence Length | Vocabulary | Dtype | Latency (ms) | Tokens/s | Peak VRAM (GB) | Status | Command |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| `h100-sxm5` | `logp_native` (log_softmax + gather) | 32 | 512 | 128256 | float16 | 23.29 | 703,528 | 19.57 | pass | `python scripts/run_profile_suite.py --device cuda --dtype float16 --batch-sizes 32 --seq-lens 512 --vocab-sizes 128256 --workloads logp-native,logp-fused` |
| `h100-sxm5` | `logp_fused` (generic CUDA `fused_logp`) | 32 | 512 | 128256 | float16 | 6.93 | 2,362,773 | 7.83 | pass | same as above |
| `h100-sxm5` | `logp_native` | 16 | 512 | 128256 | float16 | 11.37 | 720,555 | 9.79 | pass | `--batch-sizes 16 --seq-lens 512 --vocab-sizes 128256` |
| `h100-sxm5` | `logp_fused` | 16 | 512 | 128256 | float16 | 3.32 | 2,468,160 | 3.91 | pass | same as above |
| `mi300-template` | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

Full 24-row sweep (batch ∈ {8,16,32} × seq_len ∈ {128,512} × vocab ∈ {4096,128256}):
`reports/perf_report_NVIDIA_H100_80GB_HBM3.csv` (this PR). `logp_fused` selected the generic
CUDA kernel (`FusedLogpGenericOp`), not the experimental SM90 TMA kernel — see the linear-logp
note below for why SM90-specific dispatch is more nuanced.

## Sampling Results

| Environment | Backend | Batch | Vocabulary | Top-k | Top-p | Temperature | Latency (ms) | Tokens/s | Peak VRAM (GB) | Status | Command |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `h100-sxm5` | native (topk→topp→softmax→multinomial) | 64 | 128256 | 50 | 0.9 | 1.0 | 8.73 | n/a | n/a | pass | `python benchmarks/benchmark_sampling.py --g-sizes 32,64,128,256 --vocab-size 128256 --top-k 50 --top-p 0.9` |
| `h100-sxm5` | FlashInfer (`RL_Sampler`) | 64 | 128256 | 50 | 0.9 | 1.0 | 0.84 | n/a | n/a | pass | same as above |
| `h100-sxm5` | native | 256 | 128256 | 50 | 0.9 | 1.0 | 29.30 | n/a | n/a | pass | same as above |
| `h100-sxm5` | FlashInfer (`RL_Sampler`) | 256 | 128256 | 50 | 0.9 | 1.0 | 1.65 | n/a | n/a | pass | same as above |
| `mi300-template` | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

Raw artifact backing the rows above: `reports/benchmark_sampling_NVIDIA_H100_80GB_HBM3.txt`
(this PR) — full console output of the exact command listed, including all four batch sizes
(32/64/128/256), not just the two rows excerpted here.

Limitation: `benchmarks/profiler.py`'s `WORKLOAD_REGISTRY` only registers `sampling-native`
(no `sampling-fused` workload), so the FlashInfer rows above come from
`benchmarks/benchmark_sampling.py` directly rather than `run_profile_suite.py`, and it does not
report tokens/s or peak VRAM. Wiring a `sampling-fused` workload into the profiler would close
this gap (tracked as a follow-up, not done in this PR).

## Linear-LogP: SM90 warp specialization vs generic paths (H100, bf16)

`rl_engine/kernels/ops/cuda/loss/linear_logp.py` has **no generic (SM86) CUDA kernel** — on
Ampere, `linear_logp` falls back to Triton or the native materializing path. The SM90 TMA/WGMMA
kernel is therefore the only path that is both memory-efficient *and* GPU-accelerated for this
op; the question "does SM90 warp specialization help vs SM86" reduces to "does the SM90 kernel
help vs SM86's only options (Triton / native)".

Command: `python benchmarks/benchmark_linear_logp.py` (built with `KERNEL_ALIGN_FORCE_SM90=1`).
Raw artifact: `reports/benchmark_linear_logp_sm90_NVIDIA_H100_80GB_HBM3.txt` (this PR).

| shape (N×H×V) | native fwd ms | triton fwd ms | sm90 fwd ms | native fwd MB | triton fwd MB | sm90 fwd MB | sm90 vs triton | sm90 vs native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096×2048×32768 | 1.80 | 6.29 | 3.65 | 1280 | 0 | 2 | 1.72x faster | 0.49x (slower) |
| 4096×2048×50257 | 10.27 | 9.88 | 5.40 | 1965 | 0 | 2 | 1.83x faster | 1.90x faster |
| 4096×2048×131072 | 7.20 | 24.76 | 14.11 | 5120 | 0 | 2 | 1.75x faster | 0.51x (slower) |

Status: pass. Limitation (report honestly, not cherry-picked): SM90 **consistently beats Triton**
(1.7–1.9x) at ~600–2500x less memory than the native materializing path, but does **not**
uniformly beat native on raw latency — native is faster in forward at two of three vocab sizes
tested, and backward is slower on SM90 across the board (tile-recompute trades FLOPs for
memory). This also reproduces on the real Qwen3-30B-A3B `lm_head` (vocab 151,936) below.

Separately: the standalone (non-`linear_logp`) `fused_logp_sm90` kernel, gated behind
`RL_KERNEL_ENABLE_EXPERIMENTAL_SM90_LOGP=1`, currently aborts the process
(`cuTensorMapEncodeTiled` fails, and `csrc/utils/tma_utils.cuh:49` calls `exit(EXIT_FAILURE)`
instead of raising) at vocab=32768/128256 — filed as a follow-up, out of scope for this PR since
the flag defaults off and isn't part of any existing benchmark claim.

## Real-model validation: Qwen3-30B-A3B (H100, bf16)

Real weights (`Qwen/Qwen3-30B-A3B`, 56.9 GB, downloaded from the HF Hub) and a real forward pass
(not synthetic tensors) were used for the `lm_head` weight and hidden-state distribution.

Command: `python benchmarks/benchmark_qwen3_moe_real_model.py` (script added in this PR).
Raw artifact: `reports/benchmark_qwen3_30b_a3b_NVIDIA_H100_80GB_HBM3.txt` (this PR).

| Metric | Value |
| --- | --- |
| Weight VRAM | 56.87 GB |
| Total H100 VRAM | 79.11 GB |
| Headroom | 22.24 GB |

| N tokens | native extra VRAM | RL-Kernel (SM90) extra VRAM | native ms | RL-Kernel ms | status |
| ---: | ---: | ---: | ---: | ---: | --- |
| 12,288 | 17.39 GB | 0.00 GB | 25.08 | 48.03 | pass |
| 16,384 | OOM | 0.00 GB | n/a | 65.90 | native: oom |
| 24,576 | OOM | 0.00 GB | n/a | 105.50 | native: oom |

Status: pass (memory claim), with the same latency caveat as above — RL-Kernel is ~1.9–2x
slower per call than native at shapes where native still fits; its advantage here is fitting at
all within the 22.24 GB headroom, not raw speed. Hidden states came from one real 29-token
prompt, replicated to build larger N (memory/latency at this stage don't depend on token
content, only shape/dtype) — flagging so this isn't read as N independently-sampled completions.

## Reproduction

Run the profiler from the repository root:

```bash
python scripts/run_profile_suite.py \
  --device cuda \
  --dtype float16 \
  --batch-sizes 8,16,32 \
  --seq-lens 128,512 \
  --vocab-sizes 4096,128256 \
  --workloads logp-native,logp-fused \
  --output reports/logp_profile.csv
```
