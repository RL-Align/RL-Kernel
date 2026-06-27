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
| `h100-template` | H100 SXM5 | Hopper | pending | CUDA pending | pending | pending | pending |
| `mi300-template` | MI300X | CDNA 3 | pending | ROCm pending | pending | pending | pending |

## Selected LogP Results

| Environment | Backend | Batch | Sequence Length | Vocabulary | Dtype | Latency (ms) | Tokens/s | Peak VRAM (GB) | Status | Command |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| `h100-template` | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| `mi300-template` | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

## Sampling Results

| Environment | Backend | Batch | Vocabulary | Top-k | Top-p | Temperature | Latency (ms) | Tokens/s | Peak VRAM (GB) | Status | Command |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `h100-template` | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| `mi300-template` | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

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
