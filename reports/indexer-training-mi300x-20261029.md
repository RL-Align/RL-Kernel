# Item 26 MI300X operator report, 2026-10-29

Base `main`: `8588158650aff58c9c1f527674de94bad06c3d10`.
Target: `dsv4-rocm` after prerequisite PRs #590, #599, #616, #618,
#624, #628 and #629; this branch also includes those open histories.
GPU: MI300X/gfx942, one visible card in `rl-kernel-ljj`.
Numerical providers: pinned aiter `326cffa38c9eda16bef19375ba55334e4d3acd59`,
mLite `327a238239448de6771beea185dc6902a8de3eaa`, installed vLLM BI
batched GEMM; explicit `TRITON_F32_DEFAULT=ieee`. No new GPU kernel.

Validation: 101 new item-26 tests passed; 963 contract/model tests passed
with 19 warnings, no skips reported. Ruff passed on the ten added Python
files. The exact external source hashes are checked by the provisioning
script and runtime provider fingerprints. The tests are mathematical
correctness plus bytewise repeatability checks, not full-model parity.

`benchmark_indexer_training.py --repeats 10` measured checked API wall
microseconds, including admission, allocation and canonical collection.
These are not isolated kernel times or throughput claims:

| Precision | Queries | Selected | Forward record us | Canonical fwd+bwd us |
|---|---:|---:|---:|---:|
| MXFP4 | 1 | 8 | 1311 | 3516 |
| MXFP4 | 16 | 32 | 1196 | 3494 |
| MXFP4 | 64 | 128 | 1226 | 3687 |
| FP8 | 1 | 8 | 1283 | 3521 |
| FP8 | 16 | 32 | 2114 | 4501 |
| FP8 | 64 | 128 | 4534 | 6805 |

Full timings, workspace estimates and implementation fingerprints:
`/outputs/item26-20261029/benchmark.json` on the MI300X host. No
performance comparison with native vLLM/Megatron has been performed.

Outstanding: actual framework integration, CP/PP and multi-node behavior,
Graph, LoRA updates/scale barriers and complete Flash-0731 logprob parity.
The model directory was empty during this run. These remain untested.
