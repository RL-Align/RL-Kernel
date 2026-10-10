# WS2 Attention PR5 Drift Benchmark

PR5 adds the report artifact path for issue #235. It does not introduce a
production communication kernel. The benchmark is a rank-aware, torchrun-style
driver around the deterministic CP attention reference. Under a matching
two-rank CUDA/NCCL launch it executes the P2P reference transport; CPU/Gloo
remains a report-generation smoke path.

## Scope

The benchmark covers the Qwen3-8B Attention target:

- global heads: `Hq=32`, `Hkv=8`, `D=128`
- TP sweep: `TP=1/2`; TP only changes the local head shard shape
- CP sweep: `CP=1/2`
- modes: full prefill and chunked-prefill replay
- dtype path: BF16 candidate path compared with FP32 reference
- optional backward: `dq`, `dk`, `dv` drift from the PR8 reference
- optional RoPE composition before Attention, while CP Attention still consumes
  post-RoPE Q/K

The report separates two drift classes:

| Field | Meaning |
| --- | --- |
| `drift.cp_merge_fp32` | CP/chunked candidate with FP32 output vs CP=1 FP32 prefill. This isolates CP merge and split-KV order. |
| `drift.dtype_path_vs_fp32` | BF16 candidate path vs FP32 reference. This exposes arithmetic/final-write drift. |
| `merge_order_probe` | Reversed-arrival partial states vs canonical sorted merge. This verifies that arrival order is ignored. |
| `te_merge_oracle` | Optional Transformer Engine merge-oracle drift when TE is installed and passes capability probes. |
| `backward` | Optional PR8 `dq/dk/dv` drift report when `--include-backward` is used. |
| `distributed_p2p_reference` | Real NCCL P2P partial-state gather, FP32 merge, and query scatter drift. |

With `--include-dlogp`, PR5 projects both Attention outputs through the same
deterministic synthetic FP32 lm_head and reports active-token selected-logprob
drift. This closes the operator-attribution leg without pretending to replace
PR4's full Qwen3 model/runtime integration. Without that flag, dlogp is recorded
as `not_requested` rather than silently omitted.

## Commands

Local smoke:

```bash
python benchmarks/distributed/benchmark_ws2_cp_attention_drift.py --smoke --json
```

Qwen3 TP=2 / CP=2 with backward drift and a JSON artifact:

```bash
python benchmarks/distributed/benchmark_ws2_cp_attention_drift.py \
  --smoke \
  --tp-world-sizes 2 \
  --cp-world-sizes 2 \
  --kv-chunk-sizes none,1 \
  --include-backward \
  --include-dlogp \
  --output artifacts/ws2-cp-attention-drift.json
```

Strict GPU acceptance manifest (expected to fail until every required GPU/NCCL
case and the self-owned CUDA AG/RS operators are executable):

```bash
python tools/validation/distributed/ws2_attention_gpu_acceptance.py \
  --mode manifest \
  --output artifacts/ws2-attention-acceptance-manifest.json
```

Strict GPU run after stacking the issue #235 implementation PRs in one checkout:

```bash
python tools/validation/distributed/ws2_attention_gpu_acceptance.py \
  --mode run \
  --output artifacts/ws2-attention-gpu-acceptance.json
```

The orchestrator requires the Qwen3-8B `TP=2, CP=2, BF16` matrix, full and
chunked prefill, FlashInfer paged prefill/decode with disabled and fixed
Split-K, attention-domain `out/lse`, active-token `dlogp`, PR8 `dq/dk/dv`,
batch/page-layout invariance, the P2P NCCL reference, and the self-owned CUDA
AG/RS path. Missing scripts, dry-runs, requested-only Split-K provenance,
skipped collectives, or unavailable metrics fail closed.

Two-GPU NCCL transport check:

```bash
torchrun --standalone --nproc-per-node=2 \
  tools/validation/distributed/ws2_p2p_nccl_attention_reference_check.py
```

Two-GPU benchmark report with real P2P transport:

```bash
torchrun --standalone --nproc-per-node=2 \
  benchmarks/distributed/benchmark_ws2_cp_attention_drift.py \
  --smoke \
  --device cuda \
  --init-process-group \
  --tp-world-sizes 2 \
  --cp-world-sizes 2 \
  --json
```

Rank 0 prints or writes the shared report. Other ranks can run the same
rank-aware benchmark without changing the numerical reducer.  The recommended
container is the repository CUDA image built from `docker/Dockerfile.cuda`
(`ghcr.io/rl-align/rl-kernel/rl-kernel-ci:cuda` when using the repository image
workflow). It is based on PyTorch 2.4 / CUDA 12.4 and includes NCCL support.

## Transformer Engine Reuse

PR5 reuses Transformer Engine only as an optional merge oracle, not as the
source of truth. The adapter imports:

```text
transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py
```

and uses these APIs when available:

```text
flash_attn_fwd_softmax_lse_correction
flash_attn_fwd_out_correction_init
flash_attn_fwd_out_correction
```

The benchmark first builds RL-Kernel partial states:

```text
state_i = (out_i, lse_i, global_block_index_i)
```

then sorts them by `global_block_index`. TE is allowed to perform only the
online-softmax correction arithmetic for those already-sorted states. If TE is
missing, incompatible, or fails the numeric self-test, the report records a
provenance fallback and continues with the deterministic RL-Kernel merge.

## Report Contract

The JSON root contains:

```text
schema_version
issue / pr
launch.command
runtime.rank_env
target
te_context_parallel_merge
dlogp
cases[]
```

Each case records topology, RoPE/cache provenance, split-KV policy, block
metadata hash, drift summaries, per-logical-CP-rank metrics, and optional
backward drift. The merge order is always `global_block_index`, and
`downcast_at` is always `final_write`.

The PR5 report schema is `ws2_cp_attention_drift/v2`. The strict aggregate
report schema is `ws2_attention_gpu_acceptance/v1`.
