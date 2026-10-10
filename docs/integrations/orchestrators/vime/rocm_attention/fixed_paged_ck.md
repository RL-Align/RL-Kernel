# Fixed paged CK attention on ROCm

Set `RL_KERNEL_ROCM_FIXED_PAGED_TILE=128` before starting training and rollout
workers to select the RL-Kernel-owned CK instantiation. `64` selects the other
fixed schedule; `0` (default) retains the AITER entrypoint. This path reuses
installed `aiter_meta`/Composable Kernel headers and requires a ROCm C++ build
toolchain. The extension builds lazily and must be warmed before HIP Graph
capture. It was validated on MI300X VF (`gfx942`).

The supported specialization is BF16, head dimension 128, page size 16, with
no dropout, softcap, bias, or sliding window. Both training and rollout use
the same fixed tile. Implicit scalar FMA contraction is disabled in this
extension to preserve rounding across differently sized query chunks;
explicit CK MFMA instructions remain enabled. Cache addressing stays 64-bit
when either K/V view spans more than 2 GB.

The adapter pads page tables to full 128-token KV tiles and sanitizes unused
columns and inactive graph rows before CK can load their page IDs. Decode
query rows retain their request mapping as the active batch shrinks. These
operations prepare metadata without copying the dense KV tensor. A
materialized fallback remains available for other adapter call layouts and
uses the fixed arithmetic when enabled.

`RL_KERNEL_ROCM_PAGED_KV_MAX_TOKENS` optionally bounds the scheduling length.
It must cover the complete prompt-plus-response KV length; an asynchronous
device assertion rejects lengths above the bound. Leave it unset for general
workloads. Changing either setting changes the graph cache identity; restart
workers when changing settings.

## One-round R/R reproduction

In the validated AMD container, install this checkout at
`/workspace/RL-Kernel-pr390`, with Vime at `/workspace/vime`, Megatron at
`/workspace/Megatron-LM-vime`, and the model/data paths used by
`run_full_rr_single_arm_v90.py`. From the repository root, run:

```bash
python -m rl_engine.integrations.orchestrators.vime.experiments.rocm_attention.run_full_rr_fixed_paged_ck \
  --tile 128 --run-dir /app/model/vime-runs/fixed-paged-ck-rr
```

Use a new output directory. This runs one R/R rollout/training iteration,
with 8 GPUs, training TP4/CP2, two TP4 rollout engines, round-robin routing,
Qwen3-8B BF16, batch 1 with 8 samples, global batch 8, response limit 7168,
4096 tokens per training GPU, seed 1234, and KV scheduling bound 8192.
FFN and logp remain R/R. The helper is also the shared dependency of the
existing 200-round R/R runner; adding it does not enable fixed tiles in that
runner by default.

## Diagnostic probes

```bash
python -m rl_engine.integrations.orchestrators.vime.experiments.rocm_attention.probe_paged_dispatch \
  --fixed-tile 128 --guard-pages --graph
python -m rl_engine.integrations.orchestrators.vime.experiments.rocm_attention.bench_paged_dispatch
```

The probe prints raw BF16 bit comparisons on identical input data. It covers
full/suffix queries, physical KV layouts, and dynamic graph replay. The
benchmark compares AITER dynamic dispatch against fixed CK on the same
inputs. Neither replaces full training/rollout logprob validation. Performance
depends on cache layout and query shape; fixed M128 is not uniformly faster.
