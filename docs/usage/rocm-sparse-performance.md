# ROCm G11 performance path

Use the updated [ROCm companion patches](../../examples/vime_rocm_attention_ablation/companion_patches/README.md),
rebuild RL-Kernel for `gfx942`, and configure the local paths in the profile.
The README Quick start command is the canonical entry point:

```bash
./rlk run --tp 4 --rollout-tp 4 --temperature 0.7 --top-p 0.95 \
  --lr 5e-7 --kl-coef 0.01 --max-response-len 6912 --max-tokens-per-gpu 4096 --steps 200
```

The consistency arm uses Triton chunked Attention, deterministic MFMA GEMM,
the sparse HIP top-p logprob/entropy path, and independent training logprob
recomputation. `--use-rollout-logprobs` is not passed. Complete retained-token
support is transported, so arbitrary top-p values remain supported. The
launcher leaves `AMD_CPU_AFFINITY` empty by default and preserves an explicit
operator override, avoiding the old eight-core CPU bottleneck.

Select the supplied ROCm profile during setup. Training CP defaults to 2 for
TP4, rollout CP to 1, and GRPO standard-deviation normalization to enabled.
The profile uses eight samples, global batch 8, rollout batch 1 and vLLM memory
fraction 0.4. All remain overridable; use `plan` to inspect a custom profile.

The launcher records source fingerprints, backend readbacks and raw train /
rollout logprob comparisons. A consistency run fails if any compared value is
not bitwise equal. Temperature, top-p, learning rate, KL coefficient, response
length, training TP/CP and rollout TP/CP are command-line parameters.

## Historical reference

The original 200-step G10/G11 runs used the same TP4/CP2 training and TP4/CP1
rollout topology, temperature 0.7, top-p 0.95, LR 5e-7, KL 0.01 and response
cap 6912. Both runs enabled rollout-logprob reuse and generated different token
counts, so these numbers are a reference rather than a guarantee for the
current no-reuse path.

| 200-step measurement | Native G10 | Strict G11 |
|---|---:|---:|
| Mean rollout seconds | 81.4696 | 68.0234 |
| Mean training seconds | 14.7771 | 27.6626 |
| Mean step seconds | 99.2430 | 99.0377 |
| Pooled end-to-end tok/GPU/s | 62.4719 | 59.6966 |

The mean step difference was 0.2069% in favor of G11 because its rollout
savings offset the slower strict training path. A short run cannot establish
that 200-step number; compare native and consistency with the same command,
token workload and warmup policy.

## Current measured path

The latest strict run used the command above with `--steps 3`, on eight
MI300X GPUs, PyTorch 2.12 / ROCm 7.14 and Qwen3-8B. The native reference was
rerun in the same session, with the same prompts, seeds, sampling and training
settings, inherited CPU affinity, and no rollout-logprob reuse. It changes
only `--mode native`. Native and strict generated trajectories differ.

| Three-step measurement | Native reference | Published strict baseline | Current strict |
|---|---:|---:|---:|
| Mean rollout seconds | 41.8292 | 42.0642 | 40.0310 |
| Mean training seconds | 14.0723 | 16.6432 | 16.4480 |
| Mean step seconds | 57.4005 | 60.6998 | 58.2901 |
| Response tokens | 133,687 | 130,890 | 130,890 |
| End-to-end tok/GPU/ms | 0.097043 | 0.089848 | 0.093562 |

Relative to the published strict baseline, rollout improves by 4.83% and
step time by 3.97%. The token multisets match in all three steps, including
their sequence hashes. Separate fixed-workload rollout pairs use four
3072-token prefixes and 256 generated tokens each. QKV scheduling plus the
sequence-reservation change measured **1.8355 -> 1.7707 seconds (-3.53%)**.
Parallel peer polling then measured **1.7871 -> 1.7511 seconds (-2.01%)**;
reversing the run order measured **1.7692 -> 1.7530 seconds (-0.92%)**.
Each candidate matched token IDs and all 5,120 raw logprob-bit comparisons.
These are separate incremental rollout measurements, not additive speedups.

Strict remains 1.55% slower in mean step time than the native reference
(3.57% excluding step 0). Native completed all three steps but lacks the
compiled FFN execution record, so that reference remains provisional. The
historical 0.2% difference is neither reproduced nor guaranteed.

The current strict run passed source sealing and validation. Raw FP32 audit:
**130,890 distinct logprobs / 523,560 including TP replicas,
zero mismatches**. Training recomputes logprobs independently.

### Retained kernel changes

The existing path fuses eager RMSNorm/residual addition, Q/K normalization
with RoPE, and GEMM chunk reduction with SwiGLU. Smaller decode tiles,
guarded QKV/down chunk unrolling and the Attention identity-index shortcut
preserve BF16 rounding and reduction order.

The existing path also changes instruction scheduling for small split Attention
tiles and accelerates the small-batch top-p cumulative scan. The scan computes
independent Sklansky subtrees in parallel, propagates the carry along the
original left spine, then reconstructs each prefix with the original addition
order. Sorting, softmax, top-p thresholding and random sampling retain their
native semantics. There is no nucleus support cap or fixed sampling parameter.

The scan shortcut is guarded to the verified PyTorch 2.12 / gfx942 FP32 path,
batch sizes 2–7 and vocabulary sizes at least 4096. Single-row CUB, large-batch
vLLM Triton and unsupported inputs retain their native implementations. The
README command enables the shortcut automatically; no extra flag is required.
Temperature, top-p and existing training/rollout topology controls are unchanged.

Uninstrumented operator tests measured the scan at 281.29 -> 54.55 microseconds
for batch 4 and vocabulary 152,064, with exact raw bits. Attention-only paired
rollout checks improved 2.04% and 1.72% in opposite run orders. Adding the scan
improved a separate pair by 3.68%; the combined fresh pair above is the retained
end-to-end rollout estimate, rather than adding those percentages together.
The full-model trace confirms both new paths execute. Over the same final 63
decode graph replays, cumulative Attention partial time fell 51.30 -> 45.65 ms
(VGPR count 104 -> 88), and scan time fell 20.20 -> 3.60 ms across its three
kernels. These profiled attribution numbers are separate from wall time.
The new increment schedules the existing four-way-unrolled small QKV kernel
with `memory-bound-attention`. Each IPC rank now advances its own generation
after every peer has completed the previous one; the rank-zero publication
dependency is removed. One lane polls each peer's ready/done signal in parallel.
System-scope acquire/release and the fixed arithmetic reduction tree remain
unchanged. Mixed collectives and graph replay exercise the generation protocol.

The full-model trace shows QKV partial time decreasing 6.39 -> 5.91 microseconds
and mean readiness wait decreasing 9.22 -> 4.58 microseconds after the first
two changes, before parallel peer polling. Profiling is used for attribution;
uninstrumented wall-time pairs determine whether an optimization is retained.
Single-CTA reduction/readiness fusion and multi-CTA readiness fusion were
excluded after implementation review and negative complete-FFN measurements.
Explicit Gluon rewrites, transposed QK and alternate scan-carry loading were
also excluded after correctness or performance checks.

### Validation scope

The combined GEMM/Attention/SwiGLU/scan GPU suite reports **77 passed**. Tests
include changed-input graph replay, raw scan bits, top-p boundary values,
per-row top-p/top-k, temperatures 0.2/0.7/1.0/1.8 and native fallback dispatch.
The CPU collective/build suite reports **45 passed** on Linux. The added
distributed IPC check passes for TP1/2/4/8, FP32/FP16/BF16, mixed all-reduce,
all-gather and reduce-scatter, direct staging, uneven rank progress, and
changing inputs across eager execution and graph replay.

The earlier expanded GPU/integration suite had 130 passed, one skipped and
one existing integer-square-bin kernel-loading failure (HIP status 500), also
reproduced on the unchanged baseline. Counts across suites overlap. This
increment is not a new full topology matrix or 200-step acceptance run.
Raw traces and experiment scripts stay outside the repository.
