# WS1 #266 closeout on Ascend NPU (`ascend_bf16`)

Issue [#266](https://github.com/RL-Align/RL-Kernel/issues/266) is the single
execution and acceptance entry for WS1: single-GPU **model-level** train–inference
consistency for the **full Qwen3-8B Dense** model, judged by C1–C11 (#267–#277).
Its two required profiles are `cuda_bf16` and `triton_cuda_bf16`.

This document covers the Ascend version: a third required profile, **`ascend_bf16`**
(backend family `ascend`), carried through every one of C1–C11 on the same shared
contract and the same harnesses.

## What is and is not claimed

| | |
| --- | --- |
| **In scope** | Single-NPU model-level train–inference consistency for full Qwen3-8B Dense on the in-repo Ascend C (CANN) operator stack, under the same `tolerance_contract.json`, the same #150 matrix, the same #152 KV path, and the same four judgments. |
| **Out of scope** | Multi-NPU (TP/CP/SP/DP) → WS2. vime / real vLLM / real Megatron integration → WS3. FP8, MoE, throughput KPIs. |
| **Not claimed** | Cross-platform bitwise parity with CUDA or Triton. Ascend's vector units have their own reduction order; the guarantee is the same one each platform provides for itself — batch-invariant determinism under the shared contract. This mirrors the Triton-vs-CUDA situation, which #266 already treats as two independent profiles rather than one comparison. |

`ascend_bf16` is **required**, not optional. A missing or unexecuted Ascend cell is
**red**, never N/A and never a fallback to another vendor's kernel — the same rule
#266 applies to a missing Triton candidate.

## Chain node → Ascend kernel

All eleven required C2 chain nodes resolve to the `ascend` candidate:

| Chain node | Op | Ascend kernel |
| --- | --- | --- |
| `embedding` | `AscendEmbeddingOp` | `csrc/ascend/embedding_ascend.asc` |
| `rms_norm` | `RMSNormAscendOp` | `csrc/ascend/rmsnorm_ascend.asc` |
| `det_gemm` | `DetGemmAscendOp` | `csrc/ascend/gemm/det_gemm_ascend.asc` (PR #405) |
| `qk_norm` | `RMSNormAscendOp` | same RMSNorm kernel, per-head |
| `rope` | `RoPEAscendOp` | `csrc/ascend/rope_ascend.asc` |
| `attention` | `DeterministicAttentionAscendOp` | `csrc/ascend/attention/deterministic_attention_ascend.asc` |
| `swiglu` | `SwiGLUAscendOp` | `csrc/ascend/activation.asc` |
| `silu` | `SiLUAscendOp` | `csrc/ascend/activation.asc` — **new in this PR** |
| `lm_head` | `AscendLMHeadOp` | `csrc/ascend/lm_head_ascend.asc` |
| `logprob` | `FusedLogpAscendOp` | `csrc/ascend/fused_logp_ascend.asc` |
| `batch_invariant_logp` | `BatchInvariantLogpAscendOp` | `csrc/ascend/batch_invariant_logp_ascend.asc` |

Two kernel-level gaps had to be closed before the profile could be wired:

- **`silu`.** A required C2 chain node with no Ascend kernel. Added to
  `csrc/ascend/activation.asc` alongside SwiGLU, sharing its tile geometry and its
  FP32 sigmoid sequence, so `silu(x)` is bitwise equal to `swiglu(x, ones)`.
  Substituting SwiGLU-with-a-unit-operand at the dispatch layer was rejected: the
  node would then report SwiGLU provenance, which C1 treats as an undeclared backend.
- **FP32-accumulation GEMM.** The canonical row-fold VJP (the construction that makes
  a shared parameter's gradient depend only on logical row identity, not on batching)
  needs a deterministic **FP32-in** GEMM. The Ascend det_gemm kernel is BF16-in only.
  `det_gemm_rowwise_ascend_fwd_fp32` exposes the existing `lm_head_ascend` kernel —
  which already accepts FP32 and reduces each output element in one fixed per-row
  order — as a general GEMM by passing `Bᵀ`. This is exactly how CUDA builds
  `det_gemm_rowwise_fwd_fp32` from its SM90 lm_head kernel. Casting the VJP down to
  BF16 instead would have kept determinism but broken the contract's FP32-accumulation
  rule and the `gradient_accuracy` judgment.

## C1–C11 disposition

| ID | Issue | Ascend delivery |
| --- | --- | --- |
| **C1** | #267 | `tolerance_contract.json` declares `ascend_bf16 → backend_family "ascend"`; `tolerance.py` requires it in `_validate_policy`. Thresholds, dtype policy, comparison roles and the three aggregates are unchanged — there is no Ascend-private relaxation (`backend_private_tolerance_relaxation` stays `false`). TF32 is "disabled" by construction: Ascend has no TF32 mode, and `disable_tf32("npu")` reports `candidate_tf32_enabled=False`. |
| **C2** | #268 | `ws1_manifest.json` gains the `ascend_bf16` profile with all 11 required nodes `declared`, plus 23 representative cases mirroring the CUDA set one-for-one (same fixtures, same tiers, same shapes), each pinning a real `.asc` entry point. `version` bumps to `ws1-c2-v8` and `fixture_identity_sha256` is regenerated. `workload_id` is deliberately **unchanged**: the logical workload, fixtures and seed are identical, so existing CUDA/Triton evidence stays bound to the same workload. |
| **C3** | #269 | `scripts/check_forward_invariance.py` accepts `--backend-profile ascend_bf16` and runs on the profile's own device. Report provenance carries the NPU name and SoC key. |
| **C4** | #270 | `scripts/check_gradient_invariance.py` likewise; `gradient_adapter_status_matrix` now sweeps all three profiles and every required adapter resolves an `ascend` candidate with no red rows. |
| **C5** | #271 | `elementwise_inventory.py` gains an `ascend_verdict` column. Items whose audit argument is backend-independent (residual add, scale, bias, dtype cast) carry over as `pass`; real kernels (rope, silu, swiglu, mask_fill) are `tracked_red` until C3/C4 have executed on an NPU host. |
| **C6** | #272 | `kv_consistency.assert_decode_prefill_consistent` resolves the device from the profile instead of assuming CUDA; `scripts/check_decode_prefill.py` takes `ascend_bf16`. |
| **C7** | #273 | Same for `assert_stateful_kv_consistent` / `scripts/check_stateful_kv.py`. B2 stays explicitly absent, as on CUDA. |
| **C8** | #274 | `four_judgment_matrix.PROFILES` includes `ascend_bf16`, and `build_classified_matrix(profiles=…)` can be scoped to one host's profiles. `scripts/sweep_ws1_four_judgments.py --profile ascend_bf16 --execute` runs the Ascend grid. |
| **C9** | #275 | `qwen3_dense.py` is device-agnostic: the runtime observation check asserts the profile's own device type, `_family` maps `ascend`, and the canonical backward paths gained Ascend branches (`canonical_ascend_rmsnorm`, the row-fold LM head and linear with `family="ascend"` provenance). |
| **C10** | #276 | `chain_gate.py` and `scripts/ws1_chain_gate.py` run the full #150 matrix + train/infer parity on the NPU. Evidence records `gpu_name` (the NPU) and the SoC as the architecture key. |
| **C11** | #277 | `ci/run_ws1_ascend_ci.sh` is the NPU host entry (build → linkage check → tests → C2 evidence → C3/C4 → C6/C7 → C8 → C10), and `.github/workflows/ws1-chain-npu.yml` runs it on a self-hosted Ascend runner. `ci/run_ws1_chain_gate.sh` is now profile-parameterised through `WS1_PROFILES`. |

## Why a per-host profile split

A machine has a GPU or an NPU, not both. Sweeping all three profiles on one host
would force the absent vendor's cells to red for a reason that is not a defect.
So the C8 sweep, the candidate-evidence script and the chain-gate CI script all take
an explicit profile list, and each vendor's job proves its own profiles. **C11 closes
only when every required profile has gone green on its own hardware** — the split is
in where the work runs, never in what is required.

## Accelerator abstraction

`rl_engine/kernels/gtest/accelerator.py` holds the vendor-dependent facts the gates
need: availability, device resolution, device name, architecture key (`sm90` on CUDA,
the SoC string on Ascend), TF32 policy, seeding, synchronization and cache release.
It fails closed — asking for `ascend_bf16` on a host with no NPU raises
`AcceleratorUnavailable`, and pointing an Ascend profile at `cuda:0` is rejected
before any device probe rather than silently running the wrong kernels.

## Running the gates on an Ascend host

```bash
# Atlas A2 / 910B, CANN + torch_npu installed
export WS1_WEIGHTS_PATH=/path/to/Qwen3-8B            # pinned snapshot
bash ci/run_ws1_ascend_ci.sh                          # everything below, in order
```

Individual gates:

```bash
KERNEL_ALIGN_FORCE_ASCEND=1 pip install -e . --no-build-isolation --no-deps

# Operator tests
pytest -q tests/test_silu_ascend.py tests/test_det_gemm_ascend.py
pytest -q tests/test_ws1_ascend_closeout.py            # CPU-only wiring checks

# C2 runtime candidate evidence
python scripts/ws1_candidate_evidence.py --profile ascend_bf16 --all --check-grad

# C3 / C4
python scripts/check_forward_invariance.py  --op silu --candidate ascend --backend-profile ascend_bf16
python scripts/check_gradient_invariance.py --op silu --candidate ascend --backend-profile ascend_bf16

# C6 / C7
python scripts/check_decode_prefill.py --backend-profile ascend_bf16
python scripts/check_stateful_kv.py    --backend-profile ascend_bf16

# C8
python scripts/sweep_ws1_four_judgments.py --execute --profile ascend_bf16 --json

# C9 / C10
python scripts/ws1_chain_fwd_bwd.py --backend-profile ascend_bf16 --weights-path "$WS1_WEIGHTS_PATH"
WS1_PROFILES=ascend_bf16 bash ci/run_ws1_chain_gate.sh
```

## Status

Everything above is wired and green on the CPU-side checks. The on-device
evidence — C2 runtime provenance, C3/C4, C6/C7, the C8 grid and the C10 full-model
gate — has **not** been collected yet: it needs an Ascend host. Until it is, the
Ascend C5 rows stay `tracked_red` and the C8 Ascend cells stay red, which is the
correct pre-execution state and not a claim of failure.

## See also

- `docs/design/ws1-c2-268-workload-plan.md` — the workload identity this profile reuses
- `docs/design/ws1-c4-270-gradient-plan.md` — the gradient harness contract
- `docs/design/ws1-c6-c11-closeout-plan.md` — the CUDA/Triton closeout plan
- `docs/operators/det-gemm.md`, `docs/operators/activation.md` — the Ascend kernels
