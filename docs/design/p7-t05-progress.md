# P7 T05 progress record (Megatron training provider and readback)

**Task:** P7 T05 · **Branch:** `dsv4-p7-dev` · **Status date:** 2026-10-13 ·
**Design:** [p7-t05-megatron-training-provider.md](p7-t05-megatron-training-provider.md)

This records what T05 has implemented and verified so far, what is blocked and on
whom, and what is not done. Status words follow the WS1 definition of done:
**Passed** means positive evidence was obtained and checked; **Blocked** means the
item needs an upstream deliverable or resources T05 does not own; **Not done**
means T05 still has work to do. A negative test that correctly reports missing
evidence proves only that the gate fails closed; it does not complete the item.

WS1 is **not complete**. This push is a progress snapshot.

## Implemented

| Area | Module | What it does |
| --- | --- | --- |
| Read-only call capture | `rl_engine/integrations/megatron_training_readback.py` | Wraps `MegatronLogpOperator.__call__` at class level, records request, result, operator, runtime (RNG, stream) and build identity per call, each field tagged observed / declared / configured / upstream |
| Fail-closed gate | `rl_engine/alignment/cross_config/adapters/megatron_training_gate.py` | Validates one training call (schema, row layout, TP coordinates, vocab partition, weight lock, entry point, staleness, fallback, four sampling layers, LoRA, mask, FP32, target ownership) and keeps a ledger of evidence that is still missing |
| Recorded provider | `.../megatron_training_provider.py` | Replays a recorded M10 input through the real VIME request types and the strict `linear_logp` route |
| Upstream correlation | `.../megatron_training_upstream.py` | Correlates recorded P1–P6 `TraceEnvelope`s with a training call and forwards them verbatim |
| M10 export (draft) | `.../megatron_training_export.py` | `m10.json` plus FP32 active raw bytes, integrity checks, local bit self-test (no verdict authority) |
| Failure batch and weight lock (draft) | `.../megatron_training.py` | Failure-batch manifest and filter, pre-update weight lock, status names |
| Gradient write evidence | `rl_engine/integrations/megatron_training_gradients.py` | Per-step write markers on Megatron DDP (accumulate-hook calls and `grad_added_to_main_grad`), DDP buffer/bucket mapping, base-grad-absent and producer cross-checks |
| Cross-rank gradient evidence (draft) | `rl_engine/integrations/megatron_training_gradients.py` (`observe_sync`, `observe_optimizer_ownership`), `.../megatron_training_gradient_ranks.py` | Records each gradient-sync collective Megatron issues on a DDP bucket inside the step window (pass-through wrappers, no added kernel) and the distributed optimizer's owned ranges; validates rank completeness, replica agreement, one sync per bucket on the DP group with the configured collective, shard offsets and exact ownership coverage |
| Per-rank readback (draft) | `.../megatron_training_ranks.py` | Observed rank coordinates (TP/CP/DP/PP, device), rank readbacks exchanged over CPU gloo, rank-set integrity: missing or duplicate ranks, group membership, invariants, weight replicas, vocab coverage, row and target ownership, global-token mapping |

Schemas are suffixed `-draft` and are replaced by the T01 publications once frozen.
Test doubles for unpublished T01/T02 artifacts live in `tests/p7_doubles/` and are
not a public ABI.

## Verification

Environment: Python 3.11.15, torch 2.9.1+cu128, vLLM 0.16.0, Ray 2.57.0,
Megatron-LM `1dcf0da` (core 0.16.0rc0), VIME tested tree `f1e20b7`
(base `c80200e7` plus the companion `vime.patch`), matching the runtime contract
in [qwen3-vime-consistency.md](../usage/qwen3-vime-consistency.md).
Transformer Engine 2.18 is not installed.

| Gate | Result |
| --- | --- |
| T05 CPU tests (`tests/test_megatron_training_*.py`) | 145 passed, 8 skipped (GPU-only tests) |
| T05 plus the existing strict-route regression files (VIME provider, framework runtime adapters, Megatron runtime state, canonical LM-head backward, canonical CP gradients, cross-config runtime, three canonical `linear_logp` tests; 256 tests) | 247 passed, 8 skipped, 1 failed: the pre-existing `test_vllm_rocm_rotary_reuses_one_table_for_query_and_key` (see Findings) |
| ruff, black, isort, flake8 (pre-commit arguments, line length 100) | passed on all T05 files |
| GPU G1, H20 (SM90), `tests/test_megatron_training_gpu.py` | 4/4 passed: real strict route passes the gate; CPU double provenance keys match `LinearLogpWrapper`; L1 repeat bitwise identical; L2 batch composition bitwise invariant |
| GPU G2, `tests/test_megatron_training_gpu_live.py` | 1/1 passed: tiny Megatron GPT with VIME's LM-head capture hook; live and recorded results byte-identical; observation adds or removes no CUDA kernel |
| GPU G3, same file | 2/2 passed: real Megatron DDP, two micro-batches; LM-head write markers, no write on the frozen embedding; `main_grad` bitwise unchanged under observation |
| GPU G3 fused wgrad | skipped: `fused_weight_gradient_mlp_cuda` (apex/TE) not installed |
| WS2 CPU tests (`tests/test_megatron_training_ranks.py`) | 31/31 passed, including a 4-process gloo exchange |
| WS2 gradient CPU tests (`tests/test_megatron_training_gradient_ranks.py`) | 19 passed, 1 skipped (ownership does not apply to all-reduce); 2-process gloo double of Megatron's bucketed sync |
| WS2 gradients on 4×H20, `tests/megatron_training_grad_ranks_dp.py` | Real Megatron DDP, strict LM-head route, two micro-batches with different data per DP rank: TP1×DP4 and TP2×DP2 with the distributed optimizer, TP1×DP4 all-reduce, TP1×DP4 distributed optimizer with overlapped reduction (4 buckets). Every bucket synchronised once per rank with the configured collective on the DP group, each rank received its own shard, owned ranges cover every trainable parameter exactly once; missing/duplicate rank, swapped collective (`SILENT_FALLBACK`) and a dropped owned range (`INCOMPLETE_ARTIFACT`) fail closed. Test-side: synchronised gradients bitwise unchanged under observation, parameters equal across DP replicas after the optimizer step. Complete, not certified (producer gradient set missing) |
| WS2 on 4×H20, `tests/megatron_training_ranks_tp.py`, canonical TP4 | TP1 (size-1 TP group, as VIME passes it), TP2, TP4 and TP2×CP2 on the real strict TP route: FP32 bytes identical on all 77 active tokens; every rank set complete; dropping a rank reports `MISSING_RANK`, duplicating one reports `IDENTITY_DRIFT`. Rank sets are not certified: the same contract-delta and T01 items are missing |

G1–G3 were single-GPU (TP=1, world size 1) correctness runs. Calling TP1 without
a TP group selects the single-GPU fused entry, whose bytes differ from the
canonical TP entry; the evidence records the entry point, so the two are never
compared as one path.

## WS1 definition-of-done status

| Item | Status | Evidence or dependency |
| --- | --- | --- |
| Correct FP32 selected-token logprob; L1 and L2 batch bitwise | Passed | G1, G2 |
| L2 launch variants on GPU (chunked vs whole call, reuse vs recompute, eager vs CUDA graph) | Not done | Reuse/recompute covered on CPU doubles only |
| L2 canonical virtual-TP grouping on GPU | Not done | Configuration checks covered on CPU |
| Executed GEMM spec bound to the T01 harness fingerprint | Blocked (T01) | Harness and spec not published; ledger reports `gemm.executed_spec`, `gemm.harness_fingerprint` |
| LSE, global max/sum, padded mask, actual canonical split, redistribution collective, merge order observed | Blocked (contract delta) | No readback interface in the strict route; additive delta proposed |
| Evidence bound to the call and tagged by source; configured/declared never certify | Passed | Readback and gate CPU tests, G1/G2 |
| Gradient evidence: autograd `main_grad` path, buffer/bucket mapping, write markers, base-grad-absent | Passed (single GPU) | G3 |
| Gradient evidence: fused wgrad on GPU | Out of WS1 scope | The strict LM-head backward does not use fused wgrad and the reference configuration disables gradient-accumulation fusion; revisit in P4 if fusion is enabled |
| Gradient evidence: producer gradient set forwarded | Blocked (P5) | Format not published; cross-check implemented |
| Gradient evidence: cross-rank reduce-scatter and distributed-optimizer shard ownership | Passed | WS2 gradients on 4×H20 (DP2/DP4, TP2×DP2, overlapped reduction) |
| Sampling layers recorded, identity in consistency mode | Passed | Gate CPU tests, G1/G2 |
| Entry point, temperature, TP coordinates, materialization range checked; unsupported branches fail | Passed | Gate CPU tests |
| Failure-batch filter and pre-update weight lock stable across reruns | Passed (draft interface) | CPU tests; shared interface awaits T06 review |
| P1–P6 TraceEnvelope correlation and forwarding | Passed against the draft envelope; final fields depend on T01 | Upstream CPU tests |
| Negative, read-only and regression tests | Passed | See Verification |
| M10 pairing with T06 on one fixture, T09 byte verdict | Blocked (T06, T09) | Export draft delivered |
| Live pairing with a model forward (`PairedScorer`) | Not done | Moved to P4 |
| T06 review | Not done | |

WS2 (cross physical TP with a shared canonical boundary): per-rank readback and
cross physical TP bytes passed on four GPUs (see Verification); certification
waits on the same contract-delta and T01 items. TP8 and larger are outside the
available machine.

## Findings

- **Strict TP needs `KERNEL_ALIGN_DET_GEMM_SM90=1`.** The strict tensor-parallel
  entry points used by VIME training call the deterministic SM90 GEMM. An extension
  built without the flag fails closed with "built without
  KERNEL_ALIGN_DET_GEMM_SM90=1; refusing naive fallback"
  (`rl_engine/kernels/ops/cuda/matmul/det_gemm.py`). The flag is read in
  `setup.py` and is now documented in
  [qwen3-vime-consistency.md](../usage/qwen3-vime-consistency.md). Build with:

  ```bash
  KERNEL_ALIGN_FORCE_SM90=1 KERNEL_ALIGN_DET_GEMM_SM90=1 \
    pip install --no-build-isolation --no-deps -e .
  ```

- **Pre-existing test without a GPU guard.**
  `tests/test_framework_runtime_adapters.py::test_vllm_rocm_rotary_reuses_one_table_for_query_and_key`
  fails when a CUDA build of torch sees no GPU: `torch.cuda.is_current_stream_capturing()`
  raises `cudaErrorNoDevice`. Reproduced on the unmodified base commit; it passes
  with a GPU visible.
- **Gloo 4-rank test times out on slow storage.**
  `tests/test_linear_logp.py::test_native_tensor_parallel_matches_full_reference_cpu_gloo_4_ranks`
  waits 45 s for four spawned workers; with the environment on a network
  filesystem their imports take about 54 s. The same worker with a longer wait
  passes on all four ranks.
- **Pre-existing collection error.** `tests/test_rocm_aiter_api_contract.py` fails
  to import `_AITER_FWD_REQUIRED_KEYWORDS` from
  `rl_engine.kernels.ops.rocm.attention.flash_attn`. T05 does not modify
  `rl_engine/kernels`.

## Proposed contract deltas (additive)

1. The provider result carries the FP32 LSE produced by the same call (or its
   digest, dtype and shape) and the entry point that produced it.
2. The strict kernels report the actual per-row global max and sum, the effective
   padded-vocab boundary, the canonical split actually used, whether logits were
   redistributed and by which collective, and the merge order.
3. T01 publishes the fixed-K / batch-invariant GEMM harness and its fingerprint;
   the strict LM-head GEMM reports the executed spec.

None of these changes arithmetic or adds kernel launches.

## Next steps

- T05 work still open: GPU L2 launch variants and canonical virtual-TP grouping.
- After T01/T06/T09/P5 publish: swap the draft schemas, wire the M10 pairing and
  the producer gradient set, and close the blocked items.
- P4: live pairing with the training-side `PairedScorer`.
