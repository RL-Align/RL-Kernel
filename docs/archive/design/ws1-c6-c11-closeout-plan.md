# WS1 C6–C11 Closeout Plan

**Parent:** [#266](https://github.com/RL-Align/RL-Kernel/issues/266)
**Branch:** `feat/ws1-c6-c11-closeout-266` (from `feat/ws1-c1-c5-c8-gtest`)
**Depends on:** C1–C5, C8 already on this branch tip

C6–C11 are the remaining EXIT blockers. This landing does **not** reopen
#145–#154. Full-model GPU execute is intended for H20 (user-run); CPU
schema/topology tests stay in ordinary CI.

## Remaining children

| ID | Issue | What lands |
| --- | --- | --- |
| C6 | #272 | Direct decode vs prefill (attention + selected-logprob), C1 only |
| C7 | #273 | Stateful allocate→write→read→decode + generate-rescore; B2 absent |
| C9 | #275 | Full Qwen3-8B Dense BI assembly, both profiles, one-command fwd+bwd |
| C10 | #276 | #150 2×2 matrix + train/infer on the full model |
| C11 | #277 | Required CUDA + Triton BF16 CI entry + report schema |

## Locks

- Thresholds only from C1 (`tolerance_contract.json`). No `_DECODE_ATOL`.
- CUDA and Triton profiles are independent; missing Triton is red.
- Silent / cross-profile / reference-as-candidate fallback is a hard fail.
- EXIT forbids shrinking layers / hidden / heads / vocab.
- Concat-only `NativeKVCacheAttnOp` is Level A, **not** C7 B1.
- C9 green ≠ EXIT. C10 + C11 + parent A/B are required to close #266.
- C10 `backward_executed` compares real `tensor.grad` for every official
  trainable leaf. Layout cells participate. Candidate vs FP32 gold uses
  the three logprob aggregates. Full-model decode covers C6 short/long/
  varlen/padding/B=1/N.

## Local vs H20

| Surface | Where |
| --- | --- |
| C6/C7 CPU schema + gold path | ordinary CI |
| C6/C7 CUDA/Triton CLI | any CUDA GPU |
| C9 topology / node / profile resolution | ordinary CI (no 16 GB alloc) |
| C9/C10/C11 full-model execute | H20 + pinned Qwen3-8B weights |
