# WS1 local defect log (do not reopen #145–#151)

In-repo record only. Do **not** file GitHub issues from this list unless the
maintainer asks. Hopper re-runs go through the same repro commands; kernel
fixes land as PRs against this log.

## rmsnorm-dweight

**Resolved on 2026-08-13:** adapters accumulate `parameter_vjp_contributions_fp32`
per logical row, so `dweight` is bitwise 0 across chunk / N×B=1 on H20.

- **Ops:** `rms_norm`, `qk_norm`
- **Profiles:** `cuda_bf16`, `triton_cuda_bf16`
- **Judgment:** `gradient_invariance`

## det-gemm-dw

**Resolved on 2026-08-13:** same logical-row FP32 VJP protocol as RMSNorm.
Re-run `check_gradient_invariance.py --op det_gemm` to confirm on the target GPU.

- **Op:** `det_gemm`
- **Profiles:** `cuda_bf16`, `triton_cuda_bf16`
- **Judgment:** `gradient_invariance`

## cuda-logp-no-backward

**Resolved on 2026-08-13:** `FusedLogpGenericOp` now has a row-local FP32
softmax VJP bridge. H20 C4/C8 reports all `dlogits` invariance errors as 0.

- **Op:** `logp`
- **Profile:** `cuda_bf16` (C2 status is `declared`, not `missing_required`)
- **Judgment:** `gradient_accuracy` / `gradient_invariance`
- **Symptom:** `FusedLogpGenericOp` calls `_C.fused_logp` with no `torch.autograd.Function`; no `dlogits`.
- **Repro:**
  ```bash
  python tools/validation/operators/check_gradient_invariance.py --op logp --candidate cuda --backend-profile cuda_bf16
  ```

## triton-attention-left-pad

**Resolved on 2026-08-13:** the Triton kernel rebases a contiguous valid KV
interval to logical columns before both softmax reduction passes. Backward is
the matching Triton VJP (no `NativeAttentionOp`). H20 C8 execute is green.

## Tracked C2 gaps (not new defects)

Triton `embedding`, `lm_head`, and plain `logp` are declared candidates. H20
C8 execute ran them with no fallback.

## Hopper-only cells (not defects)

CUDA `embedding` / `lm_head` / `rope` / `batch_invariant_logp` are declared
`cuda-sm90`. H20 C8 execute (`docs/archive/design/ws1-c8-execute.json`) ran all four
green, including CUDA RoPE C3/C4. On non-Hopper hosts classify-only still
marks them `pending_hopper`.
