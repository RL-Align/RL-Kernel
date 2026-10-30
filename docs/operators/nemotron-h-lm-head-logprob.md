# Nemotron-H Embedding + LM-Head Logprob

`nemotron_h_lm_head_logprob` is the `lm_head_logprob` row of the Nemotron-3-Nano-30B-A3B
RFC: the untied input embedding lookup and the LM head (2688 -> 131072) fused with the
selected-token logprob, with no softcap and no embedding scaling. Training and rollout call
the same strict boundary and receive bitwise-identical FP32 `logp` / `lse`.

The op is pinned to `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` at revision `bf77c31`
(`rl_engine/models/nemotron_h/lm_head_contract.py`). Contract version:
`nemotron-h-lm-head-logp-sm90-contract-v1`.

## Entry Point

```python
from rl_engine.runtime.registry import kernel_registry

op = kernel_registry.get_op("nemotron_h_lm_head_logprob")  # SM90 only, no fallback

x = op.embed(token_ids, embed_weight)                        # [..., 2688] bf16
logp, lse = op.logprob(hidden, lm_head_weight, target_ids)   # training entry, differentiable
logp, lse, logits = op.logprob_with_logits(                  # rollout entry, inference only
    hidden, lm_head_weight, target_ids, temperature=temperature
)
op.provenance()                                              # contract + backend ids
```

## Backends

| Backend | Wrapper | Native symbols | Status |
| --- | --- | --- | --- |
| CUDA SM90 (H100/H200) | `NemotronHLMHeadLogprobSM90Op` | `_C.fused_linear_logp_sm90`, `_C.fused_linear_logp_sm90_strict_dlogits`, `_C.embedding_sm90_forward`, strict det-GEMM | Strict forward + backward. |
| ROCm / CPU / MUSA / NPU | N/A | N/A | Not implemented. `get_op` raises; there is no fallback. |

The registry lists the op on every platform with no candidates, so `get_op` fails closed
instead of silently picking a generic logp backend. Hopper builds that expose both strict
symbols prepend the SM90 op.

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `token_ids` | any | int32/int64 | `embed` only. |
| `embed_weight` | `[131072, 2688]` | bf16 | Contiguous; must not share storage with `lm_head_weight`. |
| `hidden` | `[..., 2688]` | bf16 | Output of `norm_f`. |
| `lm_head_weight` | `[131072, 2688]` | bf16 | Contiguous, HF `[out, in]` layout, no bias. |
| `target_ids` | `hidden.shape[:-1]` | int32/int64 | In `[0, 131072)`. |
| `temperature` | scalar or one per token | fp32 | Optional, strictly positive. |
| `logp`, `lse` | `hidden.shape[:-1]` | fp32 | `lse` is returned but not differentiable. |
| `logits` | `[N, 131072]` | fp32 | Rollout entry only, unscaled by temperature. |

Any other shape, dtype, layout, bias, softcap, tied weight, or out-of-range target raises
`NemotronHGeometryError`.

## Arithmetic Contract

Forward reuses the frozen `cuda-fused-linear-logp-sm90-contract-v1` kernel: vocab tile 64,
K slab 32, vocab split count 64 (a function of V only), in-tile balanced tree, ascending
cross-tile chains, IEEE `expf`/`logf`, temperature applied to the statistics and the
selected logit, `logp = min(z_t - lse, 0)`. H = 2688 = 84 x 32 and V = 131072 = 2048 x 64
both fit this tiling with no padding.

Backward (new in this contract):

1. `_C.fused_linear_logp_sm90_strict_dlogits` reruns the forward kernel's MMA main loop
   for a `[rows, vocab slice]` tile, so each FP32 logit is the one the forward reduced.
   The epilogue computes `dz = g * (onehot - expf(z / T - lse)) / T` with the forward's
   rounding points and one BF16 cast. No reduction crosses rows or columns.
2. `grad_hidden = dz @ W` and `grad_weight = dz^T @ hidden` run through the strict
   deterministic GEMM (`det_gemm_linear_input_gradient` / `det_gemm_linear_weight_gradient`:
   no split-K, FP32 accumulation, one cast in the GEMM epilogue).
3. When `N * V` exceeds the 2^28-element dlogits budget, `grad_hidden` is computed in row
   chunks (multiples of 256) over the full vocab, and `grad_weight` in vocab chunks
   (multiples of 64) over all tokens. Neither output is ever summed across chunks, so the
   chunk policy is bit-neutral. A test checks that the single-pass and chunked paths match.
4. The SM90 det-GEMM tree covers at most 32768 reduction elements (`BK << TREE_DEPTH`);
   longer reductions would leave the tile path for the scalar fallback. On that backend
   `grad_hidden` is therefore reduced over vocab in four 32768-wide segments, and
   `grad_weight` over tokens in 32768-token segments once `N` exceeds 32768. Segment
   boundaries are fixed absolute offsets, independent of the memory chunks above; the
   BF16 partials are added in ascending order in FP32 and cast once. The cuBLASLt
   no-split-K backend has no such limit and runs each GEMM in one pass.

Consequences:

- `logp`, `lse`, and each `grad_hidden` row depend only on that row. They are bitwise
  invariant to batch size, position, leading shape, and train-vs-rollout entry.
- `grad_weight` reduces over tokens in the det-GEMM's fixed ascending order. It is
  bitwise reproducible for a given token set.
- `embed` is a pure row copy, bitwise equal to `F.embedding`. Its weight gradient uses the
  existing sorted-segment deterministic reduction.

## Accuracy

The correctness oracle is `rl_engine/models/nemotron_h/reference.py`, an independent FP64
evaluation that shares no code with the CUDA path. Forward and gradients are compared with
the WS1 `logprob` rows of `rl_engine/contracts/profiles/precision/ws1.json`
(`forward_accuracy` / `gradient_accuracy`, bf16), except `grad_weight`, which is a GEMM
reduction with O(10) entries and is judged with the `reduction` gradient row: at that
magnitude BF16 output rounding alone (up to 0.049 on the test inputs) nearly exhausts the
absolute-only `logprob` budget of 0.05. Bitwise properties are always asserted
between two strict runs, never against the oracle.

## Tests

```bash
python -m pytest tests/models/nemotron_h tests/runtime/test_dispatch.py -q
python tools/validation/operators/check_operator.py --op nemotron_h_lm_head_logprob \
    --candidate cuda-sm90 --device cuda --dtype bf16 --batch 2 --seq 16 \
    --vocab 131072 --normalized-dim 2688 --check-grad
```

The geometry-gate and dispatch tests run on CPU. The forward and backward suites need a
Hopper GPU and an extension built with `KERNEL_ALIGN_FORCE_SM90=1`. The backward GEMMs
also need a strict det-GEMM backend: build with `KERNEL_ALIGN_DET_GEMM_SM90=1`, or export
`CUBLAS_WORKSPACE_CONFIG=:16:8 CUBLASLT_WORKSPACE_SIZE=1` before the process starts for
cuBLASLt no-split-K. Without either, the backward tests skip and the op raises at backward.

## Benchmark

```bash
python benchmarks/operators/logprob/benchmark_nemotron_h_lm_head_logp.py --json report.json
```

The baseline is the native production path: cuBLAS BF16 `F.linear` -> `.float()` ->
`log_softmax` -> gather (autograd backward), plus `F.embedding`. Shapes: rollout decode
N in {1, 8, 32, 128, 256, 512}, prefill N in {2048, 8192}, training N in
{2048, 4096, 8192, 16384}. The report includes the provenance dict (contract version,
det-GEMM backend id, fallback state).

## Known Limitations

- The forward kernel processes 256 rows per CTA, so small decode batches underuse the
  GPU. Numbers are reported only; tuning is out of scope for this row.
- The backward recomputes logits, once more in total when `N * V` exceeds the budget.
- TP / CP / EP and ROCm are tracked by separate RFC rows.

## Implementation Files

- `rl_engine/models/nemotron_h/lm_head_contract.py`: checkpoint fingerprint and geometry gate
- `rl_engine/models/nemotron_h/reference.py`: FP64 oracle
- `rl_engine/backends/cuda/logprob/nemotron_h_lm_head_logp.py`: strict op and backward
- `csrc/cuda/logprob/fused_linear_logp_sm90.cu`: forward kernel and strict dlogits epilogue
- `rl_engine/runtime/registry.py`, `rl_engine/validation/operators/operator_specs.py`
- `tests/models/nemotron_h/`
- `benchmarks/operators/logprob/benchmark_nemotron_h_lm_head_logp.py`
