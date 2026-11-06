# Scaled Token Embedding

## Summary

Gemma looks up the input embedding and multiplies every row by `sqrt(hidden)`. This
operator implements that lookup for the Gemma 4 text model requested in
[WS1 issue #415](https://github.com/RL-Align/RL-Kernel/issues/415) (`vocab 262144`,
`hidden 5376`) on top of the deterministic [token embedding](embedding.md) kernels.

For token id $t$, table $W$ and scale $s$:

$$
y_t = W[t] \cdot s,
\qquad
s = \operatorname{round}_{\text{dtype}(W)}\left(\sqrt{\text{hidden}}\right)
$$

$$
\frac{\partial L}{\partial W[v]} = s \sum_{t:\ \text{id}_t = v} g_t
$$

The scale is rounded to the weight dtype **before** the multiplication, exactly as HF's
`Gemma4TextScaledWordEmbedding` (`embed_scale.to(self.weight.dtype)`) and vLLM's
`normalizer` do. For bf16 and `hidden = 5376` that is `73.5`, not `73.3212`; for fp16 it
is `73.3125`. Multiplying by the unrounded fp32 value does not match either framework.

## Entry Point

```python
import torch
from rl_engine.runtime.registry import kernel_registry

weight = torch.randn(262144, 5376, device="cuda", dtype=torch.bfloat16, requires_grad=True)
token_ids = torch.randint(0, 262144, (2, 16), device="cuda")
op = kernel_registry.get_op("scaled_token_embedding", device=weight.device)
hidden = op(token_ids, weight, 5376**0.5)  # [2, 16, 5376], weight dtype
grad_weight, = torch.autograd.grad(hidden, weight, torch.randn_like(hidden))
```

The public wrappers are `NativeScaledTokenEmbeddingOp` in
`rl_engine.reference.embedding.scaled_embedding` and `TritonScaledTokenEmbeddingOp` in
`rl_engine.backends.shared.triton.embedding.scaled_embedding`. `embed_scale` is a Python
float; the wrappers round it to the weight dtype themselves. The native op also exposes
`forward_fp32`, the WS1 golden path.

## Backends and Dispatch

| Platform | Registry priority | Validation status |
| --- | --- | --- |
| NVIDIA CUDA | Triton → PyTorch native | A100 validated; evidence linked below |
| ROCm | Triton → PyTorch native | Hardware execution pending |
| CPU / MUSA / NPU | PyTorch native | Reference and registry tests |

The registry falls back when a backend cannot be loaded or instantiated. The fallback does
not change results: the native op's forward is bitwise equal to the Triton kernel, and its
backward reuses `_deterministic_embedding_grad_weight`, whose addition order matches the
Triton sorted-segment reducer bit for bit.

## Tensor and Precision Contract

| Value | Shape | Dtype | Behavior |
| --- | --- | --- | --- |
| `token_ids` | Any shape | integer | Cast to int64; values in `[0, vocab)` |
| `weight` | `[vocab, hidden]` | FP16 / BF16 / FP32 | Non-contiguous tables are copied |
| `embed_scale` | scalar | Python float | Rounded to the weight dtype before use |
| Output | `token_ids.shape + (hidden,)` | weight dtype | `cast(fp32(W[t]) * fp32(s))` |
| `grad_weight` | `[vocab, hidden]` | weight dtype | `cast(sum_fp32(fp32(g_t) * fp32(s)))` |

**Forward is bitwise equal to HF and vLLM.** A bf16 row times `73.5` has at most 16
significant bits, so the fp32 product is exact and the only rounding is the final cast,
the same single rounding PyTorch performs for `bf16 * bf16`. The same holds for fp16; for
fp32 both sides perform one fp32 multiplication.

**Backward follows the embedding reducer, not HF autograd.** Rows are stable-sorted by id
and added in fp32 in that order, with `g_t * s` computed in fp32 and no atomics; the
result is cast once. HF rounds `g_t * s` to bf16 before the embedding backward adds it,
so the two can differ by more than a fixed ulp bound when rows for one id cancel. The
documented counterexample (`g = [-0.85546875, 0.94140625, -0.193359375]`, `s = 73.5`)
gives `-7.90625` here and `-8.1875` under HF. Rollout does not run backward, so this does
not affect train/rollout logprob parity.

The Triton path is the plain embedding kernels with an fp32 `scale` argument: the forward
kernel multiplies the gathered row in fp32 before the cast, and the sorted-segment
reducer multiplies each row in fp32 before adding it. The reducer launches with
`enable_fp_fusion=False`, so the fp32 multiply and add round separately. The fused kernel
is then bitwise equal to scaling the rows in PyTorch and running the unscaled reducer,
for every dtype. At `scale = 1.0` a compile-time flag removes the multiply from both
kernels, so the plain embedding is not scaled at all. An fp32 multiply by `1.0` is not a
no-op on the GPU: it replaces NaN payloads with the canonical NaN.

Padding rows are **not** masked in backward (no `padding_idx`): vLLM's and Megatron's
`VocabParallelEmbedding` do not mask either, and the training path for Gemma 4 is
Megatron-based.

## Accuracy and Invariance

Forward correctness is checked with `torch.equal` against the literal HF expression for
fp32, bf16 and fp16, including boundary ids, `hidden = 5376` and a tiny `hidden = 7`.
Backward correctness is checked with `torch.equal` against a Python oracle of the contract
above for all three dtypes.

Batch invariance is bitwise: the same token row gives the same output in batch 1 and
batches of 2, 16 and 300, at a different position, with padding appended, after reshaping
to another `[B, S]`, and on repeated runs. Each output element depends only on its own
row and column, so the launch geometry cannot affect it.

Backward determinism is checked for a fixed flattened order: repeated runs and different
leading shapes of the same flattened sequence give bitwise equal `grad_weight`. Changing
the token order changes the addition order inside a segment; cross-layout and cross-chunk
invariance belong to the canonical wrapper, which this operator does not yet plug into.

`check_operator --check-grad` compares the candidates with the fp32 golden
`forward_fp32` under the `elementwise` rows of
`rl_engine/contracts/profiles/precision/ws1.json`.

## Tests

```bash
python -m pytest tests/ops/embedding/test_scaled_token_embedding.py -v -rs
python -m pytest tests/benchmarks/test_scaled_token_embedding_benchmark.py -v
python -m pytest tests/ops/embedding -q
python -m pytest tests/runtime/test_dispatch.py -q
python tools/validation/operators/check_operator.py --op scaled_token_embedding \
  --candidate triton --device cuda --dtype bf16 --batch 2 --seq 16 --check-grad
```

The suite covers the rounded scale per dtype, bitwise forward parity with HF, the
contract oracle, the cancelling-rows counterexample, non-contiguous tables, batch
invariance, backward repeatability, the fused-versus-prescaled reducer property,
`scale = 1.0` parity with the plain embedding op, native-versus-Triton parity under a pad
flood, registry dispatch and fallback, the accuracy harness, and a real-shape bf16 forward
smoke test (`262144 x 5376`, skipped without ~4 GB of free GPU memory).

## Benchmark

```bash
# Quick execution check before the complete run.
python benchmarks/operators/embedding/benchmark_scaled_token_embedding.py \
  --tokens 1 2048 --dtypes bf16 --warmup 3 --repeat 5 \
  --output-dir ../scaled-embedding-results/smoke

# FP16, BF16 and FP32; 1, 2048 and 32768 tokens; unique, duplicate and pad-flood ids.
python benchmarks/operators/embedding/benchmark_scaled_token_embedding.py \
  --output-dir ../scaled-embedding-results/run-1
```

The benchmark times the eager HF expression `F.embedding(ids, w) * scale` against the
Triton op with the `PerformanceProfiler` accelerator-event timer. Forward is checked
bitwise before timing. Backward is timed with three id patterns, because the sorted-segment
reducer loops over each segment in one program: `unique` (one row per id), `duplicate`
(about 64 rows per id) and `pad_flood` (half of the tokens on one id, which is the
performance cliff). The backward gap to HF autograd is recorded as a diagnostic, not a
gate. Keep generated reports outside the checkout.

## Known Limitations

- The Triton autograd wrapper supports first-order gradients only.
- The forward runs one program per row with `next_power_of_2(hidden)` lanes (`8192` for
  `5376`, so about a third are masked); the backward block is `128` columns. There is no
  autotuning.
- Row offsets are int32, as in the plain embedding: one call is limited to about 400k
  token positions at `hidden = 5376`.
- The deterministic backward walks each id segment serially; a pad token referenced
  thousands of times is slow. The native fallback's intermediate buffer is
  `[unique ids, max repeats, hidden]`, which can exhaust memory under a real-size pad
  flood.
- Token ids must be in `[0, vocab)`; the Triton path reports invalid ids asynchronously.
- The strict-mode semantic descriptor and the canonical logical-row backward are not
  wired yet; ROCm hardware validation is pending.

## Implementation Files

- `rl_engine/reference/embedding/scaled_embedding.py`
- `rl_engine/backends/shared/triton/embedding/scaled_embedding.py` (public Triton op)
- `rl_engine/backends/shared/triton/embedding/embedding.py` (kernels and autograd, `scale`)
- `rl_engine/runtime/registry.py`
- `rl_engine/validation/operators/operator_specs.py`
- `rl_engine/validation/operators/operator_inputs.py`
- `tests/ops/embedding/test_scaled_token_embedding.py`
- `benchmarks/operators/embedding/benchmark_scaled_token_embedding.py`
