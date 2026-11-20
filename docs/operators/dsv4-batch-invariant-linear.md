# DSV4 Batch-Invariant Linear (ROCm, task 9)

## Summary

`y = x @ weightᵀ + bias` whose output rows do not depend on the other rows in the
call, on ROCm: the shared GEMM with a fixed K accumulation order, bias and output
conversion of task 9 of the DSV4 ROCm plan
([#510](https://github.com/RL-Align/RL-Kernel/issues/510)), with fp32 output for
DeepSeek-V4's lm_head. x is `[M, K]` and weight is `[N, K]`.

```text
y[m, n] = OUT(bias[n] + sum_k FP32(x[m, k]) * FP32(weight[n, k]))   k ascending by K tile
```

## Entry Point

```python
from rl_engine.backends.rocm.gemm.triton_gemm import TritonBatchInvariantLinearOp

linear = TritonBatchInvariantLinearOp()  # raises unless the GPU is gfx942
logits = linear(x, weight, out_dtype=torch.float32)  # x [..., K], weight [N, K]
```

DSV4 model code builds the op explicitly (the layer end,
`rl_engine/models/deepseek_v4/layer_end.py`, with `backend="rocm"`).

## Backends

| Backend | Wrapper | Implementation | Status |
| --- | --- | --- | --- |
| ROCm (gfx942) | `TritonBatchInvariantLinearOp` | `rl_engine/backends/rocm/gemm/triton_gemm.py` (`_gemm_a16_w16_kernel`, copied from aiter `326cffa`); backward through det_gemm's fixed-order Triton GEMM in fp32 | MI300X / MI355X acceptance not run |
| CUDA | — | — | Not implemented |
| AITER | — | — | Not used; see **AITER integration boundary** |

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` | `[..., K]` | bf16 | |
| `weight` | `[N, K]` | bf16 | |
| `bias` | `[N]` | `out_dtype` | optional |
| output | `[..., N]` | `out_dtype`: bf16 (default) or fp32 | |

Leading dimensions of `x` are flattened into M. `x`, `weight` and `bias` are read
through their strides, without a copy, and must be on one device.

## Dispatch Behavior

The constructor raises unless the GPU is gfx942 and names the architecture.
`kernel_registry.get_op("batch_invariant_linear", device="cuda")` returns the same op
on ROCm, without fallback; CUDA, MUSA, CPU and NPU have no backend and `get_op` raises.
Inputs outside the tensor contract raise; an empty batch returns an empty output
without a launch. Each backward is recorded with
`record_backward("batch_invariant_linear", ...)`.

## Accuracy

bf16 inputs, fp32 accumulation, one rounding to `out_dtype`. Output and input gradient
match an fp32 PyTorch linear within the WS1 contract tolerance
(`rl_engine/contracts/profiles/precision/ws1.json`, bfloat16 `reduction` cells of
`forward_accuracy` and `gradient_accuracy`). Batch invariance is checked by bytes.

The arithmetic that fixes the bytes:

- K is never split, and the MFMA shape is fixed (`matrix_instr_nonkdim=16`,
  `kpack=1`). Inside a tile there is only `tl.dot`, so an output element is accumulated
  in K order for every M.
- With a bias, the accumulator starts from the bias (converted to fp32) and the K
  products are added after it. vLLM #59952 adds the bias after the K sum; both orders
  are batch invariant, but they round differently.
- `CONFIGS` chooses the tile shape, the grouping of output tiles and the K tile by M.
  On gfx942, K tiles 16 to 256 and every entry of `CONFIGS` give the same bytes.
- M and N are not specialized, so a new M compiles nothing.

## AITER integration boundary

At the pinned AITER revision
[326cffa](https://github.com/ROCm/aiter/tree/326cffa38c9eda16bef19375ba55334e4d3acd59),
`gemm_a16w16` has a split-K path with a reduce kernel that some configs choose by M,
so its bytes can change with M. This PR copies the kernel into RL-Kernel without the
split-K path, the activations, `skip_reduce` and the persistent and gluon paths, and
does not modify AITER.

## Performance Notes

Not measured on MI300X or MI355X. `CONFIGS` is tuned for the DeepSeek-V4-Flash lm_head
on gfx942. The benchmark times the lm_head forward (x `[M, 4096]` bf16 times weight
`[129280, 4096]` bf16, fp32 logits) for M = 1 to 8192 against AITER's `gemm_a16w16`
with its default config, and reports the error of both against an fp32 linear at
M = 1024:

```bash
PYTHONPATH=<aiter checkout> AITER_TRITON_ONLY=1 \
    python benchmarks/operators/gemm/benchmark_batch_invariant_linear.py --output linear.md
```

## Tests

```bash
# ROCm GPU
python -m pytest -q tests/backends/rocm/test_rocm_batch_invariant_linear.py \
    tests/backends/rocm/test_rocm_dynamic_shape.py
# CPU: the constructor's architecture check with a faked GPU
python -m pytest -q tests/runtime/test_dispatch.py
```

The GPU suite covers the mHC projection shapes, the WS1 manifest's GEMM shapes, a
shape whose N and K are not tile multiples and the lm_head shape, with bf16 and fp32
output, with and without bias: a row alone and in batches of M = 1 to 16384, chunking,
reordering, padding, strides and repeated runs; every entry of `CONFIGS` bit for bit;
the fp32 reference and the input gradient; empty batches; unsupported inputs; and that
unseen M values compile nothing after `warmup`. Skipped GPU tests do not establish
correctness.

## Known Limitations

- gfx942 only. `CONFIGS` and its M ranges are tuned for the lm_head shape; other
  shapes use the same ranges.
- One GPU, no tensor parallelism.
- The launch config chosen for a call is not recorded at run time.
- `dbias` is a PyTorch `sum` over rows, whose order can depend on M.
