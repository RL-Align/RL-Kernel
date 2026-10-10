# Pack and Pad

## Summary

`NativePackOp` compacts the active rows of a dense `[B, S, *tail]` tensor into
`[total_active, *tail]`. It is intended for RL losses where only generated,
non-padding tokens contribute: packing hidden states before `lm_head` avoids
materializing vocabulary logits for masked prompt and padding tokens.

The operator also returns `cu_seqlens`, the prefix sum of active tokens for each
batch row. Backward scatters gradients to the selected dense rows and writes
zero to inactive rows.

## Entry Point

```python
from rl_engine.runtime.registry import kernel_registry

pack = kernel_registry.get_op("pack")
packed, cu_seqlens = pack(hidden_states, completion_mask)
```

## Backends

| Platform | Registered implementation | Status |
| --- | --- | --- |
| CPU | `NativePackOp` | Supported |
| CUDA | `NativePackOp` | Supported |
| ROCm | `NativePackOp` | Supported |

The native implementation uses PyTorch's optimized `nonzero`, `index_select`,
and `index_copy_` primitives. A custom Triton gather/scatter was evaluated but
was not registered: on the tested RTX PRO 5000 shapes it remained slower than
the equivalent native operation after an active-row-only launch optimization.
Keeping the native backend avoids shipping a slower duplicate kernel while
preserving the memory saving from packing before the vocabulary projection.

## Tensor Contract

| Value | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` | `[B, S, *tail]` | Any dtype supported by PyTorch indexing | Same device as `mask` |
| `mask` | `[B, S]` | Boolean or numeric | Nonzero values are active |
| `packed` | `[total_active, *tail]` | Same as `x` | Row-major order over `[B, S]` |
| `cu_seqlens` | `[B + 1]` | `torch.int64` | Starts at zero; final value is `total_active` |

The public mask contract is two-dimensional. This keeps `cu_seqlens` unambiguous:
entry `i + 1` is the cumulative number of active sequence positions through
batch row `i`.

`NativePackOp.unpack(packed, mask)` performs the inverse diagnostic scatter,
returning zeros at inactive rows. If `tail_shape` is passed explicitly, it must
equal `packed.shape[1:]`.

## Fast Paths

- All active: uses a contiguous clone and skips the indexed gather.
- None active: returns an empty view and skips gather/scatter work.
- Partial mask: gathers active rows and saves their indices for backward.

Because the packed output shape depends on mask data, `nonzero()` is an
intentional device synchronization point. Callers should pack once and reuse
the result across downstream consumers when possible.

## Accuracy

Packing is an exact row selection with no floating-point arithmetic. Forward
output and backward gradients therefore match the PyTorch boolean-index
reference exactly; no tolerance is required.

## Benchmark

```bash
python benchmarks/operators/packing/benchmark_pack.py --smoke

python benchmarks/operators/packing/benchmark_pack.py \
  --device cuda \
  --dtype bfloat16 \
  --num-prompts 2 \
  --g-sizes 8 \
  --completion-lens 1024 \
  --hidden-dim 4096 \
  --vocab-sizes 32768 \
  --mask-densities 0.1,0.3,1.0
```

The latency baseline is a pack-only lower bound (`x[mask]`); it does not return
`cu_seqlens`. The memory columns compare dense vocabulary projection with
pack-before-projection. Use a production-sized vocabulary when interpreting
memory savings because tiny vocabularies can make the packing intermediates
larger than the logits they avoid.

## Tests

```bash
python -m pytest tests/ops/packing/test_pack.py -q
```

Coverage includes forward and backward correctness, non-boolean masks,
all-active and none-active fast paths, multi-dimensional tails, device and shape
validation, unpack validation, and registry dispatch.

## Related Files

- `rl_engine/reference/packing/pack.py`
- `rl_engine/runtime/registry.py`
- `tests/ops/packing/test_pack.py`
- `benchmarks/operators/packing/benchmark_pack.py`
