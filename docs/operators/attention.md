# Standard Softmax Attention

The attention operator is the reduction core of the Qwen3/Llama transformer block. It is a
**WS1 ground-truth reference** (issue #108): a pure-PyTorch, fp32-accumulating definition of
the "correct answer" that downstream fused CUDA/Triton attention kernels (FlashAttention and
friends) are validated against.

- **`NativeAttentionOp`**: `out = softmax(Q Kᵀ · scale + masks) @ V` — a hand-written naive
  softmax with a **fixed reduction order**, deliberately **not**
  `F.scaled_dot_product_attention` / flash / mem-efficient attention (whose reduction order
  is unspecified and would break the batch-invariance contract).

This op covers **only** the softmax attention. Qwen3's QK-Norm and RoPE are applied *before*
the call (see the chain), so the `q`, `k` passed in are already normalized and rotated.

```text
q --\
k ----softmax(QKᵀ/√d + mask)·V--> out
v --/
```

## Entry Point
```python
from rl_engine.kernels.registry import kernel_registry

attn = kernel_registry.get_op("attention")

# Prefill: Sq == Skv ; Decode: Sq < Skv (one/few new queries against the full cache)
out = attn(q, k, v, causal=True)                          # [B, 32, Sq, 128]
out = attn(q, k, v, causal=True, scale=1.0 / 128 ** 0.5)  # explicit scale
out = attn(q, k, v, causal=False, key_padding_mask=mask)  # mask: [B, Skv] bool, True = keep
```

The op exposes the WS1 dual-path contract:

- `forward(...)` — computes in the input dtype, returns the input dtype (Axis-B accuracy
  candidate / dtype-behavior path).
- `forward_fp32(...)` — upcasts to fp32, accumulates in fp32, returns fp32 (the ground-truth
  golden path). It disables autocast and TF32 so it stays a true fp32 reference regardless of
  the caller's ambient precision context.

> **Not the same as `"attn"`.** `kernel_registry.get_op("attn")` resolves to the production
> SDPA fallback (`PYTORCH_ATTN`); this ground-truth op is registered separately under
> `"attention"` (`PYTORCH_NATIVE_ATTENTION`). The two do not overlap.

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| PyTorch fallback | `NativeAttentionOp` | None | fp32 ground-truth reference; CPU and any GPU. |
| CUDA / ROCm / Triton | — | — | Planned: downstream fused attention kernels validate against this reference. |

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `q` | `[B, Hq, Sq, D]` | float (fp16/bf16/fp32) | Qwen3-8B: `Hq=32`, `D=128`. |
| `k` | `[B, Hkv, Skv, D]` | float | Qwen3-8B: `Hkv=8` (GQA). `Hq` must be divisible by `Hkv`. |
| `v` | `[B, Hkv, Skv, D]` | float | Same head/seq layout as `k`. |
| `causal` | — | bool (kw, default `True`) | Upper-triangular mask at offset `Skv - Sq + 1`. |
| `scale` | — | float or `None` (kw) | `None` → `1/sqrt(D)` = `1/√128`. An explicit value (incl. `0.0`) is used verbatim. |
| `key_padding_mask` | `[B, Skv]` | bool or `None` (kw) | `True` = valid / keep, `False` = padding → that key column set to `-inf`. |
| output | `[B, Hq, Sq, D]` | `forward`: input dtype · `forward_fp32`: float32 | Heads precede seq (`[B, H, S, D]`). |

**GQA** (`Hq=32`, `Hkv=8`, group `g=4`): each KV head is replicated `g` times with
`repeat_interleave(g, dim=1)` (not `repeat`), so query head `h` maps to KV head `h // g`.

**Causal offset** `Skv - Sq + 1` anchors the queries to the end of the sequence, so a single
expression is correct for both prefill (`Sq == Skv`) and decode (`Sq < Skv`, one query sees
the whole cache).

Pure function — no randomness, no in-place mutation; device and original dtype follow the
inputs. Masks are built on the inputs' device.

## Dispatch Behavior

`kernel_registry.get_op("attention")` resolves through the `OpBackend` priority map. On
`cuda` / `rocm` / `cpu` the only registered backend today is the PyTorch native op
(`PYTORCH_NATIVE_ATTENTION`), so every device dispatches to this op. Calling it (`__call__` ->
`forward(...)`) computes in the input dtype; `forward_fp32(...)` is the explicit fp32 golden
path. When fused attention kernels land, they are prepended to the priority list and the native
op becomes the fallback. The production `"attn"` op_type (SDPA-based `PYTORCH_ATTN`, FlashAttention, etc.) is
a separate dispatch chain and is unaffected.

## Accuracy

Reference semantics (`forward_fp32`, fp32 accumulation, TF32/autocast disabled):

```python
qf, kf, vf = q.float(), k.float(), v.float()
if Hkv != Hq:                                # GQA: replicate KV, 32 Q / 8 KV, r = 4
    r = Hq // Hkv
    kf = kf.repeat_interleave(r, dim=1)
    vf = vf.repeat_interleave(r, dim=1)
scale = scale if scale is not None else 1.0 / math.sqrt(D)   # D=128 → 1/√128
scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale      # [B, Hq, Sq, Skv]
if causal:                                   # offset covers prefill + decode
    m = torch.triu(torch.ones(Sq, Skv, dtype=torch.bool), diagonal=Skv - Sq + 1)
    scores = scores.masked_fill(m, float("-inf"))
if key_padding_mask is not None:             # True = keep ; False columns → -inf
    scores = scores.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))
probs = torch.softmax(scores, dim=-1)        # subtracts per-row max internally
out = torch.matmul(probs, vf)                # [B, Hq, Sq, D]
```

- **Ground truth**: `forward_fp32` always accumulates in and returns fp32, with TF32 and
  autocast disabled so it is not silently downcast by the caller's ambient context.
- **Dtype path**: `forward` runs the same math in the input dtype, so low-precision reductions
  over the key dimension drift from the fp32 reference — Axis-B accuracy therefore uses a
  tolerance, not bitwise equality.
- **Axis A — batch invariance**: each query row reduces over the keys independently of how many
  sequences share the batch, so a row's output is bitwise-identical (`torch.equal`, `atol=0`)
  across batch slicing **and chunked** (chunked-prefill) configurations — these keep the softmax
  reduction width fixed. `key_padding_mask` is the exception: padding changes the reduction width
  (e.g. Skv=10 vs 6), so the masked result only matches the valid-only result up to a small
  tolerance (`atol=1e-6`), not bitwise, in IEEE 754.
- **Axis B — tolerance**: as a `reduction` op, low-precision tolerance follows the `reduction`
  row of the WS1 numerical contract. Measured drift vs the fp32 golden path (rel-peak):

  | dtype | max_abs / peak | threshold (rel-peak) |
  | --- | --- | --- |
  | bfloat16 | ~0.56 % | 3 % |
  | float16 | ~0.07 % | 0.5 % |

## Performance Notes

Reference operator — no fused kernel or benchmark yet. Downstream fused attention kernels carry
their own benchmarks and are measured against this reference for correctness. At the LARGE
Qwen3-8B load point (`B=8`, `Skv=4096`, `Hq=32`) the fp32 scores tensor alone is ~17 GB and the
naive path peaks at ~3× that, so the LARGE smoke test is GPU-only and skips without enough
memory.

## Tests

```bash
python -m pytest tests/test_attention.py -v
```

Covers: `forward_fp32` vs an independent fp32 reference (bitwise), strict-fp32 under hostile
autocast/TF32, closed-form causal/decode checks, GQA replication and the divisibility guard,
scale defaults, key-padding masking, dtype-path accuracy (Axis-B), output shape, Axis-A batch
invariance (slice + chunked, bitwise; padding is near-equality only, see below), input purity,
gradient flow, registry dispatch, and a
GPU-only LARGE Qwen3-8B real-shape smoke test.

## Implementation Files

- `rl_engine/kernels/ops/pytorch/attention/standard_attn.py`
- `rl_engine/kernels/registry.py`
- `tests/test_attention.py`

## Known Limitations

- PyTorch fallback only; no fused CUDA/Triton backend yet (downstream work).
- `Hq` must be divisible by `Hkv` (raises `ValueError` otherwise).
- The naive path materializes the full `[B, Hq, Sq, Skv]` scores tensor — no query-chunking,
  so the LARGE load point is memory-heavy and GPU-only.
- Covers softmax attention only; QK-Norm and RoPE are applied before the call.
