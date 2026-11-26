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
from rl_engine.runtime.registry import kernel_registry

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
| CUDA deterministic | `DeterministicAttentionOp` | `_C.deterministic_attention_forward/backward` | Batch-invariant CUDA implementation (issue #147). |
| Ascend NPU deterministic | `DeterministicAttentionAscendOp` | `_C_npu.deterministic_attention_ascend` | Batch-invariant Ascend C implementation (issue #147). |

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

Pure function — no randomness, no in-place mutation; device follows the inputs. `forward(...)`
preserves the input dtype, while `forward_fp32(...)` always returns fp32. Masks are built on
the inputs' device.

## Dispatch Behavior

`kernel_registry.get_op("attention")` resolves through the `OpBackend` priority map. On
`cuda` the priority is:

1. `CUDA_DETERMINISTIC_ATTENTION` — `DeterministicAttentionOp` (batch-invariant, fixed-order).
2. `PYTORCH_NATIVE_ATTENTION` — `NativeAttentionOp` (fallback).

On `npu` the priority is:

1. `ASCEND_DETERMINISTIC_ATTENTION` — `DeterministicAttentionAscendOp` (batch-invariant,
   fixed-order Ascend C forward; bf16/fp16 inputs, head dim 128).
2. `PYTORCH_NATIVE_ATTENTION` — `NativeAttentionOp` (fallback).

The Ascend kernel implements the same algorithm as the Triton reference and the CUDA op:
every `(b, q_head, row)` is processed end-to-end by exactly one AI-core block, streaming the
keys in a fixed 64-key tile order with a two-pass (max, then sum-exp + P·V) reduction. There
is **no split-K** and no second-pass merge of per-split `(m, l, u)` summaries, so the
reduction tree for a row depends only on `Skv`, `D` and the masks — never on batch size or
block assignment. The backward recomputes the native reference forward under autograd
(Triton is unavailable on NPU), matching the Triton op's portable backward.

Calling it (`__call__` -> `forward(...)`) computes in the input dtype; `forward_fp32(...)` is
the explicit fp32 golden path (NativeAttentionOp only). The production `"attn"` op_type
(SDPA-based `PYTORCH_ATTN`, FlashAttention, etc.) is a separate dispatch chain and is unaffected.

### WS2 CP-aware dispatch

WS2 distributed callers use a separate contract-aware entry point,
`kernel_registry.get_attention_op(contract)`. It validates explicit TP/CP ownership, fixed
`(out, lse)` merge semantics, causal or packed-sequence offsets, and decode KV-cache identity
before selecting a backend. Legacy `get_op("attention")` behavior remains unchanged.

Existing WS1 implementations do not yet export attention-domain LSE or implement deterministic
CP merge, so they are declared incompatible with strict WS2 requests instead of being selected as
a silent fallback. See [WS2 CP-aware Attention contract](../contracts/ws2-cp-attention-contract.md).

Split-KV is part of that contract rather than a recorded backend extra. Strict runs allow
`disabled` or a fixed logical KV chunk size, and must export the actual per-CP-owner block
boundaries, FP32 `(out, lse)` merge order, final downcast point, backend, and fallback reason.
Runtime-selected `auto` plans are diagnostic only unless both training and rollout export and
validate the same actual plan.

The rank-aware drift benchmark can emit a CPU smoke artifact or a torchrun-friendly GPU report:

```bash
python benchmarks/distributed/benchmark_ws2_cp_attention_drift.py --smoke --json
python benchmarks/distributed/benchmark_ws2_cp_attention_drift.py --smoke --tp-world-sizes 2 \
  --cp-world-sizes 2 --kv-chunk-sizes none,1 --include-backward \
  --output artifacts/ws2-cp-attention-drift.json
```

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
  tolerance (`atol=2e-6`), not bitwise, in IEEE 754.
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
python -m pytest tests/ops/attention/test_attention.py -v
python -m pytest tests/distributed/cp/test_cp_attention.py -v
```

Covers: `forward_fp32` vs an independent fp32 reference (bitwise), strict-fp32 under hostile
autocast/TF32, closed-form causal/decode checks, GQA replication and the divisibility guard,
scale defaults, key-padding masking, dtype-path accuracy (Axis-B), output shape, Axis-A batch
invariance (slice + chunked, bitwise; padding is near-equality only, see below), input purity,
gradient flow, registry dispatch, and a
GPU-only LARGE Qwen3-8B real-shape smoke test.

## Implementation Files

- `rl_engine/reference/attention/standard_attn.py` — ground-truth reference
- `rl_engine/backends/cuda/attention/deterministic_attn.py` — CUDA deterministic op
- `csrc/cuda/attention/deterministic_attention.cu` — CUDA kernels
- `rl_engine/runtime/registry.py`
- `tests/ops/attention/test_attention.py`
- `tests/ops/attention/test_deterministic_attention_cuda.py`

## Fixed Reduction Order (CUDA Deterministic Backend)

The CUDA `DeterministicAttentionOp` pins reduction order for batch-invariance:

**QK kernel**: D-dimension FP32 accumulation, `d = 0 .. D-1`. Each `scores[b,hq,q,k]` has
exactly one writer thread.

**Softmax + LSE kernel**: Each `(b, hq, q)` row processed by one CTA with fixed 256 threads.
Max and sum-exp use a fixed shared-memory tree reduction (power-of-two stride). No split by
batch size, sequence length, or SM count.

**PV kernel**: K-dimension FP32 accumulation, `k = 0 .. Skv-1`. Each `out[b,hq,q,d]` has
exactly one writer thread.

**Backward dK/dV**: Per `(b, hkv, k, d)` output element, a single thread accumulates over
query heads in group order then query positions:
```text
for local = 0 .. g-1:       # g = Hq / Hkv
  hq = hkv * g + local
  for q = 0 .. Sq-1:
    acc += ...
```
No cross-CTA atomics. No launch-order dependent accumulation.

## Prefill / Decode / KV-cache Shared Contract

All inference modes use **the same standard attention kernels** (only Sq/Skv differ):

- **Prefill**: `Sq == Skv`. Causal mask `key_index <= Skv - Sq + query_index`.
- **Chunked-prefill**: each chunk uses `Sq = chunk_size`, `Skv = past + chunk_size`.
  Same causal offset formula produces identical results to full prefill at matching positions.
- **Decode**: `Sq = 1` (or few), `Skv = full_context`. Same kernel, same offset.
- **KV-cache**: caller does `k_full = cat([k_cache, k_new], dim=2)` then calls this op.
  No separate KV-cache softmax implementation allowed.

Hooks:
- `forward(q, k, v, ...)` — main path (registry, #108 harness). Differentiable.
- `forward_with_lse(q, k, v, ...)` — returns `(out, lse)` for LSE verification, debugging,
  and future KV-cache / training integration.
- `save_forward_state(q, k, v, ...)` captures FP32 `out/lse`, masks, position metadata,
  topology, Split-KV boundaries, and content fingerprints for the exact forward invocation.
- `backward_reference(q, k, v, dout, ..., saved_forward_state=state)` consumes that saved
  state in canonical global KV-block order. It fails closed if Q/K/V, saved tensors, masks,
  offsets, topology, or Split-KV metadata changed, and returns `dq`, `dk`, `dv`, `out`, `lse`,
  and provenance.
- `compare_cp_attention_backward(q, k, v, dout, ...)` — compares CP=1 backward against
  CP/chunked-prefill backward and emits whole-tensor plus per-logical-rank drift stats.

## Tolerance

| Scenario | Comparison | Tolerance |
| --- | --- | --- |
| Same physical shape, varying batch position/size/chunk | bitwise | `batch_invariance` (atol=0, rtol=0) |
| Chunked-prefill on/off at same position | bitwise | `batch_invariance` |
| Prefill tail vs decode slice | bitwise | `batch_invariance` |
| CUDA vs `forward_fp32` output/grad | tolerance | `accuracy.default.attention` |
| Valid-only vs padded (reduction width differs) | near-equal | accuracy tolerance (NOT bitwise) |

## Memory Tradeoff (First Version)

The first version materializes full FP32 `scores [B, Hq, Sq, Skv]` and `P [B, Hq, Sq, Skv]`.
Memory cost: `4 * B * Hq * Sq * Skv` bytes per tensor. For Qwen3-8B at B=8, Sq=Skv=4096,
Hq=32: each tensor is ~17 GB. This is acceptable for correctness verification and moderate
sequence lengths but OOM-prone for long sequences. See `benchmarks/operators/attention/benchmark_deterministic_attention.py`
for measured peak memory at representative shapes.

## Known Limitations

- First version: `D=128` only (Qwen3-8B alignment).
- Supported dtypes: BF16, FP16.
- Full materialization of scores/P limits practical sequence length.
- `Hq` must be divisible by `Hkv` (raises `ValueError` otherwise).
- CUDA KV-cache op wrapper is not in scope (caller does cat + calls this op).
- No FP8, no multi-GPU / sequence-parallel.

## MiniMax-H3 full attention

This implements the CUDA `h3_full_attention` work item in [RFC #420](https://github.com/RL-Align/RL-Kernel/issues/420), with an independent PyTorch reference. It covers the attention core after Q/K normalization and RoPE; it does not close the complete model roadmap.

### Interface and scope

```python
from rl_engine.backends.cuda.attention.h3_full_attention import H3FullAttentionOp

out = H3FullAttentionOp(chunk_rows=128)(q, k, v)
```

Q/K/V must be matching contiguous BF16 `[B,S,56,128]` tensors. They must already have the checkpoint's QK RMSNorm and partial three-axis RoPE applied. Output has the same shape and dtype. `forward_with_lse` additionally returns diagnostic FP32 `[B,S,56]` LSE without gradients.

Only non-causal MHA, single rank, TP1/CP1 and first derivatives are supported. No masks, GQA, KV cache, CP or projections are provided. CUDA requires NVIDIA SM80 or newer and an audited strict extension; unsupported inputs/builds fail closed. H20 is the measured device. ROCm is outside this PR.

`kernel_registry.get_op("h3_full_attention", device="cuda")` selects CUDA. The semantic registry explicitly declares dtype/layout/topology and rejects unsupported requests.

### Arithmetic and workspace

The CUDA path retains #240's ascending-D scalar QK reduction, 256-thread softmax tree and ascending-key PV/dQ reductions. Shared-memory staging reuses converted Q/K and dO/V operands and rows of probabilities/dS. Fused dK/dV loading shares operands while maintaining two separate ascending-query FP32 accumulation chains. Each gradient element has one writer; there are no cross-CTA atomics or split-K partial-sum merges.

Queries are processed in fixed-size chunks, while each query consumes all keys. Forward saves Q/K/V; backward recomputes probabilities using the same QK/softmax arithmetic, skipping unused PV. dK/dV retain FP32 accumulators across chunks, dK is scaled once, and both gradients are cast once at the final boundary. Fixed-chunk probability workspace is `[B,56,min(chunk_rows,S),S]` FP32; backward additionally needs dP/dS and two `[B,56,S,128]` FP32 accumulators. Workspace grows linearly with S at a fixed chunk size; compute remains quadratic.

Strict compilation uses `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`. Fast math and external NVCC prepend/append flags are rejected. An isolated `_H3_C` extension enforces this profile without changing ordinary `_C` source/compiler/linker settings. Native H3 entry points and the wrapper verify build identity. Raw-bit promises apply within a qualified device/build, not across platforms or compilers. Independent mathematical accuracy is a separate tolerance check.

### Build and validation

Use an isolated checkout and environment. The ordinary build intentionally omits the strict extension.

```bash
RL_KERNEL_H3_STRICT=1 RL_KERNEL_REQUIRE_EXT=1 TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=2 \
  python setup.py build_ext --inplace
python -m pytest -q tests/backends/cuda/test_h3_full_attention.py
python tools/validation/operators/check_operator.py --op h3_full_attention \
  --candidate cuda --device cuda --dtype bf16 --batch 2 --seq 129 --check-grad --json
```

Tests cover independent FP32/FP64 output/all-gradient accuracy, strict-native raw bits,
repeat/batch/chunk invariance, LSE, invalid inputs, checkpointing and streams. CUDA tests require the optional strict `_H3_C`;
CPU-only success or skipped GPU tests do not establish GPU qualification.

### Performance reproduction

```bash
python benchmarks/operators/attention/benchmark_deterministic_attention.py h3 \
  --output /path/to/local-evidence/matrix
```

The matrix compares strict native and CUDA chunks 64/128/256 on six shapes.
Output/all-gradient raw-bit checks precede timing; two warmups and six rotating samples
include public-wrapper copies, allocations and forward plus all QKV gradients. JSON records
raw samples, extra peak allocation, environment, backend readback and source/binary hashes.
At fixed chunk size, probability workspace grows linearly with sequence length while compute
remains quadratic. These are operator measurements, without
full-model, multi-GPU or ROCm hardware qualification.
