# Fused Linear LogP

Fused Linear LogP computes per-token selected log-probabilities directly from
hidden states and the LM-head weight `log_softmax(hidden @ Wᵀ + b)[target]`
without ever materializing the `[N, V]` logits. It targets large-vocabulary RL
post-training, where the `[B, S, V]` logits activation (and its gradient) dominate
memory. The forward streams the vocab in blocks through an online softmax; the
backward recomputes the logit tiles instead of storing them, trading compute for
memory. It is differentiable w.r.t. `hidden`, `lm_head_weight`, and `bias`.

This differs from [Fused LogP](fused-logp.md), which takes already-materialized
logits as input. Here the LM-head projection is fused into the reduction, so the
`[N, V]` tensor never lands in HBM.

## Entry Point

```python
from rl_engine.kernels.registry import kernel_registry

linear_logp = kernel_registry.get_op("linear_logp")

logp = linear_logp(
    hidden,         # [B, S, D] or [N, D]  (differentiable)
    lm_head_weight, # [V, D]               (differentiable)
    target_ids,     # [B, S] or [N]        int, in [0, V)
    bias=None,      # [V] optional         (differentiable)
)                   # -> [B, S] or [N], float32

logp.sum().backward()  # gradients flow into hidden, lm_head_weight, bias
```

## Backends

| Backend | Wrapper | Status |
| --- | --- | --- |
| CUDA SM90 (Hopper) | `FusedLinearLogpSM90Op` | TMA-streamed, Double Buffering, tensor-core forward (`mma.sync.m16n8k16`), online softmax in smem; chunked backward. Compiles for `sm_90a`; validated fp32-accurate on H100. Falls back to Triton/native for fp32/fp16 inputs or hidden dims not divisible by 32. |
| CUDA / ROCm (Triton) | `TritonLinearLogpOp` | Triton online-softmax forward; Liger-style chunked backward (cuBLAS matmuls, deterministic). Phase 1. |
| PyTorch native | `NativeLinearLogpOp` | Naive `F.linear` + `log_softmax` + `gather` reference; CPU / Triton-less fallback. |

The SM90 backend (`csrc/cuda/fused_linear_logp_sm90.cu`) streams hidden/weight
tiles via TMA (`cp.async.bulk.tensor`, mbarrier-completed, double-buffered),
contracts each `[BM, BN]` logit tile with the warp-level tensor-core MMA path
(`ldmatrix` + `mma.sync.m16n8k16`, fp32 accumulation), folds it into a per-row
online softmax in shared memory, and gathers the target logit — never
materializing `[N, V]`. It is **build-guarded**: only compiled when the extension
is built with `KERNEL_ALIGN_FORCE_SM90=1` on an SM90 device (TMA/`sm_90a`), and
the registry only selects it when `cc_major == 9` and the symbol is present. The
forward kernel requires bf16 hidden/weight with `D % 32 == 0`; for any other input
the op transparently falls back to the Triton (else native) backend. The backward
reuses the deterministic chunked path. The native op materializes the full `[N, V]`
logits and is the correctness oracle.

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `hidden` | `[N, D]` / `[B, S, D]` | bf16 / fp16 / fp32 | Differentiable; contiguous. |
| `lm_head_weight` | `[V, D]` | bf16 / fp16 / fp32 | Differentiable; contiguous. |
| `target_ids` | `[N]` / `[B, S]` | int | Token id per position, in `[0, V)`. |
| `bias` | `[V]` | float | Optional; differentiable. |
| Output | `[N]` / `[B, S]` | float32 | `z[target] − logsumexp(z)` per position. |

Gradients flow into `hidden`, `lm_head_weight`, and `bias`; `target_ids` is
integer and non-differentiable.

## Reference Semantics

```python
logits = torch.nn.functional.linear(hidden.float(), weight.float(), bias)  # [N, V]
logp = torch.log_softmax(logits, dim=-1)
out = logp.gather(-1, target_ids.long().unsqueeze(-1)).squeeze(-1)
```

The Triton kernel accumulates the matmul and softmax in float32, so it matches the
float32 reference to `atol≈1e-3`. For bf16/fp16 inputs it matches the **fp32-upcast**
reference (it is more accurate than a bf16 `F.linear`, which rounds the logits).

## Performance

```bash
python benchmarks/benchmark_linear_logp.py
python benchmarks/benchmark_linear_logp.py --configs "4096,2048,32768;4096,2048,131072"
```

Measured on an **NVIDIA H100 80GB** (SM90), bf16, N=4096, D=2048, CUDA 12.8.

**Forward latency (ms) and peak forward VRAM:**

| shape (N×D×V) | native | Triton | **SM90** | SM90 vs Triton | peak fwd VRAM (native → fused) |
| --- | --- | --- | --- | --- | --- |
| 4096×2048×32768 | 1.79 | 6.42 | **3.41** | **1.88×** | 1280 MB → **~0 MB** |
| 4096×2048×50257 | 9.96 | 9.82 | **4.88** | **2.01×** | 1965 MB → **~0 MB** |
| 4096×2048×131072 | 7.28 | 25.56 | **12.88** | **1.98×** | 5120 MB → **~0 MB** |

**Forward + backward latency (ms):**

| shape (N×D×V) | native | Triton | **SM90** |
| --- | --- | --- | --- |
| 4096×2048×32768 | 4.25 | 15.86 | 12.69 |
| 4096×2048×50257 | 23.29 | 47.20 | 42.20 |
| 4096×2048×131072 | 17.05 | 117.62 | 104.97 |

**Memory**: the native path allocates the `[N, V]` logits (forward
peak scales with `V`), while the fused op streams them online — its forward peak is
just the per-CTA shared-memory tiles, **independent of `V`** (≈0 MB of activation),
and the chunked backward only ever holds `chunk·V`. That freed memory is what lets
you grow the batch or the CoT length.

**Latency**: the SM90 forward runs at **~2× the memory-free Triton baseline**
across the vocab range — from TMA double-buffering, register-blocked `mma.sync`
M-tiling (`BM=256`, so the weight matrix is re-read `N/BM` times), and split-V over
the grid. It also beats the materializing native path at moderate vocab (2.0× at
V=50257); at the extremes the native cuBLAS GEMM is still faster in raw ms, but only
by paying the 1.3–5 GB `[N, V]` allocation the fused op avoids. The backward is the
shared deterministic chunked-recompute path (recomputes logit tiles rather than
storing them), so the fused forward+backward already beats Triton's. Closing the
remaining forward gap to native's cuBLAS GEMM (WGMMA, a register-resident softmax
epilogue) and a fully fused CUDA backward are future work.

## Tests

```bash
python -m pytest tests/test_linear_logp.py -v
```

Covers the native reference vs the materialized definition, Triton forward (fp32 and
bf16) vs native, Triton backward vs native autograd (with and without bias), leading-
shape preservation, a large-vocab smoke test, and registry dispatch. Triton tests
skip without CUDA + Triton.

## Implementation Files

- `rl_engine/kernels/ops/triton/loss/linear_logp.py`
- `rl_engine/kernels/ops/pytorch/loss/linear_logp.py`
- `rl_engine/kernels/ops/cuda/loss/linear_logp.py` (SM90 wrapper + chunked backward)
- `csrc/cuda/fused_linear_logp_sm90.cu`, `csrc/ops.cpp`, `setup.py` (SM90 kernel + build)
- `rl_engine/kernels/registry.py`
- `tests/test_linear_logp.py`
- `benchmarks/benchmark_linear_logp.py`
- `docs/design/fused-linear-logp.md`
