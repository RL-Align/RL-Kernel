# MLP Up GELU GEMM (`mlp_up_gemm_gelu`)

## Summary

Qwen-Image MMDiT feed-forward up projection fused with the tanh-approximate
GELU, `y = bf16(gelu_tanh(x @ W.T + b))`. It covers the two mathematically
identical `[3072 -> 12288]` GEMMs of the Phase A blocks -- `img_mlp.net.0`
(image stream) and `txt_mlp.net.0` (text stream) -- and its output is the
already dtype-rounded activation that `mlp_down_gemm` (`net.2`) consumes, so the
two rows are the two halves of the same MLP. Claimed row of the Qwen-Image WS1
roadmap (issue #386).

The point of the operator is train-infer consistency: the same kernel runs in
training and rollout, so a logically identical token must produce the same bytes
regardless of how it was batched. That, not cross-backend bit equality, is what
the row guarantees; see [Accuracy](#accuracy).

## Entry Point

```python
from rl_engine.kernels.registry import kernel_registry

op = kernel_registry.get_op("mlp_up_gemm_gelu", device="cuda")
y = op(x, weight, bias=bias)   # [M, K] @ [N, K].T + [N] -> GELU -> [M, N] bf16
```

Direct construction of a specific backend:

```python
from rl_engine.kernels.ops.triton.linear.mlp_up_gemm_gelu import TritonMlpUpGemmGeluOp
from rl_engine.kernels.ops.cuda.linear.mlp_up_gemm_gelu import CudaMlpUpGemmGeluOp
from rl_engine.kernels.ops.pytorch.linear.mlp_up_gemm_gelu import NativeMlpUpGemmGeluOp

y = TritonMlpUpGemmGeluOp()(x, weight, bias=bias)
```

## Contract

Two contracts, one per implementation, because the two CUDA paths make different
arithmetic promises and the choice between them is a real trade:

* **`mlp-up-gemm-gelu-tree` -- the portable contract**, implemented by the
  portable fp32 tree kernel: an FP32 32-wide-leaf mid-split tree over the K
  reduction (exact fp32 FMA per leaf), the bias added once in FP32 after the
  complete tree, the frozen fp32 GELU applied to that pre-activation, and exactly
  one BF16 cast at the store. This is the order the independent FP32 CPU
  reference computes, so this path's **`pre` is byte-equal to that reference**.
  A tree cannot use tensor cores, so it runs on FP32 CUDA cores, far slower than
  the tensor-core paths.
* **`mlp-up-gemm-gelu-mma` -- the hardware order**, implemented by the Hopper
  TMA+wgmma path and by Triton: a pinned sequence of tensor-core steps --
  ascending k-chunks of 16, one FP32 accumulator chained in place, no split-K, no
  atomics -- with the bias once in FP32 after the complete reduction, the same
  frozen GELU on that pre-activation and exactly one BF16 cast at the store. Its
  `pre` is a *different association* of the same sum, so it is **not**
  byte-equal to the fp32 CPU reference: measured at K = 3072 the two orders
  agree bit for bit on ~1.2% of elements and differ by at most one fp32 ulp
  elsewhere. The two implementations of this contract (Hopper TMA+wgmma and
  Triton) *are* byte-identical to each other. The output **`y`** is a
  declared-tolerance comparison rather than byte-equal, because of the `tanhf`
  below.

Under either contract the K loop is the operator's *entire* reduction -- nothing
is accumulated outside it -- so a logical row's bytes cannot depend on batch
size, batch position, prompt padding, launch geometry or tiling. The Hopper path
and Triton are bit-identical to each other because they execute the same pinned
k-chunk chain, and both CUDA paths share the backward's gate and `db` kernels.

## Exact Math and the Frozen Sequences

Forward, with `x:[M, 3072]`, `W:[12288, 3072]`, `b:[12288]` (all bf16):

```
pre = tree_gemm(x, Wᵀ) + b                  # fp32; the byte-equality anchor
y   = bf16_rne(gelu_tanh_fp32(pre))         # the only cast in the forward
```

`tree_gemm` is the frozen reduction tree, identical in definition to
`mlp-down-gemm-tree`: the reduction is split into leaves of 32 in leaf space,
each leaf is an ascending-k chain of correctly-rounded fp32 FMAs, and the leaves
are merged pairwise by mid-split (`merge(lo, mid) + merge(mid, hi)`). The
reduction length is the only shape input to the tree; the bias is added once in
fp32 after the complete tree. For the forward `K = 3072` is 96 leaves; for the
backward's `dx` the reduction is over `N = 12288`, i.e. 384 leaves.

The GELU is pinned op by op, with fp32 constants
`kGeluC = 0.044715`, `kGeluS = 0.7978845608` (`sqrt(2/pi)`, i.e. PyTorch's
`M_SQRT2 * M_2_SQRTPI * 0.5`, the coefficient its own tanh GELU uses on both CPU
and CUDA -- `M_SQRT1_2` belongs to the erf branch) and
`kGeluK3 = 0.134145`, and every arithmetic step is an explicit rounding
intrinsic (`__fmul_rn`, `__fmaf_rn`, `__fadd_rn`, `__fsub_rn` in CUDA; the
`mul_rn`/`add_rn`/`fma` libdevice equivalents in Triton):

```
# forward
q  = C * x
t  = fma(q, x * x, x)
th = tanhf(S * t)
y  = (0.5 * x) * (1 + th)

# backward (d = d(gelu)/dx)
x2 = x * x
q  = C * x
t  = fma(q, x2, x)
th = tanhf(S * t)
a  = 1 + th
b  = 1 - th * th
e  = fma(x2, K3, 1)
h  = ((S * x) * b) * e
d  = fma(0.5, a, 0.5 * h)
```

`tanhf` is the **only** transcendental in the row, and the device's `tanhf` is
within its documented 2 ulp of the correctly-rounded fp32 `tanh` the fp32 CPU
reference uses, so `y` and the backward gate are compared under a declared
tolerance. Under the **tree** contract everything around it -- the tree, the
pre-activation, the polynomial, the bias, the single cast, `dx`/`dW`/`db` fed the
same gate -- is byte-equal to the reference by construction; under the **mma**
contract the polynomial and the constants are the same sequence but the GEMM
association differs, so `pre`, the gate and the contractions are
declared-tolerance comparisons (Hopper and Triton remain byte-identical to each
other). No TF32, no split-K, no atomics, no fast math, no compiler-dependent
reassociation.

Backward, given `pre` (fp32) and `grad_y` (bf16):

```
gate = bf16_rne( fp32(grad_y) * gelu_tanh_grad_fp32(pre) )  # activation grad at the dtype boundary
dx   = tree_gemm(gate, W)     # [M, 3072],   reduction over N = 12288
dW   = left_fold(gate, x)     # [12288, 3072], ascending-row fp32 left fold, one fma per row
db   = left_fold(gate)        # [12288],       ascending-row fp32 left fold, one add per row
```

`gate` is the activation's gradient **at the gradient dtype boundary**: the
correctly-rounded fp32 product `grad_y * gelu'(pre)` is cast once to bf16, and
that bf16 tensor feeds all three contractions. The gate (therefore `db` too) and
the `dW`/`db` folds are shared by both CUDA paths and by Triton, exactly like
`db` in the down row, so the three gradients differ across backends only where
`tanhf` moved an element's gate.

Supporting rules:

- **Precision**: BF16 operands, FP32 accumulation, no TF32, no fast math, no
  compiler-dependent reassociation, bias once in FP32, one BF16 cast at the
  store. The build adds `--use_fast_math` only when
  `KERNEL_ALIGN_USE_FAST_MATH=1` is set (not the default).
- BF16 inputs and BF16 output. An fp32 call **fails closed** (no silent fp32
  SGEMM); the PyTorch reference serves fp32 callers.
- Every backend implements `dx`, `dW` and `db`; the CUDA/GELU-only kernels form
  the gate from the saved `pre`.

## The `pre` Save (`emit_pre`) and Its Memory Cost

The backward's gate needs the fp32 pre-activation, so the forward writes it
(`emit_pre`) **only when a gradient can actually be requested** -- i.e. under
`torch.autograd` with a differentiable operand. An inference call
(`torch.no_grad()`, or a frozen model) skips the store entirely and returns an
empty `pre`. The forward *bytes* do not depend on that decision; only whether
the extra fp32 tensor exists.

| `pre` | shape | dtype | when |
| --- | --- | --- | --- |
| written | `[M, N]` | fp32 | forward under grad mode with a differentiable operand |
| empty | `torch.empty({0}, fp32)` | fp32 | inference / no-grad forward |

The cost is one fp32 `[M, 12288]` tensor: at the 4096-token tier that is
`4096 * 12288 * 4 = 192 MiB` on top of the bf16 output (`96 MiB`), and it is the
row's explicit memory trade-off -- the same kernel serves inference at half the
activation footprint. The benchmark reports the inference footprint, the
`+pre` footprint and `pre` alone as separate columns.

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| Triton | `TritonMlpUpGemmGeluOp` | `rl_engine/kernels/ops/triton/linear/mlp_up_gemm_gelu.py` | One portable source for CUDA, ROCm and MUSA: Triton is JIT-compiled per device, so there is no `*_sm90.py` counterpart and no build switch -- the arch-specific instruction is chosen by Triton's own lowering, which is why this file is the ROCm slot. Autotune disabled, tiles pinned, no split-K. On Hopper `tl.dot` lowers to `wgmma.mma_async.m64n256k16`, byte-identical to the hand-written kernel (contract `mlp-up-gemm-gelu-mma`). Portable / ROCm fallback and cross-backend reference. |
| CUDA (Hopper) | `CudaMlpUpGemmGeluOp` | `csrc/cuda/gemm/mlp_up_gemm_gelu_sm90.cu` | TMA 2-D bulk-tensor loads (`CU_TENSOR_MAP_SWIZZLE_128B`, OOB fill zero, `mbarrier.arrive.expect_tx` / `try_wait.parity`) driving `wgmma.mma_async.m64n256k16` for the forward; `dx`/`dW` reuse the down row's `TM=256, TN=128, BK=64` four-warpgroup shape. The epilogue adds the bias in fp32, applies the frozen GELU and keeps the single bf16 cast; the same kernel writes the fp32 `pre` tile when `emit_pre` is set. Compiled only when the extension is built with `KERNEL_ALIGN_FORCE_SM90=1` (the repository-wide SM90 switch) and used only on a compute capability 9.0 device. |
| CUDA (portable) | `CudaMlpUpGemmGeluOp` | `csrc/cuda/gemm/mlp_up_gemm_gelu.cu` | The `mlp-up-gemm-gelu-tree` order on FP32 CUDA cores: a 32-wide-leaf mid-split tree with per-thread partial stacks that merge as the reference's mid-split tree does, bias once in fp32, the frozen GELU and one bf16 cast; **its `pre` is byte-equal to the FP32 CPU reference**. Always compiled; NVIDIA SM80+; the fallback whenever the SM90 build or the device is absent. Also provides the shared gate and `db` kernels. |
| PyTorch | `NativeMlpUpGemmGeluOp` | `rl_engine/kernels/ops/pytorch/linear/mlp_up_gemm_gelu.py` | Independent FP32 CPU reference (`mlp_up_gemm_gelu_reference_pre`/`_forward`/`_gate`/`_backward`): a 32-wide-leaf, mid-split FP32 tree over the same reduction length plus the frozen GELU, deliberately a *different* association order than the mma paths, so agreement there is a declared tolerance. Also the fp32 device fallback. |

## Backend Selection at a Glance

The two CUDA paths live in separate sources, the repository's SM90 convention:
the portable fp32-tree kernel is always compiled from
`csrc/cuda/gemm/mlp_up_gemm_gelu.cu`, and the Hopper path lives in
`csrc/cuda/gemm/mlp_up_gemm_gelu_sm90.cu`, built only when the repository-wide
SM90 switch is on (`KERNEL_ALIGN_FORCE_SM90=1` adds `-gencode=arch=compute_90a`
and `-DKERNEL_ALIGN_WITH_SM90`, exactly as it does for the other `*_sm90.cu`
sources). Both paths can be selected explicitly on the same machine, mirroring
`RL_KERNEL_MLP_DOWN_GEMM_BACKEND`:

```bash
RL_KERNEL_MLP_UP_GEMM_GELU_BACKEND=general  # force the portable fp32-tree kernel (mlp-up-gemm-gelu-tree)
RL_KERNEL_MLP_UP_GEMM_GELU_BACKEND=hopper   # force the Hopper path; refuse if it cannot serve
RL_KERNEL_MLP_UP_GEMM_GELU_BACKEND=auto     # default: Hopper when it can serve, general otherwise
```

`general` is how the two paths are A/B compared on one machine. The two paths
are *different contracts*, so they do not produce the same `pre`/gradient bytes:
the portable tree is byte-equal to the fp32 CPU reference and the Hopper path is
byte-identical to the Triton backend.

| Situation | Path used | Throughput (H100, bf16, K=3072, N=12288) |
| --- | --- | --- |
| Hopper (cc 9.0) and the extension built with `KERNEL_ALIGN_FORCE_SM90=1` | Hopper TMA+wgmma (`mlp-up-gemm-gelu-mma`); Triton is the same contract | forward 627-643 TFLOP/s; fwd+`dx`+`dW`+`db` 541-555 TFLOP/s (4096-6889 tokens) |
| Any NVIDIA GPU cc >= 8.0 without the Hopper build | the portable CUDA tree path (`mlp-up-gemm-gelu-tree`); Triton on the mma contract | tree: forward 14.9-15.1 TFLOP/s, fwd+`dx`+`dW`+`db` 15.9-16.2 TFLOP/s (~43x slower than Hopper -- the fp32-core price of byte equality); Triton as above |
| Below cc 8.0 | the CUDA paths **raise** (no kernel exists); Triton if it supports the target, else the fp32 reference | reference |
| ROCm / MUSA | Triton, else the fp32 reference | not measured here |
| CPU / NPU, or any fp32 call | the fp32 reference | reference |

Every CUDA call prints the route report once (in the shape `det_gemm` uses),
naming the requested backend, the actual path, the fallback state and the
arithmetic contract the call took:

```
[RL-Kernel][route] mode=<mode> module=mlp_up_gemm_gelu requested=<cuda> actual=<hopper|general> \
fallback=<true|false> contract=<mlp-up-gemm-gelu-tree|mlp-up-gemm-gelu-mma>
```

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` | `[M, K]` | bf16 | contiguous; `M` = image tokens at one of the RFC's reference shapes (`4096` for 1024^2, `6889` for 1328^2, `6032` for 1664x928, or `6889` with the 9-token padding), prompt length for the text stream (256 is the schedule's anchor), or batch x seq |
| `weight` | `[N, K]` | bf16 | contiguous, nn.Linear layout |
| `bias` | `[N]` | bf16 | optional |
| `out` | `[M, N]` | bf16 | newly allocated |
| `pre` | `[M, N]` | fp32 | saved only when a gradient can be requested; empty otherwise |

`K = 3072`, `N = 12288` for both MMDiT streams.

Supported geometry: **bf16** operands and output; the CUDA portable path on
**cc >= 8.0** (SM80+); the Hopper TMA+wgmma path only on **cc == 9.0** *and* a
build with `KERNEL_ALIGN_FORCE_SM90=1`, with the TMA constraints -- 2-D
contiguous operands, row strides that are multiples of 8 elements (16 B) and
16 B-aligned bases -- checked explicitly, otherwise the call falls back to (or,
if `hopper` was pinned, refuses in favour of) the portable tree.

## `_C` Symbols

All symbols are exported by the extension's `_C` module and gated by
`csrc/ops.cpp` (owned elsewhere). The two `.cu` sources implement exactly these:

| Symbol | Signature | Notes |
| --- | --- | --- |
| `mlp_up_gemm_gelu_cuda_forward` | `(x: bf16 [M,K], weight: bf16 [N,K], bias: bf16[N]\|None, emit_pre: bool)` -> `(out: bf16 [M,N], pre: fp32 [M,N] \| empty)` | portable fp32 tree (`mlp-up-gemm-gelu-tree`) |
| `mlp_up_gemm_gelu_cuda_gate` | `(grad: bf16 [M,N], pre: fp32 [M,N])` -> `bf16 [M,N]` | elementwise `bf16(f32(grad) * gelu_tanh_grad_fp32(pre))`, no reduction; shared by both CUDA paths |
| `mlp_up_gemm_gelu_cuda_dx` | `(g: bf16 [M,N], weight: bf16 [N,K])` -> `bf16 [M,K]` | portable tree over N |
| `mlp_up_gemm_gelu_cuda_dw` | `(g: bf16 [M,N], x: bf16 [M,K])` -> `bf16 [N,K]` | ascending-row fp32 left fold |
| `mlp_up_gemm_gelu_cuda_db` | `(g: bf16 [M,N])` -> `fp32 [N]` | ascending-row fp32 left fold; **fp32 output** |
| `mlp_up_gemm_gelu_cuda_forward_sm90` | `(x, weight, bias, emit_pre)` -> `(out, pre)` | Hopper TMA+wgmma (`mlp-up-gemm-gelu-mma`), built only with `KERNEL_ALIGN_FORCE_SM90=1` |
| `mlp_up_gemm_gelu_cuda_dx_sm90` | `(g: bf16 [M,N], weight: bf16 [N,K])` -> `bf16 [M,K]` | Hopper path; same contract as Triton |
| `mlp_up_gemm_gelu_cuda_dw_sm90` | `(g: bf16 [M,N], x: bf16 [M,K])` -> `bf16 [N,K]` | Hopper path; same contract as Triton |

The gate and `db` are provided by the portable `.cu` and reused by the Hopper
`dx`/`dW`, exactly as `db` is shared in the down row. `emit_pre=False` must not
allocate or write `pre` and returns an empty tensor; `emit_pre=True` writes the
fp32 pre-activation of the calling path -- byte-equal to the reference's
`pre_fp32` under the tree contract, and the mma order (within an fp32 ulp of it)
under the mma contract.

## Dispatch Behavior

- CUDA: the registry's candidates for the op (the CUDA backend, and the Triton
  backend where it supports the target), then the PyTorch reference. Within the
  CUDA backend, `auto` prefers the Hopper path when the extension has those
  symbols and the device is cc 9.0, and runs the portable tree otherwise;
  `hopper`/`general` override that (see above).
- ROCm / MUSA: Triton, then the PyTorch reference.
- CPU / NPU: the PyTorch reference only.

The CUDA and Triton backends raise on fp32 input and on non-CUDA tensors; the
PyTorch reference covers fp32 and every device without a kernel.

## Accuracy

The declared split is fixed by the arithmetic. Under the **tree** contract the
pre-activation (`pre`), the gate and the contractions `dx`/`dW`/`db` (fed the
same gate) are byte-equal to the independent FP32 CPU reference. Under the
**mma** contract the GEMM is a different association of the same sum, so `pre`
differs from the reference by at most one fp32 ulp elementwise and `dx`/`dW`
inherit that; the two mma implementations (Hopper TMA+wgmma and Triton) are
byte-identical to each other. The GELU value is not byte-equal under either
contract: `tanhf` (<= 2 ulp) is the only transcendental, so `y` and the gate are
declared-tolerance comparisons.

**Against the independent FP32 reference.** Declared bounds (the benchmark gates
on them): at least 99% of output elements bit-identical to the correctly-rounded
BF16 reference -- 96% for `dx`, whose reduction is the longest -- and every
element within 8 BF16 ulps of `max|reference|`. Measured on the H100 at
K = 3072, N = 12288, 32 gate rows, bf16:

| quantity | identical | worst deviation |
| --- | --- | --- |
| `pre` (tree contract) | byte-equal (`torch.equal`) | 0 |
| `pre` (mma contract) | 1.24% | 1 fp32 ulp |
| `y` | 99.81-99.83% | 0.90 bf16-ulp of `max\|ref\|` |
| `dx` (same gate) | 96.65-96.88% | 1.18 bf16-ulp |
| `dW` (same gate) | 99.79-99.81% | 0.99 bf16-ulp |
| `db` (same gate) | 99.75-99.83% | 0.90 bf16-ulp |

The `pre`/`dx`/`dW` figures are the mma contract's association error, not a
precision loss in the epilogue: on the tree contract those same quantities are
byte-equal to the reference, and the two mma implementations (Hopper and Triton)
are byte-equal to *each other* on `pre`, `y`, the gate, `dx`, `dW` and `db`.

**Across backends.** The Triton backend and the Hopper TMA+wgmma path are
bit-identical to each other (`torch.equal` on `pre`, forward, `dx`, `dW`, `db`)
because they execute the same pinned k-chunk chain; the portable fp32-tree path
is a *different* contract and is byte-equal to the fp32 CPU reference instead.
See `tests/test_mlp_up_gemm_gelu.py` and `tests/test_mlp_up_gemm_gelu_triton.py`
for the byte-equality anchors, the tolerance checks and the fp64 oracle.

## Performance Notes

The shipped constants were re-swept for this row's shape before they were frozen
(`K = 3072`, `N = 12288` -- the *transposed* shape relative to the down row, whose
constants were swept for `K = 12288`, `N = 3072`):

- **Hopper (`mlp_up_gemm_gelu_sm90.cu`)**: 71 buildable configurations across
  `FWD_STAGES` 3-6, `FWD_BK` 64/128, `FWD_KG` 1/2, `FWD_GM` 4/8/16, the
  `GeoFwd`/`GeoBwd` tile shapes and TMA L2-promotion policies. The shipped point
  (`Geo<128,256,2,ST=4,BK=64,KG=1,GM=8>` forward, `Geo<256,128,4,ST=4>` backward)
  wins or ties every operation; apparent wins of 3-14% for a `128x256` backward
  tile and for 256 B L2 promotion disappeared under a paired A/B rotation, i.e.
  they were per-run state, not configuration. This host's run-to-run spread is
  5-9%, so single-shot timings cannot resolve a few percent: the sweep used
  alternating bursts and per-pair ratios.
- **Triton**: the pinned tiles were re-searched for this row's shape -- forward
  `BLOCK_M` 64-512, `BLOCK_N` 64-256, `BLOCK_K` 32-128, `num_warps` 4/8,
  `num_stages` 2-4, plus targeted `dx`/`dW` candidates. One constant moved: the
  forward's `num_stages` 3 -> 4, worth **7.9%** on the forward (paired A/B:
  median ratio 0.921 over 30 alternating bursts, the deeper pipeline faster in 29
  of them, p10-p90 0.909-0.943), and 4 is also the largest stage count that fits
  shared memory at `128x256x64` (`num_stages=5` needs 245 760 B of the 232 448 B
  limit). The `dx` and `dW` tiles are unchanged: `dx` with `num_stages=4` at
  `BLOCK_K=128` is out of shared memory and `num_warps=8` is within noise, while
  `dW`'s pinned `num_stages=4` beats 3 by 34% (30/30 pairs). Every alternative
  tile was byte-identical to the pinned one and to the Hopper kernel, i.e. these
  knobs change scheduling only.
- **Every** swept variant was checked for byte identity against the shipped
  build (`pre`, `y`, `dx`, `dW` at two token counts) and the pinned Triton tiles
  against the Hopper kernel: all of them held, which is the evidence that these
  knobs change *scheduling only* and never the accumulation order.

`pre` is written at the memory roofline: 106.5 µs for 339 MB at 6889 tokens,
against a 101.1 µs DRAM roofline at 3.35 TB/s. The remaining distance to the
`cuBLAS` reference on the same shape (667 µs for the raw GEMM vs 907 µs for the
fused forward) is the GELU epilogue run at one CTA per SM, not the tiling.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python benchmarks/benchmark_mlp_up_gemm_gelu.py \
    --backend cuda --dtype bf16 --batch 4096 --seq 1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python benchmarks/benchmark_mlp_up_gemm_gelu.py \
    --backend cuda --dtype bf16 --batch 6032 --seq 1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python benchmarks/benchmark_mlp_up_gemm_gelu.py \
    --backend triton --dtype bf16 --batch 6889 --seq 1
```

The benchmark gates every run against the FP32 CPU reference before reporting
timings, and separates the inference forward (no `pre`) from the training
forward (with the fp32 `pre`) and the end-to-end training step.

Measured on the H100 (median of 5 warmup + 20 iterations), `bf16`,
`K = 3072`, `N = 12288`. `forward` is the inference path (no `pre`);
`forward + dx + dW + db` is one training step. `pre` MB is the fp32 buffer the
training forward materializes (`[tokens, 12288]`, 4 B/element):

| tokens | backend | forward | forward + `dx` + `dW` + `db` | `pre` MB |
| --- | --- | --- | --- | --- |
| 4096 | CUDA Hopper (TMA+wgmma) | 0.491 ms (629.2 TFLOP/s) | 1.682 ms (551.5 TFLOP/s) | 192 |
| 4096 | Triton (wgmma) | 0.545 ms (567.2 TFLOP/s) | 2.021 ms (459.1 TFLOP/s) | 192 |
| 6032 | CUDA Hopper (TMA+wgmma) | 0.727 ms (626.3 TFLOP/s) | 2.609 ms (523.6 TFLOP/s) | 283 |
| 6032 | Triton (wgmma) | 0.814 ms (559.7 TFLOP/s) | 3.072 ms (444.7 TFLOP/s) | 283 |
| 6889 | CUDA Hopper (TMA+wgmma) | 0.810 ms (642.3 TFLOP/s) | 2.806 ms (556.1 TFLOP/s) | 323 |
| 6889 | Triton (wgmma) | 0.915 ms (568.4 TFLOP/s) | 3.467 ms (450.1 TFLOP/s) | 323 |

TFLOP/s counts the forward's two GEMM-equivalents (2·tokens·K·N); the
training-step column counts all six (`forward` = 2·M·K·N and
`dx`/`dW` = 2·M·N·K each).

```text
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. RL_KERNEL_MLP_UP_GEMM_GELU_BACKEND=general \
    python benchmarks/benchmark_mlp_up_gemm_gelu.py --backend cuda --dtype bf16 --batch 6889 --seq 1
```

Measured with `RL_KERNEL_MLP_UP_GEMM_GELU_BACKEND=general` (the portable
`mlp-up-gemm-gelu-tree` order, byte-equal to the fp32 CPU reference). Its
footprint is larger because the tree runs in fp32 and up-converts both operands:

| tokens | backend | forward | forward + `dx` + `dW` + `db` |
| --- | --- | --- | --- |
| 4096 | CUDA portable (fp32 tree) | 20.681 ms (14.95 TFLOP/s) | 58.189 ms (15.94 TFLOP/s) |
| 6032 | CUDA portable (fp32 tree) | 30.493 ms (14.93 TFLOP/s) | 85.405 ms (16.00 TFLOP/s) |
| 6889 | CUDA portable (fp32 tree) | 34.367 ms (15.13 TFLOP/s) | 96.380 ms (16.19 TFLOP/s) |

That is ~43x the Hopper path and the price of the byte-equality contract: a tree
cannot use tensor cores. The row keeps it as the portable/byte-exact order and as
the fallback, exactly as the down row does.

### Kernel-level profile

`torch.profiler` (mean per call over 10 iterations) and
`cuobjdump --dump-resource-usage` on the built extension, at 6889 tokens
(`K = 3072`, `N = 12288`). `cuBLAS` on the same shape is the machine reference:

| kernel | µs | share | TFLOP/s | REG | threads/CTA | resident CTAs/SM |
| --- | --- | --- | --- | --- | --- | --- |
| cuBLAS bf16 `x @ Wᵀ` (reference) | 665 | -- | 752 | -- | -- | -- |
| `sm90` forward (bias + GELU + `pre`) | 753 | 27.8% | 691 | 255 (+16 B stack) | 128 | 1 |
| `sm90` `dx` | 698 | 25.7% | 745 | 90 | 512 | 1 |
| `sm90` `dW` | 892 | 32.9% | 583 | 90 | 512 | 1 |
| `gate` (elementwise, 677 MB) | 219 | 8.1% | ~3.1 TB/s | 40 | 256 | 4+ |
| `db` fold (677 MB read) | 144 | 5.3% | ~4.7 TB/s | 96 | 256 | 2 |

The two GEMM kernels and the forward all run at one CTA per SM: the forward
instantiation needs 255 registers (the fp32 accumulator tile the epilogue also
needs) with a 16-byte spill, and the `dx`/`dW` geometry is 512 threads x 90
registers, which also fits only once. `gate` and `db` are already at HBM
bandwidth and are not worth restructuring. This is why the fused epilogue costs
roughly 0.2 ms over the raw GEMM at this size: it runs with almost no
occupancy to hide its latency.

### The `pre` store trade-off

The training forward writes the fp32 `pre-activation` (323 MB at 6889 tokens)
that the backward's gate consumes. Measured on the same shape: the store costs
**+99 µs** on the forward (0.907 ms -> 1.006 ms of kernel time). Recomputing it
in the backward instead would cost a full extra GEMM (665-907 µs) plus the same
gate work, i.e. ~7-9x the store, so writing it is the cheaper side of the
trade -- and it is only written when a gradient can actually be requested.

### H100 environment row

Environment this row was measured and validated on:

| field | value |
| --- | --- |
| host / GPU | vast.ai container, NVIDIA H100 80GB HBM3 (SXM), compute capability 9.0 |
| driver / CUDA | 550.163.01 / CUDA toolkit 12.8 (`nvcc` 12.8.93) |
| PyTorch / Triton | 2.11.0+cu128 / 3.6.0 |
| Python | 3.12.14 |
| build | `KERNEL_ALIGN_FORCE_SM90=1 MAX_JOBS=48 pip install -e . --no-build-isolation` |
| `RL_KERNEL_MLP_UP_GEMM_GELU_BACKEND` | `auto` (resolves to `hopper` on cc 9.0; `general` forces the portable tree) |
| profiler | `ncu` is blocked on this host (`ERR_NVGPUCTRPERM`: GPU performance counters need host `RestrictProfilingToAdminUsers=0` or `cap_sys_admin` in the container, and this container has neither). The numbers here come from `torch.profiler`, `nsys` (its `QdstrmImporter` has to be run by hand on this image), `cuobjdump --dump-resource-usage` and component isolation; `compute-sanitizer --tool memcheck` reports 0 errors on both contracts. |

Build the Hopper path with the repository-wide SM90 switch, which also builds
the other `*_sm90.cu` sources:

```bash
KERNEL_ALIGN_FORCE_SM90=1 pip install -e . --no-build-isolation
```

Without it `mlp_up_gemm_gelu_sm90.cu` is not compiled, the extension links
without the `_sm90` symbols and the wrapper runs the portable fp32-tree path -- a
different, also frozen, order.

## Tests

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m pytest tests/test_mlp_up_gemm_gelu.py -q
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m pytest tests/test_mlp_up_gemm_gelu_triton.py -q
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/check_operator.py --op mlp_up_gemm_gelu \
    --candidate cuda --device cuda --dtype bf16 --batch 64 --seq 64 --k-dim 3072 --n-dim 12288 --check-grad
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/check_operator.py --op mlp_up_gemm_gelu \
    --candidate triton --device cuda --dtype bf16 --batch 83 --seq 83 --k-dim 3072 --n-dim 12288 --check-grad
```

The suites cover the schedule's structural promise (one-hot probes), determinism,
row and tiling invariance (bitwise), the `pre` byte-equality anchor against
`mlp_up_gemm_gelu_reference_pre` (the whole anchor block, the K/N tails and the
tiling/padding/batch invariances), the `y` tolerance comparison against
`mlp_up_gemm_gelu_reference_forward` (miss fraction and worst ULP, with the
deviation proven to come only from `tanh`), the three gradients compared
**byte-for-byte** when fed the *device's own* gate (isolating the transcendental),
the gate against `mlp_up_gemm_gelu_reference_gate`, the `emit_pre=False`
inference path (no `pre`, same `y` bytes), the fail-closed cases (fp32, cc < 8,
non-bf16/mismatched bias, ROCm, K mismatch, rank errors) and registry/route
dispatch. `pytest tests/test_mlp_up_gemm_gelu.py tests/test_mlp_up_gemm_gelu_triton.py -q`
reports `173 passed in 174.81s` on the H100 (the fp32 CPU reference is bounded to
a 16-row slice per shape and torch is pinned to one intra-op thread around it --
the reference is elementwise fp64 work, where a 128-thread pool costs ~20x what
the arithmetic does).

Shapes are the model's own: `K = 3072` and `N = 12288` everywhere, with the token
count taken from the RFC's reference tiers (`4096` for 1024^2, `6889` for
1328^2, `6032` for 1664x928), the 256-token schedule anchor, or a synthetic
coverage shape (odd tails).

## Known Limitations

- **The build decides the contract.** On one Hopper machine, a build with
  `KERNEL_ALIGN_FORCE_SM90=1` serves `mlp-up-gemm-gelu-mma` through `auto`, while a
  plain build serves `mlp-up-gemm-gelu-tree`; the two are different reduction
  orders, so bytes differ between them. A deployment that must not change bytes
  pins the build *and* `RL_KERNEL_MLP_UP_GEMM_GELU_BACKEND`; the route report
  carries the contract id it actually took.
- **bf16 only.** An fp32 call fails closed on the CUDA and Triton backends; the
  PyTorch reference serves fp32 callers and every non-CUDA device.
- **SM80+** for the portable fp32-tree kernel; the TMA+wgmma path needs a Hopper
  device *and* a build with `KERNEL_ALIGN_FORCE_SM90=1` (it is emitted for
  `compute_90a`, and the device gate is exactly cc 9.0), and Triton's wgmma
  lowering is Hopper-specific as well. Older targets fall back to the same pinned
  order at lower throughput.
- The Hopper entries require contiguous operands (the TMA tensor maps are built
  from the extents) and check that explicitly; the portable fp32-tree path stages
  whatever strides it is given.
- **No split-K**, by contract: a single CTA owns each output tile. Small `M` or
  `N` therefore under-fill the device, and shapes are not padded to tile
  multiples (masks handle the tails).
- The Triton backend's tiles are pinned per contraction rather than autotuned, so
  that a change of `M` cannot change `BLOCK_K`/`num_warps` and with it a row's
  bytes.
- The row's train-infer guarantee is per-device: CUDA and ROCm qualify
  separately, as the RFC requires. The ROCm slot is served by the Triton backend
  and is **unvalidated** until an AMD host runs the row's qualification.
