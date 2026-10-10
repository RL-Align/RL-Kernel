# Nemotron Nano router and dispatch

The SM90 CUDA provider implements the single-device `moe_router_dispatch` row
of [RFC #434](https://github.com/RL-Align/RL-Kernel/issues/434): projection,
sigmoid, corrected top-six selection, normalization, token packing and backward.
The ordering/arithmetic contract below is provisional pending maintainer review.
Expert MLP, weighted combine, shared
expert execution and EP communication are outside this operator.

The reference checkpoint is `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`,
revision `bf77c31`, gate forward in `modeling_nemotron_h.py`. With one group,
all 128 experts remain eligible. This proposal is `nemotron-router-sm90-v3`.
The maintainer-owned architecture fingerprint and router section 4.4 remain
pending for canonical acceptance. Implementation was authorized in
[the issue discussion](https://github.com/RL-Align/RL-Kernel/issues/434#issuecomment-5977907553).
Consumers must review the proposed ordering before assuming this ABI.

## Interface

```python
from rl_engine.kernels.registry import KernelRegistry

op = KernelRegistry().get_op("nemotron_router_dispatch", device=x.device)
ids, weights, permutation, offsets, packed = op(x, w, bias)
```

All inputs must be contiguous, finite and on the same NVIDIA SM90 device.
Autocast is not used inside the operator; unsupported geometry/dtypes fail.

| Tensor | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `x` | `[T,2688]` | BF16 or FP32 | Token rows |
| `w` | `[128,2688]` | FP32 | Router weight |
| `bias` | `[128]` | FP32 | Selection-only correction |
| `ids` | `[T,6]` | int64 | Selected experts in ascending ID order |
| `weights` | `[T,6]` | FP32 | Normalized original sigmoid scores times 2.5 |
| `permutation` | `[6*T]` | int64 | Packed position to original flat route slot |
| `offsets` | `[129]` | int64 | Exclusive expert prefix counts |
| `packed` | `[6*T,2688]` | Same as `x` | Unweighted token copies |

`0 <= T <= 65536`; larger counts fail before allocation to bound fixed-width
index products. There is no padding, dropping or capacity truncation. Packing
is expert-major then token-major; each permutation value is `token * 6 + slot`.
Offsets delimit expert segments. Integer block prefix sums give each route a
unique scatter destination without atomics. Absolute positions change with the
batch, so payload invariance is checked after undoing the permutation.

The registry has no CPU/ROCm or non-strict fallback. The hot path checks metadata
only; callers must ensure finite inputs and intermediates. The separate
`route_scores_cuda` debugging interface synchronously checks score values.

## Fixed arithmetic

Projection transposes the current weight to contiguous `[2688,128]` on every
call, without a weight cache. Fixed 32-by-64 tiles compute 21 independent
128-element K segments, each visiting four ordered 32-wide IEEE FP32 dot blocks.
A zero-padded 32-leaf FP32 tree merges them. Two warps and three stages are
fixed across token counts. TF32, autotuning, atomic sums and FP fusion are disabled.
Merge, FP32 sigmoid `1 / (1 + exp(-logit))` and routing are fused.

Ranking uses `score + bias`; exact ties choose the lower expert ID. Six argmax
reductions select distinct experts, then an eight-lane sort orders the six IDs.
Normalization sums the original sigmoid scores in ascending selected-ID order,
adds `1e-20`, divides and multiplies by 2.5. Bias has no gradient.

Routing backward fuses the sigmoid derivative as `(dscore * (1 - score)) * score`.
dX and dW use fixed 64-by-64 tiles with ordered 16-wide IEEE FP32 dot blocks and
four warps. dX has one 128-element segment. dW uses 512-token segments and a
zero-padded next-power-of-two sum tree. A single segment skips the merge; short
single segments run `ceil(K/16)` blocks, while multiple segments run 32 blocks
with masked tails. Empty reductions produce zero.

Payload backward gathers six gradients in selected-slot order in FP32.
Routing and payload dX branches each round to the input dtype before addition.
dW reduces the complete token set in its supplied order; adding independently
computed microbatch gradients is not a bitwise full-batch equivalence claim.
Top-k membership is discrete: gradients flow through the selected scores and
copied payloads, not indices or bias. Higher-order gradients are unqualified.

The target branch's `TritonDetGemmOp.forward_fp32` uses a different projection
reduction order and its backward rejects FP32 router weights through the BF16
tree path. Reusing it would require an agreed arithmetic contract and a qualified
FP32 backward. The specialized implementation is restricted to the geometry above.

## Qualification boundary

On the same architecture, compiler and software stack, an identical token and
upstream gradient must preserve IDs, weights, payload and dX bits across batch
sizes and positions. Payloads are compared after undoing dispatch. dW is
repeatable for an identical ordered complete token set. Microbatch dW addition,
cross-compiler equivalence, ROCm and TP/CP/EP equivalence are unqualified.
Native GEMM/TE/HF providers need not produce identical bits under this proposed
fixed arithmetic. Consumers using other route conventions need an explicit adapter;
the registry entry alone does not provide framework integration.

## Tests

```bash
python -m pytest tests/test_kernel_registry.py tests/nemotron/ -q
```

Tests cover an independent scalar FP64 reference, reference finite-difference
gradcheck away from selection boundaries, CUDA accuracy, ties, empty inputs,
unused-output/frozen-input gradients, raw-bit row/dX invariance, fixed-set dW
repeatability, tails, cancellation, the token-count bound, CUDA Graph weight
replacement and outer autocast. The consumer test applies independent nonlinear
experts and weighted combine to expose packing, weighting and backward errors;
it does not implement or benchmark the Nemotron expert MLP.

Test tolerances are empirical workload thresholds, not universal error bounds
or maintainer-approved thresholds. Near a top-k boundary FP32 rounding can
change selection; each provider's selected branch is qualified against FP64.

Current source SHA256:
`9ec652f622711d303b112787f3405751ddc7a5ed4f0281f74fb172f970605a46`.
On one H20 with Torch 2.9.1+cu130 / Triton 3.5.1, 106 reference, registry,
CUDA and consumer tests pass. All five outputs and dX/dW match the previous
implementation bitwise in 36 direct cases through 65536 tokens. Compute
Sanitizer 2025.3.1.0 passes all four tools on full/tail/capacity workloads.
These records cover selected operator sources/tests, not full-repository GPU CI.

## Performance reproduction

The primary production baseline is FP32 cuBLAS projection plus unchanged
Transformer Engine 2.19 fused sigmoid/top-six and index permutation/autograd.
It preserves native formats, with no timed canonicalization. Cotangents are
aligned outside timing. Expert GEMM, combine, auxiliary loss, optimizer and
communication are excluded from both providers.

```bash
# Install optional TE only in an isolated CUDA environment.
pip install 'transformer_engine[pytorch]==2.19.0'
PYTHONPATH=. python benchmarks/benchmark_nemotron_router_training.py --map-type index --mode eager --dtype bf16 --output /tmp/router-te-bf16.json
# Repeat with --dtype fp32 and --reverse, using new output paths.
```

The TE index implementation does not forward the current stream to its CUB
sort, so the comparison uses default-stream eager timing for both providers.
It includes host-launch gaps. Do not combine CUDA Graph latencies from other
experiments with these eager measurements into a speedup.
TE used its official cu13 wheel and PyTorch binding with optional EP bindings
disabled. The measured process excluded an unused incompatible optional FA4
import; router/permutation code was unchanged. An isolated environment without
FA4 avoids that import conflict.

The current evidence snapshot, supplied separately with the PR as
`nemotron-router-current-h20.json`, contains all 128 full-operator index/eager measurements: eight token counts
(1/16/128/513/1024/4096/8192/32768), both dtypes, random/concentrated routing,
forward/forward+backward and both initial provider orders. Each measurement
retains 20 alternating paired samples of five executions, medians, incremental
peak allocation and FP64 qualification. It binds implementation/benchmark hashes
and preserves regressions. Broader local qualification covered 512 measurements;
only the current conservative full-operator matrix is included here.

Full forward+backward on random inputs, in milliseconds. Latencies use the
normal initial order; speedup ranges cover both initial orders:

| Dtype | Tokens | Native ms | Strict ms | Native / strict |
| --- | --- | --- | --- | --- |
| bf16 | 8192 | 1.235402 | 1.141997 | 1.082-1.083x |
| bf16 | 32768 | 4.603255 | 4.217085 | 1.090-1.092x |
| fp32 | 8192 | 1.310125 | 1.293034 | 1.013-1.014x |
| fp32 | 32768 | 4.878426 | 4.857485 | 1.004-1.005x |

BF16 large-token training has an 8-9% speedup ratio; FP32's small difference is
near parity and does not establish robust acceleration. Small-token regressions
remain in the matrix. No universal speedup, model-quality improvement,
distributed equivalence or full-model throughput result is claimed.
