# MLP Down Bias GEMM (`mlp_down_gemm`)

## Summary

Qwen-Image MMDiT feed-forward down projection, `y = single_cast(x @ W.T + b)`.
It covers the two mathematically identical `[12288 -> 3072]` bias GEMMs of the
Phase A blocks -- `img_mlp.net.2` (image stream) and `txt_mlp.net.2` (text
stream) -- applied to the already dtype-rounded GELU output of
`mlp_up_gemm_gelu`. Claimed row of the Qwen-Image WS1 roadmap (issue #386).

The point of the operator is train-infer consistency: the same kernel runs in
training and rollout, so a logically identical token must produce the same bytes
regardless of how it was batched. That, not cross-backend bit equality, is what
the row guarantees; see [Accuracy](#accuracy).

## Entry Point

```python
from rl_engine.kernels.registry import kernel_registry

op = kernel_registry.get_op("mlp_down_gemm", device="cuda")
out = op(x, weight, bias=bias)          # [M, K] @ [N, K].T + [N] -> [M, N]
```

Direct construction of a specific backend:

```python
from rl_engine.kernels.ops.triton.linear.mlp_down_gemm import TritonMlpDownGemmOp
from rl_engine.kernels.ops.cuda.linear.mlp_down_gemm import CudaMlpDownGemmOp
from rl_engine.kernels.ops.pytorch.linear.mlp_down_gemm import NativeMlpDownGemmOp

out = TritonMlpDownGemmOp()(x, weight, bias=bias)
```

## Contract

Two contracts, one per implementation, because the two CUDA paths make different arithmetic
promises and the choice between them is a real trade:

* **`mlp-down-gemm-tree-v1` -- the portable contract**, implemented by the general CUDA path:
  an FP32 32-wide-leaf mid-split tree over the K reduction (exact fp32 FMA per leaf), bias once
  in FP32 after the complete tree, exactly one BF16 cast at the store. This is the order the
  independent FP32 CPU reference computes, so this path is **byte-equal to that reference**.
  A tree cannot use tensor cores, so it runs on FP32 CUDA cores: ~11 TFLOP/s forward on this
  part against the 448-476 of the hardware-order path.
* **`mlp-down-gemm-mma-v1` -- the hardware order**, implemented by the Hopper TMA+wgmma path
  and by Triton: a pinned sequence of tensor-core steps -- ascending k-chunks of 16, one FP32
  accumulator chained in place, no split-K, no atomics -- with bias once in FP32 after the
  complete reduction and exactly one BF16 cast at the store. On this contract the output is
  within the declared tolerance of the reference rather than byte-equal to it (see below).

Under either contract the K loop is the operator's *entire* reduction -- nothing is accumulated
outside it -- so a logical row's bytes cannot depend on batch size, batch position, prompt
padding, launch geometry or tiling. `db` is the ascending-row FP32 fold under both and is
byte-equal to the reference either way, and the Hopper path and Triton are bit-identical to
each other because they execute the same pinned k-chunk chain.

Supporting rules:

- **Precision**: BF16 operands, FP32 accumulation, no TF32, no fast math, no
  compiler-dependent reassociation, bias once in FP32, one BF16 cast at the store --
  the RFC's precision clause restated as this row's contract. The build adds
  `--use_fast_math` only when `KERNEL_ALIGN_USE_FAST_MATH=1` is set (not the default);
  enabling it was measured to leave every tensor of this row unchanged.

- BF16 inputs and BF16 output. The activations reach the kernel already rounded
  by `mlp_up_gemm_gelu`, and the single output cast is the model's rounding step.
- Unsupported dtype/device/shape **fails closed** (no silent fp32 SGEMM, no
  implicit transpose of a mismatched operand).
- Backward implementations are run through the same kernel: `dx = g @ W`,
  `dW = g.T @ x` and `db` = the ascending-row FP32 left fold of `g`.

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| Triton | `TritonMlpDownGemmOp` | `rl_engine/kernels/ops/triton/linear/mlp_down_gemm.py` | One portable source for CUDA, ROCm and MUSA: Triton is JIT-compiled per device, so there is no `*_sm90.py` counterpart and no build switch -- the arch-specific instruction is chosen by Triton's own lowering, which is why this file is the ROCm slot. CUDA default. Autotune disabled, tiles pinned, no split-K. On Hopper `tl.dot` lowers to `wgmma.mma_async.m64n256k16`; that lowering was measured byte-identical to the hand-written kernel, and is ~2.5x its forward throughput. Portable / ROCm fallback and cross-backend reference. |
| CUDA (Hopper) | `CudaMlpDownGemmOp` | `csrc/cuda/gemm/mlp_down_gemm_sm90.cu` | TMA 2-D bulk-tensor loads (`CU_TENSOR_MAP_SWIZZLE_128B`, OOB fill zero, `mbarrier.arrive.expect_tx` / `try_wait.parity`) per-contraction instantiations: the forward runs `TM=128, TN=256, BK=64` with two warpgroups of `m64n256k16` and a 4-slot `wgmma.wait_group<1>` ring, while `dx`/`dW` keep `TM=256, TN=128, BK=64` with four warpgroups of `m64n128k16`. Compiled only when the extension is built with `KERNEL_ALIGN_FORCE_SM90=1`, the repository-wide SM90 switch, in the same way as the other `*_sm90.cu` sources. |
| CUDA (portable) | `CudaMlpDownGemmOp` | `csrc/cuda/gemm/mlp_down_gemm.cu` | The `mlp-down-gemm-tree-v1` contract on FP32 CUDA cores: 64x64 smem tiles, conflict-free k-major staging, a 4x4 register block and a per-thread partial stack that merges as the reference's mid-split tree does; **byte-equal to the FP32 CPU reference**. Always compiled; NVIDIA SM80+; the fallback whenever the SM90 build or the device is absent. ~11 TFLOP/s forward. |
| PyTorch | `NativeMlpDownGemmOp` | `rl_engine/kernels/ops/pytorch/linear/mlp_down_gemm.py` | Independent FP32 CPU reference: a 32-wide-leaf, mid-split FP32 tree over the same reduction length, deliberately a *different* association order, so agreement is a declared tolerance rather than a copy. Also the fp32 device fallback. |

## Backend selection at a glance

| Situation | Path used | Throughput (H100, bf16, K=12288, N=3072) |
| --- | --- | --- |
| Hopper (cc 9.0) and the extension built with `KERNEL_ALIGN_FORCE_SM90=1` | Triton, then the CUDA TMA+wgmma path | 448/476 forward, 395/384 end to end |
| Any NVIDIA GPU cc >= 8.0 (Ampere, Ada, Hopper, Blackwell) without the Hopper build | Triton, then the portable CUDA tree path | 10.9-11.0 forward, 12.2 end to end (byte-equal to the reference) |
| Below cc 8.0 | the CUDA paths **raise** (no kernel exists); Triton if it supports the target, else the fp32 reference | reference |
| ROCm / MUSA | Triton, else the fp32 reference | not measured here |
| CPU / NPU, or any fp32 call | the fp32 reference | reference |

The two CUDA paths live in separate sources, the repository's SM90 convention: the
portable fp32-tree kernel is always compiled from `csrc/cuda/gemm/mlp_down_gemm.cu`, and the
Hopper path lives in `csrc/cuda/gemm/mlp_down_gemm_sm90.cu`, which is built only when
the repository-wide SM90 switch is on (`KERNEL_ALIGN_FORCE_SM90=1` adds
`-gencode=arch=compute_cc a` and `-DKERNEL_ALIGN_WITH_SM90`, exactly as it does for
`fused_linear_logp_sm90.cu` and its siblings). Both paths can be selected explicitly on the same machine, mirroring
`RL_KERNEL_DET_GEMM_BACKEND`:

```bash
RL_KERNEL_MLP_DOWN_GEMM_BACKEND=general  # force the portable fp32-tree kernel (mlp-down-gemm-tree-v1)
RL_KERNEL_MLP_DOWN_GEMM_BACKEND=hopper   # force the Hopper path; refuse if it cannot serve
RL_KERNEL_MLP_DOWN_GEMM_BACKEND=auto     # default: Hopper when it can serve, general otherwise
```

`general` is how the two paths are A/B compared on one machine (K=12288, N=3072):
on an H100 80GB HBM3 the portable tree runs 21.741 ms forward (14.22 TFLOP/s) and
57.723 ms end to end (16.07) at M=4096, against the Hopper path's 0.420 ms (735.6)
and 1.498 ms (619.2) on the same box. The two paths are *different contracts*, so
they do not produce the same bytes: the portable tree is byte-equal to the fp32 CPU
reference and the Hopper path is byte-identical to the Triton backend, and the
measured 35x end-to-end gap is the price of that exactness. The Hopper file is an
accelerator for cc 9.0 rather than the only implementation: the portable tree needs
no tensor cores and no TMA, which is why it serves every device and operand layout
the row supports.

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` | `[M, K]` | bf16 | contiguous; `M` = image tokens at one of the RFC's reference shapes (`4096` for 1024^2, `6889` for 1328^2, `6032` for 1664x928, or `6889` with the 9-token padding), prompt length for the text stream (256 is the schedule's anchor), or batch x seq |
| `weight` | `[N, K]` | bf16 | contiguous, nn.Linear layout |
| `bias` | `[N]` | bf16 | optional |
| `out` | `[M, N]` | bf16 | newly allocated |

`K = 12288`, `N = 3072` for both MMDiT streams.

Where byte equality holds, by contract. The **portable** path is byte-equal to the independent
FP32 CPU reference for forward, `dx`, `dW` and `db`: measured 0 mismatching bytes over the
256-token anchor in full (786k elements), 16-row slices at each RFC reference shape, the
short-K synthetic shape, the odd tails (`K = 100`, `12287`, `12289`) and a tiling shape. (The
reference is CPU-bound -- the tree costs `O(M*N*K/32)` python steps -- which is why the model
tiers are checked on a slice while the anchor and the synthetic cases run whole; the tree
depends only on `K`, and the path is separately verified row-, batch- and tile-invariant.) The
**hardware-order** paths are byte-identical to *each other* but only within the declared
tolerance of the reference (>= 99% identical, <= 8 bf16 ulps; measured 99.3% and 0.8-0.9 ulp),
which is the price of the tensor cores: a tensor core does not compute a tree. Accuracy on
both contracts is pinned against the exact result with an fp64 oracle, and it holds to within
one bf16 ulp for every element measured.

Measured accumulator law, taken from the device itself rather than from a datasheet. One k16
step behaves as: take the largest exponent over the accumulator and the sixteen exact products,
truncate every term toward zero onto the fixed-point grid `2**(e-26)`, sum those quantised terms
exactly, then truncate the result toward zero to a 24-bit significand. Neither the window nor
the normalisation rounds to nearest, and the window covers the accumulator and all sixteen
products together -- subgrouped alignments, exponent-carried windows, sticky/round-bit variants,
wider internal adders (27..40) and two-stage adds all measure worse. Against the CUDA and
Triton forwards this model matches **6.755e-07** of elements (51 of 75.5M over six seeds at
`M = 4096`; `M = 256` is 2 in 4.7M, i.e. the earlier 100% there was a lucky draw), and it
carries over to `dx` and `dW` (99.99996%).

It is not bit-exact, and no member of that family makes it so: the mismatch rate *saturates*
with `K` (3.4e-07 at `K = 1536` against 6.4e-07 at `K = 12288`, so it is not accumulation
drift), it is scale-invariant across a 1e4 spread of the product scale (so the mechanism lives
in the 24-bit round-toward-zero normalisation, not in the alignment geometry), `A = 27` fixes
all 245 of `A = 26`'s deviations on a single-k16 oracle *and creates 221 new ones* (no uniform
window exists), and a control shows the cause: 0 deviations in 100,526 samples with no
truncating addend against 0.8-8.8% where one addend falls below the window. That is the honest
limit of a CPU reference for this contract: the hardware's handling of the dropped bits is not
reproducible by any documented model, and reproducing it would mean encoding this tensor-core
generation's undocumented low-bit behaviour exactly. **Byte equality against the reference is
therefore provided by the portable contract instead** (`mlp-down-gemm-tree-v1`), and this
contract's accuracy is pinned against an independent fp64 oracle.

## Dispatch Behavior

- CUDA: Triton, then the native CUDA kernel, then the PyTorch reference.
- ROCm / MUSA: Triton, then the PyTorch reference.
- CPU / NPU: the PyTorch reference only.

The CUDA and Triton backends raise on fp32 input and on non-CUDA tensors; the
PyTorch reference covers fp32 and every device without a kernel.

## Accuracy

Two different questions, measured separately.

**Against the independent FP32 reference.** Declared bounds: at least 99% of
output elements bit-identical to the correctly-rounded BF16 reference, and every
element within 8 BF16 ulps of `max|reference|`. Measured on H100 PCIe with
model-like scaled inputs (`benchmarks/benchmark_mlp_down_gemm.py`, 32 gate rows,
`K = 12288`):

| quantity | identical | worst deviation |
| --- | --- | --- |
| forward | 99.26% (`M = 4096`) / 99.31% (`M = 6889`) | 0.88 / 0.85 bf16-ulp |
| `dx` | 99.83% | 0.73 bf16-ulp |
| `dW` | 99.998% | 0.69 bf16-ulp |
| `db` | 100.000% | 0.00 bf16-ulp |

Below `K = 128` the two fixed orders cannot disagree and the output is
bit-identical (`torch.equal`) to the reference. The harness tolerance contract
(`atol = 5e-2`, `rtol = 2e-2` for the forward, `atol = 1e-1` for the gradients)
is looser than these bounds.

**Across backends.** The Triton backend and the Hopper TMA+wgmma path are
bit-identical to each other (`torch.equal` on forward, `dx`, `dW`,
`db`) at every token count from 1 to 6889 that has been measured, including the full
`4096 x 12288 x 3072` model shape -- see `tests/test_mlp_down_gemm_triton.py` and the
row's own oracle check. The portable fp32-tree path is a *different* contract and is
byte-equal to the fp32 CPU reference instead, so it is not byte-identical to those
two; its invariance evidence is the suite's bitwise row/tiling/padding checks and the
same declared tolerance against the reference. That is the split rather than a
requirement: the RFC asks for byte equality where the arithmetic contract
requires it (fixed reduction trees and FP32 accumulators) and otherwise for
*measured* cross-backend equality with an explicit tolerance profile; this page
is that profile. The repository has the same split elsewhere: `det_gemm` checks
candidates against its reference with tolerance while `deterministic_attn` and the
Ascend `embedding` claim cross-backend bit identity.

## Performance Notes

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python benchmarks/benchmark_mlp_down_gemm.py \
    --backend triton --dtype bf16 --batch 4096 --seq 1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python benchmarks/benchmark_mlp_down_gemm.py \
    --backend cuda --dtype bf16 --batch 6889 --seq 1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python benchmarks/benchmark_mlp_down_gemm.py \
    --backend cuda --dtype bf16 --batch 6032 --seq 1
```

The benchmark gates every run against the FP32 CPU reference before reporting
timings. H100 PCIe, bf16, `K = 12288`, `N = 3072`:

Cells give the ms / TFLOP/s of a representative run and, in brackets, the range over
repeat sessions -- Triton's own numbers move by up to 7% between back-to-back sessions,
so single-run gaps below ~3% are not resolvable on this part.

| tokens | backend | forward | forward + `dx` + `dW` + `db` |
| --- | --- | --- | --- |
| 4096 | CUDA Hopper (TMA+wgmma) | 0.685 ms (452) [0.680-0.705, i.e. 440-454] | 2.226 ms (417) [2.226-2.324, i.e. 399-417] |
| 4096 | Triton (wgmma) | 0.667 ms (464) [463-465] | 2.392 ms (388) [2.354-2.523, i.e. 368-394] |
| 6032 | CUDA Hopper (TMA+wgmma) | 1.048 ms (434) | 3.546 ms (385) |
| 6032 | Triton (wgmma) | 1.046 ms (435) | 3.958 ms (345) |
| 6889 | CUDA Hopper (TMA+wgmma) | 1.086 ms (479) [1.077-1.105, i.e. 471-483] | 4.010 ms (389) [3.928-4.131, i.e. 378-397] |
| 6889 | Triton (wgmma) | 1.106 ms (470) [468-476] | 4.409 ms (354) [346-367] |

Same-session reference points: Triton 465/478 TFLOP/s and cuBLAS bf16
509/537 TFLOP/s forward (cuBLAS is not a candidate here -- it is not
batch-invariant).

The portable `mlp-down-gemm-tree-v1` contract is a different arithmetic order, so it
is benchmarked on its own rather than in the table above. Measured on one H100 80GB
HBM3 with the same harness (bf16, K=12288, N=3072; the fp32 operand up-conversion is
part of the cost, hence the extra memory):

| tokens | backend | forward | forward + `dx` + `dW` + `db` |
| --- | --- | --- | --- |
| 4096 | CUDA portable (fp32 tree) | 21.741 ms (14.2) | 57.723 ms (16.1) |
| 6032 | CUDA portable (fp32 tree) | 31.887 ms (14.3) | 85.125 ms (16.1) |
| 6889 | CUDA portable (fp32 tree) | 36.408 ms (14.3) | 96.885 ms (16.1) |

It is byte-equal to the fp32 CPU reference and roughly 38x slower end to end than
the mma contract (57.723 ms against 1.498 ms at 4096 tokens), which is exactly the
trade the row documents: the tensor cores cannot run a tree, and the tree is what
buys byte equality with the reference.

The hand-written Hopper path went from 190 to 433-454 TFLOP/s forward (2.3-2.4x)
once the operands arrive through TMA and the MMAs go through `wgmma` (190 is an
earlier per-warp-MMA revision of the Hopper file, not this PR's portable tree
path), i.e. within
3-7% of the Triton forward. End to end it is ahead by 6-10% at `M = 6889`
and by a tie-to-9% margin at `M = 4096` (two clean repetitions of the reviewer's:
398-400 TFLOP/s against Triton's 368-379; one session saw a tie, 391.3 vs 392.0),
because the backward and the bias fold are where a hand-written schedule wins. `db` alone went
from 0.238/0.379 ms to 0.049/0.143 ms: the old fold launched one thread per column
(3072 threads, ~27 CTAs) and walked a 12 KB-strided column each, so it was
latency-bound at ~105 GB/s; the current one stages row tiles through shared memory
with coalesced 16-byte reads and reads the bf16 gradient directly (bf16 -> fp32 is
exact, so the conversion is free to skip). Both CUDA paths use that fold.

The residual forward gap is measured, not mysterious. (a) **Wave quantization**: at
`M = 4096` the 256x128 tile launches 384 CTAs over 114 SMs = 3.37 waves, an 84%
fill; the same binary at `M = 4864` (4.00 waves) reaches 470 TFLOP/s, i.e. above
Triton's 459 measured in the same sweep. (b) **A staging wall**: rebuilding the
kernel with the `wgmma` calls removed leaves the TMA/mbarrier structure intact and
takes 0.511 ms at `M = 4096` (605 TFLOP/s), while the MMAs alone need 0.408 ms; the
two only half-overlap, so the copy stream caps this staging design near 600 TFLOP/s.
The binding factor is the **L2-to-smem byte volume at the achievable 1-CTA/SM
concurrency**: the copy-only build moves the tiles' 6.1 GB in 0.866 ms, i.e. 7.0
TB/s, which is exactly this part's measured L2-read ceiling for one CTA per SM (a
streaming probe gives 5.05 TB/s at 256 threads/SM and 8.5 TB/s at 2 CTAs/SM), while
the MMA floor is 0.687 ms. That volume is `M*N*K*2 * (1/TM + 1/TN)`, so only larger
tiles shrink it -- and `TM = TN = 256` is register-infeasible: four warpgroups of
`m64n256k16` need 128 accumulator registers per lane, i.e. the entire 64K register
file at 512 threads with nothing left for descriptors or the epilogue. A third round attacked the forward directly and lost ground on every lever:
`TN = 96` buys fill (89.8% vs 84.2%) but the `HGMMA.64x96x16` instruction costs the
same as `64x128x16` for 25% less work and the narrower B tile adds 22% more L2
traffic (423/420 forward, 326/313 end to end); a persistent-CTA epilogue has nothing
to hide (the epilogue is ~0.2% of the forward); and a proper producer/consumer split
with an empty-mbarrier handshake still measures 1-2% slower on the forward and 4.4%
slower end to end at `M = 4096` (`setmaxnreg` cannot help: 128-thread warpgroup
granularity forces a 640-thread CTA that caps registers at 102/thread, below what the
consumers need). A thread-block cluster with TMA multicast was the last physical hypothesis for
this wall, and it does not survive measurement: `multicast::cluster` delivers the
whole box into *every* CTA named by the mask, so it halves the L2 *reads* but not
the bytes each CTA's shared memory receives, and both protocol variants pay for the
cross-CTA readiness handshake on the critical path (a per-stage `barrier.cluster`
rendezvous cost 43% of the kernel; rebuilding it as a leader gate mbarrier fed by
remote `mapa` arrivals still cost 37%). cuBLAS reaches 0.969 ms / 537 TFLOP/s at
`M = 6889` on this part with the 2-SM CUTLASS-style pipeline, so sub-millisecond does
exist here -- but it needs the readiness protocol *pipelined several stages ahead*
rather than synchronous, which is a redesign of the producer, not a tuning knob of
this one.

A seventh round was the last physical hypothesis for that gap, and it closed the
question instead. The readiness protocol was pipelined a full stage ahead of the
arrival it authorizes -- each member's remote arrive lands on the leader's per-slot
gate roughly a microsecond before the leader waits on it -- and a `CLUSTER_N = 1`
control shows that machinery costs about nothing (the 5% that control measures is
entirely its shallower producer lead, a 2-stage prefetch instead of 3). The cluster
still loses, and an attribution build that keeps the cluster launch and the readiness
machinery but drops the multicast loses the same 25%: what remains is co-scheduling
at one CTA per SM, because a 4-CTA cluster needs four co-resident SMs in one GPC and
its CTAs must stay co-resident, so every wave drains with its slowest member. The
economics are one-sided: multicast can cut at most a quarter of the copy stream (the
per-CTA per-stage L2 read drops from 48 KB to 36 KB, i.e. 0.866 -> ~0.65 ms, about
12% of the kernel) against a 25% co-residency penalty measured on the same hardware
-- even a free protocol loses. Cluster size 2 and the other operand axis were not
pursued because both the handshake and the multicast were already excluded as causes
and the penalty scales with co-residency rather than with fan-out.

An eighth round gave the *forward* its own instantiation and closed the gap to a tie: the
forward now runs `TM=128`, `TN=256`, two warpgroups of `m64n256k16`, `BK=64`, a 4-stage
ring and `RASTER_GM=8` -- the footprint Triton uses, plus a fourth stage -- while
`dx`/`dW` keep the `TM=256/TN=128` four-warpgroup shape. Forward 438.7 TFLOP/s at
`M=4096` and 471.6 at `M=6889` against Triton's 463.0 and 473.7 in the same session (a
0.4% tie at the larger token count); end to end 403.3/398.3 against 380.6/366.7.
`M=4096` stays ~5% behind and that is wave arithmetic: both tiles give 384 CTAs there =
3.37 waves, an 84% fill, and the same kernel's steady state is 472.9 -- 441/472.9 =
93.3%, the fill ratio exactly. A smaller tile would add CTAs but raise the L2 volume
through the `1/TM + 1/TN` term that binds everywhere else in this row, so the two
effects cancel. Two TMA issuers (A and B on separate threads, ordered by a named
barrier) measured 0.5-0.9% *slower*, ruling out the single thread's program order as the
serializer, and the new tile's copy-only stream is 0.468/0.756 ms -- 9-13% faster than
the old tile's 0.515/0.866, yet still well under the full kernel -- so the remainder is
consumer-side per-stage serialization, whose levers are register- or cluster-infeasible:
two interleaved accumulator chains need 128 registers per lane at `TN=256`, and the
cluster multicast pays 25% co-residency for at most 12% of kernel time.

Two further rounds were driven by component isolation rather than by guessing, because
`ncu` cannot attach on this host (even a trivial `a + a` kernel reports "No kernels were
profiled"). Isolating the staging from the consumer at `M = 6889`: staging-only 0.756 ms
(688 TFLOP/s), consumer-only 0.881 ms (590 TFLOP/s), full 1.100 ms (473). Their sum is
1.637 ms, so the overlap recovers 33%; perfect overlap would still only reach
`max(0.756, 0.881) = 0.881 ms = 590 TFLOP/s`, i.e. the *consumer* side -- the per-stage
`wgmma.fence` + four `wgmma` + `commit_group` + `wait_group` sequence -- is the binding
term, not the L2 arrival rate. That also exposed two earlier misreadings: Triton's shipped
forward on this box uses `cp.async` + `ldmatrix` (36 `LDGSTS`, 16 `LDSM`, no TMA at all)
and is nevertheless at parity, so TMA is not the differentiator; and the SASS skeleton I
blamed for the gap (286 `BRA`, 68 `BSSY`/`BSYNC`) sits in the *epilogue*'s per-8-column
bounds-checked store loop, not in the mainloop, whose only CTA-wide barriers are four.

Three experiments followed. Decoupling the MMA pipeline from the TMA prefetch by
narrowing the stage and raising `KG` *is* possible (the 64-B-row tensor map and GMMA
descriptor were derived and every sweep row stayed bit-identical), and at equal prefetch
depth `KG=2` beats `KG=1` by 1.3% -- so `wait_group<1>` is not the serializer. But the
narrower stage doubles the stage count (192 -> 384) and the consumer's per-stage cost is
*fixed*, not per byte, so its share doubles: 407-412 TFLOP/s at `BK=32` against 441-454
at `BK=64`. Amortizing it further needs a *wider* stage, which the 227 KB smem cap
forbids. What did pay was specialising the epilogue: a CTA whose whole tile is interior
takes an unguarded store path (edge tiles keep the guarded one), worth **+3.0% at
`M=4096` and +2.1% at `M=6889`**, bit-identical, and visible in the executed rather than
the static instruction mix. With it the hand-written forward is **ahead of Triton at
`M = 6889` (479 vs 470 TFLOP/s in one session, 483 vs 476 in another) and 2-3% behind at
`M = 4096`**, where the remaining gap is the wave arithmetic above -- 384 CTAs = 3.37
waves, and a mixed grid of 26x128 + 12x64 M-tiles would give 456 CTAs = 4.00 waves. End
to end the row is ahead by 5-11% at both token counts.

An eleventh round implemented the mixed grid the fill arithmetic suggested (26 M-tiles of
128 plus 12 of 64 = 456 CTAs = 4.00 waves, with a second `TM=64` instantiation and two
bugs found and fixed on the way: an idle warpgroup still has to walk the mainloop so the
CTA-wide barriers stay matched, and the small tiles need their own A tensor map because
the K-major box height is baked into it) and **reverted it**: a half-height tile carries a
single warpgroup, so its per-CTA tensor-core rate is about half, and the extra CTAs cost
more than the fill recovers -- 413.6 TFLOP/s at `M=4096` and 427.2 at `M=6889` against the
uniform grid's 451.9 and 480.6. The fill diagnosis itself is settled by a same-kernel
control instead: at `M = 4864` the uniform grid is exactly 4.00 waves and the forward
reaches **497-498 TFLOP/s**, 10% above its own `M = 4096` figure, so the whole of that gap
is grid fill. The three RFC reference tiers therefore read: `M = 4096` 450-452 (Triton
464, the 3.37-wave point), `M = 6032` (1664x928) 434 (Triton 435, 5.05 waves), `M = 6889`
(1328x1328) 479-482 (Triton 470-474, 5.68 waves) -- ahead or level wherever the fill is
comparable, behind only where it is not.

The one remaining way to spend the forward's spare registers is closed too, and with
arithmetic rather than a shrug. The forward uses 154 registers per thread of the 255 the
SM allows (39,424 of 65,536 per SM, `LOCAL:0`), and the accumulator alone is 128 of them
(two warpgroups of `m64n256k16` at 128 lanes), so there is headroom -- but the only thing
worth buying with it is the *A operand* out of shared memory, via the `wgmma` RS form
(A in registers, B in smem), which exists for exactly this shape
(`SM90::GMMA::MMA_64x256x16_F32BF16BF16_RS`, K-major A, `ALayout_64x16`, 16 registers per
lane for two-deep `BK=64`). That would cut a stage from 48 KB (A 16 + B 32) to 32 KB and
deepen the B ring from four slots to six or seven, against the 0.219 ms of staging the
current ring cannot hide. It does not pay: the A staging is `SWIZZLE_128B` -- the choice
that itself bought 402 vs 120 TFLOP/s, because unswizzled A multiplies the TMA request
count by eight -- and `ldmatrix` cannot read it (rows 128 B apart versus the 8-row, 16-B
pitch an RS fragment gather wants). Un-swizzling A returns to the 120 TFLOP/s path, and
gathering the fragment per lane instead costs 128 shared-memory loads per CTA-stage with
2-4-way bank conflicts, roughly 0.1-0.3 us on a 1.0 us stage, i.e. 10-30% -- charged to
the *consumer* side, which is the binding term, to buy at most 20% of unhidden staging.
Two interleaved accumulator chains, the other register-shaped idea, need a second set of
128 accumulator registers per lane at `TN=256` and so exceed the 255 per-thread cap
outright.

The overlap question itself was then settled by isolation rather than argument, with four
diagnostics at `M = 6889` in one session (per stage, over 192 stages). Consumer-only runs
603.4 TFLOP/s with `wait_group<1>`; allowing three and four MMA groups in flight gives
605.4 and 606.0, i.e. +0.4%, so the consumer's rate is intrinsic and there is no
MMA-depth headroom to recover. Adding the consumer's whole read volume (80 KB per
CTA-stage of plain `ld.shared`, no MMA, no dependency) on top of the staging costs +10%
and drives the implied port rate to 101 B/clk of the 128 B/clk model with no cliff, so the
shared-memory port is not the serializer (`staging + reads` = 0.802 ms against the full
kernel's 1.079). Pinning the MMA operands to a static slot while the TMA keeps streaming
into the ring and every barrier stays on the critical path moves the time by 0.1%
(1.080 vs 1.079), so ring read/write coexistence is not it either. By elimination the
remaining 0.205 ms is the per-stage control coupling -- the mbarrier wait, the CTA-wide
`__syncthreads`, the single-threaded arm+issue in the loop tail, and
`fence`/`commit_group`/`wait_group` -- about 350 clk per stage, and the two attempts to
remove exactly those measured negative (a warp-specialized producer with full/empty
barriers, -1%; an empty-barrier handshake replacing the sync, -2%). The forward's own
ideal -- staging fully hidden -- is therefore the consumer-only time, 606 TFLOP/s; the
shipped kernel runs at 482, 79.5% of it, while the memory system runs at 59% of the
128 B/clk model (735 TFLOP/s at 1024 clk per stage).

The last untested setting on the dial that multiplies that tax -- the stage count -- was
measured rather than inferred. `BK=128` halves it to 96 logical stages (two 64-element
boxes per operand row, 96 KB per stage, two slots): bit-identical on the first build, and
clearly slower -- 351.7/402.8/383.3 TFLOP/s forward against the shipped 451.9/495.6/481.2,
and 360.7/374.8/363.7 forward+backward against 402.0/404.6/386.5. With two slots and
`KG=1` the fill lead is one stage, so the per-stage exposure to the staging rises from
0.199 us to 0.907 us while the stage count halves; the control tax may well halve with it,
but it is swamped by 4.6x the exposure, and giving B a second slot of lead needs
`3x64 + 2x32 = 256 KB`, over the 227 KB cap. The dial is closed at both ends (`BK=32` at
407-412 against `BK=64` at 441-454), and with it this row's search: the remaining 20% is
one architecture's per-stage control cost, paid by all eight warps, and every structural
way to remove it is either measured negative or does not fit the register, smem, traffic
and bit-identity budgets at once.

The one packing that could have carried that reduction was then built and refuted
structurally. A does not have to share B's granularity, so B can take three 64 KB slots
(a two-stage lead, as in today's ring) and A two 16 KB sub-slots -- 224 KB, inside the cap.
It cannot be made *correct*: an A sub-slot is consumed once per logical stage and refilled
at the end of that same stage, so its recycle interval is one logical stage, while
`wait_group<1>` deliberately keeps the commit group that read it still in flight; the A
ring therefore has to hold `APB*(KG+1) = 4` sub-slots = 64 KB, and 64 + 192 = 256 KB exceeds
the 227 KB cap by 32 KB. The verification signature is decisive rather than
data-dependent -- every shape whose K fits in one logical stage (K = 64, 96, 128) is
bit-identical, and every shape with two or more logical stages mismatches on the forward
while `dx`/`dW`/`db` stay exact -- so the race is in the steady-state recycling, not in the
arithmetic order. Dropping B to two slots restores legality but brings back the one-stage
lead that measured -20% in the previous round, and `KG=0` would drain the pipeline every
stage. The budget cannot hold a halved stage count, a legal A ring and a multi-stage B lead
at once; that is 32 KB of shared memory, not a tunable.

What is left is ~0.27 ms of per-stage handoff that a three-deep ring cannot hide:
1094 stages per SM (192 k-stages x 5.7 waves) at ~0.26 us each of serial
producer/consumer work (mbarrier wait -> four `wgmma` -> `commit_group` ->
`wait_group<KG>` -> `__syncthreads` -> single-thread TMA issue, all in one thread's
program order). Every volume lever is closed (a bigger tile is register-infeasible, a
multicast is co-scheduling-infeasible) and every handoff lever is closed (a smaller
stage keeps the bytes per stage and doubles the stage count; a separate producer warp
is 1-2% slower; deleting the `__syncthreads` buys ~1% and is unsafe). cuBLAS's
0.969 ms at `M = 6889` stays the existence proof, but it comes from a different
producer schedule than this row's register and smem budget admits, not from a knob
this one is missing.

Measured and rejected this round (each with the deciding number): `m64n256k16` on a
128x256 tile (forward comparable at 432/465, but forward+backward fell to 325/281
because the 128-register accumulator hurts `dx`/`dW`); a dedicated TMA producer warp
(13% worse); `TM=192`/3 warpgroups/5 stages (432/447); `TM=128`/`TN=128`/6 stages
(402/402 -- the +33% L2 traffic costs more than the better fill); `STAGES=2` with two
CTAs per SM (282/317); `cp.async.bulk.prefetch.tensor.2d.L2` two stages ahead
(414/420); `KG=2` prefetch (420/460); and `SWIZZLE_NONE` TMA with 16-byte box rows
(120/120 -- 8x the TMA requests, the tile is bit-identical, so this was purely a
bandwidth decision). What shipped instead: grouped L2 rasterization (`RASTER_GM=4`,
consecutive CTAs walk four M-tiles of one N-tile), which is worth ~9% on the
backward and nothing on the forward, plus a non-blocking first `mbarrier.try_wait`.

A final round targeted the under-1 ms goal for `M = 6889` (cuBLAS does it in
0.969 ms) and closed the question instead of reaching it. Finer raster steps
(`RASTER_GM = 7`, the largest block whose A+B window still fits the 50 MB L2: 47.2 MB
against 53.5 MB at `GM = 8`, which thrashing explains why `GM = 8` loses) move the
copy-only time by 1% -- a raster changes where the bytes come from, not how many
there are. `BK = 128` cannot enlarge a TMA box: `SWIZZLE_128B` caps the box inner
extent at 128 B, so it splits into the same two 64-element boxes per row while
halving the ring depth to two stages, which every prefetch-2 measurement in this row
has cost 3-8%. Two TMA issuers are bounded at 1-2% by two direct measurements (a
timing-only build with the per-stage `__syncthreads` deleted gains 1%, and replacing
it with a full/empty mbarrier handshake in an earlier round was 1-2% slower). And a
`TM = 256/TN = 256` tile, which would cut the arrival volume by a third, cannot be
expressed at this warpgroup width for the register reason above.

Build the Hopper path with the repository-wide SM90 switch, which also builds the
other `*_sm90.cu` sources:

```bash
KERNEL_ALIGN_FORCE_SM90=1 pip install -e . --no-build-isolation
```

Without it `mlp_down_gemm_sm90.cu` is not compiled, the extension links without the
`_sm90` symbols and the wrapper transparently uses the portable fp32-tree path -- a
different, also frozen, contract, so a build without the switch is byte-equal to the
fp32 reference instead of to Triton.

## Tests

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m pytest tests/test_mlp_down_gemm.py -q
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m pytest tests/test_mlp_down_gemm_triton.py -q
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/check_operator.py --op mlp_down_gemm \
    --candidate cuda --device cuda --dtype bf16 --batch 64 --seq 64 --k-dim 12288 --n-dim 3072 --check-grad
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/check_operator.py --op mlp_down_gemm \
    --candidate triton --device cuda --dtype bf16 --batch 83 --seq 83 --k-dim 12288 --n-dim 3072 --check-grad
```

The suites cover the schedule's structural promise (one-hot probes: every reduction index
contributes exactly once), determinism, row and tiling invariance (bitwise), the
reduction-length-128 exactness boundary, the declared tolerance bounds against the FP32
reference, the bias epilogue, the three gradients, the per-path contract reporting and the
fail-closed behaviour, and registry dispatch. The portable path is additionally asserted
**byte-equal to the FP32 CPU reference** (forward, `dx`, `dW`, `db`, at the anchor, the RFC
tiers, the odd tails and a tiling shape), and the tree's dispatch boundaries (33/65/.../2049
leaves) are byte-exact. `pytest tests/test_mlp_down_gemm.py tests/test_mlp_down_gemm_triton.py -q`
reports 129 passed on the SM90 build and 114 passed with 15 skipped on the default build.

Shapes are the model's own: `K = 12288` and `N = 3072` everywhere, with the token
count taken from the RFC's reference tiers (`4096` for 1024^2, `6889` for 1328^2,
`6032` for 1664x928), the 256-token schedule anchor, or the one synthetic case the
RFC allows for coverage -- a reduction below the mma chain length and odd tails
(`K = 100`, `K = 12289`).

The harness configurations follow the same geometry: the device candidates are checked at
the RFC's reference shapes (`64x64 = 4096`, `83x83 = 6889`, `104x58 = 6032` tokens), while
the fp32 CPU gold candidate runs at the harness's usual size. That is the contract, not a
gap: this row declares the *forward* as a reduction tree and the *backward* as ascending
left folds, whereas the harness compares a candidate's gradients against autograd through
the gold's forward. At 4096 rows the two fp32 orderings differ by ~1e-3 absolute, above
the shared `reduction.float32` threshold (`atol = rtol = 1e-4`), and the contract forbids
backend-private relaxation, so the gold candidate is exercised where the two orders agree
inside that threshold -- the row's own fp32 accuracy is pinned against fp64 in the suite
instead (`TestExactTruth`).

### Correctness against the exact result

Byte-equality with the reference is self-consistency, not correctness: it cannot
see a mistake the reference and the kernel share (a transposed weight, a dropped
leaf, a wrong cast), and it says nothing about the accuracy of the accumulation
order. Both suites therefore also test against an oracle computed independently of
both implementations -- an fp64 contraction of the same bf16 inputs, rounded once
through the contract's fp32 store and bias -- plus structural identities whose
value is exact in fp32 by construction:

* small integers (``|x|, |W| <= 3``, so every partial sum stays below ``2**24``
  and is representable): the output must equal the exact dot product **bit for
  bit**, which no transposed weight or mis-indexed column survives;
* ``x = 1, W = 1`` at ``K = 12288`` must be exactly ``K``, and ``x = 0`` must
  reproduce the bias exactly (and once) -- a dropped or doubled leaf moves these;
* per-leaf one-hot probes at the model's reduction length (every one of the 384
  leaves' first and last index, plus the short tail leaves at ``K = 12287`` and
  ``K = 12289``): each probe must read exactly ``1.0``.

Measured with the fp64 oracle on H100 PCIe over 24 seed/shape configurations
(~100M elements, both kernels):

| quantity | measured |
| --- | --- |
| elements within 1 bf16 ulp of the result's largest magnitude | **100%** (worst 1.0000 ulp) |
| elements bit-identical to the correctly rounded exact value | 99.30-99.37% (99.48% at one row) |
| the fp32 CPU reference itself vs the exact value, real shape | **100%**, zero deviation |
| integer / identity / ``x = 0`` cases, and ``db`` | **exact**, zero deviation |
| ``dx`` / ``dW`` vs exact fp64 autograd | 99.83% / 99.99% exact, worst 0.5 ulp |
| cuBLAS fp32 from the same inputs (third party) | 99.93-99.97% exact, worst 0.25 ulp |

The residual is bf16 **rounding ties** moved by the ~1e-5 fp32 accumulation noise
that the contract itself specifies (forensic example: an element whose exact value
sits 7.6e-6 above a bf16 tie point, where the fp32 store lands just below it).
cuBLAS shows the same phenomenon at the same order of magnitude, and the
industrial answer is the same: with fp32 accumulation frozen, this is the shape of
the achievable agreement, and every element is inside 1 ulp of it.

### Validating the ROCm path

The row's ROCm slot is served by the Triton backend: the CUDA sources are NVIDIA PTX
(`cp.async`, `ldmatrix`, `mma.sync`, and the Hopper TMA + wgmma block) and are not part
of a ROCm build, with `csrc/ops.cpp` guarding their bindings the same way it does for
`prefix_shared_attention`. The ROCm slot of this row is **not yet validated** -- the
step-by-step qualification below needs an AMD host, and each step names the result it
has to produce, so a reviewer with one can close it.

1. Build on the ROCm machine with a HIP torch (`torch.version.hip` non-null) and
   Triton: `pip install -e . --no-build-isolation`. `KERNEL_ALIGN_FORCE_SM90=1` is a
   no-op there. The registry then resolves `device="cuda"` to the `rocm` platform,
   whose candidate list for this op is `[TritonMlpDownGemmOp, NativeMlpDownGemmOp]`.
   Expected: the build succeeds, and `import rl_engine._C as C; [n for n in dir(C) if
   "mlp_down_gemm" in n]` is **empty** -- all seven `mlp_down_gemm_cuda*` symbols are
   guarded out, and the Triton backend must not need them.
2. Confirm the dispatch:
   `python -c "import torch; from rl_engine.kernels.registry import kernel_registry as r;
   print(type(r.get_op('mlp_down_gemm', device='cuda')).__name__)"`.
   Expected: `TritonMlpDownGemmOp`.
3. Correctness against the same fp32 CPU reference the CUDA paths are held to:
   `python -m pytest tests/test_mlp_down_gemm_triton.py -q` (its device marker is
   `torch.cuda.is_available()`, which is true under HIP), and
   `python scripts/check_operator.py --op mlp_down_gemm --candidate triton --device cuda
   --dtype bf16 --batch 2 --seq 16 --k-dim 12288 --n-dim 3072 --check-grad`.
   Expected: the suite passes, with `TestCudaByteEquality` **skipped** (it needs the
   SM90 build and a cc 9.0 device, which a ROCm build cannot have), and the gtest
   reports `pass_rate=1.0000` at the shared dtype policy (`atol=5e-2, rtol=2e-2` for
   the forward, `atol=1e-1, rtol=2e-2` for the gradients).
4. Numbers, gated on the reference before timing:
   `python benchmarks/benchmark_mlp_down_gemm.py --backend triton --dtype bf16
   --batch 4096 --seq 1` (and `--batch 6889`).
   Expected: the gate line reports `identical >= 99%` and `worst <= 8.0 bf16-ulp` for
   the forward, `dx` and `dW`, and `100.0000%` / `0.0 ulp` for `db`, then the timing
   table. There is no expected TFLOP/s number on ROCm -- the CUDA figures on this page
   are a different part's, not a target.
5. In-row invariance on that backend: `test_forward_deterministic_over_three_reruns`,
   `test_forward_row_invariant`, `test_dx_row_invariant`, `test_dw_padding_invariant`
   and `test_db_padding_invariant` in the same suite, all `torch.equal` (i.e.
   `atol=rtol=0`). Expected: all pass. The repository-wide C3/C4 scripts are *not*
   part of this row's gate on any backend: their `--op` choices and their shapes come
   from the adapters registered over the Qwen3-8B WS1 manifest
   (`ws1-qwen3-8b-dense-primary-v6`, whose per-profile `required_nodes` are the
   C2/C3/C4 matrix). A Qwen-Image row cannot be added there as *tracked* evidence
   without a node and profile in that workload, and running it untracked would
   exercise the harness's own geometry (its `--hidden`, 64 by default) rather than
   this row's `K = 12288, N = 3072`; today
   `check_forward_invariance.py --op mlp_down_gemm ...` exits with
   `invalid choice: 'mlp_down_gemm'`. The row's invariance evidence is its own suite
   (bitwise row/tiling/padding/rerun checks at the model geometry) plus the tier
   backward tests in `tests/test_mlp_down_gemm_triton.py`.
6. What is *not* testable there: byte-equality against the CUDA paths, which do not
   exist on ROCm. What is: the declared tolerance against the fp32 reference, and the
   backend's own batch/tiling/rerun invariance -- both covered by the suite above.

## Known Limitations

- **The build decides the contract.** On one Hopper machine, a build with
  `KERNEL_ALIGN_FORCE_SM90=1` serves `mlp-down-gemm-mma-v1` through `auto`, while a plain build
  serves `mlp-down-gemm-tree-v1`; the two are different reduction orders, so bytes differ
  between them. A deployment that must not change bytes pins the build *and*
  `RL_KERNEL_MLP_DOWN_GEMM_BACKEND`; the route report carries the contract id it actually took.
- **bf16 only.** An fp32 call fails closed on the CUDA and Triton backends; the
  PyTorch reference serves fp32 callers and every non-CUDA device.
- **SM80+** for the portable fp32-tree kernel; the TMA+wgmma path needs a Hopper
  device *and* a build with `KERNEL_ALIGN_FORCE_SM90=1` (it is emitted for
  `compute_90a`, and the device gate is exactly cc 9.0 because the kernel body only
  exists under `__CUDA_ARCH_FEAT_SM90_ALL`), and Triton's wgmma lowering is
  Hopper-specific as well. Older targets fall back to the same pinned order at lower
  throughput.
- The Hopper entries require contiguous operands (the TMA tensor maps are built
  from the extents) and check that explicitly; the portable fp32-tree path stages
  whatever strides it is given.
- **No split-K**, by contract: a single CTA owns each output tile. Small `N` or
  `M` therefore under-fill the device, and shapes are not padded to tile
  multiples (masks handle the tails).
- Odd operand row strides fall back to scalar staging (correct, slower); 16-byte
  `cp.async` staging requires 16-byte-aligned rows.
- The Triton backend's tiles are pinned per contraction rather than autotuned, so
  that a change of `M` cannot change `BLOCK_K`/`num_warps` and with it a row's
  bytes. Unusual shapes are correct but not necessarily at peak.
- The row's train-infer guarantee is per-device: CUDA and ROCm qualify
  separately, as the RFC requires. The ROCm slot is served by the Triton backend
  and is **unvalidated** until an AMD host runs the qualification above.
