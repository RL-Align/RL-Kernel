# WS2: bitwise-exact Attention on ROCm

Brings the strict ROCm Attention path to a stated, measured bitwise standard, and adds a
Triton core that is bit-identical to the native reference kernel so the arithmetic
contract is testable without the vendor kernel.

Operator-only. No model checkpoint or serving engine is loaded anywhere in this PR; the
shapes are Qwen3-8B's (`Hq=32`, `Hkv=8`, `D=128`). Measured on 8×MI300X (`gfx942`),
ROCm 7.14.60850, torch 2.12.0, Triton 3.7.0.

---

## 1. What "bitwise" means here, and what it does not

Several different guarantees get conflated in attention work, so this PR states which one
it claims at each boundary:

| Scope | Claim | Enforced by |
| --- | --- | --- |
| Varying batch composition | **bitwise** | `B > 1` rejected at the core (§2.2) |
| Varying padding | **bitwise** | `key_padding_mask` rejected (§2.3) |
| Varying TP degree | **bitwise** | one KV group per launch (§2.4) |
| Varying CP degree | **bitwise** | RCCL-as-transport AG/RS (§2.5) |
| Triton core vs native reference core | **bitwise** | §2.8, measured in §3.1 |
| CUDA production core vs ROCm production core | **not claimed** | different vendor kernels |
| ROCm production core vs reference core | **not claimed** | different kernels; gap sized in §3.4 |

The last two rows are deliberate. CUDA runs FlashAttention 4 CuTe and ROCm runs AITER CK
dense MHA; the tile decomposition, the online-softmax rescale order, and MFMA versus MMA
accumulation all differ. Nothing in the tree claims cross-platform bit equality and this
PR does not add such a claim. What is shared across platforms is the *contract*, not the
bits.

## 2. The algorithmic arrangements

The strict ROCm core does not reimplement attention. It removes every source of
run-to-run and shape-to-shape arithmetic variation from the vendor kernel and records what
was removed, so a mismatch becomes a contract violation rather than a debugging session.

**2.1 Split-KV is structurally impossible, not merely switched off.**
Split-KV partitions the KV axis and merges partial softmax states; the partition count
depends on shape and occupancy, so the reduction order moves with it. CUDA can pass
`num_splits=1` to FA4. AITER exposes no such knob, so the ROCm core binds to the dense,
non-split API entry point instead and records `split_kv_control = "dense_non_split_api"`.
`SplitKVSpec` must be `DISABLED`; a non-disabled spec raises in `__init__` rather than
being quietly honoured.

**2.2 Batch composition cannot change the bits, because `B > 1` is rejected.**
`StrictRocmAiterCKAttentionCore._validate_inputs` refuses any input with `q.size(0) != 1`:
*"strict AITER CK core executes one logical batch row at a time"*. This is stronger than
testing for batch invariance — there is no batched launch whose arithmetic could differ
from the single-row launch, because the batched launch does not exist. Callers materialise
each logical row separately.

**2.3 Padding never enters a reduction.**
`key_padding_mask` is rejected outright: the core *"materializes each unpadded logical
row"*. A padded and an unpadded run of the same logical row cannot differ, because the
padded run is not expressible.

**2.4 One KV group per launch, to make the result independent of TP degree.**
This is the ROCm-specific problem. AITER/CK's reduction order depends on how many heads
shared the launch, and TP performs no cross-rank reduction in attention — it is pure head
sharding — so a head shard computed under TP=4 was *not* bit-identical to the same shard
under TP=8 at some shapes. The provider therefore launches the core once per
`(batch row, KV group)` and concatenates, so every launch sees exactly one KV group and its
Q heads regardless of the TP degree that produced the shard. §3.2 measures both schedules
side by side; the cost is real and is reported.

**2.5 RCCL is a transport, never a reduction.**
`_RCCLRankOrderedTransport` uses RCCL only for `all_gather` and a root-owned `scatter`. Its
`reduce_scatter` first gathers every source shard and then evaluates a fixed balanced rank
tree locally, so the floating-point combine order is ours and does not depend on RCCL's
internal algorithm selection, which varies with message size and topology.

**2.6 The vendor kernel is fingerprinted, not version-pinned.**
AITER dispatches in Python, so a package version does not pin behaviour.
`_load_aiter_ck_ops()` takes a **sha256 of the `aiter.ops.mha` source file** and exports it
as `aiter_source_sha256` in the provenance, so a silent upstream change to the dispatch
logic invalidates the recorded arithmetic identity.

**2.7 Fail closed, everywhere.**
A missing AITER entry point, a missing native extension, or a dispatch that resolves to a
different backend raises. No path substitutes a different kernel to keep a run alive.

**2.8 A reference core shared with CUDA, and a Triton port that matches it bitwise.**
`csrc/cuda/attention/deterministic_attention.cu` is hipified to
`csrc/hip/attention/deterministic_attention.hip`, so the *reference* core genuinely is the
same algorithm on both platforms. This PR adds
`rl_engine/kernels/ops/triton/attention/deterministic_attn.py`, a Triton port whose
contract is bit-identity with that reference. Three things had to be reproduced rather than
re-derived:

- **Dot products stay sequential FMA chains.** The C++ kernel accumulates one element at a
  time in a single thread, so the contraction index is the *loop* and the head dim is the
  *vector*. The opposite, much faster arrangement would reassociate the sum.
- **The row softmax keeps the 256-lane partial layout.** Key `k` belongs to lane `k % 256`;
  each lane sums ascending, then a stride-halving fold combines the partials.
  `_tree_sum_256` reproduces that fold exactly.
- **`expf`/`logf` are re-emitted instruction for instruction.** Every Triton exp/log
  intrinsic — `tl.exp`, `tl.math.exp`, `libdevice.exp` — lowers to a bare `v_exp_f32`, about
  1 ULP away from the `expf` the C++ kernel calls, which alone broke parity on ~14% of
  elements. The helpers reproduce hipcc's two-term argument reduction around that same
  hardware instruction, with an inline-asm barrier to stop LLVM refolding the reduction into
  an FMA. Verified bitwise over 4M+ inputs including subnormals, ±inf and NaN.

The nvcc `expf`/`logf` sequences are not ported, so on CUDA the op refuses to construct
unless the caller passes `require_bitwise_libm=False`, rather than silently returning
non-bitwise results.

---

## 3. Results

Full report, `results.json` and figures: `benchmarks/results/ws2_rocm_mi300x/`.
Reproduce with `python benchmarks/benchmark_ws2_rocm_attention.py`.

Paths: `sdpa` (`torch.nn.functional.scaled_dot_product_attention`, **speed baseline only** —
as in PR #325, no accuracy comparison is mixed into the speed table), `strict-aiter` (the
ROCm production core), `reference-hip` (`_C.deterministic_attention_*`), `triton-bitwise`
(this PR).

### 3.1 Headline: Triton port vs the native reference core

Acceptance is 0 mismatched elements. This is the contract the Triton core exists to hold.

| dtype | S | out | lse | dQ | dK | dV | bitwise |
|---|---:|---:|---:|---:|---:|---:|:---:|
| bf16 | 512 | 0 | 0 | 0 | 0 | 0 | yes |
| bf16 | 1024 | 0 | 0 | 0 | 0 | 0 | yes |
| bf16 | 2048 | 0 | 0 | 0 | 0 | 0 | yes |
| bf16 | 4096 | 0 | 0 | 0 | 0 | 0 | yes |
| fp16 | 512 | 0 | 0 | — | — | — | yes |
| fp16 | 1024 | 0 | 0 | — | — | — | yes |
| fp16 | 2048 | 0 | 0 | — | — | — | yes |
| fp16 | 4096 | 0 | 0 | — | — | — | yes |

`dQ/dK/dV` are measured on the BF16 sweep only.

### 3.2 TP-degree invariance of the strict ROCm core

A head shard computed under TP=N versus the same slice of an unsharded run. TP performs no
cross-rank reduction in attention, so any nonzero value means the kernel's result depends on
how many heads shared the launch.

| S | TP | Local Hq | `raw_launch` out max-abs | `one_kv_group_per_launch` out max-abs |
|---:|---:|---:|---:|---:|
| 512 | 2 | 16 | 0.000000e+00 | 0.000000e+00 |
| 512 | 4 | 8 | 0.000000e+00 | 0.000000e+00 |
| 512 | 8 | 4 | 0.000000e+00 | 0.000000e+00 |
| 1024 | 2 | 16 | **7.812500e-03** | 0.000000e+00 |
| 1024 | 4 | 8 | **7.812500e-03** | 0.000000e+00 |
| 1024 | 8 | 4 | **7.812500e-03** | 0.000000e+00 |
| 2048 | 2 | 16 | 0.000000e+00 | 0.000000e+00 |
| 2048 | 4 | 8 | **3.906250e-03** | 0.000000e+00 |
| 2048 | 8 | 4 | **1.953125e-03** | 0.000000e+00 |
| 4096 | 2 | 16 | 0.000000e+00 | 0.000000e+00 |
| 4096 | 4 | 8 | 0.000000e+00 | 0.000000e+00 |
| 4096 | 8 | 4 | **3.906250e-03** | 0.000000e+00 |

Raw AITER is non-invariant at 5 of 12 points, and *which* points is shape-dependent — the
failure is invisible at S=512 and at S=4096/TP=2, which is exactly what makes it dangerous:
training at TP=4 and rolling out at TP=8 would compare fine on most shapes. The per-KV-group
schedule is bitwise at **12 of 12**. This reproduces PR #319's finding on independent inputs.

### 3.3 Single-GPU latency and memory (BF16)

| S | Path | Fwd median (ms) | vs sdpa | Fwd+bwd (ms) | vs sdpa | Fwd peak MiB | Fwd+bwd peak MiB | out max-abs vs FP64 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | sdpa | 0.0782 | 1.00x | 0.3228 | 1.00x | 12.1 | 32.2 | 8.195e-03 |
| 512 | strict-aiter | 0.2428 | 3.11x | 0.6039 | 1.87x | 14.1 | 288.2 | 2.468e-02 |
| 512 | reference-hip | 0.9659 | 12.36x | 2.9714 | 9.21x | 36.1 | 78.1 | 7.741e-03 |
| 512 | triton-bitwise | 1.3816 | 17.68x | 4.8578 | 15.05x | 36.1 | 78.1 | 7.741e-03 |
| 1024 | sdpa | 0.1319 | 1.00x | 0.4319 | 1.00x | 24.1 | 64.3 | 1.027e-02 |
| 1024 | strict-aiter | 0.2468 | 1.87x | 0.9293 | 2.15x | 28.1 | 1088.4 | 2.609e-02 |
| 1024 | reference-hip | 3.1389 | 23.79x | 12.2009 | 28.25x | 136.1 | 284.3 | 7.810e-03 |
| 1024 | triton-bitwise | 4.8240 | 36.56x | 20.1771 | 46.71x | 136.1 | 284.3 | 7.810e-03 |
| 2048 | sdpa | 0.2887 | 1.00x | 1.0735 | 1.00x | 48.3 | 128.8 | 7.994e-03 |
| 2048 | strict-aiter | 0.2962 | 1.03x | 1.9019 | 1.77x | 56.3 | 4224.8 | 2.027e-02 |
| 2048 | reference-hip | 12.8467 | 44.50x | 47.8741 | 44.60x | 528.2 | 1080.5 | 7.804e-03 |
| 2048 | triton-bitwise | 19.4226 | 67.28x | 76.7414 | 71.49x | 528.3 | 1080.5 | 7.804e-03 |
| 4096 | sdpa | 0.6936 | 1.00x | 3.2464 | 1.00x | 96.5 | 257.5 | 9.604e-03 |
| 4096 | strict-aiter | **0.5644** | **0.81x** | 5.7304 | 1.77x | 112.5 | 16641.5 | 2.138e-02 |
| 4096 | reference-hip | 49.3624 | 71.17x | 173.3365 | 53.39x | 2080.5 | 4209.0 | 7.808e-03 |
| 4096 | triton-bitwise | 86.0655 | 124.08x | 304.1032 | 93.67x | 2080.5 | 4209.0 | 7.808e-03 |

Three things worth reading off this table:

- **The strict production core is not a tax at long sequence.** At S=4096 forward it is
  *faster* than SDPA (0.81x), and its worst case across the sweep is 3.11x at S=512 where
  absolute cost is 0.24 ms. The bitwise arrangements in §2 cost almost nothing in the
  production path.
- **AITER's backward is memory-hungry.** `strict-aiter` fwd+bwd peaks at 16.6 GiB at S=4096
  versus 4.2 GiB for the materializing reference core — the reference core materializes an
  FP32 `[B, Hq, Sq, Skv]` score matrix and is *still* 4x smaller. Worth knowing before
  sizing a training run.
- **The deterministic cores are the most accurate of the four.** Against an FP64 oracle they
  sit at 7.8e-03 versus 9.6e-03 for SDPA and 2.1e-02 for AITER. Determinism here is not
  bought with accuracy.

FP16 is in the full report; the shape of the result is the same.

### 3.4 Production core versus reference core

Two different kernels, so this is a tolerance comparison, not a parity claim. It is here to
size the gap.

| S | out max-abs | out relative-L2 | lse max-abs |
|---:|---:|---:|---:|
| 512 | 3.125e-02 | 5.420e-03 | 9.537e-07 |
| 1024 | 3.125e-02 | 5.505e-03 | 1.431e-06 |
| 2048 | 1.562e-02 | 5.595e-03 | 1.907e-06 |
| 4096 | 1.562e-02 | 5.633e-03 | 3.815e-06 |

### 3.5 Batch-composition invariance

Bitwise at every sequence length for every path. For `strict-aiter` the property is
structural (§2.2) rather than measured: the batched launch does not exist.

### 3.6 Distributed CP over the RCCL AG/RS transport

Schedule: all-gather Q/K/V and the position ids over the CP group, run the strict core once
on the full sequence, reduce-scatter `(out, lse)` back to this rank's query range. Acceptance
is bitwise against a CP=1 run of the same core on the same inputs. S=4096, BF16.

| Topology | World | TP | CP | Replicas | Local Hq/Hkv | Median (ms) | p95 (ms) | Peak MiB/rank | out bitwise | lse bitwise | Repeat |
|---|---:|---:|---:|---:|---|---:|---:|---:|:---:|:---:|:---:|
| `tp1_cp2` | 2 | 1 | 2 | 1 | 32/8 | 1.8352 | 1.8726 | 160.5 | yes | yes | yes |
| `tp2_cp2` | 4 | 2 | 2 | 1 | 16/4 | 1.2114 | 1.3138 | 80.3 | yes | yes | yes |
| `tp1_cp4` | 4 | 1 | 4 | 1 | 32/8 | 1.3973 | 1.4363 | 160.5 | yes | yes | yes |
| `tp2_cp2_x2` | 8 | 2 | 2 | 2 | 16/4 | 1.2297 | 1.2948 | 80.3 | yes | yes | yes |
| `tp2_cp4` | 8 | 2 | 4 | 1 | 16/4 | 1.2594 | 1.3294 | 80.3 | yes | yes | yes |
| `tp1_cp8` | 8 | 1 | 8 | 1 | 32/8 | 1.4112 | 2.2029 | 160.5 | yes | yes | yes |

All six topologies are bitwise against CP=1 on both `out` and `lse`, with 0 mismatched
elements summed across every rank, and repeat-bitwise on every rank. `tp2_cp2_x2` is the
8-rank case PR #319 used: two independent CP groups running side by side at TP=2/CP=2.

---

## 4. Figures

![Single-device latency and memory grid](benchmarks/results/ws2_rocm_mi300x/single_gpu_grid.png)

![TP-degree invariance](benchmarks/results/ws2_rocm_mi300x/tp_degree_invariance.png)

![Distributed CP latency](benchmarks/results/ws2_rocm_mi300x/distributed_cp_latency.png)

`reference-hip` and `triton-bitwise` allocate exactly the same buffers, so their memory
curves coincide and the later-drawn series hides the earlier one.

---

## 5. Files

| Path | What |
| --- | --- |
| `rl_engine/kernels/ops/triton/attention/deterministic_attn.py` | New. Triton core, bit-identical to `_C.deterministic_attention_*`. |
| `rl_engine/kernels/ops/triton/attention/__init__.py` | Exports the new op and `BITWISE_LIBM_PARITY`. |
| `tests/test_triton_deterministic_attention.py` | New. 71 parity / invariance tests. |
| `benchmarks/benchmark_ws2_rocm_attention.py` | New. Measurement matrix and figures, reusing PR #325 / #328 helpers. |
| `benchmarks/results/ws2_rocm_mi300x/` | Report, `results.json`, figures. |
| `csrc/ops.cpp` | Fix: the merge from `feat/rocm-deterministic-collectives` dropped an `#if !defined(USE_ROCM)` around the `prefix_shared_attention` registration but kept its `#endif`, leaving 13 `#endif` against 12 `#if`. The ROCm build failed with `#endif without #if`. |

## 6. Test plan

- `pytest tests/test_triton_deterministic_attention.py` — 71 passed. 15 shape/mask/scale
  configs × {bf16, fp16} × {forward, backward}, plus end-to-end autograd through both ops,
  the fully-masked-row case, batch-slice invariance, and a direct pin on the `expf`/`logf`
  helpers against the vendor libm.
- `pytest tests/test_deterministic_attention_cuda.py` — 614 passed (the native core is
  unaffected).
- `python benchmarks/benchmark_ws2_rocm_attention.py` — the report above.

## 7. Known limitations

- **CUDA is not covered by the Triton core's bitwise claim.** The nvcc `expf`/`logf` argument
  reductions are not ported, and no CUDA device was available to derive or verify them.
  `TritonDeterministicAttentionOp` raises on CUDA unless the caller passes
  `require_bitwise_libm=False`; a test pins that behaviour on both platforms.
- **The Triton core is a parity core, not a FlashAttention replacement.** Like the native
  reference it materialises the full FP32 `[B, Hq, Sq, Skv]` score matrix and runs
  scalar-order reductions; §3.3 shows the cost.
- **`_C` does not register `deterministic_attention_forward_fp32`.** The `.cu` defines it but
  the pybind registration is missing, so the *native* `DeterministicAttentionOp.forward_fp32`
  raises `AttributeError`. Pre-existing, not touched here; the Triton `forward_fp32` works and
  its test validates against the op's own downcast instead of the native path.
- **A stale comment contradicts the shipped TP policy.**
  `rl_engine/integrations/vime/attention.py` still carries a comment saying RL-Kernel "binds
  the degree rather than paying ~3x forward time", from before the merge that introduced the
  per-KV-group launch loop. The provenance dict immediately below it correctly reports
  `tp_degree_invariant: True`. §3.2 shows the code is right and the comment is wrong; flagged
  here rather than silently rewritten.
