# P1 mHC + RMSNorm Start Kit (`P1-S0`)

The start kit unblocks every P1 sub-issue (`P1-D1`…`P1-D6`): it freezes the
layer contract, provides a bit-exact FP32 oracle for the six WS1 operators,
generates seeded golden fixtures, and ships one acceptance command that any
backend PR can run independently.

Issue: [DSV4][P1/7] mHC 与 RMSNorm 确定性前向/反向 (#2).

## The eight sub-issues of #2

`P1-N` is a development-order label; **GitHub issue numbers stay authoritative
for links**. Five WS1 operator tasks, each delivering forward *and* backward
together, plus three WS2 parallelism tasks.

| Label | Issue | Stage | Scope |
| --- | --- | --- | --- |
| `P1-S0` | this PR | — | Start kit: contract, oracle, fixtures, provider stub, acceptance command |
| `P1-1` | #14 | WS1 | `hc_split_sinkhorn` — controller mapping + 20 Sinkhorn rounds (fwd + bwd) |
| `P1-2` | #15 | WS1 | `fp32_gemm_rms` — FP32 controller projection + controller RMS (fwd + bwd). **Also carries the fixed-K / batch-invariant GEMM reference + bit-equivalence harness** that P2/P3/P5/P7 consume |
| `P1-3` | #16 | WS1 | `mhc_post` — sublayer output written back into the four residual streams (fwd + bwd) |
| `P1-4` | #17 | WS1 | `mhc_pre` / `h_aggregate` — mHC entry and four-stream aggregation (fwd + composite bwd) |
| `P1-5` | #18 | WS1 | `rmsnorm_residual` — single-stream RMSNorm + residual fork (fwd + bwd) |
| `P1-6` | #19 | WS2 | `fp32_gemm_rms` TP/SP contract — the full `K=16384` controller dot and RMS statistic may not silently become local-K |
| `P1-7` | #20 | WS2 | Rank-local invariance of the three mHC operators under TP/SP/CP/DP/PP |
| `P1-8` | #21 | WS2 | `rmsnorm_residual` TP+SP semantics and cross-rank `dgamma` reduction |

```
Foundation v1 -> P1-S0 -> {P1-1 || P1-2 || P1-3 || P1-4 || P1-5} -> P1-R0
                                     |
                                     +-> WS2: P1-6, P1-7, P1-8
```

Every WS1 task depends only on `P1-S0`, never on the others. `P1-4`'s composite
backward calls `hc_split_sinkhorn_bwd` and `fp32_gemm_rms_bwd`, but it codes
against the oracle for those, so it does not wait on `P1-1` or `P1-2`.

For the three WS2 tasks, the `placement` field on `LayerContract` and
`check_capability` are the attachment points: a provider that has not declared
a placement already fails closed on it.

## What is in the kit

| Module | Contents |
| --- | --- |
| `rl_engine/mhc/reduction.py` | The two pinned reduction trees. Defines the golden bytes for every P1 accumulation. |
| `rl_engine/mhc/contract.py` | `LayerContract`, `ResidualBatch`, `ControllerParams`, `NormParams`, `GradBoundary`, fingerprints. |
| `rl_engine/mhc/oracle.py` | FP32 reference for the six operators plus the full block forward/backward composition. |
| `rl_engine/mhc/provider.py` | `MHCProvider` protocol, `ReferenceProvider` (oracle-backed), `StubProvider` (fail-closed), `check_capability`. |
| `rl_engine/mhc/fixtures.py` | Seeded fixture cases and the golden-hash manifest (`tests/fixtures/p1/golden_hashes.json`, the CI anchor). |
| `rl_engine/mhc/trace.py` | Boundary hashes + `first_divergence` (P1-local stand-in for `TraceEnvelope`). |
| `scripts/check_p1.py` | The acceptance command. |

## The reduction trees (the heart of the contract)

Everything in `oracle.py` reduces through `reduction.py`; no `torch.sum`,
`matmul`, `mean` or `einsum` appears anywhere in the operator bodies. Two
trees, and only two:

- **Long reductions** — the `K = 4·D` controller dot, the `D`-wide
  sum-of-squares, the token-major parameter gradients — use a **single FP32
  accumulator walking ascending indices left to right**, with every multiply
  and add rounding separately. This is the order the repository's existing
  `reduce_rows_fp32` left fold already uses, so a P1 kernel and the WS1 VJP
  path agree by construction.
- **4-element stream reductions** — the four mHC residual streams, and the
  row/column sums of the 4×4 Sinkhorn matrix — use the balanced tree
  **`(a0+a1)+(a2+a3)`** pinned by #2.

Banned downstream: Split-K, Stream-K, atomic partial accumulation, and any
order that varies with batch size, token count, SM count or any other runtime
condition. `tests/test_p1_reduction.py` pins both trees with cases where the
alternatives visibly disagree.

## What Megatron actually does — and why it is not the byte reference

#2 says to prefer Megatron's implementation over vLLM's. Reading the source
(`megatron/core/transformer/hyper_connection.py`,
`megatron/core/fusions/fused_mhc_kernels.py`, both on `dev`) shows what that
can and cannot mean.

**Megatron's own two paths do not agree with each other.** `config.use_fused_mhc`
selects between a native path and a fused Triton/cuTile path:

| | native | fused |
| --- | --- | --- |
| controller RMS | `norm = x.norm(-1)`; `r = 1/(norm/sqrt(K) + eps)` | `r_val = sqrt(sum_sq / K)`; `1/(r_val + eps)` |
| controller GEMM | `torch.matmul`, FP32 | `ct.mma(..., tfloat32)` — **TF32** |
| K reduction | whatever Inductor emits (`@torch.compile`) | **`split_k = 16` when `K >= 16384`**, and also runtime-autotuned |
| 4-stream mix | `torch.bmm` (cuBLAS) | Triton kernel |
| softmax | `torch.softmax` (`exp`) | `tl.exp2(x * log2e)` |

The `sqrt(s)/sqrt(K)` vs `sqrt(s/K)` split is ~1 ulp; TF32 in the fused MMA is
~1e-3 relative. The authors are aware of this class of difference and manage it
as a tolerance budget — there is a comment in the fused kernel reading *"Square
in fp32: a bf16 square/reduction loses ~2e-3 relative on the RMS scale, which
native (fp32) does not."*

Three of #2's explicit bans are violated by Megatron's fused path **at exactly
the production shape**: Split-K is on (`K = 16384` selects `split_k = 16`),
TF32 is on, and the reduction order is autotuned per machine per run. Measured
separately: eager `.sum(-1)` on CUDA is **not batch-invariant** — rows `0:8`
produce different bytes inside a 64-row batch than in an 8-row batch. On CPU it
happens to be invariant.

None of that is a defect in Megatron. It is a correct set of performance
choices for a framework that never promised bit-exactness. But it means:

> **"参考 Megatron" can only mean its formulas and constants — never its
> reduction order.** Reproducing Megatron's bytes is not a goal that can be
> held, because Megatron does not reproduce its own.

What the source *does* settle, and this kit adopts verbatim: the affine
association `h = r * proj * alpha_ + bias`; the Sinkhorn schedule and its axis
convention; the max-shift before `exp`; the FP32 controller path
(`mark_keep_in_fp32` on the projection weight, alphas and bias); the three
scalar alphas; and the module decomposition that leaves the sublayer outside.

## Frozen numeric contract (recap of #2 + decisions made here)

From the issue:

1. All multiplies and reduction accumulations are FP32.
2. Each operator performs **exactly one** FP32→BF16 downcast, at its output:
   `mhc_pre`'s aggregated hidden, `rmsnorm_residual`'s normalized row, and
   `mhc_post`'s `R_new`. No intermediate BF16 cast anywhere.
3. `PRE`, `POST`, `C`, the controller projection `P` and the RMS scale `r`
   stay FP32 while travelling between operators.
4. Controller arithmetic is `h = ((r * P) * alpha) + bias`;
   `PRE[i] = sigmoid(h[i]) + 1e-6`; `POST[i] = 2·sigmoid(h[4+i])`;
   `L = h[8:24].reshape(4,4)`.
5. `hc_mult = 4`, layout `PRE[0:4] POST[4:8] COMB[8:24]`, `sinkhorn_iters = 20`,
   `eps = 1e-6`; `sum + eps` may not be replaced by a `clamp`.
6. `rmsnorm_residual` is `rsqrt(mean(x²) + eps)`; the controller RMS is
   `1/(sqrt(mean(x²)) + eps)`. The two forms are never interchangeable.

Decisions this kit had to freeze (flagged for review on #2; changing any of
them means regenerating the manifest and bumping the schema/profile id):

| # | Decision | Rationale |
| --- | --- | --- |
| D1 | **`q = sqrt(s) / sqrt(K)`, not `sqrt(s / K)`**, with `s` from our own fixed-tree sum-of-squares (*not* `torch.norm`, which is not bit-equal to `sqrt(x.square().sum())` on either device). Backward reuses the saved `q`. | #2 states both forms because **Megatron states both**: `native_proj_rms` computes `norm/sqrt(K)` while the fused kernel computes `sqrt(s/K)` under a comment that says `norm/sqrt(K)`. The native path is the semantic reference, so it wins. |
| D2 | **Sinkhorn `sum_row(M)[i] = Σ_j M[i,j]` (row sums, broadcast along j) and `sum_col(M)[j] = Σ_i M[i,j]`.** Schedule is literal: `softmax_row(L)+eps`, one column normalize, then 19×(row, column) — 20 column normalizations, 39 in total. | The issue names the steps but not the axis convention; this is the reading that makes rows/columns each sum to 1. |
| D3 | **`softmax_row` subtracts the row max before `exp`**, with the max taken on the same balanced 4-way tree. | Unshifted `exp` overflows on the saturating-logit fixture; the shift has to be pinned rather than left to the kernel. |
| D4 | **Sinkhorn backward walks the recorded 39 normalizations in reverse, one VJP per step.** No fixed-point / implicit-differentiation shortcut, no fused simplification. | #2 forbids "mathematically equivalent but differently associated" forms. Cross-checked against autograd in `test_p1_oracle.py`. |
| D5 | **Numeric profile `oracle-fp32-mhc-v1`**: FP32, the two trees above, mul-then-add (**no FMA fusion**). A strict CUDA kernel matches with `__fmul_rn`/`__fadd_rn` or registers its own profile. | Byte-equality needs the rounding points pinned, not just the order. |
| D6 | **`alpha` is three learnable FP32 scalars** (`alpha_pre`, `alpha_post`, `alpha_res`) broadcast over the PRE / POST / COMB segments; `bias` is a `[24]` vector. Backward returns three scalars, each a pinned segment fold on top of the token fold. | Matches Megatron exactly (`alpha_pre/post/res` are `nn.Parameter(torch.full((1,), ...))`, cat-expanded in `_compute_h`). Forward is identical to a `[24]` gain, but backward is not — a `[24]` `dAlpha` would leave the segment reduction order unpinned. |
| D7 | **The transformer sublayer is external to P1.** `y_sublayer` enters the block as data and `d_normalized` / `d_residual` enter the backward as data; `dy_sublayer` is an output boundary. | This is what makes P1 acceptance runnable with no P2–P7 code in the loop, exactly as the issue requires. |
| D8 | **`unfused` is canonical and the default.** For `rmsnorm_residual`, #18 says to prefer TE's `TEFusedResidualRMSNorm` first and self-write only if TE fails the deterministic contract — **it fails**: it refuses to expose the pre-normalization intermediate (it raises on any forward hook), so the fork and the norm cannot be hashed as separate boundaries. That is exactly the escape hatch #18 provides, so v1 is unfused. `fused-pre-norm` stays supported for an engine that physically cannot expose the intermediate. Fusion itself is not banned — changing the reduction layout or moving a downcast point is. | Train/infer byte-equality needs every boundary hashable on its own, so a divergence localizes to one operator instead of one megakernel. A fused fast path can be swapped back through the provider hook once proven byte-equal. |
| D9 | **`trainability='mixer-frozen'` returns `None` for `d_controller_weight`/`d_alpha`/`d_bias`**, not zeros. | #2: a stop-grad mixer must not leak `dMixWeight`. `None` cannot be silently summed into an optimizer; a zero tensor can. |
| D10 | **Gradients are returned FP32** (the accumulator dtype); rounding to BF16 happens only at an outer block edge. | Consistent with "FP32 reductions, BF16 boundaries". |

## Byte-equality scope

Strict byte-equality is required **between Megatron training and Miles
inference on the same numeric profile and device**. The committed manifest
anchors the CPU x86 oracle; `scripts/check_p1.py` recomputes the oracle on the
provider's device, so transcendentals (`sigmoid`, `exp`, `rsqrt`) never cross
devices inside a strict comparison. Hardware without equivalent capability
must register its own profile with an explicit tolerance — never silently
relax.

The acceptance command additionally re-runs each fixture row on its own and
checks that the bytes do not move: **same row, different batch / padding /
stride ⇒ identical output**, which is acceptance criterion 2 of #2.

## How a sub-issue PR uses the kit

1. Subclass `ReferenceProvider`, override only the operators your PR delivers
   (everything else stays on the oracle), and set `name` / `numeric_profile`:

   ```python
   from rl_engine.mhc.provider import ReferenceProvider

   class MyCudaProvider(ReferenceProvider):
       name = "my-cuda"
       numeric_profile = "cuda-ffma-strict-v1"

       def mhc_post_fwd(self, r_old, y, c, post):
           return my_cuda_kernel(r_old, y, c, post)
   ```

2. Run `python scripts/check_p1.py --provider your.module:YourProvider
   [--device cuda]`. Every boundary must be byte-equal; exit code 1 otherwise.
3. Ship the check output and your `provenance()` in the PR description.

Fixture cases: `one_row`, `packed_t16`, `packed_t7_odd`, `fused_pre_norm`,
`mixer_frozen`, plus operator edge cases `sinkhorn_edges` (saturating
sigmoids, tied logits, a zero row, and a magnitude that makes the `sum + eps`
guard load-bearing) and `rms_edges` (zero row, subnormal-ish and large
magnitudes, exact powers of two).

Fixture geometry is a scaled-down layer (`hidden = 128` ⇒ `K = 512`) so the
serial oracle stays CPU-cheap. `hc_mult`, `controller_n`, `sinkhorn_iters` and
both epsilons are the real production constants; the full DSv4 geometry
(`hidden = 4096`, `K = 16384`, `N = 24`) is pinned separately by
`LayerContract.assert_production()`.

Regenerate the manifest after an intentional contract change:

```bash
python -m rl_engine.mhc.fixtures --write-manifest
```

## Open questions for review

- **Miles must also run unfused** (D8). #2 records that Miles/XoRL fuse the
  mHC pre-mix and RMSNorm into a single launch. If that kernel keeps the same
  reduction layout and downcast points it can stay and still be byte-equal;
  if it does not, the inference side has to unfuse for v1. **This decision
  moves cost onto the Miles side and needs their sign-off.**
- **Sinkhorn iteration detail** — #2 notes that TogetherAI never published the
  per-round detail and that the checkpoint plus the Miles implementation are
  the only reference. Megatron's two implementations agree with each other and
  with the issue text, and the kit matches both; if Miles differs, D2/D4 change
  and the manifest is regenerated.
- **Upstream nit worth raising (not a bug report)** — `fused_mhc_kernels.py`
  computes `sqrt(sum_sq / K)` under a comment reading
  `# 2. Compute r = norm / sqrt(K)`. Which is the intent? A one-line answer
  confirms D1 from the other direction.
- **`dAlpha` consumers** — the three scalar gradients are DP-reduced by the
  outer DDP, but any consumer that currently expects a `[24]` `dAlpha` needs
  updating (D6).

## Non-goals of the kit

No CUDA/Triton kernels, no TE/Megatron/vLLM injection, no RoPE (that is P2's,
per #1), no attention or MoE (#4, #8/#10), and no WS2 TP/SP/CP/PP gates. The
`placement` field and the WS2 notes in the operator docstrings mark where those
gates will attach; `check_capability` already fails closed on any placement a
provider has not declared.
