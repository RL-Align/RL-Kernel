# WS1 Numerical Precision Standard: Ground Truth Definition, Judgment Methods, and Threshold Provenance

| Field | Value |
|---|---|
| Document status | **Draft Normative** |
| Document version | v1.0 |
| Document date | 2026-09-11 |
| Precision contract | `rl_engine/kernels/gtest/tolerance_contract.json` |
| Contract version | `ws1-c1-v2` |
| Implementation audit baseline | `6ceeb62e073d74e00a8c5b56af538ee1a7417773` (`origin/main`) |
| Target model | Qwen3-8B Dense (`ws1-qwen3-8b-dense-primary-v6`) |
| Required backend profiles | `cuda_bf16`, `triton_cuda_bf16` |
| Optional precision | FP16 (complete when declared; not required for WS1 exit) |
| Out of WS1 scope | FP8, ROCm, Ascend, and any optional fused path not listed in the C8 required-op matrix |

> **On document status.** The normative sections (§1–§8) define the numerical precision targets and judgment methods for WS1 and may be cited as the project's standard. Section 9 records where the current repository diverges from that standard, Section 10 bounds the applicability of historical evidence, and Section 11 gives the plan for closing the gaps. Until every blocking item in Section 11 is closed, this repository must not be described as fully satisfying this standard. Normative targets and implementation status are stated separately throughout.

---

## 1. Summary

WS1 separates numerical judgment into four classes, so that standards of different natures are not collapsed into a single "tolerance table":

| Judgment | Compared quantities | Judgment mode |
|---|---|---|
| Forward accuracy | BF16 candidate ↔ FP32 reference | Element-wise tolerance |
| Gradient accuracy | BF16 backward ↔ FP32 reference gradient | Element-wise tolerance |
| Forward / gradient invariance | Transformed config ↔ canonical config | Zero tolerance (bitwise is the target) |
| Chain-level parity | Training-style teacher forcing ↔ inference-style rollout decode | Chain aggregate metrics |

**The ground truth is defined as follows.** Take the same low-precision input values the candidate consumes, upcast them to FP32, compute the operator's original mathematical definition entirely in FP32 with TF32 and autocast explicitly disabled, and keep the output in FP32 rather than rounding it back to BF16.

Current thresholds are established from three classes of evidence:

1. **Test conventions of upstream projects**, used to establish the reasonable order of magnitude (PyTorch, NVIDIA TransformerEngine).
2. **Floating-point error analysis**, used to explain why different dtypes and operator classes require different tiers.
3. **Measurements on H20 in a fixed environment**, used to calibrate the two rows for which no upstream baseline of the same comparison semantics exists: logprob and backward reduction.

An important qualification: **an external project's threshold supplies a reference magnitude; it does not by itself prove that value is appropriate for RL-Kernel.** A final threshold must satisfy three conditions simultaneously: well-defined comparison semantics, measured coverage on a fixed workload, and traceable version information. Section 6 therefore distinguishes three modes of adoption — direct adoption, partial reference, and measured calibration — and never reports a single-field match as a full match.

---

## 2. Normative Scope and dtype Policy

### 2.1 Target dtype policy

| Item | Normative target |
|---|---|
| Candidate execution | BF16 |
| Reference computation | FP32 |
| Reduction / accumulation | FP32 by default; any exception must be declared per operator |
| Default candidate output | Same as execution dtype |
| Logprob aggregate output | FP32 |
| TF32 | Disabled for the reference and for the WS1 required candidate path |
| Autocast | Disabled on the reference path |
| Backend-private tolerance relaxation | Not permitted |

### 2.2 On `backend_private_tolerance_relaxation: false`

This field means that the CUDA and Triton profiles must both pass against the same WS1 contract, and that neither backend may relax a threshold locally inside the formal WS1 gate.

It does **not** mean that every legacy unit test in the repository has had its local tolerance removed. As of the audit baseline, `tests/` still contains 306 hard-coded `atol=` occurrences across 36 files that do not resolve through the contract. The formal WS1 gate uses the unified contract; the convergence plan for legacy tests is item 13 in Section 11.

### 2.3 Operator class partition

Thresholds are given per operator class rather than per operator, because the error magnitude is driven primarily by reduction topology:

| Op class | Operators covered | Dominant error source |
|---|---|---|
| `elementwise` | embedding, rope, silu, swiglu, pack | A single output rounding |
| `reduction` | rms_norm, qk_norm, lm_head, det_gemm | Long-reduction accumulation |
| `attention` | attention | Two GEMM stages plus a softmax reduction |
| `logprob` | logp, linear_logp, batch_invariant_logp | Vocabulary-width log-softmax |

---

## 3. Definition of the FP32 Reference

### 3.1 Normative definition

An FP32 reference that satisfies WS1 must meet all five conditions:

1. **Same-source inputs.** The candidate and the reference consume the same low-precision input values; the reference upcasts those already-quantized values to FP32 rather than generating a separate set of high-precision inputs.
2. **Computation from the original definition.** The operator's mathematical definition is evaluated in FP32, including reduction, normalization, softmax, log-softmax, and the GEMM accumulator.
3. **No implicit precision loss.** TF32 and autocast are explicitly disabled so the reference cannot silently enter a low-precision path.
4. **FP32 output.** When compared against a BF16 candidate, the reference output is not first rounded to BF16.
5. **Same-source gradients.** Backward uses gradients derived from the same FP32 forward definition.

Condition 1 is the core of this definition: it places input quantization error outside the measurement, so that a comparison reflects only the kernel's own error plus one output rounding, and can therefore be compared directly against the floating-point theoretical floor.

The implementation lives in the `forward_fp32` methods under `rl_engine/kernels/ops/pytorch/`. RMSNorm is representative (`rl_engine/kernels/ops/pytorch/norm/rms_norm.py#L136-L140`):

```python
x_f = x.float()                         # same BF16 inputs, upcast to FP32
var = x_f.pow(2).mean(dim=-1, keepdim=True)
normed = x_f * torch.rsqrt(var + eps)
out = normed * weight.float()
return out.to(output_dtype)             # output_dtype = float32
```

The `forward_fp32` methods for attention and the LM head additionally use a context manager that disables `torch.backends.cuda.matmul.allow_tf32` and autocast.

### 3.2 Relationship to the upstream reference construction

Conditions 1 through 3 share their origin with the reference construction used in the FlashAttention test suite, whose reference implementation documents its `upcast` argument as:

> `upcast`: whether to cast all inputs to fp32, do all computation in fp32, then cast output back to fp16/bf16.
>
> — Dao-AILab/flash-attention, `tests/test_flash_attn.py` (commit `0f3fb00d`)

**The two differ in output comparison strategy, and this must be stated plainly.** Several FlashAttention tests cast the reference output back down to low precision and apply a relative criterion, namely that the candidate's error may not exceed a small multiple of a plain PyTorch low-precision implementation's error:

> `assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()`

The WS1 accuracy reference keeps its output in FP32 and applies fixed `atol`/`rtol` criteria. It is therefore accurate to say that WS1 adopts the FlashAttention reference **construction**, and inaccurate to say the two use the same **judgment** method. Adoption of the relative criterion is tracked as item 10 in Section 11.

### 3.3 Why the reference must be FP32

The unit roundoff of BF16 is:

```text
u_bf16 = 2^-8 = 3.90625e-3
```

For operators containing long reductions, a low-precision reference can itself accumulate error of the same order as, or larger than, the candidate's. The classical error analysis of softmax and log-sum-exp gives a forward error bound of roughly `(n+3)u` and states explicitly when that bound ceases to constrain anything:

> they provide no useful information when n ≳ 1/u, and for fp16 this happens for n as small as 2048.
>
> — Blanchard, Higham & Higham, *IMA Journal of Numerical Analysis* 41(4)

Substituting `u_bf16` lowers the failure boundary to `n ≳ 256`. The actual reduction widths in Qwen3-8B are:

| Reduction | Width |
|---|---|
| RMSNorm hidden width | 4096 |
| Vocabulary log-softmax | 151936 |

Both are far beyond that boundary.

This argument does **not** support the claim that all BF16 results are mathematically invalid. The bound is a worst-case bound, and the same paper notes that under a probabilistic model of rounding errors `n` may be replaced by a small constant multiple of `√n`. What the argument does support is narrower and sufficient: **a BF16 implementation is unsuitable as the high-precision numerical baseline for WS1**, because the baseline's own error guarantee is no longer strong enough to adjudicate the quantity under test.

### 3.4 The limits of a reference's authority

A reference is authoritative because its operator definition is explicit and its dtype path is controlled, not because of the name of the library that produced it. cuBLAS illustrates the point; its reproducibility guarantee carries strict preconditions:

> all cuBLAS API routines from a given toolkit version, generate the same bit-wise results at every run when executed on GPUs with the same architecture and the same number of SMs
>
> — NVIDIA cuBLAS Documentation, Results Reproducibility

The same document states that bitwise reproducibility is not guaranteed across toolkit versions, that the guarantee no longer holds when multiple CUDA streams are active, and that results are not guaranteed to be bitwise reproducible when atomics mode is enabled.

Three boundaries follow:

- A cuBLAS path with unpinned version, algorithm, and workspace **cannot** serve as the bitwise oracle for WS1 invariance.
- With a fixed environment and a numerical tolerance, cuBLAS **can** still form part of an accuracy reference.
- Whether a reference qualifies is determined by its actual dtype, algorithm, and output semantics, not by which library was called.

---

## 4. Judgment Methods

### 4.1 Accuracy

Accuracy is judged element-wise, with the reference on the right-hand side:

```text
abs(candidate - reference) <= atol + rtol * abs(reference)
```

This is the same form used by PyTorch, TransformerEngine, and Liger-Kernel. The implementation is `_compare_output` in `rl_engine/kernels/gtest/op_checks.py`, which upcasts both sides to FP32 before calling `torch.allclose`.

The logprob class uses `rtol = 0`, because a log-probability can approach zero, at which point relative error loses a stable meaning.

The reported `max_abs_error` and `max_rel_error` are observations, not the basis of the verdict. In particular, `max_rel_error` uses `abs(reference).clamp_min(1e-12)` as its denominator, so elements near zero produce very large values; readers should rely on the `passed` field.

### 4.2 Invariance

The normative target for invariance is a true bitwise comparison across batch, chunk, padding, and layout transformations, requiring that:

1. shapes are identical;
2. dtypes are identical;
3. raw bit patterns are identical;
4. `+0` and `-0` are treated as distinct bit patterns;
5. NaN handling is specified explicitly rather than left to the default behavior of `allclose`.

The `atol = 0` and `rtol = 0` entries in the contract are the **numerical representation** of bitwise mode; they are not equivalent to a complete bitwise comparison implementation. The gap between the current implementation and this target is documented in §9.3.

### 4.3 Chain-level parity

Chain-level metrics are managed separately from per-operator accuracy; see Section 8.

---

## 5. Threshold Tables

### 5.1 Forward accuracy

| Op class | FP32 `atol / rtol` | BF16 `atol / rtol` | FP16 optional `atol / rtol` |
|---|---:|---:|---:|
| Elementwise | `1e-5 / 1e-5` | `2e-2 / 1.6e-2` | `1e-3 / 1e-3` |
| Reduction | `1e-4 / 1e-4` | `5e-2 / 2e-2` | `1e-3 / 1e-3` |
| Logprob | `1e-5 / 0` | `6e-2 / 0` | `5e-3 / 0` |
| Attention | `1e-4 / 1e-4` | `5e-2 / 2e-2` | `1e-3 / 1e-3` |

### 5.2 Gradient accuracy

Gradient rows are independent of forward rows and are not inherited from them.

| Op class | FP32 `atol / rtol` | BF16 `atol / rtol` | FP16 optional `atol / rtol` |
|---|---:|---:|---:|
| Elementwise | `1e-5 / 1e-5` | `2e-2 / 1.6e-2` | `1e-3 / 1e-3` |
| Reduction | `1e-4 / 1e-4` | `1e-1 / 2e-2` | `1e-3 / 1e-3` |
| Logprob | `1e-5 / 0` | `5e-2 / 0` | `5e-3 / 0` |
| Attention | `1e-4 / 1e-4` | `5e-2 / 2e-2` | `1e-3 / 1e-3` |

---

## 6. Threshold Provenance and Mode of Adoption

The table below distinguishes three modes of adoption. **A match on `atol` or `rtol` alone is not reported as a full match.**

| Contract row | Primary basis | Mode of adoption | Notes |
|---|---|---|---|
| Elementwise BF16 | TransformerEngine RMSNorm test; PyTorch BF16 default `rtol = 1.6e-2` | Field reference | `2e-2 / 1.6e-2` matches the typical acceptance magnitude for BF16 elementwise ops; PyTorch's default pair uses `atol = 1e-5`, which differs from this row |
| Reduction BF16 | TransformerEngine linear accuracy | Direct upstream reference | `5e-2 / 2e-2` applies to operators containing longer reductions |
| Attention BF16 | TransformerEngine DPA forward / backward | Partial reference | `atol = 5e-2` is the same tier; `rtol` is not uniform across tensors and directions upstream, and this row's `rtol` is wider than the upstream value |
| Gradient reduction BF16 | TransformerEngine GPT-level tolerance; H20 backward data | Combined evidence | `atol = 1e-1` is additionally calibrated against an H20 near-zero error of `0.0978`; a larger-magnitude `0.1034` sample is covered by the `rtol = 2e-2` term |
| Logprob BF16 | H20 full-model data | Measured calibration | Upstream CE and logprob tests compare within a single dtype, which is not the BF16↔FP32 comparison semantics of this row, so no directly transferable baseline exists; the historical sample was `max_abs_dlogp = 0.05064`, and the threshold is `0.06` |
| FP32 elementwise / logprob | PyTorch default tolerance | Field reference (`atol`) | PyTorch's float32 default `atol` is `1e-5`; its default `rtol` of `1.3e-6` differs from this row |
| FP32 reduction / attention | Magnitude of PyTorch OpInfo matmul and reduction tests | Magnitude reference | WS1 uses a unified `1e-4 / 1e-4` reduction tier |
| FP16 optional | PyTorch FP16 default `rtol = 1e-3` | Field reference (`rtol`) | `atol` is specified per op class and is not equivalent to PyTorch's complete default pair |

### 6.1 Parameters that are not floating-point tolerances

The contract's `clip_interval = [0.8, 1.2]` is the **policy interval** of the PPO ratio. It defines the statistical semantics of `clipfrac0` and is not a floating-point error tolerance. Its resemblance to identically named parameters in other RL frameworks is a policy convention and does not constitute an endorsement of numerical precision. This document keeps it separate from the threshold tables.

---

## 7. Thresholds and Floating-Point Error Magnitudes

### 7.1 BF16

BF16 stores 7 mantissa bits, which the implicit leading bit extends to 8 significant bits, giving `u ≈ 3.906e-3`. WS1 commonly uses `rtol` between `1.6e-2` and `2e-2`, roughly `4u` to `5u`.

This relationship shows that the thresholds sit at the same order of magnitude as the fundamental rounding scale of a BF16 output. **It does not on its own prove the thresholds are sufficient.** A final threshold must still cover:

- error dominated by `atol` near zero;
- accumulated error introduced by reduction topology;
- the differing error distributions of forward and backward;
- representative shapes, extreme-valued inputs, and long-tail tokens.

### 7.2 FP32 reduction

For a sequential dot product of length `n`, the classical worst-case bound is:

```text
gamma_n = n*u / (1 - n*u)
```

With `n = 4096` and `u_fp32 = 5.96e-8`:

```text
sqrt(n)*u  ≈ 3.8e-6      # empirical scale, contingent on a probabilistic rounding-error model
WS1 threshold  1e-4
gamma_n    ≈ 2.44e-4     # worst case for sequential accumulation
```

`1e-4` lies between the typical error scale and that worst-case bound, and is the engineering tier WS1 currently adopts: below the worst case, so it is a real constraint; above the typical scale, so normal operation does not produce false alarms.

One qualification is required. Real kernels may use pairwise trees, online reductions, MMA tiles, or multi-stage reductions, whose error behavior differs from sequential accumulation. Each op class must therefore be validated against its actual reduction topology and measured distribution; a single conclusion cannot be derived from sequential accumulation at `n = 4096` alone.

---

## 8. Chain-Level Logprob Parity

| Metric | Definition | BF16 threshold | Current status |
|---|---|---:|---|
| `max_abs_dlogp` | `max(abs(lhs_logp - rhs_logp))` | `0.06` | Required gate; calibrated from historical H20 data |
| `approx_kl0` | `mean(exp(dlogp) - 1 - dlogp)` | `0.05` | Required report and gate; identified as loose |
| `clipfrac0` | `mean(1[exp(dlogp) outside clip_interval])` | `0` | Required gate |

`dlogp` is defined as `comparison_lhs_logp - comparison_rhs_logp`, with the comparison roles fixed by report kind: for `train_infer_logprob_parity` the left and right sides are training-style teacher forcing and inference-style rollout decode respectively.

### 8.1 `max_abs_dlogp`

The current value of `0.06` derives from `max_abs_dlogp = 0.05064` recorded in a historical H20 full-model run. That calibration covers only 27 active tokens, which is too few to estimate the tail of the distribution stably. The current value should therefore be treated as a **provisional, version-scoped engineering threshold**.

Recalibration requires at minimum:

- an active-token fixture on the order of a thousand tokens;
- multiple seeds, sequence lengths, and logit scales;
- `max`, `mean`, P99, and P99.9 distribution statistics;
- the raw report, fixture hash, code commit, driver, CUDA, PyTorch, and Triton versions, and GPU model.

### 8.2 `approx_kl0`

The historical H20 samples were `1.4e-4` and `2.5e-4`. The current threshold of `5e-2` is roughly two orders of magnitude above them and exerts limited constraint.

This is not an oversight. The contract recorded the value as provisional when it was introduced:

> pending measured chain-level distributions, after which its threshold may be tightened
>
> — `tolerance_contract.json`, `approx_kl0.threshold_rationale`

The proposed value of `1e-3` is closer to the magnitude used by several RL training systems, but **until new distribution data is generated and the contract is updated, it remains a proposal and not the standard in force.**

### 8.3 Proposed changes

- Tighten `approx_kl0` to `1e-3`.
- Add `mean_abs_dlogp` with a proposed threshold of `0.005`.
- Decide, on the basis of the new measured distribution, whether `max_abs_dlogp` should be demoted from a hard gate to monitor-only.

These changes alter pass/fail semantics and must be released with a new contract version (`ws1-c1-v3` is suggested). They cannot be enacted by editing this document alone.

---

## 9. Implementation Conformance Audit

The findings below are based on audit baseline `6ceeb62e073d74e00a8c5b56af538ee1a7417773`, with each item tied to a verifiable source location. Line numbers refer to that commit.

### 9.1 Summary table

| Component | Status | Basis |
|---|---|---|
| PyTorch FP32 references (RMSNorm / attention / LM-head / activation / RoPE) | Conformant | `forward_fp32` computes entirely in FP32 with TF32 and autocast explicitly disabled |
| CUDA RMSNorm / QK-Norm | Conformant | Statistics and reduction in FP32 |
| CUDA attention | Conformant | Row max, rescale, and softmax in FP32; `csrc/cuda/attention/prefix_shared_attention.cu#L127`, `#L237-L253` |
| CUDA SiLU / SwiGLU / RoPE | Conformant | Intermediates in FP32, output stored at execution dtype |
| Triton standard attention | Conformant | Contains no `tl.dot`; 16 explicit `tl.float32` accumulators; `rl_engine/kernels/ops/triton/attention/standard_attn.py` |
| Triton linear_logp | Conformant | `tl.dot(..., input_precision="ieee")` at `#L77`; lse and logp allocated FP32 at `#L109-L110` |
| Chain LM-head / logprob / residual stream | Conformant, and stricter than defaults | `score_logits` in FP32, residual stream in FP32 throughout; `rl_engine/alignment/qwen3_dense.py#L683`, `#L707-L729` |
| **Deterministic GEMM accumulation** | **Exception not declared in the contract** | See §9.2 |
| **FP32 reference for `det_gemm`** | **Non-conformant** | See §9.3 |
| **Invariance comparator** | **Non-conformant** | See §9.4 |
| **CUDA generic logp output dtype** | **Non-conformant** | See §9.5 |
| **RMSNorm reference rounding order** | **Not pinned in the contract** | See §9.6 |
| Legacy unit-test tolerances | Pending convergence | 306 `atol=` occurrences across 36 files in `tests/` that do not resolve through the contract |
| Legacy Triton attention paths | Scope must be bounded | See §9.7 |

### 9.2 Deterministic GEMM uses FP32 leaves with a BF16 tree merge

§2.1 states that reduction and accumulation are FP32 by default and that **any exception must be declared per operator**. The deterministic GEMM takes such an exception, and the exception is documented in the kernel but is not carried in the contract.

The CUDA kernel reduces K with a mid-split tree whose leaves accumulate in FP32 and whose merges are performed in BF16 (`csrc/cuda/gemm/det_gemm_kernel.cu#L68-L84`):

```cpp
constexpr int K_TREE_LEAF = 32;

__device__ __forceinline__ nv_bf16 bf16_add(nv_bf16 a, nv_bf16 b) {
  return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
}

__device__ nv_bf16 k_tree_naive(..., int lo, int hi) {
  if (hi - lo <= K_TREE_LEAF) {
    float acc = 0.0f;                       // FP32 leaf accumulation
    for (int k = lo; k < hi; ++k)
      acc += __bfloat162float(A[row * K + k]) * __bfloat162float(B[k * N + col]);
    return __float2bfloat16(acc);           // leaf result rounded to BF16
  }
  const int mid = lo + (hi - lo) / 2;
  return bf16_add(k_tree_naive(..., lo, mid),
                  k_tree_naive(..., mid, hi));   // BF16 tree merge
}
```

The file header states the design and its rationale (`#L11-L14`):

> Both: BF16 in / FP32 accum / BF16 store / no TF32 / no split-K.
> K is reduced with a mid-split tree. A contiguous half-K GEMM is one child, so simulated TP=2 (a+b) matches TP=1. TP=8 left-fold does not.
> Leaves stay FP32 (naive: 32-wide MAC; SM90: one BK).

A `static_assert` binds the SM90 tile width to the same leaf size (`#L205`), so the SM90 and fallback paths produce the same reduction tree.

The Triton implementation matches this topology: tile accumulators are `tl.float32` (`rl_engine/kernels/ops/triton/matmul/det_gemm.py#L394`, `#L466`, `#L484`, `#L719`), while the tree workspace is allocated BF16 (`#L847-L849`), so each merge level loads to FP32, adds, and rounds back to BF16 on store.

**This is a deliberate design choice, not a defect.** The BF16 merge is what makes a simulated TP=2 split reproduce the TP=1 result bitwise, which serves the invariance objective directly. The gap is documentary rather than numerical: the contract declares `accumulation_dtype: float32` globally and requires per-operator exceptions to be declared, but carries no such declaration for `det_gemm`.

Two resolutions are available, and the project must choose one explicitly:

- **Option A — FP32 throughout the tree.** Keep the partial workspace and every merge in FP32, rounding only at the output boundary. Appropriate if `accumulation_dtype: float32` is to remain a hard global rule. This changes the TP-invariance property described in the header comment and must be re-validated against it.
- **Option B — declare the exception.** Record `det_gemm` in the contract as `FP32 leaf accumulation + BF16 tree merge + BF16 output`, with the TP-invariance rationale, and state the leaf width (`K_TREE_LEAF = 32`) as part of the declared semantics.

The implementation currently matches Option B; the contract currently asserts Option A. A specification and an implementation must not carry two mutually contradictory descriptions.

### 9.3 The `det_gemm` FP32 reference does not satisfy §3.1

The gold path for `det_gemm` resolves to `NativeGemmOp.__call__` (`rl_engine/kernels/gtest/operator_specs.py#L148-L149`), implemented as:

```python
def __call__(self, a, b):
    return torch.matmul(a, b)
```

With BF16 inputs this call returns BF16, so the gold output dtype is BF16. That violates condition 4 of §3.1 and conflicts with the contract's declared `reference_dtype: float32`. Measurement confirms `gold_dtype = torch.bfloat16`.

The module's own docstring already marks it as unsuitable for this role:

> `torch.matmul` (cuBLAS) does NOT guarantee batch-invariance … This op exists only as a correctness reference and benchmark target, NOT as a fallback.

The consequence is that the accuracy judgment for `det_gemm` compares two BF16 GEMMs rather than a BF16 candidate against an FP32 reference. Historical `det_gemm` accuracy results obtained under this condition do not constitute verification against the §3.1 definition. This defect is independent of §9.2: it concerns the reference, whereas §9.2 concerns the candidate kernel.

`op_checks.py` contains a provenance check that verifies the gold dtype equals `reference_dtype`, but it fires only when provenance is supplied. The `scripts/check_operator.py` path does not supply it, so the mismatch is not intercepted.

### 9.4 Invariance judgment is not a raw-bit comparison

`_compare_logical_tensors` in `forward_invariance.py` upcasts both sides to FP32 and then calls `torch.allclose`:

```python
canonical_fp32 = canonical.float()
transformed_fp32 = transformed.float()
...
passed = bool(torch.allclose(transformed_fp32, canonical_fp32, atol=atol, rtol=rtol))
```

(`rl_engine/kernels/gtest/forward_invariance.py#L297-L352`)

`_compare_parameter_grad` in `gradient_invariance.py` reuses the same function (`#L308-L331`), so both invariance judgments share this implementation.

Because the BF16-to-FP32 upcast is lossless, this implementation is equivalent to exact numerical equality when dtype and shape match. The gap against the §4.2 target is confined to two points: `+0` and `-0` are judged equal, and NaN handling is left to the default behavior of `allclose` rather than being specified.

**Reporting requirement.** Until the comparator is corrected, the historical C3 / C4 / C8 invariance results must be described as **zero-tolerance numeric equality** and must not be described as having completed strict bitwise verification.

### 9.5 CUDA generic logp output dtype conflicts with the policy

`fused_logp_forward` allocates its output from `logits.options()`:

```cpp
auto output = torch::empty({logits.size(0)}, logits.options());
```

(`csrc/fused_logp_kernel.cu#L565-L567`)

BF16 logits therefore produce a BF16 logprob, conflicting with the `Logprob aggregate output = FP32` entry in §2.1. The same file already provides an FP32 entry point, `fused_logp_forward_fp32` (`#L571-L573`), which specifies `at::ScalarType::Float` explicitly.

The formal WS1 candidate should call the FP32 output entry point, and the provenance validator should verify `logprob_aggregates_dtype = float32` per op class. **Changing the output dtype may change the existing `0.05064` calibration, so the logprob threshold must be re-measured afterward.**

### 9.6 RMSNorm reference rounding order is not pinned in the contract

Mainstream implementations differ in the final rounding step of RMSNorm:

| Order | Expression | Adopted by |
|---|---|---|
| HF order | `weight * x.to(bf16)`; round first, then multiply by weight | HuggingFace `Qwen3RMSNorm`; vLLM `forward_native`; SGLang with `cast_x_before_out_mul=True` |
| Kernel order | `(x * weight).to(bf16)`; multiply in FP32, round once | vLLM CUDA kernel; SGLang `forward_native` default |

The WS1 `NativeRMSNormOp` uses kernel order (`rl_engine/kernels/ops/pytorch/norm/rms_norm.py#L136-L140`):

```python
x_f = x.float()
var = x_f.pow(2).mean(dim=-1, keepdim=True)
normed = x_f * torch.rsqrt(var + eps)
out = normed * weight.float()
return out.to(output_dtype)
```

The two orders differ by roughly 1 ulp, and SGLang provides a dedicated switch specifically to align with HF semantics.

The choice itself is defensible, but **the contract does not currently declare which order it adopts**. Since the rounding order used by the training-side framework determines the achievable floor for chain-level parity, this should be pinned in the contract with its rationale recorded.

### 9.7 Bounding the TF32 claim

When Triton executes an FP32 `tl.dot` on NVIDIA GPUs, the default `input_precision` may be TF32. The audit confirms:

- **The WS1 required path handles this correctly.** `det_gemm.py` passes `allow_tf32=False` (`#L729`), `linear_logp.py` passes `input_precision="ieee"` (`#L77`), and `standard_attn.py` contains no `tl.dot` at all.
- **Two legacy paths do not.** `rl_engine/kernels/ops/triton/triton_attn.py` contains 7 `tl.dot` calls and `rl_engine/kernels/ops/triton/attention/chunked_flash_attn.py` contains 4, none with a precision specified. Neither file is referenced by `rl_engine/kernels/gtest/` or `rl_engine/alignment/`, so neither is in the C8 required-op matrix. `chunked_flash_attn.py` is reached through `rl_engine/kernels/ops/rocm/attention/flash_attn.py`, and ROCm is out of WS1 scope per the header.

The TF32 entry in §2.1 is therefore scoped to the WS1 required path. The claim "TF32 disabled throughout" must not be extended into an unverified repository-wide statement.
---

## 10. Historical Evidence and Its Applicability

The checked-in C8 H20 evidence originates from:

| Field | Value |
|---|---|
| Source commit | `fdf5bcc5165820abb506291a29370225306514ca` |
| GPU | NVIDIA H20, CC 9.0 |
| Driver | 580.76.05 |
| CUDA / PyTorch / Triton | 12.8 / 2.8.0+cu128 / 3.4.0 |
| Workload | `ws1-qwen3-8b-dense-primary-v6` / `ws1-c2-v7` |
| Result | `green = 176`, `N/A = 16`, `red = 0` |

**This result attests only to the historical execution state at that fixed source commit in that fixed environment.** It does not automatically attest to current `main`, and it does not substitute for final-commit GPU CI. Specific limitations:

- the evidence records cell status and lacks complete per-tensor raw error distributions;
- the "bitwise gate" of that period was implemented by the zero-tolerance `allclose` described in §9.3;
- `csrc`, Triton, gtest, scripts, and tests have all changed between the evidence commit and the audit baseline.

Before a new WS1 document or contract is released, evidence should be regenerated at the target commit, with an artifact containing at minimum:

- source commit and dirty-tree status;
- expected and actual backend plus the underlying kernel ID;
- input fixture ID, hash, shape, and seed;
- dtype, TF32, autocast, and key runtime readbacks;
- per-tensor max / mean / P99 / P99.9 absolute and relative error;
- dtype, shape, and raw-bit results of the bitwise comparison;
- the complete software and hardware environment.

---

## 11. Remediation Plan

### 11.1 Release-blocking items

Until these are closed, this document remains Draft Normative and the repository must not be described as fully satisfying the WS1 precision standard.

| # | Item | Section | Impact |
|---|---|---|---|
| 1 | Decide between Option A (FP32 throughout the reduction tree) and Option B (declare the `det_gemm` exception, including `K_TREE_LEAF`, in the contract), then align specification and implementation | §9.2 | Option A changes the TP-invariance property and requires re-validation; Option B requires a contract revision |
| 2 | Change the `det_gemm` gold to an FP32 GEMM (inputs upcast, TF32 disabled), with backward updated to match; supply provenance on the `check_operator.py` path so the dtype check fires | §9.3 | `det_gemm` accuracy results must be regenerated |
| 3 | Change the invariance gate to a true raw-bit comparison with explicit `±0` and NaN semantics | §9.4 | C3 / C4 / C8 invariance conclusions must be restated and re-run |
| 4 | Switch CUDA generic logp to the FP32 output entry point and add op-aware provenance validation | §9.5 | The logprob threshold must be recalibrated |
| 5 | Declare the RMSNorm rounding order in the contract and record its relationship to the training-side framework | §9.6 | Affects the explanation of the achievable chain-parity floor |
| 6 | Define the required-op scope so that legacy Triton paths without explicit TF32 disabling are either excluded or fixed | §9.7 | Affects the defensible scope of the TF32 claim |
| 7 | Re-run H20 GPU evidence at the target release commit, producing the full artifact listed in §10 | §10 | Affects the currency of every historical conclusion |

### 11.2 Threshold evolution items

These change pass/fail semantics and must ship with a new contract version (`ws1-c1-v3` suggested).

| # | Item | Section |
|---|---|---|
| 8 | Tighten `approx_kl0` from `5e-2` to `1e-3` | §8.2 |
| 9 | Add `mean_abs_dlogp` with a proposed threshold of `0.005` | §8.3 |
| 10 | Recalibrate `max_abs_dlogp` on a thousand-token fixture and decide whether it becomes monitor-only | §8.1, §8.3 |
| 11 | Introduce the FlashAttention-style relative criterion (candidate error within 2× the PyTorch BF16 path's error, 3× for backward), retaining the current absolute thresholds as caps | §3.2, §7.1 |
| 12 | Move gradient judgment to rel-peak or RMS normalization, replacing the scale-dependent absolute `atol` | §7.1 |
| 13 | Converge the 306 hard-coded tolerances in `tests/` onto the unified contract | §2.2 |

### 11.3 Release criteria

Once blocking items 1 through 7 are closed, this document should be promoted to **Normative**, with the contract version, source commit, evidence artifact hash, and verification environment pinned together in the header.

---

## 12. Reference Index

### 12.1 External sources

External references use commit permalinks so that movement of a default branch does not break traceability. The links and quoted text below were verified on 2026-09-11.

| Source | Link | Cited in |
|---|---|---|
| PyTorch tolerance implementation | [`torch/testing/_comparison.py` @ `31527a43`](https://github.com/pytorch/pytorch/blob/31527a43dbf6adf4df2e6eb5c7b38094fec6b6f6/torch/testing/_comparison.py) | §6 |
| PyTorch default tolerance table | [torch.testing documentation](https://docs.pytorch.org/docs/stable/testing.html) | §6 |
| PyTorch OpInfo tolerance overrides | [`common_methods_invocations.py`](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_methods_invocations.py) | §6 |
| TransformerEngine numerical tests | [`tests/pytorch/test_numerics.py` @ `02f8e754`](https://github.com/NVIDIA/TransformerEngine/blob/02f8e754fa9a7324896b0373c76a5dcddf70fdcc/tests/pytorch/test_numerics.py) | §6 |
| FlashAttention reference construction | [`tests/test_flash_attn.py` @ `0f3fb00d`](https://github.com/Dao-AILab/flash-attention/blob/0f3fb00d3f34196ca55f30d2f6c5dce0709b4667/tests/test_flash_attn.py) | §3.2 |
| Softmax / log-sum-exp error analysis | [arXiv:1909.03469](https://arxiv.org/abs/1909.03469) | §3.3 |
| cuBLAS reproducibility | [NVIDIA cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility) | §3.4 |
| NVIDIA floating-point compliance | [Floating Point and IEEE 754 Compliance](https://docs.nvidia.com/cuda/floating-point/index.html) | §3.4 |
| Triton FP32 dot precision | [`triton.language.dot`](https://triton-lang.org/main/python-api/generated/triton.language.dot.html) | §9.6 |
| Qwen3 modeling code | [`modeling_qwen3.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py) | §9.5 |

### 12.2 In-repository sources

Paths and line numbers refer to audit baseline `6ceeb62e073d74e00a8c5b56af538ee1a7417773`.

| Content | Path |
|---|---|
| Precision contract | `rl_engine/kernels/gtest/tolerance_contract.json` |
| Four-judgment matrix | `rl_engine/kernels/gtest/four_judgment_matrix.py` |
| Accuracy comparator | `rl_engine/kernels/gtest/op_checks.py` |
| Invariance comparators | `rl_engine/kernels/gtest/forward_invariance.py`, `gradient_invariance.py` |
| Operator gold mapping | `rl_engine/kernels/gtest/operator_specs.py` |
| PyTorch FP32 references | `rl_engine/kernels/ops/pytorch/` |
| Chain model | `rl_engine/alignment/qwen3_dense.py` |
| Workload manifest | `rl_engine/testing/ws1_manifest.json` |
| H20 C8 evidence | `docs/design/ws1-c8-execute.json` |

### 12.3 Companion research reports

The external basis for this document is drawn from two full research reports containing all verbatim quotations, URLs, and unverified-item annotations:

- `docs/design/ws1-tolerance-contract-review.md` — threshold and ground truth compared against external standards
- `docs/design/ws1-qwen3-8b-precision-and-mismatch-survey.md` — per-operator precision conventions for Qwen3-8B and the training-inference mismatch literature
