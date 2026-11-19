# MoE numerical contract: task 31 foundation

This is the first CPU implementation of the numerical-contract portion of
[Ascend roadmap task 31](https://github.com/RL-Align/RL-Kernel/issues/563).
It provides executable routing and combine references, plus offline validation
of supplied Train/Prefill/Decode/Verify combine outputs. It does **not** qualify
an Ascend implementation or complete task 31's device acceptance criteria.

## Supported profile

`MoENumericalContract` in `rl_engine.contracts.operators.moe_numerical` describes the
versioned `moe.softmax_fixed_tree.v1` profile. It is an explicit profile for
integration, not a declaration that every Qwen model uses these rules.

| Boundary | Rule |
| --- | --- |
| Input logits | Finite `[num_tokens, num_experts]`; declared fp16/bf16/fp32 input dtype |
| Expert selection | Descending **input logits**, before softmax rounding; ties use ascending expert ID |
| Softmax | Convert to FP32, subtract token-local maximum, FP32 exp; sum in ascending expert-ID order with the fixed tree |
| Selected weights | Optionally renormalize the selected slots with the same FP32 tree; multiply by the positive, effective FP32 `routing_scale` afterward |
| Combine inputs | Finite expert rows and optional branches in the declared input dtype; canonical weights are FP32 |
| Weight product | Separate FP32 multiplication before reduction, unless `expert_weight_applied=True` |
| Slot reduction | Adjacent-pair FP32 additions in canonical slot order; carry an odd last node unchanged at every level |
| Additional branches | Add already-gated shared output, then external forward residual, in FP32 |
| Output | One round-to-nearest, ties-to-even cast to declared fp16/bf16/fp32 output dtype |
| Subnormal mode | CPU references force FP32 subnormals to be preserved and restore the caller's setting afterward |

For five slots the tree is `((s0+s1)+(s2+s3))+s4`, not a left fold,
arrival-order sum, rank-dependent tree, or an implicit reduction. A multiply
must not be fused into an addition. No zero padding is used: a lone negative
zero must retain its sign. The `subnormal_mode=preserve` rule is part of the contract fingerprint. CPU references temporarily disable flush-to-zero for FP32 subnormals, serialize that process-wide setting, and restore the caller's previous setting. This does not certify an Ascend flush-to-zero mode.
Exponent underflow may produce zero weights. Non-finite inputs and non-finite
fixed-reference outputs (including final-cast overflow) are rejected.

Quantization other than `none` is rejected. Quantization/dequantization scale
arithmetic, its rounding points, and matching independent references must be
specified before enabling a quantized profile.

## Mapping and ownership

`MoECombineInputs` carries physical expert rows `[T*K, H]`, int64 token/slot/expert
vectors, and an independently captured canonical routing result: expert IDs and
FP32 weights `[T, K]`. Every `(token, slot)` pair must occur exactly once, within
the explicit domain. Selected experts are distinct within a token; physical
expert IDs must match canonical routing. Consumers compare the required
`expected_mapping_run_id` and reject stale records. Changing physical row order
together with its metadata cannot change the result. Strided storage is allowed.

Canonical routing is supplied evidence: comparing two copies of corrupted
metadata cannot establish that the original routing decision was correct.
Likewise, a same-length value permutation without corresponding metadata cannot
be identified from row counts alone.

When `expert_weight_applied=False`, combine applies canonical weights exactly
once. When true, the producer supplies already-weighted rows and combine skips
multiplication. Weights remain in the input for auditing; they are **not**
reapplied. The accuracy oracle also uses the supplied, potentially already-rounded
rows, so it cannot recover precision lost by upstream preweighting. Tensor values
cannot prove ownership; reports distinguish this declaration from verification.
No weight-sum simplex check is imposed on supplied weights, since scaling may
make their sum differ from one.

Shared output is already weighted by its shared gate. `separate`/`external`
modes require the respective `[T,H]` tensor; `excluded` modes reject it. An
external residual must not also be included by the producer. This API handles
**forward** combine: it does not move a residual through RMSNorm backward or
implement the DSV4 gradient-combine API.

Empty-token work is supported by execution references, but rejected as phase
acceptance evidence. Caller-owned tensors are revalidated on every invocation;
the frozen dataclass does not freeze tensor storage.

## Two references and shared rules

`rl_engine.reference.moe_numerical` exports:

- `route_fixed_tree` / `combine_fixed_tree`: CPU PyTorch FP32 arithmetic with
  the declared tree and output cast; these are the fixed-order targets.
- `route_accuracy_reference` / `combine_accuracy_reference`: independent
  Python/FP64 arithmetic using `math.fsum`, independent selection/mapping code,
  and no final low-precision cast. They share contract validation, but do not
  call the fixed-order implementation or its arithmetic helper.

The accuracy oracle evaluates supplied expert contributions; it does not execute
the expert MLP or router projection. Cancellation can cause a fixed-tree result
to fail the independent accuracy criterion. Both verdicts remain visible, and
the validator does not silently relax thresholds to make a result pass.

`moe_rule_manifest` links attention, GEMM, Norm, logprob and combine output
judgments to the existing shared numerical contract. GEMM/Norm/combine use the
reduction class; logprob uses FP32. It includes both contract fingerprints and
the resolved forward/gradient accuracy and invariance rules. These are **output
judgment bindings**, not implementations of those operators' internal arithmetic
or a claim that gradient acceptance has been executed.

## Offline cross-phase comparison

Create one `MoEPhaseOutput` for each of `train`, `prefill`, `decode`, and `verify`,
using independently captured outputs for the same logical sample. Call
`compare_moe_phases` from `rl_engine.validation.moe_numerical.phase_contract` with the
contract, combine inputs, the four records, and explicit expected run and
parameter versions. Each record must include the contract fingerprint, run ID,
parameter version, actual backend label and implementation fingerprint.

The validator checks each actual tensor against both references:

- Exact element bytes against fixed-order **combine** output, including signed zero and
  one-ULP differences. First coordinates are `[token, hidden]` in row-major order.
- Shared forward-accuracy atol/rtol against the independent FP64 result.
- Shape/dtype failures report zero compared elements and metadata errors.
- Non-finite actual outputs fail; report serialization does not emit NaN/Inf.
- Missing/duplicate phases and stale contract/run/parameter labels fail closed.

The top-level report exposes `forward_invariance_passed` and
`forward_accuracy_passed` separately. `passed` requires both under the declared
`passed_policy`; a fixed-tree result can therefore be correct for the declared
implementation while still failing the independent FP64 accuracy oracle under
severe cancellation. Routing evidence uses exact expert IDs and tolerance-based
weight comparison when routing observations are available; this phase API currently
compares supplied combine outputs only and does not claim routing capture. The CPU
`torch.exp` reference is not a cross-device bitwise claim.

The report explicitly states `evidence_scope=supplied_combine_outputs`,
`device_execution_verified=False`, and `producer_identity_verified=False`.
Matching a parameter-version label does not hash or verify parameter content.
Backend labels are caller assertions. Passing CPU fixtures demonstrates the
checker, not four real model execution paths. Contract fingerprints alone are
never acceptance evidence.

## Integration work remaining

Tasks 13–16 own native routing, dispatch/inverse mapping and combine. They can
consume these rules and fixtures without adopting a CPU reference as an NPU
backend. Task 31 still requires agreement with those implementations, actual
Train/Prefill/Decode/Verify captures after parameter updates, model-specific
operator arithmetic and enabled quantization-scale rules, native backward
coverage, and Ascend build/device measurements against independent and
fixed-order references. No executable Ascend registry descriptor is added here.

Run the foundation tests without accelerator hardware:

```shell
python -m pytest tests/contracts/operators/test_moe_contract.py tests/reference/test_moe_numerical_reference.py -q
```
