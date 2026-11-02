# ISSUE-108 Session Log

This document records the engineering decisions made while building the ISSUE-108 kernel correctness checker. It is intentionally concise and review-oriented: it explains what was added, why it was added, how to use it, and what is still out of scope.

## Logging Rules

- Record the reason for each meaningful change, not only the files touched.
- Keep changes minimal and independently verifiable.
- Be explicit when a path is only a smoke test or an experimental path.
- Do not present failed CUDA paths as supported capabilities.
- Gold implementations must come from `rl_engine.reference`.

## Goal

The goal of this work is to add a minimal, reusable operator correctness framework for post-training kernels.

The framework should:

- Generate deterministic operator inputs.
- Run PyTorch gold implementations and backend candidates on the same inputs.
- Compare every tensor output with dtype/operator-class tolerances.
- Report absolute error, relative error, pass rate, and final pass/fail status.
- Expose a CLI so a developer can validate a backend candidate without editing test files.

## Final Layout

```text
rl_engine/kernels/gtest/
  __init__.py
  op_checks.py
  operator_inputs.py
  operator_specs.py
  tolerance.py
  tolerance_contract.json

tools/validation/operators/check_operator.py

tests/validation/operators/test_op_checks.py
tests/validation/operators/test_operator_inputs.py
tests/contracts/test_tolerance_contract.py
```

## Key Design Decisions

### Tolerance Contract

Files:

```text
rl_engine/contracts/numerical.py
rl_engine/contracts/profiles/precision/ws1.json
tests/contracts/test_tolerance_contract.py
```

Decision:

- Store tolerance values in a small contract file rather than hard-coding them inside tests.
- Resolve tolerance by `op_class + dtype`, with optional `arch_key` overrides.
- Treat `default` as the generic fallback, not as CPU-specific tolerance.

Current accuracy classes:

```text
elementwise
reduction
logprob
```

### Operator Check Runner

Files:

```text
rl_engine/validation/operators/op_checks.py
tests/validation/operators/test_op_checks.py
```

Decision:

- `OperatorCase` describes one deterministic test case: name, op class, dtype, inputs, and gold function.
- `CandidateSpec` describes one implementation under test: name, function, backend, and optional arch key.
- `run_operator_suite()` runs candidates against gold outputs and returns structured reports.
- The runner compares forward outputs only in this minimal version.

Review follow-up:

- `op_checks.py` includes a TODO for optional gradient checks on differentiable operators.
- Gradient checks require additional metadata and input cloning rules, so they are intentionally tracked as follow-up work instead of being silently implied by this PR.

### Operator Inputs

Files:

```text
rl_engine/validation/operators/operator_inputs.py
tests/validation/operators/test_operator_inputs.py
```

Decision:

- Build standard semantic inputs for each operator.
- Support both `random` and `constant` initialization.
- Make random inputs reproducible with `--seed`.
- Preserve semantic shapes such as `[B, S, V]`; do not flatten inputs for backend-specific kernels inside input generation.

Current input builders cover:

```text
rms_norm
matmul
attention
logp
rope
silu
swiglu
embedding
lm_head
kv_cache_attention
```

### Operator Specs

File:

```text
rl_engine/validation/operators/operator_specs.py
```

Decision:

- Keep operator-specific registration outside `tools/validation/operators/check_operator.py`.
- Register PyTorch gold paths and backend candidate paths in one place.
- Require `gold_path` to point into `rl_engine.reference`.

Current minimal registered operator:

```text
op: logp
op_class: logprob
gold: rl_engine.reference.logprob.logp.NativeLogpOp
candidates:
  pytorch      -> NativeLogpOp
  cuda         -> FusedLogpGenericOp
  cuda-generic -> FusedLogpGenericOp
  cuda-sm90    -> FusedLogpSM90Op
```

Important note:

- `candidate=pytorch` is only a smoke test for the checker itself.
- CUDA, Triton, ROCm, and future hardware-specific implementations are candidates.
- Registry dispatch is tested separately from this accuracy harness because it is hardware-dependent.
- Do not compare two operators that implement different math, such as ordinary `logp` and `linear_logp`.

### SM90 Adapter Exception

Current code contains `_LogpSM90CandidateAdapter` in `operator_specs.py`.

Reason:

- The existing SM90 logp wrapper accepts flattened inputs, while the checker standard input for `logp` is `[B, S, V]` logits and `[B, S]` token ids.
- The adapter exists only to validate the checker path against the current SM90 wrapper.

Long-term rule:

- Backend wrappers should align with the standard operator interface whenever possible.
- New operators should not rely on permanent test-side adapters for ordinary shape or parameter-name differences.

## CLI Usage

CPU smoke check against the PyTorch candidate:

```bash
python tools/validation/operators/check_operator.py   --op logp   --candidate pytorch   --device cpu   --dtype fp32   --batch 1   --seq 2   --vocab 17
```

CUDA candidate check against the PyTorch gold path:

```bash
python tools/validation/operators/check_operator.py   --op logp   --candidate cuda   --device cuda   --dtype bf16   --arch-key sm90   --batch 1   --seq 1   --vocab 4096
```

JSON report:

```bash
python tools/validation/operators/check_operator.py   --op logp   --candidate pytorch   --device cpu   --dtype fp32   --batch 1   --seq 2   --vocab 17   --json
```

Supported key options:

```text
--op              Operator name. The minimal version supports logp and linear_logp.
--candidate       Candidate backend, for example pytorch, cuda, cuda-generic, cuda-sm90, triton.
--dtype           fp32, bf16, or fp16.
--device          auto, cpu, cuda, or another torch device string.
--arch-key        Optional tolerance override key such as sm90.
--batch           Batch size.
--seq             Sequence length.
--vocab           Vocabulary size.
--input-mode      random or constant.
--constant-value  Floating-point value for constant mode.
--token-value     Token id for constant mode, reduced modulo vocab.
--seed            Random seed for reproducible random inputs.
--check-grad      Also compare gradients for inputs declared by the operator spec.
--grad-mode       Upstream gradient mode for --check-grad: random by default; ones for smoke tests.
--grad-seed       Random upstream gradient seed for --grad-mode random.
--json            Print the full structured report as JSON.
```

Example output:

```text
suite=logp passed=True pass_rate=1.0000
candidate=cuda-logp backend=cuda passed=True pass_rate=1.0000
  case=logp-torch.bfloat16-1x1x4096 output=0 shape=(1, 1) dtype=torch.bfloat16 max_abs=2.69813538e-02 mean_abs=2.69813538e-02 max_rel=3.03093810e-03 tol=(atol=5.000e-02, rtol=0.000e+00) passed=True
```

## Adding a New Operator

To add a new operator, keep the shared checker flow unchanged. Add only operator-specific inputs, specs, and tests.

### 1. Add Input Generation

File:

```text
rl_engine/validation/operators/operator_inputs.py
```

Update `make_operator_inputs()`:

```python
builders = {
    ...
    "new_op": _make_new_op_inputs,
}
```

Update `operator_shape_name()`:

```python
names = {
    ...
    "new_op": f"{batch}x{seq}x...",
}
```

Add the input builder:

```python
def _make_new_op_inputs(args, dtype, device):
    batch, seq = _batch_seq(args)
    return {
        "x": _floating_tensor((batch, seq, ...), args, dtype, device, offset=0),
    }
```

Rules:

- Inputs should represent the operator's standard semantic interface.
- Do not generate backend-specific flattened inputs here.
- Support deterministic random inputs and constant inputs where practical.

### 2. Register Gold and Candidates

File:

```text
rl_engine/validation/operators/operator_specs.py
```

Add an `OperatorSpec` entry:

```python
"new_op": OperatorSpec(
    name="new_op",
    op_class="elementwise",
    gold_path="rl_engine.reference....NativeNewOp",
    candidate_paths={
        "pytorch": "rl_engine.reference....NativeNewOp",
        "cuda": "rl_engine.backends.cuda....CudaNewOp",
        "triton": "rl_engine.backends.shared.triton....TritonNewOp",
    },
)
```

Rules:

- `gold_path` must come from `rl_engine.reference`.
- Backend implementations are candidates only.
- `candidate=pytorch` is for checker smoke tests only.
- Do not compare operators with different math.

### 3. Update Tolerances If Needed

File:

```text
rl_engine/contracts/profiles/precision/ws1.json
```

Reuse an existing class when possible:

```text
elementwise
reduction
logprob
```

If a new class is needed, add dtype tolerances and set `op_class` accordingly in `operator_specs.py`.

### 4. Add Tests

Files:

```text
tests/validation/operators/test_operator_inputs.py
tests/validation/operators/test_op_checks.py
```

Minimum expected coverage:

- Add the operator to the `test_operator_inputs_support_all_issue_108_ops` parametrized list.
- Add a PyTorch-vs-PyTorch smoke case if the operator adds new runner behavior.
- Add a bad-candidate case if the operator introduces new comparison behavior.

### 5. Validate

```bash
python -m pytest tests/contracts/test_tolerance_contract.py tests/validation/operators/test_op_checks.py tests/validation/operators/test_operator_inputs.py -q
```

Then run the CLI:

```bash
python tools/validation/operators/check_operator.py   --op new_op   --candidate pytorch   --device cpu   --dtype fp32
```

For CUDA:

```bash
python tools/validation/operators/check_operator.py   --op new_op   --candidate cuda   --device cuda   --dtype bf16   --arch-key sm90
```

## CUDA Validation Notes

H100 environment observed during development:

```text
GPU: NVIDIA H100 80GB HBM3
Driver: 580.95.05
CUDA driver capability: 13.0
nvcc: 13.0
torch: 2.12.0+cu130
compute capability: (9, 0)
```

Generic CUDA `logp` passed on H100 for vocab sizes 256, 512, 1024, 2048, and 4096 with bf16 inputs under the current tolerance contract.

SM90 fused logp is not marked as a passing path in this PR. It compiled and loaded in some experiments, but runtime failures and accuracy failures were observed separately. Treat SM90 fused logp as a separate CUDA kernel validation task unless `check_operator.py` reports `passed=True` for the target case.

## Validation Performed

```bash
python -m pytest tests/contracts/test_tolerance_contract.py tests/validation/operators/test_op_checks.py tests/validation/operators/test_operator_inputs.py -q
```

CPU CLI smoke test:

```bash
python tools/validation/operators/check_operator.py   --op logp   --candidate pytorch   --device cpu   --dtype fp32   --batch 1   --seq 2   --vocab 17
```

Backward CLI smoke test:

```bash
python tools/validation/operators/check_operator.py   --op logp   --candidate pytorch   --device cpu   --dtype fp32   --batch 1   --seq 2   --vocab 17   --check-grad
```

## PR Review Updates

### LogP Gradient Coverage

Files:

```text
tests/ops/logprob/test_logp.py
docs/contributing/issue-108-session-log.md
```

Change:

- Added a forward-gradient test for `NativeLogpOp.forward_fp32`.

Reasoning:

- The checker PR already validates forward output values, but review feedback called out that logprob coverage should also prove gradient propagation and batch invariance.
- The new gradient test compares the op gradient against a direct PyTorch `log_softmax + gather` reference under a non-unit upstream gradient.
- Batch invariance was already covered by `TestNativeLogpOpBatchInvariance` in `tests/ops/logprob/test_logp.py`, so no duplicate batch-invariance test was added.

### GTest Backward Check Support

Files:

```text
rl_engine/validation/operators/op_checks.py
rl_engine/validation/operators/operator_specs.py
tools/validation/operators/check_operator.py
tests/validation/operators/test_op_checks.py
docs/contributing/issue-108-session-log.md
```

Change:

- Added `OperatorCase.grad_input_names` to declare which inputs should be checked for gradients.
- Added `run_operator_suite(..., check_grad=True)`.
- Added `_run_case_backward()` to compare candidate forward outputs and selected input gradients against the PyTorch gold path.
- Added `OperatorSpec.grad_input_names`; `logp` declares `("logits",)`.
- Added `tools/validation/operators/check_operator.py --check-grad`.
- Added `--grad-mode ones|random` and `--grad-seed` for backward checks.

Reasoning:

- Forward-only checks can miss incorrect or disconnected backward paths.
- Gradient inputs must be declared per operator because not every floating tensor should receive gradients.
- Input generation remains independent of autograd; the runner clones inputs and enables `requires_grad` only inside the backward check path.
- `grad_mode=random` is the default and catches backward bugs hidden by all-one upstream gradients.
- `grad_mode=ones` remains available for quick smoke tests and preserves the old `output.sum().backward()` behavior.

Known backend limitation:

- `cuda` `logp` currently calls the compiled `_C.fused_logp` forward path directly and does not produce an autograd-connected output.
- Running `--check-grad` against `candidate=cuda` fails with a missing `grad_fn`; this is a backend implementation gap, not a tolerance issue.
- To support `cuda logp` backward, the backend must add or wrap a real backward path, usually via `torch.autograd.Function` or an explicit CUDA backward kernel.

### Linear LogP Triton GTest Smoke Support

Files:

```text
rl_engine/validation/operators/operator_inputs.py
rl_engine/validation/operators/operator_specs.py
tests/validation/operators/test_operator_inputs.py
docs/contributing/issue-108-session-log.md
```

Change:

- Added `linear_logp` input construction with `hidden`, `lm_head_weight`, `target_ids`, and `bias=None`.
- Added a `linear_logp` operator spec using `NativeLinearLogpOp.apply` as the PyTorch gold path.
- Added `triton` as a `linear_logp` candidate backend.
- Declared `("hidden", "lm_head_weight")` as gradient inputs for backward checks.

Reasoning:

- `linear_logp` is the smallest real fused op in the repository with an implemented Triton backward path.
- The first gtest integration keeps bias disabled to avoid optional-gradient handling in the initial smoke path.
- The op reuses the `logprob` tolerance class because it produces selected-token log probabilities.
- It gives the checker a real non-PyTorch differentiable candidate for end-to-end forward/backward reporting.

Example Triton smoke command:

```bash
python tools/validation/operators/check_operator.py   --op linear_logp   --candidate triton   --device cuda   --dtype bf16   --batch 1   --seq 2   --vocab 1024   --normalized-dim 4096   --check-grad   --grad-mode random   --grad-seed 123
```

Observed bf16 result on H100:

```text
suite=linear_logp passed=False pass_rate=0.0000
candidate=triton-linear_logp backend=triton passed=False pass_rate=0.6667
  case=linear_logp-torch.bfloat16-1x2x4096x1024 output=0 shape=(1, 2) dtype=torch.float32 max_abs=4.76226807e-01 mean_abs=4.74334717e-01 max_rel=2.96434597e-03 tol=(atol=5.000e-02, rtol=0.000e+00) passed=False
  case=linear_logp-torch.bfloat16-1x2x4096x1024 output=1 gradient:hidden shape=(1, 2, 4096) dtype=torch.bfloat16 max_abs=0.00000000e+00 mean_abs=0.00000000e+00 max_rel=0.00000000e+00 tol=(atol=5.000e-02, rtol=0.000e+00) passed=True
  case=linear_logp-torch.bfloat16-1x2x4096x1024 output=2 gradient:lm_head_weight shape=(1024, 4096) dtype=torch.bfloat16 max_abs=0.00000000e+00 mean_abs=0.00000000e+00 max_rel=0.00000000e+00 tol=(atol=5.000e-02, rtol=0.000e+00) passed=True
```

Interpretation:

- The checker flow is complete: CLI parsing, input construction, PyTorch gold loading, Triton candidate loading, forward execution, backward execution, gradient collection, comparison, and report formatting all ran successfully.
- The suite is intentionally not marked as passing because the bf16 forward output exceeded the current `logprob` absolute tolerance.
- The current tolerance uses `atol=5.0e-2` and `rtol=0.0`; the observed forward absolute error is about `4.76e-1`, while the relative error is about `2.96e-3`.
- `linear_logp` includes a large bf16 matrix multiply before selected-token logprob, so it likely needs an operator-specific gold policy or tolerance rather than reusing plain `logprob` tolerances unchanged.
- In this case, both checked gradients passed, so the failure is a forward accuracy/tolerance calibration issue, not a failure to execute the backward checker path.
