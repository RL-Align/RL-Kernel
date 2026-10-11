# gtest usage guide (operator candidate vs gold)

> **Audience:** contributors implementing train–inference / batch-invariant operators
> **Entry point:** `tools/validation/operators/check_operator.py` + `rl_engine/kernels/gtest/*`
> **Numerical SSOT:** [#267](https://github.com/RL-Align/RL-Kernel/issues/267) four-judgment contract

This is the official how-to for the gtest harness: register an op, build inputs, run the CLI for forward/backward checks, and obtain tolerances from the shared contract (not private `atol`/`rtol`).

---

## 1. What gtest is for

gtest validates a **single operator**:

| Capability | Meaning |
|------------|---------|
| Gold | Usually a PyTorch / `forward_fp32` reference path |
| Candidate | CUDA / Triton / arch-specific implementation |
| Forward check | Outputs within contract tolerance (`forward_accuracy`) |
| Backward check | Selected input gradients within contract tolerance (`gradient_accuracy`, **independent of forward**) |

It is **not**:

- The full Qwen3-8B model-level gate (#266 C9/C10)
- The final cross-config invariance harness (C3/C4 build on the same contract)
- Real vLLM vs Megatron engine alignment

The CLI primarily covers **accuracy** (candidate vs gold).
**Invariance** (bitwise across configs) and **train/infer aggregates** use the contract APIs / later harnesses—do not invent private gate thresholds in tests.

---

## 2. End-to-end flow

```text
1) (Optional) register the op in the runtime registry
        ↓
2) gtest/operator_specs.py  → OP_SPECS: gold + candidates
        ↓
3) gtest/operator_inputs.py → build input shapes / values
        ↓
4) tools/validation/operators/check_operator.py → run suite, load tolerance_contract.json
        ↓
5) report max_abs / tol / passed
```

### 2.1 Key files

| Path | Role |
|------|------|
| `rl_engine/validation/operators/operator_specs.py` | `OP_SPECS`: name, `op_class`, gold, candidates, grad inputs |
| `rl_engine/validation/operators/operator_inputs.py` | Default Qwen3-8B dims + `make_operator_inputs` |
| `rl_engine/validation/operators/op_checks.py` | Suite execution and comparison |
| `rl_engine/contracts/profiles/precision/ws1.json` | Numerical contract SSOT |
| `rl_engine/contracts/numerical.py` | `load_contract` / `resolve_tolerance` / chain aggregates |
| `tools/validation/operators/check_operator.py` | **CLI entry** (accuracy) |
| `rl_engine/validation/operators/gradient_invariance.py` | C4 gradient invariance API |
| `rl_engine/validation/operators/gradient_adapters.py` | C4 enumerable adapters + status matrix |
| `rl_engine/validation/operators/elementwise_inventory.py` | C5 elementwise / RoPE inventory |
| `rl_engine/validation/operators/four_judgment_matrix.py` | C8 four-judgment matrix schema |
| `tools/validation/operators/check_gradient_invariance.py` | C4 GPU evidence CLI |
| `rl_engine/validation/operators/kv_consistency.py` | C6/C7 decode–prefill + stateful KV |
| `rl_engine/models/qwen3/reference.py` | C9 full Qwen3-8B Dense BI model |
| `rl_engine/validation/models/chain_gate.py` | C10 model-level #150 + train/infer gate |
| `tools/validation/operators/check_decode_prefill.py` | C6 GPU CLI |
| `tools/validation/operators/check_stateful_kv.py` | C7 GPU CLI |
| `tools/validation/models/ws1_chain_fwd_bwd.py` | C9 one-command fwd+bwd (assembly only) |
| `tools/validation/models/ws1_chain_gate.py` | C10/C11 full-model required gate |

---

## 3. Step 1: register the op in `OP_SPECS`

Edit `rl_engine/validation/operators/operator_specs.py` and add an entry to `OP_SPECS`. Example shape (logp / linear_logp):

```python
"logp": OperatorSpec(
    name="logp",
    op_class="logprob",          # selects the contract op_class row
    gold_path="rl_engine.reference.logprob.logp.NativeLogpOp",
    gold_method="forward_fp32",  # method invoked on the gold instance
    candidate_paths={
        "pytorch": "rl_engine.reference.logprob.logp.NativeLogpOp",
        "cuda": "rl_engine.backends.cuda.logprob.logp.FusedLogpGenericOp",
        "cuda-sm90": "rl_engine.backends.cuda.logprob.logp.FusedLogpSM90Op",
    },
    grad_input_names=("logits",),  # inputs compared under --check-grad
),
```

### 3.1 `OperatorSpec` fields

| Field | Meaning |
|-------|---------|
| `name` | Value for CLI `--op` |
| `op_class` | Contract class: `elementwise` / `reduction` / `logprob` / `attention` |
| `gold_path` | Gold class path `module.Class` |
| `gold_method` | Method name, e.g. `forward_fp32`, `apply`, `__call__` |
| `candidate_paths` | Map `candidate name → implementation class`; CLI `--candidate cuda` looks up this map |
| `grad_input_names` | With `--check-grad`, enable grads and compare these inputs; missing config errors |

**Only ops registered in `OP_SPECS` can be invoked via `check_operator.py`.**

Currently registered (source of truth is the code):

```text
rms_norm, qk_norm, attention, logp, linear_logp, embedding, lm_head,
det_gemm, rope, silu, swiglu, batch_invariant_logp, pack
```

`qk_norm` is a first-class `OP_SPECS` key that reuses the RMSNorm kernels on
`head_dim` (per-head), not the full hidden width.

`pack` is a WS1 layout helper, not a per-profile CUDA/Triton kernel. It is in
`OP_SPECS` so `check_operator.py --op pack` and the C3/C4 CPU adapters can
prove logical packing/unpacking. C8 marks every pack cell **N/A** with that
C2/C4 reason.

`linear_logp` is registered for the CLI but is **not** a WS1 required chain
node. C2 status is `optional_fused_path`; C4 is `optional_fused`; C8 does not
require a four-judgment row.

---

## 4. Step 2: build inputs

File: `rl_engine/validation/operators/operator_inputs.py`.

### 4.1 Default model dims (Qwen3-8B Dense semantics)

Macros at the top of the file (local experiments may change them; WS1 full-model EXIT uses the official config fingerprint):

```text
DEFAULT_HIDDEN       = 4096
DEFAULT_N_HEADS      = 32
DEFAULT_N_KV_HEADS   = 8
DEFAULT_HEAD_DIM     = 128
DEFAULT_INTERMEDIATE = 12288
DEFAULT_VOCAB        = 151936
DEFAULT_ROPE_THETA   = 1.0e6
DEFAULT_RMS_EPS      = 1.0e-6
```

### 4.2 Shape names and input builders

- `operator_shape_name(op_name, args)` — human-readable case name (e.g. `2x16x257`)
- `_make_*_inputs` / `make_operator_inputs` — build the input dict from `--op` and CLI args
  - `random`: reproducible randomness (`--seed` plus per-tensor offsets)
  - `constant`: fixed values for debugging (`--constant-value` / `--token-value`)

When adding an op: extend the shape map and implement the matching `_make_xxx_inputs`.

### 4.3 Suggested GRPO-oriented shapes (local sweeps)

For GRPO, `B = P × G`. With `G=8`, batch is often a multiple of 8.
`B=1` is fine for smoke; fuller sweeps may use:

```text
B ∈ {1, 8, 16, 32, 64}
S ∈ {1, 31, 33, 127, 129, 255, 256, 257, 512, 1024, 4096, 8192}
```

Prefer short `S` when VRAM is tight; full-model gates are owned by #266 / C2.

---

## 8. C6–C11 closeout commands

C6 (direct decode, both profiles; chunked-prefill is not a substitute):

```bash
python tools/validation/operators/check_decode_prefill.py --backend-profile cuda_bf16
python tools/validation/operators/check_decode_prefill.py --backend-profile triton_cuda_bf16
```

C7 (B1 stateful allocate→write→read→decode + generate-rescore). Concat-only
`NativeKVCacheAttnOp` is not B1. B2 is explicitly `absent`.

```bash
python tools/validation/operators/check_stateful_kv.py --backend-profile cuda_bf16
python tools/validation/operators/check_stateful_kv.py --backend-profile triton_cuda_bf16
```

C9 (assembly only; not EXIT). Official 36-layer Qwen3-8B Dense, pinned weights:

```bash
python tools/weights/prepare_ws1_weights.py --output "$QWEN3_8B" --verify-only
python tools/validation/models/ws1_chain_fwd_bwd.py --backend-profile cuda_bf16 --weights hf --weights-path $QWEN3_8B
python tools/validation/models/ws1_chain_fwd_bwd.py --backend-profile triton_cuda_bf16 --weights hf --weights-path $QWEN3_8B
```

C10/C11 required full-model gate (H20; no skip / xfail / synthetic-as-pass):

```bash
python tools/validation/models/ws1_chain_gate.py --backend-profile cuda_bf16 --model qwen3-8b-dense --dtype bfloat16 --weights required --weights-path $QWEN3_8B --json
python tools/validation/models/ws1_chain_gate.py --backend-profile triton_cuda_bf16 --model qwen3-8b-dense --dtype bfloat16 --weights required --weights-path $QWEN3_8B --json
```

Omitting `--seed` uses the manifest-pinned execution seed. A supplied seed is
applied to both PyTorch and CUDA and recorded separately from `workload_seed`.
The weight loader verifies the pinned index SHA-256, every shard size/SHA-256,
and the aggregate content hash before allocating the 8B model.

C10 compares `tensor.grad` after a real training-style backward over every
official Qwen3-8B Dense trainable leaf
(`gradient_scope=all_required_trainable_parameters`,
`all_parameter_gradients=true`). Logprob accuracy vs FP32 gold uses only
`max_abs_dlogp` / `approx_kl0` / `clipfrac0`. The JSON also records GPU name,
representative `case_id`s, workflow URL, C8 evidence path, and backward
runtime kernel identities.

To keep the full-model gate within Hopper device memory, leaf-gradient snapshots
are transferred to CPU without FP32 expansion and released after comparison. The
The chain GPU job writes C8 outside the checkout; the artifact validator requires clean C8 evidence from the exact C10 commit and
explicit packed-versus-FP32 forward and gradient accuracy rows.

---

## 5. Step 3: run the CLI

```bash
# From the repo root; prefer an editable install: pip install -e .
python tools/validation/operators/check_operator.py --op logp --candidate pytorch --device cpu --dtype fp32 --batch 1 --seq 2 --vocab 17
```

### 5.1 Common examples

**Smoke (CPU / PyTorch self-check)**

```bash
python tools/validation/operators/check_operator.py \
  --op logp --candidate pytorch --device cpu --dtype fp32 \
  --batch 1 --seq 2 --vocab 17
```

**Triton `linear_logp` + backward (BF16)**

```bash
python tools/validation/operators/check_operator.py \
  --op linear_logp --candidate triton --device cuda --dtype bf16 \
  --batch 1 --seq 2 --vocab 1024 --normalized-dim 4096 \
  --check-grad
```

**CUDA deterministic attention + gradients**

```bash
python tools/validation/operators/check_operator.py \
  --op attention --candidate cuda --device cuda --dtype bf16 \
  --batch 2 --seq 64 --check-grad --grad-mode random
```

**Full JSON report**

```bash
python tools/validation/operators/check_operator.py --op rms_norm --candidate cuda --dtype bf16 --device cuda --json
```

### 5.2 CLI flags

| Flag | Meaning |
|------|---------|
| `--op` | Operator name from `OP_SPECS` |
| `--candidate` | Backend: `pytorch` / `cuda` / `cuda-generic` / `cuda-sm90` / `triton` / … (see that op’s `candidate_paths`) |
| `--dtype` | `fp32` / `bf16` / `fp16`; selects input dtype and contract row |
| `--device` | `auto` / `cpu` / `cuda` |
| `--batch` / `--seq` | Batch size and sequence length for inputs |
| `--vocab` | Vocab size; logp logits `[B,S,V]`; linear_logp weight `[V,H]` |
| `--input-mode` | `random` (default) or `constant` |
| `--constant-value` | Float fill in constant mode |
| `--token-value` | Token id in constant mode |
| `--normalized-dim` | Hidden dim for rms_norm / linear_logp, etc. |
| `--k-dim` / `--n-dim` | Matmul / det_gemm dims |
| `--theta` | RoPE theta |
| `--eps` | RMSNorm epsilon |
| `--seed` | Input RNG seed (per-tensor offsets still apply) |
| `--arch-key` | Arch override key, e.g. `sm90` (contract `arch_overrides`) |
| `--check-grad` | Also compare gradients (requires `grad_input_names`) |
| `--grad-mode` | `random` (default, stricter) / `ones` (≈ `output.sum().backward()`) |
| `--grad-seed` | Seed for random upstream gradients |
| `--json` | Print the full structured report |

---

## 6. Where tolerances come from (after #267)

### 6.1 Before vs after C1

| Before | After (C1 / #267) |
|--------|-------------------|
| Mostly `accuracy[op_class][dtype]` | **Four judgments**: forward/gradient × accuracy/invariance |
| Forward and grad often shared one tol | **Grad uses `gradient_accuracy` only** (no silent forward inheritance) |
| Flat threshold table | Plus dtype policy, comparison roles, chain logprob aggregates |

### 6.2 Which judgments the CLI / `op_checks` use

`run_operator_suite` / `check_operator.py`:

| Comparison | Judgment |
|------------|----------|
| Output vs gold | `forward_accuracy` |
| Gradient vs gold | `gradient_accuracy` |

Batch/chunk **bitwise invariance** and train/infer **three aggregates** are not separate `check_operator.py` switches. Use C3/C4. C3 now runs the same enumerable WS1 ops as C4 (`make_forward_runner`):

```python
from rl_engine.validation.operators import (
    assert_forward_batch_invariant,
    assert_gradient_batch_invariant,
)
from rl_engine.validation.operators.gradient_adapters import get_adapter

# C4: training-style gradient accuracy + invariance (thresholds from C1 only)
adapter = get_adapter("rms_norm")
report = assert_gradient_batch_invariant(
    op,
    contract=contract,
    backend_profile="cuda_bf16",
    provenance=provenance,
    gold_fn=gold_fn,
    grad_tensors=adapter.tensors,
    op_class=adapter.op_class,
)
```

`max_abs_dlogp`, `approx_kl0`, and `clipfrac0` are the **sole** chain-level
logprob / ablation aggregates. The three aggregates judge **outputs only**.
Gradient pass/fail uses only independent `gradient_accuracy` /
`gradient_invariance` verdicts. GPU evidence:

```bash
python tools/validation/operators/check_gradient_invariance.py \
  --op rms_norm --candidate cuda --backend-profile cuda_bf16
```

C4 does not claim the full-model C10 gate. Use:

```python
from rl_engine.contracts.numerical import (
    load_contract,
    resolve_tolerance,
    compute_logprob_aggregates,
    judge_logprob_aggregates,
    default_clip_interval,
)

contract = load_contract()
# Cross-config invariance (gate path)
inv = resolve_tolerance(
    contract,
    judgment="forward_invariance",  # or gradient_invariance
    op_class="attention",
    dtype="bfloat16",
    backend_profile="cuda_bf16",
)
# inv.mode == "bitwise", inv.atol == inv.rtol == 0

# Train vs infer selected-logprob
agg = compute_logprob_aggregates(
    train_logp,
    rollout_logp,
    active_mask,
    contract=contract,
    report_kind="train_infer_logprob_parity",
    clip_interval=default_clip_interval(contract),
    comparison_lhs_role="training_style_teacher_forcing",
    comparison_rhs_role="inference_style_rollout_decode",
)
verdict = judge_logprob_aggregates(agg, contract, execution_dtype="bfloat16")
```

### 6.3 Policy locks (WS1)

| Item | Value |
|------|--------|
| Execution | BF16 mandatory for EXIT (CLI may still exercise fp32/fp16) |
| Reference / accumulation | FP32 |
| FP8 | Out of scope (resolve hard-fails) |
| TF32 | Disabled |
| Profiles | `cuda_bf16` and `triton_cuda_bf16` share **the same** thresholds |

WS1 evidence must attach checked provenance to its candidate report:

```python
from rl_engine.validation.operators import BackendProvenance, CandidateSpec

provenance = BackendProvenance(
    backend_profile="cuda_bf16",  # use triton_cuda_bf16 + triton for Triton
    requested_backend="cuda",
    actual_backend="cuda",
    execution_dtype="bfloat16",
    accumulation_dtype="float32",
    output_dtype="bfloat16",
    reference_dtype="float32",
    candidate_tf32_enabled=False,
    reference_tf32_enabled=False,
)
candidate = CandidateSpec(
    name="cuda-candidate",
    backend="cuda",
    fn=op,
    provenance=provenance,
)
```

The suite rejects backend fallback, dtype drift, TF32 enablement, and observed output
dtypes that disagree with this provenance before producing a passing report.

`check_operator.py` is a local debugging CLI and does not construct provenance on its
own. A WS1 gate must create `CandidateSpec(..., provenance=provenance)` in its harness;
use `--json` to retain the resolved judgment and comparison-role fields in CLI reports.

**Do not** use private `atol=1e-5` (etc.) as WS1 gate evidence. Migrate a gate to
the shared resolver before using it as WS1 evidence.

### 6.4 Report `tol=(atol=..., rtol=...)`

The CLI summary line:

```text
tol=(atol=..., rtol=...)
```

comes from the shared resolver—not hard-coded constants inside `check_operator.py`.

---

## 7. Recommended local test order

```text
1. --candidate pytorch --device cpu --dtype fp32
   → registration / inputs / plumbing smoke

2. Same shape with --dtype bf16 --device cuda --candidate triton|cuda
   → real candidate forward

3. Add --check-grad --grad-mode random
   → gradients (random upstream grads catch more bugs than ones)

4. --arch-key sm90 only when you need arch-specific contract overrides

5. Cross batch/layout: C3 `check_forward_invariance.py` / C4 `check_gradient_invariance.py`
```

---

## 8. Common failures

| Symptom | Likely cause |
|---------|----------------|
| Unsupported / missing `--op` choice | Not registered in `OP_SPECS` |
| `--check-grad` missing grad inputs | Empty/wrong `grad_input_names` vs input keys |
| Candidate import error | Bad `candidate_paths` or extension not built |
| BF16 over tolerance | Confirm gold is `forward_fp32`; check contract row; do not loosen private atol |
| Missing SM90 symbols | Build without SM90 / non-sm90 GPU; pick another candidate or rebuild |
| Want FP8 | Hard-fail under WS1 contract; out of scope |

---

## 9. Relationship to pytest

| Path | Use |
|------|-----|
| `python tools/validation/operators/check_operator.py ...` | Fast single-op shape/debug loops |
| `pytest tests/test_*.py` | Regression, invariance, integration |
| `pytest tests/contracts/test_tolerance_contract.py` | Contract schema / resolver |

Both paths should take thresholds from `tolerance_contract.json`.
New pytest code should call `resolve_tolerance` instead of copying magic numbers.

---

## 10. Minimal checklist for a new operator

- [ ] Implementation under `rl_engine/kernels/ops/{pytorch,cuda,triton}/...`
- [ ] (Optional) runtime `registry` registration
- [ ] `OP_SPECS` entry: gold + candidates + `op_class` + `grad_input_names`
- [ ] `operator_inputs` shape name + input builder
- [ ] `check_operator.py` smoke + bf16 + `--check-grad` green
- [ ] Contract already has the `op_class` row (extend schema + `test_tolerance_contract` if not)
- [ ] No new private `atol`/`rtol` as gate evidence
- [ ] Operator docs point at the contract for thresholds (do not restate ad-hoc numbers)

---

## 11. Further reading

| Doc | Content |
|-----|---------|
| [testing.md](testing.md) | Short testing entry points |
| Issues [#266](https://github.com/RL-Align/RL-Kernel/issues/266) / [#267](https://github.com/RL-Align/RL-Kernel/issues/267) | WS1 closeout and C1 contract |

---

## 12. Changelog

| Date | Notes |
|------|--------|
| 2026-08-11 | Initial English guide aligned with C1; documents CLI, `OP_SPECS`, inputs, and contract usage |
| 2026-08-13 | Document C4 `assert_gradient_batch_invariant` and `check_gradient_invariance.py` |
| 2026-08-13 | C4 adapters run on `config.physical_layout` (packed / chunked / padded / permuted) and return physical tensors restored through the C2 map; a new adapter must vary with the layout or its bitwise verdicts are tautologies |
| 2026-08-13 | C3 `check_forward_invariance.py` / `make_forward_runner` cover every C2 required chain op plus pack, not only logp |
| 2026-08-13 | C5 inventory + C8 `sweep_ws1_four_judgments.py`; C2 v5 adds remaining operator case_ids |
