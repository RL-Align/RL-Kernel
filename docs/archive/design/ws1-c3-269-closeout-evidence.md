# WS1 C3 (#269) closeout evidence

**Parent:** #266 · **Depends on:** #267 / #268 · **Scope:** shared forward harness only

## Acceptance map

| #269 criterion | Evidence |
| --- | --- |
| Accuracy and invariance separate | `ForwardInvarianceReport.accuracy_reports` and `invariance_reports` |
| Batch/chunk bitwise after logical unpadding | C1 `forward_invariance` resolver plus exact C2 logical-key validation |
| C2 transforms | `build_config_matrix`: fixed 2×2 matrix, permutation, packing, left/right padding |
| Diagnostics | tensor name, config pair, max/mean absolute error, max relative error |
| Backend provenance | profile, requested/actual backend, candidate/kernel id, device, CC, dtype, seed, fallback reason |
| Silent/cross-profile fallback | missing or mismatched provenance fails; CLI rejects candidate/profile mismatch |
| Selected-logprob smoke | C1 `max_abs_dlogp`, `approx_kl0`, and `clipfrac0` verdict |
| CUDA and Triton same schema | one API/CLI/report schema; both profile contracts are parametrically tested |
| No private thresholds | all tensor and aggregate thresholds resolve through the C1 contract |

CPU-safe contract regression:

```bash
python -m pytest -q \
  tests/contracts/test_tolerance_contract.py \
  tests/validation/operators/test_ws1_workload.py \
  tests/validation/operators/test_forward_invariance.py \
  tests/validation/operators/test_op_checks.py
```

Required-profile runtime examples (must run on CUDA hardware and must not be skipped):

```bash
python tools/validation/operators/check_forward_invariance.py \
  --op logp --candidate cuda \
  --backend-profile cuda_bf16 --json

python tools/validation/operators/check_forward_invariance.py \
  --op batch_invariant_logp --candidate triton \
  --backend-profile triton_cuda_bf16 --json
```

The CLI exits red when CUDA is unavailable, a candidate is absent, the C2 node is
`missing_required`, the compute capability cannot run a declared SM90 candidate, provenance
does not match the profile, or any accuracy/invariance/logprob verdict fails.

## Runtime verification

Verified on NVIDIA GeForce RTX 3060 Laptop GPU (`sm86`) with PyTorch 2.8.0+cu128:

| Gate | Result |
| --- | --- |
| Full pytest suite | `1524 passed, 121 skipped` |
| Full pre-commit | trailing whitespace, EOF, YAML, large-file, black, isort, flake8 passed |
| `cuda_bf16` / generic CUDA logp C3 matrix | passed; all invariance max-abs errors `0.0` |
| `triton_cuda_bf16` / Triton batch-invariant-logp C3 matrix | passed; all invariance max-abs errors `0.0` |
| CUDA operator accuracy check | passed, max absolute error `0.0287590` |
| Triton operator accuracy check | passed, max absolute error `9.536743e-07` |

The CUDA profile uses the manifest-declared generic CUDA logp candidate on SM86. No SM90
candidate or fallback path is claimed on this device.

## Parent boundary

This closes only C3. It supplies the report and canonicalization contract that C10 must reuse.
It does not claim the full-model, backward, KV-cache, or CI EXIT requirements of #266.
