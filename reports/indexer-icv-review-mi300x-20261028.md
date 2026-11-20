# Item 16 review follow-up: MI300X, 2026-10-28

This updates PR #599; target remains `dsv4-rocm`. Numerical GPU work still uses
aiter decode/GEMMs and vLLM FP8 normalization. No new numerical kernel was added.

## Changes

- Explicit `borrowed` versus `snapshot` binding in both adapters. Borrowed storage
  is a caller-owned no-write lifetime contract, not immutable storage proven by
  `_version`. Snapshots isolate original `.data`, independently shared storage,
  and NumPy aliases, and are validated after copying. Inference tensors snapshot.
- Missing/None/unreadable provider source fails with `IndexerICVError`, rather
  than a `Path(None)` TypeError or an invented provenance identity.
- Explicit magnitude checks for both E2M1 nibbles preserve both signs of zero.
  Exhaustive 256-byte tests cover special finite scales and NaN scale rejection.
- FP8 oracle now preserves separate FP32 per-head/per-row multipliers and applies
  their product once after the dot. Earlier reference code DID apply scales
  before the dot; it did not omit scales or use exponent bytes. The new tests
  distinguish this ABI from MXFP4 E8M0 bytes, including boundary exponents.

## Verification

On AMD Instinct MI300X VF (gfx942), ROCm/HIP 7.14.60850, PyTorch
2.12.0+rocm7.14.0a20260608, Triton 3.7.0:

- Full contract + DeepSeek-V4 suite: **382 passed**, 0 skipped, 0 failures,
  49.79 seconds with coverage; independent repeat **382 passed**, 42.50 seconds.
- Includes 113 CPU-safe item-16 tests, 127 GPU/process/transport tests and
  142 existing regression tests. Both runs report 15 existing warnings.
- Wrapper coverage: **365/365 statements, 90/90 branches**. This is Python
  contract/adapter/backend coverage, NOT native GPU kernel instruction coverage.
- Ruff lint and formatting checks pass for all seven changed Python files.
- Re-ran benchmark with 30 repetitions and updated provider fingerprints.
  Example M=8,H=64,N=512,D=128 MXFP4: prepared aiter dot 42.82 us,
  materialized pipeline 149.86 us, bounded stream 199.60 us. Predecoded FP32
  torch.einsum reference 24.33 us. Different arithmetic; no speedup claim.

The original acceptance limitations still apply: no full model, GPU CP/RCCL,
Graph, backward or multi-node acceptance is claimed by this forward-only item.
Snapshot copying has additional source-sized allocation and bandwidth costs;
benchmark numbers above use the default borrowed policy.
