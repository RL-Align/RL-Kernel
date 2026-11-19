# Proposed title

[RFC #642][Task 12] Fix projection GEMM FP32 outputs and qualify Hopper invariance

# Proposed description

CUDA `DetGemmOp.forward_fp32()` currently raises because its native symbol is
not exported; its C++ implementation also selects BF16 output. Export the
existing forward entry points and select the existing FP32 store, preserving
all BF16 K-tree rounding points. Document the separate full-FP32 accumulation
profile and restore Triton forward tests that were silently omitted by a stale
import. No backward implementation or new numerical kernel is introduced.

Add forward qualification for ten projection K/N pairs derived from the pinned
Qwen3.6-35B-A3B config and official call chain, including narrow GDN/gate outputs
and the full vocabulary. Check independent FP64 accuracy, original output bits
across single-row/batch/chunk execution, dtype/casts, layout/alignment and K/M
boundaries. Independent CPU FP64 dyadic goldens distinguish the SM90 tile tree
from the scalar/Triton element tree for non-power-of-two K.

Target `main`: these shared components are independently testable without model
assembly or miles integration, following the maintainer's exception in #642.
Base: `49941801f942428fd52ac5b66a1b0c070418724e`. Coordinate the small import fix
with #450; leave #679's scalar recursion/index fix and Task 13 gradients separate.

Validation on one H100 80GB HBM3, CUDA 13.4, PyTorch
2.14.0a0+4fdf77b940.nv26.08, Triton 3.8.0+git4c7e62b7.nv26.8:

- Original binary: four new cast regressions fail on the missing binding.
- Projection matrix: 158 passed, no skips/failures. FP64 maximum absolute
  error 0.037069 passes unchanged repository limits; raw-bit comparisons have
  zero mismatches. Twenty pytest XML property-format warnings are documented.
- Independent K-tree goldens and selected existing CUDA/Triton forward tests:
  38 passed, 65 deselected. Portable-reference check: 6 passed.
- Original/repaired binaries: 26 output comparisons, zero differing bytes.
- CPU layout: 33 passed, one extension-smoke skip. Attention projection: 8 passed.
- Black, isort, flake8, focused mypy and diff checks pass. Profiler confirms
  both the actual SM90 and scalar float-output kernels.

CUDA-event microbenchmarks use 10 warmups and 30 samples. BF16 representative
median changes are -0.45% to +1.63%; repaired FP32 latency is comparable to the
original output-equivalent BF16-to-FP32 widening. The M=1 full-vocabulary FP32
case adds about 61.10 MiB peak allocation for its padded FP32 temporary. Full
timings, p95, environment, hashes and reproduction commands are recorded in
`reports/experiments/task12-projection-gemm.md` and its compact JSON companion.

MUSA validation is **pending**, with platform-specific arithmetic and commands
documented for a collaborator. No cross-vendor, H200, engine/full-model,
multi-node, gradient or complete H+M acceptance claim is made. This is a
CUDA/Hopper forward-operator interim result.

Relates to #642 / Task 12.
