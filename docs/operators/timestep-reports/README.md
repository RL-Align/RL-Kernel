# Timestep MLP GPU validation

See [operator contract and usage](../timestep-embed-mlp.md) for the math, precision
policy, deterministic layouts, backend dispatch and supported gradients.
This report covers operator-level A100 (sm80) and H100 (sm90) validation, not
full-model train/infer consistency or maintainer approval.

## Results

| Check | A100, 2026-10-02 | H100, 2026-10-07 |
| --- | --- | --- |
| H3072, B1/3/16, seeds 386/9386, CUDA/Triton, FP32/BF16 | 24/24 passed | 24/24 passed |
| H17, B33/64, long sample reduction | 8/8 passed | 8/8 passed |
| Official forward/backward CLI combinations | 4/4 passed | 4/4 passed |
| Original related pytest suite | 60 passed | 60 passed |
| Suite including reference VJP finite differences | Not run | 63 passed, no skips |
| Related gtest framework regressions | Not claimed | 155 passed, no skips |
| Independent reference accuracy audit | Not run | 8/8 passed at FP32 phase boundary |
| Profiler confirms real candidates without fallback | Four traces | Four traces |

Configuration checks cover repeated execution, chunking, permutation, padding,
strided inputs and singleton row comparisons. Parameter-gradient invariance is
only for the same complete logical sample set. Independently accumulated BF16
microbatch gradients are outside that guarantee.

## Core evidence

- [a100-results.json](evidence/a100-results.json): final production matrix,
  long reductions, four official CLI outputs, four production traces, pytest log,
  environment, source manifest, and the baseline failure/gradient diagnosis.
- [h100-results.json](evidence/h100-results.json): production and long-reduction
  matrices, independent reference audit, profiler traces, stage summary, source
  manifests, environment/package inventory and original execution logs.
- [SHA256SUMS](evidence/SHA256SUMS): checksums of these consolidated JSON files.

Each bundle has an `artifacts` map keyed by the original artifact filename.
For `format: json`, `content` is the original parsed JSON; for `format: text`,
it is the unmodified text. `original_sha256` identifies the original file bytes,
not the reserialized JSON. Paths within results and logs describe the original
run and are not links to files in this reduced checkout. Values, error metrics,
timing samples, traces and source hashes have not been recomputed or changed.
Redundant development reports, local diagnostics and intermediate GPU matrices
were removed. Full pre-consolidation artifacts remain in Git commit
`7f5b0c33c5f29683909e47017383036eec162f2f`.

Example: inspect the production results and verify bundle integrity:

```bash
(cd docs/operators/timestep-reports/evidence && sha256sum -c SHA256SUMS)
python - <<'PY'
import json
from pathlib import Path
root = Path('docs/operators/timestep-reports/evidence')
for gpu, key in [('a100', 'a100-delivery-matrix.json'), ('h100', 'production.json')]:
    bundle = json.loads((root / f'{gpu}-results.json').read_text())
    matrix = bundle['artifacts'][key]['content']
    print(gpu, 'passed:', matrix['passed'], 'cases:', len(matrix['cases']))
PY
```

## Source and environment

The original H100 acceptance used snapshot
`ea55fa686706892ce94ebe7020829c19c108496c`. Later identity/DCO corrections changed
commit IDs without changing runtime files. Audit tooling/tests were added later;
`source-manifest.json` and `audit-source-manifest.json` in the H100 bundle record
those snapshots separately. A100 results are historical and use their own source
hashes. This documentation cleanup does not represent a new GPU run.

H100: one NVIDIA H100 SXM 80GB HBM3, compute capability 9.0, MIG disabled;
Ubuntu 22.04, Python 3.10.12, CUDA Toolkit 12.4.131, driver 580.126.09,
GCC 11.4, PyTorch 2.6.0+cu124, Triton 3.2.0, pytest 8.3.5. TF32 was disabled.
A fresh isolated sm90 JIT build passed; original acceptance took 113.06 seconds
including compilation but excluding environment installation. A100 environment
and per-file hashes are retained in its bundle.

## Independent reference audit

`scripts/audit_timestep_reference.py` compares the old row-wise `torch.mv`
reference, the compensated FP32 reference, ordinary CPU FP64 `F.linear`/`F.silu`
with native autograd, and both GPU candidates. Input, parameter and upstream
bytes are hashed. The FP64 oracle does not reuse candidate reduction primitives
or the handwritten linear backward. All six tensors (output and five gradients)
use the unchanged official reduction tolerances.

Eight H3072 cases cover both dtypes: B16/seed9386 with CUDA and CPU upstream RNG,
B3/seed1701 and B16/seed20261007 with CPU upstream RNG. Two distinct comparisons
must not be conflated:

1. **Acceptance at the FP32 phase boundary:** construct frequencies and phase
   using the specified FP32 arithmetic, then evaluate trig/MLP/autograd in FP64,
   retaining the analytic phase Jacobian. The current reference and both
   candidates pass all eight cases; candidates also pass against the reference.
2. **All-FP64 diagnostic:** compute frequencies, phase and the remaining math in
   FP64. This measures a different rounding boundary and is not an acceptance
   substitute. Three of four FP32 cases exceed the official tolerance for at
   least one tensor; all four BF16 diagnostic cases pass.

For the historical FP32 B16/seed9386/CUDA-upstream case, the maximum timestep
gradient error divided by its per-element tolerance is 5.311 for the old
reference and 0.642 for the current reference. The all-FP64 diagnostic reaches
80.892. These are evidence for the specified FP32-phase contract, not equivalence
to continuous all-FP64 mathematics or a guarantee for every possible input.

The handwritten linear VJP also passes independent finite-difference
`torch.autograd.gradcheck` at B1/3/33, K5/H7 for input, weight and bias derivatives.
This isolated test uses FP64; the production reference remains FP32.

## Performance and limitations

H100 CUDA forward+backward microbenchmarks measured approximately 1.51–1.85x
relative to the ordinary batched PyTorch baseline (20 CUDA-event samples,
median). B16 forward alone and most Triton cases were slower. Raw timing samples
are retained in the production matrices. This is not a full-model speedup claim.
No Hopper WGMMA/TMA-specific optimization was implemented.

Full-model training, multi-GPU execution and full-repository native extension
builds remain unverified. Autocast and second-order gradients are unsupported.
BF16 is storage/output precision with FP32 intermediates, not layerwise BF16
rounding. Cross-GPU bit equality is not claimed; equal seeds need not generate
identical upstream tensors on different GPU architectures.

The initial H100 framework attempts failed because the trimmed source package
omitted benchmarks and a design fixture. Adding the unchanged fixed-snapshot
files yielded 155 passing tests; initial and final logs are preserved. An older
local double-free remains undiagnosed; the isolated H100 suites passed.
The target `test-qwenimage` was not included in the repository's main/test CI
triggers at validation time. These runs do not establish full-repository CI success.

## Reproduction

Use a compatible CUDA toolkit, compiler and the recorded Python dependencies.
From the repository root, with an activated environment and a fresh output root:

```bash
# Select the entrypoint matching the GPU.
TIMESTEP_RUN_ROOT=/absolute/new-a100-run bash scripts/run_timestep_official.sh
TIMESTEP_RUN_ROOT=/absolute/new-h100-run bash scripts/run_timestep_sm90.sh

python scripts/validate_timestep_official.py --hidden 17 --batches 33 64 \
  --seeds 386 --output /absolute/new-long-reduction.json

OMP_NUM_THREADS=4 TIMESTEP_CUDA_JIT_ONLY=1 TORCH_CUDA_ARCH_LIST=9.0 \
  python scripts/audit_timestep_reference.py --output /absolute/new-audit.json
python -m pytest tests/test_timestep_reference_audit.py -q
```

The sm90 entrypoint requires compute capability 9.0, forces fresh JIT loading
from the current source, and isolates CUDA/Triton caches. It preserves the
original arithmetic and tolerances. The audit refuses to overwrite its output;
its pass status uses the phase-boundary comparison, candidate/reference agreement
and real-backend checks, while retaining old-reference and all-FP64 failures.
