# H3 gate/up and attention output projections

CUDA WS1 candidates for `h3_ffn_gate_up_gemm` and `h3_attention_o_gemm` in
[issue #420](https://github.com/RL-Align/RL-Kernel/issues/420).
Checkpoint: `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`.

| Operator | Input width | Output width | Native block-0 weight |
| --- | ---: | ---: | --- |
| FFN gate/up | 5376 | 28672 | `ff.net.0.proj.weight` |
| Attention output | 7168 | 5376 | `attn.to_out.0.weight` |

Both accept BF16 `[M,K]` or `[B,S,K]`, native `[N,K]` weights and no bias.
Gate/up returns the packed projection in checkpoint row order without a SwiGLU
activation or row permutation. Attention output consumes flattened 56 x 128
head channels in their original order. Inputs support 1 to 131072 rows.
The expanded range passed the H100 operator tests below.
One NVIDIA SM90 device, eager execution and TF32 disabled are required.

The shared adapter reuses `_det_gemm_fp32_kernel` with fixed 64x64x32 tiles,
four warps, two stages, no fusion, no autotuning and no split-K. Each 32-wide
BF16 dot tile produces FP32; tile sums are added in increasing K order and
cast to BF16 once. Forward, dX and dW use the same arithmetic policy.
There is no fallback and no newly introduced device kernel. Registry entries
remain test-only, with `gpu_qualified=false` until reviewed GPU qualification.
This separate branch has no dependency on unmerged PR #535 and imports none of
its historical GPU evidence. Common code can be consolidated when both land.

```python
import torch
from rl_engine.ops.gemm.h3_projections import H3FFNGateUpGemmOp, H3AttentionOGemmOp
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session

torch.backends.cuda.matmul.allow_tf32 = False
op = H3FFNGateUpGemmOp()  # Use H3AttentionOGemmOp for attention output.
with canonical_backward_session() as session:
    y = op(x, weight, logical_keys=keys, parameter_id="block0.ff.gate_up.weight")
    y.backward(dy)
    session.validate_complete()
```

Training with weight gradients requires one session spanning all forward uses
and one backward call. Int64 keys `[rows,2]` or `[rows,3]` must identify unique
active logical rows; a negative first component marks padding with zero dY.
The same parameter must retain one weight Tensor and one stable name. dW rows
are sorted canonically, zero-padded to a multiple of 32 and reduced once.
Repeated accumulation, retained-graph replay, double backward, graph compilation,
TP and full-model integration are outside this candidate's contract.

## Local validation and GPU qualification

CPU tests cover contracts, session ordering, chunks, permutations, padding,
partial gradients, non-contiguous version counters, fail-closed registry
resolution, reference math, fixture byte ranges and runner orchestration.
CPU surrogates exercise control flow only; they do not emulate GPU arithmetic.
The H100 run at source revision
`b58e276d5702c6ca70fb229f1cebb8318016aba0` passed 71 focused tests,
8 smoke cases (176 checks), 80 full-matrix cases (1,886 checks), and 12
task/boundary cases (288 checks). All 12 task replay cases passed; all 36
y/dX/dW digests matched across separate processes and compilation caches.
The worst error divided by tolerance was 0.194678. Every invariance check
passed, and explicit execution traces recorded 64-bit addressing, the requested
Triton kernel and no fallback. The retained cubin files match the launch hashes.
The 131072-row boundary and all three task-shaped row counts passed on both operators.

Additional native-extension/TP regression could not complete because `_C` was
unavailable. This is outside the two operators' WS1/TP1 acceptance scope;
the evidence bundle retains the outcome and error types, with detailed logs
preserved separately. The generic Triton regression selection did not execute;
its skip summary is retained because this change touches shared code. Neither
suite is counted as passed. Three direct GPU checks of the shared FP32
default-index path passed accuracy, repeat and permutation checks, but do not
replace the uncompleted regression suites.

The machine used one H100 80GB (SM90) and an Intel Xeon Platinum 8470 CPU.
Software was Python 3.10.12, PyTorch 2.6.0+cu124, Triton 3.2.0 and driver
580.126.09. The optional global core extension was unavailable; its import-time
warning does not describe these operators, whose actual kernel/cubin traces
are retained. Registry entries remain test-only pending broader qualification.

Use one non-MIG H100 80GB, at least 16 CPU cores, 64GB host RAM and 100GB free
storage. A CUDA 12.4 environment with Python 3.10 is the primary test profile:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements/h3-projections.txt
export H3_EVIDENCE_DIR="${PWD}/artifacts/h3-projections"
bash ci/scripts/run_h3_projections.sh smoke
bash ci/scripts/run_h3_projections.sh full
bash ci/scripts/run_h3_projections.sh tasks
bash ci/scripts/run_h3_projections.sh tasks-replay
```

The full matrix covers both operators, 13 row counts from 1 through 32768 and
random, normalized and cancellation inputs: 78 synthetic cases plus two real
checkpoint-weight cases. Fixture activations are explicitly synthetic. The
range downloader retrieves only the two required tensors and records revision,
weight key and weight/index/header digests. It rejects ignored or malformed
HTTP ranges. Captured model inputs remain an additional integration qualification target.

Each case checks independent FP32 y/dX/dW accuracy from identical BF16 inputs,
repeat, row permutation, non-divisible chunks, non-contiguous layout, inactive padding,
batch reshaping, singleton rows and mutation of unrelated rows. Published shared
tolerances are used unchanged; NaN/Inf and signed-zero differences fail invariance.
Synchronized wall timings include host checks, copies and canonical preparation;
GPU peak allocation and every sample are retained. Ordinary PyTorch BF16 is a
performance comparator, with no assumed byte-invariance qualification.
Fresh-process replay uses a separate compilation cache and compares all three
output digests. Reports retain actual JIT/cubin identity, source hashes and
software versions. Hardware identity is limited to necessary device facts.

## Measured performance and evidence

Median synchronized wall times in milliseconds, random BF16 inputs with seed
420 and 32,768 rows at `b58e276`; two warmups and five measured repetitions:

| Operator | Candidate forward | PyTorch BF16 forward | Candidate forward/backward | PyTorch BF16 forward/backward |
| --- | ---: | ---: | ---: | ---: |
| Gate/up | 83.2183 | 14.6619 | 236.8656 | 41.9501 |
| Attention output | 20.5741 | 3.6446 | 51.7265 | 10.4624 |

The candidate is slower than ordinary PyTorch BF16 in these measurements.
The baseline has no byte-invariance qualification, and the candidate timing
includes validation, copies and canonical gradient preparation. [Compact historical results](h3-projections-results.json) contain per-operator
accuracy and invariance summaries for every suite, case coverage, complete replay
comparisons, representative timing samples, regression failures/skips and original
hashes. Verify with `sha256sum -c h3-projections-results.sha256` in this directory.

Full reports, historical runs, raw logs and compiled artifacts remain privately
preserved and hash-verified (261 current-run and 170 earlier-run files). The compact
JSON is a selected derivative, not a lossless export or new execution. Previously
published reports remain in Git history. Private archives do not establish public
reproducibility. The earlier evidence reduction did not change numerical
implementation fingerprints; the later shared FP32 repair is described below.

Historical local checks at `3537a0b` passed 91 tests with 16 GPU-dependent skips
(registry, full-attention and projection selections). No runtime tests or GPU runs
were repeated for this documentation/evidence reduction.

Real weights use synthetic activations. Captured inputs, independent-environment
replay, full-model integration and TP remain untested. `acceptance_complete=false`;
these results do not establish production readiness or a performance improvement.

## Implementation and reuse

- Operator/session integration: `rl_engine/ops/gemm/h3_projections.py` and
  `rl_engine/ops/autograd/canonical_backward.py`.
- NVIDIA adapter: `rl_engine/backends/cuda/gemm/h3_projection_fp32.py`.
  The historical shared import remains a thin compatibility alias.
- Mathematical reference: `rl_engine/reference/gemm/h3_projections.py`;
  comparison and training orchestration remain in `rl_engine/validation/operators/`.
- Device computation: existing `_det_gemm_fp32_kernel` in the shared `det_gemm.py`.
  These operators select 64-bit indexing; default callers retain their index policy.
  No new native CUDA kernel or silent native fallback is introduced.
- Tests: `tests/ops/gemm/test_h3_projections.py` and fixture/runner tests in
  `tests/validation/operators/`.

Fixed ordered tiles match the declared arithmetic policy; the adapter supplies
weight layout and canonical cross-call dW. CUTLASS, FlashInfer, SGLang and vLLM
remain possible performance comparators, not qualified replacements for this
training/session contract. No external-backend benchmark or speed advantage is
claimed. The maintained plotting tool accepts full raw report bundles, not this
compact summary; charts and their source reports remain private.

## Task-shaped operator acceptance

The earlier 32768-row ceiling was below a realistic 768p task. Geometry follows
pinned diffusers v0.40.0
[frame conversion](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/modular_pipelines/minimax_h3/modular_pipeline.py)
and [packed sequence construction](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/modular_pipelines/minimax_h3/before_denoise.py):
`T = ((F - 5) // 17) * 5 + 2`, video rows `T * 24 * 42`, stereo audio rows
`2 * round(F / 24 * 40)`. The fixture explicitly selects 256 text rows and two
keyframes (2016 conditioning rows). These are task-shaped synthetic inputs,
not captured model activations. No full model is required to exercise these
row-local GEMMs and their canonical weight gradients.

| Frames | Latent frames | Video rows | Audio rows | Total packed rows |
| ---: | ---: | ---: | ---: | ---: |
| 124 | 37 | 37296 | 414 | 39982 |
| 243 | 72 | 72576 | 810 | 75658 |
| 345 | 102 | 102816 | 1150 | 106238 |

The largest declared task fixture is 345 frames, B=1, 1344x768; the 131072-row
boundary is also tested separately. This is not a claim to cover every aspect
ratio, prompt length, reference count or larger batch. Device fit and correctness passed on the tested H100 80GB. This does not
establish fit for full-model training or a different device/software profile.
Long cases use non-divisible chunks of 4093 rows; historical-sized cases retain
31-row chunks. Byte mismatch counts are accumulated in bounded chunks to avoid
an unnecessarily large temporary during long-case verification.

```bash
bash ci/scripts/run_h3_projections.sh smoke
bash ci/scripts/run_h3_projections.sh full
bash ci/scripts/run_h3_projections.sh tasks
bash ci/scripts/run_h3_projections.sh tasks-replay
```

Both task runs exercise 32768, 131072 and all three packed row counts for both
operators, plus checkpoint fixtures; replay uses a fresh process/cache and
compares all output digests. The full historical matrix was rerun because the
index specialization changed. The new task runs currently use random inputs;
normalized/cancellation coverage remains in the full regression matrix.

Negative tests now explicitly cover mixed-device inputs/keys, malformed key
shape/dtype/count, duplicates, session row overflow and unsupported registry
WS/TP combinations. Direct eager calls are local operators and do not negotiate
distributed topology; distributed callers must use the strict registry. Existing
coverage includes empty/wrong-dtype inputs, unsupported devices/backends,
non-contiguous inputs, padding gradients and stale autograd versions.

## Shared FP32 regression and CI

`tests/ops/gemm/test_det_gemm_fp32.py` covers the shared helper and public
`forward_fp32` entry point with real launches using the unchanged
`INDEX_64=False` default. Its 24-case matrix combines `(M,K,N)` shapes
`(1,65,73)`, `(33,129,67)` and `(129,5376,73)`, BF16/FP32 and mixed inputs,
and contiguous/strided layouts. Each case resets seed 420 and checks FP32
reference accuracy at the existing `atol=rtol=1e-4`, byte-identical repeats,
row reversal and 31-row chunks. This extends the earlier BF16-only checks;
it is not an exact replay of their inputs.

```bash
RL_KERNEL_REQUIRE_FP32_GPU=1 python -m pytest -q tests/ops/gemm/test_det_gemm_fp32.py
```

This command requires NVIDIA SM90+ and Triton and fails on missing prerequisites;
ordinary unsupported-device runs skip. The H3 smoke profile retains its exact
SM90 requirement and runs this command first. The proposed GPU workflow adds
`test-h3` with only an SM90 job, retains the `needs-gpu-ci` label gate and uses
same-device-model fallback. Other targets retain the existing architecture matrix.
The shared runner executes the selection strictly on SM90+ when the checked-out
source contains it, before native extension smoke checks but after the editable
build. Older PRs without this file are unaffected; lower-SM jobs do not establish
shared FP32 acceptance. Because the trusted workflow and runner come from the
base branch, a maintainer must enable this route there or provide another SM90+
CI route. No actual CI execution of this selection has been recorded.

The target-branch kernel at `97a7ab9` failed all eight long-K accuracy cases
under matched H100 inputs, including against an FP64 product rounded to FP32;
these are pre-existing precision failures. The NVIDIA FP32-output helper now
uses compensated addition across fixed K tiles with explicit round-to-nearest
PTX operations, retaining disabled TF32, fixed tile sizes/order and no split-K.
Four exact cancellation cases cover the compensation for each dtype pair.
This changes the general NVIDIA FP32 path; direct H3 adapters and non-NVIDIA
callers retain the uncompensated default. Shared FP32 backward support is unchanged.

The NVIDIA helper uses eight warps; the H3 direct adapter retains four warps.
Manual H100 SM90 validation on 2026-10-27 (UTC+08:00; shared-module SHA256 prefix
`b2619e122fbb8d33`; Python 3.11.10, PyTorch 2.4.1+cu124, Triton 3.0.0,
driver 595.91.07) passed all 28 shared tests with zero failures or skips.
All 164 observed launches used `INDEX_64=False`, compensation and eight warps.
All 24 matched diagnostics passed against FP32 and FP64-rounded-to-FP32
references, with worst error/tolerance ratios of 0.384756 and 0.161371.
Outputs matched the previous four-warp compensated version byte for byte in
all 24 cases. The pre-compensation PR module passed 16/24 cases; it is a
separate baseline from the target-branch comparison above.

The three-path performance comparison retains all 432 samples: ten warmups
per path and all six path-order permutations, with twenty calls per timed
batch using CUDA events. Timings include wrapper copies/allocation; clocks
were not pinned. Eight warps reduced the four previously large regressions
to candidate/baseline ratios of 1.072 to 1.182 in this run. One case remains
flagged for review: contiguous FP32/BF16 `(33,129,67)` measured 0.035488 ms
versus 0.028384 ms for the pre-compensation baseline (1.250282x) and
0.031197 ms for four-warps compensation. Its six candidate samples range
from 0.021970 to 0.062435 ms; the cause of this variation is unresolved.
The 1.25 flag is a local review threshold, not an upstream acceptance rule.
Across all 24 cases, candidate/baseline ratios range from 0.426 to 1.250282.
These shared-helper measurements do not establish H3 performance or broad
performance acceptance. The earlier four-warp regressions (2.74 to 7.11x)
and all other original results remain privately preserved unchanged.

Historical H100 execution on 2026-10-26 passed six full-width H3 cases at
1, 33 and 129 rows, including y/dX/dW accuracy and chunk invariance, with
zero failures or skips. That H3 evidence is reused without rerunning: its
direct kernel call, kernel body and numerical dependencies are unchanged.
The adapter was relocated byte for byte to its NVIDIA-owned directory and the
reference functions moved unchanged to the reference layer; canonical imports
and provenance paths were updated, with compatibility imports retained.
Affected CPU regression execution for this relocation remains incomplete. Traces used `INDEX_64=True` without compensation. The current
change does not establish actual CI execution, native-extension regression,
full H3 workload performance or qualification on other software/device
profiles. Original results, failed attempts, source snapshots and compiled
artifacts remain private; private integrity checks are not public evidence
availability. Correctness passes do not establish merge readiness.

Additional regression commands (neither is included in passing operator counts):

```bash
CUBLAS_WORKSPACE_CONFIG=:16:8 CUBLASLT_WORKSPACE_SIZE=1 \
  python -m pytest -q tests/backends/cuda/test_canonical_backward_fusion.py
python -m pytest -q tests/ops/gemm/test_det_gemm.py -k triton
```

The first recorded 27 failures due to unavailable `_C` attributes; the second
recorded 14 skips and 57 deselections. Three shared FP32 checks passed separately,
but do not replace these missing regression suites. Full logs and the standalone
shared-path reproducer remain privately preserved.
