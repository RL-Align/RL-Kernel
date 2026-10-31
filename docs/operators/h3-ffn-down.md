# H3 deterministic FFN down projection

CUDA WS1 candidate for `zzf2024`'s `h3_ffn_down_gemm` work item in
[issue #420](https://github.com/RL-Align/RL-Kernel/issues/420).
Measured base: `test-h3@32b765ec992cd1206517104ec66506881203c91c`.
Updated for the directory refactor in `test-h3@16b8117`.
Checkpoint: `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`.

## Contract and scope

Backend `rlkernel.h3_ffn_down.triton.fp32_tiles.v1` reuses the existing
`rl_engine.backends.shared.triton.gemm.det_gemm._det_gemm_fp32_kernel`.
The historical backend, class and logical kernel identifiers remain stable;
`fp32_tiles` denotes accumulation precision, not FP32 multiplication. With
`PROMOTE_INPUTS=False`, BF16 operands feed 32-wide BF16 tensor-core MMA
tiles with FP32 accumulation and ascending FP32 tile additions, followed
by one BF16 cast. The schedule is fixed at 64x64x32, four warps and two stages, with
`enable_fp_fusion=False`, TF32 disabled, no tuning, no split-K and no fallback.
No new CUDA or Triton device kernel is added.

The eager operator accepts BF16 `[M,14336]` or `[B,S,14336]`, flattened row count
1..32768, native `[5376,14336]` weights and no bias on one SM90 CUDA device.
It computes `Y=X@W.T`, `dX=dY@W`, and canonical `dW=dY.T@X`. Strided tensors
are normalized by copies. The adapter passes transposed weight/gradient views
using explicit operand strides; storage layout does not change logical K order.
The historical forward/dX/dW matrix includes this path. Weight-gradient rows
are sorted by logical identity
and padded to a fixed 32-row extent before a single reduction.
The 32768-row limit also applies to the cumulative active rows for one parameter
across a backward session.

The operator rejects invalid shapes, types, devices, TF32, duplicate active
keys, inconsistent weight identity and nonzero padding gradients. Unsupported
hardware or backend fails closed. Graph compilation, double backward and
retained-graph replay are unsupported. The registry descriptor remains
**test-only**; strict production resolution requires further qualification.
TP, whole-model integration, ROCm and cross-platform equality are outside this PR.

Inference requires no backward session:

```python
import torch
from rl_engine.ops.gemm.h3_ffn_down import H3FFNDownGemmOp

torch.backends.cuda.matmul.allow_tf32 = False
op = H3FFNDownGemmOp()
with torch.no_grad():
    y = op(swiglu_output, down_weight)
print(op.last_execution)
```

Training requires one session spanning **all forward uses and one backward call**:

```python
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session

with canonical_backward_session() as session:
    outputs = [op(x, down_weight, logical_keys=keys,
                  parameter_id="transformer.blocks.0.ffn.down.weight")
               for x, keys in zip(parts, key_parts, strict=True)]
    torch.cat(outputs).backward(upstream_gradient)
    session.validate_complete()
```

Keys are same-device int64 `[rows,2]` or `[rows,3]`, such as
`[sample_id,token_id]`, with stable unique identities across packing, permutation
and microbatching. A negative first component marks inactive padding. The same
weight Tensor must retain one parameter ID. Do not sum independently rounded
microbatch weight gradients; call `validate_complete()` to detect missing branches.
Execution traces record requested/actual backend, kernel, arithmetic, schedule,
topology, checkpoint, separate `input_gradient_kernel` / `weight_gradient_kernel`
fields and actual JIT/cubin hashes. A deferred canonical weight reduction keeps
its kernel field unset until that use executes the reduction.

Each forward validates and constructs its backend once and retains it for
backward. Padding row indices are snapshotted during forward registration;
unpadded uses skip the padding-related backward readback. Padded uses still
synchronously reject nonzero upstream padding gradients. Forward registration
and canonical row preparation can still synchronize; this is not an
asynchronous execution guarantee or a measured speedup.

## H100 results

### Current-head validation (2026-10-31)

The clean `92c98f0792f2f26349210f5cff61f572dd15a3c3` source was validated on
one non-MIG H100 80GB HBM3 (SM90) with Python 3.10.12, torch 2.6.0+cu124,
Triton 3.2.0 and CUDA toolkit 12.4. Smoke checks passed 28 CPU tests with four
GPU cases deselected, then 32 SM90 tests and three validator cases. The SM90
test suite includes the M=4097 ordered/permuted/padded canonical reduction.
The explicit full matrix passed **49/49 cases and 1,141/1,141 comparisons**
across the three input families and real checkpoint weights with synthetic
activations. It covers forward, input gradient, canonical weight gradient and
tail shapes. Execution traces include actual Triton JIT and cubin hashes; no
fallback was used.

All 49 measured candidate forward and forward-plus-backward cases were slower
than ordinary torch BF16, with median time ratios of **4.897x** and **5.108x**.
These are five-repeat synchronized wall timings for the whole operator path,
including wrapper work. No speedup or production qualification is claimed.
The original report, failed setup attempts and logs are retained privately.
The public evidence derivative records the full case/check matrix and source,
environment, fixture, trace and timing provenance.

The report retains `acceptance_complete=false`: captured H3 activations,
independent-environment replay, maintainer review of arithmetic and covered
layouts, full-model/TP integration and a performance decision remain pending.

The 12 files below were anonymously downloaded from the immutable Gist revision
and byte-matched to the reviewed sanitized derivative. The original report and
failed setup logs remain in the private archive. The provenance README explains
the path and device-identifier replacements; the checksum manifest pins all
other files.

| Current-head evidence download | SHA256 |
|---|---|
| [README.md](https://gist.githubusercontent.com/zzf2024/a348109cb5f6510159cda306b8358d06/raw/157c0a10410b4e24b3fc51421d3fd988a04eacb0/README.md) | `a0075312e52846fc5fc7df27093cffc0fcd3fa6cae469179d8dd5db5e59ce4a4` |
| [SHA256SUMS](https://gist.githubusercontent.com/zzf2024/a348109cb5f6510159cda306b8358d06/raw/157c0a10410b4e24b3fc51421d3fd988a04eacb0/SHA256SUMS) | `ce6b473fdf3e16439c74a1b8ab4d365f8e66f7f02037790954d561d725526f3d` |
| [run-metadata.json](https://gist.githubusercontent.com/zzf2024/a348109cb5f6510159cda306b8358d06/raw/157c0a10410b4e24b3fc51421d3fd988a04eacb0/run-metadata.json) | `0cf9405763c4f36668814988880448e09e09d260c5ff885fd44f3ec611d0aa80` |
| [smoke-report.json](https://gist.githubusercontent.com/zzf2024/a348109cb5f6510159cda306b8358d06/raw/157c0a10410b4e24b3fc51421d3fd988a04eacb0/smoke-report.json) | `229041f4f13161a0be4cbf1dd479bfd02c37cf23ca065e674639eaf242f00698` |
| [h100-random.json](https://gist.githubusercontent.com/zzf2024/a348109cb5f6510159cda306b8358d06/raw/157c0a10410b4e24b3fc51421d3fd988a04eacb0/h100-random.json) | `d2c4d4471532db6c6167950a567bf790a31df3716fe3ae42ebd4a2f72afce310` |
| [h100-swiglu-like.json](https://gist.githubusercontent.com/zzf2024/a348109cb5f6510159cda306b8358d06/raw/157c0a10410b4e24b3fc51421d3fd988a04eacb0/h100-swiglu-like.json) | `93dcdb230928c68964c0d8721c202f8c8ac1c82ccc272b842538da5dd49dbf2e` |
| [h100-cancellation.json](https://gist.githubusercontent.com/zzf2024/a348109cb5f6510159cda306b8358d06/raw/157c0a10410b4e24b3fc51421d3fd988a04eacb0/h100-cancellation.json) | `c3799c6a173aae97699f42168a1ac21e72bbeac1c48919d77d925c6927450075` |
| [h100-checkpoint-fixture.json](https://gist.githubusercontent.com/zzf2024/a348109cb5f6510159cda306b8358d06/raw/157c0a10410b4e24b3fc51421d3fd988a04eacb0/h100-checkpoint-fixture.json) | `c3c374dbe8f7b8ce89fb8dab9e9a6c6a6b28fda21986357e39d48d6e2b0e6f0a` |
| [fixture-metadata.json](https://gist.githubusercontent.com/zzf2024/a348109cb5f6510159cda306b8358d06/raw/157c0a10410b4e24b3fc51421d3fd988a04eacb0/fixture-metadata.json) | `60dafdff4bd4d50cbef4f50bcaf8ffe8fd6b2fc883ef924cceafcec999f8ac6f` |
| [smoke-cpu-tests.txt](https://gist.githubusercontent.com/zzf2024/a348109cb5f6510159cda306b8358d06/raw/157c0a10410b4e24b3fc51421d3fd988a04eacb0/smoke-cpu-tests.txt) | `f4e5bbfa2cb2d1427174429287fc5aa4e7ecd150c70886556c2a7e73bfd30d38` |
| [smoke-gpu-tests.txt](https://gist.githubusercontent.com/zzf2024/a348109cb5f6510159cda306b8358d06/raw/157c0a10410b4e24b3fc51421d3fd988a04eacb0/smoke-gpu-tests.txt) | `0b2a18e7842d63f8dcbc886d8a664f99ed4a6980a50a839490a89b74a3054718` |
| [full-validation.log](https://gist.githubusercontent.com/zzf2024/a348109cb5f6510159cda306b8358d06/raw/157c0a10410b4e24b3fc51421d3fd988a04eacb0/full-validation.log) | `86707b9c7fdeb99264e48a4c80760a602aa70de9004cb84ecf78917753b92696` |

### Historical validation (2026-10-10)

Measured on 2026-10-10: dedicated non-MIG H100 80GB HBM3, capability 9.0,
81,559 MiB, driver 570.211.01, Ubuntu 22.04, Python 3.10.12, torch 2.6.0+cu124,
CUDA 12.4 and primary Triton 3.2.0.

- **64 GPU/profile tests passed** before PR formatting.
- **49 cases / 1,141 comparisons passed**. Random, synthetic SwiGLU-like and
  cancellation inputs cover M=1,31,32,33,127,128,129,256,513,1024,2048,4096,
  8192,16385,24577,32768; a real-weight fixture adds M=129.
- Forward/dX/dW match an independent FP32 PyTorch autograd reference with the
  shared `ws1-c1-v2` tolerances. Maximum error/tolerance is **0.18233**, below 1.
  No tolerance was weakened.
- Raw-byte checks cover repeats, logical-row permutation, 31-row chunking with
  one backward, padding, non-contiguous layouts, batch reshaping, singleton rows
  and mutation of unrelated rows. NaN/Inf and signed-zero differences fail.
- A separate process and fresh Triton 3.2.0 cache passed 10 cases with **30 matching
  y/dX/dW hashes**. A separately pinned Triton 3.4.0 cache passed five selected
  cases with **15 matching hashes**. Both profiles used the same GPU and PyTorch.

The checkpoint fixture uses `transformer_blocks.0.ff.net.2.weight`, BF16
`[5376,14336]`, fetched through validated HTTP ranges without downloading the
whole model. Weight SHA256 is
`e19d4e1649e4b55b1931842e5cc4bb46892c2e635b9315b2996630a86284e88f`.
Activations and upstream gradients are **synthetic**, not captured model data.
The validation CLI preserves fixture labels as declarations and records
`activation_provenance_verified=false`. Even a fixture labeled `captured` keeps
independent capture-provenance verification in its remaining acceptance list;
a file checksum or self-declared capture hash does not establish provenance.

The explicit legacy backend `rlkernel.h3_ffn_down.sm90.midtree.v1` failed real-weight
forward accuracy at M=129: 23,353 failing elements, maximum error/tolerance
5.00441. Its failure is preserved; the BF16 midpoint tree is never selected
as an automatic fallback.

Five-sample synchronized wall medians for primary-profile random inputs:

| M | Candidate forward ms | torch BF16 forward ms | Candidate forward+backward ms | torch BF16 forward+backward ms |
|---:|---:|---:|---:|---:|
| 1 | 0.4641 | 0.0822 | 5.8811 | 1.2091 |
| 129 | 0.5118 | 0.1100 | 4.3671 | 1.2572 |
| 8192 | 7.1128 | 1.7194 | 18.4472 | 5.2220 |
| 32768 | 41.6259 | 7.1671 | 107.5935 | 21.5373 |

These include wrapper checks, copies, allocations and canonical preparation,
not isolated kernel latency. This candidate is slower than ordinary torch BF16.
The comparator has not been qualified for byte invariance.

Vendored vLLM commit `7797b6022c129b862e45ae6aed08822e65d1bccb` requires the separately
pinned Triton 3.4.0 profile because 3.2.0 rejects `tl.range(..., flatten=True)`.
Its partial probe passed y/dX/dW accuracy and y/dX chunk/singleton invariance.
Within 3.4.0 at random M=32768, vLLM forward took 9.6604 ms versus this adapter's
41.5963 ms; its dW primitive took 40.0210 ms versus 20.9800 ms for fixed FP32.
The vLLM primitive excludes this adapter's checks/copies and has no qualified
canonical autograd session or full dW invariance matrix. It remains prior art.

## Evidence and implementation inventory

[Compact historical results](h3-ffn-down-results.json) retain the complete matrix
summary, forward/dX/dW tolerances and worst errors, replay digests, representative
timings, rejected native results and original bundle hashes. Full reports, samples,
traces and raw originals are preserved privately. This is a selected derivative,
not a lossless report or a new GPU run. The complete historical report derivatives
are available with device UUIDs removed at the links below. These
downloads preserve every case, check, trace and timing sample, including rejected
native results. Raw private originals are unchanged. The historical
JSON retains its original FP32-dot trace wording; the corrected BF16 MMA / FP32
accumulation description above applies to that unchanged device computation.

| Historical report download | SHA256 |
|---|---|
| [Summary, real-weight fixture and rejected native results](https://gist.githubusercontent.com/zzf2024/2225006b9a7c25a05e7df4be423d1331/raw/87293a25635b4d92965cb2c8f1dd1d571c55ad29/h100-summary.json) | `d6011dd275ffa302abd33814a195d0b31b535da90a6d69988377ec9403bfef8d` |
| [Random matrix](https://gist.githubusercontent.com/zzf2024/2225006b9a7c25a05e7df4be423d1331/raw/87293a25635b4d92965cb2c8f1dd1d571c55ad29/h100-random.json) | `66fcd857d26a57ea03ce0984f8bf20db1785336c61d302dce74dfd6a4cf213fb` |
| [Synthetic SwiGLU-like matrix](https://gist.githubusercontent.com/zzf2024/2225006b9a7c25a05e7df4be423d1331/raw/87293a25635b4d92965cb2c8f1dd1d571c55ad29/h100-swiglu-like.json) | `dcc64cf4360bbb9dca0211319f635cffed1cd91ebc6ddfd24052882a7201ba9d` |
| [Cancellation matrix](https://gist.githubusercontent.com/zzf2024/2225006b9a7c25a05e7df4be423d1331/raw/87293a25635b4d92965cb2c8f1dd1d571c55ad29/h100-cancellation.json) | `2088317c29cc9de80540f215f54b7d6d660044a568c1f50f6194449b2c9b4fba` |
| [Cold-cache and compiler-version replays](https://gist.githubusercontent.com/zzf2024/2225006b9a7c25a05e7df4be423d1331/raw/87293a25635b4d92965cb2c8f1dd1d571c55ad29/h100-replays.json) | `31a40fa89e76b80f37b39b12fa2a16a98b3b4bf9d677be5387638ab1ca933cd5` |

[Download checksum manifest](https://gist.githubusercontent.com/zzf2024/2225006b9a7c25a05e7df4be423d1331/raw/87293a25635b4d92965cb2c8f1dd1d571c55ad29/SHA256SUMS)
has SHA256 `d1a268a31cb19458c30ef38f9bf2dc4e340bda2f35ab2525c93cb47d41a3f029`.
These are historical 2026-10-10 executions with portable paths and device UUIDs redacted,
not evidence that the current shared GEMM and canonical reducer have been qualified.

[Transformation provenance](https://gist.githubusercontent.com/zzf2024/2225006b9a7c25a05e7df4be423d1331/raw/87293a25635b4d92965cb2c8f1dd1d571c55ad29/README.md),
SHA256 `788a09f028a322ef54eef850fd6ba8394896ad89b8b97cbb52f9b64bf5c7c7a9`,
records the source and derivative hashes. Across the five reports, only four
device UUID occurrences changed; all measurements, failures and original
artifact hashes remain.

- Operator: `rl_engine/ops/gemm/h3_ffn_down.py`.
- Triton adapter: `rl_engine/backends/shared/triton/gemm/h3_down_fp32_backend.py`;
  actual device computation reuses `_det_gemm_fp32_kernel` in `det_gemm.py`.
- Canonical gradients: `rl_engine/ops/autograd/canonical_backward.py`.
- Native CUDA comparator: existing SM90 midpoint-tree backend; rejected for the
  accuracy failure above, never an automatic fallback.
- Tests: `tests/ops/gemm/test_h3_ffn_down.py`,
  `tests/backends/shared/triton/gemm/test_h3_down_backend.py`,
  `tests/benchmarks/test_h3_down_report.py` and
  `tests/validation/operators/test_h3_down_{fixture,tooling}.py`.

Historical local checks at `d8ae6cf` passed 58 tests with 13 GPU-dependent skips
(registry, full-attention and H3 down selections). No tests or GPU runs were
repeated for that documentation/evidence reduction. The subsequent review fix
corrects the arithmetic description and changes wrapper preparation and gradient
tracing, while preserving the device kernel, launch settings and canonical
reduction arithmetic. The H100 results above remain historical and were not
rerun for this fix; they do not measure the updated wrapper or establish its
GPU acceptance. Original reports, summary bytes and evidence pins are unchanged.

Local review-fix checks on 2026-10-26: **56 passed, 3 skipped** across
`tests/ops/gemm/test_h3_ffn_down.py`, `tests/runtime/test_kernel_registry.py`
and `tests/runtime/test_layout_compatibility.py`. The three skips require SM90;
the new tests exercise CPU orchestration, not GPU arithmetic. Lint and formatting
checks passed. No GPU run or performance measurement was made for this fix.

A follow-up to the same review passed **33 tests with 3 SM90 skips** across
the operator and tooling suites after strengthening actual-call assertions
and testing unverified fixture labels. The earlier registry/layout results
were reused; their runtime sources are unchanged. Follow-up lint and formatting
checks passed. No GPU execution was repeated in that 2026-10-26 review fix.

## Shared-kernel compatibility update

The adapter now supplies a null bias pointer and `HAS_BIAS=False` for the
bias-aware shared GEMM signature. CPU regression tests bind the actual forward,
dX and dW calls to the real JIT function signature, including transpose strides
and launch settings. Accuracy charts leave missing/nonfinite ratios unavailable
and mark failed checks explicitly; finite failed ratios remain visible.

Focused checks on 2026-10-30 with torch 2.6.0+cpu and Triton 3.2.0:
**46 passed, 4 SM90 tests skipped** across the backend, benchmark, operator, fixture
and tooling suites. Lint and formatting passed. CPU binding uses a synthetic
launcher and does not compile or execute a Triton kernel.

The shared GEMM and canonical reducer changed in the later upstream merge;
the 2026-10-10 H100 results do not qualify these current dependencies.
The current-head execution above covers the updated device path and its
full-width forward/dX/dW cases. Its scope remains operator-level qualification.

Timing tables and chart legends now identify the recorded backend when available;
reports without that field retain the generic candidate label. The report and
runbook checks passed **16 report/tooling tests** on 2026-10-30, including
four CPU simulations of failure priority and evidence-directory output. Lint,
formatting and Bash syntax checks passed. The earlier compatibility results
above are retained as a separate execution. No GPU execution was repeated during
those 2026-10-30 report/runbook checks.

## Reproduction and remaining qualification

The runbook preserves GPU-test, validation and plotting logs, always prints the
evidence directory after plotting, and returns failures in that priority order.
Its source patch captures tracked edits only; retain any executed untracked
sources separately. Native comparisons also require matching build records;
runtime provenance records the loaded binary hash but does not verify build flags.

Use a dedicated non-MIG H100 80GB and a CUDA 12.4 development environment.
Install matching Python 3.10 virtual-environment and development-header packages
(for example, `python3.10-venv` and `python3.10-dev` on Ubuntu). Triton JIT
compiles a CUDA helper even when the optional native extension is disabled:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements/h3-down.txt
python tools/weights/prepare_h3_down_fixture.py --rows 129 --output ${FIXTURE_ROOT}/h3-down-fixture.pt
H3_FIXTURE=${FIXTURE_ROOT}/h3-down-fixture.pt bash ci/scripts/run_h3_down_h100.sh smoke
python tools/validation/operators/validate_h3_ffn_down.py --seed 420 --repeats 5 \
  --rows 1,31,32,33,127,128,129,256,513,1024,2048,4096,8192,16385,24577,32768 \
  --families random,swiglu_like,cancellation \
  --fixture ${FIXTURE_ROOT}/h3-down-fixture.pt --output /tmp/h3-down-full.json
```

The default needs Triton, not the native extension. `H3_BUILD_NATIVE=1` optionally
builds the legacy comparison with fast math disabled; the runbook requires nvcc 12.4.
`--compare-vllm` is an explicit partial probe; use a separate Triton 3.4.0
environment without replacing the primary profile.

This change is ready for **operator review**, with `acceptance_complete=false`.
Captured H3 activations, review of arithmetic/covered layouts, formal runtime
integration and a performance decision remain before production promotion.
It does not close the full H3 roadmap or assert portable equality across devices.
