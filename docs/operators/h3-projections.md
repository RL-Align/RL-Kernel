# H3 gate/up and attention output projections

Test-only BF16 WS1 candidates for [issue #420](https://github.com/RL-Align/RL-Kernel/issues/420),
using `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`.

| Operator | Input width | Output width | Native block-0 weight |
| --- | ---: | ---: | --- |
| `h3_ffn_gate_up_gemm` | 5376 | 28672 | `ff.net.0.proj.weight` |
| `h3_attention_o_gemm` | 7168 | 5376 | `attn.to_out.0.weight` |

Inputs are BF16 `[M,K]` or `[B,S,K]`, weights are native `[N,K]`, and bias is
unsupported. Gate/up preserves packed checkpoint row order without SwiGLU.
Attention output consumes flattened 56 x 128 head channels. The implemented
row range is 1 through 131072. Execution requires one NVIDIA SM90 device,
eager mode and disabled TF32. Registry entries remain test-only with
`gpu_qualified=false`; `acceptance_complete=false`.

## Arithmetic and training contract

The NVIDIA adapter reuses the existing Triton `_det_gemm_fp32_kernel`:
64x64x32 tiles, four warps, two stages, no fusion, autotuning or split-K.
Each 32-wide BF16 dot produces FP32; tile sums are added in increasing K order
and cast to BF16 once. Forward, dX and dW use this policy and 64-bit indexing.
The adapter explicitly disables bias, compensation, K chunks and aligned-load
shortcuts. The shared kernel retains upstream bias/chunk features for other callers.
There is no new native CUDA C++ kernel and no fallback.

```python
import torch
from rl_engine.ops.gemm.h3_projections import H3FFNGateUpGemmOp, H3AttentionOGemmOp
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session

torch.backends.cuda.matmul.allow_tf32 = False
op = H3FFNGateUpGemmOp()  # Or H3AttentionOGemmOp.
with canonical_backward_session() as session:
    y = op(x, weight, logical_keys=keys, parameter_id="block0.ff.gate_up.weight")
    y.backward(dy)
    session.validate_complete()
```

Weight gradients require one session spanning all forward uses and one backward
call. Int64 keys `[rows,2]` or `[rows,3]` identify unique active logical rows;
a negative first component marks padding, whose dY must be zero. Each parameter
retains one weight Tensor and one stable name. Active rows are sorted
lexicographically, padded with zeros to a multiple of 32 and reduced once.
A complete parameter session containing only padding returns zero dW without
launching a weight-gradient GEMM. Its trace records `weight_gradient_zero=true`
and `canonical_active_rows=0`, leaving `weight_gradient_executed=false`.
Padding-only chunks before active chunks remain supported.

Repeated accumulation, retained-graph replay, double backward, compilation,
TP and full-model integration are outside this candidate's contract.

## Implementation and validation entry points

| Responsibility | Location |
| --- | --- |
| Public operator and session integration | `rl_engine/ops/gemm/h3_projections.py` |
| NVIDIA launch adapter | `rl_engine/backends/cuda/gemm/h3_projection_fp32.py` |
| Reused Triton kernel | `rl_engine/backends/shared/triton/gemm/det_gemm.py` |
| Independent mathematical reference | `rl_engine/reference/gemm/h3_projections.py` |
| Reusable validation harness | `rl_engine/validation/operators/h3_projection_validation.py` |
| Operator, adapter and reference tests | `tests/ops/gemm/test_h3_projections.py`, `tests/backends/cuda/test_h3_projection_fp32.py`, `tests/reference/gemm/test_h3_projections.py` |

The historical shared adapter import aliases the NVIDIA module, sharing its
cache. Fixture/runner tests live in `tests/validation/operators/`.
The range downloader retrieves only the two pinned checkpoint tensors, records
weight/index/header hashes and rejects malformed or ignored HTTP ranges.
Activations are synthetic; a checkpoint weight is not a captured activation.

For the full operator matrix, use a non-MIG H100 80GB, at least 16 CPU cores,
64GB host RAM and 100GB free storage. The original matrix used Python 3.10.12,
PyTorch 2.6.0+cu124, Triton 3.2.0 and driver 580.126.09.

```bash
python -m pip install -r requirements/h3-projections.txt
export H3_EVIDENCE_DIR="${EVIDENCE_ROOT}/h3-projections"
bash ci/scripts/run_h3_projections.sh smoke
bash ci/scripts/run_h3_projections.sh full
bash ci/scripts/run_h3_projections.sh tasks
bash ci/scripts/run_h3_projections.sh tasks-replay
```

Set `EVIDENCE_ROOT` to a writable output volume before running. Reports include
source/software identity, actual kernel/cubin hashes, every timing sample and
forward/dX/dW accuracy and invariance results. The full matrix covers random,
normalized and cancellation inputs, 13 row counts through 32768, and two
checkpoint-weight cases. Tasks add 131072 rows and packed row counts 39982,
75658 and 106238 for both operators, with random synthetic activations.
These correspond to B=1, 1344x768 task fixtures at 124, 243 and 345 frames,
256 text rows and two conditioning keyframes. Geometry follows pinned diffusers
v0.40.0 [frame conversion](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/modular_pipelines/minimax_h3/modular_pipeline.py)
and [sequence packing](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/modular_pipelines/minimax_h3/before_denoise.py).
This does not cover every aspect ratio, prompt length, reference count or batch.

Checks include an independent FP32 reference from identical BF16 inputs, repeat,
row permutation, non-divisible chunks, strided layouts, padding, batch reshaping,
singleton rows and unrelated-row mutation. Published shared tolerances are
unchanged; nonfinite results and signed-zero changes fail invariance. Long cases
use 4093-row chunks; smaller cases use 31. Replay compares all three output
digests in a fresh process/cache. Wall timings include synchronization, copies,
host checks and canonical preparation; ordinary PyTorch BF16 is a performance
comparator without byte-invariance qualification.

## Shared FP32 regression and CI

`tests/ops/gemm/test_det_gemm_fp32.py` exercises actual default
`INDEX_64=False` launches: 24 accuracy/repeat/permutation/chunk cases across
three shapes, four BF16/FP32 dtype pairs and contiguous/strided layouts;
four exact cancellation cases; four bias/aligned-load cases; and two upstream
chunked-reduction cases with K tails. The NVIDIA full-K FP32-output helper uses
eight warps and explicitly rounded compensated tile addition. Direct H3 calls
retain their four-warp uncompensated arithmetic; upstream chunked callers retain
their two-half reduction and never select compensation.

```bash
RL_KERNEL_REQUIRE_FP32_GPU=1 python -m pytest -q tests/ops/gemm/test_det_gemm_fp32.py
```

Strict mode fails on missing NVIDIA SM90+ or Triton prerequisites. The H3 smoke
script requires exact SM90 and runs it first. The proposed GPU workflow adds
`test-h3` with an SM90-only job, retaining the `needs-gpu-ci` label gate and
the existing matrix on other targets. The trusted runner executes this suite
before native extension smoke checks when the checked-out head contains it.
The workflow and runner are read from the trusted base branch: a maintainer must
enable that route there, or provide another SM90+ CI route. No actual CI result
is claimed. Local preparation does not enable or trigger remote jobs.

## Results and evidence status

All GPU results below are historical, from before integration with `test-h3`
at `59725d83bd045ebb1c56dc27ccd1be3163df7c7d`. The merged shared kernel and
new all-padding path still require target-device validation. Old results do not
certify the merged source, its generated kernels or current CI.

| Historical execution | Outcome and scope |
| --- | --- |
| H100 at `b58e276d5702c6ca70fb229f1cebb8318016aba0` | 71 focused tests; 8 smoke / 176 checks; 80 full / 1886 checks; 12 task / 288 checks passed. All 36 task replay y/dX/dW digests matched. Worst error/tolerance ratio 0.194678. |
| Additional regression at that source | Three standalone shared FP32 checks passed. Native-extension selection: 27 failures from unavailable `_C` attributes. Generic Triton selection: 14 skips and 57 deselections. Neither selection passed. |
| Manual H100, 2026-10-27, shared module SHA256 `b2619e122fbb8d33206037e441e93ead325fbd760b79fd23168360316a0739bf` | 28 shared tests passed, no failures/skips; 164 real default-index launches. All 24 FP32/FP64-reference diagnostics passed. Pre-compensation PR baseline passed 16/24; pre-existing long-K accuracy failures motivated compensation. |
| Manual H100 follow-up at `c8045837c7bdc9961559761e04076d502e875c2c` | 58 CPU checks passed; six GPU tests deselected. Two processes retained all 864 timing observations. The earlier 1.250282x timing flag was not reproduced; contiguous eager event ratios were 0.941509/0.964632, strided 1.056958/1.061701. Earlier adverse measurements remain retained; broad performance acceptance is incomplete. |
| Local merged-source preparation, 2026-10-30 | 76 affected CPU tests passed; 10 GPU tests deselected. All 34 shared GPU tests skipped because no CUDA device was available. CPU surrogates and launch-contract fixtures establish control flow only. |

The 2026-10-27 H100 checks used Python 3.11.10, PyTorch 2.4.1+cu124,
Triton 3.0.0 and driver 595.91.07. The timing follow-up used 100 warmups,
36 balanced path orders per mode and 100 eager wrapper calls per batch;
no trimming or pooling processes. Graph replay was diagnostic only.

Historical median synchronized wall times (milliseconds), random BF16 seed 420,
32768 rows at `b58e276`, two warmups and five repetitions:

| Operator | Candidate forward | PyTorch BF16 forward | Candidate forward/backward | PyTorch BF16 forward/backward |
| --- | ---: | ---: | ---: | ---: |
| Gate/up | 83.2183 | 14.6619 | 236.8656 | 41.9501 |
| Attention output | 20.5741 | 3.6446 | 51.7265 | 10.4624 |

These candidates were slower than PyTorch BF16. No speedup is claimed.
Full reports, raw logs, original checksums, compiled artifacts and failed
attempts remain preserved outside the merge diff. A private review copy retains
original/derivative hashes and sanitization provenance; public delivery of that
copy is pending separate authorization and access verification. No immutable
full-bundle download link is available yet. Previously published summaries
remain in Git history; they do not replace full evidence delivery.

Captured inputs, independent-environment replay, native-extension regression,
TP and full-model acceptance remain incomplete. Historical device fit does not
establish full-model training fit, production readiness or current merge readiness.
