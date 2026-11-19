# RFC #642 Task 12: deterministic projection GEMM

CUDA/Hopper interim delivery; MUSA remains pending. This is synthetic forward
operator qualification, not full-model, distributed, engine or gradient acceptance.
Relates to #642 / Task 12. Task 13 gradients are unchanged.

## Baseline and changes

Base: `main@49941801f942428fd52ac5b66a1b0c070418724e`. The refreshed
`test-qwen36-miles@2668098a0e14e0efefb3ad1595a935323d221b6b` has the same
relevant GEMM code. These shared forward fixes have no model/miles dependency,
so main follows the maintainer's [shared-component exception][baseline-rule].

On the original binary, four new cast regressions fail because
`det_gemm_fwd_fp32` is not exported. Its existing C++ body also selects the BF16
output specialization. Export the existing entry point and select its existing
float store while retaining every BF16 leaf/node rounding point. Export the
already implemented SM90 rowwise FP32 entry point used by `forward_accum_fp32`.
No new numerical kernel, per-row fallback or gradient algorithm is introduced.

Restore the stale Triton test import and re-raise unexpected import failures;
otherwise existing tests silently drop Triton. This small correction overlaps
the test-import portion of [#450][pr450]; its gradient/grid work is not copied.
The scalar recursion/int32 fixes in [#679][pr679] remain separate.

## Target dimensions and arithmetic

The pinned [model config][config] and official [Transformers v5.13.1 call chain][model]
give H=2048, Q=16, KV=2, attention head dimension=256, GDN key/value
dimensions=2048/4096, expert width=512, experts=256 and vocabulary=248320.
The test groups equivalent K,N pairs:

| Projections | K → N |
| --- | --- |
| Gated attention Q; GDN QKV | 2048 → 8192 |
| Attention K/V; separate shared gate/up | 2048 → 512 |
| Attention output; GDN output | 4096 → 2048 |
| GDN z | 2048 → 4096 |
| GDN a/b | 2048 → 32 |
| Per-expert packed gate/up | 2048 → 1024 |
| Expert/shared down | 512 → 2048 |
| Router logits | 2048 → 256 |
| Shared expert gate | 2048 → 1 |
| LM head | 2048 → 248320 |

This does not certify engine-specific packed QKV, grouped expert scheduling,
router softmax/top-k, fused selected logprob or actual checkpoint weights.

CUDA SM90 uses BM=128, BN=64, BK=32, FP32 tensor-core leaves rounded to BF16,
then a midpoint tree with BF16 rounding at every node. M is padded to BM.
Scalar tails split K elements, while SM90 splits BK tiles; ragged/non-power-of-two
trees are independently qualified, not assumed cross-path equal. Triton BF16 has
its own fixed element-midpoint tree. Native CUDA FP32 output only widens the
tree root. Full FP32 accumulation uses a separate fixed 256-thread reduction;
Triton FP32 uses fixed-tile dots. No cross-profile equality is claimed.

## Environment and commands

2026-11-19 UTC, one NVIDIA H100 80GB HBM3 (SM90), MIG disabled; other seven
devices left unused. GPU UUID `GPU-25dc68ae-c0e6-49b1-23b5-c4da9d63e8ef`.
Driver 580.126.20; nvcc 13.4.59; Python 3.12.3; PyTorch
`2.14.0a0+4fdf77b940.nv26.08` (CUDA 13.4); Triton
`3.8.0+git4c7e62b7.nv26.8`; gcc 13.3.0. The task venv inherits this system
stack without modifying it. CUDA compilation emits sm_90a, no fast math; the
current PyTorch extension builder supplies C++20 to nvcc.

From a checkout with a matching PyTorch/CUDA environment:

```bash
export CUDA_VISIBLE_DEVICES=<one-idle-Hopper-GPU-UUID>
export CUDA_HOME=/usr/local/cuda MAX_JOBS=4
export KERNEL_ALIGN_DET_GEMM_SM90=1 KERNEL_ALIGN_FORCE_SM90=1
export KERNEL_ALIGN_USE_FAST_MATH=0 RL_KERNEL_REQUIRE_EXT=1
export RL_KERNEL_DET_GEMM_BACKEND=sm90 OMP_NUM_THREADS=4
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python setup.py build_ext --inplace
python -m pytest tests/ops/gemm/test_projection_gemm.py -q -rs
python -m benchmarks.operators.gemm.benchmark_projection_gemm --json artifacts/task12/timing.json
CUDA_VISIBLE_DEVICES='' python ci/run.py layout -q
CUDA_VISIBLE_DEVICES='' python -m pytest tests/ops/attention/test_attention_projection.py -q
```

## Validation evidence

- Original binary: 4 cast failures, 4 inapplicable-profile skips, 144 deselected;
  first failure is the missing forward-FP32 binding. No baseline accuracy claim.
- Post-fix new matrix: **158 passed**, no skips/failures, 596.93 s including
  first Triton compilation. Twenty pytest xunit2 property warnings; the values
  are present in XML (use `-o junit_family=xunit1` to avoid those warnings).
- Independent CPU FP64 dyadic K-tree goldens, non-power-of-two negative control
  and selected existing forward regressions: **38 passed, 65 deselected**,
  83.12 s. No Task 13 gradient suite was included.
- Final portable-reference helper check: **6 passed, 152 deselected**, 6.80 s.
  The MUSA branch uses CPU FP64; CUDA reference device/precision are unchanged.
- Across all 20 target backend/shape cases, maximum absolute error versus FP64
  is **0.037068993886350654**; LM head maximum is 0.036685844184830785. Existing
  accuracy limits pass; every declared raw-bit comparison has zero mismatches.
- Profiler confirms float-output SM90 tensor-core kernel at [129,2048,512]
  and float-output scalar kernel at [129,2048,32], rather than a silent substitute.
- Original versus repaired binary comparison: **26 outputs, zero differing
  bytes**, M=7, seed 649, all ten target pairs plus (65,33), (96,64), (96,65).
  Compare original BF16 output and original BF16-to-FP32 widening with repaired
  BF16 output and repaired native FP32 output, in independent processes.
- CPU layout: 33 passed, 1 extension-smoke skip. Attention projection: 8 passed.
- Focused Black 24.4.2, isort 5.13.2, flake8 7.0.0 and wrapper mypy pass.

Accuracy and invariance have separate checks. The mathematical oracle is FP64
matmul on the original BF16 inputs, with the unchanged repository reduction
limits (BF16 atol=0.05, rtol=0.02; full FP32 atol=rtol=0.0001). Random hidden
inputs have unit RMS; weights use the model initializer range 0.02. No tolerance
is used for raw-bit comparisons, and no outputs are downcast for comparison.
The tree's intermediate BF16 rounding is also tested by an analytic cancellation
case whose exact dot is 2 and CUDA/Triton BF16 tree result is 1.

M=129 target-shape tests compare all output elements against FP64, the same rows
at positions 0/64/128 as singletons, and concatenated chunks 7/31/64/127.
Batch boundaries M=1/2/31/32/63/64/65/127/128/129/255/256/1024/4096 use
K=2048,N=512, including row reversal and extra padding. Layout/K cases cover
K=1/31/32/33/63/64/65/96/512/2048/4096 and N=1/32/64/65, noncontiguous and
unaligned views. Native FP32 casts cover every target pair plus aligned/ragged
small cases; separate BF16/FP32-input full-accumulation probes are small-shape
qualification only. FP16 rejection and signed-zero comparison controls are tested.

## Performance

Before/after CUDA-event measurements use seed 642, one device, 10 warmup calls
and 30 individual samples per API/shape; record median, p95, min/max and peak
allocated memory. Calls include public-wrapper work, transpose/materialization,
padding, allocation and launches; no CUDA graphs or clock locking. This is a
microbenchmark, not complete RL-iteration performance or a cuBLAS speedup claim.
The original forward_fp32 API cannot run; compare the repaired API with the
output-equivalent original `gemm(a,b).float()` and keep BF16 `.linear` separate.
Base is `49941801f942428fd52ac5b66a1b0c070418724e`; repaired code is
`b3ddb9df22d7810116ad168182e689f6e699d843`. Both benchmark production-source
diffs are empty. Binary SHA256 values and full small timing summaries (including
p95 and memory) are in [task12-projection-gemm.json](task12-projection-gemm.json).
Medians in milliseconds:

| M×K×N | BF16 before | BF16 after | FP32 widen before | Native FP32 after |
| --- | ---: | ---: | ---: | ---: |
| 1×2048×512 | 0.094048 | 0.095584 | 0.114432 | 0.107488 |
| 129×2048×512 | 0.102528 | 0.102832 | 0.116192 | 0.111104 |
| 129×4096×2048 | 0.180528 | 0.179712 | 0.248224 | 0.245936 |
| 129×2048×32 | 0.116704 | 0.117952 | 0.117664 | 0.114752 |
| 129×2048×1 | 0.091200 | 0.091920 | 0.094384 | 0.092112 |
| 1×2048×248320 | 1.951088 | 1.942544 | 5.755360 | 5.759792 |

BF16 median changes range from -0.45% to +1.63% in this sample. These are
single-run measurements with variability, not a statistical regression gate.
LM-head native FP32 peak allocation is 3,181,010,944 bytes versus 3,116,944,384
for original widening: about **61.10 MiB extra**, from the wider padded-M
temporary. BF16 peak allocation is unchanged at 2,099,329,024 bytes for that case.

To reproduce the original comparison with the new harness, after setting the
same build environment above (choose separate paths if they already exist):

```bash
TASK12_CHECKOUT=$PWD
git worktree add --detach ../task12-baseline 49941801f942428fd52ac5b66a1b0c070418724e
(cd ../task12-baseline && python setup.py build_ext --inplace && \
  PYTHONPATH=. python "$TASK12_CHECKOUT/benchmarks/operators/gemm/benchmark_projection_gemm.py" \
  --json artifacts/task12/timing-before.json)
python -m benchmarks.operators.gemm.benchmark_projection_gemm --json artifacts/task12/timing-after.json
```

## Pending and MUSA handoff

No MUSA hardware was available. Native MUSA currently uses ascending-K FP32
accumulation with a final cast; it is not the CUDA BF16 tree profile. Start with
the shared accuracy/batch/chunk/layout/dtype cases, excluding CUDA tree-specific
cast/rounding probes, and qualify MUSA output casts against its own source
contract. Use a single idle S5000 device, record actual board/driver/compiler/
torch_musa, native extension and Triton availability, and retain first mismatch.
Do not report skipped cases as MUSA passes. Cross-vendor equality is Task 70.
The shared MUSA accuracy probes use CPU FP64 mathematical references.

```bash
# In a separately prepared matching MUSA checkout/environment:
MUSA_VISIBLE_DEVICES=<idle-device> TASK12_DEVICE=musa \
  python -m pytest tests/ops/gemm/test_projection_gemm.py -q -rs \
  -k 'not native_fp32_is_tree_output_cast and not canonical_k_rounding_is_observable'
```

Also pending: H200 and other software stacks, multi-device/multi-node/TP K-shard
acceptance, cuBLASLt qualification, non-finite/overflow/subnormal arithmetic,
zero-sized dimensions, K>4096 (including #679's deep-stack/index cases),
production distributions/checkpoints and broad rowwise-FP32 performance.
Hopper operator results do not close H+M delivery or Task 13.

[baseline-rule]: https://github.com/RL-Align/RL-Kernel/issues/642#issuecomment-6626542520
[pr450]: https://github.com/RL-Align/RL-Kernel/pull/450
[pr679]: https://github.com/RL-Align/RL-Kernel/pull/679
[config]: https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/345d467b117d1fd5eda9b879a5988de9ae5f47a7/config.json
[model]: https://github.com/huggingface/transformers/blob/v5.13.1/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py
