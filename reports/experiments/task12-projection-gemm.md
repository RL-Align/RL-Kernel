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

## Historical implementation validation

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

## Review-fix validation

At tested commit `e505cf704db5afc31da0720711f55fc8dd8c987d`, reran on one H100:

- Projection matrix **159 passed** (9.74 s), including the added generic canonical
  mathematical probe. All 20 target accuracy cases and raw-bit checks pass;
  maximum absolute FP64-reference error is again 0.037068993886350654. Tolerances
  and CUDA arithmetic are unchanged.
- Independent tree profiles and the same existing forward regressions **38 passed**
  (6.88 s); no Task13 gradient test/body changed.
- Measurement lifetime/accounting **2 passed** (2.35 s), including a real CUDA
  allocation. CPU-only run **16 passed, 1 real-CUDA skip** (3.20 s), covering
  lifecycle, root/staged/unstaged/index/environment provenance, explicit device
  marks and the CPU FP64 oracle policy. The skipped allocation ran successfully
  in the separate H100 run. No torch_musa import or MUSA runtime was simulated.
- Historical 26-output binary equality, kernel-profiler trace, CPU layout and
  attention-projection results above are retained evidence, **not rerun here**.
- Final configured Black 24.4.2 / isort 5.13.2 / flake8 7.0.0, mypy on the
  benchmark, provenance helper and CUDA wrapper, JSON/history/hash checks,
  whitespace/size checks, `git diff --check`, and strict MkDocs build pass.

Commands (with the CUDA environment above, XML/logs outside the repository):

```bash
python -m pytest tests/ops/gemm/test_projection_gemm.py -q -rs
python -m pytest tests/ops/gemm/test_projection_gemm_profiles.py \
  tests/ops/gemm/test_det_gemm.py -q -k \
  'canonical_midpoint or non_power_of_two_profiles or rhs_transposed or forward_batch_invariance or forward_chunked_prefill or forward_padding_invariance or forward_correctness or triton_ragged_tiles or triton_tree_matches_python_reference_forward or triton_final_tree_level_direct_output'
python -m pytest tests/benchmarks/test_projection_gemm_measurement.py -q -rs
CUDA_VISIBLE_DEVICES='' python -m pytest tests/benchmarks/test_projection_gemm_measurement.py \
  tests/benchmarks/test_projection_gemm_provenance.py \
  tests/ops/gemm/test_projection_gemm_platform_policy.py -q -rs
```

## Performance: corrected review rerun

All six benchmark shapes were remeasured on the same H100 and original software
stack, seed 642, 10 warmups and 30 CUDA-event samples per API. The **same corrected
benchmark file** ran against baseline `49941801f942428fd52ac5b66a1b0c070418724e`
and tested repair `e505cf704db5afc31da0720711f55fc8dd8c987d`. BF16 linear is reported separately.
FP32 compares original BF16-tree GEMM **then `.float()`** with repaired direct
FP32 tree output: the numerical tree is the same. The original binary has no
callable forward_fp32 binding; it is explicitly `binding_unavailable`. The
separate full-FP32-accumulation algorithm is not used in this performance table.

Each warmup result is deleted; each timed result remains alive through
`end.record()`, then its dtype is cached and the tensor deleted before the next
call. After timing is synchronized and all results are gone, memory uses an
**independent single call**: capture `memory_allocated()` baseline, reset peak,
call and synchronize with its result alive, read `max_memory_allocated()`, delete
result and synchronize. Report absolute peak, `peak_increment_bytes = peak -
baseline`, and post-release allocated bytes. Every measured case returned to its
pre-call baseline. No reserved-memory statistic is used. CUDA-event timing covers
public-wrapper copies/allocations/kernels, without CUDA graphs or clock locking;
it is not complete RL-iteration performance or a statistical performance gate.
Weakref CPU regression checks release before every warmup/timed/memory/API call
and liveness at end.record; a real-CUDA regression verifies exactly one 4 MiB
output in the independent memory pass.

The retained baseline extension was reused, not newly rebuilt. Its SHA256 matched
the original build/benchmark record; it was loaded from an independent checkout
at the baseline commit, verified SM90 compiled and no forward-FP32 binding, then
used for the new run. The repair's build_ext ran at the tested commit (unchanged
source objects, relink); its hash remains the original repaired hash. Tracked
`csrc` and `rl_engine` trees match code commit b3ddb9d. Binary identifiers:

| Binary | SHA256 |
| --- | --- |
| Original | `e8c207324aad3ca02feabad6735417c6e2001de4b0dc638922612e01232fd488` |
| Repaired | `33e4e5b80996c7c678b3c1762aad50aca493c0fe6049f6880dd2b08d77d1bc4f` |

Benchmark SHA256 `f14ef0bf18bd4dd0dcf49ebe32de2230794dec01c32b42cbc29687f385371ab3` is identical in both runs;
provenance-helper SHA256 is `805d0307f690bcecb159a109b3c8b01f715ec01491d9be919cea3bda8f486f37`.
New records preserve exact commit, committed tree, separate staged/unstaged source
diff hashes, source status and loaded binary path/hash. Both source diffs are
empty and source status clean; committed trees differ. Root-anchored pathspecs
and sanitized Git location/index variables keep this valid from subdirectories
and inherited environments. Git metadata alone is not a build attestation.
The compact [JSON](task12-projection-gemm.json) contains p95/min/max, all memory
fields, environment, timestamps and hashes of the complete external raw JSON.

New medians in milliseconds:

| M×K×N | BF16 original | BF16 repaired | Original FP32 widen | Repaired direct FP32 |
| --- | ---: | ---: | ---: | ---: |
| 1×2048×512 | 0.094448 | 0.093040 | 0.114112 | 0.106912 |
| 129×2048×512 | 0.102112 | 0.101536 | 0.115424 | 0.108096 |
| 129×4096×2048 | 0.179584 | 0.177776 | 0.249264 | 0.241024 |
| 129×2048×32 | 0.114944 | 0.114896 | 0.115072 | 0.112560 |
| 129×2048×1 | 0.089648 | 0.089680 | 0.092544 | 0.090368 |
| 1×2048×248320 | 1.936336 | 1.941632 | 5.751968 | 5.751824 |

BF16 median changes range from **-1.49% to +0.27%** in
these single-run samples. This is not evidence of an optimization or a regression
threshold. Original inputs, shape order, seed and warmup/repeat counts match.

New **incremental allocated** peaks in MiB (1 MiB = 2²⁰ bytes):

| M×K×N | BF16 linear, both | Original FP32 widen | Repaired direct FP32 |
| --- | ---: | ---: | ---: |
| 1×2048×512 | 0.625977 | 2.625977 | 2.751953 |
| 129×2048×512 | 1.375977 | 3.375977 | 3.751953 |
| 129×4096×2048 | 3.503906 | 19.503906 | 22.000000 |
| 129×2048×32 | 0.133301 | 0.024414 | 0.016113 |
| 129×2048×1 | 0.000488 | 0.001465 | 0.000977 |
| 1×2048×248320 | 61.598633 | 1031.598633 | 1092.697266 |

The focused LM head M=1,K=2048,N=248320 separates the input baseline from the
absolute peak; the baseline includes both weight layouts already allocated by
the harness:

| API | Baseline allocated MiB | Absolute peak allocated MiB | Increment MiB |
| --- | ---: | ---: | ---: |
| BF16 linear (both binaries) | 1940.003906 | 2001.602539 | 61.598633 |
| Original BF16 tree then .float() | 1940.003906 | 2971.602539 | 1031.598633 |
| Repaired direct FP32 tree output | 1940.003906 | 3032.701172 | 1092.697266 |

New absolute FP32 peaks are **3,115,951,104 bytes** original
widen and **3,180,017,664 bytes** direct FP32. Their difference
is **64,066,560 bytes = 61.09863281 MiB**, also the difference between incremental
peaks because the baselines match. BF16 peak is identical in both binaries at
**2,098,832,384 bytes** (increment 64,590,848).
The extra padded FP32 temporary remains observable with the corrected method.

**Historical data and correction:** the original method retained the previous
output across the next fn(), contaminating absolute peak allocations. The whole
old dataset is preserved under `historical` in JSON, with its original values and
provenance-metadata note; none of the new data is relabeled as the old run. For
this LM head, old FP32 absolute peaks were 3,116,944,384 / 3,181,010,944 bytes.
Both new peaks are 993,280 bytes lower (one prior logical FP32 output). The old
61.10 MiB difference is therefore **reconfirmed by new measurements**, although
the old absolute peaks were wrong. The old BF16 peak 2,099,329,024 is 496,640
bytes too high. Historical -0.45%…+1.63% latency observations remain historical;
use the new table above for this corrected measurement method.

To reproduce after preparing matching binaries in separate checkouts:

```bash
# Same environment/one idle H100 as above. Keep outputs outside either checkout.
TASK12_CHECKOUT=$PWD
(cd ../task12-pr718-artifacts/baseline-checkout && \
  PYTHONPATH=. python "$TASK12_CHECKOUT/benchmarks/operators/gemm/benchmark_projection_gemm.py" \
  --warmup 10 --repeats 30 --json ../benchmark-before.json)
python -m benchmarks.operators.gemm.benchmark_projection_gemm \
  --warmup 10 --repeats 30 --json ../task12-pr718-artifacts/benchmark-after.json
```

## Pending and MUSA handoff

**MUSA hardware validation pending.** No MUSA hardware was available. Native MUSA
currently uses ascending-K FP32 accumulation with a final cast; it is not the CUDA
BF16 tree profile. The two CUDA-specific cast/rounding tests have explicit skipif
marks with profile reasons; no command-line exclusion is required. The generic canonical-K
mathematical probe remains and uses CPU FP64 on MUSA. Qualify MUSA output casts
against its own source contract. Use a single idle S5000 device, record actual
board/driver/compiler/
torch_musa, native extension and Triton availability, and retain first mismatch.
Do not report skipped cases as MUSA passes. Cross-vendor equality is Task 70.
The shared MUSA accuracy probes use CPU FP64 mathematical references.

```bash
# In a separately prepared matching MUSA checkout/environment:
MUSA_VISIBLE_DEVICES=<idle-device> TASK12_DEVICE=musa \
  python -m pytest tests/ops/gemm/test_projection_gemm.py -q -rs
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
