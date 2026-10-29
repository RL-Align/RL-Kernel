# Item 18: packaged ROCm-image revalidation — 2026-10-29

This update belongs to the existing item-18 PR #616, not a new roadmap task.
The branch also includes current main `8588158650aff58c9c1f527674de94bad06c3d10`
through merge `c9c186e`.

## Why compatibility was needed

The new MI300X image ships aiter 0.1.16's self-contained HCA implementation.
It exposes `flydsl_hca_compress_attn` rather than the later split-forward
entrypoint and does not ship the later `fused_compress_attn_common` module.
The original source pin correctly rejected that unsupported ABI.

The adapter now recognizes **two exact upstream source hashes**. For the
packaged composite ABI, it calls the existing two-kernel library entrypoint
with `k_split_num_waves=8` and `slice_size=64`, bypassing its N-based tuning.
Its norm/RoPE kernel is intrinsically one wave. No numerical kernel or library
package was changed. Unknown source revisions still fail before a launch.

The selected ABI and source hashes are included in the implementation
fingerprint. This does not claim byte identity between library revisions.
Training and rollout still have to agree on one exact fingerprint.

## Acceptance

**240 passed, zero skipped**, twice (33.52 s and 33.31 s; one warning each):

* 67 C128 CPU contract/reference tests;
* 31 actual ROCm acceptance tests;
* 142 shared operator-contract regressions.

The existing numerical, gradient and BI assertions were retained. Instrumentation
now observes the packaged ABI's actual FP32 scratch pool as well as the newer
ABI's returned pool. Both are tested for exact bytes between full and one-group
launches, with explicit fixed pooling/norm configuration checks.

Coverage includes independent FP64 forward math, native/mLite VJPs,
microbatch/backward ordering, ragged and permuted multi-request groups,
padding, metadata lifetime, fresh processes, nondefault streams, and the real
two-process CPU Gloo ownership fixture. That last fixture uses one shared GPU
and is **not RCCL or multi-GPU CP signoff**.

```bash
RLK_MLITE_DSV4_BACKWARD_SOURCE=/workspace/upstream/megatron_backward.py \
python3 -m pytest tests/contracts/operators tests/models/deepseek_v4 -q \
  --disable-warnings --basetemp=/outputs/item18-compat-20261029/repeat-temp \
  --junitxml=/outputs/item18-compat-20261029/repeat-tests.xml
```

Ruff checks pass. Artifacts and tested source hashes are under
`/outputs/item18-compat-20261029/`.

## Native provider and timings

MI300X/gfx942, Torch `2.11.0+gitd0c8b1f`, HIP `7.2.53211`,
Triton `3.6.0`, FlyDSL `0.2.0`.

* HCA source: `e7aaf267eb78dd16c24f2967c6f5f825781a8dbac81d837dad74f8218a2aad5a`.
* Tensor shim: `a11d02c7fc89a1b8452b8da30f9924a3168f36b893ee32eeed4469578af302fc`.
* Pinned mLite VJP: `6a1e78ca89b83f56391b5e5ac409e765338f1e3a625251d0d4c877c1e80b4a68`.
* Implementation: `f4fc6a7b73394beecb47cf99a802359002862ce4d499f39abed586dd57af7f78`.

30 warmed repetitions, synchronized averages including Python/launch costs:

| Complete groups | Tokens | Admission µs | Checked forward µs | Raw two-kernel backend µs |
|---:|---:|---:|---:|---:|
| 1 | 128 | 317.30 | 445.88 | 63.49 |
| 8 | 1024 | 946.82 | 434.30 | 60.56 |
| 32 | 4096 | 6755.31 | 443.34 | 61.72 |
| 128 | 16384 | 20840.99 | 501.40 | 68.35 |

The benchmark JSON retains the older field name `pool_us`; that field actually
times `backend.forward`, including **both pooling and norm/RoPE**, not pooling
alone. Admission is a separate host-validated operation. No full-model throughput
or speedup claim is made.

This remains the eight-wave HCA profile. It does not turn item 21's different
single-wave C128 state arithmetic into the same profile, and it does not add
Graph/cache-runtime signoff.
