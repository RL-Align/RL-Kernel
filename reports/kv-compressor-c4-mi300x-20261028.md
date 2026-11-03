# Item 19 C4 KV compressor — MI300X validation

Branch: `feat/dsv4-kv-c4-bi`
Target: `dsv4-rocm`
Hardware: AMD Instinct MI300X VF (`gfx942`)
Environment: ROCm/HIP 7.14.60850, PyTorch 2.12.0+rocm7.14.0a20260608,
Triton 3.7.0, FlyDSL 0.3.4.1, pinned aiter commit
`326cffa38c9eda16bef19375ba55334e4d3acd59`, pinned mLite commit
`327a238239448de6771beea185dc6902a8de3eaa`.

## Test evidence

- CPU-safe C4 tests: **53 passed**.
- Real MI300X C4 tests: **26 passed**, 0 skipped, 1 existing warning, 19.49s.
- Full targeted contract + DeepSeek-V4 suite: **221 passed**, 0 skipped, 1
  existing warning, 20.33s without coverage and 26.54s with coverage.
- Scoped Python wrapper coverage: **336/371 statements (90%)** and **80/86
  branches**. Coverage does not measure FlyDSL/native GPU instructions; the
  unexecuted wrapper lines are provider-constructor and direct native VJP
  plumbing branches exercised by the real GPU tests.
- Ruff format/lint passed for all ten Python files.
- Real GPU coverage includes aiter overlap compression, true `k_per_block=64`
  paged output mapping, mLite ratio-4 functional VJP, fixed-order overlap
  gradients, streams, fresh processes/JIT isolation and CP ownership.
- The forward oracle is independent FP64 softmax/pool/RMS/rotation arithmetic.
- The backward checks compare fixed-order microbatch/reverse-order gradients to
  independent autograd and finite-difference results.

This item does not claim RCCL/multi-GPU CP, Graph replay, live cache commit,
prefill/decode runtime or full-model acceptance.

The benchmark records bind, provider forward and full operator timing with
Python validation, paged-cache gather and synchronization included. It is not a
kernel-only or throughput claim.

## Synchronized MI300X benchmark (30 repetitions)

| Complete groups | Tokens | Bind (us) | Provider (us) | Full forward (us) |
|---:|---:|---:|---:|---:|
| 1 | 4 | 429.67 | 95.44 | 440.68 |
| 8 | 32 | 944.07 | 221.16 | 781.28 |
| 64 | 256 | 2,805.11 | 533.89 | 1,096.21 |
| 128 | 512 | 5,045.03 | 380.60 | 912.69 |

These are synchronized average wall times with warmup, Python validation,
paged-cache gather and dispatch included. They are not kernel-only latency or
a speedup claim. The raw benchmark records the provider source fingerprints.
