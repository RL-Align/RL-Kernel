Implements the F2 `qknorm_scaled_rope` operator from #501 for Muse Glimmer: weightless per-head Q/K RMSNorm, Q-only scale, and shared-table rotate-half RoPE on local layers. Global layers skip RoPE. CUDA and Triton provide forward/backward with explicit first-order autograd, fixed FP32 reduction pairing and BF16 cast boundaries.

The backward uses FP32 normalized inputs to avoid premature `r³` underflow (VJP v2). Public local dispatch requires a verified shared RoPE table pair; loading validates its trusted identity and ordinary in-place mutation invalidates the pair. Backend selection is explicit, with no silent fallback. The SM80 shared-memory/shuffle mapping preserves the ordered addition tree.

Validation results:

| Environment | Build tested | Result | Coverage |
| --- | --- | --- | --- |
| A100 SM80, PyTorch 2.14.1+cu130, Triton 3.8.0 | Standard extension | 120 passed, 0 skipped | Forward/backward parity, independent FP64, batch/chunk invariance, bounds, table identity |
| H100 SM90, PyTorch 2.14.1+cu132, Triton 3.8.0 | Isolated F2 extension | 121 passed, 0 skipped | Same targeted suite plus A100 baseline comparison |
| Local CPU after target-branch synchronization | Python/reference/build checks | 32 passed, 87 GPU tests skipped | No local GPU available |
| A100 ↔ H100, identical inputs and shared table | VJP v2 comparison | 24 cases, 144 tensor comparisons, 0 bit mismatches | Q/K outputs, FP32 rstd, dQ/dK; local/global, token splits, head subsets |

GPU results above were collected before transplanting the uncommitted work onto `test-museglimmer`. F2 kernel and test source hashes are unchanged by synchronization; the full extension build and complete GPU suite have not been rerun on the synchronized branch.

Native eager baseline at T=2048, 32 Q heads / 2 K heads, D=128, local layers:

| Implementation | Forward µs | Backward µs | Forward + backward µs | Combined speedup |
| --- | ---: | ---: | ---: | ---: |
| Native eager | 532.5 | 1058.8 | 1582.9 | 1.00× |
| CUDA F2 | 178.6 | 242.4 | 342.8 | 4.62× |
| Triton F2 | 221.8 | 232.0 | 451.2 | 3.51× |

These are medians of 21 synchronized public-call wall-time samples after 5 warmups, including lookup, validation, allocation and launch overhead. Table construction is excluded. The baseline uses exact native eager function bodies extracted from the pinned Transformers source, with optional Hub replacement disabled. Backward-only excludes forward graph construction; combined timing measures both together, so separate medians need not add up. This is an operator-chain comparison, not whole-model throughput or a torch.compile baseline.

See `docs/contracts/qknorm-scaled-rope.md`, `reports/f2-native-benchmark-2026-10-24/README.md`, and `reports/f2-h100-2026-10-24/README.md` for arithmetic, provenance and evidence.

| Limitation | Scope |
| --- | --- |
| Platform qualification | Tested SM80/cu130 and SM90/cu132 cases only; no universal cross-architecture or ROCm claim. H100 used an isolated F2 build. |
| Arithmetic | Strict gradients need not match native eager bits; final acceptance thresholds remain for review. |
| CUDA graphs | Local bounds checking synchronizes; capture is unsupported. |
| Table ownership | Expected identity must come from a trusted producer. Concurrent/external writes bypassing PyTorch version counters are unsupported. |
| Integration | Runtime model adapters remain separate work. |

Build integration adds the F2 NVIDIA source and bindings, rejects F2 in fast-math builds, and lets PyTorch select its required C++ standard (the tested PyTorch requires C++20).

Refs #501; implements its F2 row only, not the entire issue.
