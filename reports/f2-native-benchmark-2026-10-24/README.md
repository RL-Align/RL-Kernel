# Native eager F2 baseline on A100

Compared exact AST-extracted functions from the pinned Transformers source (SHA-256 recorded in JSON): `MuseGlimmerRMSNorm`, `rotate_half`, and `apply_rotary_pos_emb`. The optional Hub-kernel decorator is explicitly removed to measure the native eager body. No model weights or full-model execution are involved.

Shared CUDA-built BF16 tables, full 32/2 heads, D=128, BF16 inputs, eps=1e-5, Q scale=3.87. Both paths perform local table lookup inside the timed call; table construction and registration are excluded. Native code has no extra position-bounds scan. Fused measurements use the validated public API including its checks.

Timing: 5 warmups, 21 samples per case, synchronized wall time including Python, allocation, launch and device work. Backward-only builds a fresh forward graph outside timing; training measures forward plus `autograd.grad` together. No graph capture, torch.compile or fast math. Separate medians need not add up. Samples are one run, not confidence intervals.

| T | Layer | Native F+B µs | CUDA F+B µs | Speedup | Triton F+B µs | Speedup |
|---:|---|---:|---:|---:|---:|---:|
| 1 | global | 706.0 | 206.9 | 3.41× | 340.5 | 2.07× |
| 1 | local | 1271.4 | 284.1 | 4.48× | 439.1 | 2.90× |
| 128 | global | 748.3 | 218.8 | 3.42× | 357.5 | 2.09× |
| 128 | local | 1322.1 | 293.1 | 4.51× | 430.3 | 3.07× |
| 2048 | global | 1021.1 | 231.0 | 4.42× | 362.8 | 2.81× |
| 2048 | local | 1582.9 | 342.8 | 4.62× | 451.2 | 3.51× |

## T=2048 separate measurements

| Layer | Backend | Forward µs | Backward µs |
|---|---|---:|---:|
| global | native_eager | 281.4 | 778.4 |
| global | cuda | 110.7 | 168.8 |
| global | triton | 144.2 | 228.3 |
| local | native_eager | 532.5 | 1058.8 |
| local | cuda | 178.6 | 242.4 |
| local | triton | 221.8 | 232.0 |

Maximum measured relative L2 differences versus native eager: forward 5.43751e-06, backward 4.31127e-05. This smoke check uses 0.03, not a final upstream-approved threshold. Native reduction/autograd arithmetic is not the strict fixed-tree/VJP-v2 target and bitwise native equality is not asserted. The separate 120-test suite supplies strict parity and independent FP64 coverage.

Environment: A100 80GB PCIe SM80, PyTorch 2.14.1+cu130, CUDA toolkit 13.0.88, Triton 3.8.0. Raw samples: `rlk-f2-native-benchmark.json`; console: `rlk-f2-native-benchmark.log`.

Reproduce from repo root with the pinned source file:

```bash
PYTHONPATH=. python benchmarks/benchmark_qknorm_scaled_rope.py \
  --model-source /path/to/modeling_muse_glimmer.py --out results.json
```

The post-run script only adds no_grad to the untimed numerical-error calculation to suppress a warning; measured paths are unchanged. This is an operator-chain native-eager baseline, not an optimized torch.compile baseline, attention/full-model benchmark or training-throughput claim. Cross-GPU validation remains pending.
