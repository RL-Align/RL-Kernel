# MiniMax-H3 RMSNorm and AdaLN Modulation

## Summary

`h3_rmsnorm` covers every RMSNorm in MiniMax-H3 (RFC #420, WS1 step 4). That is the block
`norm1`/`norm2`, the token-refiner norms, the refiner `final_norm` and `norm_out.norm`. All of
them are `nn.RMSNorm(5376, eps=1e-5)` with a BF16 affine weight. In a transformer block and in
`norm_out`, the normalised rows are immediately modulated by per-row AdaLN parameters, and the
op fuses that step:

```text
n   = rms_norm(x, weight, eps)
out = n * (1.0 + scale[index]) + shift[index]
```

`index` is `adaln_indices` in a block (rows of the [projection](h3-adaln-projection.md) table)
and `timestep_indices` in `norm_out`. `shift`/`scale` are `(R, H)` row views of the AdaLN table,
gathered inside the kernel, so the `(S, H)` tensors that
[`adaln_row_gather`](h3-adaln-row-gather.md) materialises are never needed.

## Entry Point

```python
op = kernel_registry.get_op("h3_rmsnorm", device="cuda")
n = op(x, weight)                                              # plain RMSNorm
out = op.forward_modulated(x, weight, shift_msa, scale_msa, adaln_indices)
```

## Backends

| Backend | Wrapper | Native symbols | Status |
| --- | --- | --- | --- |
| CUDA (SM80+, validated on SM100) | `rl_engine.backends.cuda.model_specific.minimax_h3.rmsnorm.H3RMSNormCudaOp` | `rl_engine._C.h3_rmsnorm_{forward,backward}` | Bitwise equal to `nn.RMSNorm` and to the diffusers modulation |
| PyTorch reference | `rl_engine.reference.minimax_h3.rmsnorm.NativeH3RMSNormOp` | n/a | `forward*`: provider path; `forward*_fp32`: FP64 golden (the modulated one stores `norm(x)` and `1 + scale` in BF16, straight-through, as the model does) |
| ROCm | n/a | n/a | Falls back to the PyTorch reference |

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` | `(..., S, N)` | bf16 / fp16 / fp32 | `N % 4 == 0`; H3: `N = 5376` |
| `weight` | `(N,)` | `x.dtype` | |
| `shift`, `scale` | `(R, N)` row views, same stride | `x.dtype` | Unit column stride |
| `index` | `(S,)` | int64 | In `[0, R)`; one entry per position, shared across the batch |

Out-of-range indices, mismatched dtypes or shapes, a non-positive eps, and `N % 4 != 0` all fail
closed.

## Numerics

- **Statistics (contract `h3-rmsnorm-v1`).** The kernel replays PyTorch's own
  `vectorized_layer_norm_kernel<T, float, rms_norm>` (torch `cf30153`):
  - one `(32, 4)` block per row, reading 4-element vectors;
  - thread `t` sums vectors `t, t+128, …` in order;
  - a shuffle-down tree (16 → 1), then a cross-warp tree;
  - `rsqrtf(Σx²/N + eps)`, then `w * (rstd * x)` with one cast.

  The plain norm is therefore **bitwise equal to `nn.RMSNorm`**. Rows are independent of batch
  size and position.
- **Modulation.** `1 + scale`, `n * (…)` and `+ shift` are each rounded to the tensor dtype,
  exactly where the eager expression rounds. The fused output is **bitwise equal to diffusers'
  `norm(x) * (1.0 + scale.index_select(0, i)) + shift.index_select(0, i)`**.
- **Backward.** Everything is in FP32 with one cast per output and no atomics:
  - `dx` is row-local, using the same reduction tree;
  - `dweight` is folded over fixed 256-row tiles in ascending order;
  - `dshift`/`dscale` are segmented sums over positions sorted stably by table row, in the
    same scheme as [`adaln_row_gather`](h3-adaln-row-gather.md).

  Diffusers' `index_select` backward accumulates `dshift`/`dscale` with BF16 atomics, so it is
  non-deterministic and about 15–20× less accurate (its error varies from run to run).

## Performance Notes

```bash
python benchmarks/models/benchmark_h3_conditioning.py --op h3_rmsnorm
```

B200, block `norm1` + MSA modulation, B = 1, H = 5376, BF16:

| S | CUDA fwd | diffusers fwd | CUDA bwd | diffusers bwd |
| --- | --- | --- | --- | --- |
| 4097 | 0.08 ms | 0.12 ms | 1.23 ms | 0.85 ms |
| 32768 | 0.35 ms | 0.76 ms | 2.18 ms | 4.40 ms |
| 131072 | 1.21 ms | 2.90 ms | 6.17 ms | 17.55 ms |

The forward is a single pass that never materialises the gathered rows. At small S the backward
is dominated by the fixed cost of the stable sort and tile setup. Backward timings and peak
memory exclude leaf creation and the forward pass. Candidate/provider execution order alternates
each iteration and is recorded in the report.

## Evidence

![h3_rmsnorm on B200: latency and backward accuracy](../../reports/experiments/h3-rmsnorm-b200/figure.png)

The data is in [`report.json`](../../reports/experiments/h3-rmsnorm-b200/report.json), written by
`tools/validation/models/h3_evidence.py` from a clean tree at commit `80e4609`. It also records:

- bitwise equality with `nn.RMSNorm` for all four pinned norm weights;
- bitwise equality with diffusers for the modulation;
- row invariance.

## Existing implementations (RFC #420 reuse rule)

![norm_modulate vs existing implementations](../../reports/experiments/h3-prior-art-b200/norm_modulate.png)

| Implementation | Batch-invariant | size 4097: fwd err / worst grad err / fwd+bwd | size 32768: fwd err / worst grad err / fwd+bwd |
|---|---|---|---|
| diffusers composition (F.rms_norm + index_select modulation) | **no** (param/table grads not repeatable) | 8.6e-03 / 4.5e-02 / 872 µs | 7.5e-03 / 1.9e-01 / 4434 µs |
| torch F.rms_norm (no modulation) | yes | 2.1e-03 / 2.1e-03 / 331 µs | 2.0e-03 / 2.8e-03 / 850 µs |
| TE 2.20.2 RMSNorm (no modulation) | yes | 2.1e-03 / 2.1e-03 / 471 µs | 2.0e-03 / 2.8e-03 / 1597 µs |
| Liger 0.8.4 modulated RMSNorm + index_select | **no** (param/table grads not repeatable) | 6.6e-03 / 4.7e-02 / 810 µs | 6.9e-03 / 2.0e-01 / 3911 µs |
| Liger 0.8.4 modulated RMSNorm + rl-kernel row gather | yes | 6.6e-03 / 6.5e-03 / 1824 µs | 6.9e-03 / 8.0e-03 / 4763 µs |
| SGLang 0.5.21 fused_norm_scale_shift (forward only) | yes | 4.4e-03 / — / 85 µs (fwd only) | 4.1e-03 / — / 436 µs (fwd only) |
| rl-kernel H3RMSNormCudaOp.forward_modulated | yes | 8.6e-03 / 4.2e-03 / 1093 µs | 7.5e-03 / 5.0e-03 / 2653 µs |

Errors are max|err| / max|ref| against the same computation in FP64; latency is the median
forward + backward time on an otherwise idle B200. Batch invariance is bitwise and covers
three checks: every row computed alone vs inside full batches of 64, 257 and 2048 rows; the full
131072-token batch vs sub-batches that together cover every row; and a dense batch-size sweep. A
"no" means that at least one row, sub-batch or gradient differed. [`norm_modulate.json`](../../reports/experiments/h3-prior-art-b200/norm_modulate.json)
was written from a clean tree at `ee83dec` by

```bash
python tools/validation/models/h3_prior_art.py --op norm_modulate --out reports/experiments/h3-prior-art-b200/norm_modulate.json
python tools/validation/models/plot_h3_prior_art.py reports/experiments/h3-prior-art-b200/norm_modulate.json
```

Libraries that do not import are skipped and recorded as unavailable in the report.

## Tests

```bash
export RL_KERNEL_H3_WEIGHTS=<dir written by tools/weights/prepare_h3_weights.py>
python -m pytest tests/models/minimax_h3/test_h3_rmsnorm.py -v               # operator
python -m pytest tests/models/minimax_h3/test_h3_conditioning_e2e.py -v      # end to end, incl. block norm1 + modulation
python tools/validation/operators/check_operator.py --op h3_rmsnorm --candidate cuda --device cuda \
    --dtype bf16 --batch 2 --seq 257 --normalized-dim 5376 --check-grad
python tools/validation/models/h3_evidence.py --op h3_rmsnorm --out reports/experiments/h3-rmsnorm-b200/report.json
python tools/validation/models/plot_h3_evidence.py reports/experiments/h3-rmsnorm-b200/report.json
```

## Known Limitations

- Bitwise parity with `nn.RMSNorm` is tied to PyTorch's vectorized layer-norm path (`N % 4 == 0`,
  aligned tensors). Other shapes are rejected rather than approximated.
- The token-refiner blocks' own norm weights live in shard 14, which is not pinned. They are the
  same module (`nn.RMSNorm(5376, eps=1e-5)`) as the four pinned norms tested here.
- There is no ROCm kernel. ROCm dispatches the PyTorch reference.
