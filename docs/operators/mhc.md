# DeepSeek-V4 mHC (pre, post, head) and Layer End

## Summary

Batch-invariant DeepSeek-V4 manifold-constrained hyper-connections (mHC) on ROCm,
DSV4 RFC tasks 1–4 and 8 ([#508](https://github.com/RL-Align/RL-Kernel/issues/508)).
DeepSeek-V4 keeps `n = 4` residual streams of hidden size `C` per token. Before each sublayer, mHC pre mixes them into one input; after
it, mHC post mixes the output back; after the last layer, mHC head contracts the
streams for the final norm and lm_head. A token's result must not depend on the
other tokens in the batch, so that rollout and training produce the same bits.

## Entry Point

```python
from rl_engine.runtime.registry import kernel_registry

pre = kernel_registry.get_op("mhc_pre", device="cuda")
post_mix, comb_mix, layer_input = pre(residual, fn, scale, base)

post = kernel_registry.get_op("mhc_post", device="cuda")
new_residual = post(y, residual, post_mix, comb_mix)

head = kernel_registry.get_op("mhc_head", device="cuda")
hidden = head(residual, head_fn, head_scale, head_base)

from rl_engine.models.deepseek_v4.layer_end import mhc_layer_end

stages = mhc_layer_end(residual, head_fn, head_scale, head_base,
                       norm_weight, lm_head_weight, target_ids)
# {"head", "final_norm", "logits", "logprob"}
```

`rl_engine.ops.mhc` exposes `mhc_pre`, `mhc_post` and `mhc_head` as functions
that look the op up in `kernel_registry` for the input's device. The arithmetic
each backend follows is recorded in `rl_engine/contracts/operators/mhc.py`
(`describe("mhc_pre")` and so on).

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| CUDA | — | — | Not implemented; the registry raises |
| ROCm | `AiterMHCPreOp`, `AiterMHCPostOp`, `AiterMHCHeadOp` in `rl_engine/backends/rocm/mhc/aiter_mhc.py` | aiter `mhc_pre_dsv4`, `mhc_post_dsv4`, `mhc_head_dsv4` (Triton) | Forward batch invariant; backward is aiter's PyTorch recompute |
| PyTorch fallback | `NativeMHCPreOp`, `NativeMHCPostOp`, `NativeMHCHeadOp` in `rl_engine/reference/mhc/` | — | fp32 reference following DeepSeek-V4-Flash-0731 `inference/model.py`; CPU only |

The ROCm ops need an aiter build with the DeepSeek-V4 mHC ops (aiter PR #5492 or
later; measured with aiter `326cffa` and `AITER_TRITON_ONLY=1`).

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `residual` | `[..., n, C]` | bf16 | |
| `fn` | `[2n + n², n·C]` | fp32 | `hc_attn_fn` / `hc_ffn_fn` |
| `scale`, `base` | `[3]`, `[2n + n²]` | fp32 | |
| `post_mix`, `comb_mix` | `[..., n]`, `[..., n, n]` | fp32 | `post_mix` may also be `[..., n, 1]` |
| `layer_input`, `y`, `hidden` | `[..., C]` | bf16 | |
| `head_fn`, `head_scale`, `head_base` | `[n, n·C]`, `[1]`, `[n]` | fp32 | |
| `norm_weight` | `[C]` | fp32 | |
| `lm_head_weight` | `[V, C]` | bf16 | as `head.weight` in the checkpoint; other dtypes are rounded to bf16 |
| `target_ids` | `[T]` | int64 | |

`comb_mix` is indexed [source stream, destination stream]. Leading dimensions are
flattened into the token count M.

## Dispatch Behavior

On ROCm the registry hands out only the aiter ops: a missing aiter raises instead
of falling back to an op whose bits change with the batch. Inputs outside the
tensor contract (other dtypes, n ≠ 4, mismatched shapes) raise as well. CPU gets
the PyTorch references; CUDA, MUSA and NPU have no mHC backend, and `get_op`
raises there. The semantic catalog declares the aiter ops for bf16 on
ROCm with `batch_invariant: True` and `backward_batch_invariant: False`.

aiter chooses mhc_pre's launch config by M and, on gfx942, splits K into 8, 4 or
2 segments from M = 1024 on, so a row's bits change with the batch. The ops pass
the config through aiter's public `config` argument instead, chosen by
`rl_engine/backends/rocm/config_utils.py`:

- The GPU arch is that of the input's device. On gfx942 with n = 4, C = 4096
  the config comes per M from
  `configs/gfx942/mhc_pre/MHC_PRE-n=4-C=4096.json` and
  `configs/gfx942/mhc_head/MHC_HEAD-n=4-C=4096.json` (`M_LEQ_x` buckets, the
  smallest x ≥ M, else `"any"`). Other GPUs and shapes use one config for every
  M: `FIXED_PRE_CONFIG` (no K split) and aiter's own head config
  (`FIXED_HEAD_CONFIG`).
- Checked when a table is loaded, with `ValueError` otherwise: every key is
  `M_LEQ_x` or `"any"`, the `"any"` entry exists, and every entry agrees on the
  parameters that change the output bits. Measured on MI325X, Triton 3.7:

| Op | Free to change per M | Change the bits |
| --- | --- | --- |
| mhc_pre | `BLOCK_M` (16–128), `BLOCK_C` (≤ 128), `num_warps`, `num_stages`, `waves_per_eu` | `NUM_KSPLIT`, `BLOCK_K`, `kpack`, `BLOCK_C=256`, `BLOCK_M=16` with `num_warps=8` |
| mhc_head | `num_stages`, `waves_per_eu` | `BLOCK_M`, `BLOCK_K`, `BLOCK_C`, `num_warps`, `kpack` |

Any fixed `NUM_KSPLIT` used for every M is batch invariant. The mhc_pre table
splits K into 4 segments with `BLOCK_K=128` at every M. mhc_head's public
`config` cannot set `num_warps` or `num_stages`, so its table only changes
`waves_per_eu`. mhc_post's config does not depend on M and is aiter's.

`rl_engine.backends.rocm.config_utils.selected_configs()` lists the configs that
ran in the process, one record per GPU arch, op, table file and bucket: `arch`,
`op`, `file`, `bucket` (`M_LEQ_x`, `"any"`, or `"default"` without a table),
`source` (the table path or `"default"`), `config`, and `m_min` / `m_max`, the
smallest and largest M that used it.

## Accuracy

- The PyTorch references compute in fp32 and cast `layer_input`, the post output
  and the head output to bf16, as DeepSeek-V4 does; `forward_fp32` keeps fp32.
  The aiter ops match `forward_fp32`, including the residual gradient, within
  the WS1 contract tolerance (`rl_engine/contracts/profiles/precision/ws1.json`,
  bfloat16 `reduction` cells of `forward_accuracy` and `gradient_accuracy`).
- The RMS factor is `rsqrt(mean(x²) + eps)`, as in DeepSeek-V4 and aiter.
- aiter rounds `fn` and `head_fn` to bf16 before the projection; the checkpoint
  stores them in fp32 and almost none of their values are exact in bf16. The
  reference projects in fp32. Which precision the project uses is open.
- aiter sums the four streams of `layer_input` as
  `fma(w2, x2, w0·x0) + fma(w3, x3, w1·x1)` in fp32 and rounds to bf16 once.
- `mhc_layer_end`'s lm_head uses `batch_invariant_linear` with bf16 inputs and
  fp32 output (see [Batch-Invariant Linear](batch-invariant-linear.md)).

## Performance Notes

Forward device time of mHC pre, n = 4, C = 4096, bf16 residual, fp32 `fn`, 20
Sinkhorn rounds; ms per call from GPU events, median over 5 trials of 50 calls.
Baseline: aiter's own config (not batch invariant). Ours: the `mhc_pre` op with the
gfx942 table. Three runs on two nodes; each cell gives the range of the three
medians, and the ratio column the range of ours / baseline per run.

| M | baseline: aiter's own config (ms) | ours: `mhc_pre` op (ms) | ours / baseline (ratio) |
| --- | --- | --- | --- |
| 1 | 0.1985–0.1986 | 0.0965–0.1176 | 0.49–0.59 |
| 16 | 0.2431–0.2432 | 0.0982–0.1183 | 0.40–0.49 |
| 17 | 0.2482–0.2484 | 0.0991–0.1181 | 0.40–0.48 |
| 64 | 0.2503–0.2514 | 0.0998–0.1192 | 0.40–0.47 |
| 100 | 0.2546–0.2553 | 0.0987–0.1187 | 0.39–0.46 |
| 256 | 0.2549–0.2560 | 0.0986–0.1181 | 0.39–0.46 |
| 1024 | 0.0910–0.0962 | 0.0996–0.1196 | 1.09–1.24 |
| 2048 | 0.0910–0.0961 | 0.0991–0.1184 | 1.07–1.23 |
| 3000 | 0.1315–0.1330 | 0.1000–0.1179 | 0.76–0.89 |
| 4096 | 0.1395–0.1424 | 0.1158–0.1224 | 0.82–0.86 |
| 8192 | cannot launch: needs 128 KB LDS, gfx942 has 64 KB | 0.1978–0.2076 | n/a: baseline cannot launch |
| 16384 | 0.4458–0.5202 | 0.4047–0.5137 | 0.91–0.99 |
| 32768 | 0.8665–0.8963 | 0.7747–0.8114 | 0.89–0.91 |
| 65536 | 1.6500–1.7960 | 1.4913–1.6645 | 0.90–0.93 |

Forward device time of mHC head, n = 4, C = 4096, bf16 residual, fp32 `head_fn`;
ms per call from GPU events around a CUDA graph of 20 calls, median over 5
replays, one run. Baseline: aiter's own head config, which is itself batch
invariant. Ours: the `mhc_head` op with the gfx942 table. The output bits of ours
and the baseline were the same at every M.

| M | baseline: aiter's head config (ms) | ours: `mhc_head` op (ms) | ours / baseline (ratio) |
| --- | --- | --- | --- |
| 1 | 0.8195 | 0.7440 | 0.91 |
| 16 | 0.8319 | 0.7454 | 0.90 |
| 17 | 0.8311 | 0.7476 | 0.90 |
| 64 | 0.8374 | 0.7497 | 0.90 |
| 100 | 0.8399 | 0.7458 | 0.89 |
| 256 | 0.8396 | 0.7446 | 0.89 |
| 1024 | 0.9042 | 0.7780 | 0.86 |
| 2048 | 0.9101 | 0.7815 | 0.86 |
| 3000 | 0.9121 | 0.7848 | 0.86 |
| 4096 | 0.9153 | 0.7879 | 0.86 |
| 5000 | 1.7825 | 1.3547 | 0.76 |
| 8192 | 1.8948 | 1.4265 | 0.75 |
| 16384 | 3.8658 | 2.8688 | 0.74 |
| 32768 | 6.7690 | 5.1442 | 0.76 |
| 65536 | 12.0018 | 10.7097 | 0.89 |

Measured before the port, on the KLR branch with the same aiter kernels (aiter
`326cffa`, `AITER_TRITON_ONLY=1`), the same launch-config tables and the same
benchmark code, on AMD Instinct MI325X (gfx942), Triton 3.7, torch 2.12, ROCm 7.14.
Rerun the commands below to measure this branch.

```bash
python benchmarks/operators/mhc/benchmark_mhc_pre.py --output mhc_pre.md
python benchmarks/operators/mhc/benchmark_mhc_head.py --output mhc_head.md
```

`benchmarks/tuning/tune_mhc_pre_config.py` and
`benchmarks/tuning/tune_mhc_head_config.py` regenerate the tables.

## Tests

```bash
python -m pytest tests/reference/test_mhc_reference.py tests/ops/mhc \
    tests/contracts/operators/test_mhc_contract.py \
    tests/backends/rocm/test_rocm_launch_config.py \
    tests/backends/rocm/test_rocm_mhc_pre.py tests/backends/rocm/test_rocm_mhc_post.py \
    tests/backends/rocm/test_rocm_mhc_head.py \
    tests/backends/rocm/test_rocm_mhc_batch_invariance.py tests/models/deepseek_v4 -v
```

`tests/backends/rocm/test_rocm_mhc_batch_invariance.py` checks mHC pre, mHC post
and mHC head, and `tests/models/deepseek_v4/test_layer_end_batch_invariance.py`
the layer end, with `rl_engine/validation/common/batch_invariance.py`: a row alone
and in batches of M = 1 to 16384 (including the WS1 manifest's 8 and 59),
chunking, reordering and padding, other input strides, repeated runs, and forward
with and without grad. The mHC op tests also record which launch configs ran and
run at C = 7168, which has no table, so the default configs are covered.
`test_rocm_mhc_pre.py` isolates the projection, RMS, Sinkhorn and stream
aggregation with exact inputs, and checks the projection and RMS under chunking,
reordering, padding and other strides as well. Accuracy against the fp32
references uses the WS1 contract tolerance. Unsupported inputs and the layer end
against a PyTorch reference are tested as well, and empty batches for mHC pre,
mHC post, mHC head and the batch-invariant linear.

## Known Limitations

- Backward is not batch invariant: aiter recomputes it in PyTorch. Those tests are
  marked xfail.
- The tables are measured for gfx942 with Triton 3.7 and DeepSeek-V4-Flash's
  n = 4, C = 4096 only; other shapes and GPUs use the default configs.
- aiter's own mhc_pre config cannot launch for M = 8192–16383 on gfx942 with
  C = 4096; the tables avoid it.
- No CUDA kernel. The registry has no mhc_pre, mhc_post or mhc_head backend on
  CUDA, MUSA or NPU, so `get_op` raises there.
- `mhc_layer_end` does not support an empty batch: with zero tokens the PyTorch
  RMSNorm reference it uses for the final norm raises
  (`rl_engine/reference/norm/rms_norm.py` reshapes the empty tensor with
  `reshape(*shape[:-1], -1, 32)`, whose `-1` cannot be inferred).
- aiter `326cffa` declares M as `tl.constexpr` in `_mhc_fused_kernel` and
  `_mhc_fused_split_kernel` (mHC pre) and `_mhc_head_kernel` (mHC head), so every
  new token count M compiles these kernels again; the tuning runs saw about 10 s
  per config and M on MI325X. `_mhc_reduce_apply_kernel`,
  `_mhc_asymmetric_sinkhorn_kernel` and `_mhc_post_kernel` take M as a runtime
  argument.
- CUDA/HIP graph capture is not covered here; it belongs to the framework
  integration (DSV4 RFC task 7, #510).
- Open for the RFC author: the RFC asks for a controller RMS of
  `1 / (sqrt(mean) + eps)`, while aiter and the DeepSeek-V4 reference use
  `rsqrt(mean + eps)`; and aiter rounds `fn` to bf16 before the projection. The
  tests record aiter's behavior for both.
