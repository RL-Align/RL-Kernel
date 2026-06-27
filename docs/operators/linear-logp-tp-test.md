# Fused Linear LogP TP Test Runbook

This runbook describes how to validate the tensor-parallel path of
`linear_logp` on a 4-GPU Hopper machine. The script exercises the public
operator API with vocab-sharded LM-head weights:

- each rank owns `lm_head_weight[vocab_start:vocab_end]`;
- `target_ids` are global token ids and are replicated across TP ranks;
- forward merges the global log-sum-exp through TP collectives;
- backward all-reduces `hidden.grad` and keeps weight/bias gradients local.

## Test Script

```bash
scripts/test_linear_logp_tp.py
```

Launch it with `torchrun`; do not run it with plain `python` unless you are
debugging argument parsing only.

```bash
torchrun --standalone --nproc_per_node=4 scripts/test_linear_logp_tp.py
```

The script has two phases:

| Phase | Default | What it checks |
| --- | --- | --- |
| `correctness` | Always on | Builds a full materialized reference on every rank and compares TP output, `hidden.grad`, local `weight.grad`, and local `bias.grad`. |
| `stress` | `--run-stress` | Runs a larger TP-only shape without materializing full logits/reference; checks finite output/gradients, elapsed time, and peak memory. |

## Prerequisites

From the repository root:

```bash
python -m pip install -e ".[dev]"
```

For SM90 direct-backend testing, rebuild the extension on the Hopper host:

```bash
KERNEL_ALIGN_FORCE_SM90=1 python -m pip install -e .
```

Recommended environment:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
```

If the machine uses a scheduler, make sure the job really owns four GPUs on the
same node before launching.

## Quick Start

Start with a small fp32 run. This should be numerically tight and easy to debug:

```bash
torchrun --standalone --nproc_per_node=4 scripts/test_linear_logp_tp.py \
  --dtype fp32 \
  --tokens 128 \
  --hidden-size 256 \
  --vocab-size 4096 \
  --uneven-shards
```

Then run the bf16 path that matches the intended Hopper use case:

```bash
torchrun --standalone --nproc_per_node=4 scripts/test_linear_logp_tp.py \
  --dtype bf16 \
  --tokens 256 \
  --hidden-size 512 \
  --vocab-size 8192 \
  --reference-mode fp32 \
  --uneven-shards
```

Finally run a larger TP smoke without the full reference:

```bash
torchrun --standalone --nproc_per_node=4 scripts/test_linear_logp_tp.py \
  --dtype bf16 \
  --tokens 256 \
  --hidden-size 512 \
  --vocab-size 8192 \
  --reference-mode fp32 \
  --run-stress \
  --stress-tokens 4096 \
  --stress-hidden-size 2048 \
  --stress-vocab-size 32768
```

## Recommended Test Matrix

Run these in order. Stop at the first failure and keep the full terminal log.

### 1. Native TP math sanity

This bypasses registry/backend selection and directly tests the shared TP
autograd path.

```bash
torchrun --standalone --nproc_per_node=4 scripts/test_linear_logp_tp.py \
  --op-source native \
  --dtype fp32 \
  --tokens 128 \
  --hidden-size 256 \
  --vocab-size 4096 \
  --uneven-shards
```

Expected: every metric is `PASS` with max errors around `1e-4` on CUDA/NCCL.

### 2. Registry path, bf16 fp32 reference

This is the main merge gate for the current TP implementation. The fused and
Triton bf16 paths accumulate matmuls in fp32, so `fp32` reference mode compares
against the same semantic target.

```bash
torchrun --standalone --nproc_per_node=4 scripts/test_linear_logp_tp.py \
  --op-source registry \
  --dtype bf16 \
  --reference-mode fp32 \
  --tokens 256 \
  --hidden-size 512 \
  --vocab-size 8192 \
  --uneven-shards
```

Expected: `output`, `hidden_grad`, `weight_grad`, and `bias_grad` all pass.

### 3. Optional same-dtype drift check

This compares bf16 TP against a materialized full-vocab `F.linear` in the input
dtype. It is useful for understanding PyTorch full-GEMM vs shard-GEMM drift, but
the main correctness target is the fp32-accumulation reference above.

```bash
torchrun --standalone --nproc_per_node=4 scripts/test_linear_logp_tp.py \
  --op-source registry \
  --dtype bf16 \
  --reference-mode matching \
  --tokens 128 \
  --hidden-size 512 \
  --vocab-size 8192 \
  --atol 0.75 \
  --rtol 0.75
```

Record the drift numbers separately from the merge gate.

### 4. Triton wrapper delegation

Use this when Triton is installed. The TP kwargs should route through the shared
TP path instead of the local non-TP Triton kernel.

```bash
torchrun --standalone --nproc_per_node=4 scripts/test_linear_logp_tp.py \
  --op-source triton \
  --dtype bf16 \
  --reference-mode fp32 \
  --tokens 256 \
  --hidden-size 512 \
  --vocab-size 8192
```

### 5. SM90 wrapper delegation

Use this only after rebuilding with `KERNEL_ALIGN_FORCE_SM90=1`. The direct SM90
op still delegates to the shared TP path once TP kwargs are present.

```bash
torchrun --standalone --nproc_per_node=4 scripts/test_linear_logp_tp.py \
  --op-source sm90 \
  --dtype bf16 \
  --reference-mode fp32 \
  --tokens 256 \
  --hidden-size 512 \
  --vocab-size 8192
```

### 6. Larger stress

This checks the end-to-end distributed path at a more realistic shape without
building full reference logits.

```bash
torchrun --standalone --nproc_per_node=4 scripts/test_linear_logp_tp.py \
  --op-source registry \
  --dtype bf16 \
  --reference-mode fp32 \
  --tokens 256 \
  --hidden-size 512 \
  --vocab-size 8192 \
  --run-stress \
  --stress-tokens 4096 \
  --stress-hidden-size 2048 \
  --stress-vocab-size 32768
```

Expected: `finite=PASS`. Record `max_rank_elapsed_ms` and
`max_rank_peak_memory_gb`.

## Reading Output

Successful output looks like:

```text
[correctness]
  dtype=torch.bfloat16, reference_mode=fp32, atol=0.08, rtol=0.08
  tokens=256, hidden=512, vocab=8192
  shard_boundaries=[0, 2048, 4096, 6144, 8192]
  PASS output: max_abs=...
  PASS hidden_grad: max_abs=...
  PASS weight_grad: max_abs=...
  PASS bias_grad: max_abs=...

[result]
  PASS
```

If `--uneven-shards` is set, the shard boundaries will not be equal-sized. That
is intentional and validates `vocab_start_index` handling.

## Important Flags

| Flag | Meaning |
| --- | --- |
| `--op-source registry` | Use `kernel_registry.get_op("linear_logp")`; recommended default. |
| `--op-source native` | Directly test the shared PyTorch TP path. |
| `--op-source triton` | Test Triton wrapper TP delegation. |
| `--op-source sm90` | Test SM90 wrapper TP delegation; requires SM90 extension. |
| `--dtype bf16` | Hopper target dtype. |
| `--reference-mode fp32` | Full reference upcasts hidden/weight/bias to fp32; best for fused/Triton bf16 correctness. |
| `--reference-mode matching` | Full reference uses same-dtype `F.linear`; useful for measuring PyTorch full-GEMM vs shard-GEMM drift. |
| `--uneven-shards` | Builds non-equal vocab shards to test range handling. |
| `--run-stress` | Adds a large TP-only finite/peak-memory smoke. |
| `--no-bias` | Tests the no-bias path. |

## Troubleshooting

### NCCL timeout or hang

Check that every process reaches the same code path and that `nproc_per_node`
matches the number of visible GPUs:

```bash
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
```

Re-run with:

```bash
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

### `target_ids must be covered by exactly one TP vocab shard`

The rank-local `vocab_start_index` or shard sizes are inconsistent. The script
prints `shard_boundaries`; verify they form a contiguous `[0, vocab_size)`
partition.

### bf16 fp32-reference passes but matching-reference drifts

This indicates numerical drift between PyTorch's full bf16 GEMM and the TP
vocab-sharded GEMMs. Record the reported `max_abs` and `max_rel`; use the
fp32-reference result as the main TP correctness signal.

### CUDA OOM in correctness phase

The correctness phase materializes full `[tokens, vocab]` logits on every rank.
Reduce `--tokens`, `--hidden-size`, or `--vocab-size`. Use `--run-stress` for
larger shapes after small correctness has passed.

### `fused_linear_logp_sm90 is not compiled`

This only affects `--op-source sm90`. Rebuild with:

```bash
KERNEL_ALIGN_FORCE_SM90=1 python -m pip install -e .
```

The default `--op-source registry` should still work by falling back to Triton or
native.

## What To Attach To The PR

For the PR update, paste:

- command lines for each run;
- the `[env]` block;
- all `[correctness]` metric lines;
- the `[stress]` finite, elapsed time, and peak memory lines;
- GPU model and CUDA/PyTorch versions;
- any matching-reference drift numbers if that optional run fails or requires loose tolerances.
