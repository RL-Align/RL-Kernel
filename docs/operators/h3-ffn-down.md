# H3 deterministic FFN down projection

CUDA WS1 candidate for `zzf2024`'s `h3_ffn_down_gemm` work item in
[issue #420](https://github.com/RL-Align/RL-Kernel/issues/420).
Base: `test-h3@32b765ec992cd1206517104ec66506881203c91c`.
Checkpoint: `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`.

## Contract and scope

Backend `rlkernel.h3_ffn_down.triton.fp32_tiles.v1` reuses the existing
`triton.matmul.det_gemm._det_gemm_fp32_kernel`. BF16 operands feed FP32
32-wide dot products and ascending FP32 tile additions, followed by one BF16
cast. The schedule is fixed at 64x64x32, four warps and two stages, with
`enable_fp_fusion=False`, TF32 disabled, no tuning, no split-K and no fallback.
No new CUDA or Triton device kernel is added.

The eager operator accepts BF16 `[M,14336]` or `[B,S,14336]`, flattened row count
1..32768, native `[5376,14336]` weights and no bias on one SM90 CUDA device.
It computes `Y=X@W.T`, `dX=dY@W`, and canonical `dW=dY.T@X`. Strided tensors
are normalized by copies. Weight-gradient rows are sorted by logical identity
and padded to a fixed 32-row extent before a single reduction.

The operator rejects invalid shapes, types, devices, TF32, duplicate active
keys, inconsistent weight identity and nonzero padding gradients. Unsupported
hardware or backend fails closed. Graph compilation, double backward and
retained-graph replay are unsupported. The registry descriptor remains
**test-only**; strict production resolution requires further qualification.
TP, whole-model integration, ROCm and cross-platform equality are outside this PR.

Inference requires no backward session:

```python
import torch
from rl_engine.kernels.ops.h3_ffn_down import H3FFNDownGemmOp

torch.backends.cuda.matmul.allow_tf32 = False
op = H3FFNDownGemmOp()
with torch.no_grad():
    y = op(swiglu_output, down_weight)
print(op.last_execution)
```

Training requires one session spanning **all forward uses and one backward call**:

```python
from rl_engine.kernels.ops.canonical_backward import canonical_backward_session

with canonical_backward_session() as session:
    outputs = [op(x, down_weight, logical_keys=keys,
                  parameter_id="transformer.blocks.0.ffn.down.weight")
               for x, keys in zip(parts, key_parts, strict=True)]
    torch.cat(outputs).backward(upstream_gradient)
    session.validate_complete()
```

Keys are same-device int64 `[rows,2]` or `[rows,3]`, such as
`[sample_id,token_id]`, with stable unique identities across packing, permutation
and microbatching. A negative first component marks inactive padding. The same
weight Tensor must retain one parameter ID. Do not sum independently rounded
microbatch weight gradients; call `validate_complete()` to detect missing branches.
Execution traces record requested/actual backend, kernel, arithmetic, schedule,
topology, checkpoint, executed gradients and actual JIT/cubin hashes.

## H100 results

Measured on 2026-10-10: dedicated non-MIG H100 80GB HBM3, capability 9.0,
81,559 MiB, driver 570.211.01, Ubuntu 22.04, Python 3.10.12, torch 2.6.0+cu124,
CUDA 12.4 and primary Triton 3.2.0.

- **64 GPU/profile tests passed** before PR formatting.
- **49 cases / 1,141 comparisons passed**. Random, synthetic SwiGLU-like and
  cancellation inputs cover M=1,31,32,33,127,128,129,256,513,1024,2048,4096,
  8192,16385,24577,32768; a real-weight fixture adds M=129.
- Forward/dX/dW match an independent FP32 PyTorch autograd reference with the
  shared `ws1-c1-v2` tolerances. Maximum error/tolerance is **0.18233**, below 1.
  No tolerance was weakened.
- Raw-byte checks cover repeats, logical-row permutation, 31-row chunking with
  one backward, padding, non-contiguous layouts, batch reshaping, singleton rows
  and mutation of unrelated rows. NaN/Inf and signed-zero differences fail.
- A separate process and fresh Triton 3.2.0 cache passed 10 cases with **30 matching
  y/dX/dW hashes**. A separately pinned Triton 3.4.0 cache passed five selected
  cases with **15 matching hashes**. Both profiles used the same GPU and PyTorch.

The checkpoint fixture uses `transformer_blocks.0.ff.net.2.weight`, BF16
`[5376,14336]`, fetched through validated HTTP ranges without downloading the
whole model. Weight SHA256 is
`e19d4e1649e4b55b1931842e5cc4bb46892c2e635b9315b2996630a86284e88f`.
Activations and upstream gradients are **synthetic**, not captured model data.

The explicit legacy backend `rlkernel.h3_ffn_down.sm90.midtree.v1` failed real-weight
forward accuracy at M=129: 23,353 failing elements, maximum error/tolerance
5.00441. Its failure is preserved; the BF16 midpoint tree is never selected
as an automatic fallback.

Five-sample synchronized wall medians for primary-profile random inputs:

| M | Candidate forward ms | torch BF16 forward ms | Candidate forward+backward ms | torch BF16 forward+backward ms |
|---:|---:|---:|---:|---:|
| 1 | 0.4641 | 0.0822 | 5.8811 | 1.2091 |
| 129 | 0.5118 | 0.1100 | 4.3671 | 1.2572 |
| 8192 | 7.1128 | 1.7194 | 18.4472 | 5.2220 |
| 32768 | 41.6259 | 7.1671 | 107.5935 | 21.5373 |

These include wrapper checks, copies, allocations and canonical preparation,
not isolated kernel latency. This candidate is slower than ordinary torch BF16.
The comparator has not been qualified for byte invariance.

Vendored vLLM commit `7797b6022c129b862e45ae6aed08822e65d1bccb` requires the separately
pinned Triton 3.4.0 profile because 3.2.0 rejects `tl.range(..., flatten=True)`.
Its partial probe passed y/dX/dW accuracy and y/dX chunk/singleton invariance.
Within 3.4.0 at random M=32768, vLLM forward took 9.6604 ms versus this adapter's
41.5963 ms; its dW primitive took 40.0210 ms versus 20.9800 ms for fixed FP32.
The vLLM primitive excludes this adapter's checks/copies and has no qualified
canonical autograd session or full dW invariance matrix. It remains prior art.

## Core JSON evidence

Only the following measured JSON files accompany this document:

- [Summary, fixture, tests and rejected native candidate](evidence/h3-down-h100-20261010/h100-summary.json).
- Full matrix: [random](evidence/h3-down-h100-20261010/h100-random.json),
  [SwiGLU-like](evidence/h3-down-h100-20261010/h100-swiglu-like.json), and
  [cancellation](evidence/h3-down-h100-20261010/h100-cancellation.json).
- [Cold-cache/compiler replay and vLLM comparison](evidence/h3-down-h100-20261010/h100-replays.json).

The split matrix is lossless: combine the three `cases` arrays with the summary's
`fixture_cases`, sort by their `case_indexes` / `fixture_case_indexes`, and attach
`full_report_header`. That reproduces the original measured report exactly.
Every file is below the repository's 500 KiB pre-commit limit. JSON retains all
checks, tolerances, timing samples, peak memory, output hashes, compiled-kernel
identities and source fingerprints. Original logs, PNG/CSV, source snapshots,
archives and rental details remain local and are excluded from the PR.

GPU reports identify the measured source. Subsequent Black/isort formatting
is recorded separately with original/current hashes, unchanged non-import AST
and unchanged import bindings; no GPU rerun after formatting is claimed.

## Reproduction and remaining qualification

Use a dedicated non-MIG H100 80GB and a CUDA 12.4 development environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r scripts/requirements-h3-down.txt
python scripts/prepare_h3_down_fixture.py --rows 129 --output /workspace/h3-down-fixture.pt
H3_FIXTURE=/workspace/h3-down-fixture.pt bash scripts/run_h3_down_h100.sh smoke
python scripts/validate_h3_ffn_down.py --seed 420 --repeats 5 \
  --rows 1,31,32,33,127,128,129,256,513,1024,2048,4096,8192,16385,24577,32768 \
  --families random,swiglu_like,cancellation \
  --fixture /workspace/h3-down-fixture.pt --output /tmp/h3-down-full.json
```

The default needs Triton, not the native extension. `H3_BUILD_NATIVE=1` optionally
builds the legacy comparison with fast math disabled; that build requires matching
Python development headers and nvcc 12.4. `--compare-vllm` is an explicit partial
probe; use a separate Triton 3.4.0 environment without replacing the primary profile.

This change is ready for **operator review**, with `acceptance_complete=false`.
Captured H3 activations, review of arithmetic/covered layouts, formal runtime
integration and a performance decision remain before production promotion.
It does not close the full H3 roadmap or assert portable equality across devices.
