# P6 T05 fused MoE combine

This implementation targets the proposed profile from PR #448 at
`05bc53aa0396ffaf2c9241559b388159563e6b36`. NVIDIA sm80+ is supported;
the A100 can execute local WS1 tests. Hardware certification requires actual
results from the named device. Foundation/live integration remains unverified.

The target is the DSV4 routed/shared-expert forward boundary in RL training and
rollout. Identical logical weighted rows must produce identical token outputs,
regardless of packed return order or batch layout. This is forward only; T06 owns
backward. The proposed profile does not assert engine-level train/rollout parity.

## Backends, dispatch and accuracy

| Backend | Entry point | Status |
| --- | --- | --- |
| NVIDIA sm80+ | `rl_engine.p6.combine.prepare_combine` | Triton CUDA fused forward |
| CPU reference | `rl_engine.p6.reference` T02/T03/T04 | Independent scalar oracle, not a fallback |
| ROCm / Ascend | None | Explicitly unsupported in this implementation |

The direct binding never falls back. For conformance, the thin
`TritonCombineProvider` registers through T01's existing `ProviderRegistry`:

```python
from rl_engine.p6.combine_provider import TritonCombineProvider
from rl_engine.p6.provider import ProviderRegistry

registry = ProviderRegistry()
registry.register("live", TritonCombineProvider("cuda:0"))
# registry.run("live", "fused_moe_combine_fwd", frozen_case_id, frozen_inputs)
```

Here `live` means actual device readback, not production approval. The adapter
accepts only the nine input-bound frozen cases and compares every actual stage
before issuing `REFERENCE_BYTES_PASS`. It is not a fast path. Arbitrary fast-path
inputs use `prepare_combine`. Global `KernelRegistry`, production autograd and
Foundation ABI registration await T01/owner integration; this PR does not claim
those pending interfaces are settled or implement T07/T08/T09.

Acceptance uses exact raw-byte equality against the proposed P6 policy and
independent T02→T03→T04 serial reference, not a private tolerance or WS1 Qwen3
logprob threshold. Signed zero is compared as bytes. TF32 and route weighting are
not part of this elementwise gather/add operation.

| Tensor | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `rows` | `[P,H]` | BF16 | Already weighted; contiguous; same NVIDIA device |
| `shared`, `residual` | `[T,H]` | BF16 | Contiguous; added once in that order |
| `output` | `[T,H]` | BF16 | One final RNE downcast |
| lookup (private) | `[T,6]` | int64 | Validated opaque-token inverse mapping |

Inputs may share storage with each other, but must not overlap output, debug or
status buffers. Lazy negative/conjugate views, unsupported shapes, grad requests,
invalid metadata and unsupported numerical values fail closed. Offsets must fit
signed 32-bit indexing, H must be positive and grid.y must not exceed 65535 tiles.

## API and arithmetic

```python
from rl_engine.p6.combine import prepare_combine

prepared = prepare_combine(plan, context, rows.device)
result = prepared.forward(rows, shared, residual, debug=False)
output_bf16 = result.output
saved = result.saved_forward
```

The plan is validated on the host once; the uploaded lookup is `[T,6]` int64.
Rows are contiguous BF16 `[P,H]`, already weighted upstream. Shared/residual
are contiguous BF16 `[T,H]` on the same NVIDIA device. Slots are gathered by
canonical identity, first-valid copied, later-valid added in ascending slot
order with `add.rn.f32`, then shared and residual added in FP32. The output is
rounded once to BF16 RNE. Invalid rows are not loaded; all-invalid starts at
positive zero. The default compute path launches one kernel and allocates no
canonical `[T,6,H]` or slot-partial tensors. Prepared launch also writes one
numeric status per tile.

`result = prepared.forward(..., debug=True)` adds stores for
`canonical_fp32`, six `slot_partials_fp32`, `routed_fp32`, `after_shared_fp32`,
`precast_fp32` and `output_bf16`. `result.stage_bytes()` reads actual tensor
bytes; `result.trace()` hashes them under #448's existing stage names. These
diagnostics synchronize and belong outside timed/captured production execution.
The public result binding and stage keys remain provisional with #448.

The checked API rejects nonfinite active inputs/results, subnormal arithmetic,
FP32/BF16 overflow, unsupported layout/dtype/device and autograd requests.
Padding may contain nonfinite values because invalid rows are never loaded.
T05 supplies forward only; backward fan-in is T06. Route weighting, residual
ownership, communication and live-provider identity are upstream obligations.

## CUDA Graph

Allocate and compile before capture:

```python
buffers = prepared.allocate(debug=False)
prepared.launch(rows, shared, residual, buffers)  # warm-up on the capture stream
buffers.check_status()
# Capture prepared.launch(...) with torch.cuda.graph using normal side-stream warm-up.
# After each replay, outside capture and on the synchronized consumer stream:
graph.replay()
buffers.check_status()
output_bf16 = buffers.output
```

`launch` does not perform host readback; its output is valid only after
`check_status` succeeds. Every execution overwrites the entire status buffer,
so a later good replay recovers from an earlier invalid input. Checked eager
forward, preparation, allocation and readback cannot run inside capture.
Keep tensors/plan/lookup alive, preserve input addresses for replay, and obey
CUDA stream dependencies before reading/reusing buffers. Distinct concurrent
launches need distinct buffers. The opt-in graph test contains a full example.

## Remote A100: check availability before testing

Log in to the GPU host, then run these **before any GPU test**:
The query/monitor commands are documented in the
[NVIDIA nvidia-smi manual](https://docs.nvidia.com/deploy/nvidia-smi/).

```bash
nvidia-smi
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
nvidia-smi pmon -c 3
nvidia-smi dmon -c 5
```

Match the selected GPU's UUID against the process list. An available GPU
normally has low utilization over multiple samples, ample free memory and
no active compute process on that GPU. A single 0% sample does not prove
that it is unused. MIG instances and shared/scheduled nodes need the site's
allocation rules; use your assigned device. Do not stop another user's process.
If utilization or occupied memory persists, inspect the PID with
`ps -fp PID` and use another device or wait for the allocation.

For physical GPU 2, select it in the shell (replace `2` with your assigned
index). Inside Python it becomes logical `cuda:0`:

```bash
export CUDA_VISIBLE_DEVICES=2
```

## Get the code and check the Python environment

The branch is `codex/p6-t05-fused-moe-combine`. Fetch it from your fork after
you publish it. Until then, transfer the dedicated worktree/patch; checking
out #448 alone does not include T05. Run commands at the T05 checkout root.

Use a dedicated environment, Python >=3.10, NVIDIA PyTorch >=2.4.1,
Triton >=3.2 and <4, NumPy and pytest. If PyTorch/Triton already work on the
server and meet these version floors, retain the matched pair. Otherwise create a virtual environment
and install a NVIDIA PyTorch wheel matching the server's driver using
the [official installation selector](https://pytorch.org/get-started/locally/).
NVIDIA PyTorch supplies its matched Triton dependency. Native RL-Kernel CUDA
extensions, Transformers and model weights are not needed for these tests.

```bash
# Only if these dependencies are missing; keep a working torch/Triton pair.
uv pip install --python "$(which python)" pytest numpy ruff
python - <<'PY'
import torch, triton
print('torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
print('triton:', triton.__version__, 'available:', torch.cuda.is_available())
assert torch.cuda.is_available() and torch.version.hip is None
print('GPU:', torch.cuda.get_device_name(0), 'capability:', torch.cuda.get_device_capability(0))
assert torch.cuda.get_device_capability(0)[0] >= 8
PY
```

## Correctness, then performance

Start with CPU contract tests:

```bash
OMP_NUM_THREADS=1 python -m pytest tests/p6 -q
```

Then explicitly enable the real T05 kernel tests. An unavailable GPU or
missing Triton fails rather than silently skipping when this flag is set:

```bash
mkdir -p p6-t05-results
# Pick a fresh absolute directory OUTSIDE the checkout for immutable raw-byte evidence.
# Never reuse or delete an existing sealed attempt to make a rerun pass.
P6_T05_ARTIFACT_DIR=/tmp/p6-t05-a100-attempt-001 \
P6_RUN_T05_GPU=1 OMP_NUM_THREADS=1 python -m pytest tests/p6/test_fused_combine.py -q -x \
  --junitxml=p6-t05-results/a100.xml
python -m rl_engine.p6 verify /tmp/p6-t05-a100-attempt-001
git rev-parse HEAD
git status --short
```

The 12 offline-compile parametrizations are optional and skip unless
`P6_COMPILE_T05=1`; the 48 GPU cases and 43 CPU cases must pass in the run above
(91 passed, 12 skipped). This verifies
all nine frozen goldens and debug stages, signed zero/ties/cancellation,
block sizes 128/256/512, layout/dtype/shape/autograd rejection, unused poison
padding, physical permutation, batch partitioning through H=4096, FP32 and
BF16 overflow, subnormal intermediate cancellation, sparse signed-zero
initialization and changed-input graph replay/status recovery. Added acceptance
checks cover explicit T02→T03→T04 byte equality, debug-on/off CPU/CUDA RNG and
plan/order identity, candidate-versus-recorded registry envelopes, sealed actual
readback, default buffer allocation and zero additional PyTorch launch allocation,
output/debug storage alias rejection, and P4 mock EP1/2/4/8 with zero-count peers,
placement changes, chunk sizes, reversed/delayed completion and two-stream overlap.
These are local recorded schedules, not real EP transport or NCCL certification.

The sealed attempt reuses T01's artifact schema. Its top-level provenance denotes
the CPU reference recordings; `checks[].candidate` holds nine actual CUDA byte
envelopes with device/build hashes. The GPU test compares them before publication
and again after sealed readback. `verify` checks seal integrity/CPU replay and does
not rerun the GPU or certify production. Keep the whole attempt directory.

An additional no-device compile check covers sm80/sm90, debug on/off and
block sizes 128/256/512 with H=4097:

```bash
P6_COMPILE_T05=1 python -m pytest tests/p6/test_fused_combine.py -q -k offline_cuda_compile
```

Run memory checking if the server has Compute Sanitizer:

```bash
P6_RUN_T05_GPU=1 compute-sanitizer --tool memcheck --error-exitcode 1 \
  python -m pytest tests/p6/test_fused_combine.py -q -x
```

Only after correctness passes, benchmark:

```bash
python -m benchmarks.p6_combine --tokens 32 --hidden-size 4096 --repeats 100 \
  > p6-t05-results/a100-t32-h4096.json
python -m benchmarks.p6_combine --tokens 256 --hidden-size 4096 --repeats 100 \
  > p6-t05-results/a100-t256-h4096.json
```

The benchmark refuses byte mismatches before timing, compares an ordered
eager PyTorch reference against prepared fusion and checked eager fusion,
and reports device/build/input/output identities, GPU-event latency and
wall latency. Checked eager includes allocation and numeric status readback;
prepared launch excludes those costs and needs the explicit post-execution
check. Process peak allocation includes fixtures/debug/reference buffers;
it is not per-kernel VRAM. These local measurements do not certify EP/NCCL,
live Foundation, backward or H100 behavior. There is no approved minimum speedup
in this proposed contract; report measurements without inventing one. Timing on
a GPU running Ray/another workload is not valid performance evidence. Return XML,
the sealed directory, environment/git logs, two JSON files from an idle allocated
GPU and any sanitizer output. See the [merge checklist](../design/p6-t05-merge-checklist.md)
for local CI commands and remaining owner/hardware gates.
