# Standard Qwen-Image timestep embedding MLP

`timestep_embed_mlp` implements the standard 256-channel cosine-first sinusoidal
embedding, Linear(256, 3072), SiLU, Linear(3072, 3072), and first-order backward.
It does not add guidance or conditioning. Smaller hidden dimensions are accepted
for testing. Weights follow `torch.nn.Linear` layout `[out_features, in_features]`.

```python
from rl_engine.ops.embedding.timestep_embed_mlp import TimestepEmbedMLPOp

op = TimestepEmbedMLPOp("cuda")  # or "triton"; "pytorch" is the slow independent gold
output, trace = op(timestep, weight1, bias1, weight2, bias2, return_trace=True)
output.backward(upstream)
print(trace.actual_backend, trace.kernels)
```

Parameters use FP32 or BF16 storage, all on the timestep device. Timesteps are
one-dimensional FP32/BF16 or int32/int64 values at the embedding-module boundary.
Integer timesteps support forward and parameter gradients, not timestep gradients.
All intermediate values and accumulations use FP32; output and parameter gradients
return to parameter dtype. Timestep gradients return to timestep dtype. Autocast
and higher-order differentiation are unsupported. This fused precision policy
avoids intermediate BF16 quantization; it is not a bitwise emulation of the
unfused Diffusers BF16 module.

For `i=0..127`, frequencies are `exp((-log(10000)*i)/128)` and phase is
`(timestep*frequency)*1000`, in that order. Output embedding concatenates cosine
then sine. Fixed FP32 frequencies are generated normally and cached per operator
and device. There are no CPU library hashes, MKL dispatch requirements or
frequency-fingerprint gates.

## Accuracy and deterministic layouts

The operator is registered in `rl_engine/validation/operators/operator_specs.py` with `op_class="reduction"`
and all five differentiable inputs. The unchanged shared contract resolves forward
and gradient accuracy independently. The independent gold uses pure PyTorch FP32
compensated product sums and explicit linear VJPs, plus torch sigmoid autograd.
It does not invoke candidate primitives. A separate conventional `F.linear` path
is used for semantic regression and performance comparison.

CUDA long dot products use compensated FP32 lane accumulation and a fixed warp
tree; Triton uses a fixed compensated FP32 tree. Short reductions use ordered FMA.
Neither backend uses atomic accumulation, TF32, tensor-core reduced precision, or
batch-dependent autotuning. Backend-to-backend bit equality is not promised;
configuration invariance is checked separately, byte for byte.

`chunk_size` changes forward row partitioning. Backward always reduces the entire
logical active sample set in a fixed order. For permuted/padded physical input:

```python
output = op(t, w1, b1, w2, b2,
            sample_ids=logical_ids, active_mask=mask, chunk_size=2)
```

`sample_ids` is a device-local int64 `[B]` with unique active IDs. `active_mask` is
a device-local bool `[B]`. Active rows are evaluated in ascending ID order and
outputs are restored to physical positions; padding output and timestep gradients
are zero. This makes parameter gradients invariant for the **same full logical
sample set**. Ordinary external microbatch `.backward()` calls that add already
rounded BF16 parameter gradients are not covered. Singleton invariance covers
outputs and row-local timestep gradients only, not parameter gradients for a
different sample set.

## Backends, build and trace

The CUDA backend first imports the independent `rl_engine._timestep_cuda` extension
built by `setup.py`, with strict FP32 flags independent of other operators' fast
math options. In a source checkout it can JIT-build the same CUDA source when the
prebuilt module is absent. JIT needs a compatible CUDA toolkit, C++ compiler,
Python headers and Ninja. Set `TORCH_EXTENSIONS_DIR` and `TRITON_CACHE_DIR` to
isolated writable directories. The Triton backend needs a compatible Triton install.

Unavailable backends raise by default. `allow_fallback=True` explicitly enables
the PyTorch reference; the returned trace records requested/actual backend and the
reason. Kernel execution errors are not silently converted into fallback results.
Trace objects belong to individual calls and are updated by their own backward.
Launch records are diagnostic labels; the validation script separately requires
observed CUDA profiler events from embedding, matrix, activation and timestep VJP
kernels. The repository's unrelated `_C unavailable` import warning does not
identify this independently built extension's execution path.

## Reproduce

```bash
OMP_NUM_THREADS=4 python -m pytest tests/ops/embedding/test_timestep_official.py \
    tests/contracts/test_tolerance_contract.py tests/validation/operators/test_operator_inputs.py -q

python tools/validation/operators/check_operator.py --op timestep_embed_mlp --candidate cuda \
    --device cuda --dtype bf16 --batch 3 --seq 1 --seed 386 --check-grad --json
# Repeat with --candidate triton and --dtype fp32.

python tools/validation/operators/validate_timestep_official.py --trace --benchmark \
    --output /absolute/isolated/output/matrix.json
```

The default matrix uses production H=3072, batches 1/3/16, seeds 386/9386,
FP32/BF16 and both GPU backends. Accuracy gold runs on CPU with the same input and
upstream values. The JSON records source hashes, resolved tolerances, per-output
errors, full invariance verdicts, observed kernels, environment, and timing samples.
CUDA-event timings include operator dispatch and allocation overhead after warmup;
compilation is excluded. The performance baseline is ordinary batched PyTorch with
the same fused FP32 compute policy, not the deliberately slow compensated oracle.

`TIMESTEP_RUN_ROOT=/absolute/isolated/root bash ci/scripts/run_timestep_official.sh`
provides an A100-only fresh-cache reproduction including pytest, four official CLI
runs, the matrix, package inventory and device information. Activate the desired
venv and set CUDA_HOME/CPATH for your environment first. It does not connect to a
server, install dependencies or alter system configuration.

## Implementation and historical validation

| Component | Canonical location |
| --- | --- |
| Semantic API and autograd | `rl_engine/ops/embedding/timestep_embed_mlp.py` |
| Independent PyTorch reference | `rl_engine/reference/embedding/timestep_embed_mlp.py` |
| Native CUDA source and loader | `csrc/cuda/embedding/timestep_embed_mlp.cu`, `rl_engine/backends/cuda/embedding/timestep_embed_mlp.py` |
| Triton kernels and adapter | `rl_engine/backends/cuda/embedding/timestep_embed_mlp_triton.py` |
| API and reference tests | `tests/ops/embedding/test_timestep_official.py`, `tests/reference/test_timestep_reference_audit.py` |
| Acceptance and reference audit | `tools/validation/operators/validate_timestep_official.py`, `tools/validation/operators/audit_timestep_reference.py` |

The Triton implementation uses NVIDIA libdevice and is registered only for NVIDIA
CUDA devices. Legacy Python modules remain aliases of these canonical modules,
sharing class identities and the CUDA extension cache. Recorded operator identities
remain stable across the move. New code and reproduction commands use canonical paths.

These are **historical executions**, not new tests of the directory migration:

| Check | A100 / sm80, 2026-10-02 | H100 / sm90, 2026-10-07 |
| --- | --- | --- |
| H3072, B1/3/16, seeds 386/9386, CUDA/Triton, FP32/BF16 | 24/24 passed | 24/24 passed |
| H17, B33/64 long reductions | 8/8 passed | 8/8 passed |
| Official forward/backward CLI checks | 4/4 passed | 4/4 passed |
| Original related tests | 60 passed | 60 passed |
| Reference VJP finite differences / expanded suite | Not run | 63 passed, no skips |
| Related gtest framework regressions | Not claimed | 155 passed, no skips |
| Independent FP32-phase reference audit | Not run | 8/8 passed |
| Profiler confirmation without fallback | Four traces | Four traces |

The H100 acceptance source is `ea55fa686706892ce94ebe7020829c19c108496c`;
source manifests, dependencies, commands, seeds, thresholds, raw measurements,
traces and failed attempts remain preserved outside the current tree. The native
CUDA source and reference are unchanged by the migration; backend arithmetic and
launch settings are unchanged, with only imports and source locations adjusted.
GPU acceptance was not rerun for the migration; local import/build-path checks
do not establish a fresh device execution.

The independent audit compares all six tensors against FP64 trig/MLP/autograd
at the specified **FP32 phase boundary**. It passed all eight H3072 cases.
A separate all-FP64-phase diagnostic exceeded tolerance in 3/4 FP32 cases;
it is a different contract, not an acceptance substitute. In the historical
B16/seed9386 case, timestep-gradient error/tolerance was 5.311 for the old
reference, 0.642 for the current reference, and 80.892 for the all-FP64 diagnostic.

H100 CUDA forward+backward microbenchmarks measured approximately 1.51–1.85x
speedup over ordinary batched PyTorch using 20 CUDA-event samples and medians.
B16 forward and most Triton cases were slower. No full-model speedup or
Hopper-specific WGMMA/TMA optimization is claimed. Full-model training,
multi-GPU execution and full-repository native builds remain unverified.
The target branch's GitHub workflows did not execute these GPU acceptance runs.
An older local double-free remains undiagnosed; the isolated H100 suites passed.

## Historical evidence downloads

Complete A100/H100 results are published separately from the merge diff. These are historical executions, not a new GPU run. The A100 derivative replaces one device UUID with `<GPU_UUID>`; measurements and original artifact hashes are unchanged. H100 is byte-identical to the preserved bundle.

- [Provenance and limitations](https://gist.githubusercontent.com/zzf2024/013d1510aa34a3a29d27574bf9159bfa/raw/d15136a4a9eefc586c33fea4a1a1b396a0675bf4/README.md)
- [Checksum manifest](https://gist.githubusercontent.com/zzf2024/013d1510aa34a3a29d27574bf9159bfa/raw/d15136a4a9eefc586c33fea4a1a1b396a0675bf4/SHA256SUMS)

| Bundle | SHA256 |
| --- | --- |
| [a100-results.json](https://gist.githubusercontent.com/zzf2024/013d1510aa34a3a29d27574bf9159bfa/raw/d15136a4a9eefc586c33fea4a1a1b396a0675bf4/a100-results.json) | `6871f6908d1f51df5d668f14f70a341b147d0ab154f8a562b832e2cdc5a34493` |
| [h100-results.json](https://gist.githubusercontent.com/zzf2024/013d1510aa34a3a29d27574bf9159bfa/raw/d15136a4a9eefc586c33fea4a1a1b396a0675bf4/h100-results.json) | `21184e8ed4acb6b7edd4786236603ee857d3534bf46b3b055a3e8da13ed21762` |

Additional reproduction commands, with isolated output paths:

```bash
TIMESTEP_RUN_ROOT=/absolute/new-h100-run bash ci/scripts/run_timestep_sm90.sh
python tools/validation/operators/validate_timestep_official.py --hidden 17 \
    --batches 33 64 --seeds 386 --output /absolute/new-long-reduction.json
OMP_NUM_THREADS=4 TIMESTEP_CUDA_JIT_ONLY=1 TORCH_CUDA_ARCH_LIST=9.0 \
    python tools/validation/operators/audit_timestep_reference.py \
    --output /absolute/new-audit.json
python -m pytest tests/reference/test_timestep_reference_audit.py -q
```
