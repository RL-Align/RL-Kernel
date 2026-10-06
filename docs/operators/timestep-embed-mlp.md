# Standard Qwen-Image timestep embedding MLP

`timestep_embed_mlp` implements the standard 256-channel cosine-first sinusoidal
embedding, Linear(256, 3072), SiLU, Linear(3072, 3072), and first-order backward.
It does not add guidance or conditioning. Smaller hidden dimensions are accepted
for testing. Weights follow `torch.nn.Linear` layout `[out_features, in_features]`.

```python
from rl_engine.kernels.ops.timestep_embed_mlp import TimestepEmbedMLPOp

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

The operator is registered in `gtest/operator_specs.py` with `op_class="reduction"`
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
OMP_NUM_THREADS=4 python -m pytest tests/test_timestep_official.py \
    tests/test_tolerance_contract.py tests/test_operator_inputs.py -q

python scripts/check_operator.py --op timestep_embed_mlp --candidate cuda \
    --device cuda --dtype bf16 --batch 3 --seq 1 --seed 386 --check-grad --json
# Repeat with --candidate triton and --dtype fp32.

python scripts/validate_timestep_official.py --trace --benchmark \
    --output /absolute/isolated/output/matrix.json
```

The default matrix uses production H=3072, batches 1/3/16, seeds 386/9386,
FP32/BF16 and both GPU backends. Accuracy gold runs on CPU with the same input and
upstream values. The JSON records source hashes, resolved tolerances, per-output
errors, full invariance verdicts, observed kernels, environment, and timing samples.
CUDA-event timings include operator dispatch and allocation overhead after warmup;
compilation is excluded. The performance baseline is ordinary batched PyTorch with
the same fused FP32 compute policy, not the deliberately slow compensated oracle.

`TIMESTEP_RUN_ROOT=/absolute/isolated/root bash scripts/run_timestep_official.sh`
provides an A100-only fresh-cache reproduction including pytest, four official CLI
runs, the matrix, package inventory and device information. Activate the desired
venv and set CUDA_HOME/CPATH for your environment first. It does not connect to a
server, install dependencies or alter system configuration.

Detailed Chinese stage reports, including failed attempts and limitations, are in
[timestep-reports/README.md](timestep-reports/README.md). A100 is the delivery target;
SM90/H100 performance, full-repository native extension CI and full-model training
are not established by these operator-level results.
