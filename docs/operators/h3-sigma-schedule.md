# H3 Sigma Schedule

The MiniMax-H3 rectified-flow sigma grid (RFC #420, `h3_sigma_schedule`):
`linspace(1, 0, N)` in fp32, the exponential shift
`shift·s / (1 + (shift−1)·s)`, `unique_consecutive` deduplication, and
`timesteps = 1 − sigmas[:-1]`. Video uses `shift=12`, audio `shift=3`; the two
grids are independent per-request inputs, never a shared step index. The
default backend on every platform is the PyTorch reference (CPU construction,
outputs copied to the requested device); a device-side NVIDIA Triton backend
is available as an explicit opt-in.

## Entry Point

```python
from rl_engine.runtime.registry import kernel_registry

op = kernel_registry.get_op("sigma_schedule_h3", device="cuda")   # PyTorch reference
schedule = op(25, shift=12.0, device="cuda")   # device= selects output placement only
schedule.sigmas          # fp32 [M], strictly decreasing, terminal exactly 0.0
schedule.timesteps       # fp32 [M-1] = 1 - sigmas[:-1]
schedule.num_model_evals # M - 1 (deduplicated length drives the count)
```

`get_op` returns the PyTorch reference on every platform, CUDA included: the
grid is built on CPU and both outputs are moved to `device`.

## Opt-in: device-side Triton backend

Two explicit routes select `H3SigmaScheduleCudaOp`; neither is reachable
through a legacy `get_op` candidate list. The semantic path resolves with
provenance (`requested_backend` `rlkernel.h3_sigma_schedule.triton`; the
reference is `.reference`):

```python
from rl_engine.runtime.registry import kernel_registry
from rl_engine.runtime.semantic_registry import OperatorRequirements

session = kernel_registry.semantic.session()
resolution = session.resolve(
    semantic_op="sigma_schedule_h3",
    requested_backend="rlkernel.h3_sigma_schedule.triton",
    target="rollout",
    requirements=OperatorRequirements(device="cuda", dtype="float32"),
)
op = session.instantiate(resolution)
provenance = session.instance_provenance(resolution, op)
```

Or construct the class directly:

```python
from rl_engine.backends.cuda.model_specific.minimax_h3.sigma_schedule import (
    H3SigmaScheduleCudaOp,
)

op = H3SigmaScheduleCudaOp()   # raises ImportError without Triton or on HIP/ROCm
schedule = op(25, shift=12.0, device="cuda")
```

Both routes produce the same bytes as the default reference path (the CUDA
suite asserts this); opting in only moves the arithmetic onto the device.

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| CUDA (Triton) | `rl_engine.backends.cuda.model_specific.minimax_h3.sigma_schedule.H3SigmaScheduleCudaOp` | 3 Triton kernels (grid / dedup scan / timesteps) | **Opt-in** (semantic `requested_backend` or direct construction; not a legacy `get_op` candidate). Validated on RTX 4090D (torch 2.5.1+cu121, Triton 3.1.0): bitwise-identical to the pinned upstream oracle across the full device suite |
| ROCm | — | — | **CUDA only.** The ROCm column stays open. The inline PTX `div.rn.f32` is NVIDIA-only, so the backend refuses construction on HIP; ROCm dispatches the PyTorch reference. |
| PyTorch reference | `rl_engine.reference.minimax_h3.sigma_schedule.NativeH3SigmaScheduleOp` | `torch.linspace` + `unique_consecutive` | **Default on every platform** (byte-frozen oracle; fixture generated on Linux CPU under torch 2.14.1, reproduced on 2.4.1 and 2.5.1). Builds on CPU and copies the outputs to the requested device |

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `num_inference_steps` | scalar | int | `>= 2`; terminal zero included in the count |
| `shift` | scalar | python float | `> 0` (video 12.0 / audio 3.0) |
| `device` | — | — | Output placement. The default reference accepts any device (builds on CPU, copies to `device`); the opt-in Triton backend requires an NVIDIA CUDA device (refuses construction on HIP/ROCm) |
| → `sigmas` | `[M]`, `M ≤ N` | float32 | strictly decreasing; `sigmas[0] == 1.0` and `sigmas[-1] == 0.0` exactly for released shifts |
| → `timesteps` | `[M-1]` | float32 | `1 − sigmas[:-1]`, strictly increasing from 0.0 |
| → `num_model_evals` | scalar | int | `M − 1` |

## Dispatch Behavior

The legacy candidate list for `sigma_schedule_h3` is `[PyTorch reference]` on
every platform — CUDA, ROCm, MUSA, NPU and CPU alike — so `get_op` always
builds the grid on CPU and moves the outputs to the requested device. This is a
deliberate default (review 2026-10-25, PR #566): the 4090D measurements below
show the device-side build is 2.5–3.2x slower at typical H3 sizes, and no
consumer requires on-device construction. The Triton backend is not a legacy
candidate anywhere; it is selected only through the semantic
`requested_backend="rlkernel.h3_sigma_schedule.triton"` or by direct
construction (see Opt-in above). Its construction-time guards are unchanged —
`H3SigmaScheduleCudaOp()` raises `ImportError` when Triton is not importable
or on a HIP/ROCm build, so an opt-in that cannot be honored fails explicitly
(semantic `instantiate` raises `OperatorInstantiationError`) instead of
silently swapping arithmetic (see
[Runtime Dispatch](../architecture/runtime-dispatch.md)).

## Accuracy

The acceptance oracle is the pinned fixture
`tests/reference/minimax_h3/h3_sigma_schedule_upstream_fixture.json`,
generated by executing the original upstream
`MiniMaxH3Scheduler.set_timesteps` (diffusers commit `f53d552`, blob
`ca69ae6a`); six cases N∈{2,4,25} × shift∈{3,12} are compared as fp32 bit
patterns. The device kernels reproduce the pinned CPU construction bit law:
fp32 step, two-sided fp64 accumulation with single rounding, pre-rounded
python-float scalars, correctly-rounded division via libdevice `div_rn` for
the linspace step and inline PTX `div.rn.f32` for the shift transform
(on Triton 3.0.0 / RTX 4090D the
`tl.fdiv(..., ieee_rounding=True)` flag and the `/` operator were measured
bit-identical and NOT correctly rounded, so they are not used),
`enable_fp_fusion=False` on floating-point launches, and IEEE-equality dedup
(NaN rows are kept, matching `torch.unique_consecutive`). NaN and near-zero
parity follows the torch-CPU oracle (validated on RTX 4090D, independent
audit 2026-10-16): a NaN `fl32(shift)` fills the grid with its own quieted
NaN bits — sign included, so `-NaN` yields `0xffc00000` and `+NaN`
`0x7fc00000` (host-side scalar fast path; the device kernel's `0x7fffffff`
canonicalization would otherwise differ); an infinite `fl32(shift)` makes
every element an invalid operation, filled with the x86 default quiet NaN
`0xffc00000`; a subnormal `fl32(shift)` (e.g. `1e-45` → the minimum
subnormal `2^-149`) takes the normal device path with a `+inf` first element
and subnormal quotients (upstream keeps 10 of 25 rows at `1e-45`); a shift
whose `fl32` underflows to `+0.0` (e.g. `1e-46`) makes element 0 the invalid
`+0.0/+0.0`, canonicalized in-kernel to `0xffc00000`. Odd inputs follow
upstream parity: float `N` truncates via `int()`, non-finite or
fp32-overflowing shifts yield NaN grids, extreme finite shifts can collapse
the grid (e.g. `1e30` → `[1.0, 0.0]`).

## Performance Notes

```bash
python benchmarks/models/benchmark_h3_sigma_schedule.py --samples 100
```

Measures **per-call end-to-end host wall latency** on three paths — A:
reference built on CPU, outputs on CPU; B: reference built on CPU, sigmas and
timesteps moved to the GPU via `device="cuda"` (the registry default path on
every platform); C: the opt-in Triton backend building on the device. Each call is
timed individually (GPU paths synchronize before each sample and after each
call); raw samples, mean/median/stdev/min/max are reported per case, and the
median is the headline number. Before any timing on a GPU host the script
verifies path C bitwise against path A for every measured case (exit 2 on
mismatch); Triton compilation happens inside that gate and its wall time is
recorded once as `correctness_gate_wall_ms`, outside all timed samples. The
script runs from an uninstalled checkout. The grid is tiny (≤4 KB for
N=1000): launch and host-sync overhead may well exceed the arithmetic — the
benchmark reports what is measured, it does not assume the device path wins.
N=1000000 is a dedup stress case, not a typical workload.

Measured on RTX 4090D (torch 2.5.1+cu121, Triton 3.1.0, 100 samples/case):
at N=25/50/1000 the device path C is 2.5–3.2x slower than B (B/C median
0.315–0.403, every case separated from dispersion). C wins only at the N=1e6
dedup stress case: 51.98x at shift=12 (separated), while shift=3 is 7.9x and
within dispersion, so it is not claimed. The N=1e6 CPU medians are noisy
(B stdev ≈ 44 ms at shift=12), so the 51.98x ratio passes the dispersion
check only by a small margin. See Known Limitations and the raw JSON, CUDA
test results, source hashes, and reproduction commands in the
[4090D evidence report](https://github.com/RL-Align/RL-Kernel/tree/test-h3/reports/experiments/h3-sigma-schedule).

These numbers are the basis for the default: the reference path (B) is the
legacy `get_op` backend on CUDA because it is 2.5–3.2x faster at every typical
H3 size and produces identical bytes. The trade reverses only at the N=1e6
dedup stress case — at shift=12 the reference median is 48.37 ms against
0.93 ms for Triton (51.98x), so a consumer building grids of that size should
opt into the Triton backend explicitly; it is not a released-model workload.

## Tests

```bash
python -m pytest tests/reference/minimax_h3 tests/runtime/test_h3_sigma_schedule_dispatch.py -v
python -m pytest tests/backends/cuda/test_h3_sigma_schedule_cuda.py -v   # NVIDIA CUDA hardware
```

The device suite also carries the upstream-parity edge matrix (NaN/±inf and
extreme shifts, float-`N` truncation) against the CPU reference, which is
itself bit-identical to the executed upstream method on those inputs.

## Known Limitations

- Validated on an x86_64 host with NVIDIA RTX 4090D only (driver 580, torch 2.5.1+cu121,
  Triton 3.1.0); other GPUs/drivers/Triton versions are not validated. On
  Triton 3.0.0 the `tl.fdiv(..., ieee_rounding=True)` flag was measured NOT
  correctly rounded, which is why the linspace step uses libdevice `div_rn`
  and the shift transform uses inline PTX `div.rn.f32`.
- Invalid-operation NaN bits are pinned to that x86 CPU oracle (`0xffc00000`).
  A live CPU reference on ARM can produce `0x7fc00000` instead for infinite,
  fp32-overflowing, or underflowing shifts. Bitwise parity for those edge cases
  is not portable across host architectures; the released shifts (3 and 12)
  do not take these paths. NVIDIA GPUs on ARM hosts have not been validated.
- **CUDA only.** The ROCm column stays open: the Triton backend refuses
  construction on HIP because its division is emitted as inline NVIDIA PTX;
  ROCm dispatches the PyTorch reference.
- Measured end-to-end latency on the 4090D: for the typical H3 sizes
  (N = 25/50/1000) the device-side Triton build is 2.5–3.2x SLOWER than
  building on CPU and copying to the GPU (B/C median 0.315–0.403, all separated
  from dispersion) — which is why it is opt-in rather than the default. The
  device path wins only at the N = 1e6 dedup stress case: 51.98x at shift=12
  (separated) and 7.9x at shift=3 (within dispersion, not claimed); the default
  reference path is correspondingly slower there (48.37 ms vs 0.93 ms at
  shift=12). These numbers are single-operator microbenchmark results on one
  card and say nothing about whole-model speed.
- `N` beyond 2^31 is not supported (int32 indexing); the shift arithmetic is
  internal-fp64, which is slow on consumer GPUs but irrelevant at these sizes.
