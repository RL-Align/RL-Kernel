# Gated DeltaNet sequence and state contract

This is the CPU semantic baseline for [RFC #642 Task 20][rfc] under the
[maintainer's accepted scope][scope]: shared numerical/state contract, reference
and tests. It builds on #659's corrected reference derived from #469, and does
not qualify a device backend, engine, model, TP/CP layout or provider pair.

## Public boundary

`rl_engine.contracts.operators.gdn.GDNContract` describes geometry, scale,
normalization and numerical profile. The sole executable profile is
`vllm_decode_v1`, naming the retained arithmetic rather than certifying vLLM
agreement. Unknown profile names fail explicitly. Geometry is independent of
model/provider head widths; HV must be divisible by H.

`rl_engine.reference.linear_attn.gated_delta_rule_sequence` takes:

| Argument | Contract |
| --- | --- |
| `mixed_qkv` | `[T, 2*H*K + HV*V]`, contiguous logical q\|k\|v sections; FP32, BF16 or FP16 |
| `a`, `b` | `[T, HV]`, same dtype as QKV; raw decay/beta logits |
| `A_log`, `dt_bias` | `[HV]`, FP32 |
| `state` | `[blocks, HV, V, K]`, FP32 or BF16 |
| `indices` | `[B]`, int32/int64, distinct nonnegative slots in the state bank |
| `cu_seqlens` | CPU int32/int64 `[B+1]`, nondecreasing, starts at 0, ends at T |
| `contract` | Keyword-only `GDNContract(H, HV, K, V, scale=..., use_qk_l2norm=True)` |

State `[blocks, HV, V, K]` is the reference's logical V-major representation.
Backends may use K-major, tiled or other physical storage; adapters own layout
conversion and must preserve logical state meaning. K=V can hide axis swaps,
so shape equality alone is insufficient. Packed `q|k|v` is this reference's input
schema, not a required physical projection layout. Miles/group-major projection
channels need explicit reordering, with consistent convolution weight and cache
channel order.

All data tensors and indices share a device. `GDNContract.validate()` validates
this normalized reference invocation; its CPU offsets and host boundary
extraction do not define a universal GPU/MUSA launch interface. Backend launch
metadata and validation may differ while preserving the same logical semantics.
The execution entry explicitly accepts CPU only.
Strided tensors are allowed. Bad geometry, nonfinite scale, unknown profile,
wrong shapes/dtypes/devices, offsets or duplicate/out-of-range slots raise
`ValueError`, including when all sequences are empty. Tensor values are not
scanned for finiteness; ordinary PyTorch floating-point propagation applies.

Return values are packed output `[T, HV, V]` in QKV dtype and a complete,
independent state bank in its input dtype. No inputs are mutated. Unused slots
remain unchanged. Empty sequences produce no tokens and do not advance state;
`cu_seqlens=[0]` describes zero sequences. Even empty calls return independent
state storage and preserve the identity gradient to the initial state.

Every shared slot, including zero, can hold state. There is **no null sentinel**
in this sequence API. Internally a private dummy slot translates logical slots
to the existing step's positive slots. The old `GatedDeltaRuleRecurrentStepOp`
API continues to skip indices <=0; `CausalConv1dUpdateOp` continues to skip only
0 and reject negatives. These legacy policies are not universal state rules.
An adapter calling legacy convolution must also translate logical slot0;
passing zero directly would skip its update. The dummy-slot convention is
private to this CPU sequence reference, not required of GPU/MUSA backends.
Initialize by passing an explicitly zeroed bank; continuation passes the entire
returned bank and the corresponding logical indices without detaching it.

```python
import torch
from rl_engine.contracts.operators.gdn import GDNContract
from rl_engine.reference.linear_attn import gated_delta_rule_sequence

contract = GDNContract(1, 1, 1, 1, scale=1.0, use_qk_l2norm=False)
out, state = gated_delta_rule_sequence(
    torch.ones(2, 3), torch.zeros(2, 1), torch.zeros(2, 1),
    torch.zeros(1), torch.zeros(1), torch.zeros(1, 1, 1, 1),
    torch.tensor([0]), torch.tensor([0, 2]), contract=contract,
)
```

## Recurrence and numerical profile

For value head j use Q/K head `j // (HV/H)`. With V-major state S:

```text
q, k, v = unpack(qkv).float()
if normalize: x = x / sqrt(R32(x*x) + 1e-6)  # separately for q and k
q = q * scale
g = -exp(A_log.float()) * softplus20(a.float() + dt_bias.float())
beta = sigmoid(b.float())
D = S.float() * exp(g)
delta = (v - R32(D * k)) * beta
U = D + outer(delta, k)
output = R32(U * q).to(qkv.dtype)
S_next = U.to(state.dtype)
```

For `vllm_decode_v1`, all products, normalization, elementary functions,
contractions and accumulators above are FP32. `softplus20(x)` is
`log1p(exp(x))` for x<=20, x otherwise;
the unused exponential branch is made finite for backward. No normalized-Q/K,
beta or delta cast back to input dtype occurs. Output is read **before** the
state store rounds. BF16 state storage rounds after **every token**, not just
at a chunk boundary. Output rounding does not feed back into the recurrence.

Gate calls operate on one active sequence's `[HV]` row at a time, as fixed in
#659 at `73ee54e8`. Calling sigmoid/exp over an entire `[B,HV]` tensor can select
different vector/scalar-tail paths when B changes on AVX2. The per-sequence
schedule removes that batch dependence; it does not require AVX2 and AVX512 to
produce identical bits. The sequence entry executes one sequence/token per step;
tests also check exact agreement with the grouped single-step interface.

`R32` pads the last axis to a multiple of 32, reshapes to length-32 groups,
sums each group, then sums the group results. PyTorch supplies the within-axis
sum implementation and elementary functions. This specifies the retained
execution schedule, not a universal hardware-independent reduction tree or
transcendental bit pattern. Software/device changes require new qualification.

Sequences execute independently, each in increasing token order. A temporal
chunk is just another call with the prior returned bank. Chunk boundaries add
no cast, reset, rounding or detach. This is not Task 24's optimized chunk scan.
Autograd traverses the reference and state handoffs; no new backward kernel or
guarantee of bitwise parameter-gradient accumulation order is introduced.

## Provider scope

Only `vllm_decode_v1` is implemented. Its [single-step reference][corrected-step]
derives from #469 and includes #659's per-sequence CPU gate correction. This
preserves the documented recurrence and rounding schedule, not exact bitwise
extraction fidelity to the original batched #469 reference.
The schedule above is specific to this profile. Future profiles need explicitly implemented and
validated execution paths; accepting a new profile string alone is insufficient.
Extending profiles must preserve the existing profile's semantics.

[#507's interface][author] uses a restricted configuration of this profile:
BF16 QKV/a/b, FP32 parameters and state, K=V=128, and positive state slots.
It does not need a separate arithmetic profile. Its older reference copy must
be reconciled during integration before claiming provider agreement.

The [miles recurrence][miles] computes beta in input dtype and calls chunked
kernels with a different arithmetic schedule. It is not a supported profile.
Mathematical equivalence alone does not establish bitwise agreement with
optimized chunk/device implementations. CPU accuracy or within-dispatch
invariance does not certify vLLM, miles, GPU/TP4, cross-ISA or
cross-PyTorch-version agreement.

## Validation and scope

Run with CPU PyTorch (no provider, native extension or GPU required):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest tests/reference/linear_attn tests/ops/linear_attn -q -rs
python -m mypy rl_engine/contracts/operators/gdn.py rl_engine/reference/linear_attn
python ci/run.py layout -q
```

To qualify each CPU dispatch separately, set `ATEN_CPU_CAPABILITY=avx2` or
`ATEN_CPU_CAPABILITY=avx512` before starting a fresh Python process, verify
`torch.backends.cpu.get_cpu_capability()` reports that dispatch, then run the
same suites. Only run a dispatch supported by the host. These are independent
within-ISA checks, not a cross-ISA equality test. CI's existing AVX2 step also
executes the sequence/grouped-step regression across gate head widths.

`test_gdn_sequence.py` includes a hand-worked recurrence, an independent FP64
per-head matrix oracle, normalization/head/width/dtype coverage, fixed beta,
state-rounding and cancellation fixtures, full/chunked and packed/permuted
raw-bit comparisons, empty and non-mutating state checks, and independent
autograd comparison for all floating inputs. FP64 oracle comparisons use tight
FP32 tolerances or storage-ULP allowances for low precision; these are accuracy
checks. Raw-bit regrouping checks use the identical per-token CPU path.
Gradient comparisons allow FP32 accumulation-order error.

The #507 fixture uses BF16 QKV/a/b, FP32 parameters and
`[blocks, HV, 128, 128]` FP32 state with positive slots. The width4 BF16
no-bias/SiLU convolution fixture composes existing conv steps with this GDN
entry and checks prompt gradients across both state boundaries. No #507 CUDA
bridge is copied. Its positive slots and tensors can be supplied directly;
this reference additionally supports slot0 and other geometry. A subprocess
blocks engine/backend/provider/extension imports and actually executes the
references. Single-step regression tests retain the existing APIs.

These tests cover the recurrence, state schema/dtype, cast/reduction schedule,
temporal chunk boundaries, packing, isolation, ownership and gradients.
No GPU/TP4, miles, cross-hardware or performance gate is passed by these tests.
Tasks 21–27 still own projection/device conv, qualified gates, device recurrent
and optimized chunk kernels, device backward, adapter state lifecycle and
cross-phase/CP/TP qualification. [RFC #563][rfc563] separates accuracy,
consistency and performance; [canonical layout][layout] assigns contracts and
references their respective ownership. This work does not close RFC #642.

[rfc]: https://github.com/RL-Align/RL-Kernel/issues/642
[scope]: https://github.com/RL-Align/RL-Kernel/issues/642#issuecomment-6579049792
[corrected-step]: https://github.com/RL-Align/RL-Kernel/blob/8d5b8e2e37e972cef4a62c549a90048705bc58b2/rl_engine/reference/linear_attn/gated_delta_rule.py
[author]: https://github.com/RL-Align/RL-Kernel/pull/507#issuecomment-6583519108
[miles]: https://github.com/radixark/miles/blob/65feb4a616bd935890cdac6ca06b1dca5cee28dd/miles_plugins/models/linear_attn.py
[rfc563]: https://github.com/RL-Align/RL-Kernel/issues/563
[layout]: ../architecture/repository-layout.md
