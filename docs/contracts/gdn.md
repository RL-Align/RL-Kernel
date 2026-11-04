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

All data tensors and indices share a device; `GDNContract.validate` describes
device-neutral metadata. The execution entry explicitly accepts CPU only.
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

All products, normalization, elementary functions, contractions and accumulators
above are FP32. `softplus20(x)` is `log1p(exp(x))` for x<=20, x otherwise;
the unused exponential branch is made finite for backward. No normalized-Q/K,
beta or delta cast back to input dtype occurs. Output is read **before** the
state store rounds. BF16 state storage rounds after **every token**, not just
at a chunk boundary. Output rounding does not feed back into the recurrence.

Gate calls operate on one active sequence's `[HV]` row at a time, as fixed in
#659 at `73ee54e8`. Calling sigmoid/exp over an entire `[B,HV]` tensor can select
different vector/scalar-tail paths when B changes on AVX2. The per-sequence
schedule removes that batch dependence; it does not require AVX2 and AVX512 to
produce identical bits. PR B already executes one sequence/token per step, but
also tests exact agreement with the grouped single-step interface.

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

## Source pins and semantic comparison

Initial investigation: 2026-11-03; #659 refresh: 2026-11-04 (Asia/Shanghai).

| Source | Fixed revision / status |
| --- | --- |
| PR B baseline / #659 | `8d5b8e2e37e972cef4a62c549a90048705bc58b2`, open/unmerged when fixed; includes per-sequence gate fix `73ee54e8` |
| Superseded PR B baseline | `10db999c366d29c0b633b34e3e75eac334190d6b`; old local PR B preserved on `archive/gdn-task20-before-avx-update` |
| #469 original | `2b64780b2e1be5d97e936405bb1e899fea14215c`; merged into test-qwennext at `d0541b87f991f91169027bcf8d4077c23d687b01` |
| #507 | `d849e180adbe5655a29a45cdef3f0093719efedb`, open |
| RFC RL-Kernel inspection anchor | `59f3ae01c76c5607a51a368ab1e8f6c00556cbc0`; historical anchor, not PR B base |
| miles RFC anchor | `65feb4a616bd935890cdac6ca06b1dca5cee28dd` |
| miles current main observed | `e8046accd214b8038e8593000ad4c39243a5aaa4`; relevant linear-attention file last changed at `44a1dda6b35b18176b5f1b766d12e414824b8b1a`, same blob `507901b1f80755e9c34c89c544089e4860633cf4` as RFC anchor |
| FLA in pinned miles Dockerfile | 0.5.2, tag resolved to `9c8e42e762fce087c27b673af4922795d9edb85e` |
| FlashQLA in pinned miles Dockerfile | `7c7dfe16416ad21b1d03258189fc8d3b8460ae06` |
| Qwen configuration observed | `995ad96eacd98c81ed38be0c5b274b04031597b0` |

The [Qwen configuration][config] selects qwen3_5_moe, H=16, HV=32, K=V=128,
width-4 convolution and FP32 recurrent state. The [pinned guide][guide] and
[model arguments][model] select Qwen3.5-derived GDN. These describe target
geometry and integration, not arithmetic compatibility. The public configuration
was also reread at the fixed revision above.

In the following matrix **known** means directly observed source behavior;
**inferred** means a mathematical relationship, not measured provider agreement;
**unknown** means no qualification or insufficient source evidence.

| Item / shared meaning | #659 / #469 | #507 | miles and actual kernel evidence | Status |
| --- | --- | --- | --- | --- |
| Delta recurrence, grouped heads | `_step`: decay, residual, outer update, output; `j//(HV/H)` | `_provider`, `_PackedDecode`: same vLLM decode kernel | FLA `chunk_delta_h.py`: same grouped indexing, WY/dot chunk updates; FlashQLA Hopper `fused_fwd.py`: chunk GEMMs | source paths known; mathematical equivalence inferred; numerical agreement unknown |
| QKV / scale | `_step`: q\|k\|v; scale q after norm | `_validate`: BF16, K=V=128, default `128**-0.5` | miles `convolve`/`project`: group-major rows split into Q/K/V, z/b/a projections; FLA/FlashQLA default K^-0.5 | known; conversion/projection outside this contract |
| Normalization | `_l2_normalize`: FP32 division by sqrt, eps=1e-6 | requests normalization inside decode provider | FLA `l2norm.py::l2norm_fwd_kernel`: FP32 sum and reciprocal sqrt multiplication, then store to Q/K dtype; FlashQLA dispatch calls its own `l2norm_fwd` | FLA difference known; FlashQLA detailed norm arithmetic unknown |
| Beta / decay | `_step`, `_softplus`: FP32 beta and decay, branched threshold20 | same provider; backward recomputes retained reference | miles `gdn_recurrence`: input-dtype sigmoid, FP32 softplus/exp decay; supplied to kernels | known difference; no miles support claim |
| Accumulators / casts / order | FP32, padded R32, per-token state dtype, QKV output dtype | FP32 state, BF16 output | FLA `chunk.py`, `chunk_delta_h.py`, `chunk_o.py`: chunk cumsum, WY, FP32 state accumulators, low-precision intermediate state/dot operands, FP32 final state; FlashQLA Hopper: FP32 fragments copied to low-precision shared memory and GEMMs | source casts known; compiled pairing/contraction and provider bit agreement unknown |
| State layout / indices / null | V-major; <=0 skipped, independent cloned bank | sequence indices positive and unique; whole independent bank | wrapper requests no initial/final state; FLA/FlashQLA default K-major and offer V-major option, no vLLM paged slot ABI | known; a universal slot0 rule is unjustified |
| Packing / empty / chunks | step only | `packed_recurrent_sequence`: CPU offsets, empty allowed, position loop over provider | wrapper supplies packed/CP metadata; FLA/FlashQLA real chunk kernels, not per-token stores | known; miles empty/continuation agreement unknown |
| State gradients | differentiable `.to`/clone/index updates | custom backward recomputes reference and returns initial-state gradient | FLA/FlashQLA backward accept state gradients; miles wrapper does not expose persistent initial/final state | known source behavior; integration continuity unknown |
| Conv layout / numeric path | `_update`: DS or SD, token cast to cache dtype, promoted operand product then FP32 bias-first sequential sum, SiLU; 0 skipped, negatives rejected | BF16 `[blocks,dim,3]`, width4 no bias/SiLU; separate provider bridge | miles `causal_short_conv`: FLA mix/Triton, channels split only for indexing; FLA conv dispatch uses distinct implementations | known dispatch/layout; miles conv product rounding unknown |

Primary source locations: [retained recurrence][step], [retained conv][conv],
[#507 GDN bridge][bridge], [#507 conv bridge][convbridge], [author interface][author],
[miles linear attention][miles], [Qwen3.5 family][qwen], [Docker pins][docker],
[FLA chunk entry][fla], [FLA state arithmetic][flah], [FLA output arithmetic][flao],
[FLA normalization][flan], [FlashQLA entry][qla], [FlashQLA Hopper arithmetic][qlah].
No provider code is imported or copied by the new implementation.

The #469 source links describe provenance. Current CPU gate scheduling is the
[corrected #659 step][corrected-step], not an exact extraction of #469. The
original batched reference fails two batch-invariance cases under AVX2 in the
local PyTorch environment; AVX512 passes those cases. Historical #469 bitwise
extraction evidence does not extend to the corrected reference. #507's older
copy must be reconciled during integration; no GPU/provider fidelity is implied.

Only one profile is implemented: #507 is a restricted configuration of the
existing vLLM profile, not separate arithmetic. Miles' input-dtype beta and
chunk arithmetic justify keeping it unqualified; an incomplete guessed miles
profile would not be a reference. Unknown details above do not block this
explicit CPU contract. They block provider-compatibility claims and must be
resolved before a future implementation advertises that compatibility.

## Tests, #507 relationship and remaining tasks

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
references. PR A regression tests retain their original single-step APIs.

Task 20 checklist: explicit recurrence, state schema/dtype, cast/reduction
schedule, temporal chunk boundaries, packed/empty/isolation semantics,
non-mutation and gradient continuity, CPU oracle and compatibility fixtures.
No GPU/TP4, miles, cross-hardware or performance gate is passed by these tests.
Tasks 21–27 still own projection/device conv, qualified gates, device recurrent
and optimized chunk kernels, device backward, adapter state lifecycle and
cross-phase/CP/TP qualification. [RFC #563][rfc563] separates accuracy,
consistency and performance; [canonical layout][layout] assigns contracts and
references their respective ownership. This work does not close RFC #642.

[rfc]: https://github.com/RL-Align/RL-Kernel/issues/642
[scope]: https://github.com/RL-Align/RL-Kernel/issues/642#issuecomment-6579049792
[config]: https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/995ad96eacd98c81ed38be0c5b274b04031597b0/config.json
[guide]: https://github.com/radixark/miles/blob/65feb4a616bd935890cdac6ca06b1dca5cee28dd/docs/models/qwen/qwen3-6-moe.md
[model]: https://github.com/radixark/miles/blob/65feb4a616bd935890cdac6ca06b1dca5cee28dd/scripts/models/qwen3.6-35B-A3B.py
[step]: https://github.com/RL-Align/RL-Kernel/blob/2b64780b2e1be5d97e936405bb1e899fea14215c/rl_engine/reference/linear_attn/gated_delta_rule.py
[corrected-step]: https://github.com/RL-Align/RL-Kernel/blob/8d5b8e2e37e972cef4a62c549a90048705bc58b2/rl_engine/reference/linear_attn/gated_delta_rule.py
[conv]: https://github.com/RL-Align/RL-Kernel/blob/2b64780b2e1be5d97e936405bb1e899fea14215c/rl_engine/reference/linear_attn/causal_conv1d.py
[bridge]: https://github.com/RL-Align/RL-Kernel/blob/d849e180adbe5655a29a45cdef3f0093719efedb/rl_engine/integrations/engines/train/vllm/qwen3_next_gdn.py
[convbridge]: https://github.com/RL-Align/RL-Kernel/blob/d849e180adbe5655a29a45cdef3f0093719efedb/rl_engine/integrations/engines/train/vllm/qwen3_next_conv.py
[author]: https://github.com/RL-Align/RL-Kernel/pull/507#issuecomment-6583519108
[miles]: https://github.com/radixark/miles/blob/65feb4a616bd935890cdac6ca06b1dca5cee28dd/miles_plugins/models/linear_attn.py
[qwen]: https://github.com/radixark/miles/blob/65feb4a616bd935890cdac6ca06b1dca5cee28dd/miles_plugins/models/qwen3_5.py
[docker]: https://github.com/radixark/miles/blob/65feb4a616bd935890cdac6ca06b1dca5cee28dd/docker/Dockerfile
[fla]: https://github.com/fla-org/flash-linear-attention/blob/9c8e42e762fce087c27b673af4922795d9edb85e/fla/ops/gated_delta_rule/chunk.py
[flah]: https://github.com/fla-org/flash-linear-attention/blob/9c8e42e762fce087c27b673af4922795d9edb85e/fla/ops/common/chunk_delta_h.py
[flao]: https://github.com/fla-org/flash-linear-attention/blob/9c8e42e762fce087c27b673af4922795d9edb85e/fla/ops/common/chunk_o.py
[flan]: https://github.com/fla-org/flash-linear-attention/blob/9c8e42e762fce087c27b673af4922795d9edb85e/fla/modules/l2norm.py
[qla]: https://github.com/QwenLM/FlashQLA/blob/7c7dfe16416ad21b1d03258189fc8d3b8460ae06/flash_qla/ops/gated_delta_rule/chunk/__init__.py
[qlah]: https://github.com/QwenLM/FlashQLA/blob/7c7dfe16416ad21b1d03258189fc8d3b8460ae06/flash_qla/ops/gated_delta_rule/chunk/hopper/fused_fwd.py
[rfc563]: https://github.com/RL-Align/RL-Kernel/issues/563
[layout]: ../architecture/repository-layout.md
