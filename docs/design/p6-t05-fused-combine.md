# P6 T05 fused forward

Base: PR #448, `05bc53aa0396ffaf2c9241559b388159563e6b36`.

Implement the proposed `p6.synthetic-bf16.v1` forward on NVIDIA CUDA
with a Triton kernel. Input rows are already weighted BF16 `[P,H]`;
shared and residual are BF16 `[T,H]`. The validated host CombinePlan
supplies a compact `[T,6]` physical-row lookup. The kernel gathers each
slot, copies the first valid row, adds later valid rows in ascending slot
order in FP32, adds shared and residual, then casts once with BF16 RNE.
The all-invalid accumulator starts at positive zero. No atomics,
generic reduction, route weighting or canonical `[T,6,H]` materialization
is used in the default path.

`prepare_combine(plan, context, device)` performs host metadata validation
and uploads the lookup once. `prepared.forward(rows, shared, residual,
debug=False)` returns buffers containing the CUDA BF16 output, SavedForward
and tensor debug stages when requested. `fused_moe_combine_fwd(plan, rows, shared, residual,
context, debug=False)` is the convenience binding. This binding stays
isolated from the scalar oracle and global KernelRegistry. T06 owns
backward; this forward binding rejects tensors requiring gradients.
The input-bound `TritonCombineProvider` supplies T01 ProviderRegistry registration
for nine frozen conformance cases, using actual GPU readback and existing trace
envelopes. It is not the timed path or a production Foundation adapter.

Debug uses the same kernel arithmetic with additional stores under a
compile-time switch. It exposes the start-kit stage names, including six
slot partials, and must produce the same output bytes as debug-off.
Host serialization/hashing is an explicit diagnostic operation outside
the production launch. No second boundary schema is invented.

Unsupported devices, dtypes, strided layouts, autograd and arithmetic
outside the synthetic profile fail closed. Numeric profile checking is
performed in the fused kernel, which writes a small status buffer; the
eager checked API reads that status before returning. CUDA Graph use
requires the prepared launch API and explicit status validation after
replay, outside capture. This keeps graph launches allocation-free.
Inputs must not alias any output/debug/status write buffers: canonical reads may
otherwise race with another CTA's stores. Contiguous byte-range checks include
storage offsets and reject overlaps before launching.

CPU tests cover plan preparation and fail-closed behavior. Opt-in GPU
tests compare frozen raw bytes, every debug stage, debug-off, different
block sizes, physical row permutations, padding, batch partitioning,
nonfinite/subnormal failures and changed-input CUDA Graph replay.
The final-audit matrix additionally covers explicit serial T02-T04 equality,
debug RNG/logical identity, EP1/2/4/8 mock schedules, chunk/placement/delayed
completion/two-stream overlap, actual provider envelopes, sealed byte readback
and default-path allocation. No real EP communication is inferred from mocks.
A100 is supported for WS1 validation; H100 and live Foundation/WS2
certification remain pending. The user reported the full final-audit A100 suite
passing (91 passed, 12 optional compile tests skipped), and attempt-002 passed
artifact integrity/CPU replay verification (not GPU reexecution).
Benchmark reports checked eager and allocation-free prepared-launch timings;
CUDA Event intervals are not a pure-kernel or end-to-end latency certificate.
All six idle-A100 GPU-6 runs were BYTE_EQUAL, with a code digest matching this
binding/kernel. Prepared-launch Event median ratios against the measured eager
reference are 10.23-10.49x at T=32 and 8.75-9.41x at T=256; checked eager includes
allocation/status costs and is not consistently faster at T=256.
Raw results are retained in `docs/validation/p6-t05-a100-idle-gpu6-2026-10-03`.

Provisional integration points: public binding/return shape, P1 residual
identity, Foundation ABI and invalid/first-valid/signed-zero policy remain
owner-reviewed items in #448. The kernel follows that exact proposed
profile; changes require versioned conformance inputs, not silent casts.
