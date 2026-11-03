# P5 Expert Start Kit (`P5-S0`)

The start kit unblocks every P5 sub-issue (P5-1…P5-9): it freezes the data
contract, provides a bit-exact FP32 oracle for the five WS1 operators,
generates seeded golden fixtures, and ships one acceptance command that any
backend PR can run independently.

## Sub-issue naming (development order posted on #8)

`P5-N` is the development-order label from the sequencing comment on #8;
GitHub issue numbers stay authoritative for links.

| Label | Scope |
| --- | --- |
| P5-S0 | This start kit (contract, oracle, fixtures, acceptance command) |
| P5-1 | `mxfp8_act_quant` (fwd + STE bwd) |
| P5-2 | `clamp_swiglu_weighted` (fwd + dgate/dup/dp_s) |
| P5-3 | `shared_grouped_lora_delta` (fwd + dX/dA/dB) |
| P5-4 | `mxfp8_mxfp4_grouped_gemm` (fwd + dX only) |
| P5-5 | `shared_expert_mlp` (fwd + dX only) |
| P5-6 | `moe_provider_adapter` (Megatron + vLLM injection) |
| P5-7 | WS2: EP placement, `expert_tensor_parallel_size = 1` gate |
| P5-8 | WS2: shared expert TP/SP + shared-once gate |
| P5-9 | WS2: adapter fail-closed under EP>1 / placements |

## What is in the kit

| Module | Contents |
| --- | --- |
| `rl_engine/moe/mx_format.py` | OCP MX codecs: E8M0 / E4M3 / E2M1, block-32 quantize/dequantize, nibble packing. Defines the golden bytes for P5-1/P5-4. |
| `rl_engine/moe/contract.py` | `ExpertBatch`, `SharedBatch`, `LoRAParams`, clamp constants, tensor fingerprints. P5-local subset of the Foundation `ExpertBatch` ABI (`p5-expertbatch-v1`). |
| `rl_engine/moe/oracle.py` | FP32 reference for the five operators plus the full routed/shared forward–backward compositions. |
| `rl_engine/moe/provider.py` | `ExpertProvider` protocol, `ReferenceProvider` (oracle-backed), `StubProvider` (fail-closed). |
| `rl_engine/moe/fixtures.py` | Seeded fixture cases and the golden-hash manifest (`tests/fixtures/p5/golden_hashes.json`, the CI anchor). |
| `rl_engine/moe/trace.py` | Boundary hashes + `first_divergence` (P5-local stand-in for `TraceEnvelope`). |
| `scripts/check_p5.py` | The acceptance command. |

## Frozen numeric contract (recap of #8 + decisions made here)

From the issues:

1. LoRA-only fine-tuning; base weights frozen — **no `dW` anywhere**.
2. Routed base is MXFP8 activation × MXFP4 frozen weight; block = 32, scale =
   E8M0, elements = E4M3 / E2M1 (OCP Microscaling v1.0).
3. Backward is BF16 (no MXFP8 re-quant); every reduction uses FP32 accumulators.
4. Route weight `p_s` is applied in `clamp_swiglu_weighted`
   (`h = SiLU(min(gate,10)) · clamp(up,−10,10) · p_s`), exactly once globally.
5. `mxfp8_act_quant` amax is a row-local 32-element reduction; backward is STE.
6. One-round SwiGLU: FP32 math, a single BF16 round on the output.

Decisions this kit had to freeze (flagged for review on #8; changing any of
them requires regenerating the manifest and bumping the schema/profile id):

| # | Decision | Rationale |
| --- | --- | --- |
| D1 | **E4M3 encode = clamp to ±448 in FP32, then RNE cast** (torch `float8_e4m3fn`). Bare torch cast maps overflow to NaN; clamp+cast equals PTX `cvt.satfinite`. | Matches hardware satfinite; pinned by golden tests. |
| D2 | **E8M0 scale recipe**: `shared_exp = floor(log2(amax)) − emax_elem` (8 for E4M3, 2 for E2M1); all-zero block → code 127 (scale 1). `floor(log2)` computed exactly via `frexp`. | OCP-recommended recipe; exact integer arithmetic. |
| D3 | **Oracle numeric profile `oracle-fp32-serial-v1`**: serial ascending-index accumulation, mul-then-add rounding (**no FMA fusion**). A strict CUDA kernel must use `__fmul_rn`/`__fadd_rn` to match, or register its own profile. | Reduction order must be pinned for byte-equality; serial ascending is auditable. |
| D4 | **LoRA inter-GEMM rounding**: `U = X·Aᵀ` rounds to BF16 before `Y = U·Bᵀ·α`; in backward, `dY·α` and `dU` also round to BF16 between GEMMs. | Matches a two-GEMM BF16 pipeline; must hold on both engines. |
| D5 | **Clamp subgradients are zero exactly at the bounds** (strict inequalities pass gradient). | Tie-break must be deterministic; pinned by tests. |
| D6 | **Shared expert applies no clamp** (`h = SiLU(gate)·up`), reusing the one-round SwiGLU with `p_s = None`, per the fixed math in P5-5 (#64). | P5-5 (#64) prose says "reuse clamp_swiglu_weighted (without p_s)" but its math shows no clamp — **open question raised on the issue**. |
| D7 | Gradients returned by backward are FP32 (the accumulator dtype); rounding at the next operator edge is BF16. | Consistent with "BF16 backward, FP32 reductions". |

## Byte-equality scope

Strict byte-equality is required **between train and infer on the same
numeric profile and device**. The committed manifest anchors the CPU x86
oracle; `scripts/check_p5.py` recomputes the oracle on the provider's device,
so transcendentals (sigmoid) never cross devices inside a strict comparison.
Hardware without equivalent capability (no FP8 MMA, fnuz formats, native MX
instructions) must register its own profile with an explicit tolerance —
never silently relax (P5-4/P5-6 contract).

## How a sub-issue PR uses the kit

1. Subclass `ReferenceProvider`, override only the operators your PR delivers
   (everything else stays on the oracle), and set `name`/`numeric_profile`.
2. Run `python scripts/check_p5.py --provider your.module:YourProvider
   [--device cuda]`. Every boundary must be byte-equal; exit code 1 otherwise.
3. Ship the check output (and your `provenance()`) in the PR description.

Fixture cases: `base_only_one_row`, `base_only_packed`, `lora_only`,
`base_plus_lora`, `uneven_experts` (zero-row experts), `shared_t1`,
`shared_t16`, plus operator edge cases `act_quant_edges` (powers of two,
RNE ties, zero rows) and `swiglu_boundary` (values at/inside/beyond clamps).

Regenerate the manifest after an intentional contract change:

```bash
python -m rl_engine.moe.fixtures --write-manifest
```

## Non-goals of the kit

No CUDA/Triton kernels, no Megatron/vLLM injection (P5-6), no EP transport or
combine (P4/P6), no multi-rank gates (P5-7…P5-9). `output_slot` is carried
through untouched for P6.
