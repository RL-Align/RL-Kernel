# WS1 C2 (#268) Closeout Evidence

**Issue:** #268 · **Parent:** #266 · **Workload:** `ws1-qwen3-8b-dense-primary-v6`

**Branch:** `feat/ws1-c2-canonical-workload-268`

## Deliverables

| Path | Role |
| --- | --- |
| `rl_engine/config/workloads/qwen3_8b.json` | SSOT workload identity / matrix / profiles / cases |
| `rl_engine/config/workload.py` | Load, validate, logical identity, pad/pack/chunk restore |
| `tools/validation/models/ws1_reference.py` | One-command reference emission |
| `tools/validation/operators/ws1_candidate_evidence.py` | Executable CUDA/Triton candidate provenance |
| `tests/validation/operators/test_ws1_workload.py` | CPU acceptance tests |
| `docs/archive/design/ws1-c2-268-workload-plan.md` | Landing plan |
| `docs/archive/design/ws1-c2-268-closeout-evidence.md` | This map |

## Acceptance criteria map

| #268 AC | Status | Evidence |
| --- | --- | --- |
| Manifest pins numerics-affecting fields | **Pass** | model, seed, tokens, prompt/completion lenses, masks, positions, dtypes, clip, aggregates, RNG, TF32 ref |
| Full Qwen3-8B Dense identity + weight hash | **Pass** | config fingerprint + shard SHA-256 `content_hash` |
| Same workload ID → same fixture/reference identity | **Pass** | `fixture_identity_sha256` + `fixture_hash` tests |
| pad/pack/chunk restore logical identity | **Pass** | `apply_padding` / `apply_packing` / `apply_chunking` + restore tests |
| B1 singleton_aggregate vs BN same multiset | **Pass** | `singleton_aggregate_plan` test |
| Naming: singleton_aggregate ≠ C1 roles; no bare baseline | **Pass** | `forbidden_comparison_roles` + `report_naming` |
| 2×2 + perm + multi-chunk non-divisible + pad/varlen | **Pass** | primary matrix + varlen samples `[11,16,13,19]` |
| clip_interval for clipfrac0 | **Pass** | `[0.8, 1.2]` aligned with C1 |
| Dropout/sampling/RNG policy; undeclared hard-fail | **Pass** | `stochastic_policy` + helper test |
| Short + representative fixtures hit declared candidates | **Pass** | fixture-derived shapes + runtime candidate evidence runner |
| Stable case_id for C8/C10/C11 reference | **Pass** | `representative_cases[].case_id` |
| expected + actual backend/kernel + algorithm property | **Pass** | runner executes each case, records actual class path, compares it to expected, and checks outputs |
| One command emits reference (workload ID, seed, dtype) | **Pass** | `tools/validation/models/ws1_reference.py` |
| Packing / QK-Norm / required ops status | **Pass** | packing supported + packed fixture; qk_norm required |
| Both profiles enumerate required nodes; no untracked missing | **Pass** | Triton required nodes are all `declared` |
| `linear_logp` is not WS1 required | **Pass** | `required_chain_ops` status `optional_fused_path`; no C8 row |
| Packing is a layout helper | **Pass** | C2 `packing.supported` + C3/C4 CPU pack adapters; C8 N/A |

C2 executes all representative cases. Full-model dispatch provenance remains owned by C3/C8/C10/C11; this does not claim the C9/C10 full-model gate.

## Verification commands

```bash
# From repo root with PYTHONPATH=repo root (or editable install)
python -m pytest tests/validation/operators/test_ws1_workload.py -q
python tools/validation/models/ws1_reference.py --dtype bf16 --cell-id BN/full --emit-json -
python tools/validation/operators/ws1_candidate_evidence.py --emit-json ws1-c2-runtime-evidence.json
```

Expected: all C2 tests green; the reference CLI emits identity/digests; the candidate runner executes all CUDA/Triton cases and reports `passed: true` with runtime-observed actual paths.

Validated on 2026-08-12:

- NVIDIA GeForce RTX 3060 Laptop GPU, SM86, single GPU
- PyTorch `2.8.0+cu128`, CUDA runtime `12.8`, Triton `3.4.0`, Python `3.13.3`
- Representative runtime evidence: 10/10 CUDA + Triton cases passed
- Focused review/workload/contract suite: 81 passed (including CUDA/Triton runtime evidence)
- Full repository CUDA/Triton pytest: 1622 tests collected; exit code 0 (1501 passed, 121 hardware/CI skips)
- `pre-commit run --all-files`: all 7 hooks passed

## Residual (explicitly not #268)

| Item | Owner |
| --- | --- |
| Full-model runtime observed actual backend | C3 / C8 / C10 / C11 |
| Triton embedding / lm_head / logp | declared candidates; C8 execute owns green/red |
| #150 numerical asserts / full-model e2e | C9 / C10 |
| Full WS1 EXIT | #266 after C1–C11 |

## Close recommendation

Close **#268** once this branch is merged. Do **not** claim #266 WS1 EXIT from C2 alone.
