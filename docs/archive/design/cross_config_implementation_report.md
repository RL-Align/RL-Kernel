# Cross-Configuration Alignment Implementation Report

Status: V1 framework snapshot, 2026-07-19

Related work:

- [Roadmap #83](https://github.com/RL-Align/RL-Kernel/issues/83)
- [Cross-configuration alignment #111](https://github.com/RL-Align/RL-Kernel/issues/111)
- [Numerical contract #108](https://github.com/RL-Align/RL-Kernel/issues/108)
- [V1 contract](../../contracts/cross_config_logprob_drift_contract.md)

## Result and claim boundary

This change provides a small framework for planning and executing paired
rollout/training logprob comparisons across controlled configuration changes. It
includes strict configuration loading, bounded case planning, exact semantic
operator selection, lifecycle-aware runtime materialization, paired read-only
scoring, fixed-contract comparison, append-only artifacts, and validated resume.

The included executable path is deliberately CPU-only. It validates framework
plumbing with a synthetic model and temporary selected-logprob backends; it does
not claim production vLLM, FSDP, TP, CP, accelerator, or distributed numerical
alignment. The S1, S2, and S3 examples are plans, not execution evidence.

## Architecture

The implementation keeps configuration, operator resolution, runtime ownership,
and execution separate:

```text
JSON -> ExperimentConfig -> Planner -> ExperimentPlan
                                      |
                            build_execution_plan
                                      |
                    operator-bound ExecutionPlan
                         /                     \
                operator session       RuntimeMaterializer
                                              |
                                       RuntimeBinding
                                              |
                       ArtifactStore <- PairedRunner -> fixed comparator
```

| Boundary | Responsibility |
| --- | --- |
| `config.py` and `planner.py` | Load strict, versioned JSON; normalize the ten supported knobs; emit an `ExperimentPlan` containing a baseline plus declared OAT or explicit pairwise cases under a fixed 256-case cap; compute stable semantic case IDs without importing a runtime. |
| `build_execution_plan` | Resolve rollout/training selections, bind them into immutable case identity, and emit canonical operator-bound `ExecutionPlan` rows shared by planning and execution. |
| `SemanticOperatorCatalog` | Store immutable backend descriptors and their target, device, dtype, per-target required topology, lifecycle, factory, and observability constraints. |
| `OperatorSession` | Resolve and instantiate exact rollout/training implementations for one case, cache only within that case, and produce concrete provenance. |
| `RuntimeMaterializer` | Apply each normalized knob through an owning adapter and report requested, materialized, and actual values with status and lifecycle evidence. |
| `RuntimeBinding` | Carry only backend-neutral batch, side-configuration, topology, scorer, operator-backend, and runtime-kind mappings. Runtime-specific engine objects stay behind the adapter boundary. |
| `PairedRunner` | Supervise isolated rollout/training scoring children, enforce timeout and read-only model state, validate ranks and exact operator instances, compare selected logprobs, and coordinate resume/publication. |
| `ArtifactStore` | Publish immutable attempt directories and write `COMPLETE` last with SHA-256 seals for every required payload. Resume accepts only an attempt whose identity, execution, provenance, tensors, and comparison still validate. |

Runtime bindings deep-freeze their execution handoff. Rollout topology contains
only rollout-owned world/TP/CP state, while training topology contains only
training-owned world/sharding state; neither side is padded with fields owned by
the other.

The fixed-threshold comparator applies the repository numerical contract only to
active selected tokens. Identity violations, invalid artifacts, non-finite
scores, and zero active tokens cannot become passes. Diagnostics remain separate
from the pass/fail rule.

## Configuration and operator selection

Execution controls do not live in experiment JSON. `scenario` is metadata; the
CLI chooses planning versus execution, the runtime adapter, temporary-operator
authorization, timeout, and resume policy.

`logp.backend` is the concise choice when rollout and training use the same
selected-logprob implementation. The optional top-level `operators` mapping is
the extension point for independent sides and per-backend options:

```json
{
  "baseline": {
    "logp": {"backend": "rlkernel.reference_logp"}
  },
  "operators": {
    "selected_logprob": {
      "rollout": "rlkernel.reference_logp",
      "training": {
        "backend": "vendor.training_logp",
        "options": {"mode": "exact"}
      }
    }
  }
}
```

The explicit rollout backend must agree with the baseline `logp.backend`.
Explicit operators cannot be combined with `logp.backend` interventions because
that would make the planned knob differ from the implementation actually used.
Unknown fields, threshold overrides, hidden execution controls, duplicate JSON
keys, and non-finite values are rejected.

## CLI

Planning validates and records a plan without constructing a runtime:

```bash
python -m rl_engine.validation.cross_config plan \
  configs/experiments/cross_config/cross_config_s3_qwen3_8b_tp4_cp4_bf16.json
```

The only shipped execution adapter is the explicit CPU smoke runtime:

```bash
python -m rl_engine.validation.cross_config run \
  configs/experiments/cross_config/cross_config_s0_cpu_smoke.json \
  --runtime cpu-smoke \
  --allow-smoke-operators
```

Both commands accept `--output-root`. `run` also exposes a per-attempt timeout
and `--no-resume`; these policies are intentionally absent from the JSON schema.

## CPU testing adapter

`rl_engine.validation.fixtures.cpu_cross_config` owns the synthetic causal model,
stateless CPU scorer, `CpuSmokeMaterializer`, canonical batch construction, and
CPU experiment helpers. Keeping these objects outside the core package prevents
test hardware and model assumptions from becoming runtime abstractions.

Temporary selected-logprob implementations live under
`rl_engine/alignment/testing/smoke_ops`. They advertise CPU as their only device,
are not registered by default, and require explicit policy authorization. The
reference backend performs `log_softmax` plus gather; the offset backend is used
only by focused mismatch tests. The shipped S0 example selects reference on both
sides and contains one baseline case.

## Scenario evidence

| Scenario | Planned cases | Current evidence |
| --- | ---: | --- |
| S0 CPU framework smoke | 1 | One reference/reference case passes on CPU; a second invocation validates and resumes the same complete attempt. |
| S1 distributed smoke | 5 | Configuration loading and planning only. |
| S2 vLLM TP versus FSDP | 10 | Configuration loading and planning only. |
| S3 Qwen3-8B TP=4, CP=4, BF16 | 11 | Configuration loading and planning only. |

Planning success does not imply that the requested production topology can be
materialized. Unsupported or unobservable runtime settings fail strict execution
instead of silently falling back.

## Validation snapshot

```text
Focused contract/runtime/runner/CLI and existing regression tests:
60 passed in 2.20s

CPU-collectable repository suite:
418 passed, 242 skipped in 15.83s

Named scenario checks:
S0 run: 1 pass, then 1 validated resume
S1/S2/S3 plan: 5 / 10 / 11 cases, no runtime constructed
```

The final review also checks JSON syntax, formatting, static typing, strict
documentation build, and `git diff --check`.

The full CPU command excludes `test_grpo_loss.py` and `test_ratio_kl.py`
because those modules require Triton during collection. It ran outside the
restricted sandbox so Gloo and POSIX shared-memory tests could use host
resources.

## Extension path and known gaps

A production backend extends the semantic catalog with a descriptor and factory,
then supplies injection/read-back hooks through its runtime adapter. The planner
and runner do not need backend-specific branches. Operator correctness remains
owned by the operator implementation; the framework verifies exact selection,
materialization evidence, paired identity, comparison, and provenance.

Production execution still requires:

- verified selected-logprob injection and read-back for the rollout engine;
- a read-only pre-update training scorer for FSDP and distributed rank evidence;
- context-parallel application/read-back and process-group orchestration;
- accelerator-backed lifecycle and cleanup tests; and
- the production kernels tracked by their owning workstreams.

Until those adapters exist, S1-S3 remain reproducible planning inputs and the CPU
smoke remains a framework claim only.
