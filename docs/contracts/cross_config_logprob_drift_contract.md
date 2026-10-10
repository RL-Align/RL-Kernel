# Cross-Configuration Logprob Drift Contract

Status: V1 implementation contract

Related work:

- [Roadmap #83](https://github.com/RL-Align/RL-Kernel/issues/83)
- [Cross-configuration alignment #111](https://github.com/RL-Align/RL-Kernel/issues/111)
- [Numerical contract #108](https://github.com/RL-Align/RL-Kernel/issues/108)

## Goal and boundary

This framework isolates configuration changes that can make rollout-selected
log probabilities differ from training-side recomputation. It provides typed
plans, lifecycle-aware runtime materialization, exact semantic-operator
selection, paired read-only scoring, append-only artifacts, and safe resume.

It does not implement production AG, RS, GEMM, attention, logprob, TP-invariant,
CP-aware, or deterministic collective kernels. Those implementations remain
owned by their operator workstreams and integrate through the semantic operator
catalog described below.

## The only pass/fail rule

For every active selected response/action token:

```text
abs(training_logprob - rollout_logprob) > fixed_threshold
```

The fixed threshold is loaded from the repository numerical contract. It is not
a config field, CLI flag, experiment axis, workload policy, or operator option.
Equality with the threshold is not a mismatch.

```python
mismatch_mask = active_mask & (
    torch.abs(training_logprobs - rollout_logprobs) > fixed_threshold
)
```

Mean, percentiles, maximum absolute difference, mismatch ratio, worst-token
location, and approximate KL are diagnostics only. They never change pass/fail.
The token-level artifact persists both logprob tensors, the active mask, and the
resolved threshold so the mask can be recomputed offline.

Zero active tokens produce `ZERO_ACTIVE_TOKENS`, never a pass. Non-finite or
non-floating active scores produce `INVALID_ARTIFACT`.

## Identity before numerics

A comparison is valid only when both scorers use the same logical input:

- immutable checkpoint and model version;
- tokenizer ID and tokenization policy;
- generated token IDs and selected-token IDs;
- active and attention masks;
- pre-update model state;
- required position, cache, and packing metadata.

The training scorer teacher-forces the already generated sequence. It cannot
generate replacement tokens, use a KV cache when the frozen identity forbids
one, own an optimizer, update parameters or buffers, or leave the model in a
different mode. An identity violation is `INVALID_IDENTITY`, not numerical
drift.

## V1 configuration

The user supplies one explicit baseline plus declared interventions. Lists do
not imply a Cartesian product.

```json
{
  "schema_version": "cross_config.experiment_config.v1",
  "experiment_id": "qwen3-8b-alignment",
  "scenario_id": "qwen3-8b-tp4-cp4-bf16",
  "contract_source": "ws1",
  "contract_version": "current",
  "strategy": "one_at_a_time",
  "strict_fallback": true,
  "identity": {"...": "frozen scoring identity"},
  "baseline": {
    "batch": {"size": 8},
    "rollout": {
      "tensor_parallel_size": 4,
      "context_parallel_size": 4,
      "dtype": "bfloat16",
      "enable_prefix_caching": true,
      "enforce_eager": false
    },
    "training": {
      "attention_backend": "flash_attention_2",
      "compute_dtype": "bfloat16",
      "sharding": "fsdp"
    },
    "logp": {"backend": "native"}
  },
  "interventions": [
    {"path": "batch.size", "values": [1]},
    {"path": "logp.backend", "values": ["rlkernel.reference_logp"]}
  ],
  "scenario": {"level": "S3", "device": "cuda"}
}
```

`scenario` is metadata. Execution mode and authorization policy belong to the
CLI, so a config cannot hide `plan_only`, expected test outcomes, or permission
to activate temporary operators.

### Exact knob allowlist

| Knob | Minimum lifecycle | Meaning |
|---|---|---|
| `batch.size` | request | Canonical sample chunking only. |
| `rollout.tensor_parallel_size` | process | Rollout TP world. |
| `rollout.context_parallel_size` | process | Rollout CP world. |
| `rollout.dtype` | engine construction | Rollout numerical dtype. |
| `rollout.enable_prefix_caching` | engine construction | Engine cache policy. |
| `rollout.enforce_eager` | engine construction | Eager versus optimized/graph path. |
| `training.attention_backend` | engine construction | Training scorer attention implementation. |
| `training.compute_dtype` | engine construction | Training scorer compute dtype. |
| `logp.backend` | engine construction | Both-sides selected-logprob shortcut. |
| `training.sharding` | process | Training topology, such as unsharded or FSDP. |

TP/vocabulary layout is derived and recorded, not user-settable. Tokenization,
masks, positions, checkpoint identity, and pre-update state are invariants, not
ordinary knobs. Quantization, FP8, MoE, speculative decoding, pipeline
parallelism, and arbitrary runtime fields are deferred.

### Planning

`one_at_a_time` emits one baseline and cases that change exactly one declared
path. `pairwise` is opt-in and expands only explicitly listed path pairs. The
planner normalizes aliases, validates the allowlist and capability constraints,
and reports structured issues without creating engines. A fixed 256-case
framework cap stops OAT or pairwise expansion before unbounded accumulation.

Stable case IDs hash normalized requested values, identity, contract version,
and scenario definition. Runtime readback never rewrites a case ID. A retry gets
a new attempt ID under the same case.

## Architecture and extension points

The core has one-way responsibilities:

```text
strict config -> Planner -> ExperimentPlan -> build_execution_plan
                                              |
                                  operator-bound ExecutionPlan
                                              |
                                   runtime adapter -> RuntimeBinding
                                              |
                                         paired runner
                                          /         \
                                artifact store     comparator
```

- `config.py` owns the external schema and strict JSON loading.
- `schema.py` owns immutable, versioned domain records.
- `planner.py` owns normalization, the knob catalog, OAT, and pairwise cases.
- `execution_plan.py` binds the selected rollout/training operators into each
  immutable case and produces canonical rows shared by planning and execution.
- `runtime.py` owns the adapter protocol, three-stage materialization, lifecycle
  fingerprints, and a backend-neutral execution binding.
- `comparison.py` owns identity validation and the fixed-threshold result.
- `runner.py` coordinates paired execution and atomic publication; private
  execution, provenance, and resume modules isolate process supervision and
  validation details.
- `artifacts.py` owns append-only attempts and resume discovery.
- `semantic_registry.py` owns generic operator descriptors and case-local
  resolution sessions; it is shared by future alignment features.

The package root exposes only the common planning and execution facade. Runtime,
artifact, schema, and operator internals remain in their owning modules.

### Runtime adapters

A runtime adapter receives the normalized case and returns one application
record per knob:

```text
requested -> materialized -> actual
```

Each record includes status (`applied`, `fallback`, `unsupported`,
`unobservable`, or `error`), evidence, and lifecycle. The facade derives
construction, distributed-context, and process fingerprints. Reuse is allowed
only when all relevant fingerprints and operator bindings match.

Adapters may construct repository-native vLLM, training, or stateless config
objects internally. The core runner receives only backend-neutral batch, side
configuration, topology, scorer, operator-backend, and runtime-kind mappings,
so adding a runtime does not add branches to the planner or runner.

Strict execution rejects fallback, ignored settings, unobservable critical
values, stale registry state, and incompatible reuse. Fallback is measurable
only when it is itself the declared intervention.

## Semantic operator selection

The first semantic operator is `selected_logprob`. Rollout and training can
select implementations independently:

```json
{
  "operators": {
    "selected_logprob": {
      "rollout": "rlkernel.reference_logp",
      "training": {
        "backend": "rlkernel.reference_logp",
        "options": {}
      }
    }
  }
}
```

When `operators` is absent, `logp.backend` selects the same backend on both
sides. An explicit mapping is bound into execution identity before a runtime is
created. A `logp.backend` intervention cannot be combined with a fixed explicit
mapping because that would create a knob that no longer changes execution.

Each backend descriptor declares:

- semantic operation and backend ID;
- supported target tags, devices, dtypes, and per-target required topology values;
- alignment properties and lifecycle;
- implementation factory and version/build fingerprint;
- explicit fallback policy and temporary-test marker.

`SemanticOperatorCatalog` stores immutable descriptors. Each case creates an
`OperatorSession` for resolution, instantiation, caching, and provenance. Failed
or cached state cannot leak into the next case. Strict resolution never invokes
legacy priority fallback.

Adding a production implementation requires its existing semantic interface,
one descriptor, runtime injection hooks where needed, operator-owned correctness
tests, and one framework case. It does not require a planner change.

## Artifacts and resume

Attempts are append-only:

```text
runs/<experiment>/
  experiment.json
  plan.jsonl
  cases/<case>/<attempt>/
    requested.json
    materialized.json
    actual.json
    identity.json
    score_rollout.pt
    score_training.pt
    comparison.json
    token_diffs.pt
    COMPLETE
```

`COMPLETE` is published last and seals every required payload with a SHA-256
digest. Resume accepts only a complete attempt whose case, identity,
materialization, scorer, operator, environment, comparison, and tensor artifacts
match the current execution key. Partial, malformed, or tampered attempts are
ignored; an older valid attempt may still be reused. Existing files are never
overwritten.

## CPU smoke boundary

The only executable adapter delivered here is under
`rl_engine.validation.fixtures.cpu_cross_config`. It is explicitly CPU-only and
uses a deterministic synthetic model plus read-only stateless scoring. Named
distributed and accelerator scenarios are configuration/plan coverage only.

Temporary selected-logprob backends live together under
`rl_engine/alignment/testing/smoke_ops`:

- `smoke_only.logp_reference`: PyTorch `log_softmax` plus gather;
- `smoke_only.logp_offset`: the same result with an authorized deterministic
  offset used to prove mismatch detection.

They are CPU-only, marked `is_smoke_only`, unregistered by default, and require
both explicit registration and execution policy authorization. Their exact
removal procedure is in `SMOKE_OPERATORS.md`. Remove them when equivalent
production operators pass the same framework cases, then remove the opt-in flag
and temporary test marker.

## Scenario levels and claims

| Level | Purpose | Current claim |
|---|---|---|
| S0 | Local CPU framework smoke | Executable: config, planner, operator selection, paired scoring, comparison, artifacts, resume. |
| S1 | Small distributed lifecycle smoke | Plan only until suitable hardware/runtime adapters exist. |
| S2 | Named vLLM TP versus training FSDP comparison | Plan only. |
| S3 | Qwen3-8B TP=4, CP=4, BF16 milestone | Plan only; no production alignment claim. |

Run the shipped examples with:

```bash
python -m rl_engine.validation.cross_config plan \
  configs/experiments/cross_config/cross_config_s3_qwen3_8b_tp4_cp4_bf16.json

python -m rl_engine.validation.cross_config run \
  configs/experiments/cross_config/cross_config_s0_cpu_smoke.json \
  --runtime cpu-smoke \
  --allow-smoke-operators
```

Passing S0 proves framework plumbing only. It does not prove accelerator,
distributed, production-operator, or roadmap numerical alignment.
