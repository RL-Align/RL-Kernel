# DSV4 compatibility check for provider-lock PR 622

PR 622 is `[ROCm] Add provider lock and execution checks`, targeting
`test-qwennext`. The isolated check applies only its 13 added files at
`468bf2ddf6eef54f2e77e16a89b830892633c917`; it does not merge that branch into
DSV4 or supply mHC arithmetic.

The contract, runtime/session, identity collector, observer protocol, CLI and
validation CPU suite passes 166 tests. These protocol fixtures do not qualify
actual GPU dispatch. The real four-layer checkpoint was also read using the
provided identity collectors, with no model or cache mutation.

The indexed `checkpoint_identity` entry rejects this single-file checkpoint:
`StrictProviderError(code="unsafe_artifact_path", field="model.safetensors.index.json")`.
That index is absent. The existing `file_inventory` collector supports the actual
`config.json`, `model.safetensors`, tokenizer and tokenizer-config files and
produces their content hashes. The weight file matches
`1bd423ab1d98f43d40b19617fcb30d973d94436cc7b150fe3f744de79c2de086`.

Run the standalone collector probe after applying those companion helper files:

```bash
python -m examples.operators.dsv4_4l.provider622_probe "$CHECKPOINT" "$NEW_RESULT_JSON"
```

The result explicitly declares `evidence_kind="disk_checkpoint"`,
`actual_worker_integration=false` and `actual_gpu_execution_verified=false`.
No completion callback is invented. The DSV4 worker does not yet implement the
PR's resolved KernelPlan/recheck/real GPU-completion ProviderObserver or attach
its StrictProviderSession to actual dispatch. Existing P1 traces and versioned
weight readback do not substitute for that protocol. This remains separate from
the actual actor/rollout numerical acceptance and does not block it.
