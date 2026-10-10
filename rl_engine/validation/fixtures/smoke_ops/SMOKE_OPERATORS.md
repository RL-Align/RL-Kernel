# Cross-configuration smoke-only operators

These files are temporary test scaffolding. They validate operator selection,
strict resolution, active-token scoring, and provenance; they do not establish
production numerical alignment.

| File | Backend | Purpose | Replacement owner / issue |
| --- | --- | --- | --- |
| `smoke_only_logp_reference.py` | `smoke_only.logp_reference` | CPU PyTorch `log_softmax` plus gather reference for rollout/training injection tests. | Production selected-logprob operator workstream; roadmap issue #83 / WS1 contract issue #108. |
| `smoke_only_logp_offset.py` | `smoke_only.logp_offset` | Adds an explicit deterministic active-token offset so comparator mismatch detection can be tested. | Test-only fault injection; no production replacement should preserve the offset. |
| `__init__.py` | registration boundary | Keeps registration disabled by default and requires `allow_smoke_operators=True`. | Remove with both smoke implementations. |

Removal trigger: delete this package once equivalent production RL-Kernel
selected-logprob operators are integrated and the same framework tests pass using
those production backends on both rollout and training sides.

Exact deletion steps:

1. Change `tests/validation/cross_config/test_cross_config_runtime.py` to exercise the production backend
   IDs while preserving disabled/unavailable, capability, paired-output, and
   provenance coverage.
2. Remove the `smoke_operator` test marker if no other temporary smoke operator
   tests use it.
3. Delete `rl_engine/alignment/testing/smoke_ops/` and remove its exports from
   `rl_engine/validation/fixtures/__init__.py`.
4. Search the repository for `smoke_only.`, `allow_smoke_operators`, and
   `RL_KERNEL_ALLOW_SMOKE_OPS`; remove configuration and documentation references
   that no longer describe an active test boundary.
5. Run the cross-configuration contract, runtime, runner, and production-backend
   tests before merging the deletion.
